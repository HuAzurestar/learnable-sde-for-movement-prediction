"""Managed synthetic qualification on the original K=2 cumulative budget.

This is a shared computation, not a new motion capability or method family.
It publishes immutable store artifacts/receipts only; no data grant is created.
"""

from datetime import datetime
import json
import sys

from application.research_budget import BudgetSpec
from application.research_evidence import (authorize_study, require_unexpired_disclosure,
                                          verify_study_disclosure)
from application.research_supervisor import ResearchSupervisor
from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, atomic_write, digest, encode
from infrastructure.research_visibility import combine_visibility, study_visibility
from .evidence import (MAXIMUM_RESULT_BYTES, _cost_history, source_attempts_hash,
                       source_run_ids, study_definition_binding)
from .qualification import IdentifiabilitySpec, METRICS
from .approximation import pending_qualification

MAXIMUM_REQUEST_BYTES = 4*1024*1024
MAXIMUM_BOOTSTRAP_ELEMENTS = 50_000_000
COMPUTATION_ID = 'pirc28-qualification-computation-v1'


def computation_plan(definition, maximum_total_bytes, *, source_recipes=()):
    definition.validate()
    elements = (len(definition.scenario_axes)*len(definition.replicate_ids)
                *definition.bootstrap_draws*(len(METRICS)+1))
    if (elements > MAXIMUM_BOOTSTRAP_ELEMENTS or type(maximum_total_bytes) is not int
            or not 1 <= maximum_total_bytes <= 2**30):
        raise ResearchError('RESOURCE_PLAN_REJECTED', 'qualification read/bootstrap allocation exceeds quota')
    from .approximation import RECIPE_SCHEMA
    from .pipeline import recipe_allocation
    from .runtime import switching_plugin
    from application.research_registry import plan_resources
    prefix_elements = 0
    for recipe in source_recipes:
        if recipe['schema_version'] == RECIPE_SCHEMA:
            allocation = recipe_allocation(recipe, 2)
            plan_resources(switching_plugin().registry_entry, allocation['config'], allocation['inputs'], matrix_cells=1)
            prefix_elements = max(prefix_elements, allocation['combined_tensor_elements'])
    if (elements+prefix_elements)*8 > 512*1024*1024:
        raise ResearchError('RESOURCE_PLAN_REJECTED', 'additive bootstrap/prefix numeric envelope exceeds 512 MiB')
    return {'maximum_result_bytes': MAXIMUM_RESULT_BYTES,
        'maximum_request_bytes': MAXIMUM_REQUEST_BYTES, 'maximum_total_source_bytes': maximum_total_bytes,
        'maximum_rows': len(definition.scenario_axes)*len(definition.replicate_ids)*2,
        'maximum_bootstrap_elements': elements, 'bootstrap_batch_draws': 32,
        'maximum_prefix_replay_elements': prefix_elements,
        'maximum_combined_numeric_bytes': (elements+prefix_elements)*8,
        'prefix_replay_resource_rule': 'additive-bootstrap-plus-one-complete-registered-source-envelope; no-reference-resampling',
        'recovery': 'seed-deterministic-full-restart-from-frozen-source-attempts'}


def _sources(store, frozen, *, authorize=True):
    """Current source scope and authority; no numerical aggregation here."""
    studies, labels = [], []
    for study_id, expected in sorted(frozen['study_hashes'].items()):
        grant = frozen['authorizations'][study_id]
        if authorize:
            authorize_study(store, study_id, grant, 'evaluate')
        else:
            verify_study_disclosure(store, study_id, grant, 'evaluate')
        document = store.manifest('study-'+study_id)
        study = document['spec']
        if document['spec_hash'] != expected or digest(study) != expected:
            raise ResearchError('IDENTITY_CONFLICT', 'qualification computation source definition changed')
        if study.get('admission', {}).get('mode') != frozen['source_stage']:
            raise ResearchError('IDENTITY_CONFLICT', 'qualification computation source stage changed')
        label = study_visibility(store.manifest, study)
        if (not {cell['block_id'] for cell in study['cells']} <= set(grant['block_ids'])
                or label not in grant['visibilities']):
            raise ResearchError('UNAUTHORIZED_DATA', 'qualification grant does not cover complete source metadata')
        labels.append(label)
        studies.append(study)
    if source_attempts_hash(store, studies) != frozen['source_attempts_hash']:
        raise ResearchError('INDEX_STALE', 'qualification computation source attempts changed')
    # Results can carry a stricter admission/source label than their cells.
    runs = source_run_ids(studies)
    for attempt in store.attempts().values():
        if attempt['run_id'] in runs and attempt.get('artifact_id'):
            metadata = store.manifest('artifact-'+attempt['artifact_id'])
            labels.append(metadata['visibility'])
    visibility = combine_visibility(labels)
    if any(visibility not in grant['visibilities'] for grant in frozen['authorizations'].values()):
        raise ResearchError('UNAUTHORIZED_DATA', 'source grants cannot disclose aggregate visibility')
    return studies, visibility


def _final_clock(store, frozen):
    # No valid-path I/O after the earliest expiry check.
    grants = frozen['authorizations']
    study_id = min(grants, key=lambda key: datetime.fromisoformat(grants[key]['expires_at']))
    require_unexpired_disclosure(store, study_id, grants[study_id], 'evaluate')


def validate_computation_result(store, value, request):
    frozen, report = request['frozen'], value.get('report', {})
    if (value.get('schema_version') != 'pirc28-qualification-computation-result-v1'
            or value.get('request_hash') != digest(request)
            or value.get('report_hash') != digest(report) or code_hash() != frozen['runtime_code_hash']
            or report.get('schema_version') != 'pirc28-shared-qualification-v1'
            or report.get('store_id') != store.store_id
            or report.get('definition_hash') != digest(frozen['definition'])
            or report.get('source_attempts_hash') != frozen['source_attempts_hash']
            or report.get('code_hash') != frozen['runtime_code_hash']
            or report.get('aggregate_hash') != digest(report.get('rows'))
            or report.get('real_data_authorized') is not False
            or type(report.get('qualified_identifiability')) is not bool
            or len(encode(value)) > frozen['resource_plan']['maximum_result_bytes']):
        raise ResearchError('CONTRACT_MISMATCH', 'qualification computation output/source binding differs')
    definition = IdentifiabilitySpec.from_dict(frozen['definition'])
    if (len(report['rows']) != frozen['resource_plan']['maximum_rows']
            or report['gate'].get('aggregate_hash') != report['aggregate_hash']
            or report['gate'].get('spec_hash') != digest(frozen['definition'])
            or report['gate'].get('real_data_authorized') is not False
            or report['state'] != ('FIXTURE_ONLY' if frozen['source_stage'] == 'fixture' else report['gate'].get('state'))
            or report['qualified_identifiability'] is not False
            or (report['state'] == 'FIXTURE_ONLY' and not frozen['allow_fixture'])):
        raise ResearchError('CONTRACT_MISMATCH', 'qualification gate/fixture disposition differs')
    studies, _ = _sources(store, frozen)
    if computation_plan(definition, frozen['resource_plan']['maximum_total_source_bytes'],
            source_recipes=[study['switching_recipe'] for study in studies]) != frozen['resource_plan']:
        raise ResearchError('RESOURCE_PLAN_REJECTED', 'owner qualification replay allocation differs')
    if (digest(report.get('approximation_qualification')) != digest(pending_qualification(studies))
            or (frozen['source_stage'] != 'fixture' and report['state'] == 'PASSED')):
        raise ResearchError('CONTRACT_MISMATCH', 'managed approximation proof is missing; scientific promotion refused')
    reference = study_definition_binding(store, studies, definition)
    if ('study_definition_hash' not in report
            or report['study_definition_hash'] != reference
            or type(report.get('study_definition_bound')) is not bool
            or report['study_definition_bound'] != (reference is not None)
            or (report['qualified_identifiability'] and reference is None)):
        raise ResearchError('CONTRACT_MISMATCH', 'qualification master-definition binding differs')
    from .managed_evidence import validate_collected_prefix_bindings
    validate_collected_prefix_bindings(store, studies, report, frozen['authorizations'],
                                      frozen['resource_plan']['maximum_total_source_bytes'])
    _final_clock(store, frozen)


class QualificationRunner:
    def __init__(self, store):
        self.store = store

    def _prepare(self, definition, study_ids, authorizations, allow_fixture, maximum_total_bytes,
                 budget, parent_attempt_id, reason):
        if (not isinstance(definition, IdentifiabilitySpec) or type(study_ids) is not list
                or not study_ids or len(study_ids) > 10_000
                or any(type(study_id) is not str or not study_id for study_id in study_ids)
                or len(study_ids) != len(set(study_ids))
                or type(authorizations) is not dict or set(authorizations) != set(study_ids)
                or type(allow_fixture) is not bool):
            raise ResearchError('CONTRACT_MISMATCH', 'canonical definition, studies and exact grants required')
        plan = computation_plan(definition, maximum_total_bytes)
        with self.store._read_transaction():
            studies = []
            for study_id in sorted(study_ids):
                authorize_study(self.store, study_id, authorizations[study_id], 'evaluate')
                studies.append(self.store.manifest('study-'+study_id)['spec'])
            runtime_hash = code_hash()
            if any(study['code_hash'] != runtime_hash or study.get('runtime_binding') !=
                    {'root': str(self.store.path.parent), 'store_id': self.store.store_id}
                    or study.get('switching_recipe', {}).get('identifiability_spec') != definition.to_dict()
                    for study in studies):
                raise ResearchError('IDENTITY_CONFLICT', 'computation must use the frozen worker source/store/definition')
            modes = {study.get('admission', {}).get('mode') for study in studies}
            if (not modes <= {'fixture', 'pilot', 'formal'} or ('fixture' in modes and not allow_fixture)
                    or (modes == {'fixture'} and budget.category != 'smoke')
                    or ('pilot' in modes and budget.category != 'pilot')
                    or ('formal' in modes and budget.category != 'job') or len(modes) != 1):
                raise ResearchError('CONTRACT_MISMATCH', 'one source stage and its shared computation budget cap required')
            plan = computation_plan(definition, maximum_total_bytes,
                                    source_recipes=[study['switching_recipe'] for study in studies])
            # Caller cannot choose a cheap new arm or relabel a method family.
            arms = [next((arm for arm in study['arms'] if arm['arm_id'] == cell['arm_id']), None)
                    for study in studies for cell in study['cells'] if cell.get('fit_K') == 2]
            if not arms or arms[0] is None or any(arm != arms[0] for arm in arms):
                raise ResearchError('IDENTITY_CONFLICT', 'qualification must retain one original K=2 arm')
            frozen = {'definition': definition.to_dict(), 'study_hashes': {study['study_id']: digest(study) for study in studies},
                'source_attempts_hash': source_attempts_hash(self.store, studies), 'authorizations': authorizations,
                'source_stage': next(iter(modes)),
                'runtime_code_hash': runtime_hash, 'resource_plan': plan, 'allow_fixture': allow_fixture,
                'runtime_binding': {'root': str(self.store.path.parent), 'store_id': self.store.store_id}}
            if len(encode(frozen)) > MAXIMUM_REQUEST_BYTES-4096:
                raise ResearchError('RESOURCE_PLAN_REJECTED', 'qualification request exceeds metadata quota')
            _, visibility = _sources(self.store, frozen)
            identity = digest(frozen)
            derived = {'schema_version': self.store.SCHEMA, 'study_id': 'pirc28-qualify-'+identity,
                'experiment_id': COMPUTATION_ID, 'comparison_family': studies[0]['comparison_family'],
                'protocol_hash': digest(frozen['study_hashes']), 'data_hash': frozen['source_attempts_hash'],
                'code_hash': runtime_hash, 'feature_hash': digest(definition.to_dict()), 'selection_hash': digest('complete-matrix'),
                'arms': [arms[0]], 'cells': [{'arm_id': arms[0]['arm_id'], 'block_id': 'qualification-computation',
                    'seed': definition.bootstrap_seed, 'visibility': visibility, 'resource_class': 'cpu', 'plugin_id': COMPUTATION_ID}],
                'pirc28_qualification_computation': frozen}
            self.store.register(derived, digest(derived))
            run_id = self.store.register_run(derived['study_id'], derived['cells'][0])
            success = next((attempt for attempt in self.store.attempts().values()
                            if attempt['run_id'] == run_id and attempt['state'] == 'SUCCEEDED'), None)
            if success:
                return derived, None, success
            attempt_id = self.store.new_attempt(run_id, parent_attempt_id=parent_attempt_id, reason=reason)
            return derived, attempt_id, None

    def run(self, definition, study_ids, authorizations, *, budget=BudgetSpec(900, category='smoke'),
            allow_fixture=False, maximum_total_bytes=512*1024*1024, parent_attempt_id=None, reason=None):
        budget.validate()
        derived, attempt_id, success = self._prepare(definition, study_ids, authorizations,
            allow_fixture, maximum_total_bytes, budget, parent_attempt_id, reason)
        frozen = derived['pirc28_qualification_computation']
        if success:
            result = {'attempt_id': success['attempt_id'], 'state': 'SUCCEEDED',
                'artifact_id': success['artifact_id'], 'exit_code': 0, 'reused': True}
        else:
            request = {'schema_version': 'pirc28-qualification-computation-request-v1',
                'study_id': derived['study_id'], 'attempt_id': attempt_id, 'spec_hash': digest(derived), 'frozen': frozen}
            def command(path):
                self.store.publish('pirc28-qualification-request-'+attempt_id, request)
                atomic_write(path.parent/'qualification-request.json', encode(request))
                return [sys.executable, '-B', '-m', 'experiments.pirc28.qualification_worker',
                    str(self.store.path.parent), self.store.store_id, str(path.parent/'qualification-request.json'), str(path)]

            def checkpoint(state, progress, deadline):
                if state != {'request_hash': digest(request), 'restart': frozen['resource_plan']['recovery']}:
                    raise ResearchError('CONTRACT_MISMATCH', 'qualification restart checkpoint binding differs')
                import time
                if time.monotonic() >= deadline:
                    raise ResearchError('TIMEOUT', 'qualification checkpoint deadline reached')
                metadata = self.store.artifact(encode(state), role='checkpoint', visibility=derived['cells'][0]['visibility'],
                    block_ids=['qualification-computation'], study_id=derived['study_id'])
                return {'artifact_id': metadata['artifact_id'], 'resume_level': 'seed-deterministic'}

            result = ResearchSupervisor(self.store).run(attempt_id, command, budget,
                resource_plan=frozen['resource_plan'], checkpoint_handler=checkpoint,
                result_validator=lambda value: validate_computation_result(self.store, value, request))
            result['reused'] = False
            if result['state'] != 'SUCCEEDED':
                return result
        # Receipt publication is crash-retryable without another computation.
        with self.store._read_transaction():
            attempt = self.store.attempts()[result['attempt_id']]
            metadata = self.store.manifest('artifact-'+attempt['artifact_id'])
            if (type(metadata.get('size_bytes')) is not int or not 0 <= metadata['size_bytes'] <= MAXIMUM_RESULT_BYTES
                    or metadata['study_id'] != derived['study_id'] or metadata['visibility'] != derived['cells'][0]['visibility']):
                raise ResearchError('CORRUPT_ARTIFACT', 'qualification computation artifact metadata differs')
            content = self.store._verified_artifact_content(metadata)
            if len(content) > MAXIMUM_RESULT_BYTES or metadata['study_id'] != derived['study_id']:
                raise ResearchError('CORRUPT_ARTIFACT', 'qualification computation artifact scope differs')
            value = json.loads(content)
            request = self.store.manifest('pirc28-qualification-request-'+attempt['attempt_id'])
            if (request['frozen'] != frozen or request['spec_hash'] != digest(derived)
                    or request['study_id'] != derived['study_id']
                    or request['attempt_id'] != attempt['attempt_id'] or content != encode(value)):
                raise ResearchError('CORRUPT_ARTIFACT', 'qualification request/canonical result binding differs')
            validate_computation_result(self.store, value, request)
            events = [event for event in self.store.events() if event['payload'].get('attempt_id') == attempt['attempt_id']]
            cost_study = {**derived, 'admission': {'mode': next(iter({study['admission']['mode']
                for study in _sources(self.store, frozen)[0]}))}}
            history = _cost_history(self.store, cost_study, derived['cells'][0], [attempt], {attempt['attempt_id']: events})
            kinds = [event['event_kind'] for event in events]
            if (attempt['state'] != 'SUCCEEDED' or kinds.count('WORKER_STARTED') != 1
                    or kinds.count('WORKER_TREE_STOPPED') != 1 or history[0]['cost'] is None):
                raise ResearchError('UNQUALIFIED', 'qualification computation worker/cost provenance incomplete')
            settlement = next(event for event in events if event['event_kind'] == 'SETTLE')
            reserve = next(event for event in events if event['event_kind'] == 'RESERVE')
            started = next(event for event in events if event['event_kind'] == 'WORKER_STARTED')
            stopped = next(event for event in events if event['event_kind'] == 'WORKER_TREE_STOPPED')
            if (not reserve['sequence'] < started['sequence'] < stopped['sequence'] < settlement['sequence']
                    or started['payload'].get('reservation_id') != settlement['payload']['reservation_id']
                    or stopped['payload'].get('reservation_id') != settlement['payload']['reservation_id']
                    or stopped['payload'].get('observed_elapsed_ms') != history[0]['cost']['monotonic_elapsed_ms']):
                raise ResearchError('UNQUALIFIED', 'qualification launch/stop/settlement identities differ')
            receipt = {'schema_version': 'pirc28-qualification-computation-receipt-v1',
                'study_id': derived['study_id'], 'spec_hash': digest(derived), 'attempt_id': attempt['attempt_id'],
                'request_hash': digest(request), 'result_artifact_id': attempt['artifact_id'], 'report_hash': value['report_hash'],
                'source_attempts_hash': frozen['source_attempts_hash'], 'settlement_event_hash': settlement['hash'],
                'cost': {**history[0]['cost'], 'arm_id': derived['arms'][0]['arm_id'], 'unit': 'slot-ms'},
                'cost_scope': 'whole-managed-qualification-job-on-original-K2-arm',
                'qualified_identifiability': value['report']['qualified_identifiability'], 'real_data_authorized': False}
            receipt_id = 'pirc28-qualification-receipt-'+attempt['attempt_id']
            self.store.publish(receipt_id, receipt)
            _sources(self.store, frozen, authorize=False)
        _final_clock(self.store, frozen)
        return {**result, 'receipt_id': receipt_id, 'receipt_hash': digest(receipt),
                'qualification_state': value['report']['state'], 'real_data_authorized': False}
