"""Authorized saved case views and offline, budget-supervised frozen graphics."""

import json
import re
import sys

from application.research_budget import BudgetSpec
from application.research_case_figures import case_plan, validate_case_layout, validate_case_package, MAX_SOURCE_BYTES, MAX_OUTPUT_BYTES, MAX_OPERATIONS
from application.research_computation import _verify_job_cost
from application.research_evidence import authorize_study
from application.research_supervisor import ResearchSupervisor
from experiments.pirc25.affine import ROOT, code_hash
from infrastructure.research_store import ResearchError, atomic_write, digest, encode, identifier
from infrastructure.research_visibility import study_visibility, combine_visibility


def read_case_source(store, source_id, grant):
    if not isinstance(source_id, str) or not re.fullmatch(r'[0-9a-f]{64}', source_id):
        raise ResearchError('UNAUTHORIZED_DATA', 'invalid case artifact ID')
    metadata = store.manifest('artifact-' + source_id)
    if metadata['role'] != 'result' or metadata['media_type'] != 'application/json':
        raise ResearchError('CONTRACT_MISMATCH', 'case source must be a saved JSON result')
    if metadata['size_bytes'] > MAX_SOURCE_BYTES:
        raise ResearchError('TOO_LARGE', 'case source exceeds 2 MiB')
    authorize_study(store, metadata['study_id'], grant, 'preview')
    content = store.read_artifact(source_id, purpose='preview', authorization=grant)
    result = json.loads(content)
    spec = store.manifest('study-' + metadata['study_id'])['spec']
    if (not isinstance(result, dict) or result.get('spec_hash') != digest(spec) or
            result.get('protocol_hash') != spec['protocol_hash'] or
            result.get('cell_hash') not in {digest(c) for c in spec['cells']} or
            not isinstance(result.get('forecast', {}), dict)):
        raise ResearchError('CONTRACT_MISMATCH', 'case source identity differs from registration')
    samples = result.get('forecast', {}).get('samples')
    if isinstance(samples, list) and (len(samples) > 64 or
            any(isinstance(path, list) and len(path) > 512 for path in samples)):
        raise ResearchError('TOO_LARGE', 'preview exceeds 64 trajectories or 512 points')
    return result, metadata, spec, content


def _validate_result(value, request, source):
    try:
        if (value['schema_version'] != 'pirc25-case-graph-result-v1' or value['request_hash'] != digest(request) or
                value['computation_ref'] != request['computation_ref'] or
                value['figure_index']['computation_ref'] != request['computation_ref'] or
                value['figure_index']['resource_plan'] != request['resource_plan'] or len(encode(value)) > MAX_OUTPUT_BYTES):
            raise ValueError('case worker binding differs')
        validate_case_package(source, request['source_artifact_id'], value['figure_index'], value['figures'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError('CORRUPT_ARTIFACT', 'case worker output binding differs') from exc


def verified_case_job(store, package, source):
    """Owner integrity reads only. Public source/figure reads need fresh grants."""
    try:
        with store._read_transaction():
            reference = package['computation_ref']
            proof = store._manifest(reference['manifest_id'])
            request = store._manifest(reference['request_manifest_id'])
            attempt = store._attempts().get(reference['attempt_id'])
            if (proof['schema_version'] != 'pirc25-case-graph-receipt-v1' or
                    proof['computation_ref'] != reference or request['computation_ref'] != reference or
                    proof['request_hash'] != digest(request) or not attempt or attempt['state'] != 'SUCCEEDED' or
                    attempt['run_id'] != reference['run_id'] or proof['result_artifact_id'] != attempt['artifact_id']):
                raise ValueError('case receipt or successful attempt differs')
            run = store._manifest('run-' + attempt['run_id'])
            derived = store._manifest('study-' + run['study_id'])['spec']
            original = store._manifest('study-' + package['study_id'])['spec']
            cell = next(c for c in original['cells'] if digest(c) == source['cell_hash'])
            arm = next(a for a in original['arms'] if a['arm_id'] == cell['arm_id'])
            if (request['schema_version'] != 'pirc25-case-graph-request-v1' or
                    run['spec_hash'] != reference['computation_spec_hash'] or
                    run['study_id'] != request['computation_study_id'] or derived['arms'] != [arm] or
                    run['arm_id'] != cell['arm_id'] or package['case_id'] != request['source_artifact_id'] or
                    reference['source_artifact_id'] != package['case_id'] or
                    derived['case_origin'] != request['case_origin'] or
                    request['case_origin'] != {'study_id': package['study_id'], 'source_artifact_id': package['case_id'],
                        'spec_hash': source['spec_hash'], 'cell_hash': source['cell_hash'],
                        'runtime_code_hash': request['runtime_code_hash'], 'resource_plan': request['resource_plan']}):
                raise ValueError('case source or original charged arm differs')
            metadata = store._manifest('artifact-' + attempt['artifact_id'])
            if (metadata['role'] != 'result' or metadata['study_id'] != run['study_id'] or
                    digest(metadata) != attempt['artifact_manifest_hash']):
                raise ValueError('case job result metadata differs')
            value = json.loads(store._verified_artifact_content(metadata))
            _validate_result(value, request, source)
            if (package['schema_version'] != 'pirc25-case-package-v1' or
                    package['figure_index'] != value['figure_index'] or proof['figure_index_hash'] != digest(value['figure_index']) or
                    package['figures'] != [{**e, 'artifact_id': e['sha256']} for e in value['figure_index']['figures']]):
                raise ValueError('public case package differs from managed output')
            _verify_job_cost(store, reference, run, proof)
            return proof, value
    except (KeyError, TypeError, ValueError, StopIteration) as exc:
        raise ResearchError('CORRUPT_ARTIFACT', 'case computation provenance differs') from exc


def case_view(store, source_id, grant):
    with store._read_transaction():
        source, _, _, _ = read_case_source(store, source_id, grant)
        # Layout validation is not allocation admission. Valid saved data stays
        # readable even when rendering every horizon would exceed job quotas.
        validate_case_layout(source)
        packages = []
        for event in store._events():
            if event['event_kind'] == 'MANIFEST' and event['payload']['object_id'].startswith('case-'):
                item = store._manifest(event['payload']['object_id'])
                if item.get('case_id') == source_id and item.get('study_id') == grant['study_id']:
                    packages.append(item)
        viewed = {'schema_version': 'pirc25-case-view-v1', 'case_id': source_id, 'result': source,
                  'figure_status': 'UNAVAILABLE', 'figure_reason': 'No supervised frozen case figure is saved.'}
        if not packages:
            return viewed
        package = packages[-1]  # A corrupt newest record must not silently fall back.
        proof, value = verified_case_job(store, package, source)
        for entry in package['figures']:
            metadata = store.manifest('artifact-' + entry['artifact_id'])
            if (metadata['study_id'] != grant['study_id'] or metadata['role'] != 'case-figure' or
                    metadata['visibility'] not in grant['visibilities'] or
                    not set(metadata['block_ids']) <= set(grant['block_ids']) or
                    metadata['size_bytes'] != entry['size_bytes'] or metadata['media_type'] != entry['media_type']):
                raise ResearchError('UNAUTHORIZED_DATA', 'case figure metadata outside grant')
        return {**viewed, 'figure_status': 'AVAILABLE', 'package': package,
                'figure_index': value['figure_index'], 'computation_receipt': proof}


class CaseGraphRunner:
    def __init__(self, store):
        self.store = store

    def run(self, source_id, *, authorization_id, budget=BudgetSpec(), max_operations=MAX_OPERATIONS,
            parent_attempt_id=None, reason=None):
        budget.validate()
        with self.store._read_transaction():
            grant = self.store._manifest('authorization-' + identifier(authorization_id))
            source, metadata, original, content = read_case_source(self.store, source_id, grant)
            plan = case_plan(source, max_operations)
            if plan is None:
                return {'state': 'UNAVAILABLE', 'case_id': source_id, 'exit_code': 0, 'reason': 'No saved sample paths.'}
            cell = next(c for c in original['cells'] if digest(c) == source['cell_hash'])
            arm = next(a for a in original['arms'] if a['arm_id'] == cell['arm_id'])
            visibility = combine_visibility([metadata['visibility'], study_visibility(self.store._manifest, original)])
        runtime_hash = code_hash()
        identity = digest(['saved-case-graphs-v1', source_id, runtime_hash, plan])
        origin = {'study_id': original['study_id'], 'source_artifact_id': source_id, 'spec_hash': digest(original),
                  'cell_hash': source['cell_hash'], 'runtime_code_hash': runtime_hash, 'resource_plan': plan}
        derived = {'schema_version': original['schema_version'], 'study_id': 'casejob-' + identity,
                   'experiment_id': 'saved-case-graphs-v1', 'comparison_family': original['comparison_family'],
                   'protocol_hash': original['protocol_hash'], 'data_hash': source_id, 'code_hash': runtime_hash,
                   'feature_hash': original['feature_hash'], 'selection_hash': original['selection_hash'], 'arms': [arm],
                   'cells': [{'arm_id': arm['arm_id'], 'block_id': cell['block_id'], 'seed': 0,
                              'plugin_id': 'saved-case-graphs-v1', 'resource_class': 'cpu', 'visibility': visibility}],
                   'case_origin': origin}
        self.store.register(derived, digest(derived))
        run_id = self.store.register_run(derived['study_id'], derived['cells'][0])
        prior = [a for a in self.store.attempts().values() if a['run_id'] == run_id]
        success = next((a for a in prior if a['state'] == 'SUCCEEDED'), None)
        if success:
            outcome = {'state': 'SUCCEEDED', 'attempt_id': success['attempt_id'], 'artifact_id': success['artifact_id'],
                       'exit_code': 0, 'reused': True}
        else:
            attempt_id = self.store.new_attempt(run_id, parent_attempt_id=parent_attempt_id, reason=reason)
            reference = {'manifest_id': 'case-receipt-' + attempt_id, 'request_manifest_id': 'case-request-' + attempt_id,
                         'attempt_id': attempt_id, 'run_id': run_id, 'computation_spec_hash': digest(derived),
                         'reservation_id': digest([self.store.store_id, attempt_id]), 'source_artifact_id': source_id,
                         'allocation': plan['allocation']}
            request = {'schema_version': 'pirc25-case-graph-request-v1', 'computation_ref': reference,
                       'computation_study_id': derived['study_id'], 'source_artifact_id': source_id,
                       'runtime_code_hash': runtime_hash, 'resource_plan': plan, 'case_origin': origin}

            def command(result_path):
                self.store.publish(reference['request_manifest_id'], request)
                atomic_write(result_path.parent / 'case-request.json', encode(request))
                atomic_write(result_path.parent / 'case-source.json', content)
                return [sys.executable, '-B', str(ROOT / 'experiments/pirc25/case_worker.py'),
                        str(result_path.parent / 'case-request.json'), str(result_path.parent / 'case-source.json'), str(result_path)]

            def validate(value):
                _validate_result(value, request, source)
                if code_hash() != runtime_hash:
                    raise ResearchError('CONTRACT_MISMATCH', 'case implementation moved during execution')

            outcome = ResearchSupervisor(self.store).run(attempt_id, command, budget, result_validator=validate,
                resource_plan={'maximum_result_bytes': MAX_OUTPUT_BYTES})
            outcome['reused'] = False
            if outcome['state'] != 'SUCCEEDED':
                return outcome
        with self.store._read_transaction():
            attempt = self.store._attempts()[outcome['attempt_id']]
            job_metadata = self.store._manifest('artifact-' + attempt['artifact_id'])
            value = json.loads(self.store._verified_artifact_content(job_metadata))
            reference = value['computation_ref']
            request = self.store._manifest(reference['request_manifest_id'])
            _validate_result(value, request, source)
            settles = [e for e in self.store._events() if e['event_kind'] == 'SETTLE' and
                       e['payload'].get('reservation_id') == reference['reservation_id']]
            if len(settles) != 1:
                raise ResearchError('CORRUPT_ARTIFACT', 'case job settlement missing')
            settled = settles[0]
            proof = {'schema_version': 'pirc25-case-graph-receipt-v1', 'computation_ref': reference,
                     'request_hash': digest(request), 'result_artifact_id': attempt['artifact_id'],
                     'figure_index_hash': digest(value['figure_index']), 'settlement_event_hash': settled['hash'],
                     'cost': {'arm_id': arm['arm_id'], 'charged_ms': settled['payload']['charged_ms'], 'unit': 'slot-ms',
                              'scope': 'whole-shared-computation-job', 'basis': 'measured-monotonic'}}
            self.store.publish(reference['manifest_id'], proof)
            package = {'schema_version': 'pirc25-case-package-v1', 'case_id': source_id, 'study_id': original['study_id'],
                       'computation_ref': reference, 'figure_index': value['figure_index'],
                       'figures': [{**entry, 'artifact_id': entry['sha256']} for entry in value['figure_index']['figures']]}
            verified_case_job(self.store, package, source)
            # Long jobs and reused jobs must pass fresh source permissions before publication/disclosure.
            read_case_source(self.store, source_id, grant)
            for entry in package['figures']:
                self.store.artifact(value['figures'][entry['filename']].encode(), role='case-figure', visibility=visibility,
                    study_id=original['study_id'], block_ids=metadata['block_ids'], media_type='image/svg+xml')
            self.store.publish('case-' + identity, package)
        return {**outcome, 'case': package}
