"""Owner-side, purpose-authorized qualification collection from the shared store.

No new store, run, grant or budget is created. Existing read/reuse audit receipts
are appended by the shared APIs. A fixture report cannot authorize scientific
interpretation. Even scientific qualification never grants real-data access.
"""

from dataclasses import replace
import hashlib
import json
import math

import numpy as np

from application.research_budget import BudgetLedger
from application.research_evidence import authorize_study, require_unexpired_disclosure
from application.research_preregistration import hash_reference, validate_preregistration
from application.research_reuse import verified_reuse, verified_failure
from infrastructure.research_store import ResearchError, digest, encode
from .pipeline import validate_recipe, validate_failed_fit
from .failures import valid_numerical_diagnostic
from models.switching import SwitchingError
from .qualification import METRICS, IdentifiabilitySpec, phase_manifest, qualify
from .approximation import withhold_scientific_pass
from .runtime import PLUGIN_ID, switching_plugin
from .synthetic import ScenarioSpec

SELECTION_RULE = 'one-row-per-independent-replicate; immutable-success-else-latest-attempt'
MAXIMUM_RESULT_BYTES = 64*1024*1024


def _run_id(study, cell, spec_hash=None):
    return digest({'study_id': study['study_id'], 'experiment_id': study['experiment_id'],
        'spec_hash': digest(study) if spec_hash is None else spec_hash, 'arm_id': cell['arm_id'], 'cell_hash': digest(cell),
        'seed': cell.get('seed'), 'cell': cell})


def source_run_ids(studies):
    """Derive the exact registered identities with one study hash per pass."""
    runs = set()
    for study in studies:
        study_hash = digest(study)
        runs.update(_run_id(study, cell, study_hash) for cell in study['cells'])
    return runs


def source_attempts_hash(store, studies, *, attempts=None):
    """Scoped immutable outcome snapshot, excluding unrelated computation jobs."""
    runs = source_run_ids(studies)
    states = store.attempts() if attempts is None else attempts
    return digest({key: value for key, value in states.items() if value['run_id'] in runs})


def study_definition_binding(store, studies, definition, *, events=None):
    """Verify bounded master metadata, never fit or aggregate scientific rows."""
    references = [study.get('pirc28_study_definition_hash') for study in studies]
    if references and all(reference is None for reference in references):
        return None
    if (not references or not all(hash_reference(reference) for reference in references)
            or len(set(references)) != 1):
        raise ResearchError('NEEDS_PREREGISTRATION', 'one complete pre-fit study definition is required')
    from .study import SyntheticStudyDefinition
    reference = references[0]
    frozen = SyntheticStudyDefinition.from_dict(store.manifest('pirc28-study-definition-'+reference))
    plan, sources = frozen.to_dict(), {study['code_hash'] for study in studies}
    if (frozen.definition_hash != reference or plan['identifiability_spec'] != definition.to_dict()
            or sources != {plan['runtime_code_hash']}):
        raise ResearchError('IDENTITY_CONFLICT', 'complete study definition/evaluator/source identity differs')
    frozen.matches_studies(studies, {'root': str(store.path.parent), 'store_id': store.store_id})
    events = store.events() if events is None else events
    publications = [event for event in events if event['event_kind'] == 'MANIFEST'
        and event['payload'].get('object_id') == 'pirc28-study-definition-'+reference]
    source_runs = source_run_ids(studies)
    admitted = [event for event in events if event['event_kind'] == 'ADMISSION'
        and event['payload'].get('run_id') in source_runs]
    if not publications or any(publications[0]['sequence'] >= event['sequence'] for event in admitted):
        raise ResearchError('NEEDS_PREREGISTRATION', 'complete source matrix was published after fitting began')
    return reference


def _cost_history(store, study, cell, attempts, events_by_attempt):
    """Keep every attempt's charged or outstanding cost, including failed retries."""
    history = []
    for attempt in attempts:
        attempt_id = attempt['attempt_id']
        matching = events_by_attempt.get(attempt_id, [])
        reserves = [event for event in matching if event['event_kind'] == 'RESERVE']
        settlements = [event for event in matching if event['event_kind'] == 'SETTLE']
        terminal = [event for event in matching if event['event_kind'] == 'ATTEMPT' and event['payload'] == attempt]
        if len(reserves) > 1 or len(settlements) > 1 or not terminal:
            raise ResearchError('UNQUALIFIED', 'attempt cost/terminal provenance is incomplete')
        cost = None
        if reserves:
            reserve = reserves[0]
            binding = {'attempt_id': attempt_id, 'run_id': attempt['run_id'], 'study_id': study['study_id'],
                'arm_id': cell['arm_id'], 'reservation_id': digest([store.store_id, attempt_id])}
            if any(reserve['payload'].get(key) != value for key, value in binding.items()):
                raise ResearchError('UNQUALIFIED', 'attempt reservation belongs to a different source/arm')
            reserved = reserve['payload'].get('reserved_ms')
            cap = {'fixture': 900_000, 'pilot': 1_800_000, 'formal': 7_200_000}[study['admission']['mode']]
            if type(reserved) is not int or not 0 < reserved <= cap:
                raise ResearchError('UNQUALIFIED', 'attempt reservation violates the shared stage cap')
            if settlements:
                settlement = settlements[0]
                cost = settlement['payload']
                elapsed = cost.get('monotonic_elapsed_ms')
                if (any(cost.get(key) != value for key, value in binding.items()) or cost.get('settled') is not True
                        or cost.get('reserved_ms') != reserved or type(cost.get('charged_ms')) is not int
                        or (elapsed is not None and (type(elapsed) is not int or elapsed < 0))
                        or cost['charged_ms'] != (reserved if elapsed is None else elapsed)
                        or (attempt['state'] in store.TERMINAL and cost.get('outcome') != attempt['state'])
                        or not reserve['sequence'] < settlement['sequence'] < terminal[-1]['sequence']):
                    raise ResearchError('UNQUALIFIED', 'attempt settlement cost/binding/order differs')
            elif attempt['state'] in store.TERMINAL:
                raise ResearchError('UNQUALIFIED', 'a reserved terminal attempt has no settlement')
        elif settlements or attempt.get('started_at') is not None:
            raise ResearchError('UNQUALIFIED', 'attempt started/settled without its own reservation')
        history.append({'attempt_id': attempt_id, 'state': attempt['state'],
            'artifact_id': attempt.get('artifact_id'), 'artifact_manifest_hash': attempt.get('artifact_manifest_hash'),
            'error_code': attempt.get('error_code'), 'parent_attempt_id': attempt.get('parent_attempt_id'),
            'cost': None if not reserves else {
                'reservation_id': reserves[0]['payload']['reservation_id'],
                'reserved_ms': reserves[0]['payload']['reserved_ms'],
                'charged_ms': None if cost is None else cost['charged_ms'],
                'monotonic_elapsed_ms': None if cost is None else cost['monotonic_elapsed_ms'],
                'settled': cost is not None}})
    return history


def _source_matrix(store, study, definition):
    reference = study.get('phase_matrix_hash')
    if not hash_reference(reference):
        raise ResearchError('NEEDS_PREREGISTRATION', 'a reconstructible scientific phase matrix is required')
    value = store.manifest('pirc28-phase-matrix-' + reference)
    if (digest(value) != reference or type(value) is not dict or value.get('schema_version') != 'pirc28-phase-matrix-v1'
            or len(encode(value)) > 2*1024*1024):
        raise ResearchError('UNQUALIFIED', 'bounded phase matrix identity/schema differs')
    reconstructed = phase_manifest(ScenarioSpec.from_dict(value['baseline_scenario']), value['axes'])
    if digest(reconstructed) != reference:
        raise ResearchError('UNQUALIFIED', 'phase axes do not reconstruct the actual model/schedule/noise sources')
    scenarios = {row['scenario']['scenario_id']: row for row in value['scenarios']}
    if {name: row['axes'] for name, row in scenarios.items()} != definition.scenario_axes:
        raise ResearchError('UNQUALIFIED', 'qualification applicable ranges differ from physical phase sources')
    return reference, scenarios


def _preregistration(store, study, definition, cell, events):
    """Synthetic pilot qualification also needs a pre-fit frozen scientific plan."""
    protocol = store.manifest('protocol-' + study['admission']['protocol_id'])
    if digest(protocol) != study['protocol_hash'] or protocol['study_id'] != study['study_id']:
        raise ResearchError('IDENTITY_CONFLICT', 'qualification protocol identity differs')
    expected = hashlib.sha256(encode(study['switching_recipe'])).hexdigest()
    selected = [block for block in protocol['blocks'] if block['block_id'] == cell['block_id']]
    if (len(selected) != 1 or selected[0]['dataset_id'] != 'synthetic-pirc28'
            or selected[0]['sha256'] != expected or any(block['dataset_id'] != 'synthetic-pirc28'
                or block['sha256'] != expected for block in protocol['blocks'])):
        raise ResearchError('UNAUTHORIZED_DATA', 'qualification source is not the frozen synthetic recipe')
    reference = protocol.get('preregistration_hash')
    if not hash_reference(reference):
        raise ResearchError('NEEDS_PREREGISTRATION', 'synthetic qualification preregistration is missing')
    prereg = store.manifest('preregistration-' + reference)
    validate_preregistration(prereg)
    from application.research_preregistration import protocol_binding
    if (digest(prereg) != reference or prereg.get('pirc28_identifiability_spec_hash') != digest(definition.to_dict())
            or prereg.get('pirc28_phase_matrix_hash') != study.get('phase_matrix_hash')
            or prereg.get('result_selection_rule') != SELECTION_RULE or study['study_id'] not in prereg['study_ids']
            or protocol_binding(protocol) not in prereg['protocol_bindings']):
        raise ResearchError('NEEDS_PREREGISTRATION', 'qualification definition/selection/source was not frozen')
    if study.get('pirc28_study_definition_hash') is not None and prereg.get('pirc28_study_definition_hash') != study['pirc28_study_definition_hash']:
        raise ResearchError('NEEDS_PREREGISTRATION', 'complete recipe/initialization/replicate seed matrix was not frozen')
    k1 = {candidate['arm_id'] for candidate in study['cells'] if candidate.get('fit_K') == 1}
    k2 = {candidate['arm_id'] for candidate in study['cells'] if candidate.get('fit_K') == 2}
    if ('position_energy_score_d2' not in prereg['primary_metrics'] or len(k1) != 1 or len(k2) != 1
            or not any(comparison.get('reference') in k1 and k2 <= set(comparison.get('candidates', []))
                       for comparison in prereg['comparisons'])):
        raise ResearchError('NEEDS_PREREGISTRATION', 'paired proper-score comparison was not frozen')
    published = [event for event in events if event['event_kind'] == 'MANIFEST'
        and event['payload'].get('object_id') == 'preregistration-' + reference]
    study_hash = digest(study)
    runs = {_run_id(study, candidate, study_hash) for candidate in study['cells']}
    admitted = [event for event in events if event['event_kind'] == 'ADMISSION'
        and event['payload'].get('run_id') in runs]
    if not published or any(published[0]['sequence'] >= event['sequence'] for event in admitted):
        raise ResearchError('NEEDS_PREREGISTRATION', 'qualification plan was published after a fit began')
    return reference


def validate_initialization_history(study, cell, fit, generated, *, allow_failed_selection=False):
    _, initial_models, inference, prior_mean, prior_covariance = validate_recipe(study['switching_recipe'], cell['fit_K'])
    selection = fit.get('selection', {})
    candidates = fit.get('initializations', [])
    expected_ids = ['init-'+str(index) for index in range(len(study['switching_recipe']['initializations'][str(cell['fit_K'])]))]
    if (type(candidates) is not list or any(type(candidate) is not dict for candidate in candidates)
            or [candidate.get('initialization_id') for candidate in candidates] != expected_ids):
        raise ResearchError('UNQUALIFIED', 'complete initialization history differs')
    no_selection = selection is None
    if no_selection:
        if not allow_failed_selection or any(candidate.get('state') != 'FAILED' for candidate in candidates):
            raise ResearchError('UNQUALIFIED', 'selection is missing without all fits failing')
        selection = {'selected_initialization': None, 'candidates': [{**candidate, 'score': None} for candidate in candidates]}
    elif (type(selection) is not dict or type(selection.get('candidates')) is not list
            or any(type(candidate) is not dict for candidate in selection['candidates'])
            or selection.get('split_role') != 'selection' or selection.get('selection_hash') != generated[1]['observations_hash']
            or selection.get('rule') != 'largest-approximate-selection-objective-stable-tie'
            or [candidate.get('initialization_id') for candidate in selection.get('candidates', [])] != expected_ids):
        raise ResearchError('UNQUALIFIED', 'initialization selection source or complete diagnostic history differs')
    selected = selection.get('selected_initialization')
    eligible = []
    probability = np.ones(cell['fit_K'])/cell['fit_K']
    means = np.tile(prior_mean, (cell['fit_K'], 1))
    covariance = np.tile(prior_covariance, (cell['fit_K'], 1, 1))
    for index, (candidate, selection_row, model) in enumerate(zip(candidates, selection['candidates'], initial_models)):
        identifier = expected_ids[index]
        if candidate.get('state') == 'FAILED':
            try:
                validate_failed_fit(candidate, model, index, generated[0]['observations_hash'],
                                    probability, means, covariance, inference)
            except SwitchingError as error:
                raise ResearchError('UNQUALIFIED', 'failed initialization disclosure differs') from error
            if selection_row != {**candidate, 'score': None}:
                raise ResearchError('UNQUALIFIED', 'failed fit cannot be eligible for selection')
            continue
        if (candidate.get('state') != 'SUCCEEDED' or candidate.get('schema_version') != 'pirc28-fit-v1'
                or candidate.get('initial_model_hash') != digest(model.to_dict())
                or type(candidate.get('model')) is not dict or candidate['model'].get('K') != cell['fit_K']
                or candidate.get('configured_K') != cell['fit_K']
                or candidate.get('model_hash') != digest(candidate.get('model'))
                or candidate.get('inference_spec') != inference.to_dict()
                or type(candidate.get('collapsed')) is not bool
                or selection_row.get('collapsed') != candidate['collapsed']
                or selection_row.get('stopping_reason') != candidate.get('stopping_reason')):
            raise ResearchError('UNQUALIFIED', 'fitted initialization/model/selection history differs')
        trace, rejected = candidate.get('objective_trace'), candidate.get('rejected_objectives')
        if (type(trace) is not list or not 1 <= len(trace) <= inference.max_iterations+1
                or type(rejected) is not list or len(rejected) > inference.max_iterations*inference.line_search_steps
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in trace+rejected)
                or np.any(np.diff(trace) < 0)
                or candidate.get('stopping_reason') not in {'MAX_ITERATIONS', 'REJECTED_STEP', 'CONVERGED_APPROXIMATE_OBJECTIVE'}):
            raise ResearchError('UNQUALIFIED', 'successful initialization accepted/rejected progress differs')
        if selection_row.get('state') == 'FAILED':
            if (not valid_numerical_diagnostic(selection_row, 'selection', identifier)
                    or selection_row.get('score') is not None):
                raise ResearchError('UNQUALIFIED', 'failed selection must retain typed failure without a score')
        elif (selection_row.get('state') == 'SUCCEEDED'
                and type(selection_row.get('score')) in (int, float) and math.isfinite(selection_row['score'])
                and selection_row.get('error_code') is None):
            eligible.append((selection_row['score'], -index, identifier))
        else:
            raise ResearchError('UNQUALIFIED', 'selection candidate is neither applicable nor a typed failure')
    if not eligible and allow_failed_selection and selected is None:
        return None
    if not eligible or selected != max(eligible, key=lambda value: value[:2])[2]:
        raise ResearchError('UNQUALIFIED', 'selected initializer is not the preregistered applicable-score winner')
    return next(candidate for candidate in candidates if candidate['initialization_id'] == selected)


def _success_row(definition, study, cell, result, artifact_hash):
    scenario, _, _, _, _ = validate_recipe(study['switching_recipe'], cell['fit_K'])
    fit, diagnostic = result.get('fit', {}), result.get('fit', {}).get('recovery_diagnostics', {})
    if (result.get('source_schema') != 'pirc28-synthetic-result-v1'
            or fit.get('model', {}).get('K') != cell['fit_K']
            or fit.get('recipe_hash') != digest({'recipe': study['switching_recipe'], 'K': cell['fit_K'], 'seed': cell['seed']})
            or diagnostic.get('identifiability_spec_hash') != digest(definition.to_dict())):
        raise ResearchError('UNQUALIFIED', 'qualification result recipe/K/evaluator identities differ')
    rng = np.random.default_rng(cell['seed'])
    seeds = rng.integers(0, 2**63, size=3, dtype=np.int64)
    generated = fit.get('generated_inputs')
    if type(generated) is not list or len(generated) != 3:
        raise ResearchError('UNQUALIFIED', 'generated role-specific source identities missing')
    for index, role in enumerate(('train', 'selection', 'test')):
        frozen = replace(scenario, seed=int(seeds[index]), sample_count=study['switching_recipe'][role+'_paths'],
                         scenario_id=scenario.scenario_id+'-'+role)
        if (generated[index].get('role') != role or generated[index].get('scenario_hash') != frozen.scenario_hash
                or not hash_reference(generated[index].get('observations_hash'))
                or not hash_reference(generated[index].get('truth_hash'))):
            raise ResearchError('IDENTITY_CONFLICT', 'generated roles do not match frozen independent streams')
    metrics = {'position_energy_score_d2': result['metrics']['position_energy_score_d2']}
    if cell['fit_K'] == 2:
        parameter = diagnostic.get('parameter_recovery', {}).get('metrics', {})
        mode = diagnostic.get('mode_diagnostics', {})
        jump = diagnostic.get('jump_diagnostics', {})
        values = {**parameter, **{name: mode[name] for name in ('mode_brier', 'mode_ece', 'occupancy_error') if name in mode},
            'position_marginal_coverage_error': diagnostic.get('position_marginal_coverage_error'),
            **{name: jump[name] for name in ('any_jump_brier', 'different_endpoint_mode_brier') if name in jump}}
        if set(values) != set(METRICS) or any(value is None for value in values.values()):
            raise ResearchError('UNQUALIFIED', 'K=2 source artifact lacks preregistered recovery/calibration metrics')
        metrics.update(values)
    chosen = validate_initialization_history(study, cell, fit, generated)
    # This diagnostic remains separate from test posterior mode alignment.
    if chosen is None or chosen.get('model') != fit.get('model') or chosen.get('model_hash') != digest(fit.get('model')):
        raise ResearchError('UNQUALIFIED', 'selected fit and retained initialization history differ')
    return {'scenario_id': scenario.scenario_id, 'replicate_id': cell['replicate_id'], 'K': cell['fit_K'],
        'state': 'SUCCEEDED', 'input_hash': digest(generated), 'artifact_hash': artifact_hash,
        'metrics': metrics, 'posterior_occupancy': chosen['posterior_occupancy'], 'collapsed': chosen['collapsed']}


def collect_qualification(store, definition, study_ids, authorizations, *, check_budget,
                          allow_fixture=False, maximum_total_bytes=512*1024*1024,
                          expected_attempts_hash=None):
    """Collect a complete declared matrix; no retry or initialization cherry-pick.

    authorizations maps study IDs to existing evaluate-purpose grant documents.
    Fixture collection is explicitly opt-in and can only produce FIXTURE_ONLY.
    Pilot results may establish the synthetic gate with a pre-fit scientific
    plan; formal blind-test admission is not incorrectly required before that
    prerequisite gate. All grants are checked again before aggregate disclosure.
    """
    if not isinstance(definition, IdentifiabilitySpec):
        raise ResearchError('CONTRACT_MISMATCH', 'canonical identifiability definition required')
    definition.validate()
    if (type(study_ids) is not list or not study_ids or len(study_ids) > 10_000
            or any(type(value) is not str or not value for value in study_ids)
            or len(set(study_ids)) != len(study_ids) or type(authorizations) is not dict
            or set(authorizations) != set(study_ids) or type(allow_fixture) is not bool
            or type(maximum_total_bytes) is not int or not 1 <= maximum_total_bytes <= 2**30):
        raise ResearchError('CONTRACT_MISMATCH', 'bounded studies, exact grant scopes and byte quota required')
    with store._read_transaction():
        for study_id in study_ids:
            authorize_study(store, study_id, authorizations[study_id], 'evaluate')
        events, attempts = store.events(), store.attempts()
    events_by_attempt, attempts_by_run = {}, {}
    for index, event in enumerate(events):
        if index % 256 == 0:
            check_budget()
        attempt_id = event['payload'].get('attempt_id')
        if attempt_id is not None:
            events_by_attempt.setdefault(attempt_id, []).append(event)
    for attempt in attempts.values():
        attempts_by_run.setdefault(attempt['run_id'], []).append(attempt)
    expected = {(scenario, replicate, k) for scenario in definition.scenario_axes for replicate in definition.replicate_ids for k in (1, 2)}
    cells, families, sources, matrices, study_hashes = {}, {}, set(), {}, {}
    fixture = False
    studies = []
    preregistrations = set()
    for study_id in sorted(study_ids):
        check_budget()
        document = store.manifest('study-' + study_id)
        study = document['spec']
        studies.append(study)
        study_hashes[study_id] = digest(study)
        if (document.get('spec_hash') != digest(study) or study['study_id'] != study_id
                or study.get('runtime_binding') != {'root': str(store.path.parent), 'store_id': store.store_id}
                or study.get('switching_recipe', {}).get('identifiability_spec') != definition.to_dict()):
            raise ResearchError('IDENTITY_CONFLICT', 'study/store/frozen qualification identities differ')
        grant = authorizations[study_id]
        from infrastructure.research_visibility import study_visibility
        if (not {cell['block_id'] for cell in study['cells']} <= set(grant['block_ids'])
                or study_visibility(store.manifest, study) not in grant['visibilities']):
            raise ResearchError('UNAUTHORIZED_DATA', 'grant cannot disclose the complete source/cost matrix')
        mode = study.get('admission', {}).get('mode')
        if mode not in ('fixture', 'pilot', 'formal') or (mode == 'fixture' and not allow_fixture):
            raise ResearchError('UNQUALIFIED', 'fixture evidence is not scientific qualification')
        fixture |= mode == 'fixture'
        sources.add(study['code_hash'])
        if mode != 'fixture':
            reference = study.get('phase_matrix_hash')
            if not hash_reference(reference):
                raise ResearchError('NEEDS_PREREGISTRATION', 'scientific matrix identity is missing')
            if reference not in matrices:
                reference, matrices[reference] = _source_matrix(store, study, definition)
            actual = study['switching_recipe']['scenario']
            expected_scenario = matrices[reference].get(actual['scenario_id'])
            if expected_scenario is None or digest(expected_scenario['scenario']) != digest(actual):
                raise ResearchError('UNQUALIFIED', 'registered recipe is not the reconstructed scientific phase scenario')
            preregistrations.add(_preregistration(store, study, definition, study['cells'][0], events))
        for cell in study['cells']:
            k, replicate = cell.get('fit_K'), cell.get('replicate_id')
            scenario = study['switching_recipe']['scenario']['scenario_id']
            if type(k) is not int or type(replicate) is not str:
                raise ResearchError('CONTRACT_MISMATCH', 'cell must bind K and an independent replicate ID')
            key = (scenario, replicate, k)
            if (key not in expected or key in cells or cell.get('plugin_id') != PLUGIN_ID
                    or cell.get('visibility') != 'synthetic'):
                raise ResearchError('CONTRACT_MISMATCH', 'undeclared/duplicate/non-synthetic qualification cell')
            validate_recipe(study['switching_recipe'], k)
            arm = next((arm for arm in study['arms'] if arm['arm_id'] == cell['arm_id']), None)
            if arm is None or arm.get('budget_seconds') != 86400:
                raise ResearchError('CONTRACT_MISMATCH', 'qualification arm cap differs')
            family = tuple(arm[name] for name in ('arm_id', 'model_family_id', 'method_family_id', 'objective_id'))
            if k in families and families[k] != family:
                raise ResearchError('IDENTITY_CONFLICT', 'scenario or replicate changed its original arm family')
            families[k] = family
            cells[key] = (study, cell)
    if (set(cells) != expected or len(sources) != 1 or families[1][0] == families[2][0]
            or families[1][1] == families[2][1] or (not fixture and len(matrices) != 1)):
        raise ResearchError('UNQUALIFIED', 'full paired matrix, one code source and separate fixed K arms required')
    study_definition_hash = study_definition_binding(store, studies, definition, events=events)
    snapshot_hash = source_attempts_hash(store, studies, attempts=attempts)
    if expected_attempts_hash is not None and snapshot_hash != expected_attempts_hash:
        raise ResearchError('INDEX_STALE', 'frozen source attempts changed before qualification')
    # Recipe/seed pairing is checked even when neither cell has run yet.
    for scenario in definition.scenario_axes:
        seeds = set()
        for replicate in definition.replicate_ids:
            left, right = cells[(scenario, replicate, 1)], cells[(scenario, replicate, 2)]
            seed = left[1].get('seed')
            if (type(seed) is not int or not 0 <= seed < 2**64 or seed != right[1].get('seed')
                    or digest(left[0]['switching_recipe']) != digest(right[0]['switching_recipe']) or seed in seeds):
                raise ResearchError('IDENTITY_CONFLICT', 'paired cells differ or duplicate an independent source seed')
            seeds.add(seed)
    rows, lineage, disclosed, prefix_evidence = [], [], [], []
    consumed = 0
    plugin = switching_plugin()
    balances = {k: BudgetLedger(store).balance(family[0]) for k, family in families.items()}
    for key in sorted(cells):
        check_budget()
        study, cell = cells[key]
        run_id = _run_id(study, cell, study_hashes[study['study_id']])
        related = attempts_by_run.get(run_id, [])
        if related:
            run = store.manifest('run-' + run_id)
            if run.get('spec_hash') != digest(study) or run.get('cell') != cell or run.get('run_id') != run_id:
                raise ResearchError('IDENTITY_CONFLICT', 'qualification run source differs')
        history = _cost_history(store, study, cell, related, events_by_attempt)
        failures = {}
        for attempt in related:
            if attempt['state'] != 'FAILED' or attempt.get('artifact_id') is None:
                continue
            check_budget()
            artifact_id = attempt['artifact_id']
            metadata = store.manifest('artifact-'+artifact_id)
            size = metadata.get('size_bytes')
            if type(size) is not int or not 0 <= size <= MAXIMUM_RESULT_BYTES or consumed+size > maximum_total_bytes:
                raise ResearchError('RESOURCE_LIMIT', 'failure artifact/matrix bytes exceed frozen quota')
            grant = authorizations[study['study_id']]
            content = store.read_artifact(artifact_id, purpose='evaluate', authorization=grant)
            consumed += len(content)
            result = json.loads(content)
            if len(content) != size or hashlib.sha256(content).hexdigest() != artifact_id or content != encode(result):
                raise ResearchError('CORRUPT_ARTIFACT', 'authorized failure bytes differ from frozen canonical metadata')
            verified_failure(store, attempt, study, cell, plugin)
            from .managed_evidence import check_cell_prefix
            prefix = check_cell_prefix(store, study, cell, attempt, result, metadata, grant, history,
                                       check_budget=check_budget)
            if prefix is not None:
                prefix_evidence.append(prefix)
            failures[attempt['attempt_id']] = result['failure']
            next(value for value in history if value['attempt_id'] == attempt['attempt_id'])['failure'] = result['failure']
            disclosed.append((artifact_id, grant))
        successful = [attempt for attempt in related if attempt['state'] == 'SUCCEEDED']
        if len(successful) > 1:
            raise ResearchError('IDENTITY_CONFLICT', 'a single independent cell has multiple successful outcomes')
        selected = successful[0] if successful else related[-1] if related else None
        lineage.append({'scenario_id': key[0], 'replicate_id': key[1], 'K': key[2],
            'study_id': study['study_id'], 'cell_hash': digest(cell), 'run_id': run_id,
            'selected_attempt_id': None if selected is None else selected['attempt_id'], 'attempts': history})
        if successful:
            artifact_id = selected['artifact_id']
            metadata = store.manifest('artifact-' + artifact_id)
            size = metadata.get('size_bytes')
            if type(size) is not int or not 0 <= size <= MAXIMUM_RESULT_BYTES or consumed+size > maximum_total_bytes:
                raise ResearchError('RESOURCE_LIMIT', 'qualification artifact/matrix bytes exceed the frozen quota')
            grant = authorizations[study['study_id']]
            content = store.read_artifact(artifact_id, purpose='evaluate', authorization=grant)
            consumed += len(content)
            if len(content) != size or hashlib.sha256(content).hexdigest() != artifact_id:
                raise ResearchError('CORRUPT_ARTIFACT', 'authorized qualification result bytes differ from metadata')
            result = json.loads(content)
            if content != encode(result):
                raise ResearchError('CORRUPT_ARTIFACT', 'qualification source is not canonical result JSON')
            verified_reuse(store, selected, study, cell, plugin)
            from .managed_evidence import check_cell_prefix
            prefix = check_cell_prefix(store, study, cell, selected, result, metadata, grant, history,
                                       check_budget=check_budget)
            if prefix is not None:
                prefix_evidence.append(prefix)
            row = _success_row(definition, study, cell, result, artifact_id)
            lineage[-1]['initialization_diagnostics'] = [
                {'initialization_id': fit['initialization_id'], 'fit_state': fit['state'],
                 'selection_state': selection['state'], 'fit_error_code': fit.get('error_code'),
                 'selection_error_code': selection.get('error_code'), 'selection_score': selection['score'],
                 'collapsed': fit.get('collapsed'), 'stopping_reason': fit.get('stopping_reason'),
                 'last_accepted_state_hash': None if fit.get('last_accepted_state') is None
                     else digest(fit['last_accepted_state'])}
                for fit, selection in zip(result['fit']['initializations'], result['fit']['selection']['candidates'])]
            rows.append(row)
            disclosed.append((artifact_id, grant))
        else:
            state = 'NOT_RUN'
            if selected is None and balances[key[2]]['closed']:
                state = 'BUDGET_EXHAUSTED'
            elif selected is not None:
                if selected['state'] == 'BUDGET_EXHAUSTED':
                    state = 'BUDGET_EXHAUSTED'
                elif selected['state'] == 'TIMEOUT':
                    state = 'TIMEOUT'
                elif selected['state'] in store.TERMINAL:
                    state = 'INFERENCE_INAPPLICABLE' if selected.get('error_code') == 'INFERENCE_INAPPLICABLE' else 'FAILED'
            rows.append({'scenario_id': key[0], 'replicate_id': key[1], 'K': key[2], 'state': state,
                'input_hash': None if selected is None or selected['attempt_id'] not in failures
                    else digest(failures[selected['attempt_id']]['generated_inputs']),
                'artifact_hash': None if selected is None or selected['attempt_id'] not in failures else selected['artifact_id'],
                'metrics': None, 'posterior_occupancy': None, 'collapsed': None})
    report = qualify(definition, rows, digest(rows), check_budget=check_budget)
    if not fixture and report['state'] == 'PASSED' and study_definition_hash is None:
        report['state'] = 'NEEDS_PREREGISTRATION'
        report['missing_fields'].append('pirc28_study_definition_hash')
        if 'applicable_ranges' in report:
            report['applicable_scenarios'], report['applicable_ranges'] = [], []
    approximation = withhold_scientific_pass(report, studies, fixture=fixture)
    if len(encode({'rows': rows, 'lineage': lineage, 'gate': report,
                   'prefix_evidence': prefix_evidence})) > MAXIMUM_RESULT_BYTES:
        raise ResearchError('RESOURCE_LIMIT', 'qualification aggregate exceeds its bounded byte quota')
    # Integrity reuse and interval computation occur after the actual read;
    # purpose grants must still be valid at this final disclosure boundary.
    with store._read_transaction():
        for study_id in study_ids:
            authorize_study(store, study_id, authorizations[study_id], 'evaluate')
        for artifact_id, grant in disclosed:
            store.verify_artifact_read(artifact_id, purpose='evaluate', authorization=grant)
        if source_attempts_hash(store, studies) != snapshot_hash:
            raise ResearchError('INDEX_STALE', 'source attempts changed during qualification')
    for study_id in study_ids:
        require_unexpired_disclosure(store, study_id, authorizations[study_id], 'evaluate')
    scientific = False  # No implemented, managed approximation proof adapter yet.
    return {'schema_version': 'pirc28-shared-qualification-v1', 'store_id': store.store_id,
        'definition_hash': digest(definition.to_dict()), 'aggregate_hash': digest(rows), 'rows': rows,
        'lineage': lineage, 'gate': report, 'state': 'FIXTURE_ONLY' if fixture else report['state'],
        'qualified_identifiability': scientific, 'real_data_authorized': False,
        'approximation_qualification': approximation,
        'cell_approximation_evidence': prefix_evidence,
        'cell_approximation_evidence_hash': digest(prefix_evidence),
        'preregistration_hashes': sorted(preregistrations), 'code_hash': next(iter(sources)),
        'study_definition_hash': study_definition_hash, 'study_definition_bound': study_definition_hash is not None,
        'arm_balances': balances, 'read_bytes': consumed, 'selection_rule': SELECTION_RULE,
        'source_snapshot_event_hash': None if not events else events[-1]['hash'],
        'source_attempts_hash': snapshot_hash,
        'evidence_binding': 'shared-authorized-result-read-and-verified-reuse-provenance'}
