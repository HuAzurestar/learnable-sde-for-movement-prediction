"""Authorized artifact/cost bindings for independently checked prefix diagnostics.

Numerical checks belong in the funded qualification worker. The publication
owner repeats bounded source/metadata binding, not fitting or reference sampling.
Neither path promotes engineering diagnostics to scientific qualification.
"""

import hashlib
import json

import numpy as np

from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, digest, encode
from .approximation import ApproximationSpec, RECIPE_SCHEMA
from .approximation_evidence import reference_seed, replay_test_prefix, validate_prefix_numbers
from .pipeline import validate_recipe
from .selection_evidence import validate_selected_fit
from .input_evidence import expected_input_receipt, validate_generated_inputs
from .reference_execution import validate_reference_execution
from .failures import POST_REFERENCE_STAGES

SCHEMA = 'pirc28-authorized-prefix-evidence-v1'


def _require(condition, detail):
    if not condition:
        raise ResearchError('UNQUALIFIED', detail)


def _projection(result, study, cell):
    recipe = study['switching_recipe']
    if recipe['schema_version'] != RECIPE_SCHEMA:
        return None
    _require(study['code_hash'] == code_hash(), 'prefix replay requires the original current code source')
    if cell['fit_K'] == 1:
        if result['status'] == 'SUCCEEDED':
            fit = result.get('fit')
            scope = ('engineering-only; not-managed-authority-proof' if study['admission']['mode'] == 'fixture'
                     else 'admitted-source-diagnostic-only; not-managed-authority-proof')
            _require(type(fit) is dict and fit.get('approximation_execution_scope') == scope,
                     'K1 diagnostic execution scope differs')
            report = fit.get('k1_boundary_checks')
            _require(type(report) is dict and report.get('state') == 'NUMERICAL_CHECKS_PASSED',
                     'successful V2 K1 artifact lacks complete passed boundary report')
        elif result['status'] == 'FAILED' and result.get('source_schema') == 'pirc28-synthetic-failure-v4':
            from .evidence import validate_initialization_history
            body = result['failure']
            chosen = validate_initialization_history(study, cell, body, body['generated_inputs'])
            _require(chosen is not None, 'K1 boundary failure has no selected fit')
            fit = {name: body[name] for name in ('recipe_hash', 'generated_inputs', 'initializations', 'selection')}
            fit.update(model=chosen['model'], k1_boundary_execution=body['k1_boundary_execution'],
                       interpretation_level='synthetic-statistical-only', qualified_identifiability=False)
            report = body['k1_boundary_checks']
        else:
            return None
        _require(type(result.get('admission_hash')) is str, 'original K1 admission missing')
        return fit, report, result['admission_hash']
    if result['status'] == 'SUCCEEDED':
        fit = result.get('fit')
        scope = ('engineering-only; not-managed-authority-proof' if study['admission']['mode'] == 'fixture'
                 else 'admitted-source-diagnostic-only; not-managed-authority-proof')
        _require(type(fit) is dict and fit.get('approximation_execution_scope') == scope, 'prefix execution scope differs')
        report = fit.get('approximation_checks')
        _require(type(report) is dict and report.get('state') == 'NUMERICAL_CHECKS_PASSED',
                 'successful V2 K2 artifact lacks complete passed prefix report')
    elif result['status'] == 'FAILED' and result.get('source_schema') in ('pirc28-synthetic-failure-v2', 'pirc28-synthetic-failure-v3'):
        from .evidence import validate_initialization_history
        body = result['failure']
        chosen = validate_initialization_history(study, cell, body, body['generated_inputs'])
        _require(chosen is not None, 'failed reference has no selected fit')
        fit = {name: body[name] for name in ('recipe_hash', 'generated_inputs', 'initializations', 'selection')}
        fit.update(model=chosen['model'], interpretation_level='synthetic-statistical-only',
                   qualified_identifiability=False)
        fit['reference_execution'] = body.get('reference_execution')
        report = body['approximation_checks']
        post_reference = body['stage'] in POST_REFERENCE_STAGES
        expected_state = 'NUMERICAL_CHECKS_PASSED' if post_reference else 'INFERENCE_INAPPLICABLE'
        _require(not post_reference and report is None or type(report) is dict and report.get('state') == expected_state,
                 'failed reference report disposition differs')
    else:
        return None
    if report is not None:
        _require(all(report.get(name) is False for name in (
            'managed_evidence_bound', 'qualified_identifiability', 'real_data_authorized')),
            'diagnostic report cannot invent scientific authority')
    _require(type(result.get('admission_hash')) is str, 'source result admission identity missing')
    return fit, report, result['admission_hash']


def _receipt(store, study, cell, attempt, metadata, grant, history, projection, native_events):
    fit, report, admission_hash = projection
    k = cell['fit_K']
    if k == 1:
        from .k1_execution import validate_execution
        execution = fit.get('k1_boundary_execution')
        validate_execution(execution, study, cell, attempt, fit['model'], report, admission_hash=admission_hash)
        reference_complete = False
    else:
        execution = fit.get('reference_execution')
        reference_complete = validate_reference_execution(execution, study, cell, attempt, fit['model'], report,
                                                           admission_hash=admission_hash)
    _, _, inference, mean, covariance = validate_recipe(study['switching_recipe'], k)
    view = {name: fit[name] for name in ('recipe_hash', 'model', 'selection', 'initializations', 'generated_inputs')}
    if report is not None:
        _require(report.get('spec_hash') == ApproximationSpec.from_dict(
            study['switching_recipe']['approximation_spec']).spec_hash
            and (k == 1 or report.get('reference_seed') == reference_seed(study['switching_recipe'], cell['seed']))
            and report.get('model_hash') == digest(fit['model'])
            and report.get('inference_hash') == digest(inference.to_dict())
            and report.get('origin_index') == study['switching_recipe']['forecast_origin_index']
            and report.get('initial_state_hash') == digest({'probability': (np.ones(k)/k).tolist(),
                'means': np.tile(mean, (k, 1)).tolist(), 'covariances': np.tile(covariance, (k, 1, 1)).tolist()}),
            'prefix report expected model/prior/recipe identity differs')
    # Attached failure bodies are disclosed separately; this projection binds
    # EVERY attempt's immutable identity, parent, state, artifact and full cost.
    costs = [{name: value for name, value in item.items() if name != 'failure'} for item in history]
    stops = [event for event in native_events if event['event_kind'] == 'WORKER_TREE_STOPPED']
    starts = [event for event in native_events if event['event_kind'] == 'WORKER_STARTED']
    settlements = [event for event in native_events if event['event_kind'] == 'SETTLE']
    cost = next(item['cost'] for item in history if item['attempt_id'] == attempt['attempt_id'])
    _require(len(stops) == len(starts) == len(settlements) == 1 and cost is not None
        and stops[0]['payload'].get('reservation_id') == digest([store.store_id, attempt['attempt_id']])
        and type(stops[0]['payload'].get('observed_elapsed_ms')) is int
        and stops[0]['payload']['observed_elapsed_ms'] == cost['monotonic_elapsed_ms']
        and stops[0]['payload'].get('confirmation') == 'native-job-or-process-group-no-running-descendants'
        and starts[0]['sequence'] < stops[0]['sequence'] < settlements[0]['sequence'],
        'prefix source has no actual native whole-tree stop/cost proof')
    receipt = {'schema_version': SCHEMA, 'study_id': study['study_id'], 'cell_hash': digest(cell),
        'run_id': attempt['run_id'], 'attempt_id': attempt['attempt_id'], 'arm_id': cell['arm_id'],
        'runtime_binding': {'root': str(store.path.parent), 'store_id': store.store_id},
        'source_code_hash': study['code_hash'], 'artifact_id': attempt['artifact_id'],
        'artifact_manifest_hash': digest(metadata), 'admission_hash': admission_hash,
        'authorization_hash': digest(grant), 'all_attempt_costs_hash': digest(costs),
        'recipe_cell_hash': fit['recipe_hash'], 'selection_projection_hash': digest(view),
        'selection_input_hash': fit['generated_inputs'][1]['observations_hash'],
        'selected_model_hash': digest(fit['model']),
        'generated_input_replay_hash': digest(expected_input_receipt(
            fit['generated_inputs'], study['switching_recipe'], cell['seed'], k=k)),
        'reference_report_hash': None if report is None else digest(report),
        'observation_prefix_hash': None if report is None else report['observation_prefix_hash'],
        'reference_seed': reference_seed(study['switching_recipe'], cell['seed']) if k == 2 else None,
        'state': 'NO_REFERENCE_REPORT' if report is None else report['state'],
        'selection_scores_recomputed': True, 'prefix_inputs_replayed': report is not None,
        'prefix_diagnostics_consistent': report is not None,
        'authorized_native_source_and_cost_bound': True,
        'reference_execution_hash': digest(execution),
        'full_generated_input_hashes_verified': True, 'reference_sampling_verified': reference_complete,
        'managed_evidence_bound': False, 'qualified_identifiability': False, 'real_data_authorized': False}
    if k == 1:
        receipt.update(reference_execution_hash=None, k1_boundary_execution_hash=digest(execution),
                       k1_boundary_verified=report is not None and report['state'] == 'NUMERICAL_CHECKS_PASSED')
    return receipt


def check_cell_prefix(store, study, cell, attempt, result, metadata, grant, history, *, check_budget):
    """Caller must first authorize/read canonical bytes and verify native reuse/failure."""
    _require(callable(check_budget), 'explicit funded prefix evidence poll required')
    check_budget()
    if study['switching_recipe']['schema_version'] == RECIPE_SCHEMA:
        from application.research_evidence import authorize_study
        from application.research_reuse import verified_failure, verified_reuse
        from .runtime import switching_plugin
        authorize_study(store, study['study_id'], grant, 'evaluate')
        store.verify_artifact_read(attempt['artifact_id'], purpose='evaluate', authorization=grant)
        _require(digest(result) == attempt['artifact_id']
            and digest(metadata) == attempt['artifact_manifest_hash'], 'checked source artifact projection changed')
        if attempt['state'] == 'SUCCEEDED':
            verified_reuse(store, attempt, study, cell, switching_plugin())
        else:
            verified_failure(store, attempt, study, cell, switching_plugin())
        check_budget()
    projection = _projection(result, study, cell)
    if projection is None:
        return None
    fit, report, _ = projection
    k = cell['fit_K']
    validate_generated_inputs(fit['generated_inputs'], study['switching_recipe'], cell['seed'], k=k,
                              check_budget=check_budget)
    model, _ = validate_selected_fit(fit, study['switching_recipe'], cell['seed'], k=k,
                                     check_budget=check_budget)
    if report is not None:
        prefix, _ = replay_test_prefix(study['switching_recipe'], cell['seed'], k=k, check_budget=check_budget)
        _, _, inference, mean, covariance = validate_recipe(study['switching_recipe'], k)
        from .k1_boundary import validate_k1_boundary
        verifier = validate_k1_boundary if k == 1 else validate_prefix_numbers
        extras = {} if k == 1 else {'seed': reference_seed(study['switching_recipe'], cell['seed'])}
        receipt = verifier(report, prefix, model, np.ones(k)/k,
            np.tile(mean, (k, 1)), np.tile(covariance, (k, 1, 1)), inference,
            ApproximationSpec.from_dict(study['switching_recipe']['approximation_spec']),
            origin_index=study['switching_recipe']['forecast_origin_index'],
            check_budget=check_budget, **extras)
        if result['status'] == 'SUCCEEDED':
            name = 'k1_boundary_consistency' if k == 1 else 'approximation_consistency'
            _require(digest(fit.get(name)) == digest(receipt),
                     'source diagnostic consistency receipt differs from independent check')
    check_budget()
    events = [event for event in store.events() if event['payload'].get('attempt_id') == attempt['attempt_id']]
    return _receipt(store, study, cell, attempt, metadata, grant, history, projection, events)


def validate_collected_prefix_bindings(store, studies, report, authorizations, maximum_total_bytes):
    """Owner-side bounded authorized metadata checks; NEVER repeat numerics here."""
    from application.research_reuse import verified_failure, verified_reuse
    from .evidence import (MAXIMUM_RESULT_BYTES, _cost_history, _run_id,
        _success_row, _non_success_row, _initialization_diagnostics, _snapshot_balances)
    from .qualification import IdentifiabilitySpec
    from .failure_disclosure import disclose_failure
    from .runtime import switching_plugin
    claimed = report.get('cell_approximation_evidence')
    _require(type(claimed) is list and report.get('cell_approximation_evidence_hash') == digest(claimed),
             'complete prefix evidence roster/hash missing')
    study_hashes = {study['study_id']: digest(study) for study in studies}
    events_by_attempt, by_run = {}, {}
    events = store.events()
    snapshot = next((index for index, event in enumerate(events)
                     if event['hash'] == report.get('source_snapshot_event_hash')), None)
    _require(snapshot is not None, 'owner source event snapshot is missing')
    snapshot_events = events[:snapshot+1]
    for event in events:
        identifier = event['payload'].get('attempt_id')
        if identifier is not None:
            events_by_attempt.setdefault(identifier, []).append(event)
    for attempt in store.attempts().values():
        by_run.setdefault(attempt['run_id'], []).append(attempt)
    cells = sorted((study['switching_recipe']['scenario']['scenario_id'], cell['replicate_id'], cell['fit_K'], study, cell)
                   for study in studies for cell in study['cells'])
    expected, rows, lineage, consumed = [], [], [], 0
    definition = IdentifiabilitySpec.from_dict(studies[0]['switching_recipe']['identifiability_spec'])
    arms = {}
    for _, _, k, study, cell in cells:
        _require(k not in arms or arms[k] == cell['arm_id'], 'owner source arm family changed')
        arms[k] = cell['arm_id']
    balances = _snapshot_balances(snapshot_events, arms)
    _require(digest(report.get('arm_balances')) == digest(balances), 'owner source snapshot balances differ')
    plugin = switching_plugin()
    for scenario, replicate, k, study, cell in cells:
        run_id = _run_id(study, cell, study_hashes[study['study_id']])
        related = by_run.get(run_id, [])
        _require(all(event['sequence'] <= events[snapshot]['sequence']
            for attempt in related for event in events_by_attempt.get(attempt['attempt_id'], [])
            if event['event_kind'] in ('ATTEMPT', 'RESERVE', 'SETTLE', 'WORKER_TREE_STOPPED')),
            'owner source attempts occurred after the claimed snapshot')
        history = _cost_history(store, study, cell, related, events_by_attempt)
        successful = [item for item in related if item['state'] == 'SUCCEEDED']
        _require(len(successful) <= 1, 'owner cell has multiple successful outcomes')
        selected = successful[0] if successful else related[-1] if related else None
        entry = {'scenario_id': scenario, 'replicate_id': replicate, 'K': k,
            'study_id': study['study_id'], 'cell_hash': digest(cell), 'run_id': run_id,
            'selected_attempt_id': None if selected is None else selected['attempt_id'], 'attempts': history}
        lineage.append(entry)
        failures, success_result = {}, None
        # Match collector order: every failed reference artifact, then success.
        ordered = [item for item in related if item['state'] == 'FAILED' and item.get('artifact_id')]
        ordered += [item for item in related if item['state'] == 'SUCCEEDED']
        for attempt in ordered:
            artifact_id = attempt['artifact_id']
            metadata = store.manifest('artifact-'+artifact_id)
            size = metadata['size_bytes']
            _require(type(size) is int and 0 <= size <= MAXIMUM_RESULT_BYTES
                     and consumed+size <= maximum_total_bytes, 'owner prefix source byte quota exceeded')
            grant = authorizations[study['study_id']]
            content = store.read_artifact(artifact_id, purpose='evaluate', authorization=grant)
            consumed += len(content)
            result = json.loads(content)
            _require(len(content) == size and hashlib.sha256(content).hexdigest() == artifact_id
                     and content == encode(result), 'owner authorized prefix source bytes differ')
            if attempt['state'] == 'SUCCEEDED':
                verified_reuse(store, attempt, study, cell, plugin)
            else:
                verified_failure(store, attempt, study, cell, plugin)
                failures[attempt['attempt_id']] = result['failure']
                next(value for value in history if value['attempt_id'] == attempt['attempt_id'])['failure'] = disclose_failure(
                    result, study, artifact_id)
            if attempt['state'] == 'SUCCEEDED':
                success_result = result
            projection = _projection(result, study, cell)
            if projection is not None:
                expected.append(_receipt(store, study, cell, attempt, metadata, grant, history, projection,
                                         events_by_attempt.get(attempt['attempt_id'], [])))
        if successful:
            rows.append(_success_row(definition, study, cell, success_result, selected['artifact_id']))
            entry['initialization_diagnostics'] = _initialization_diagnostics(success_result)
        else:
            rows.append(_non_success_row(store, (scenario, replicate, k), selected, failures, balances[k]))
    _require(digest(claimed) == digest(expected), 'prefix source/artifact/grant/all-cost roster differs')
    _require(digest(report.get('rows')) == digest(rows), 'owner canonical source rows differ')
    _require(digest(report.get('lineage')) == digest(lineage), 'owner complete source/failure/cost lineage differs')
    _require(report.get('read_bytes') == consumed and type(report.get('read_bytes')) is int,
             'owner complete source read byte count differs')
    return {'rows': rows, 'lineage': lineage, 'proofs': expected}
