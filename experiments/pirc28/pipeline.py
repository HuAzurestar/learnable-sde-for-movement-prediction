"""Frozen synthetic recipe executor, including held-out evaluator-only truth.

This layer has no store writer or budget authority. The managed worker supplies
the admitted recipe, shared poll, progress sink and accepted-iteration checkpoint.
"""

from dataclasses import replace
import copy
import math

import numpy as np

from estimation.switching import fit_switching, select_initialization
from inference.switching import InferenceSpec, filter_switching, forecast_switching
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingError, SwitchingSDEModel, finite_array, psd_array
from .synthetic import ScenarioSpec, generate_scenario
from .failures import (NUMERICAL_EXCEPTIONS, NumericalCellFailure, numerical_diagnostic,
                       valid_numerical_diagnostic, POST_REFERENCE_STAGES)
from .qualification import IdentifiabilitySpec, forecast_coverage, jump_diagnostics, mode_diagnostics, parameter_recovery
from .approximation import ApproximationSpec, RECIPE_SCHEMA
from .approximation_checks import reference_design_preflight


def validate_recipe(recipe, k):
    keys = {'schema_version', 'scenario', 'initializations', 'inference', 'train_paths',
        'selection_paths', 'test_paths', 'forecast_paths', 'forecast_origin_index', 'horizon_indices',
        'prior_mean', 'prior_covariance'}
    if type(recipe) is dict and recipe.get('schema_version') == RECIPE_SCHEMA:
        keys = keys | {'approximation_spec'}
    if (type(recipe) is not dict or set(recipe) not in (keys, keys | {'identifiability_spec'})
            or recipe['schema_version'] not in ('pirc28-synthetic-recipe-v1', RECIPE_SCHEMA)
            or type(k) is not int or k not in (1, 2)):
        raise SwitchingError('CONTRACT_MISMATCH', 'synthetic recipe schema or first-round K differs')
    scenario = ScenarioSpec.from_dict(recipe['scenario'])
    scenario.validated()
    if 'identifiability_spec' in recipe:
        qualification = IdentifiabilitySpec.from_dict(recipe['identifiability_spec'])
        if scenario.scenario_id not in qualification.scenario_axes:
            raise SwitchingError('IDENTITY_CONFLICT', 'recipe scenario is outside the declared qualification matrix')
    if type(recipe['inference']) is not dict:
        raise SwitchingError('CONTRACT_MISMATCH', 'inference configuration is required')
    try:
        inference = InferenceSpec(**recipe['inference']).validate()
    except TypeError as error:
        raise SwitchingError('CONTRACT_MISMATCH', 'inference fields differ') from error
    for name, upper in (('train_paths', 1000), ('selection_paths', 1000), ('test_paths', 64), ('forecast_paths', 1024)):
        if type(recipe[name]) is not int or not 1 <= recipe[name] <= upper:
            raise SwitchingError('RESOURCE_LIMIT', name + ' exceeds the bounded recipe quota')
    models = recipe['initializations'].get(str(k))
    if type(models) is not list or not 1 <= len(models) <= inference.max_initializations:
        raise SwitchingError('CONTRACT_MISMATCH', 'registered initializations are missing or excessive')
    models = [SwitchingSDEModel.from_dict(value) for value in models]
    if any(model.n_modes != k for model in models):
        raise SwitchingError('CONTRACT_MISMATCH', 'initialization K differs from the arm')
    origin, horizons = recipe['forecast_origin_index'], recipe['horizon_indices']
    if (type(origin) is not int or not 0 <= origin < len(scenario.times)-1 or type(horizons) is not list
            or not horizons or any(type(index) is not int or not origin < index < len(scenario.times) for index in horizons)
            or horizons != sorted(set(horizons))):
        raise SwitchingError('CONTRACT_MISMATCH', 'forecast origin or horizon indices differ')
    mean = finite_array(recipe['prior_mean'], (4,), 'frozen filtering prior mean')
    covariance = psd_array(recipe['prior_covariance'], (4, 4), 'frozen filtering prior covariance')
    if recipe['schema_version'] == RECIPE_SCHEMA:
        approximation = ApproximationSpec.from_dict(recipe['approximation_spec'])
        approximation.prefix_preflight(inference, scenario.times, origin,
                                       max(model.rate_bounds[1] for model in models))
    return scenario, models, inference, mean, covariance


def recipe_allocation(recipe, k):
    """Additive live source/forecast/reference envelope, metadata only.

    No phase reuse is assumed: keep the original fit and forecast envelopes and
    add retained/generated source buffers, numeric metadata and the complete
    sequential reference envelope. Four-vector padding uses the existing shared
    count vocabulary, while the actual observation count remains explicit.
    This declaration does not reserve a budget or admit/execute the reference.
    """
    scenario, models, inference, _, _ = validate_recipe(recipe, k)
    paths = max(recipe['train_paths'], recipe['selection_paths'], recipe['test_paths'] * recipe['forecast_paths'])
    # The quota is a real algorithm refusal boundary, not an optimistic estimate.
    steps = max(inference.max_substeps, scenario.max_jumps, len(scenario.times),
                len(models) * (inference.max_iterations+1) + 3)
    config = {'paths': paths, 'steps': steps, 'mixtures': k, 'components': len(models),
              'dtype': 'float64', 'device': 'cpu', 'noise_dim': 2}
    # Fit histories and forecast sample paths are separate live allocations.
    # Do not multiply the large forecast sample count into each EM history.
    config['components'] = max(recipe['train_paths'], recipe['selection_paths']) * len(models)
    source_paths = sum(recipe[name] for name in ('train_paths', 'selection_paths', 'test_paths'))
    n, d = len(scenario.times), len(scenario.H)
    # Observation/state/mode arrays, two jump histories and dwell, including
    # generation lists -> arrays -> retained copies. Count integer slots at the
    # same conservative eight-byte width as float64; no truth enters inference.
    source_data = 3*source_paths*(n*(d+4+1)+2*(scenario.max_jumps+1)+4)+4*n+d*4+d*d
    # Model/accepted-state/selection diagnostics and fixed bounded line searches.
    fit_metadata = 8*len(models)*(k*k+14*k+inference.max_iterations*(1+inference.line_search_steps)+k)
    # Retained per-path forecasts, actual jump/mode histories and serialization
    # copies are ADDITIONAL to the original continuous forecast workspace.
    forecast_retained = 3*recipe['test_paths']*(recipe['forecast_paths']*(
        5*(len(recipe['horizon_indices'])+1)+2*(scenario.max_jumps+1)+1)
        +4*k*(len(recipe['horizon_indices'])+1))
    core = None
    if recipe['schema_version'] == RECIPE_SCHEMA and k == 2:
        core = reference_design_preflight(ApproximationSpec.from_dict(recipe['approximation_spec']),
            inference, scenario.times, recipe['forecast_origin_index'], recipe['test_paths'],
            max(model.rate_bounds[1] for model in models), d)['core_plan']
    reference_elements = core['tensor_elements'] if core is not None else 0
    pool_elements = source_data+fit_metadata+forecast_retained+reference_elements
    inputs = {'observations': source_paths*n, 'workspace_vectors': (pool_elements+3)//4, 'state_dim': 4}
    fit_elements = 24*config['components']*steps*k*k*16
    forecast_elements = paths*steps*4
    elements = fit_elements+forecast_elements+4*inputs['workspace_vectors']
    return {'schema_version': 'pirc28-recipe-allocation-v1', 'config': config, 'inputs': inputs,
        'source_data_elements': source_data, 'fit_metadata_elements': fit_metadata,
        'retained_forecast_elements': forecast_retained, 'reference_core_plan': core,
        'reference_elements': reference_elements, 'padded_pool_elements': 4*inputs['workspace_vectors'],
        'fit_workspace_elements': fit_elements, 'forecast_workspace_elements': forecast_elements,
        'combined_tensor_elements': elements, 'combined_tensor_bytes': elements*8,
        'combination_rule': 'additive-concurrent-envelope; no-phase-reuse',
        'resource_admitted': False, 'managed_executor_implemented': False}


def recipe_resources(recipe, k):
    allocation = recipe_allocation(recipe, k)
    return allocation['config'], allocation['inputs']


def energy_score(samples, target):
    samples, target = np.asarray(samples), np.asarray(target)
    first = np.linalg.norm(samples-target, axis=1).mean()
    total = 0.0
    # Bounded sample-pair blocks; never an all-trajectory similarity matrix.
    for start in range(0, len(samples), 32):
        total += np.linalg.norm(samples[start:start+32, None] - samples[None], axis=-1).sum()
    return float(first - 0.5 * total / len(samples)**2)


def validate_failed_fit(value, initial_model, index, observations_hash, probability, means, covariance, inference):
    """Bind retained failure/accepted progress before skipping it on recovery."""
    identifier = 'init-'+str(index)
    if (not valid_numerical_diagnostic(value, 'fit', identifier)
            or set(value) != {'schema_version', 'state', 'initialization_id', 'stage', 'error_code',
                             'exception_type', 'initial_model_hash', 'last_accepted_state'}
            or value['initial_model_hash'] != digest(initial_model.to_dict())):
        raise SwitchingError('IDENTITY_CONFLICT', 'failed initialization checkpoint identity differs')
    state = value['last_accepted_state']
    if state is None:
        return
    identity = {'data_hash': observations_hash, 'spec_hash': digest(inference.to_dict()),
        'initial_distribution_hash': digest([np.asarray(item).tolist() for item in (probability, means, covariance)]),
        'initialization_id': identifier, 'initial_model_hash': digest(initial_model.to_dict())}
    if (type(state) is not dict or set(state) != {'schema_version', 'identity', 'model', 'objective_trace',
            'rejected_objectives', 'completed_iterations', 'complete_reason'}
            or state['schema_version'] != 'pirc28-em-checkpoint-v1' or state['identity'] != identity):
        raise SwitchingError('IDENTITY_CONFLICT', 'failed initialization accepted-state identity differs')
    completed, trace, rejected = state['completed_iterations'], state['objective_trace'], state['rejected_objectives']
    if (type(completed) is not int or not 0 <= completed <= inference.max_iterations
            or type(trace) is not list or type(rejected) is not list or len(trace) != completed+1
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in trace+rejected)
            or np.any(np.diff(trace) < 0)
            or state['complete_reason'] not in {None, 'MAX_ITERATIONS', 'REJECTED_STEP', 'CONVERGED_APPROXIMATE_OBJECTIVE'}):
        raise SwitchingError('CONTRACT_MISMATCH', 'failed initialization accepted progress differs')
    model = SwitchingSDEModel.from_dict(state['model'])
    if model.n_modes != initial_model.n_modes or model.rate_bounds != initial_model.rate_bounds:
        raise SwitchingError('IDENTITY_CONFLICT', 'failed initialization model family differs')


def execute_recipe(recipe, k, seed, *, check_budget, progress_callback=None, resume_state=None,
                   fixture_approximation_executor=None):
    context = {'stage': 'validation'}
    try:
        return _execute_recipe(recipe, k, seed, check_budget=check_budget, progress_callback=progress_callback,
                               resume_state=resume_state, context=context,
                               fixture_approximation_executor=fixture_approximation_executor)
    except NumericalCellFailure:
        raise
    except NUMERICAL_EXCEPTIONS as error:
        if context['stage'] == 'validation':
            raise
        cause = numerical_diagnostic(error, context['stage'], None)
        failure = NumericalCellFailure(context['stage'], context['initializations'], context['selection'],
                                       error_code=cause['error_code'], cause=cause)
        failure.diagnostic.update(recipe_hash=context['recipe_hash'], generated_inputs=context['generated_inputs'])
        if context['stage'] == 'approximation':
            failure.diagnostic['approximation_checks'] = None
        elif context['stage'] in POST_REFERENCE_STAGES and 'approximation_checks' in context:
            failure.diagnostic['approximation_checks'] = context['approximation_checks']
        raise failure from error


def _execute_recipe(recipe, k, seed, *, check_budget, progress_callback, resume_state, context,
                    fixture_approximation_executor):
    scenario, initial_models, inference, prior_mean, prior_covariance = validate_recipe(recipe, k)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise SwitchingError('CONTRACT_MISMATCH', 'cell seed must be unsigned 64-bit')
    if fixture_approximation_executor is not None and (not callable(fixture_approximation_executor)
            or recipe['schema_version'] != RECIPE_SCHEMA):
        raise ResearchError('CONTRACT_MISMATCH', 'explicit V2 engineering executor required')
    if recipe['schema_version'] == RECIPE_SCHEMA and fixture_approximation_executor is None:
        # Metadata/resource preflight is not an implemented diagnostic adapter.
        # Refuse before any generation/fitting; do not create a numerical-failure
        # artifact for a software implementation gap or invent successful proof.
        raise ResearchError('NEEDS_IMPLEMENTATION', 'managed approximation executor/evidence adapter is not implemented')
    if fixture_approximation_executor is not None:
        from application.research_registry import plan_resources
        from .runtime import switching_plugin
        allocation = recipe_allocation(recipe, k)
        plan_resources(switching_plugin().registry_entry, allocation['config'], allocation['inputs'], matrix_cells=1)
    identity = digest({'recipe': recipe, 'K': k, 'seed': seed})
    context.update(stage='generation', recipe_hash=identity, generated_inputs=[], selection=None,
        initializations=[{'initialization_id': 'init-'+str(index), 'state': 'NOT_RUN',
                          'initial_model_hash': digest(model.to_dict())} for index, model in enumerate(initial_models)])
    rng = np.random.default_rng(seed)
    split_seeds = [int(value) for value in rng.integers(0, 2**63, size=3, dtype=np.int64)]
    rng_state = {'base_pcg64': rng.bit_generator.state, 'split_seeds': split_seeds,
        'child_stream_rule': 'PCG64-per-path-role-v1', 'resume_strategy': 'recompute-role-streams'}
    fit_states = [None] * len(initial_models)
    fit_failures = [None] * len(initial_models)
    if resume_state is not None:
        required = {'schema_version', 'recipe_hash', 'fit_states', 'rng_state'}
        legacy = type(resume_state) is dict and resume_state.get('schema_version') == 'pirc28-pipeline-checkpoint-v1'
        if (type(resume_state) is not dict or set(resume_state) != (required if legacy else required | {'fit_failures'})
                or resume_state['schema_version'] not in {'pirc28-pipeline-checkpoint-v1', 'pirc28-pipeline-checkpoint-v2'}
                or resume_state['recipe_hash'] != identity or resume_state['rng_state'] != rng_state
                or type(resume_state['fit_states']) is not list
                or len(resume_state['fit_states']) != len(fit_states)):
            raise SwitchingError('IDENTITY_CONFLICT', 'pipeline checkpoint recipe, cell or random streams differ')
        fit_states = copy.deepcopy(resume_state['fit_states'])
        if not legacy:
            if type(resume_state['fit_failures']) is not list or len(resume_state['fit_failures']) != len(fit_states):
                raise SwitchingError('IDENTITY_CONFLICT', 'failed initialization checkpoint count differs')
            fit_failures = copy.deepcopy(resume_state['fit_failures'])

    def progress():
        if progress_callback is not None:
            progress_callback(copy.deepcopy({'schema_version': 'pirc28-pipeline-checkpoint-v2', 'recipe_hash': identity,
                'fit_states': fit_states, 'fit_failures': fit_failures, 'rng_state': rng_state}))

    progress()
    batches, truths, generated = [], [], []
    context['generated_inputs'] = generated
    for role_index, role in enumerate(('train', 'selection', 'test')):
        check_budget()
        registered = replace(scenario, seed=split_seeds[role_index], sample_count=recipe[role + '_paths'],
                             scenario_id=scenario.scenario_id + '-' + role)
        observation, truth, manifest = generate_scenario(registered, check_budget=check_budget)
        batches.append(observation)
        truths.append(truth)
        generated.append({'role': role, 'scenario_hash': manifest['scenario_hash'],
                          'observations_hash': manifest['observations_hash'], 'truth_hash': manifest['truth_hash']})
    probability, means, covariance = np.ones(k)/k, np.tile(prior_mean, (k, 1)), np.tile(prior_covariance, (k, 1, 1))
    fits, histories = [], []
    context.update(stage='fit', initializations=histories)
    for index, initial_model in enumerate(initial_models):
        check_budget()
        failure = fit_failures[index]
        if failure is not None:
            validate_failed_fit(failure, initial_model, index, digest(batches[0].to_dict()),
                                probability, means, covariance, inference)
            if failure['last_accepted_state'] != fit_states[index]:
                raise SwitchingError('IDENTITY_CONFLICT', 'failed initialization checkpoint progress differs')
            histories.append(failure)
            continue
        def fit_progress(state):
            fit_states[index] = state
            progress()
        try:
            fit = fit_switching(batches[0], initial_model, probability, means, covariance, inference,
                initialization_id='init-' + str(index), check_budget=check_budget, progress_callback=fit_progress,
                resume_state=fit_states[index])
        except NUMERICAL_EXCEPTIONS as error:
            failure = numerical_diagnostic(error, 'fit', 'init-'+str(index))
            failure.update(initial_model_hash=digest(initial_model.to_dict()),
                           last_accepted_state=copy.deepcopy(fit_states[index]))
            fit_failures[index] = failure
            histories.append(failure)
            progress()
            continue
        # The registered output preserves parameter/iteration diagnostics and
        # test posteriors; completed training histories can be recomputed.
        fit.filtered = ()
        fits.append(fit)
        histories.append({**fit.to_dict(include_posteriors=False), 'state': 'SUCCEEDED',
                          'initial_model_hash': digest(initial_model.to_dict())})
    def complete_selection(selection):
        candidates = {value['initialization_id']: value for value in selection['candidates']}
        selection['candidates'] = [candidates.get(value['initialization_id'], {**value, 'score': None}) for value in histories]
        return selection
    try:
        if not fits:
            raise NumericalCellFailure('fit', histories)
        context['stage'] = 'selection'
        selected, selection = select_initialization(fits, batches[1], probability, means, covariance,
            inference, split_role='selection', check_budget=check_budget)
    except NumericalCellFailure as error:
        error.diagnostic.update(initializations=histories, recipe_hash=identity, generated_inputs=generated)
        if error.diagnostic['selection'] is not None:
            complete_selection(error.diagnostic['selection'])
        raise
    selection = complete_selection(selection)
    context['selection'] = selection
    approximation_checks, approximation_consistency = None, None
    if recipe['schema_version'] == RECIPE_SCHEMA and k == 2:
        from .approximation_evidence import reference_seed, validate_prefix_numbers
        context['stage'] = 'approximation'
        approximation = ApproximationSpec.from_dict(recipe['approximation_spec'])
        frozen_seed = reference_seed(recipe, seed)
        approximation_checks = fixture_approximation_executor(batches[2], selected.model, probability, means,
            covariance, inference, approximation, origin_index=recipe['forecast_origin_index'],
            seed=frozen_seed, check_budget=check_budget)
        approximation_consistency = validate_prefix_numbers(approximation_checks, batches[2], selected.model,
            probability, means, covariance, inference, approximation, origin_index=recipe['forecast_origin_index'],
            seed=frozen_seed, check_budget=check_budget)
        if approximation_checks['state'] != 'NUMERICAL_CHECKS_PASSED':
            cause = numerical_diagnostic(SwitchingError('INFERENCE_INAPPLICABLE', 'fixed prefix gates failed'),
                                         'approximation', None)
            failure = NumericalCellFailure('approximation', histories, selection, cause=cause)
            failure.diagnostic.update(recipe_hash=identity, generated_inputs=generated,
                                      approximation_checks=approximation_checks)
            raise failure
        # Preserve the complete passed prefix when later cell work fails.
        # This is diagnostic retention, not successful metrics or authority.
        context['approximation_checks'] = approximation_checks
    forecasts, scores, test_objectives, posterior_probabilities, coverage = [], [], [], [], []
    origin_index, horizon_indices = recipe['forecast_origin_index'], recipe['horizon_indices']
    offsets = [float(batches[2].times[index] - batches[2].times[origin_index]) for index in horizon_indices]
    for path_index in range(recipe['test_paths']):
        check_budget()
        context['stage'] = 'test-inference'
        filtered = filter_switching(batches[2], selected.model, probability, means, covariance, inference,
                                   path_index=path_index, check_budget=check_budget)
        context['stage'] = 'forecast'
        forecast = forecast_switching(selected.model, filtered.posteriors[origin_index], offsets,
            sample_count=recipe['forecast_paths'], seed=split_seeds[2]+path_index,
            max_jumps=scenario.max_jumps, check_budget=check_budget)
        context['stage'] = 'evaluation'
        samples = np.asarray(forecast['samples'])
        if 'identifiability_spec' in recipe:
            qualification = IdentifiabilitySpec.from_dict(recipe['identifiability_spec'])
            coverage.append(forecast_coverage(samples,
                truths[2].states[path_index, horizon_indices], nominal=qualification.coverage_nominal))
        for horizon_index, time_index in enumerate(horizon_indices):
            scores.append(energy_score(samples[:, horizon_index, :2], truths[2].states[path_index, time_index, :2]))
        forecasts.append(forecast)
        test_objectives.append(filtered.log_likelihood / len(batches[2].times))
        posterior_probabilities.append([posterior.probability.tolist() for posterior in filtered.posteriors])
    metrics = {'position_energy_score_d2': float(np.mean(scores)),
        'approximate_test_log_likelihood_per_observation': float(np.mean(test_objectives)),
        'posterior_min_occupancy': float(selected.occupancy.min()), 'mode_collapse': int(selected.collapsed),
        'test_path_count': recipe['test_paths'], 'configured_K': k}
    units = {'position_energy_score_d2': 'm', 'approximate_test_log_likelihood_per_observation': 'nats/observation',
        'posterior_min_occupancy': '1', 'mode_collapse': '1', 'test_path_count': 'count', 'configured_K': 'count'}
    # Truth is evaluator-only, and only synthetic modes can have a Brier score.
    unavailable, legacy_mode_diagnostics = {}, {}
    if scenario.model['K'] == k == 2:
        probabilities = np.asarray(posterior_probabilities)
        truth = np.eye(2)[truths[2].modes]
        losses = [np.mean(np.sum((probabilities[..., order]-truth)**2, axis=-1)) for order in ([0, 1], [1, 0])]
        legacy_mode_diagnostics['permutation_invariant_mode_brier'] = {'value': float(min(losses)), 'unit': '1',
            'role': 'synthetic-filtering-diagnostic', 'label_matching': 'minimum-over-two-permutations'}
    else:
        unavailable['permutation_invariant_mode_brier'] = 'truth/model K differ or single-mode control'
    recovery_diagnostics = None
    if 'identifiability_spec' in recipe:
        qualification = IdentifiabilitySpec.from_dict(recipe['identifiability_spec'])
        recovery_diagnostics = {'identifiability_spec_hash': digest(qualification.to_dict()),
            'state': 'NEEDS_PREREGISTRATION' if qualification.missing_fields else 'UNQUALIFIED_CELL_DIAGNOSTICS',
            'missing_fields': qualification.missing_fields,
            'position_marginal_coverage_error': abs(float(np.mean([value['position_marginal_coverage'] for value in coverage]))
                                                   - qualification.coverage_nominal),
            'position_marginal_coverage_nominal': qualification.coverage_nominal,
            'sampling_unit': 'one-independent-replicate; paths-and-points-within-cell-not-replicates'}
        recovery_diagnostics['jump_diagnostics'] = jump_diagnostics(forecasts, truths[2].jump_times,
            truths[2].modes, scenario.times, origin_index, horizon_indices)
        if scenario.model['K'] == k == 2:
            truth_model = SwitchingSDEModel.from_dict(scenario.model)
            # Scenario labels relabel evaluator modes/dwell, not physical model
            # dynamics. Match against the model expressed in those truth labels.
            label_order = np.argsort(scenario.label_permutation or tuple(range(k))).tolist()
            truth_model = truth_model.permuted(label_order)
            matched = parameter_recovery(selected.model, truth_model,
                                         qualification.parameter_scales)
            modes = mode_diagnostics(np.asarray(posterior_probabilities), truths[2].modes,
                matched['label_order'], truths[2].dwell_by_mode, float(scenario.times[-1]),
                bins=qualification.calibration_bins)
            recovery_diagnostics.update(parameter_recovery=matched, mode_diagnostics=modes,
                posterior_occupancy=modes['posterior_occupancy'], collapsed=bool(selected.collapsed),
                occupied_K=int(sum(value >= inference.min_occupancy for value in selected.occupancy)))
        else:
            recovery_diagnostics['unavailable'] = 'parameter/mode recovery is unavailable when truth/fit K differ or K=1'
    fit_payload = {'model': selected.model.to_dict(), 'selection': selection,
        'initializations': histories,
        'generated_inputs': generated, 'recipe_hash': identity, 'unavailable_metrics': unavailable,
        'mode_diagnostics': legacy_mode_diagnostics,
        'interpretation_level': 'synthetic-statistical-only', 'qualified_identifiability': False}
    if recovery_diagnostics is not None:
        fit_payload['recovery_diagnostics'] = recovery_diagnostics
    if recipe['schema_version'] == RECIPE_SCHEMA:
        fit_payload.update(approximation_checks=approximation_checks,
            approximation_consistency=approximation_consistency,
            approximation_execution_scope='engineering-only; not-managed-authority-proof')
    # Shared case views consume this registered first-path preview. The full
    # synthetic paths remain separately named, rather than confused with n blocks.
    preview = forecasts[0]
    forecast_payload = {**preview, 'per_path': forecasts, 'posterior_probabilities': posterior_probabilities,
        'preview': {'case_selection_rule': 'registered-first-synthetic-test-path',
            'sample_selection_rule': 'all-generated-samples', 'sample_ids': list(range(recipe['forecast_paths'])),
            'n_samples': recipe['forecast_paths'], 'generation_version': 'pirc28-event-driven-v1'}}
    check_budget()
    return {'metrics': metrics, 'metric_units': units, 'forecast': forecast_payload,
            'fit': fit_payload, 'source_schema': 'pirc28-synthetic-result-v1'}
