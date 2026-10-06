"""Frozen synthetic recipe executor, including held-out evaluator-only truth.

This layer has no store writer or budget authority. The managed worker supplies
the admitted recipe, shared poll, progress sink and accepted-iteration checkpoint.
"""

from dataclasses import replace
import math

import numpy as np

from estimation.switching import fit_switching, select_initialization
from inference.switching import InferenceSpec, filter_switching, forecast_switching
from infrastructure.research_store import digest
from models.switching import SwitchingError, SwitchingSDEModel, finite_array, psd_array
from .synthetic import ScenarioSpec, generate_scenario


def validate_recipe(recipe, k):
    keys = {'schema_version', 'scenario', 'initializations', 'inference', 'train_paths',
        'selection_paths', 'test_paths', 'forecast_paths', 'forecast_origin_index', 'horizon_indices',
        'prior_mean', 'prior_covariance'}
    if (type(recipe) is not dict or set(recipe) != keys
            or recipe['schema_version'] != 'pirc28-synthetic-recipe-v1' or type(k) is not int or k not in (1, 2)):
        raise SwitchingError('CONTRACT_MISMATCH', 'synthetic recipe schema or first-round K differs')
    scenario = ScenarioSpec.from_dict(recipe['scenario'])
    scenario.validated()
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
    return scenario, models, inference, mean, covariance


def recipe_resources(recipe, k):
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
    inputs = {'observations': sum(recipe[name] for name in ('train_paths', 'selection_paths', 'test_paths')) * len(scenario.times),
              'state_dim': 4}
    return config, inputs


def energy_score(samples, target):
    samples, target = np.asarray(samples), np.asarray(target)
    first = np.linalg.norm(samples-target, axis=1).mean()
    total = 0.0
    # Bounded sample-pair blocks; never an all-trajectory similarity matrix.
    for start in range(0, len(samples), 32):
        total += np.linalg.norm(samples[start:start+32, None] - samples[None], axis=-1).sum()
    return float(first - 0.5 * total / len(samples)**2)


def execute_recipe(recipe, k, seed, *, check_budget, progress_callback=None, resume_state=None):
    scenario, initial_models, inference, prior_mean, prior_covariance = validate_recipe(recipe, k)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise SwitchingError('CONTRACT_MISMATCH', 'cell seed must be unsigned 64-bit')
    identity = digest({'recipe': recipe, 'K': k, 'seed': seed})
    rng = np.random.default_rng(seed)
    split_seeds = [int(value) for value in rng.integers(0, 2**63, size=3, dtype=np.int64)]
    rng_state = {'base_pcg64': rng.bit_generator.state, 'split_seeds': split_seeds,
        'child_stream_rule': 'PCG64-per-path-role-v1', 'resume_strategy': 'recompute-role-streams'}
    fit_states = [None] * len(initial_models)
    if resume_state is not None:
        required = {'schema_version', 'recipe_hash', 'fit_states', 'rng_state'}
        if (type(resume_state) is not dict or set(resume_state) != required
                or resume_state['schema_version'] != 'pirc28-pipeline-checkpoint-v1'
                or resume_state['recipe_hash'] != identity or resume_state['rng_state'] != rng_state
                or len(resume_state['fit_states']) != len(fit_states)):
            raise SwitchingError('IDENTITY_CONFLICT', 'pipeline checkpoint recipe, cell or random streams differ')
        fit_states = list(resume_state['fit_states'])

    def progress():
        if progress_callback is not None:
            progress_callback({'schema_version': 'pirc28-pipeline-checkpoint-v1', 'recipe_hash': identity,
                               'fit_states': list(fit_states), 'rng_state': rng_state})

    progress()
    batches, truths, generated = [], [], []
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
    fits = []
    for index, initial_model in enumerate(initial_models):
        def fit_progress(state):
            fit_states[index] = state
            progress()
        fit = fit_switching(batches[0], initial_model, probability, means, covariance, inference,
            initialization_id='init-' + str(index), check_budget=check_budget, progress_callback=fit_progress,
            resume_state=fit_states[index])
        # The registered output preserves parameter/iteration diagnostics and
        # test posteriors; completed training histories can be recomputed.
        fit.filtered = ()
        fits.append(fit)
    selected, selection = select_initialization(fits, batches[1], probability, means, covariance,
        inference, split_role='selection', check_budget=check_budget)
    forecasts, scores, test_objectives, posterior_probabilities = [], [], [], []
    origin_index, horizon_indices = recipe['forecast_origin_index'], recipe['horizon_indices']
    offsets = [float(batches[2].times[index] - batches[2].times[origin_index]) for index in horizon_indices]
    for path_index in range(recipe['test_paths']):
        check_budget()
        filtered = filter_switching(batches[2], selected.model, probability, means, covariance, inference,
                                   path_index=path_index, check_budget=check_budget)
        forecast = forecast_switching(selected.model, filtered.posteriors[origin_index], offsets,
            sample_count=recipe['forecast_paths'], seed=split_seeds[2]+path_index,
            max_jumps=scenario.max_jumps, check_budget=check_budget)
        samples = np.asarray(forecast['samples'])
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
    unavailable, mode_diagnostics = {}, {}
    if scenario.model['K'] == k == 2:
        probabilities = np.asarray(posterior_probabilities)
        truth = np.eye(2)[truths[2].modes]
        losses = [np.mean(np.sum((probabilities[..., order]-truth)**2, axis=-1)) for order in ([0, 1], [1, 0])]
        mode_diagnostics['permutation_invariant_mode_brier'] = {'value': float(min(losses)), 'unit': '1',
            'role': 'synthetic-filtering-diagnostic', 'label_matching': 'minimum-over-two-permutations'}
    else:
        unavailable['permutation_invariant_mode_brier'] = 'truth/model K differ or single-mode control'
    fit_payload = {'model': selected.model.to_dict(), 'selection': selection,
        'initializations': [fit.to_dict(include_posteriors=False) for fit in fits],
        'generated_inputs': generated, 'recipe_hash': identity, 'unavailable_metrics': unavailable,
        'mode_diagnostics': mode_diagnostics,
        'interpretation_level': 'synthetic-statistical-only', 'qualified_identifiability': False}
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
