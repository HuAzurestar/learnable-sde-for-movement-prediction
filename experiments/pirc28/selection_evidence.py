"""Independent selection-input/score replay, NOT training or store authority.

Checks the fixed roster of disclosed fitted models using selection observations
only. Does not rerun GEM, authenticate fit failures or execute reference batches.
The original-arm supervisor must still bind source, artifacts, costs and grants.
"""

from dataclasses import replace
import math
import re

import numpy as np

from application.research_registry import plan_resources
from inference.switching import filter_switching
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingSDEModel
from .approximation_evidence import _array, _close, _integer, _keys, _number, _require, _same_declaration
from .evidence import validate_initialization_history
from .failures import NUMERICAL_EXCEPTIONS, numerical_diagnostic
from .pipeline import recipe_allocation, validate_recipe
from .runtime import switching_plugin
from .synthetic import generate_scenario


SUCCESS_FIELDS = {'schema_version', 'model', 'model_hash', 'initialization_id', 'objective', 'accuracy',
    'objective_trace', 'rejected_objectives', 'stopping_reason', 'posterior_occupancy', 'configured_K',
    'occupied_K', 'collapsed', 'inference_spec', 'indistinguishable_modes', 'resume_level', 'state', 'initial_model_hash'}
FAILURE_FIELDS = {'schema_version', 'state', 'initialization_id', 'stage', 'error_code', 'exception_type',
                  'initial_model_hash', 'last_accepted_state'}


def _trace(value, maximum, poll):
    _require(type(value) is list and len(value) <= maximum, 'bounded objective history required')
    for number in value:
        poll(); _number(number)


def _model(value, initial, k, poll):
    _keys(value, initial.to_dict(), 'fitted model')
    _integer(value['K'], k, k)
    for name, shape in (('Q', (k, k)), ('A', (k, 2, 2)), ('B', (k, 2, 2)),
                        ('a', (k, 2)), ('L', (k, 2, 2)), ('rate_bounds', (2,))):
        _array(value[name], shape, poll)
    model = SwitchingSDEModel.from_dict(value)
    _require(model.rate_bounds == initial.rate_bounds and digest(model.to_dict()) == digest(value),
             'fitted model canonical identity or allowed rate bounds differ')
    return model


def _history(history, initial, index, k, inference, poll):
    _require(type(history) is dict and history.get('initialization_id') == 'init-'+str(index),
             'complete ordered fitted initializer roster required')
    failed = history.get('state') == 'FAILED'
    _keys(history, FAILURE_FIELDS if failed else SUCCESS_FIELDS, 'initializer history')
    if failed:
        state = history['last_accepted_state']
        if state is not None:
            _keys(state, {'schema_version', 'identity', 'model', 'objective_trace', 'rejected_objectives',
                          'completed_iterations', 'complete_reason'}, 'accepted failure progress')
            _keys(state['identity'], {'data_hash', 'spec_hash', 'initial_distribution_hash',
                'initialization_id', 'initial_model_hash'}, 'accepted-state identity')
            _integer(state['completed_iterations'], 0, inference.max_iterations)
            _trace(state['objective_trace'], inference.max_iterations+1, poll)
            _trace(state['rejected_objectives'], inference.max_iterations*inference.line_search_steps, poll)
            _model(state['model'], initial, k, poll)
        return None
    _trace(history['objective_trace'], inference.max_iterations+1, poll)
    _trace(history['rejected_objectives'], inference.max_iterations*inference.line_search_steps, poll)
    _require(history['objective'] == 'split-step-moment-matched-observation-log-likelihood'
        and history['accuracy'] == 'approximate' and history['resume_level'] == 'numerical-tolerance',
        'registered approximate objective or recovery declaration differs')
    _same_declaration(history['inference_spec'], inference.to_dict(), poll)
    model = _model(history['model'], initial, k, poll)
    _integer(history['configured_K'], k, k); _integer(history['occupied_K'], 0, k)
    _require(type(history['indistinguishable_modes']) is bool, 'typed indistinguishable-mode declaration required')
    occupancy = _array(history['posterior_occupancy'], (k,), poll)
    _require(np.all(occupancy >= 0) and math.isclose(float(occupancy.sum()), 1, abs_tol=1e-10, rel_tol=0),
             'bounded normalized training occupancy declaration required')
    _require(history['occupied_K'] == int(np.sum(occupancy >= inference.min_occupancy)),
             'occupied-mode declaration differs')
    return model


def validate_selected_fit(fit, recipe, cell_seed, *, k, check_budget):
    """Return the numerically selected model and bounded non-authority receipt.

    Evaluator-only fit diagnostics are not selection inputs and are not validated
    by this leaf. Its hash binds exactly the recipe/roster/models/selection/input
    projection, not a claim that every field of the parent artifact was checked.
    """
    _require(callable(check_budget), 'explicit selection verifier budget poll required')
    _integer(cell_seed, 0, 2**64-1); _integer(k, 1, 2)
    check_budget()
    scenario, initial_models, inference, mean, covariance = validate_recipe(recipe, k)
    allocation = recipe_allocation(recipe, k)
    plan_resources(switching_plugin().registry_entry, allocation['config'], allocation['inputs'], matrix_cells=1)
    _require(type(fit) is dict and fit.get('recipe_hash') == digest({'recipe': recipe, 'K': k, 'seed': cell_seed})
        and fit.get('interpretation_level') == 'synthetic-statistical-only'
        and fit.get('qualified_identifiability') is False, 'expected recipe/cell or no-authority binding differs')
    generated = fit.get('generated_inputs')
    _require(type(generated) is list and len(generated) == 3, 'complete ordered source role roster required')
    seeds = np.random.default_rng(cell_seed).integers(0, 2**63, size=3, dtype=np.int64)
    sources = []
    for index, role in enumerate(('train', 'selection', 'test')):
        check_budget()
        source = replace(scenario, seed=int(seeds[index]), sample_count=recipe[role+'_paths'],
                         scenario_id=scenario.scenario_id+'-'+role)
        _keys(generated[index], {'role', 'scenario_hash', 'observations_hash', 'truth_hash'}, 'source role')
        _require(generated[index]['role'] == role and generated[index]['scenario_hash'] == source.scenario_hash,
                 'expected source role/seed/scenario differs')
        _require(all(type(generated[index][name]) is str and re.fullmatch('[0-9a-f]{64}', generated[index][name])
                     for name in ('observations_hash', 'truth_hash')), 'bounded source hash required')
        sources.append(source)
    histories = fit.get('initializations')
    _require(type(histories) is list and len(histories) == len(initial_models), 'complete fixed initializer roster required')
    models = [_history(history, initial, index, k, inference, check_budget)
              for index, (history, initial) in enumerate(zip(histories, initial_models))]
    selection_receipt = fit.get('selection')
    _keys(selection_receipt, {'schema_version', 'split_role', 'selection_hash', 'rule',
                             'selected_initialization', 'candidates'}, 'selection receipt')
    _require(selection_receipt['schema_version'] == 'pirc28-initialization-selection-v1', 'selection receipt version differs')
    candidates = selection_receipt['candidates']
    _require(type(candidates) is list and len(candidates) == len(histories), 'complete selection candidate roster required')
    for history, claimed in zip(histories, candidates):
        if history['state'] == 'FAILED':
            _keys(claimed, FAILURE_FIELDS | {'score'}, 'retained failed fit candidate')
        else:
            fields = {'initialization_id', 'collapsed', 'stopping_reason', 'state', 'score'}
            if type(claimed) is dict and claimed.get('state') == 'FAILED':
                fields |= {'schema_version', 'stage', 'error_code', 'exception_type'}
            _keys(claimed, fields, 'selection candidate')
            _require(type(claimed['collapsed']) is bool, 'typed selection collapse declaration required')
            if claimed['state'] == 'SUCCEEDED':
                _number(claimed['score'])
    chosen = validate_initialization_history({'switching_recipe': recipe}, {'fit_K': k}, fit, generated)
    _require(chosen is not None and chosen['model'] == fit.get('model'), 'selected fit model differs from retained roster')
    check_budget()
    selection, _, manifest = generate_scenario(sources[1], check_budget=check_budget)
    _require(manifest['observations_hash'] == generated[1]['observations_hash']
        and manifest['truth_hash'] == generated[1]['truth_hash'], 'replayed selection source hashes differ')
    probability, means, covariances = np.ones(k)/k, np.tile(mean, (k, 1)), np.tile(covariance, (k, 1, 1))
    eligible, rows = [], []
    for index, (history, claimed, model) in enumerate(zip(histories, candidates, models)):
        check_budget()
        identifier = 'init-'+str(index)
        if model is None:
            # Do not retry a failed fit or pretend that its occurrence was
            # numerically authenticated by inspecting a failure record.
            rows.append({'initialization_id': identifier, 'state': 'FIT_FAILURE_RETAINED', 'score': None})
            continue
        try:
            score = 0.0
            for path in range(recipe['selection_paths']):
                check_budget()
                score += filter_switching(selection, model, probability, means, covariances, inference,
                    path_index=path, check_budget=check_budget).log_likelihood
            if not math.isfinite(score):
                from models.switching import SwitchingError
                raise SwitchingError('NONFINITE', 'replayed selection objective is nonfinite')
        except NUMERICAL_EXCEPTIONS as error:
            actual = numerical_diagnostic(error, 'selection', identifier)
            _require(all(claimed.get(name) == expected for name, expected in actual.items())
                and claimed.get('score') is None, 'replayed numerical selection failure differs')
            rows.append({**actual, 'score': None})
        else:
            _require(claimed.get('state') == 'SUCCEEDED', 'successful replay cannot be censored as selection failure')
            _close(claimed.get('score'), score)
            eligible.append((score, -index, identifier, model))
            rows.append({'initialization_id': identifier, 'state': 'SUCCEEDED', 'score': float(score)})
    _require(bool(eligible), 'no numerically eligible disclosed candidate')
    winner = max(eligible, key=lambda value: value[:2])
    _require(fit['selection']['selected_initialization'] == winner[2]
        and chosen['model_hash'] == digest(winner[3].to_dict()), 'replayed stable-tie selection winner differs')
    projection = {'recipe_hash': fit['recipe_hash'], 'model': fit['model'], 'selection': fit['selection'],
                  'initializations': histories, 'generated_inputs': generated}
    check_budget()
    return winner[3], {'schema_version': 'pirc28-selection-consistency-v1',
        'selection_projection_hash': digest(projection), 'recipe_cell_hash': fit['recipe_hash'],
        'selection_input_hash': digest(selection.to_dict()), 'selected_initialization': winner[2],
        'selected_model_hash': digest(winner[3].to_dict()), 'candidates': rows,
        'selection_input_replayed': True, 'selection_scores_recomputed': True,
        'fit_training_execution_verified': False, 'fit_failures_execution_verified': False,
        'full_generated_input_hashes_verified': False, 'reference_sampling_verified': False,
        'managed_evidence_bound': False, 'qualified_identifiability': False, 'real_data_authorized': False}
