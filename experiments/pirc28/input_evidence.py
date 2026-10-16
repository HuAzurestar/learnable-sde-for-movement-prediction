"""Complete deterministic generated-input replay, not fitting or authority.

Future test observations and evaluator-only truth are hashed, never passed to
inference. Replay uses the frozen generator; it is not an independent proof of
that generator's scientific model or of the reference particle execution.
"""

from dataclasses import replace
import re

import numpy as np

from application.research_registry import plan_resources
from infrastructure.research_store import digest
from .approximation_evidence import _integer, _keys, _require
from .pipeline import recipe_allocation, validate_recipe
from .runtime import switching_plugin
from .synthetic import generate_scenario


def _sources(generated, recipe, cell_seed, k):
    _integer(cell_seed, 0, 2**64-1)
    _integer(k, 1, 2)
    scenario, _, _, _, _ = validate_recipe(recipe, k)
    _require(type(generated) is list and len(generated) == 3, 'complete ordered generated role roster required')
    seeds = np.random.default_rng(cell_seed).integers(0, 2**63, size=3, dtype=np.int64)
    sources = []
    for index, role in enumerate(('train', 'selection', 'test')):
        source = replace(scenario, seed=int(seeds[index]), sample_count=recipe[role+'_paths'],
                         scenario_id=scenario.scenario_id+'-'+role)
        row = generated[index]
        _keys(row, {'role', 'scenario_hash', 'observations_hash', 'truth_hash'}, 'generated role')
        _require(type(row['role']) is str and row['role'] == role,
                 'ordered registered generated role differs')
        _require(all(type(row[name]) is str and len(row[name]) == 64
                     and re.fullmatch('[0-9a-f]{64}', row[name])
                     for name in ('scenario_hash', 'observations_hash', 'truth_hash')),
                 'bounded generated source hash required')
        _require(row['scenario_hash'] == source.scenario_hash, 'registered generated scenario differs')
        sources.append(source)
    return sources


def expected_input_receipt(generated, recipe, cell_seed, *, k):
    """Owner's bounded EXPECTED metadata, not a standalone execution proof."""
    _sources(generated, recipe, cell_seed, k)
    return {'schema_version': 'pirc28-generated-input-consistency-v1',
        'recipe_cell_hash': digest({'recipe': recipe, 'K': k, 'seed': cell_seed}),
        'generated_inputs_hash': digest(generated), 'roles': generated,
        'replay_scope': 'complete-frozen-generator-role-streams; hash-only-truth-and-future',
        'full_generated_input_hashes_verified': True,
        'source_authority_verified': False, 'fit_training_execution_verified': False,
        'reference_sampling_verified': False, 'managed_evidence_bound': False,
        'qualified_identifiability': False, 'real_data_authorized': False}


def validate_generated_inputs(generated, recipe, cell_seed, *, k, check_budget):
    """Sequential full-role hash replay under the caller's original-arm poll."""
    _require(callable(check_budget), 'explicit input replay budget poll required')
    check_budget()
    sources = _sources(generated, recipe, cell_seed, k)
    allocation = recipe_allocation(recipe, k)
    plan_resources(switching_plugin().registry_entry, allocation['config'], allocation['inputs'], matrix_cells=1)
    for source, wanted in zip(sources, generated):
        check_budget()
        observations, truth, manifest = generate_scenario(source, check_budget=check_budget)
        actual = {'role': wanted['role'], 'scenario_hash': source.scenario_hash,
                  'observations_hash': digest(observations.to_dict()), 'truth_hash': digest(truth.to_dict())}
        _require(actual == wanted and manifest['scenario_hash'] == actual['scenario_hash']
            and manifest['observations_hash'] == actual['observations_hash']
            and manifest['truth_hash'] == actual['truth_hash'], 'complete generated source hashes differ')
        # Neither truth nor complete test observations escape into fitting,
        # selection, filtering or reference execution. Free before next role.
        del observations, truth, manifest
    check_budget()
    return expected_input_receipt(generated, recipe, cell_seed, k=k)
