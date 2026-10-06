"""Prepare explicit scientific metadata only, never fit or authorize a study."""

import argparse
import copy
from dataclasses import fields
import json
import math
from pathlib import Path

import numpy as np

from experiments.pirc25.affine import code_hash
from inference.switching import InferenceSpec
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, digest, encode, identifier
from models.switching import SwitchingError, SwitchingSDEModel
from .qualification import AXES, PARAMETERS, IdentifiabilitySpec, _wilson, phase_manifest
from .study import SCHEMA, SyntheticStudyDefinition, arm_bindings
from .synthetic import ScenarioSpec

PLAN_SCHEMA = 'pirc28-scientific-plan-v1'
MAXIMUM_PLAN_BYTES = 64*1024
REFERENCE_PLAN = Path(__file__).with_name('plans')/'identifiability-v1.json'
PARAMETER_FIELDS = {'Q', 'A', 'B', 'a', 'L', 'rate_bounds'}


def _exact(value, expected, description):
    if type(value) is not dict or set(value) != set(expected):
        raise ResearchError('CONTRACT_MISMATCH', description+' fields must be complete and explicit')


def load_plan(path):
    path = Path(path).absolute()
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ResearchError('CONTRACT_MISMATCH', 'duplicate scientific plan field')
            value[key] = item
        return value
    try:
        with opened_regular_file(path.parent, path, maximum_bytes=MAXIMUM_PLAN_BYTES) as (stream, _, _):
            content = stream.read(MAXIMUM_PLAN_BYTES+1)
        value = json.loads(content, object_pairs_hook=unique)
        encode(value)
        return value
    except (ValueError, UnicodeError) as error:
        if isinstance(error, ResearchError):
            raise
        raise ResearchError('CONTRACT_MISMATCH', 'scientific plan must be finite JSON') from error


def prepare_definition(plan):
    """Expand all axes/roles/seeds at current source; no store or data access."""
    try:
        return _prepare(plan)
    except SwitchingError as error:
        raise ResearchError(error.code, str(error)) from error
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ResearchError):
            raise
        raise ResearchError('CONTRACT_MISMATCH', 'scientific plan is incomplete or invalid') from error


def _prepare(plan):
    keys = {'schema_version', 'plan_id', 'truth_parameters', 'axes', 'initial_state', 'initializations',
            'inference', 'sampling', 'forecast_origin_s', 'forecast_horizons_s', 'qualification', 'scope', 'arm_ids'}
    _exact(plan, keys, 'scientific plan')
    if plan['schema_version'] != PLAN_SCHEMA:
        raise ResearchError('CONTRACT_MISMATCH', 'scientific plan schema differs')
    identifier(plan['plan_id'])
    content = encode(plan)
    if len(content) > MAXIMUM_PLAN_BYTES:
        raise ResearchError('RESOURCE_LIMIT', 'scientific plan exceeds its explicit byte quota')
    frozen = json.loads(content)
    profile_hash = digest(frozen)
    _exact(frozen['truth_parameters'], PARAMETER_FIELDS, 'truth parameters')
    truth = SwitchingSDEModel(**frozen['truth_parameters'])
    if truth.n_modes != 2:
        raise ResearchError('CONTRACT_MISMATCH', 'six-axis study needs K=2 synthetic truth')
    _exact(frozen['axes'], AXES, 'six-axis coverage')
    if any(type(values) is not list or len(values) < 2 for values in frozen['axes'].values()):
        raise ResearchError('CONTRACT_MISMATCH', 'scientific phase diagram must vary every one of the six axes')
    _exact(frozen['initial_state'], {'mean', 'covariance'}, 'common initial state')
    _exact(frozen['sampling'], {'replicate_count', 'train_paths', 'selection_paths', 'test_paths',
                              'forecast_paths', 'max_jumps'}, 'sampling')
    sampling = frozen['sampling']
    count = sampling['replicate_count']
    if type(count) is not int or not 2 <= count <= 1000:
        raise ResearchError('CONTRACT_MISMATCH', 'bounded independent replicate count required')
    for name in ('train_paths', 'selection_paths', 'test_paths', 'forecast_paths', 'max_jumps'):
        if type(sampling[name]) is not int or sampling[name] <= 0:
            raise ResearchError('CONTRACT_MISMATCH', 'positive explicit sampling counts required')
    _exact(frozen['inference'], {field.name for field in fields(InferenceSpec)}, 'inference')
    inference = InferenceSpec(**frozen['inference']).validate()
    _exact(frozen['initializations'], {'1', '2'}, 'initializations')
    initializations = {}
    for k in (1, 2):
        values = frozen['initializations'][str(k)]
        if type(values) is not list or not 1 <= len(values) <= inference.max_initializations:
            raise ResearchError('CONTRACT_MISMATCH', 'explicit initializer roster required')
        models = []
        for value in values:
            _exact(value, PARAMETER_FIELDS, 'initializer parameters')
            model = SwitchingSDEModel(**value)
            if model.n_modes != k or model.rate_bounds != truth.rate_bounds:
                raise ResearchError('CONTRACT_MISMATCH', 'K/parameter family differs between original arms')
            models.append(model.to_dict())
        initializations[str(k)] = models
    mean, covariance = frozen['initial_state']['mean'], frozen['initial_state']['covariance']
    base = ScenarioSpec(frozen['plan_id'], truth.to_dict(), (0.0, 1.0), sampling['train_paths'], 0,
        (0.5, 0.5), (mean, mean), (covariance, covariance), np.eye(4)[:2].tolist(),
        np.eye(2).tolist(), sampling['max_jumps'])
    matrix = phase_manifest(base, frozen['axes'])
    points = {row['scenario']['scenario_id']: row for row in matrix['scenarios']}
    if len(points)*count*2 > 10_000:
        raise ResearchError('RESOURCE_LIMIT', 'full paired scientific matrix exceeds cell quota')
    _exact(frozen['scope'], {'negative_control_separation', 'required_qualified_points', 'other_points'}, 'scope')
    scope = frozen['scope']
    required = scope['required_qualified_points']
    if (type(scope['negative_control_separation']) not in (int, float)
            or scope['negative_control_separation'] != 0 or scope['other_points'] != 'phase-diagram-only'
            or type(required) is not list or not required or len(required) > len(points)):
        raise ResearchError('CONTRACT_MISMATCH', 'true identical-mode controls and explicit qualified points required')
    required_hashes = set()
    for point in required:
        _exact(point, AXES, 'required qualified point')
        reference = digest(point)
        if reference in required_hashes or not any(row['axes'] == point for row in points.values()):
            raise ResearchError('IDENTITY_CONFLICT', 'required point duplicated or outside the full matrix')
        required_hashes.add(reference)
    roles = {name: 'required-unidentifiable' if row['axes']['mode_separation'] == 0
        else 'required-qualified' if digest(row['axes']) in required_hashes else 'phase-diagram-only'
        for name, row in points.items()}
    if sum(role == 'required-qualified' for role in roles.values()) != len(required):
        raise ResearchError('CONTRACT_MISMATCH', 'qualified points cannot be identical-mode controls')
    _exact(frozen['qualification'], {field.name for field in fields(IdentifiabilitySpec)}
           - {'scenario_axes', 'replicate_ids', 'scenario_roles'}, 'qualification')
    replicate_ids = tuple('rep-'+str(index).zfill(3) for index in range(count))
    qualification = IdentifiabilitySpec({name: row['axes'] for name, row in points.items()}, replicate_ids,
        scenario_roles=roles, **frozen['qualification']).validate()
    if (qualification.missing_fields or qualification.minimum_occupancy <= 0
            or qualification.maximum_collapse_fraction >= 1 or qualification.minimum_recovery_fraction <= 0
            or any(value <= 0 for value in qualification.thresholds.values())):
        raise ResearchError('NEEDS_PREREGISTRATION', 'non-vacuous numeric scientific gates required')
    if _wilson(count, count, qualification.confidence)[0] < qualification.minimum_recovery_fraction:
        raise ResearchError('CONTRACT_MISMATCH', 'even all-success Wilson lower bound cannot meet the registered recovery gate')
    for name, role in roles.items():
        if role != 'required-qualified':
            continue
        model = points[name]['scenario']['model']
        arrays = {key: np.asarray(model[key], dtype=float) for key in ('Q', 'A', 'B', 'a')}
        l = np.asarray(model['L'], dtype=float)
        arrays['diffusion_covariance'] = l @ l.swapaxes(-1, -2)
        for parameter, metric in PARAMETERS.items():
            zero_error = float(np.sqrt(np.mean(arrays[parameter]**2)))
            # A genuinely zero true coefficient is a correct null, not a
            # vacuous gate. Otherwise even an individual zero estimate must
            # fail this parameter's recovery criterion at a required point.
            if zero_error > 0 and qualification.thresholds[metric] >= zero_error:
                raise ResearchError('CONTRACT_MISMATCH',
                    'required '+parameter+' recovery threshold admits a zero-parameter baseline')
    recipes, seeds = {}, {}
    origin, horizons = frozen['forecast_origin_s'], frozen['forecast_horizons_s']
    if (type(origin) not in (int, float) or not math.isfinite(origin) or origin < 0
            or type(horizons) is not list or not horizons
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in horizons)):
        raise ResearchError('CONTRACT_MISMATCH', 'finite common physical forecast schedule required')
    used = set()
    for name, row in sorted(points.items()):
        times = row['scenario']['times']
        def index_of(time):
            matches = [index for index, value in enumerate(times) if math.isclose(value, time, rel_tol=0, abs_tol=1e-12)]
            if len(matches) != 1:
                raise ResearchError('IDENTITY_CONFLICT', 'common physical forecast schedule is absent from an observation grid')
            return matches[0]
        # Refuse a quota known insufficient at any allowed fitted rate, not only
        # at a favorable initializer. No filtering or synthetic data is computed.
        delta = min(inference.max_step_seconds, inference.max_rate_step/truth.rate_bounds[1]) if truth.rate_bounds[1] else inference.max_step_seconds
        if sum(math.ceil(float(dt)/delta) for dt in np.diff(times)) > inference.max_substeps:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'frozen rate family exceeds the declared substep quota')
        recipes[name] = {'schema_version': 'pirc28-synthetic-recipe-v1', 'scenario': row['scenario'],
            'initializations': copy.deepcopy(initializations), 'inference': copy.deepcopy(frozen['inference']),
            **{key: sampling[key] for key in ('train_paths', 'selection_paths', 'test_paths', 'forecast_paths')},
            'prior_mean': copy.deepcopy(mean), 'prior_covariance': copy.deepcopy(covariance),
            'forecast_origin_index': index_of(origin), 'horizon_indices': [index_of(origin+h) for h in horizons]}
        seeds[name] = {}
        for replicate in replicate_ids:
            seed = int(digest([profile_hash, 'independent-source', name, replicate])[:16], 16)
            if seed in used:
                raise ResearchError('IDENTITY_CONFLICT', 'independent source-seed collision; no silent reseeding')
            used.add(seed)
            seeds[name][replicate] = seed
    _exact(frozen['arm_ids'], {'1', '2'}, 'original arm IDs')
    return SyntheticStudyDefinition.from_dict({'schema_version': SCHEMA, 'runtime_code_hash': code_hash(),
        'phase_matrix': matrix, 'identifiability_spec': qualification.to_dict(), 'recipes': recipes,
        'replicate_seeds': seeds, 'arms': arm_bindings(frozen['arm_ids']['1'], frozen['arm_ids']['2']),
        'forecast_origin_s': origin, 'forecast_horizons_s': horizons})


def describe_plan(plan):
    definition = prepare_definition(plan)
    value = definition.to_dict()
    qualification = IdentifiabilitySpec.from_dict(value['identifiability_spec'])
    return {'schema_version': 'pirc28-scientific-plan-summary-v1', 'plan_id': plan['plan_id'],
        'plan_hash': digest(plan), 'definition_hash': definition.definition_hash,
        'runtime_code_hash': value['runtime_code_hash'], 'definition_bytes': len(definition.content),
        'scenario_count': len(value['recipes']), 'independent_replicates_per_scenario': len(qualification.replicate_ids),
        'paired_cell_count': len(value['recipes'])*len(qualification.replicate_ids)*2,
        'role_counts': {role: list(qualification.scenario_roles.values()).count(role)
            for role in ('required-qualified', 'required-unidentifiable', 'phase-diagram-only')},
        'all_success_recovery_wilson_lower_bound': _wilson(len(qualification.replicate_ids),
            len(qualification.replicate_ids), qualification.confidence)[0],
        'thresholds': qualification.thresholds, 'prepared_only': True, 'registered_studies': False,
        'scientific_groups_executed': 0, 'qualified_identifiability': False, 'real_data_authorized': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='explicit scientific plan JSON; no fixture defaults')
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.path)
        print(encode(describe_plan(plan)).decode())
        return 0
    except (ResearchError, OSError) as error:
        print(encode({'code': error.code if isinstance(error, ResearchError) else 'MISSING_INPUT', 'message': str(error)}).decode())
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
