"""Immutable pre-fit synthetic matrix/recipe/seed definitions, not data grants."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from application.research_preregistration import hash_reference, protocol_binding
from application.research_admission import data_binding
from application.research_evidence import authorize_study, require_unexpired_disclosure
from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, digest, encode, identifier
from models.switching import SwitchingError, SwitchingSDEModel
from .pipeline import validate_recipe
from .qualification import IdentifiabilitySpec, phase_manifest
from .runtime import bind_recipe_execution
from .synthetic import ScenarioSpec

SCHEMA = 'pirc28-synthetic-study-definition-v1'
MAXIMUM_DEFINITION_BYTES = 4*1024*1024
FAMILY = 'pirc28-synthetic-switching'
EXPERIMENT = 'pirc28-identifiability-v1'
SELECTION_RULE = 'one-row-per-independent-replicate; immutable-success-else-latest-attempt'


def arm_bindings(k1_arm_id, k2_arm_id):
    """Explicit arm IDs, fixed algorithm/objective families; never per-cell arms."""
    identifier(k1_arm_id)
    identifier(k2_arm_id)
    if k1_arm_id == k2_arm_id:
        raise ResearchError('IDENTITY_CONFLICT', 'separate original K arms required')
    return {str(k): {'arm_id': arm_id, 'model_family_id': 'pirc28-affine-k'+str(k),
        'method_family_id': 'pirc28-approximate-gem', 'objective_id': 'position-energy',
        'budget_seconds': 86400} for k, arm_id in ((1, k1_arm_id), (2, k2_arm_id))}


def _validate(value):
    fields = {'schema_version', 'runtime_code_hash', 'phase_matrix', 'identifiability_spec',
              'recipes', 'replicate_seeds', 'arms', 'forecast_origin_s', 'forecast_horizons_s'}
    if type(value) is not dict or set(value) != fields or value['schema_version'] != SCHEMA:
        raise ResearchError('CONTRACT_MISMATCH', 'synthetic study definition schema differs')
    if not hash_reference(value['runtime_code_hash']):
        raise ResearchError('CONTRACT_MISMATCH', 'frozen runtime code hash required')
    matrix = value['phase_matrix']
    if type(matrix) is not dict or matrix != phase_manifest(ScenarioSpec.from_dict(matrix['baseline_scenario']), matrix['axes']):
        raise ResearchError('IDENTITY_CONFLICT', 'physical phase matrix is not canonical/reconstructible')
    qualification = IdentifiabilitySpec.from_dict(value['identifiability_spec'])
    if qualification.missing_fields or qualification.scenario_roles is None:
        raise ResearchError('NEEDS_PREREGISTRATION', 'numeric thresholds and explicit pre-fit scenario roles required')
    scenarios = {row['scenario']['scenario_id']: row for row in matrix['scenarios']}
    if {name: row['axes'] for name, row in scenarios.items()} != qualification.scenario_axes:
        raise ResearchError('IDENTITY_CONFLICT', 'physical scenario axes differ from the qualification plan')
    if 'required-unidentifiable' not in qualification.scenario_roles.values():
        raise ResearchError('CONTRACT_MISMATCH', 'a physically verified identical-mode negative control is required')
    recipes, seeds, arms = value['recipes'], value['replicate_seeds'], value['arms']
    if (type(recipes) is not dict or set(recipes) != set(scenarios)
            or type(seeds) is not dict or set(seeds) != set(scenarios)
            or type(arms) is not dict or set(arms) != {'1', '2'}
            or arms != arm_bindings(arms['1']['arm_id'], arms['2']['arm_id'])):
        raise ResearchError('CONTRACT_MISMATCH', 'exact recipe/seed matrix and fixed original arm families required')
    origin, horizons = value['forecast_origin_s'], value['forecast_horizons_s']
    if (type(origin) not in (int, float) or not math.isfinite(origin) or origin < 0
            or type(horizons) is not list or not horizons or len(horizons) > 1000
            or any(type(h) not in (int, float) or not math.isfinite(h) or h <= 0 for h in horizons)
            or horizons != sorted(set(horizons))):
        raise ResearchError('CONTRACT_MISMATCH', 'common physical forecast origin/horizons must be explicit')
    used_seeds = set()
    for name, row in scenarios.items():
        recipe = recipes[name]
        if type(recipe) is not dict or 'identifiability_spec' in recipe or recipe.get('scenario') != row['scenario']:
            raise ResearchError('IDENTITY_CONFLICT', 'recipes must bind exact physical sources without duplicate qualification overrides')
        if type(seeds[name]) is not dict or set(seeds[name]) != set(qualification.replicate_ids):
            raise ResearchError('CONTRACT_MISMATCH', 'every independent replicate needs an explicit source seed')
        for seed in seeds[name].values():
            if type(seed) is not int or not 0 <= seed < 2**64 or seed in used_seeds:
                raise ResearchError('IDENTITY_CONFLICT', 'source seeds must be independent across the whole phase matrix')
            used_seeds.add(seed)
        full = {**recipe, 'identifiability_spec': value['identifiability_spec']}
        for k in (1, 2):
            scenario, models, inference, _, _ = validate_recipe(full, k)
            if (inference.max_iterations < 1 or (k == 2 and len(models) < 2)
                    or len({digest(model.to_dict()) for model in models}) != len(models)):
                raise ResearchError('CONTRACT_MISMATCH', 'actual fitting and distinct predeclared K=2 initializations required')
            actual_origin = scenario.times[recipe['forecast_origin_index']]
            actual_h = [scenario.times[index]-actual_origin for index in recipe['horizon_indices']]
            if (not math.isclose(actual_origin, origin, rel_tol=0, abs_tol=1e-12) or len(actual_h) != len(horizons)
                    or any(not math.isclose(left, right, rel_tol=0, abs_tol=1e-12) for left, right in zip(actual_h, horizons))):
                raise ResearchError('IDENTITY_CONFLICT', 'recipe forecast indices do not implement the common physical horizons')
        truth = SwitchingSDEModel.from_dict(row['scenario']['model'])
        identical = (all(np.array_equal(getattr(truth, field).numpy()[0], getattr(truth, field).numpy()[1])
                         for field in ('A', 'B', 'a'))
                     and np.array_equal((truth.L @ truth.L.mT).numpy()[0], (truth.L @ truth.L.mT).numpy()[1]))
        role = qualification.scenario_roles[name]
        if (role == 'required-unidentifiable' and not identical) or (role == 'required-qualified' and identical):
            raise ResearchError('IDENTITY_CONFLICT', 'scenario role contradicts actual drift/diffusion observation laws')
        # No fitting or data reads: still reject infeasible declared allocations.
        cells = [{'arm_id': arms[str(k)]['arm_id'], 'fit_K': k, 'replicate_id': replicate,
                  'block_id': replicate, 'seed': seeds[name][replicate]}
                 for replicate in qualification.replicate_ids for k in (1, 2)]
        bind_recipe_execution({'switching_recipe': full, 'cells': cells})
    return value


@dataclass(frozen=True)
class SyntheticStudyDefinition:
    """Canonical bytes own all nested state; returned dicts are detached copies."""
    content: bytes

    def __post_init__(self):
        if type(self.content) is not bytes or not 0 < len(self.content) <= MAXIMUM_DEFINITION_BYTES:
            raise ResearchError('RESOURCE_LIMIT', 'study definition bytes exceed the immutable quota')
        try:
            value = json.loads(self.content)
            if encode(value) != self.content:
                raise ResearchError('IDENTITY_CONFLICT', 'study definition bytes must be canonical JSON')
            _validate(value)
        except SwitchingError as error:
            raise ResearchError(error.code, str(error)) from error
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ResearchError):
                raise
            raise ResearchError('CONTRACT_MISMATCH', 'incomplete synthetic study definition') from error

    @classmethod
    def from_dict(cls, value):
        try:
            return cls(encode(value))
        except (TypeError, ValueError) as error:
            if isinstance(error, ResearchError):
                raise
            raise ResearchError('CONTRACT_MISMATCH', 'study definition must be finite canonical JSON') from error

    def to_dict(self):
        return json.loads(self.content)

    @property
    def definition_hash(self):
        return hashlib.sha256(self.content).hexdigest()

    @property
    def study_ids(self):
        return ['pirc28-synthetic-'+digest([self.definition_hash, name]) for name in sorted(self.to_dict()['recipes'])]

    def proposals(self, runtime_binding):
        if (type(runtime_binding) is not dict or set(runtime_binding) != {'root', 'store_id'}
                or type(runtime_binding['root']) is not str or not Path(runtime_binding['root']).is_absolute()
                or type(runtime_binding['store_id']) is not str):
            raise ResearchError('MISSING_INPUT', 'explicit existing shared runtime binding required')
        identifier(runtime_binding['store_id'])
        plan = self.to_dict()
        qualification = IdentifiabilitySpec.from_dict(plan['identifiability_spec'])
        proposals = []
        for name, study_id in zip(sorted(plan['recipes']), self.study_ids):
            recipe = {**plan['recipes'][name], 'identifiability_spec': plan['identifiability_spec']}
            value = {'study_id': study_id, 'experiment_id': EXPERIMENT, 'comparison_family': FAMILY,
                'code_hash': plan['runtime_code_hash'], 'feature_hash': digest('no-terrain-position-observations'),
                'selection_hash': digest({'rule': qualification.initialization_selection, 'recipe': recipe}),
                'arms': [plan['arms'][str(k)] for k in (1, 2)], 'runtime_binding': dict(runtime_binding),
                'switching_recipe': recipe, 'phase_matrix_hash': digest(plan['phase_matrix']),
                'pirc28_study_definition_hash': self.definition_hash,
                'cells': [{'arm_id': plan['arms'][str(k)]['arm_id'], 'fit_K': k, 'replicate_id': replicate,
                    'block_id': replicate, 'seed': plan['replicate_seeds'][name][replicate]}
                    for replicate in qualification.replicate_ids for k in (1, 2)]}
            bind_recipe_execution(value)
            # Not a registered pirc25 spec: protocols/admission are still required.
            proposals.append(value)
        return proposals

    def preregistration_fields(self, protocols):
        plan = self.to_dict()
        if type(protocols) is not dict or set(protocols) != set(self.study_ids):
            raise ResearchError('CONTRACT_MISMATCH', 'exact complete source protocol matrix required')
        bindings = []
        for name, study_id in zip(sorted(plan['recipes']), self.study_ids):
            protocol = protocols[study_id]
            blocks = protocol.get('blocks', [])
            recipe = {**plan['recipes'][name], 'identifiability_spec': plan['identifiability_spec']}
            expected = hashlib.sha256(encode(recipe)).hexdigest()
            if (protocol.get('schema_version') != 'pirc25-data-protocol-v1' or protocol.get('study_id') != study_id
                    or {block.get('block_id') for block in blocks} != set(plan['identifiability_spec']['replicate_ids'])
                    or len(blocks) != len(plan['identifiability_spec']['replicate_ids'])
                    or any(block.get('dataset_id') != 'synthetic-pirc28' or block.get('sha256') != expected
                           or block.get('split_role') != 'train' or block.get('fit_scope') is not True for block in blocks)):
                raise ResearchError('UNAUTHORIZED_DATA', 'protocols must physically bind every predeclared synthetic recipe/block')
            bindings.append(protocol_binding(protocol))
        return {'schema_version': 'pirc25-preregistration-v1', 'test_mode': 'exploratory',
            'study_ids': self.study_ids, 'protocol_bindings': bindings, 'primary_metrics': ['position_energy_score_d2'],
            'selection_rule': plan['identifiability_spec']['initialization_selection'],
            'stopping_rule': plan['identifiability_spec']['stop_rule'],
            'comparisons': [{'reference': plan['arms']['1']['arm_id'], 'candidates': [plan['arms']['2']['arm_id']]}],
            'pirc28_identifiability_spec_hash': digest(plan['identifiability_spec']),
            'pirc28_phase_matrix_hash': digest(plan['phase_matrix']), 'pirc28_study_definition_hash': self.definition_hash,
            'result_selection_rule': SELECTION_RULE}

    def matches_studies(self, studies, runtime_binding):
        expected = {value['study_id']: value for value in self.proposals(runtime_binding)}
        if {study['study_id'] for study in studies} != set(expected) or len(studies) != len(expected):
            raise ResearchError('IDENTITY_CONFLICT', 'complete pre-fit source study matrix required')
        for study in studies:
            if any(study.get(key) != value for key, value in expected[study['study_id']].items()):
                raise ResearchError('IDENTITY_CONFLICT', 'registered study differs from the pre-fit complete definition')

    def matches_study(self, study, runtime_binding):
        expected = next((value for value in self.proposals(runtime_binding) if value['study_id'] == study.get('study_id')), None)
        if expected is None or any(study.get(key) != value for key, value in expected.items()):
            raise ResearchError('IDENTITY_CONFLICT', 'registered study differs from the pre-fit complete definition')


def stage_definition(store, definition):
    """Publish definition metadata only; no study, attempt, grant or ledger init."""
    if not isinstance(definition, SyntheticStudyDefinition):
        raise ResearchError('CONTRACT_MISMATCH', 'canonical immutable study definition required')
    plan = definition.to_dict()
    if plan['runtime_code_hash'] != code_hash():
        raise ResearchError('IDENTITY_CONFLICT', 'study definition source differs from the current runtime')
    with store._read_transaction():
        from .evidence import source_run_ids
        existing = []
        for study_id in definition.study_ids:
            try:
                existing.append(store.manifest('study-'+study_id)['spec'])
            except ResearchError as error:
                if error.code != 'MISSING_INPUT':
                    raise
        source_runs = source_run_ids(existing)
        attempts = store.attempts()
        if any(attempt['run_id'] in source_runs for attempt in attempts.values()):
            raise ResearchError('NEEDS_PREREGISTRATION', 'scientific definition must precede every source attempt')
        store.publish('pirc28-phase-matrix-'+digest(plan['phase_matrix']), plan['phase_matrix'])
        store.publish('pirc28-identifiability-spec-'+digest(plan['identifiability_spec']), plan['identifiability_spec'])
        store.publish('pirc28-study-definition-'+definition.definition_hash, plan)
    return {'definition_hash': definition.definition_hash, 'phase_matrix_hash': digest(plan['phase_matrix']),
            'identifiability_spec_hash': digest(plan['identifiability_spec']), 'study_ids': definition.study_ids,
            'registered_studies': False, 'real_data_authorized': False}


def register_definition_studies(store, definition, admissions):
    """Register the complete frozen source matrix using existing operator inputs.

    No protocol, package, grant, external-history attestation, attempt or new arm
    is manufactured. Ordinary SharedRunner admission still precedes execution.
    """
    if not isinstance(definition, SyntheticStudyDefinition):
        raise ResearchError('CONTRACT_MISMATCH', 'canonical immutable study definition required')
    plan = definition.to_dict()
    if (plan['runtime_code_hash'] != code_hash()
            or type(admissions) is not dict or set(admissions) != set(definition.study_ids)):
        raise ResearchError('IDENTITY_CONFLICT', 'current runtime and complete original admission inputs required')
    with store._read_transaction():
        if store.manifest('pirc28-study-definition-'+definition.definition_hash) != plan:
            raise ResearchError('NEEDS_PREREGISTRATION', 'complete definition must be staged before source registration')
        proposals = definition.proposals({'root': str(store.path.parent), 'store_id': store.store_id})
        protocols, grants = {}, {}
        for value in proposals:
            study_id, settings = value['study_id'], admissions[value['study_id']]
            if (type(settings) is not dict or settings.get('mode') != 'pilot' or settings.get('purpose') != 'fit'
                    or not {'protocol_id', 'authorization_id', 'package_hash'} <= set(settings)
                    or not hash_reference(settings.get('package_hash'))):
                raise ResearchError('CONTRACT_MISMATCH', 'synthetic qualification uses explicit pilot admission')
            protocol = store.manifest('protocol-'+identifier(settings['protocol_id']))
            grant = store.authorization(settings['authorization_id'], version=settings.get('authorization_version'))
            authorize_study(store, study_id, grant, 'fit')
            if (grant.get('protocol_hash') != digest(protocol)
                    or not {cell['block_id'] for cell in value['cells']} <= set(grant['block_ids'])):
                raise ResearchError('UNAUTHORIZED_DATA', 'original grant does not cover the frozen source matrix')
            protocols[study_id], grants[study_id] = protocol, grant
        expected = definition.preregistration_fields(protocols)
        for value in proposals:
            study_id = value['study_id']
            protocol, settings = protocols[study_id], admissions[study_id]
            reference = protocol.get('preregistration_hash')
            if not hash_reference(reference):
                raise ResearchError('NEEDS_PREREGISTRATION', 'registered complete pre-fit plan required')
            prereg = store.manifest('preregistration-'+reference)
            if digest(prereg) != reference or any(prereg.get(key) != field for key, field in expected.items()):
                raise ResearchError('NEEDS_PREREGISTRATION', 'operator preregistration does not freeze the complete study definition')
            value.update(schema_version=store.SCHEMA, protocol_hash=digest(protocol), data_hash=data_binding(protocol),
                admission=json.loads(encode(settings)), comparison_plan={
                    'reference_arm_id': plan['arms']['1']['arm_id'], 'candidate_arm_ids': [plan['arms']['2']['arm_id']],
                    'preregistration_hash': reference, 'failure_policy': 'retain-and-exclude-incomplete-blocks'})
        # All grants/protocols/preregistration inputs passed before any study write.
        result = [store.register(value, digest(value)) for value in proposals]
        for study_id, grant in grants.items():
            authorize_study(store, study_id, grant, 'fit')
    earliest = min(grants, key=lambda key: datetime.fromisoformat(grants[key]['expires_at']))
    require_unexpired_disclosure(store, earliest, grants[earliest], 'fit')
    return result
