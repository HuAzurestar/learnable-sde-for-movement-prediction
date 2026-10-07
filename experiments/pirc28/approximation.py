"""Frozen approximation requirements, not numerical or managed proof.

No diagnostic executor or source authority is implemented in this specification
module. Its legacy pending helpers are not aggregate proof. Managed publication
uses scientific_release after funded checks and independent canonical bindings;
legacy recipes remain engineering diagnostics, never scientific authorization.
"""

from dataclasses import dataclass, replace
import json
import math

from inference.switching import InferenceSpec
from infrastructure.research_store import ResearchError, digest, encode

SCHEMA = 'pirc28-approximation-spec-v1'
RECIPE_SCHEMA = 'pirc28-synthetic-recipe-v2'
CHECKS = ('step-halving-prefix-filter', 'event-driven-prefix-particle-reference',
          'known-mode-affine-Kalman-boundary')
ERRORS = {'mode_total_variation': 0.25, 'position_mean_max_abs_m': 1.0,
          'velocity_mean_max_abs_m_s': 1.0, 'log_likelihood_per_observation_abs': 1.0}


def _exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ResearchError('CONTRACT_MISMATCH', label+' fields must be complete and exact')


def _positive(value, upper, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
        raise ResearchError('CONTRACT_MISMATCH', label+' must be finite, positive and bounded')


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ResearchError('RESOURCE_PLAN_REJECTED', label+' exceeds the frozen diagnostic quota')


def _validate(value):
    literals = {'schema_version': SCHEMA, 'sampling_unit': 'independent-synthetic-replicate',
        'scope': 'all-registered-K2-cells; selected-fit; every-heldout-prefix-path',
        'observation_role': 'test-prefix-through-frozen-forecast-origin; no-future-no-truth',
        'budget_rule': 'original-K2-arm; shared-supervisor; no-extra-budget',
        'failure_state': 'INFERENCE_INAPPLICABLE',
        'aggregation_rule': 'all-required-checks; no-failure-dropping; no-optional-stopping'}
    _exact(value, {*literals, 'step_halving', 'event_driven_reference', 'known_mode_boundary'}, 'approximation')
    if any(value[name] != expected for name, expected in literals.items()):
        raise ResearchError('CONTRACT_MISMATCH', 'approximation scope, truth boundary or budget rule differs')
    half = value['step_halving']
    _exact(half, {'factor', 'maximum_substeps', 'thresholds'}, 'step halving')
    if type(half['factor']) is not int or half['factor'] != 2:
        raise ResearchError('CONTRACT_MISMATCH', 'both time and rate step must be halved')
    _integer(half['maximum_substeps'], 1, 100_000, 'fine substeps')
    reference = value['event_driven_reference']
    _exact(reference, {'algorithm', 'particles', 'independent_batches', 'maximum_jumps_per_particle_interval',
        'minimum_ess_fraction', 'confidence', 'maximum_mc_half_width_fraction', 'thresholds'}, 'reference')
    if reference['algorithm'] != 'event-driven-bootstrap-particle-filter-v1':
        raise ResearchError('CONTRACT_MISMATCH', 'event-driven conditional-affine reference required')
    _integer(reference['particles'], 64, 4096, 'particles')
    _integer(reference['independent_batches'], 4, 32, 'independent reference batches')
    _integer(reference['maximum_jumps_per_particle_interval'], 1, 10_000, 'reference jumps')
    _positive(reference['minimum_ess_fraction'], 1, 'minimum ESS fraction')
    _positive(reference['maximum_mc_half_width_fraction'], 0.5, 'MC uncertainty relative to tolerance')
    if type(reference['confidence']) not in (int, float) or not 0.9 <= reference['confidence'] <= 0.999:
        raise ResearchError('CONTRACT_MISMATCH', 'reference confidence must be finite within [0.9, 0.999]')
    for part in (half, reference):
        _exact(part['thresholds'], ERRORS, 'approximation errors')
        for name, upper in ERRORS.items():
            _positive(part['thresholds'][name], upper, name)
    boundary = value['known_mode_boundary']
    _exact(boundary, {'algorithm', 'absolute_tolerance', 'relative_tolerance'}, 'known-mode boundary')
    if boundary['algorithm'] != 'conditional-affine-K1-Kalman-v1':
        raise ResearchError('CONTRACT_MISMATCH', 'known-mode conditional affine control required')
    for name in ('absolute_tolerance', 'relative_tolerance'):
        _positive(boundary[name], 1e-6, name)


@dataclass(frozen=True)
class ApproximationSpec:
    """Immutable canonical bytes; no inherited numerical defaults."""
    content: bytes

    def __post_init__(self):
        if type(self.content) is not bytes or not 0 < len(self.content) <= 16*1024:
            raise ResearchError('RESOURCE_LIMIT', 'approximation specification bytes exceed quota')
        try:
            value = json.loads(self.content)
            if encode(value) != self.content:
                raise ResearchError('IDENTITY_CONFLICT', 'approximation specification is not canonical')
            _validate(value)
        except (ValueError, TypeError, UnicodeError) as error:
            if isinstance(error, ResearchError):
                raise
            raise ResearchError('CONTRACT_MISMATCH', 'approximation specification must be finite JSON') from error

    @classmethod
    def from_dict(cls, value):
        try:
            return cls(encode(value))
        except (TypeError, ValueError) as error:
            if isinstance(error, ResearchError):
                raise
            raise ResearchError('CONTRACT_MISMATCH', 'approximation specification must be finite JSON') from error

    def to_dict(self):
        return json.loads(self.content)

    @property
    def spec_hash(self):
        return digest(self.to_dict())

    def fine_inference(self, coarse):
        """Reduce BOTH limiting steps; retain all other fitting parameters."""
        if not isinstance(coarse, InferenceSpec):
            raise ResearchError('CONTRACT_MISMATCH', 'canonical coarse inference required')
        coarse.validate()
        half = self.to_dict()['step_halving']
        return replace(coarse, max_step_seconds=coarse.max_step_seconds/half['factor'],
            max_rate_step=coarse.max_rate_step/half['factor'], max_substeps=half['maximum_substeps']).validate()

    def prefix_preflight(self, inference, times, origin_index, rate_upper):
        """Metadata only: worst permitted rate on the exact held-out prefix.

        This is not a shared resource admission. A future executor must account
        for all filter/particle storage and work in its registered resource plan.
        """
        fine = self.fine_inference(inference)
        delta = min(fine.max_step_seconds, fine.max_rate_step/rate_upper) if rate_upper else fine.max_step_seconds
        steps = sum(math.ceil(float(right-left)/delta) for left, right in zip(times[:origin_index], times[1:origin_index+1]))
        if steps > fine.max_substeps:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'fine prefix exceeds the registered substep limit at an allowed fitted rate')
        return {'maximum_prefix_substeps': steps, 'fine_inference': fine.to_dict(),
                'resource_admitted': False, 'executor_implemented': False}


def pending_qualification(studies):
    """Derive a fail-closed status from source recipes, never worker assertions."""
    specs = []
    for study in studies:
        recipe = study['switching_recipe']
        if recipe.get('schema_version') == RECIPE_SCHEMA:
            specs.append(ApproximationSpec.from_dict(recipe['approximation_spec']).spec_hash)
        else:
            specs.append(None)
    if len(set(specs)) != 1:
        raise ResearchError('IDENTITY_CONFLICT', 'one complete approximation specification required across the source matrix')
    reference = specs[0] if specs else None
    return {'schema_version': 'pirc28-approximation-qualification-v1', 'spec_hash': reference,
        'state': 'NEEDS_IMPLEMENTATION' if reference is not None else 'NEEDS_PREREGISTRATION',
        'required_checks': list(CHECKS), 'qualified': False, 'managed_evidence_bound': False}


def withhold_scientific_pass(report, studies, *, fixture):
    """Keep mathematical/negative results, but never claim unsupported accuracy."""
    pending = pending_qualification(studies)
    if not fixture and report['state'] == 'PASSED':
        report['identifiability_only_state'] = 'PASSED'
        report['state'] = 'NEEDS_APPROXIMATION_QUALIFICATION'
        if 'applicable_ranges' in report:
            report['applicable_scenarios'], report['applicable_ranges'] = [], []
    return pending
