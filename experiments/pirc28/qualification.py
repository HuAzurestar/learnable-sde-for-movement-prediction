"""Evaluator-only synthetic diagnostics and fail-closed identifiability gates.

This module cannot authorize execution or real-data access. A mathematically
passing aggregate still needs the shared admission/evidence chain before use.
Independent synthetic replicate IDs, never points or initializations, are the
sampling units. Missing, failed and budget-exhausted rows remain denominators.
"""

from dataclasses import dataclass
import itertools
import math
from statistics import NormalDist

import numpy as np

from infrastructure.research_store import digest
from models.switching import SwitchingError, SwitchingSDEModel, finite_array
from .synthetic import ScenarioSpec

AXES = ('mode_separation', 'dwell_time_s', 'observation_interval_s',
        'observation_noise_m', 'trajectory_length_s', 'mode_occupancy')
PARAMETERS = {'Q': 'rate_rmse_s_inv', 'A': 'position_drift_rmse_s_inv2',
              'B': 'velocity_drift_rmse_s_inv', 'a': 'acceleration_rmse_m_s2',
              'diffusion_covariance': 'diffusion_covariance_rmse_m2_s3'}
METRICS = (*PARAMETERS.values(), 'mode_brier', 'mode_ece', 'occupancy_error',
           'position_marginal_coverage_error', 'any_jump_brier', 'different_endpoint_mode_brier')
TERMINALS = ('SUCCEEDED', 'FAILED', 'TIMEOUT', 'BUDGET_EXHAUSTED',
             'INFERENCE_INAPPLICABLE', 'NOT_RUN')


def _number(value, name, lower=0, upper=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise SwitchingError('CONTRACT_MISMATCH', name + ' must be finite within its declared range')
    return float(value)


def phase_scenarios(base, axes, *, maximum_scenarios=256):
    """Bounded Cartesian six-axis matrix; no simulation, fitting or ledger write.

    dwell_time_s denotes mean stationary sojourn time, not the CTMC relaxation
    time. mode_occupancy is the stationary probability of synthetic state-1.
    mode_separation scales affine drift contrasts; diffusion is not silently
    changed. Observation noise is an isotropic position standard deviation.
    """
    if not isinstance(base, ScenarioSpec) or type(axes) is not dict or set(axes) != set(AXES):
        raise SwitchingError('CONTRACT_MISMATCH', 'all six scenario axes must be explicit')
    model, _, _, means, covariance, h, _, _ = base.validated()
    if model.n_modes != 2 or not np.array_equal(h, np.eye(4)[:2]):
        raise SwitchingError('CONTRACT_MISMATCH', 'phase matrix requires K=2 with position observations')
    if type(maximum_scenarios) is not int or not 1 <= maximum_scenarios <= 10_000:
        raise SwitchingError('RESOURCE_LIMIT', 'matrix quota is invalid')
    size = 1
    for name in AXES:
        values = axes[name]
        if type(values) is not list or not values or len(values) > 64:
            raise SwitchingError('CONTRACT_MISMATCH', 'axis values must be bounded nonempty lists')
        for value in values:
            _number(value, name)
        if len(set(values)) != len(values):
            raise SwitchingError('CONTRACT_MISMATCH', 'duplicate scenario axis value')
        size *= len(values)
        if size > maximum_scenarios:
            raise SwitchingError('RESOURCE_LIMIT', 'phase matrix exceeds the frozen scenario quota')
    scenarios = []
    for values in itertools.product(*(axes[name] for name in AXES)):
        point = dict(zip(AXES, values))
        separation, dwell, interval, noise, length, occupancy = values
        if min(dwell, interval, noise, length) <= 0 or not 0 < occupancy < 1 or length < 2*interval:
            raise SwitchingError('CONTRACT_MISMATCH', 'positive scales, interior occupancy and two intervals required')
        steps = int(math.floor(length/interval))
        if steps > 99_999:
            raise SwitchingError('RESOURCE_LIMIT', 'phase schedule exceeds the point quota')
        times = np.arange(steps+1, dtype=float) * interval
        if not math.isclose(times[-1], length, rel_tol=0, abs_tol=1e-12):
            times = np.append(times, length)
        if len(times) > 100_000:
            raise SwitchingError('RESOURCE_LIMIT', 'phase schedule exceeds the point quota')
        parameters = {name: getattr(model, name).numpy().copy() for name in ('A', 'B', 'a', 'L')}
        for name in ('A', 'B', 'a'):
            average = parameters[name].mean(axis=0)
            parameters[name] = average + separation*(parameters[name]-average)
        q01, q10 = 1/(2*dwell*occupancy), 1/(2*dwell*(1-occupancy))
        changed = SwitchingSDEModel([[-q01, q01], [q10, -q10]], **parameters,
                                   rate_bounds=model.rate_bounds)
        scenario = ScenarioSpec('phase-' + digest(point)[:20], changed.to_dict(), tuple(times),
            base.sample_count, base.seed, (occupancy, 1-occupancy), tuple(means), tuple(covariance),
            tuple(h), tuple(np.eye(2)*noise**2), base.max_jumps, base.label_permutation)
        scenario.validated()
        scenarios.append({'axes': point, 'scenario': scenario.to_dict(), 'scenario_hash': scenario.scenario_hash})
    return scenarios


def parameter_recovery(estimate, truth, scales):
    """Two-label matching on unit-explicit normalized parameter error.

    L itself is not identifiable up to noise rotations; compare L L^T instead.
    The returned order maps true label indices into estimated label indices.
    """
    if (not isinstance(estimate, SwitchingSDEModel) or not isinstance(truth, SwitchingSDEModel)
            or estimate.n_modes != truth.n_modes or type(scales) is not dict or set(scales) != set(PARAMETERS)):
        raise SwitchingError('CONTRACT_MISMATCH', 'matching model K and all unit-specific scales required')
    scales = {name: _number(value, name, np.finfo(float).tiny) for name, value in scales.items()}
    arrays = []
    for model in (estimate, truth):
        value = {name: getattr(model, name).numpy() for name in ('Q', 'A', 'B', 'a')}
        l = model.L.numpy()
        value['diffusion_covariance'] = l @ l.swapaxes(-1, -2)
        arrays.append(value)
    candidates = []
    for order in itertools.permutations(range(truth.n_modes)):
        errors = {}
        for name, metric in PARAMETERS.items():
            aligned = arrays[0][name][list(order)]
            if name == 'Q':
                aligned = aligned[:, list(order)]
            errors[metric] = float(np.sqrt(np.mean((aligned-arrays[1][name])**2)))
        objective = sum(errors[metric]/scales[name] for name, metric in PARAMETERS.items())
        candidates.append((objective, order, errors))
    _, order, errors = min(candidates, key=lambda value: value[0])
    return {'label_order': list(order), 'metrics': errors, 'matching': 'minimum-normalized-parameter-error',
            'matching_scales': scales, 'interpretation_level': 'synthetic-statistical-only'}


def phase_manifest(base, axes):
    """Canonical reconstruction evidence for axis meanings, not just axis labels."""
    return {'schema_version': 'pirc28-phase-matrix-v1', 'baseline_scenario': base.to_dict(),
            'axes': axes, 'scenarios': phase_scenarios(base, axes)}


def mode_diagnostics(probabilities, modes, order, dwell, duration, *, bins=10):
    probabilities = np.asarray(probabilities, dtype=float)
    modes = np.asarray(modes)
    if (probabilities.ndim != 3 or probabilities.shape[-1] != 2 or modes.shape != probabilities.shape[:2]
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-10, rtol=0)
            or modes.dtype.kind not in 'iu' or np.any((modes < 0) | (modes > 1))
            or sorted(order) != [0, 1] or any(type(value) is not int for value in order)
            or type(bins) is not int or not 2 <= bins <= 100):
        raise SwitchingError('CONTRACT_MISMATCH', 'mode diagnostic arrays or calibration bins differ')
    duration = _number(duration, 'duration', np.finfo(float).tiny)
    dwell = finite_array(dwell, (len(modes), 2), 'true continuous dwell')
    if np.any(dwell < 0) or not np.allclose(dwell.sum(axis=-1), duration, atol=1e-8, rtol=0):
        raise SwitchingError('CONTRACT_MISMATCH', 'dwell must cover every full independent path')
    aligned = probabilities[..., order]
    labels = np.eye(2)[modes]
    p, target = aligned[..., 0].ravel(), labels[..., 0].ravel()
    membership = np.minimum((p*bins).astype(int), bins-1)
    ece = sum(float(np.mean(membership == index)) * abs(float(p[membership == index].mean())
              - float(target[membership == index].mean())) for index in range(bins) if np.any(membership == index))
    posterior_occupancy = aligned.mean(axis=(0, 1))
    true_dwell_occupancy = dwell.mean(axis=0)/duration
    return {'mode_brier': float(np.mean(np.sum((aligned-labels)**2, axis=-1))), 'mode_ece': ece,
            'occupancy_error': float(np.max(abs(posterior_occupancy-true_dwell_occupancy))),
            'posterior_occupancy': posterior_occupancy.tolist(), 'true_dwell_occupancy': true_dwell_occupancy.tolist(),
            'calibration_bins': bins, 'role': 'synthetic-filtering-diagnostic',
            'sampling_unit': 'independent-synthetic-replicate', 'within_path_points_are_not_replicates': True}


def forecast_coverage(samples, target, *, nominal=0.9):
    samples, target = np.asarray(samples, dtype=float), np.asarray(target, dtype=float)
    nominal = _number(nominal, 'nominal marginal coverage', np.finfo(float).tiny, 1-np.finfo(float).eps)
    if (samples.ndim != 3 or samples.shape[-1] != 4 or len(samples) < 2
            or target.shape != samples.shape[1:] or not np.isfinite(samples).all() or not np.isfinite(target).all()):
        raise SwitchingError('CONTRACT_MISMATCH', 'forecast coverage expects samples by horizon by four-state')
    lower, upper = np.quantile(samples[..., :2], [(1-nominal)/2, (1+nominal)/2], axis=0)
    coverage = float(np.mean((target[..., :2] >= lower) & (target[..., :2] <= upper)))
    return {'position_marginal_coverage': coverage, 'position_marginal_coverage_nominal': nominal,
            'position_marginal_coverage_error': abs(coverage-nominal),
            'definition': 'coordinate-marginal-equal-tail-intervals; not joint-region coverage'}


def jump_diagnostics(forecasts, jump_times, modes, times, origin_index, horizon_indices):
    """Two different binary events scored against evaluator-only full jump paths."""
    times, modes = np.asarray(times, dtype=float), np.asarray(modes)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or modes.shape != (len(forecasts), len(times)) or modes.dtype.kind not in 'iu'
            or len(jump_times) != len(forecasts) or not forecasts or type(origin_index) is not int
            or not 0 <= origin_index < len(times)-1 or type(horizon_indices) is not list or not horizon_indices
            or any(type(index) is not int or not origin_index < index < len(times) for index in horizon_indices)):
        raise SwitchingError('CONTRACT_MISMATCH', 'jump event evaluation schedule or paths differ')
    scores = {'any_jump_brier': [], 'different_endpoint_mode_brier': []}
    for forecast, jumps, path_modes in zip(forecasts, jump_times, modes):
        jumps = np.asarray(jumps, dtype=float)
        if jumps.ndim != 1 or not np.isfinite(jumps).all() or np.any(np.diff(jumps) <= 0):
            raise SwitchingError('CONTRACT_MISMATCH', 'jump times must be finite and ordered')
        if forecast.get('role') != 'filtering-origin' or forecast.get('origin') != times[origin_index]:
            raise SwitchingError('CONTRACT_MISMATCH', 'jump event forecast origin differs')
        if not np.array_equal(np.asarray(forecast.get('horizons')), times[horizon_indices]-times[origin_index]):
            raise SwitchingError('CONTRACT_MISMATCH', 'jump event horizons differ')
        any_jump = finite_array(forecast['probability_any_jump'], (len(horizon_indices),), 'any jump probabilities')
        endpoint = finite_array(forecast['probability_different_endpoint_mode'], (len(horizon_indices),), 'endpoint probabilities')
        if np.any(any_jump < 0) or np.any(any_jump > 1) or np.any(endpoint < 0) or np.any(endpoint > any_jump+1e-12):
            raise SwitchingError('INVALID_PROBABILITY', 'jump/endpoint probabilities are inconsistent')
        actual_any = np.array([np.any((jumps > times[origin_index]) & (jumps <= times[index])) for index in horizon_indices])
        actual_endpoint = path_modes[horizon_indices] != path_modes[origin_index]
        scores['any_jump_brier'].extend(((any_jump-actual_any)**2).tolist())
        scores['different_endpoint_mode_brier'].extend(((endpoint-actual_endpoint)**2).tolist())
    return {**{name: float(np.mean(values)) for name, values in scores.items()},
            'event_any_jump': 'at-least-one-true-jump-in-(origin,origin+h]',
            'event_different_endpoint': 'true-mode-at-origin-differs-from-true-mode-at-origin+h',
            'sampling_unit': 'one-independent-replicate; within-path-events-are-not-replicates'}


@dataclass(frozen=True)
class IdentifiabilitySpec:
    scenario_axes: dict
    replicate_ids: tuple[str, ...]
    thresholds: dict
    parameter_scales: dict
    minimum_occupancy: float | None = None
    maximum_collapse_fraction: float | None = None
    minimum_recovery_fraction: float | None = None
    minimum_effective_replicates: int | None = None
    confidence: float = 0.95
    bootstrap_seed: int = 0
    bootstrap_draws: int = 1000
    calibration_bins: int = 10
    coverage_nominal: float = 0.9
    initialization_selection: str = 'selection-observation-objective-only'
    collapse_criterion: str = 'inference-frozen-low-occupancy-or-identical-dynamics'
    stop_rule: str = 'shared-arm-closure-stops-all-cells; no-budget-reset'
    scenario_roles: dict | None = None

    def validate(self):
        if (type(self.scenario_axes) is not dict or not self.scenario_axes or len(self.scenario_axes) > 10_000
                or any(type(name) is not str or not name for name in self.scenario_axes)):
            raise SwitchingError('CONTRACT_MISMATCH', 'explicit bounded scenarios required')
        for values in self.scenario_axes.values():
            if type(values) is not dict or set(values) != set(AXES):
                raise SwitchingError('CONTRACT_MISMATCH', 'each scenario must disclose six axes')
            for name, value in values.items():
                _number(value, name)
            if (min(values[name] for name in AXES[1:5]) <= 0 or not 0 < values['mode_occupancy'] < 1
                    or values['trajectory_length_s'] < 2*values['observation_interval_s']):
                raise SwitchingError('CONTRACT_MISMATCH', 'qualification scenario physical scales differ')
        if (type(self.replicate_ids) not in (tuple, list) or not 1 <= len(self.replicate_ids) <= 1000
                or len(set(self.replicate_ids)) != len(self.replicate_ids)
                or any(type(value) is not str or not value for value in self.replicate_ids)):
            raise SwitchingError('CONTRACT_MISMATCH', 'unique bounded independent replicate IDs required')
        if type(self.thresholds) is not dict or set(self.thresholds) != set(METRICS):
            raise SwitchingError('CONTRACT_MISMATCH', 'all recovery/calibration thresholds must be present')
        for name, value in self.thresholds.items():
            if value is not None:
                _number(value, name)
        if type(self.parameter_scales) is not dict or set(self.parameter_scales) != set(PARAMETERS):
            raise SwitchingError('CONTRACT_MISMATCH', 'all parameter matching scales required')
        for name, value in self.parameter_scales.items():
            _number(value, name, np.finfo(float).tiny)
        for name in ('minimum_occupancy', 'maximum_collapse_fraction', 'minimum_recovery_fraction'):
            value = getattr(self, name)
            if value is not None:
                _number(value, name, 0, 0.5 if name == 'minimum_occupancy' else 1)
        if self.minimum_effective_replicates is not None and (type(self.minimum_effective_replicates) is not int
                or not 2 <= self.minimum_effective_replicates <= len(self.replicate_ids)):
            raise SwitchingError('CONTRACT_MISMATCH', 'effective independent replicate minimum differs')
        _number(self.confidence, 'confidence', 0.5, 0.999)
        _number(self.coverage_nominal, 'nominal coverage', 0.01, 0.99)
        if (type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**64
                or type(self.bootstrap_draws) is not int or not 100 <= self.bootstrap_draws <= 10_000
                or type(self.calibration_bins) is not int or not 2 <= self.calibration_bins <= 100
                or self.initialization_selection != 'selection-observation-objective-only'
                or self.collapse_criterion != 'inference-frozen-low-occupancy-or-identical-dynamics'
                or self.stop_rule != 'shared-arm-closure-stops-all-cells; no-budget-reset'):
            raise SwitchingError('CONTRACT_MISMATCH', 'frozen interval, selection or stopping rule differs')
        if len(self.scenario_axes)*len(self.replicate_ids)*2 > 10_000:
            raise SwitchingError('RESOURCE_LIMIT', 'paired qualification matrix exceeds cell quota')
        if self.scenario_roles is not None and (type(self.scenario_roles) is not dict
                or set(self.scenario_roles) != set(self.scenario_axes)
                or any(role not in ('required-qualified', 'required-unidentifiable', 'phase-diagram-only')
                       for role in self.scenario_roles.values())
                or 'required-qualified' not in self.scenario_roles.values()):
            raise SwitchingError('CONTRACT_MISMATCH', 'exact preregistered scenario roles and a required qualification range needed')
        return self

    def to_dict(self):
        self.validate()
        fields = dict(self.__dict__)
        if self.scenario_roles is None:
            fields.pop('scenario_roles')
        return {'schema_version': 'pirc28-identifiability-spec-v1' if self.scenario_roles is None else 'pirc28-identifiability-spec-v2', **fields,
                'replicate_ids': list(self.replicate_ids), 'sampling_unit': 'independent-synthetic-replicate'}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or value.get('schema_version') not in ('pirc28-identifiability-spec-v1', 'pirc28-identifiability-spec-v2'):
            raise SwitchingError('CONTRACT_MISMATCH', 'identifiability specification schema differs')
        try:
            supplied = {name: value[name] for name in cls.__dataclass_fields__ if name != 'scenario_roles'}
            if value['schema_version'] == 'pirc28-identifiability-spec-v2':
                supplied['scenario_roles'] = value['scenario_roles']
            supplied['replicate_ids'] = tuple(supplied['replicate_ids'])
            result = cls(**supplied)
        except (KeyError, TypeError) as error:
            raise SwitchingError('CONTRACT_MISMATCH', 'identifiability fields incomplete') from error
        if digest(result.to_dict()) != digest(value):
            raise SwitchingError('IDENTITY_CONFLICT', 'identifiability specification is not canonical')
        return result

    @property
    def missing_fields(self):
        self.validate()
        return [name for name, value in self.thresholds.items() if value is None] + [name for name in
            ('minimum_occupancy', 'maximum_collapse_fraction', 'minimum_recovery_fraction',
             'minimum_effective_replicates') if getattr(self, name) is None]


def _interval(values, spec, rng, poll):
    # Batch bootstrap avoids an unbounded replicate-by-draw allocation.
    means = []
    for start in range(0, spec.bootstrap_draws, 32):
        poll()
        count = min(32, spec.bootstrap_draws-start)
        means.extend(values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1).tolist())
    alpha = (1-spec.confidence)/2
    return [float(value) for value in np.quantile(means, [alpha, 1-alpha])]


def _wilson(successes, total, confidence):
    # Unlike the empirical bootstrap at the all-success boundary, this does
    # not turn a tiny successful sample into a certainty of population recovery.
    z = NormalDist().inv_cdf((1+confidence)/2)
    p, denominator = successes/total, 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def qualify(spec, rows, aggregate_hash, *, check_budget):
    """Evaluate a frozen aggregate; never issue real-data authorization.

    Rows are unique (scenario, independent replicate, K), not initialization
    successes. K=1 and K=2 must share input hashes. Unknown/missing metrics cannot
    pass. Source/attempt/admission binding is a separate shared-store obligation.
    """
    spec.validate()
    if type(rows) is not list or len(rows) > 10_000 or digest(rows) != aggregate_hash:
        raise SwitchingError('IDENTITY_CONFLICT', 'bounded aggregate hash differs')
    expected = {(scenario, replicate, k) for scenario in spec.scenario_axes for replicate in spec.replicate_ids for k in (1, 2)}
    indexed = {}
    independent_inputs = {}
    required = {'scenario_id', 'replicate_id', 'K', 'state', 'input_hash', 'artifact_hash',
                'metrics', 'posterior_occupancy', 'collapsed'}
    for row in rows:
        check_budget()
        if type(row) is not dict or set(row) != required:
            raise SwitchingError('CONTRACT_MISMATCH', 'qualification row schema differs')
        if (type(row['scenario_id']) is not str or type(row['replicate_id']) is not str or type(row['K']) is not int):
            raise SwitchingError('CONTRACT_MISMATCH', 'qualification row identity types differ')
        key = (row['scenario_id'], row['replicate_id'], row['K'])
        if key not in expected or key in indexed or row['state'] not in TERMINALS:
            raise SwitchingError('CONTRACT_MISMATCH', 'duplicate, undeclared or invalid qualification row')
        if row['state'] == 'SUCCEEDED':
            if (type(row['metrics']) is not dict or type(row['collapsed']) is not bool
                    or any(type(row[name]) is not str or not row[name] for name in ('input_hash', 'artifact_hash'))):
                raise SwitchingError('CONTRACT_MISMATCH', 'successful row lacks result/source identity')
            occupied = finite_array(row['posterior_occupancy'], (row['K'],), 'posterior occupancy')
            if np.any(occupied < 0) or not math.isclose(float(occupied.sum()), 1, abs_tol=1e-10):
                raise SwitchingError('CONTRACT_MISMATCH', 'posterior occupancy is not normalized')
            for name, value in row['metrics'].items():
                if type(name) is not str:
                    raise SwitchingError('CONTRACT_MISMATCH', 'metric names must be strings')
                _number(value, name, 0 if name in METRICS or name == 'position_energy_score_d2' else -math.inf)
            source = (row['scenario_id'], row['input_hash'])
            if source in independent_inputs and independent_inputs[source] != row['replicate_id']:
                raise SwitchingError('IDENTITY_CONFLICT', 'one input block cannot become multiple independent replicates')
            independent_inputs[source] = row['replicate_id']
        elif any(row[name] is not None for name in ('metrics', 'posterior_occupancy', 'collapsed')):
            raise SwitchingError('CONTRACT_MISMATCH', 'non-successful rows cannot carry successful-fit metrics')
        indexed[key] = row
    report = {'schema_version': 'pirc28-identifiability-gate-v1' if spec.scenario_roles is None else 'pirc28-identifiability-gate-v2', 'spec_hash': digest(spec.to_dict()),
        'aggregate_hash': aggregate_hash, 'real_data_authorized': False,
        'evidence_binding': 'requires-shared-store-source-attempt-admission-validation',
        'interpretation_level': 'synthetic-statistical-only', 'expected_rows': len(expected),
        'observed_rows': len(indexed), 'missing_fields': spec.missing_fields, 'scenarios': {}}
    if spec.scenario_roles is not None:
        report.update(scenario_roles=dict(spec.scenario_roles), applicable_scenarios=[], applicable_ranges=[],
                      scope_dispositions={}, scope_rule='preregistered-required-ranges-and-controls; no-posthoc-range-selection')
    if spec.missing_fields:
        report['scenarios'] = {name: {'state': 'NEEDS_PREREGISTRATION', 'axes': axes,
            'registered_replicates': len(spec.replicate_ids), 'effective_replicates': None}
            for name, axes in spec.scenario_axes.items()}
        return {**report, 'state': 'NEEDS_PREREGISTRATION'}
    rng = np.random.default_rng(spec.bootstrap_seed)
    for scenario in sorted(spec.scenario_axes):
        check_budget()
        candidates, paired, missing, failures, successful_k2 = [], [], [], [], []
        for replicate in spec.replicate_ids:
            one, two = indexed.get((scenario, replicate, 1)), indexed.get((scenario, replicate, 2))
            if two is not None and two['state'] == 'SUCCEEDED':
                successful_k2.append(two)
            if one is None or two is None or one['state'] == 'NOT_RUN' or two['state'] == 'NOT_RUN':
                missing.append(replicate)
                continue
            if one['state'] != 'SUCCEEDED' or two['state'] != 'SUCCEEDED':
                failures.append({'replicate_id': replicate, 'K1': one['state'], 'K2': two['state']})
                continue
            if one['input_hash'] != two['input_hash']:
                raise SwitchingError('IDENTITY_CONFLICT', 'K=1/K=2 input blocks differ')
            if set(METRICS)-set(two['metrics']) or 'position_energy_score_d2' not in one['metrics'] or 'position_energy_score_d2' not in two['metrics']:
                missing.append(replicate)
                continue
            candidates.append(two)
            paired.append(two['metrics']['position_energy_score_d2']-one['metrics']['position_energy_score_d2'])
        # Collapse and K=2 recovery diagnostics are observable even when its
        # reference K=1 fit fails. Only genuinely paired scores/recovery gates
        # require the reference; do not censor independent K=2 outcomes.
        collapsed = sum(value['collapsed'] for value in successful_k2)
        good = sum(not value['collapsed'] and min(value['posterior_occupancy']) >= spec.minimum_occupancy
                   and all(value['metrics'][name] <= limit for name, limit in spec.thresholds.items()) for value in candidates)
        total = len(spec.replicate_ids)
        metric_candidates = [value for value in successful_k2 if not set(METRICS)-set(value['metrics'])]
        intervals = {name: _interval(np.asarray([value['metrics'][name] for value in metric_candidates]), spec, rng, check_budget)
                     for name in METRICS} if len(metric_candidates) >= 2 else {}
        # Failure-inclusive fractions use all registered replicates. Bootstrap
        # confidence intervals for successful fits are separately conditioned.
        recovery_interval = _wilson(good, total, spec.confidence)
        reasons = []
        if len(candidates) < spec.minimum_effective_replicates:
            reasons.append('INSUFFICIENT_INDEPENDENT_REPLICATES')
        if recovery_interval[0] < spec.minimum_recovery_fraction:
            reasons.append('RECOVERY_OR_CALIBRATION_GATE_FAILED')
        if collapsed/total > spec.maximum_collapse_fraction:
            reasons.append('COLLAPSE_GATE_FAILED')
        budget_failed = any(value[k] in ('TIMEOUT', 'BUDGET_EXHAUSTED') for value in failures for k in ('K1', 'K2'))
        state = 'INCOMPLETE' if missing else 'BUDGET_EXHAUSTED' if budget_failed else 'UNIDENTIFIABLE' if reasons else 'PASSED'
        report['scenarios'][scenario] = {'state': state, 'axes': spec.scenario_axes[scenario],
            'reasons': reasons, 'registered_replicates': total,
            'effective_replicates': len(candidates), 'recovering_replicates': good, 'recovery_fraction': good/total,
            'recovery_fraction_interval': recovery_interval,
            'successful_K2_fits': len(successful_k2), 'known_K2_collapse_count': collapsed,
            'observed_collapse_fraction': None if not successful_k2 else collapsed/len(successful_k2),
            'observed_collapse_denominator': 'successful-K2-fits',
            'registered_collapse_lower_bound': collapsed/total,
            'collapse_threshold_denominator': 'all-registered-replicates',
            'unknown_K2_collapse_replicates': [replicate for replicate in spec.replicate_ids
                if not any(value['replicate_id'] == replicate for value in successful_k2)],
            'metric_effective_replicates': len(metric_candidates),
            'failures': failures, 'missing_replicates': missing, 'successful_fit_metric_mean_intervals': intervals,
            'paired_energy_difference_interval': _interval(np.asarray(paired), spec, rng, check_budget) if len(paired) >= 2 else None,
            'interval_sampling_unit': 'independent-synthetic-replicate', 'interval_method': 'bounded-percentile-bootstrap',
            'recovery_fraction_interval_method': 'Wilson-binomial-with-failure-inclusive-denominator',
            'within_path_samples_are_not_replicates': True}
    states = {value['state'] for value in report['scenarios'].values()}
    report['state'] = ('INCOMPLETE' if 'INCOMPLETE' in states else 'BUDGET_EXHAUSTED' if 'BUDGET_EXHAUSTED' in states
                       else 'UNIDENTIFIABLE' if 'UNIDENTIFIABLE' in states else 'PASSED')
    if spec.scenario_roles is not None:
        required_failed, controls_failed, budget_failures = [], [], []
        for scenario, point in report['scenarios'].items():
            check_budget()
            role = spec.scenario_roles[scenario]
            complete_pairs = (point['effective_replicates'] == point['registered_replicates']
                              and not point['failures'] and not point['missing_replicates'])
            if role == 'required-qualified':
                met = point['state'] == 'PASSED'
                if not met:
                    required_failed.append(scenario)
                else:
                    report['applicable_scenarios'].append(scenario)
                    # Exact frozen points, not an untested Cartesian bounding box.
                    report['applicable_ranges'].append({'scenario_id': scenario, 'axes': point['axes']})
            elif role == 'required-unidentifiable':
                met = point['state'] == 'UNIDENTIFIABLE' and complete_pairs
                if not met:
                    controls_failed.append(scenario)
            else:
                met = point['state'] not in ('INCOMPLETE', 'BUDGET_EXHAUSTED')
            if any(value[k] in ('TIMEOUT', 'BUDGET_EXHAUSTED') for value in point['failures'] for k in ('K1', 'K2')):
                budget_failures.append(scenario)
            report['scope_dispositions'][scenario] = {'role': role, 'observed_state': point['state'],
                'expectation_met': met, 'complete_successful_pairs': complete_pairs}
        report.update(required_qualification_failures=required_failed, negative_control_failures=controls_failed,
                      budget_failure_scenarios=budget_failures)
        report['state'] = ('INCOMPLETE' if 'INCOMPLETE' in states else 'BUDGET_EXHAUSTED' if 'BUDGET_EXHAUSTED' in states or budget_failures
                           else 'UNIDENTIFIABLE' if required_failed else 'CONTROL_GATE_FAILED' if controls_failed else 'PASSED')
        report['passed_required_scenarios'] = list(report['applicable_scenarios'])
        if report['state'] != 'PASSED':
            report['applicable_scenarios'], report['applicable_ranges'] = [], []
    return report
