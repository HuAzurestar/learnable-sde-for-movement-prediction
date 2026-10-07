"""Independent prefix/diagnostic consistency, NOT managed evidence authority.

Replays deterministic synthetic prefix inputs and small filtering/boundary checks;
rederives batch arithmetic without invoking the reference sampler or trusting its
PASS/errors/widths. Authentic reference execution, selection, admission, cost and
grants still require the supervisor/artifact path. No scientific gate is lifted.
"""

from dataclasses import replace
import math

import numpy as np

from inference.switching import InferenceSpec, filter_switching
from infrastructure.research_store import ResearchError, digest, encode
from models.switching import SwitchingSDEModel
from .approximation import ApproximationSpec, ERRORS, RECIPE_SCHEMA
from .approximation_checks import MC_METHOD, _initial, numerical_plan, observation_prefix, student_t_critical
from .synthetic import ObservationBatch, generate_scenario

MAXIMUM_REPORT_BYTES = 64*1024*1024
LITERALS = {'schema_version': 'pirc28-approximation-numerical-check-v1',
    'reference_continuous_state': 'path-conditional-affine-Kalman; no-state-particle-sampling',
    'reference_initial_conditioning': 'analytic-Gaussian-mode-mixture; iid-from-initial-posterior',
    'reference_resampling': 'multinomial-before-subsequent-intervals; no-redundant-initial-resampling',
    'reference_interval_method': MC_METHOD,
    'comparison': 'marginal-mode-probabilities-and-state-means; not-full-distribution-distance',
    'accuracy': 'finite-particle-reference; not-exact-joint-likelihood-or-posterior',
    'managed_evidence_bound': False, 'qualified_identifiability': False, 'real_data_authorized': False}
TOP_FIELDS = {*LITERALS, 'spec_hash', 'model_hash', 'observation_prefix_hash', 'initial_state_hash',
    'inference_hash', 'reference_seed', 'origin_index', 'state', 'paths', 'resource_plan'}
PATH_FIELDS = {'path_index', 'prefix_hash', 'state', 'reasons', 'step_halving_errors', 'reference_errors',
    'reference_half_widths', 'coarse_summary', 'fine_summary', 'reference_summary', 'reference_batch_summaries',
    'reference_minimum_ess_fraction', 'reference_jump_count', 'reference_batches', 'reference_summary_hash',
    'known_mode_boundary'}


def _require(condition, message):
    if not condition:
        raise ResearchError('UNQUALIFIED', message)


def _keys(value, keys, label):
    _require(type(value) is dict and set(value) == set(keys), label+' fields differ')


def _number(value):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, 'finite nonboolean diagnostic number required')
    return value


def _integer(value, low, high):
    _require(type(value) is int and low <= value <= high, 'bounded nonboolean diagnostic integer required')


def _same_declaration(value, expected, poll):
    """Bound shape/types against owner metadata BEFORE untrusted serialization."""
    poll()
    _require(type(value) is type(expected), 'resource declaration type differs')
    if type(expected) is dict:
        _keys(value, expected, 'resource declaration')
        for name in expected:
            _same_declaration(value[name], expected[name], poll)
    elif type(expected) is list:
        _require(len(value) == len(expected), 'resource declaration length differs')
        for actual, wanted in zip(value, expected):
            _same_declaration(actual, wanted, poll)
    else:
        _require(value == expected, 'expected numerical resource declaration differs')


def _array(value, shape, poll):
    # Verify lengths/types BEFORE allocating numpy copies of untrusted lists.
    def visit(item, dimensions):
        poll()
        if not dimensions:
            _number(item)
        else:
            _require(type(item) is list and len(item) == dimensions[0], 'diagnostic array shape differs')
            for child in item:
                visit(child, dimensions[1:])
    visit(value, shape)
    return np.asarray(value, dtype=float)


def _close(value, expected):
    if expected is None:
        _require(value is None, 'nonpositive likelihood interval must remain unavailable')
    else:
        _require(math.isclose(_number(value), float(expected), abs_tol=1e-10, rel_tol=1e-10),
                 'diagnostic arithmetic differs from independent derivation')


def _map(value, expected):
    _keys(value, expected, 'diagnostic arithmetic')
    for name in expected:
        _close(value[name], expected[name])


def _summary(value, n, k, poll):
    _keys(value, {'probabilities', 'means', 'log_likelihood'}, 'compact summary')
    p = _array(value['probabilities'], (n, k), poll)
    m = _array(value['means'], (n, 4), poll)
    _require(np.all(p >= 0) and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10), 'summary probabilities differ')
    return {'probabilities': p, 'means': m, 'log_likelihood': _number(value['log_likelihood'])}


def _match_summary(value, expected, n, k, poll):
    actual = _summary(value, n, k, poll)
    for name in ('probabilities', 'means'):
        _require(np.allclose(actual[name], expected[name], rtol=1e-10, atol=1e-10),
                 'published summary differs from independent derivation')
    _close(actual['log_likelihood'], expected['log_likelihood'])


def _filter(prefix, model, p, m, c, inference, path, poll):
    result = filter_switching(prefix, model, p, m, c, inference, path_index=path, check_budget=poll)
    return {'probabilities': np.asarray([row.probability for row in result.posteriors]),
        'means': np.asarray([row.probability@row.means for row in result.posteriors]),
        'log_likelihood': result.log_likelihood,
        'covariances': np.asarray([row.covariances for row in result.posteriors])}


def _difference(left, right, n):
    return {'mode_total_variation': float(np.max(np.linalg.norm(left['probabilities']-right['probabilities'], ord=1, axis=1)/2)),
        'position_mean_max_abs_m': float(np.max(np.abs(left['means'][:, :2]-right['means'][:, :2]))),
        'velocity_mean_max_abs_m_s': float(np.max(np.abs(left['means'][:, 2:]-right['means'][:, 2:]))),
        'log_likelihood_per_observation_abs': float(abs(left['log_likelihood']-right['log_likelihood'])/n)}


def _batches(values, ref, n, k, path, seed, initial, poll):
    _require(type(values) is list and len(values) == ref['independent_batches'], 'complete fixed batch roster required')
    ps, ms, traces, ess, jumps = [], [], [], [], []
    for index, value in enumerate(values):
        _keys(value, {'batch_index', 'seed_entropy', 'probabilities', 'means', 'log_likelihood_trace',
                     'minimum_ess_fraction', 'jump_count'}, 'batch')
        _integer(value['batch_index'], index, index)
        _require(type(value['seed_entropy']) is list and len(value['seed_entropy']) == 3
            and all(type(item) is int for item in value['seed_entropy'])
            and value['seed_entropy'] == [seed, path, index], 'batch seed/path/role identity differs')
        p = _array(value['probabilities'], (n, k), poll)
        m = _array(value['means'], (n, 4), poll)
        trace = _array(value['log_likelihood_trace'], (n,), poll)
        _require(np.all(p >= 0) and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10), 'batch probabilities differ')
        _require(np.allclose(p[0], initial['probabilities'][0], rtol=1e-10, atol=1e-10)
            and np.allclose(m[0], initial['means'][0], rtol=1e-10, atol=1e-10), 'analytic initial mixture summary differs')
        # The full initial evidence is the only contribution at time zero.
        _close(value['log_likelihood_trace'][0], initial['initial_log_likelihood'])
        minimum = _number(value['minimum_ess_fraction'])
        _require(1/ref['particles']-1e-12 <= minimum <= 1+1e-10, 'impossible batch ESS fraction')
        _integer(value['jump_count'], 0, ref['particles']*(n-1)*ref['maximum_jumps_per_particle_interval'])
        ps.append(p); ms.append(m); traces.append(trace); ess.append(minimum); jumps.append(value['jump_count'])
    p, m, ll = np.asarray(ps), np.asarray(ms), np.asarray(traces)
    count = len(values)
    critical = student_t_critical(count-1, ref['confidence'], check_budget=poll)
    # Independent arithmetic, not the producer's reference_summary/_errors.
    average_p, average_m = p.sum(axis=0)/count, m.sum(axis=0)/count
    def half_width(samples, average):
        return critical*np.sqrt(np.sum((samples-average)**2, axis=0)/(count-1)/count)
    p_width, m_width = half_width(p, average_p), half_width(m, average_m)
    maximum = float(np.max(ll[:, -1]))
    likelihood = np.exp(ll[:, -1]-maximum)
    average = float(np.sum(likelihood)/count)
    width = float(half_width(likelihood, average))
    log_width = None if width >= average else max(math.log1p(width/average), -math.log1p(-width/average))/n
    summary = {'probabilities': average_p, 'means': average_m, 'log_likelihood': maximum+math.log(average)}
    widths = {'mode_total_variation': float(np.max(np.sum(p_width, axis=1)/2)),
        'position_mean_max_abs_m': float(np.max(m_width[:, :2])),
        'velocity_mean_max_abs_m_s': float(np.max(m_width[:, 2:])),
        'log_likelihood_per_observation_abs': log_width}
    return summary, widths, min(ess), sum(jumps)


def _boundary(prefix, model, means, covariances, inference, settings, path, poll):
    results = []
    for mode in range(model.n_modes):
        one = SwitchingSDEModel([[0]], *(getattr(model, name).numpy()[mode:mode+1] for name in ('A', 'B', 'a', 'L')),
                               rate_bounds=model.rate_bounds)
        actual = _filter(prefix, one, [1.0], means[mode:mode+1], covariances[mode:mode+1], inference, path, poll)
        mean, covariance, likelihood = means[mode].copy(), covariances[mode].copy(), 0.0
        direct_m, direct_c = [], []
        for index, time in enumerate(prefix.times):
            poll()
            if index:
                f, a, noise = [value.numpy() for value in one.conditional_affine_transition(0, float(time-prefix.times[index-1]))]
                mean, covariance = f@mean+a, f@covariance@f.T+noise
            residual = prefix.values[path, index]-prefix.H@mean
            innovation = prefix.H@covariance@prefix.H.T+prefix.R
            root = np.linalg.cholesky((innovation+innovation.T)/2)
            cross = covariance@prefix.H.T
            gain = np.linalg.solve(root.T, np.linalg.solve(root, cross.T)).T
            whitened = np.linalg.solve(root, residual)
            likelihood -= (len(residual)*math.log(2*math.pi)+2*np.log(np.diag(root)).sum()+whitened@whitened)/2
            mean, covariance = mean+gain@residual, covariance-gain@cross.T
            covariance = (covariance+covariance.T)/2
            direct_m.append(mean.copy()); direct_c.append(covariance.copy())
        expected = (np.asarray(direct_m), np.asarray(direct_c), likelihood)
        observed = (actual['means'], actual['covariances'][:, 0], actual['log_likelihood'])
        passed = all(np.allclose(left, right, atol=settings['absolute_tolerance'], rtol=settings['relative_tolerance'])
                     for left, right in zip(observed, expected))
        results.append({'mode': mode, 'passed': bool(passed),
            'mean_max_abs': float(np.max(np.abs(observed[0]-expected[0]))),
            'covariance_max_abs': float(np.max(np.abs(observed[1]-expected[1]))),
            'log_likelihood_abs': float(abs(observed[2]-expected[2]))})
    return results


def validate_prefix_numbers(report, observations, model, probability, means, covariances, inference, spec,
                            *, origin_index, seed, check_budget):
    """Recheck reported numbers against caller-owned inputs/model; no authority."""
    _require(callable(check_budget), 'explicit verifier budget poll required')
    _integer(seed, 0, 2**64-1)
    _require(isinstance(spec, ApproximationSpec) and isinstance(model, SwitchingSDEModel)
             and isinstance(inference, InferenceSpec), 'canonical expected inputs required')
    check_budget()
    _keys(report, TOP_FIELDS, 'numerical report')
    _require(all(type(report[name]) is type(value) and report[name] == value for name, value in LITERALS.items()),
             'unsupported numerical method or authority assertion')
    prefix = observation_prefix(observations, origin_index)
    plan = numerical_plan(spec, inference, prefix, model)
    p, m, c = _initial(model, probability, means, covariances)
    identity = {'spec_hash': spec.spec_hash, 'model_hash': digest(model.to_dict()),
        'observation_prefix_hash': digest(prefix.to_dict()),
        'initial_state_hash': digest({'probability': p.tolist(), 'means': m.tolist(), 'covariances': c.tolist()}),
        'inference_hash': digest(inference.to_dict())}
    _require(all(type(report[name]) is str and report[name] == value for name, value in identity.items()),
             'expected prefix/model/prior/spec/inference identity differs')
    _integer(report['origin_index'], origin_index, origin_index); _integer(report['reference_seed'], seed, seed)
    _same_declaration(report['resource_plan'], plan, check_budget)
    rows = report['paths']
    _require(type(rows) is list and len(rows) == len(prefix.values), 'every expected prefix path required')
    n, k, settings, states = len(prefix.times), model.n_modes, spec.to_dict(), []
    for index, row in enumerate(rows):
        check_budget()
        _keys(row, PATH_FIELDS, 'path diagnostic')
        _integer(row['path_index'], index, index)
        expected_hash = digest({'times': prefix.times.tolist(), 'values': prefix.values[index].tolist(),
                                'H': prefix.H.tolist(), 'R': prefix.R.tolist()})
        _require(row['prefix_hash'] == expected_hash, 'expected path input hash differs')
        coarse = _filter(prefix, model, p, m, c, inference, index, check_budget)
        fine = _filter(prefix, model, p, m, c, spec.fine_inference(inference), index, check_budget)
        _match_summary(row['coarse_summary'], coarse, n, k, check_budget)
        _match_summary(row['fine_summary'], fine, n, k, check_budget)
        first = ObservationBatch(prefix.times[:1], prefix.values[:, :1], prefix.H, prefix.R, prefix.scenario_hash)
        # filter_switching requires propagation; derive the initial evidence via
        # its first posterior plus a separate exact mixture likelihood instead.
        scores = []
        for mode in range(k):
            if p[mode] > 0:
                residual = first.values[index, 0]-first.H@m[mode]
                covariance = first.H@c[mode]@first.H.T+first.R
                root = np.linalg.cholesky(covariance)
                whitened = np.linalg.solve(root, residual)
                scores.append(math.log(p[mode])-(len(residual)*math.log(2*math.pi)
                    +2*np.log(np.diag(root)).sum()+whitened@whitened)/2)
        maximum = max(scores)
        coarse['initial_log_likelihood'] = maximum+math.log(sum(math.exp(value-maximum) for value in scores))
        reference, widths, ess, jumps = _batches(row['reference_batch_summaries'], settings['event_driven_reference'],
            n, k, index, seed, coarse, check_budget)
        _match_summary(row['reference_summary'], reference, n, k, check_budget)
        _require(row['reference_summary_hash'] == digest(row['reference_summary']), 'published reference hash differs')
        halving, differences = _difference(coarse, fine, n), _difference(coarse, reference, n)
        _map(row['step_halving_errors'], halving); _map(row['reference_errors'], differences)
        _map(row['reference_half_widths'], widths); _close(row['reference_minimum_ess_fraction'], ess)
        _integer(row['reference_jump_count'], jumps, jumps)
        batches = settings['event_driven_reference']['independent_batches']
        _integer(row['reference_batches'], batches, batches)
        boundary = _boundary(prefix, model, m, c, inference, settings['known_mode_boundary'], index, check_budget)
        _require(type(row['known_mode_boundary']) is list and len(row['known_mode_boundary']) == k, 'all fixed-mode boundaries required')
        for actual, expected in zip(row['known_mode_boundary'], boundary):
            _keys(actual, expected, 'boundary diagnostic')
            _integer(actual['mode'], expected['mode'], expected['mode'])
            _require(type(actual['passed']) is bool and actual['passed'] is expected['passed'], 'boundary disposition differs')
            for name in ('mean_max_abs', 'covariance_max_abs', 'log_likelihood_abs'):
                _close(actual[name], expected[name])
        reasons, ref = [], settings['event_driven_reference']
        if any(halving[name] > settings['step_halving']['thresholds'][name] for name in ERRORS): reasons.append('STEP_HALVING_GATE_FAILED')
        if ess < ref['minimum_ess_fraction']: reasons.append('REFERENCE_ESS_GATE_FAILED')
        if any(widths[name] is None or widths[name] > ref['maximum_mc_half_width_fraction']*ref['thresholds'][name] for name in ERRORS):
            reasons.append('REFERENCE_MC_PRECISION_FAILED')
        if any(widths[name] is None or differences[name]+widths[name] > ref['thresholds'][name] for name in ERRORS):
            reasons.append('REFERENCE_DISCREPANCY_GATE_FAILED')
        if any(not value['passed'] for value in boundary): reasons.append('KNOWN_MODE_BOUNDARY_GATE_FAILED')
        state = 'INFERENCE_INAPPLICABLE' if reasons else 'NUMERICAL_CHECKS_PASSED'
        _require(type(row['reasons']) is list and row['reasons'] == reasons and row['state'] == state, 'complete gate/failure disposition differs')
        states.append(state)
    state = 'INFERENCE_INAPPLICABLE' if 'INFERENCE_INAPPLICABLE' in states else 'NUMERICAL_CHECKS_PASSED'
    _require(report['state'] == state, 'whole-prefix gate disposition differs')
    content = encode(report)
    _require(len(content) <= MAXIMUM_REPORT_BYTES, 'numerical evidence byte quota exceeded')
    check_budget()
    return {'schema_version': 'pirc28-prefix-consistency-v1', 'report_hash': digest(report),
        'state': state, 'input_identity_bound': True, 'numerical_diagnostics_consistent': True,
        'reference_sampling_verified': False, 'selected_fit_selection_verified': False,
        'managed_evidence_bound': False, 'qualified_identifiability': False, 'real_data_authorized': False}


def reference_seed(recipe, cell_seed):
    """Frozen domain separation, never selected-fit/outcome-conditioned reseeding."""
    _integer(cell_seed, 0, 2**64-1)
    identity = digest({'recipe': recipe, 'K': 2, 'seed': cell_seed})
    return int(digest([identity, 'pirc28-prefix-reference-v1'])[:16], 16)


def replay_test_prefix(recipe, cell_seed, *, check_budget, k=2):
    """Observation-only deterministic replay; no future generation/fit/store."""
    from .pipeline import recipe_allocation, validate_recipe
    from .runtime import switching_plugin
    from application.research_registry import plan_resources
    _require(callable(check_budget), 'explicit input replay budget poll required')
    _integer(cell_seed, 0, 2**64-1)
    _integer(k, 1, 2)
    _require(type(recipe) is dict and recipe.get('schema_version') == RECIPE_SCHEMA, 'frozen V2 recipe required')
    check_budget()
    scenario, _, _, _, _ = validate_recipe(recipe, k)
    allocation = recipe_allocation(recipe, k)
    # A small reference alone does not authorize its concurrent source/fit/
    # forecast envelope. Enforce the registered SUM before generating anything.
    plan_resources(switching_plugin().registry_entry, allocation['config'], allocation['inputs'], matrix_cells=1)
    check_budget()
    seeds = np.random.default_rng(cell_seed).integers(0, 2**63, size=3, dtype=np.int64)
    full = replace(scenario, scenario_id=scenario.scenario_id+'-test', seed=int(seeds[2]), sample_count=recipe['test_paths'])
    short = replace(full, times=full.times[:recipe['forecast_origin_index']+1])
    observations, _, _ = generate_scenario(short, check_budget=check_budget)
    # Full source identity is unchanged even though future values were NEVER
    # generated; per-path/role seed streams make this prefix equal to full replay.
    prefix = replace(observations, scenario_hash=full.scenario_hash)
    return prefix, {'schema_version': 'pirc28-replayed-prefix-input-v1',
        'recipe_cell_hash': digest({'recipe': recipe, 'K': k, 'seed': cell_seed}),
        'test_split_seed': int(seeds[2]), 'test_scenario_hash': full.scenario_hash,
        'observation_prefix_hash': digest(prefix.to_dict()),
        'reference_seed': reference_seed(recipe, cell_seed) if k == 2 else None,
        'prefix_inputs_replayed': True, 'full_generated_input_hashes_verified': False,
        'source_authority_verified': False, 'managed_evidence_bound': False}
