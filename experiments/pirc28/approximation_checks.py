"""Bounded observation-prefix numerical checks, NOT managed qualification.

Reference particles sample autonomous CTMC paths and integrate the continuous
Gaussian state conditionally (Rao-Blackwellisation). They do not use endpoint-
mode affine emissions or sample future observations. Finite-particle posterior
and log-likelihood bias are not bounded by the nominal batch MC intervals.
No store, budget authority, retry, fit selection or scientific grant is created.
"""

import math

import numpy as np

from inference.switching import (InferenceSpec, covariance_checked, filter_switching,
                                 normalize_log_weights)
from infrastructure.research_store import ResearchError, digest
from models.switching import SwitchingError, SwitchingSDEModel, finite_array, psd_array
from .approximation import ApproximationSpec, ERRORS
from .synthetic import ObservationBatch

MAXIMUM_ELEMENTS = 20_000_000
MAXIMUM_SEGMENTS = 50_000_000
MC_METHOD = 'nominal-pointwise-independent-batch-Student-t; normal-approximation; finite-particle-bias-not-bounded'


def observation_prefix(observations, origin_index):
    """Copy only admitted-shape prefixes; even nonfinite FUTURE values are unused."""
    if type(observations) is not ObservationBatch:
        raise SwitchingError('CONTRACT_MISMATCH', 'only observation batches are accepted')
    times, values = np.asarray(observations.times), np.asarray(observations.values)
    if (times.ndim != 1 or type(origin_index) is not int or not 1 <= origin_index < len(times)
            or values.ndim != 3 or values.shape[1] != len(times) or not 1 <= len(values) <= 64
            or values.shape[2] not in (1, 2, 3, 4)):
        raise SwitchingError('CONTRACT_MISMATCH', 'bounded paths and a nonempty propagation prefix required')
    prefix_times = finite_array(times[:origin_index+1], (origin_index+1,), 'prefix times')
    if prefix_times[0] != 0 or np.any(np.diff(prefix_times) <= 0):
        raise SwitchingError('CONTRACT_MISMATCH', 'prefix schedule must start at zero and increase')
    if values.shape[0]*(origin_index+1)*values.shape[2] > MAXIMUM_ELEMENTS:
        raise SwitchingError('RESOURCE_LIMIT', 'prefix observations exceed the allocation quota')
    y = finite_array(values[:, :origin_index+1], (len(values), origin_index+1, values.shape[2]), 'prefix observations')
    h = finite_array(observations.H, (values.shape[2], 4), 'prefix H')
    r = psd_array(observations.R, (values.shape[2], values.shape[2]), 'prefix R')
    return ObservationBatch(prefix_times, y, h, r, observations.scenario_hash)


def numerical_plan(spec, inference, prefix, model):
    """Conservative bounded core allocation/work declaration, not admission.

    Coarse/fine histories and batch computations are sequential; only compact
    observation-time summaries survive them. The future adapter must additionally
    account for concurrent source-fit/forecast allocations and register this plan.
    """
    if not isinstance(spec, ApproximationSpec) or not isinstance(model, SwitchingSDEModel):
        raise SwitchingError('CONTRACT_MISMATCH', 'canonical specification and model required')
    return schedule_plan(spec, inference, prefix.times, len(prefix.times)-1,
        len(prefix.values), model.n_modes, model.rate_bounds[1], prefix.values.shape[-1])


def schedule_plan(spec, inference, times, origin_index, paths, k, rate_upper, observation_dimension):
    """The same core bounds from schedule metadata, without allocating fake data."""
    if (not isinstance(spec, ApproximationSpec) or not isinstance(inference, InferenceSpec)
            or type(paths) is not int or not 1 <= paths <= 64
            or type(k) is not int or k not in (1, 2)
            or type(observation_dimension) is not int or not 1 <= observation_dimension <= 4
            or type(rate_upper) not in (int, float) or not math.isfinite(rate_upper) or rate_upper < 0):
        raise SwitchingError('CONTRACT_MISMATCH', 'bounded canonical reference schedule required')
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or type(origin_index) is not int or not 1 <= origin_index < len(times)
            or origin_index >= 100_000):
        raise SwitchingError('CONTRACT_MISMATCH', 'nonempty bounded reference prefix required')
    times = times[:origin_index+1]
    if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise SwitchingError('CONTRACT_MISMATCH', 'finite increasing reference prefix required')
    inference.validate()
    settings = spec.to_dict()
    fine = spec.prefix_preflight(inference, times, origin_index, rate_upper)
    delta = min(inference.max_step_seconds, inference.max_rate_step/rate_upper) if rate_upper else inference.max_step_seconds
    coarse_steps = sum(math.ceil(float(dt)/delta) for dt in np.diff(times))
    if coarse_steps > inference.max_substeps:
        raise SwitchingError('INFERENCE_INAPPLICABLE', 'coarse prefix exceeds its permitted-rate quota')
    n = len(times)
    ref = settings['event_driven_reference']
    # Retain the complete published numeric report while independently parsing
    # and serializing it. These are ADDITIONAL to particle/filter workspaces;
    # do not quietly charge live report copies against a discarded phase.
    report_elements = 32+paths*(19+3*n*(k+4)+ref['independent_batches']*(n*(k+5)+6)+5*k)
    copy_elements = 2*report_elements
    trace_elements = 32+16*paths*ref['independent_batches']
    elements = (max(coarse_steps, fine['maximum_prefix_substeps'])*k*k*16*24
        + ref['particles']*96 + paths*ref['independent_batches']*n*(k+5)
        + paths*n*(k+4+16)*4 + paths*n*observation_dimension+report_elements+copy_elements+3*trace_elements)
    segments = paths*ref['independent_batches']*ref['particles']*(n-1)*(ref['maximum_jumps_per_particle_interval']+1)
    if elements > MAXIMUM_ELEMENTS or segments > MAXIMUM_SEGMENTS:
        raise SwitchingError('RESOURCE_LIMIT', 'reference allocation or worst-case segment work exceeds the core quota')
    return {'schema_version': 'pirc28-approximation-core-plan-v1', 'tensor_elements': elements,
        'tensor_bytes': elements*8, 'maximum_event_segments': segments,
        'coarse_prefix_substeps': coarse_steps, 'fine_prefix_substeps': fine['maximum_prefix_substeps'],
        'reference_particles': ref['particles'], 'independent_batches': ref['independent_batches'],
        'retained_report_numeric_elements': report_elements, 'owner_report_copy_elements': copy_elements,
        'completed_batch_trace_elements': trace_elements, 'owner_trace_copy_elements': 2*trace_elements,
        'reference_storage': 'sequential-path-and-batch; no-particle-or-jump-history',
        'resource_admitted': False, 'managed_executor_implemented': False}


def reference_design_preflight(spec, inference, times, origin_index, paths, rate_upper, observation_dimension):
    """Pre-data K2 design sanity, NOT empirical precision or shared admission.

    A balanced IID Bernoulli benchmark is only a necessary engineering check;
    actual weighted/resampled filtering batches can have much greater variance.
    Uniformisation bounds jump counts by Poisson(rate_upper*dt). Chernoff bounds
    and a union bound need no independence of ancestor paths or intervals.
    """
    plan = schedule_plan(spec, inference, times, origin_index, paths, 2, rate_upper, observation_dimension)
    ref = spec.to_dict()['event_driven_reference']
    critical = student_t_critical(ref['independent_batches']-1, ref['confidence'], check_budget=lambda: None)
    benchmark = critical*0.5/math.sqrt(ref['particles']*ref['independent_batches'])
    limit = ref['maximum_mc_half_width_fraction']*ref['thresholds']['mode_total_variation']
    if benchmark > limit:
        raise SwitchingError('CONTRACT_MISMATCH', 'balanced IID reference precision benchmark exceeds its frozen MC limit')
    m = ref['maximum_jumps_per_particle_interval']+1
    tail_sum = 0.0
    for dt in np.diff(np.asarray(times, dtype=float)[:origin_index+1]):
        lam = rate_upper*float(dt)
        if rate_upper == 0:
            bound = 0.0
        elif m <= lam:
            bound = 1.0
        else:
            # A positive-rate mathematical tail must not become an exact zero
            # assertion when multiplication/exponentiation underflows.
            smallest = float(np.nextafter(0.0, 1.0))
            bound = smallest if lam == 0 else max(smallest, math.exp(-lam+m*(1+math.log(lam)-math.log(m))))
        tail_sum += bound
    union = min(1.0, paths*ref['independent_batches']*ref['particles']*tail_sum)
    return {'schema_version': 'pirc28-reference-design-sanity-v1', 'core_plan': plan,
        'balanced_iid_nominal_half_width_benchmark': benchmark, 'registered_mode_mc_half_width_limit': limit,
        'particle_interval_jump_quota_exceedance_union_bound': union,
        'jump_tail_method': 'Poisson-uniformisation-Chernoff-union; exceeding-quota-still-fails',
        'actual_mc_precision_proven': False, 'finite_particle_bias_bounded': False,
        'qualified_identifiability': False}


def _initial(model, probability, means, covariances):
    k = model.n_modes
    p = finite_array(probability, (k,), 'initial probability')
    if np.any(p < 0) or not math.isclose(float(p.sum()), 1, abs_tol=1e-12, rel_tol=0):
        raise SwitchingError('INVALID_PROBABILITY', 'initial probabilities must normalize')
    return p, finite_array(means, (k, 4), 'initial means'), psd_array(covariances, (k, 4, 4), 'initial covariance')


def _kalman_update(mean, covariance, y, h, r):
    # Independent direct implementation: Cholesky score and covariance subtraction,
    # not the IMM's slogdet/Joseph measurement_update implementation.
    residual = y-h@mean
    innovation = h@covariance@h.T+r
    root = np.linalg.cholesky((innovation+innovation.T)/2)
    whitened = np.linalg.solve(root, residual)
    gain = np.linalg.solve(innovation, h@covariance).T
    updated = covariance_checked(covariance-gain@h@covariance)
    log_score = -0.5*(len(y)*math.log(2*math.pi)+2*np.log(np.diag(root)).sum()+whitened@whitened)
    if not math.isfinite(float(log_score)):
        raise SwitchingError('NONFINITE', 'reference observation likelihood is nonfinite')
    return mean+gain@residual, updated, float(log_score)


def _summary(filtered):
    value = {'probabilities': np.asarray([p.probability for p in filtered.posteriors]),
        'means': np.asarray([p.probability@p.means for p in filtered.posteriors]),
        'log_likelihood': filtered.log_likelihood,
        'conditional_covariances': np.asarray([p.covariances for p in filtered.posteriors])}
    if not all(np.isfinite(item).all() for item in value.values()):
        raise SwitchingError('NONFINITE', 'filter prefix summary is nonfinite')
    return value


def _errors(left, right, n):
    return {'mode_total_variation': float(np.max(np.sum(abs(left['probabilities']-right['probabilities']), axis=-1)/2)),
        'position_mean_max_abs_m': float(np.max(abs(left['means'][:, :2]-right['means'][:, :2]))),
        'velocity_mean_max_abs_m_s': float(np.max(abs(left['means'][:, 2:]-right['means'][:, 2:]))),
        'log_likelihood_per_observation_abs': abs(left['log_likelihood']-right['log_likelihood'])/n}


def _compact(summary):
    return {'probabilities': summary['probabilities'].tolist(), 'means': summary['means'].tolist(),
            'log_likelihood': float(summary['log_likelihood'])}


def student_t_critical(degrees, confidence, *, check_budget):
    """Integer-df t inversion via cos-power integration; no new dependency.

    Substituting x=sqrt(df)*tan(theta) in the Student density reduces its
    positive CDF integral to cos(theta)^(df-1), evaluated by finite recurrence.
    This numerical critical value does NOT justify normal MC-batch assumptions.
    """
    if type(degrees) is not int or not 3 <= degrees <= 31 or type(confidence) not in (int, float) or not 0.9 <= confidence <= 0.999:
        raise SwitchingError('CONTRACT_MISMATCH', 'reference batch confidence/df differ')
    coefficient = math.exp(math.lgamma((degrees+1)/2)-0.5*math.log(math.pi)-math.lgamma(degrees/2))
    def cdf(x):
        theta = math.atan(x/math.sqrt(degrees))
        sine, cosine = math.sin(theta), math.cos(theta)
        power = degrees-1
        integral = theta if power%2 == 0 else sine
        for exponent in range(2 if power%2 == 0 else 3, power+1, 2):
            integral = sine*cosine**(exponent-1)/exponent+(exponent-1)/exponent*integral
        return 0.5+coefficient*integral
    target, low, high = (1+confidence)/2, 0.0, 64.0
    if cdf(high) < target:
        raise SwitchingError('NONFINITE', 'Student critical value outside bounded inversion bracket')
    for _ in range(64):
        check_budget()
        middle = (low+high)/2
        if cdf(middle) < target:
            low = middle
        else:
            high = middle
    return (low+high)/2


def _event_predict(model, mode, mean, covariance, dt, rng, maximum_jumps, kernels, check_budget):
    q, elapsed, jumps = model.Q.numpy(), 0.0, 0
    while elapsed < dt:
        check_budget()
        rate = float(-q[mode, mode])
        wait = float(rng.exponential(1/rate)) if rate else math.inf
        if wait <= 0 or elapsed+wait <= elapsed:
            raise SwitchingError('NONFINITE', 'reference jump clock lost resolution')
        segment = min(wait, dt-elapsed)
        f, c, noise = kernels[mode] if segment == dt else [item.numpy() for item in model.conditional_affine_transition(mode, segment)]
        mean = f@mean+c
        covariance = covariance_checked(f@covariance@f.T+noise)
        elapsed += segment
        if wait <= segment:
            if jumps >= maximum_jumps:
                raise SwitchingError('INFERENCE_INAPPLICABLE', 'reference particle/interval jump quota exceeded')
            weights = q[mode].copy()
            weights[mode] = 0
            mode = int(rng.choice(model.n_modes, p=weights/weights.sum()))
            jumps += 1
    return mode, mean, covariance, jumps


def _reference_batch(prefix, model, probability, means, covariances, settings, seed, batch_index, path_index, check_budget):
    particles, n, k = settings['particles'], len(prefix.times), model.n_modes
    # Role separation: initial mode, event path and resampling. Never borrow
    # generator/truth RNG or consume extra batches after seeing an MC interval.
    streams = [np.random.default_rng(np.random.SeedSequence([seed, path_index, batch_index, role])) for role in range(3)]
    rng_initial, rng_events, rng_resample = streams
    # Condition the finite initial Gaussian mixture analytically BEFORE sampling.
    # This supplies exact y0 evidence/summaries and IID particles from p(z0|y0),
    # not a noisy importance-weighted sample from p(z0). Zero-prior modes need
    # neither log(0) nor a meaningless singular Kalman update.
    initial_means, initial_covariances = means.copy(), covariances.copy()
    initial_scores = np.full(k, -math.inf)
    for mode in range(k):
        check_budget()
        if probability[mode] > 0:
            initial_means[mode], initial_covariances[mode], score = _kalman_update(
                means[mode], covariances[mode], prefix.values[path_index, 0], prefix.H, prefix.R)
            initial_scores[mode] = math.log(probability[mode])+score
    initial_posterior, likelihood = normalize_log_weights(initial_scores)
    modes = rng_initial.choice(k, size=particles, p=initial_posterior)
    particle_means, particle_covariances = initial_means[modes].copy(), initial_covariances[modes].copy()
    scores, weights = np.zeros(particles), np.ones(particles)/particles
    probabilities, marginal_means = [initial_posterior], [initial_posterior@initial_means]
    ess, log_trace, jumps = [1.0], [likelihood], 0
    for index in range(1, n):
        check_budget()
        # Fresh initial particles already have the exact initial posterior law.
        # Resampling them again would add avoidable genealogical variance.
        if index > 1:
            ancestors = rng_resample.choice(particles, size=particles, p=weights)
            modes = modes[ancestors].copy()
            particle_means, particle_covariances = particle_means[ancestors].copy(), particle_covariances[ancestors].copy()
        dt = float(prefix.times[index]-prefix.times[index-1])
        kernels = [[item.numpy() for item in model.conditional_affine_transition(mode, dt)] for mode in range(k)]
        for particle in range(particles):
            check_budget()
            mode, mean, covariance, count = _event_predict(model, int(modes[particle]),
                particle_means[particle], particle_covariances[particle], dt, rng_events,
                settings['maximum_jumps_per_particle_interval'], kernels, check_budget)
            modes[particle], particle_means[particle], particle_covariances[particle] = mode, mean, covariance
            jumps += count
            particle_means[particle], particle_covariances[particle], scores[particle] = _kalman_update(
                particle_means[particle], particle_covariances[particle], prefix.values[path_index, index], prefix.H, prefix.R)
        weights, normalization = normalize_log_weights(scores)
        likelihood += normalization-math.log(particles)
        probabilities.append(np.bincount(modes, weights=weights, minlength=k))
        marginal_means.append(weights@particle_means)
        ess.append(float(1/(particles*np.sum(weights**2))))
        log_trace.append(likelihood)
    return {'probabilities': np.asarray(probabilities), 'means': np.asarray(marginal_means),
        'log_likelihood': np.asarray(log_trace), 'minimum_ess_fraction': min(ess), 'jump_count': jumps}


def reference_summary(batches, confidence, *, check_budget):
    """Nominal batch-mean intervals; never average log likelihoods as likelihood.

    Likelihoods are averaged on their original positive scale (rescaled stably),
    then logged. The transformed interval refuses a nonpositive lower bound.
    This does not remove finite-particle/finite-batch log or posterior bias.
    """
    if type(batches) is not list or not 4 <= len(batches) <= 32:
        raise SwitchingError('CONTRACT_MISMATCH', 'the whole frozen independent batch roster is required')
    critical = student_t_critical(len(batches)-1, confidence, check_budget=check_budget)
    ps = np.asarray([value['probabilities'] for value in batches], dtype=float)
    ms = np.asarray([value['means'] for value in batches], dtype=float)
    ll = np.asarray([value['log_likelihood'] for value in batches], dtype=float)
    if (ps.ndim != 3 or not 2 <= ps.shape[1] <= 100_000 or ps.shape[-1] not in (1, 2) or ms.shape != (*ps.shape[:2], 4)
            or ll.shape != ps.shape[:2] or not all(np.isfinite(x).all() for x in (ps, ms, ll))
            or np.any(ps < 0) or not np.allclose(ps.sum(axis=-1), 1, atol=1e-10, rtol=0)):
        raise SwitchingError('CONTRACT_MISMATCH', 'finite complete reference batch summaries required')
    if any(type(value['jump_count']) is not int or value['jump_count'] < 0
           or type(value['minimum_ess_fraction']) not in (int, float)
           or not math.isfinite(value['minimum_ess_fraction'])
           or not 0 < value['minimum_ess_fraction'] <= 1+1e-10 for value in batches):
        raise SwitchingError('CONTRACT_MISMATCH', 'reference ESS/jump disclosures differ')
    p, m = ps.mean(axis=0), ms.mean(axis=0)
    p_width = critical*ps.std(axis=0, ddof=1)/math.sqrt(len(batches))
    m_width = critical*ms.std(axis=0, ddof=1)/math.sqrt(len(batches))
    maximum = float(ll[:, -1].max())
    likelihoods = np.exp(ll[:, -1]-maximum)
    average = float(likelihoods.mean())
    width = float(critical*likelihoods.std(ddof=1)/math.sqrt(len(batches)))
    logarithm = maximum+math.log(average)
    log_width = None if width >= average else max(-math.log1p(-width/average), math.log1p(width/average))/ps.shape[1]
    return {'probabilities': p, 'means': m, 'log_likelihood': logarithm,
        'half_widths': {'mode_total_variation': float(np.max(p_width.sum(axis=-1)/2)),
            'position_mean_max_abs_m': float(np.max(m_width[:, :2])),
            'velocity_mean_max_abs_m_s': float(np.max(m_width[:, 2:])),
            'log_likelihood_per_observation_abs': log_width},
        'confidence': confidence, 'interval_method': MC_METHOD,
        'minimum_ess_fraction': min(value['minimum_ess_fraction'] for value in batches),
        'total_jump_count': sum(value['jump_count'] for value in batches),
        'likelihood_aggregation': 'log-of-mean-positive-likelihoods; not-mean-log-likelihood'}


def _known_mode_checks(prefix, model, means, covariances, inference, settings, path_index, check_budget):
    results = []
    for mode in range(model.n_modes):
        check_budget()
        one = SwitchingSDEModel([[0]], *(getattr(model, name).numpy()[mode:mode+1] for name in ('A', 'B', 'a', 'L')),
                               rate_bounds=model.rate_bounds)
        filtered = filter_switching(prefix, one, [1.0], means[mode:mode+1], covariances[mode:mode+1], inference,
                                    path_index=path_index, check_budget=check_budget)
        direct_means, direct_covariances, likelihood = [], [], 0.0
        mean, covariance = means[mode].copy(), covariances[mode].copy()
        for index in range(len(prefix.times)):
            check_budget()
            if index:
                f, c, noise = [item.numpy() for item in one.conditional_affine_transition(0, float(prefix.times[index]-prefix.times[index-1]))]
                mean, covariance = f@mean+c, covariance_checked(f@covariance@f.T+noise)
            mean, covariance, score = _kalman_update(mean, covariance, prefix.values[path_index, index], prefix.H, prefix.R)
            direct_means.append(mean.copy())
            direct_covariances.append(covariance.copy())
            likelihood += score
        actual = _summary(filtered)
        expected = (np.asarray(direct_means), np.asarray(direct_covariances), likelihood)
        observed = (actual['means'], actual['conditional_covariances'][:, 0], actual['log_likelihood'])
        good = all(np.allclose(left, right, atol=settings['absolute_tolerance'], rtol=settings['relative_tolerance'])
                   for left, right in zip(observed, expected))
        results.append({'mode': mode, 'passed': bool(good),
            'mean_max_abs': float(np.max(abs(observed[0]-expected[0]))),
            'covariance_max_abs': float(np.max(abs(observed[1]-expected[1]))),
            'log_likelihood_abs': abs(observed[2]-expected[2])})
    return results


def _published_batch(value, index, seed, path):
    return {'batch_index': index, 'seed_entropy': [seed, path, index],
        'probabilities': value['probabilities'].tolist(), 'means': value['means'].tolist(),
        'log_likelihood_trace': value['log_likelihood'].tolist(),
        'minimum_ess_fraction': value['minimum_ess_fraction'], 'jump_count': value['jump_count']}


def check_prefix_approximation(observations, model, probability, means, covariances, inference, spec,
                               *, origin_index, seed, check_budget, execution_callback=None):
    """Check every prefix path without fitting/selecting or granting qualification."""
    if not callable(check_budget) or type(seed) is not int or not 0 <= seed < 2**64:
        raise SwitchingError('CONTRACT_MISMATCH', 'explicit budget poll and unsigned reference seed required')
    if execution_callback is not None and not callable(execution_callback):
        raise SwitchingError('CONTRACT_MISMATCH', 'explicit callable batch completion sink required')
    prefix = observation_prefix(observations, origin_index)
    plan = numerical_plan(spec, inference, prefix, model)
    p, m, c = _initial(model, probability, means, covariances)
    settings, paths = spec.to_dict(), []
    for path_index in range(len(prefix.values)):
        check_budget()
        coarse = _summary(filter_switching(prefix, model, p, m, c, inference, path_index=path_index, check_budget=check_budget))
        fine = _summary(filter_switching(prefix, model, p, m, c, spec.fine_inference(inference),
                                        path_index=path_index, check_budget=check_budget))
        halving = _errors(coarse, fine, len(prefix.times))
        batches = []
        for batch in range(settings['event_driven_reference']['independent_batches']):
            value = _reference_batch(prefix, model, p, m, c, settings['event_driven_reference'],
                                     seed, batch, path_index, check_budget)
            check_budget()
            if execution_callback is not None:
                from .reference_execution import batch_completion_record
                execution_callback(batch_completion_record(path_index, _published_batch(value, batch, seed, path_index)))
            batches.append(value)
        reference = reference_summary(batches, settings['event_driven_reference']['confidence'], check_budget=check_budget)
        differences = _errors(coarse, reference, len(prefix.times))
        limits = settings['event_driven_reference']['thresholds']
        widths = reference['half_widths']
        reasons = []
        if any(halving[name] > settings['step_halving']['thresholds'][name] for name in ERRORS):
            reasons.append('STEP_HALVING_GATE_FAILED')
        if reference['minimum_ess_fraction'] < settings['event_driven_reference']['minimum_ess_fraction']:
            reasons.append('REFERENCE_ESS_GATE_FAILED')
        if any(widths[name] is None or widths[name] > settings['event_driven_reference']['maximum_mc_half_width_fraction']*limits[name] for name in ERRORS):
            reasons.append('REFERENCE_MC_PRECISION_FAILED')
        if any(widths[name] is None or differences[name]+widths[name] > limits[name] for name in ERRORS):
            reasons.append('REFERENCE_DISCREPANCY_GATE_FAILED')
        boundary = _known_mode_checks(prefix, model, m, c, inference, settings['known_mode_boundary'], path_index, check_budget)
        if any(not value['passed'] for value in boundary):
            reasons.append('KNOWN_MODE_BOUNDARY_GATE_FAILED')
        paths.append({'path_index': path_index,
            'prefix_hash': digest({'times': prefix.times.tolist(), 'values': prefix.values[path_index].tolist(),
                                   'H': prefix.H.tolist(), 'R': prefix.R.tolist()}),
            'state': 'INFERENCE_INAPPLICABLE' if reasons else 'NUMERICAL_CHECKS_PASSED', 'reasons': reasons,
            'step_halving_errors': halving, 'reference_errors': differences, 'reference_half_widths': widths,
            'coarse_summary': _compact(coarse), 'fine_summary': _compact(fine),
            'reference_summary': _compact(reference),
            'reference_batch_summaries': [_published_batch(value, index, seed, path_index)
                for index, value in enumerate(batches)],
            'reference_minimum_ess_fraction': reference['minimum_ess_fraction'],
            'reference_jump_count': reference['total_jump_count'], 'reference_batches': len(batches),
            'reference_summary_hash': digest(_compact(reference)),
            'known_mode_boundary': boundary})
    check_budget()
    return {'schema_version': 'pirc28-approximation-numerical-check-v1', 'spec_hash': spec.spec_hash,
        'model_hash': digest(model.to_dict()), 'observation_prefix_hash': digest(prefix.to_dict()),
        'initial_state_hash': digest({'probability': p.tolist(), 'means': m.tolist(), 'covariances': c.tolist()}),
        'inference_hash': digest(inference.to_dict()), 'reference_seed': seed, 'origin_index': origin_index,
        'state': 'INFERENCE_INAPPLICABLE' if any(value['reasons'] for value in paths) else 'NUMERICAL_CHECKS_PASSED',
        'paths': paths, 'resource_plan': plan, 'reference_continuous_state': 'path-conditional-affine-Kalman; no-state-particle-sampling',
        'reference_initial_conditioning': 'analytic-Gaussian-mode-mixture; iid-from-initial-posterior',
        'reference_resampling': 'multinomial-before-subsequent-intervals; no-redundant-initial-resampling',
        'reference_interval_method': MC_METHOD,
        'comparison': 'marginal-mode-probabilities-and-state-means; not-full-distribution-distance',
        'accuracy': 'finite-particle-reference; not-exact-joint-likelihood-or-posterior',
        'managed_evidence_bound': False, 'qualified_identifiability': False, 'real_data_authorized': False}
