"""Bounded additive-affine endpoint kernels for the registered research adapter.

Ordinary callers may use these kernels for unit qualification. Research execution
must use the shared supervisor: this module has no process, ledger or store API.
"""

from __future__ import annotations

import math

import numpy as np
from domain.errors import DataValidationError, NumericalError
from domain.frozen_dynamics import FrozenDynamicsPackage, content_hash
from domain.nonlinear_dynamics import FrozenNonlinearPackage
from domain.propagation import ErrorComponent, FunctionalResult, NumericalErrorBudget
from .affine_oracle import exact_moments, endpoint_halfspace_probability, _covariance


def _inputs(package, request):
    if not isinstance(package, (FrozenDynamicsPackage, FrozenNonlinearPackage)):
        raise DataValidationError("a supported immutable dynamics package is required")
    request.validate()
    package.validate()
    if package.package_hash != request.model_package_hash:
        raise DataValidationError("request model package differs")
    covariance = _covariance(request.initial_covariance, "initial covariance").numpy()
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values < 0):
        raise NumericalError("initial covariance square root would require PSD projection")
    parameters = package.manifest()["parameters"]
    return (np.array(parameters["A"], dtype=float), np.array(parameters["b"], dtype=float),
            np.array(parameters["L"], dtype=float), np.array(request.initial_mean, dtype=float),
            vectors * np.sqrt(values)[None, :])


def _scheme(A, b, L, dt, solver):
    if solver == "euler":
        return np.eye(4) + dt*A, dt*b, L
    if solver == "additive-heun":
        return np.eye(4) + dt*A + 0.5*dt**2*(A@A), dt*b + 0.5*dt**2*(A@b), (np.eye(4)+0.5*dt*A)@L
    raise DataValidationError("unsupported additive-affine solver")


def _rng(request, sample_id, level, phase):
    # Per-sample independent streams make chunk size and resume position neutral.
    root_hash = content_hash({"coupling_id": request.coupling_id, "scheme": "per-sample-seedsequence-v1"})
    root_words = [int(root_hash[start:start+8], 16) for start in range(0, 32, 8)]
    return np.random.default_rng(np.random.SeedSequence([request.seed, sample_id, level, phase, *root_words]))


def _functional(endpoint, request):
    if request.functional == "endpoint-x":
        return endpoint[:, 0]
    normal = np.array(request.normal)
    scale = float(np.max(np.abs(normal)))
    value, threshold = endpoint @ (normal/scale), request.threshold/scale
    return (value >= threshold if request.closed else value > threshold).astype(float)


def _expectation(mean, covariance, request):
    if request.functional == "endpoint-x":
        return float(mean[0])
    return endpoint_halfspace_probability(mean, covariance, request.normal, request.threshold, closed=request.closed)


def discrete_moments(package, request, *, solver="euler", steps=None):
    if not isinstance(package, FrozenDynamicsPackage):
        raise DataValidationError("exact discrete Gaussian moments require affine dynamics")
    A, b, L, mean, _ = _inputs(package, request)
    steps = request.steps if steps is None else steps
    if type(steps) is not int or not 1 <= steps <= 8192:
        raise DataValidationError("discrete step count exceeds registered kernel limit")
    dt = request.horizons[0] / steps
    F, offset, noise = _scheme(A, b, L, dt, solver)
    covariance = np.array(request.initial_covariance, dtype=float)
    for _ in range(steps):
        mean, covariance = F@mean+offset, F@covariance@F.T + dt*(noise@noise.T)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise NumericalError("nonfinite discrete affine moments")
    return mean, _covariance(covariance, "discrete covariance").numpy()


def _error_budget(package, request, solver, steps, standard_error, *, exact=False):
    if isinstance(package, FrozenNonlinearPackage):
        from .nonlinear_propagation import nonlinear_error_budget
        return nonlinear_error_budget(request, standard_error)
    units = "m" if request.functional == "endpoint-x" else "1"
    exact_mean, exact_covariance = exact_moments(package, request.initial_mean, request.initial_covariance, request.horizons[0])
    if exact:
        bias = 0.0
    else:
        mean, covariance = discrete_moments(package, request, solver=solver, steps=steps)
        bias = _expectation(mean, covariance, request) - _expectation(exact_mean, exact_covariance, request)
    return NumericalErrorBudget(
        ErrorComponent(None, units, "float64 analytic reference; no certified roundoff bound", "NOT_IDENTIFIABLE"),
        ErrorComponent(bias, units, "discrete affine Gaussian expectation minus continuous reference", "IDENTIFIED"),
        ErrorComponent(0.0, units, "constant affine Gaussian model; no nonlinear closure", "IDENTIFIED"),
        ErrorComponent(standard_error, units, "estimator standard error", "ESTIMATED" if standard_error is not None else "NOT_IDENTIFIABLE"),
        ErrorComponent(None, units, "no observed real dynamics in synthetic recipe", "NOT_IDENTIFIABLE"))


def analytic_estimate(package, request, *, discrete=False, solver="euler"):
    _inputs(package, request)
    mean, covariance = (discrete_moments(package, request, solver=solver) if discrete else
        exact_moments(package, request.initial_mean, request.initial_covariance, request.horizons[0]))
    return FunctionalResult(request.request_hash, "gaussian-discrete-v1" if discrete else "affine-exact-v1", "analytic",
        _expectation(mean, covariance, request), 0.0, None, "analytic-no-sampling", 0,
        _error_budget(package, request, solver, request.steps, 0.0, exact=not discrete), "SUCCEEDED",
        (("assumption", "constant affine Gaussian dynamics and Gaussian initial distribution"),))


def endpoint_chunks(package, request, *, solver="euler", level=0, paired=False,
                    proposal=(0.0, 0.0), phase=0, start_sample=0, sample_count=None):
    """Yield sample ranges, fine/coarse endpoints and unnormalized log weights.

    Coarse increments are sums of the exact same adjacent fine increments.
    Proposal drift is L @ proposal, hence remains in the declared noise support.
    """
    A, b, L, initial, root = _inputs(package, request)
    nonlinear = isinstance(package, FrozenNonlinearPackage)
    if nonlinear:
        from .nonlinear_propagation import nonlinear_drift
        parameters = package.manifest()["parameters"]
        def advance(state, increments, step_size):
            drift = nonlinear_drift(A, b+L@u, parameters["amplitude"], parameters["length_scale"], state)
            noise = np.einsum("bi,ji->bj", increments, L, optimize=False)
            predictor = state + step_size*drift + noise
            if solver == "euler":
                return predictor
            corrected = nonlinear_drift(A, b+L@u, parameters["amplitude"], parameters["length_scale"], predictor)
            return state + 0.5*step_size*(drift+corrected) + noise
    if type(level) is not int or not 0 <= level <= 8 or paired and level == 0:
        raise DataValidationError("invalid coupled level")
    steps = request.steps * 2**level
    if steps > 8192:
        raise DataValidationError("fine step count exceeds registered kernel limit")
    count = request.samples if sample_count is None else sample_count
    if (type(count) is not int or not 1 <= count <= 1_000_000 or type(start_sample) is not int
            or start_sample < 0 or start_sample + count > 1_000_000 or type(phase) is not int or phase < 0):
        raise DataValidationError("invalid sample range or independent phase stream")
    if type(proposal) is not tuple or len(proposal) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in proposal):
        raise DataValidationError("proposal must be a finite two-dimensional noise shift")
    u = np.array(proposal, dtype=float)
    dt = request.horizons[0] / steps
    F, offset, noise = _scheme(A, b + L@u, L, dt, solver)
    if paired:
        coarse_F, coarse_offset, coarse_noise = _scheme(A, b + L@u, L, 2*dt, solver)
    for start in range(start_sample, start_sample+count, request.chunk_size):
        stop = min(start+request.chunk_size, start_sample+count)
        streams = [_rng(request, sample, level, phase) for sample in range(start, stop)]
        # O(chunk * state/noise), without storing steps-by-path Brownian arrays.
        state = np.stack([initial + root @ stream.standard_normal(4) for stream in streams])
        coarse = state.copy() if paired else None
        log_weight = np.zeros(stop-start)
        pending = np.zeros((stop-start, 2)) if paired else None
        for step in range(steps):
            increments = np.stack([stream.standard_normal(2) for stream in streams]) * math.sqrt(dt)
            # Fixed reduction order, independent of the BLAS batch-size path.
            state = (advance(state, increments, dt) if nonlinear else
                np.einsum("bi,ji->bj", state, F, optimize=False) + offset + np.einsum("bi,ji->bj", increments, noise, optimize=False))
            log_weight -= np.einsum("bi,i->b", increments, u, optimize=False) + 0.5*float(u@u)*dt
            if paired:
                pending += increments
                if step % 2:
                    coarse = (advance(coarse, pending, 2*dt) if nonlinear else
                        np.einsum("bi,ji->bj", coarse, coarse_F, optimize=False) + coarse_offset + np.einsum("bi,ji->bj", pending, coarse_noise, optimize=False))
                    pending.fill(0)
            if not np.isfinite(state).all() or not np.isfinite(log_weight).all() or paired and not np.isfinite(coarse).all():
                raise NumericalError("nonfinite path, coupled path or importance weight")
        yield start, stop, state, coarse, log_weight


def _merge_moments(n, mean, m2, values):
    count = len(values)
    local_mean = float(np.mean(values))
    local_m2 = float(np.sum((values-local_mean)**2))
    delta = local_mean - mean
    total = n + count
    return total, mean + delta*count/total, m2 + local_m2 + delta*delta*n*count/total


def monte_carlo(package, request, *, solver="euler", resume_state=None, checkpoint=None):
    from .propagation_recovery import ChunkState
    _inputs(package, request)
    state = ChunkState(request, solver, (request.samples,), (request.steps,), restored=resume_state)
    stats = state.statistics[0]
    for _, stop, endpoint, _, _ in endpoint_chunks(package, request, solver=solver,
            start_sample=state.position, sample_count=request.samples-state.position) if state.level == 0 else ():
        values = _functional(endpoint, request)
        stats["n"], stats["mean"], stats["m2"] = _merge_moments(stats["n"], stats["mean"], stats["m2"], values)
        stats["hits"] += int(np.count_nonzero(values))
        state.completed(0, stop, checkpoint)
    n, mean, m2, hits = (stats[key] for key in ("n", "mean", "m2", "hits"))
    se = math.sqrt(m2/(n-1)/n)
    interval, kind = (mean-1.959963984540054*se, mean+1.959963984540054*se), "normal-approximation-95"
    # Explicit exact one-sided upper bound for zero independent Bernoulli hits.
    if request.functional == "endpoint-halfspace" and hits == 0:
        se, interval, kind = None, (0.0, -math.expm1(math.log(0.05)/n)), "exact-binomial-one-sided-95"
    return FunctionalResult(request.request_hash, solver+"-path-mc-v1", "functional_estimate", mean, se, interval, kind, n,
        _error_budget(package, request, solver, request.steps, se), "SUCCEEDED",
        (("brownian_scheme", "per-sample-seedsequence-v1"), ("coupling_id", request.coupling_id), ("hits", hits)))


def importance_sampling(package, request, *, proposal, resume_state=None, checkpoint=None):
    if request.functional != "endpoint-halfspace":
        raise DataValidationError("importance estimator supports only the registered endpoint event")
    from .propagation_recovery import ChunkState
    _inputs(package, request)
    state = ChunkState(request, "importance", (request.samples,), (request.steps,), proposal=proposal, restored=resume_state)
    stats = state.statistics[0]
    def add_log(key, values):
        previous = -math.inf if stats[key] is None else stats[key]
        stats[key] = float(np.logaddexp(previous, np.logaddexp.reduce(values)))
    for start, stop, endpoint, _, log_weights in endpoint_chunks(package, request, proposal=proposal,
            start_sample=state.position, sample_count=request.samples-state.position) if state.level == 0 else ():
        events = _functional(endpoint, request).astype(bool)
        stats["n"] += stop-start
        stats["hits"] += int(events.sum())
        add_log("log_w", log_weights)
        add_log("log_w2", 2*log_weights)
        stats["max_log_w"] = max(-math.inf if stats["max_log_w"] is None else stats["max_log_w"], float(log_weights.max()))
        if events.any():
            add_log("log_event", log_weights[events])
            add_log("log_event2", 2*log_weights[events])
        state.completed(0, stop, checkpoint)
    n, hits = stats["n"], stats["hits"]
    log_sum_weight, log_sum_square, max_log_weight = (stats[key] for key in ("log_w", "log_w2", "max_log_w"))
    log_sum_event, log_sum_event_square = stats["log_event"], stats["log_event2"]
    try:
        estimate = math.exp(log_sum_event-math.log(n)) if hits else 0.0
        second = math.exp(log_sum_event_square-math.log(n)) if hits else 0.0
        variance = max(0.0, (second-estimate**2)*n/(n-1))
    except OverflowError as exc:
        raise NumericalError("importance estimator weight moments overflow") from exc
    if not math.isfinite(estimate) or not math.isfinite(variance):
        raise NumericalError("nonfinite importance estimator moments")
    ess = math.exp(2*log_sum_weight-log_sum_square)
    se = math.sqrt(variance/n) if hits else None
    interval = (estimate-1.959963984540054*se, estimate+1.959963984540054*se) if hits else None
    status = "INSUFFICIENT_EVENTS" if not hits else "LOW_ESS" if ess < 2 else "SUCCEEDED"
    return FunctionalResult(request.request_hash, "velocity-drift-is-unnormalized-v1", "weighted", estimate, se, interval,
        "iid-weighted-normal-approximation-95" if hits else "unavailable-no-weighted-hits", n,
        _error_budget(package, request, "euler", request.steps, se), status,
        (("proposal", proposal), ("proposal_hash", content_hash(proposal)),
         ("ess", ess), ("log_sum_weights", log_sum_weight), ("log_sum_squared_weights", log_sum_square),
         ("max_log_weight", max_log_weight), ("hits", hits), ("self_normalized", False)))


def mlmc_estimate(package, request, *, level_samples, phase=1, resume_state=None, checkpoint=None, pilot=False):
    """A fixed registered allocation; pilot proposals do not start extra work."""
    if (type(level_samples) is not tuple or not 1 <= len(level_samples) <= 9
            or any(type(n) is not int or not 2 <= n <= 1_000_000 for n in level_samples)
            or sum(level_samples) > 1_000_000):
        raise DataValidationError("invalid bounded MLMC allocation")
    request.validate()
    if type(pilot) is not bool or pilot and (type(phase) is not int or phase != 2):
        raise DataValidationError("MLMC pilot requires its independent fixed stream phase")
    if request.steps*2**(len(level_samples)-1) > 8192:
        raise DataValidationError("MLMC finest grid exceeds registered step limit")
    from .propagation_recovery import ChunkState
    _inputs(package, request)
    costs = tuple(request.steps*2**level + (request.steps*2**(level-1) if level else 0) for level in range(len(level_samples)))
    state = ChunkState(request, "mlmc", level_samples, costs, phase=phase, restored=resume_state, measure_cost=pilot)
    for level in range(state.level, len(level_samples)):
        count, stats = level_samples[level], state.statistics[level]
        offset = state.position if level == state.level else 0
        chunks = iter(endpoint_chunks(package, request, level=level, paired=level>0,
                                     phase=phase, start_sample=offset, sample_count=count-offset))
        while True:
            if pilot:
                import time
                started_ns = time.perf_counter_ns()
            try:
                _, stop, fine, coarse, _ = next(chunks)
            except StopIteration:
                break
            values = _functional(fine, request)
            if level:
                values = values - _functional(coarse, request)
            stats["n"], stats["mean"], stats["m2"] = _merge_moments(stats["n"], stats["mean"], stats["m2"], values)
            stats["hits"] += int(np.count_nonzero(values))
            if pilot:
                stats["compute_ns"] += max(1, time.perf_counter_ns()-started_ns)
            state.completed(level, stop, checkpoint)
    means = [stats["mean"] for stats in state.statistics]
    variances = [stats["m2"]/(stats["n"]-1) for stats in state.statistics]
    estimate = math.fsum(means)
    se = math.sqrt(math.fsum(v/n for v, n in zip(variances, level_samples)))
    # Never clip a noisy signed MLMC probability estimator to [0,1].
    status = "UNRESOLVED_SAMPLING" if request.functional == "endpoint-halfspace" and se == 0 else "SUCCEEDED"
    diagnostics = (("level_samples", level_samples), ("level_means", tuple(means)), ("level_variances", tuple(variances)),
         ("coupling", "coarse-increment=sum(two-fine-increments)"), ("bias_bound", "unknown; affine signed bias in error budget"))
    if pilot:
        diagnostics += (("pilot_phase", 2), ("production_phase", 1),
            ("level_compute_ns", tuple(stats["compute_ns"] for stats in state.statistics)),
            ("level_work_per_sample", costs), ("cost_scope", "compute-only-excludes-checkpoint-ACK-not-budget-charge"))
    return FunctionalResult(request.request_hash, "coupled-euler-mlmc-pilot-v1" if pilot else "coupled-euler-mlmc-v1", "functional_estimate", estimate,
        None if status != "SUCCEEDED" else se, (estimate-1.959963984540054*se, estimate+1.959963984540054*se) if status == "SUCCEEDED" else None,
        "independent-level-normal-approximation-95" if status == "SUCCEEDED" else "unavailable-zero-observed-level-variance",
        sum(level_samples), _error_budget(package, request, "euler", request.steps*2**(len(level_samples)-1),
                                        se if status == "SUCCEEDED" else None), "PILOT_ONLY" if pilot and status == "SUCCEEDED" else status,
        diagnostics)


def allocate_mlmc(variances, costs, sampling_tolerance, *, maximum_samples=1_000_000):
    """Propose N_l ∝ sqrt(V_l/C_l) from an independently registered pilot."""
    if (type(variances) is not tuple or type(costs) is not tuple or len(variances) != len(costs)
            or not 1 <= len(costs) <= 9 or type(sampling_tolerance) not in (int, float)
            or not math.isfinite(sampling_tolerance) or sampling_tolerance <= 0
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in variances)
            or any(type(c) not in (int, float) or not math.isfinite(c) or c <= 0 for c in costs)
            or type(maximum_samples) is not int or not 2 <= maximum_samples <= 1_000_000):
        raise DataValidationError("invalid MLMC pilot variance/cost or tolerance")
    try:
        normalizer = math.fsum(math.sqrt(v)*math.sqrt(c) for v, c in zip(variances, costs)) / sampling_tolerance**2
        allocation = tuple(max(2, math.ceil(normalizer*math.sqrt(v/c))) for v, c in zip(variances, costs))
    except (ArithmeticError, ValueError) as exc:
        raise DataValidationError("pilot allocation is outside finite sample limits") from exc
    if sum(allocation) > maximum_samples:
        raise DataValidationError("pilot allocation exceeds the frozen sample cap; no automatic expansion")
    return allocation
