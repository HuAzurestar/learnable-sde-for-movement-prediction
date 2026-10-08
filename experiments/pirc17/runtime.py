"""CPU inference timing, explicitly separate from fitting and offline scoring."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import platform
import importlib.metadata
import sys
import threading
import time

import numpy as np

RUNTIME_VERSION = "pirc17-cpu-runtime-v2"


@dataclass
class StageTimes:
    """Nested spans are inclusive and must not be summed as disjoint stages."""
    milliseconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def span(self, name):
        if name not in {"checkpoint_and_input_io", "rollout", "terrain_io_and_query"}:
            raise ValueError("unknown inference timing stage")
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.milliseconds[name] = self.milliseconds.get(name, 0.) + (time.perf_counter_ns()-start)/1e6


def summarize_trials(trials, *, horizon_seconds):
    if not np.isfinite(horizon_seconds) or horizon_seconds <= 0 or not trials:
        raise ValueError("positive forecast horizon and nonempty trials required")
    latencies = [float(t["end_to_end_ms"]) for t in trials if t["status"] == "success"]
    failures = sum(t["status"] != "success" for t in trials)
    if any(not np.isfinite(v) or v<0 for v in latencies):
        raise ValueError("invalid latency")
    return {"trial_count":len(trials),"success_count":len(latencies),"failure_count":failures,
        "failure_rate":failures/len(trials),"latency_denominator":"successful_trials_only; all failures retained separately",
        "total_latency_p50_ms":float(np.quantile(latencies,.5)) if latencies else None,
        "total_latency_p95_ms":float(np.quantile(latencies,.95)) if latencies else None,
        "mean_ms_per_forecast_minute":float(np.mean(latencies)/(horizon_seconds/60)) if latencies else None}


def hardware_manifest():
    import psutil
    import torch
    result = {"platform":platform.platform(),"processor":platform.processor(),"python":platform.python_version(),
        "numpy":np.__version__,"torch":torch.__version__,"psutil":psutil.__version__,
        "logical_cpus":psutil.cpu_count(),"physical_cpus":psutil.cpu_count(logical=False),
        "ram_total_bytes":psutil.virtual_memory().total,"device":"cpu",
        "torch_threads":torch.get_num_threads(),"torch_interop_threads":torch.get_num_interop_threads(),
        "thread_environment":{name:os.environ.get(name) for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS")}}
    result["numba_threads"] = sys.modules["numba"].get_num_threads() if "numba" in sys.modules else "not_loaded"
    result["optional_map_packages"] = {}
    for package in ("duckdb","numba","rasterio","pyarrow"):
        try:
            result["optional_map_packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["optional_map_packages"][package] = None
    try:
        from threadpoolctl import threadpool_info
        result["loaded_threadpools"] = threadpool_info()
    except ImportError:
        result["loaded_threadpools"] = {"status":"unavailable","reason":"threadpoolctl not installed"}
    return result


def _measure(operation, *, memory_sample_seconds):
    import psutil
    process = psutil.Process()
    baseline = process.memory_info().rss
    observed = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(memory_sample_seconds):
            observed.append(process.memory_info().rss)
    sampler = threading.Thread(target=sample,daemon=True)
    sampler.start()
    stages = StageTimes()
    start = time.perf_counter_ns()
    error = None
    try:
        operation(stages)
    except Exception as exc:
        # Preserve the failed trial and its incurred time; never substitute zero.
        error = type(exc).__name__
    finally:
        elapsed = (time.perf_counter_ns()-start)/1e6
        observed.append(process.memory_info().rss)
        stop.set()
        sampler.join()
    memory = process.memory_info()
    return {"status":"success" if error is None else "failure","error_type":error,"end_to_end_ms":elapsed,
        "inclusive_stage_ms":stages.milliseconds,"baseline_rss_bytes":baseline,
        "sampled_peak_rss_bytes":max(observed),"sampled_peak_increment_bytes":max(observed)-baseline,
        "process_lifetime_peak_rss_bytes":getattr(memory,"peak_wset",None),
        "memory_sample_seconds":memory_sample_seconds,
        "gpu_memory":{"status":"not_applicable","reason":"CPU-only timed workload"}}


def benchmark_cpu(operation_factory, *, repetitions, horizon_seconds, particles, step_seconds,
                  batch_size, precision, history_step_seconds=None, warmup=1, memory_sample_seconds=.01):
    """Factory returns callable(StageTimes); neither may train or score.

    Cold means a new provider/checkpoint object per trial, NOT a flushed OS disk
    cache or fresh Python interpreter. Warm means reuse after explicit warm-up.
    GPU work is deliberately unsupported: this harness cannot pretend unsynced
    accelerator timings are valid CPU measurements.
    """
    for name,value in (("repetitions",repetitions),("particles",particles),("batch_size",batch_size),("warmup",warmup)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f"{name} must be a positive integer")
    if not all(np.isfinite(v) and v>0 for v in (horizon_seconds,step_seconds,memory_sample_seconds)):
        raise ValueError("positive finite horizon, step and memory interval required")
    if precision not in {"float32","float64"}:
        raise ValueError("explicit supported CPU precision required")
    if history_step_seconds is not None and (not np.isfinite(history_step_seconds) or history_step_seconds <= 0):
        raise ValueError("positive finite history cadence required")
    hardware = hardware_manifest()
    cold = []
    for _ in range(repetitions):
        held = []
        def cold_call(stages):
            operation = operation_factory()
            held.append(operation)
            operation(stages)
        trial = _measure(cold_call,memory_sample_seconds=memory_sample_seconds)
        for operation in held:
            close = getattr(operation,"close",None)
            if close is not None:
                close()  # Resource teardown is outside inference latency.
        cold.append(trial)
    warmup_failures = []
    try:
        warm_operation = operation_factory()
    except Exception as exc:
        warmup_failures.append(type(exc).__name__)
        warm = [{"status":"failure","phase":"warm_initialization","error_type":type(exc).__name__,
                 "end_to_end_ms":None,"not_executed":True} for _ in range(repetitions)]
    else:
        for _ in range(warmup):
            try:
                warm_operation(StageTimes())
            except Exception as exc:
                warmup_failures.append(type(exc).__name__)
        try:
            warm = [_measure(warm_operation,memory_sample_seconds=memory_sample_seconds) for _ in range(repetitions)]
        finally:
            close = getattr(warm_operation,"close",None)
            if close is not None:
                close()
    return {"schema_version":RUNTIME_VERSION,"hardware":hardware,
        "settings":{"horizon_seconds":horizon_seconds,"particles":particles,"step_seconds":step_seconds,
                    "history_step_seconds":history_step_seconds,
                    "batch_size":batch_size,"precision":precision,"warmup_calls":warmup},
        "cold_definition":"provider_recreated_OS_cache_uncontrolled_interpreter_warm",
        "warm_definition":"same_provider_after_explicit_warmup",
        "stage_accounting":"inclusive_nested_spans_do_not_sum; end_to_end_is_independently_timed",
        "memory_scope":"sampled_process_RSS_is_a_lower_bound; lifetime_peak_is_not_per_trial",
        "excluded_costs":["training","offline_scoring"],"warmup_failures":warmup_failures,
        "cold":{"trials":cold,"summary":summarize_trials(cold,horizon_seconds=horizon_seconds)},
        "warm":{"trials":warm,"summary":summarize_trials(warm,horizon_seconds=horizon_seconds)}}
