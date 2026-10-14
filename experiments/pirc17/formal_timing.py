"""One registered CPU trial, measured without fitting, scoring or exporting.

This is inference latency, not the outer controller's charged elapsed time.
Memory sampling is a lower bound; the process lifetime peak is not a trial
peak. Integrity and resource failures propagate instead of becoming timings.
"""
from copy import deepcopy
import math
import threading
import time

import psutil

from .protocol_core import sha256, unpack
from .runtime import StageTimes

VERSION = 'pirc17-formal-runtime-trial-v1'
SAMPLE_SECONDS = .01
STAGES = ('checkpoint_and_input_io', 'rollout', 'terrain_io_and_query')
SCOPE = {
    'inputs': 'Prepared causal inputs and fitted parameters resident; their original load/fit cost belongs to separately charged preparation, not zero-cost end-to-end deployment.',
    'cold': 'Fresh map provider for each terrain trial; method operations are stateless over resident parameters. OS cache and interpreter remain warm.',
    'warm': 'Same provider after the one registered warmup for this subject; no cross-subject warmed-provider substitution.',
    'includes': ['provider_initialization', 'prediction_setup', 'full_rollout', 'prediction_time_map_reads_and_queries'],
    'excludes': ['training', 'original_input_and_checkpoint_preparation', 'offline_scoring', 'artifact_export_and_validation', 'provider_teardown'],
    'stages': 'Inclusive nested spans; terrain_io_and_query is inside rollout. Do not sum nested stages.',
    'memory': 'Sampled process RSS is a lower bound; process lifetime peak is not per-trial peak. CPU only.',
}


class RuntimeTrial:
    """Bounded-memory sampler; no exception swallowing and no repeat loop."""
    def __init__(self):
        self.stages = StageTimes({name: 0. for name in STAGES})
        self.measurement = None

    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop, self.errors = threading.Event(), []
        def sample():
            while not self.stop.wait(SAMPLE_SECONDS):
                try:
                    self.peak = max(self.peak, self.process.memory_info().rss)
                except Exception as exc:
                    self.errors.append(type(exc).__name__)
                    break
        self.sampler = threading.Thread(target=sample, name='pirc17-runtime-rss', daemon=True)
        self.sampler.start()
        self.started = time.perf_counter_ns()
        return self

    def __exit__(self, kind, error, traceback):
        ended = time.perf_counter_ns()
        self.stop.set()
        self.sampler.join(timeout=1.)
        if self.sampler.is_alive() or self.errors:
            raise RuntimeError('registered runtime memory observation failed') from error
        memory = self.process.memory_info()
        peak = max(self.peak, memory.rss)
        self.measurement = dict(started_monotonic_ns=self.started, ended_monotonic_ns=ended,
            end_to_end_ms=(ended-self.started)/1e6, inclusive_stage_ms=deepcopy(self.stages.milliseconds),
            baseline_rss_bytes=self.baseline, sampled_peak_rss_bytes=peak, sampled_peak_increment_bytes=peak-self.baseline,
            process_lifetime_peak_rss_bytes=getattr(memory, 'peak_wset', None), memory_sample_seconds=SAMPLE_SECONDS,
            gpu_memory=dict(status='not_applicable', reason='CPU-only timed workload'))
        return False


def validate_timing(value, *, work, case, attempted, prediction_seconds):
    """Validate saved timing semantics, not independently remeasure the past.

The actual controller must also bind the worker's source/runtime and charge
the entire operation including IO, verification and instrumentation overhead.
"""
    required = work['kind'].startswith('runtime_') and bool(attempted)
    if not required:
        if value is not None:
            raise ValueError('unmeasured or nonruntime work cannot claim runtime evidence')
        return
    fields = {'schema_version', 'condition', 'hardware', 'provider_instance_id', 'provider_created',
              'warmup_result_sha256', 'warmup_status', 'actual_horizon_seconds', 'scope', 'trial'}
    if (not isinstance(value, dict) or set(value) != fields or value['schema_version'] != VERSION
            or value['condition'] != work['kind'] or value['scope'] != SCOPE
            or value['actual_horizon_seconds'] != float(case.score_seconds[-1])
            or type(value['provider_created']) is not bool
            or value['provider_created'] != (work['matrix'] == 'terrain' and work['kind'] in {'runtime_cold', 'runtime_warmup'})):
        raise ValueError('registered runtime condition/scope differs')
    sha256(value['provider_instance_id'])
    hardware = unpack(value['hardware'])
    hardware_fields = {'platform', 'processor', 'python', 'numpy', 'torch', 'psutil', 'logical_cpus', 'physical_cpus',
        'ram_total_bytes', 'device', 'torch_threads', 'torch_interop_threads', 'thread_environment', 'numba_threads',
        'optional_map_packages', 'loaded_threadpools'}
    if set(hardware) != hardware_fields or hardware.get('device') != 'cpu':
        raise ValueError('actual CPU hardware manifest required')
    if work['kind'] == 'runtime_warm':
        sha256(value['warmup_result_sha256'])
        if value['warmup_status'] not in {'success', 'failed'}:
            raise ValueError('actual registered warmup disposition required')
    elif value['warmup_result_sha256'] is not None or value['warmup_status'] is not None:
        raise ValueError('cold/warmup trial cannot claim a preceding warmup')
    trial = value['trial']
    keys = {'started_monotonic_ns', 'ended_monotonic_ns', 'end_to_end_ms', 'inclusive_stage_ms', 'baseline_rss_bytes',
            'sampled_peak_rss_bytes', 'sampled_peak_increment_bytes', 'process_lifetime_peak_rss_bytes', 'memory_sample_seconds', 'gpu_memory'}
    if not isinstance(trial, dict) or set(trial) != keys:
        raise ValueError('actual registered timing/memory fields required')
    start, end = trial['started_monotonic_ns'], trial['ended_monotonic_ns']
    if (type(start) is not int or type(end) is not int or not 0 <= start <= end
            or type(trial['end_to_end_ms']) not in (int, float) or not math.isfinite(trial['end_to_end_ms'])
            or trial['end_to_end_ms'] != (end-start)/1e6 or prediction_seconds != trial['end_to_end_ms']/1000
            or trial['memory_sample_seconds'] != SAMPLE_SECONDS or trial['gpu_memory'] != dict(status='not_applicable', reason='CPU-only timed workload')):
        raise ValueError('timing clock/units/sampling scope differs')
    stages = trial['inclusive_stage_ms']
    if (not isinstance(stages, dict) or set(stages) != set(STAGES)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in stages.values())
            or stages['terrain_io_and_query'] > stages['rollout']
            or stages['rollout']+stages['checkpoint_and_input_io'] > trial['end_to_end_ms']):
        raise ValueError('runtime nested stage accounting differs')
    base, peak, increment = (trial[k] for k in ('baseline_rss_bytes', 'sampled_peak_rss_bytes', 'sampled_peak_increment_bytes'))
    lifetime = trial['process_lifetime_peak_rss_bytes']
    if (any(type(v) is not int for v in (base, peak, increment)) or not 0 < base <= peak or increment != peak-base
            or (lifetime is not None and (type(lifetime) is not int or lifetime < peak))):
        raise ValueError('runtime measured RSS identities differ')
