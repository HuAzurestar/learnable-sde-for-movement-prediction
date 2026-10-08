import numpy as np
import pytest

from experiments.pirc17.runtime import StageTimes, benchmark_cpu, summarize_trials


def test_timing_arithmetic_keeps_failures_and_uses_forecast_minutes():
    rows = [{"status":"success","end_to_end_ms":10},{"status":"success","end_to_end_ms":30},
            {"status":"failure","end_to_end_ms":100}]
    result = summarize_trials(rows,horizon_seconds=120)
    assert result["total_latency_p50_ms"] == 20
    assert result["total_latency_p95_ms"] == 29
    assert result["mean_ms_per_forecast_minute"] == 10
    assert result["failure_rate"] == pytest.approx(1/3)
    all_failed = summarize_trials(rows[-1:],horizon_seconds=120)
    assert all_failed["mean_ms_per_forecast_minute"] is None and all_failed["failure_rate"] == 1


def test_cold_and_warm_caches_are_explicit_and_memory_is_not_fake_peak():
    constructions = []
    calls = []
    closed = []
    def factory():
        identifier = len(constructions)
        constructions.append(identifier)
        def execute(stages):
            calls.append(identifier)
            with stages.span("rollout"):
                with stages.span("terrain_io_and_query"):
                    assert np.ones(100).sum() == 100
        execute.close = lambda:closed.append(identifier)
        return execute
    result = benchmark_cpu(factory,repetitions=2,horizon_seconds=60,particles=4,step_seconds=1,
                           history_step_seconds=5.,batch_size=1,precision="float64")
    assert result["settings"]["history_step_seconds"] == 5.
    assert constructions == [0,1,2]
    assert calls == [0,1,2,2,2]
    assert closed == [0,1,2]
    for mode in ("cold","warm"):
        assert result[mode]["summary"]["failure_count"] == 0
        for trial in result[mode]["trials"]:
            assert trial["sampled_peak_rss_bytes"] >= trial["baseline_rss_bytes"] > 0
            assert trial["end_to_end_ms"] >= trial["inclusive_stage_ms"]["rollout"]
    assert "OS_cache_uncontrolled" in result["cold_definition"]
    assert "lower_bound" in result["memory_scope"]
    assert "numba_threads" in result["hardware"]
    assert "NUMBA_NUM_THREADS" in result["hardware"]["thread_environment"]


def test_failed_calls_are_preserved():
    def factory():
        def execute(stages):
            raise RuntimeError("fixture failure")
        return execute
    result = benchmark_cpu(factory,repetitions=1,horizon_seconds=60,particles=2,step_seconds=1,
                           batch_size=1,precision="float64")
    assert result["cold"]["summary"]["failure_rate"] == 1
    assert result["warm"]["summary"]["failure_rate"] == 1
    assert result["warmup_failures"] == ["RuntimeError"]


def test_scoring_cannot_be_named_as_an_inference_stage():
    with pytest.raises(ValueError):
        with StageTimes().span("offline_score"):
            pass


def test_provider_initialization_failure_does_not_erase_cold_trials():
    def factory():
        raise ValueError("fixture missing checkpoint")
    result = benchmark_cpu(factory,repetitions=2,horizon_seconds=60,particles=2,step_seconds=1,
                           batch_size=1,precision="float64")
    assert result["cold"]["summary"]["failure_count"] == 2
    assert result["warm"]["summary"]["failure_count"] == 2
    assert result["warm"]["trials"][0]["end_to_end_ms"] is None
