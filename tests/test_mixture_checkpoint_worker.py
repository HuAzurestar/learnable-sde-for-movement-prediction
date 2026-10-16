"""Pacing never substitutes numerical steps or extends a running deadline."""
import pytest

from tests import mixture_checkpoint_worker as worker


@pytest.mark.parametrize("elapsed,wait", [(.005, .015), (.1, 0.)])
def test_pacing_forwards_exact_real_step_and_result(monkeypatch, elapsed, wait):
    readings = iter((10., 10.+elapsed))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(readings))
    delays, calls, result = [], [], object()
    monkeypatch.setattr(worker.time, "sleep", delays.append)
    def step(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    assert worker.paced_step(step, "components", grid=17) is result
    assert calls == [(("components",), {"grid": 17})]
    assert delays == [pytest.approx(wait)]


def test_step_failure_is_not_caught_or_replaced(monkeypatch):
    monkeypatch.setattr(worker.time, "sleep", lambda _: pytest.fail("failure was paced"))
    def failed():
        raise ValueError("actual numeric failure")
    with pytest.raises(ValueError, match="actual numeric failure"):
        worker.paced_step(failed)


def test_injection_keeps_native_wrapper_deadline_arguments_and_environment(monkeypatch):
    import application.research_supervisor as supervisor
    calls, result = [], object()
    def spawn(command, *args, **kwargs):
        calls.append((command, args, kwargs))
        return result
    monkeypatch.setattr(supervisor.subprocess, "Popen", spawn)
    worker.install(monkeypatch)
    command = ["python", "native-wrapper.py", "heartbeat", "20", "python", "-m",
        "experiments.pirc27.worker", "root", "store", "study", "cell", "output",
        "resume-state", "--admission-hash", "receipt"]
    options = {"env": {"actual-control": "unchanged"}, "stdin": "pipe"}
    assert supervisor.subprocess.Popen(command, **options) is result
    rewritten, args, kwargs = calls[0]
    assert rewritten[:5] == command[:5] and rewritten[5] == "-c"
    assert rewritten[7:] == command[7:] and kwargs == options and args == ()
    assert "mixture_checkpoint_worker import main" in rewritten[6]
    assert supervisor.subprocess.Popen(["python", "unrelated.py"]) is result
    assert calls[1] == (["python", "unrelated.py"], (), {})
