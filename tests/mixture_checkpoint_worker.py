"""Test-only pacing of real mixture steps; not a research execution adapter.

The small fixed-work kernel may finish before a useful physical save window on
fast CI hosts. Pace its genuine step boundaries, without substituting results,
checkpoint frames, clocks, admission, fsync, ACK or native process containment.
"""
import time


def paced_step(step, *args, **kwargs):
    started = time.monotonic()
    result = step(*args, **kwargs)
    time.sleep(max(0., .02 - (time.monotonic() - started)))
    return result


def main():
    import inference.mixture_propagation as mixture
    from experiments.pirc27.worker import main as worker
    original = mixture.mixture_step
    mixture.mixture_step = lambda *args, **kwargs: paced_step(original, *args, **kwargs)
    worker()


def install(monkeypatch):
    """Inject only the disposable test worker, keeping its actual native owner."""
    import application.research_supervisor as supervisor
    original = supervisor.subprocess.Popen
    def popen(command, *args, **kwargs):
        command = list(command)
        marker = "experiments.pirc27.worker"
        if marker in command:
            index = command.index(marker)
            assert command[index-1] == "-m"
            command[index-1:index+1] = ["-c",
                "import sys; from tests.mixture_checkpoint_worker import main; "
                "sys.argv[0]='experiments.pirc27.worker'; main()"]
        return original(command, *args, **kwargs)
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
