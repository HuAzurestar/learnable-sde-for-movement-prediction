"""Real early checkpoint stop is not billed for later coordinator cleanup.

Use the original 300-step worker and 3s/6s deadlines. Only actual owner-side
post-stop journal/close work is delayed; no clock, worker result or native
stop observation is substituted. RNG continuation must survive reopening.
"""

import json
import time

import pytest

from application.research_budget import BudgetLedger, BudgetSpec
from application.research_recovery import SharedRecovery
from experiments.pirc25.runner import SharedRunner
from infrastructure.process_tree import ProcessTree
from infrastructure.research_control import CheckpointExchange
from infrastructure.research_store import ResearchStore, digest
from tests.native_checkpoint_trace import NativeCheckpointTrace
from tests.test_research_live_checkpoint import prepared


@pytest.mark.parametrize("level", ["exact", "chunk"])
@pytest.mark.parametrize("delay_at", ["stop-journal", "native-close"])
def test_early_real_checkpoint_stop_excludes_later_owner_work_and_resumes(
        tmp_path, monkeypatch, level, delay_at):
    continuous, baseline, baseline_registry, _, _ = prepared(
        tmp_path / "continuous", level, "continuous")
    expected = SharedRunner(continuous, baseline_registry).run_cell(
        "continuous", digest(baseline["cells"][0]), budget=BudgetSpec(6))
    assert expected["state"] == "SUCCEEDED", expected
    expected_output = json.loads(
        (continuous.path / "artifacts" / expected["artifact_id"]).read_bytes())

    store, value, registry, adapters, grant = prepared(
        tmp_path / "resumed", level, "resumed")
    original_wait, original_close = ProcessTree.wait_stopped, ProcessTree.close
    original_append, original_ack = store.append, CheckpointExchange.acknowledge
    original_request, original_response = CheckpointExchange.request, CheckpointExchange.response
    stopped, acknowledgements, delayed = [], [], []
    trace = NativeCheckpointTrace()
    checkpoint_deadline = [None]

    def request(exchange):
        checkpoint_deadline[0] = exchange.deadline
        return trace.call("request", lambda: original_request(exchange), deadline=exchange.deadline)

    def response(exchange):
        return trace.call("response", lambda: original_response(exchange),
                          deadline=exchange.deadline, frame=True, omit_empty=True)

    def wait_stopped(tree, *args, **kwargs):
        result = trace.call("native-stop", lambda: original_wait(tree, *args, **kwargs),
                            deadline=checkpoint_deadline[0], native_code=lambda: tree.process.poll())
        if result:
            # Keep the actual native observation and actual wrapper exit code.
            stopped.append((time.monotonic(), tree.process.poll()))
        return result

    def acknowledge(exchange, artifact_id):
        result = trace.call("ack", lambda: original_ack(exchange, artifact_id), deadline=exchange.deadline)
        acknowledgements.append((time.monotonic(), exchange.deadline, artifact_id))
        return result

    def delay():
        assert stopped and stopped[0][1] == 85, "actual checkpoint worker did not exit"
        assert acknowledgements and stopped[0][0] < acknowledgements[0][1], (
            "actual native stop did not precede the unchanged deadline")
        delayed.append(time.monotonic())
        time.sleep(0.8)  # Coordinator only; the whole native tree is already stopped.

    def append(kind, *args, **kwargs):
        operation = {"CHECKPOINT_REQUESTED": "journal-request", "CHECKPOINT_SAVED": "journal-save",
                     "WORKER_TREE_STOPPED": "journal-stop"}.get(kind)
        result = (trace.call(operation, lambda: original_append(kind, *args, **kwargs),
                             deadline=checkpoint_deadline[0])
                  if operation else original_append(kind, *args, **kwargs))
        if delay_at == "stop-journal" and kind == "WORKER_TREE_STOPPED" and not delayed:
            delay()
        return result

    def close(tree):
        result = trace.call("close", lambda: original_close(tree),
                            deadline=checkpoint_deadline[0], native_code=lambda: tree.process.poll())
        if delay_at == "native-close" and not delayed:
            delay()
        return result

    monkeypatch.setattr(ProcessTree, "wait_stopped", wait_stopped)
    monkeypatch.setattr(ProcessTree, "close", close)
    monkeypatch.setattr(CheckpointExchange, "acknowledge", acknowledge)
    monkeypatch.setattr(CheckpointExchange, "request", request)
    monkeypatch.setattr(CheckpointExchange, "response", response)
    monkeypatch.setattr(store, "append", append)
    try:
        interrupted = SharedRunner(store, registry, recovery_registry=adapters).run_cell(
            "resumed", digest(value["cells"][0]), budget=BudgetSpec(3))
    except Exception:
        # The original owner's containment/cleanup path has unwound.
        # Output only after failure, never add critical-path I/O or claim exit.
        print("Native checkpoint failure timeline:", json.dumps(trace.report(), sort_keys=True))
        raise
    assert len(delayed) == 1, ("actual post-stop boundary was not reached", trace.report())
    assert len(acknowledgements) == 1 and acknowledgements[0][0] < acknowledgements[0][1], trace.report()
    assert time.monotonic() >= acknowledgements[0][1], ("counterexample did not cross the fuse", trace.report())
    assert interrupted["state"] == "FAILED" and interrupted["exit_code"] != 0, interrupted
    assert store.attempts()[interrupted["attempt_id"]]["error_code"] == "CHECKPOINT_SAVED"
    events = store.events()
    stop = [event["payload"] for event in events if event["event_kind"] == "WORKER_TREE_STOPPED"]
    saved = [event["payload"] for event in events if event["event_kind"] == "CHECKPOINT_SAVED"]
    settlements = [event["payload"] for event in events if event["event_kind"] == "SETTLE"]
    assert len(stop) == len(saved) == len(settlements) == 1
    assert 2400 <= saved[0]["elapsed_ms"] <= stop[0]["observed_elapsed_ms"] < 3000
    assert acknowledgements[0][2] == saved[0]["artifact_id"] == interrupted["checkpoint"]["artifact_id"]
    assert interrupted["elapsed_ms"] == stop[0]["observed_elapsed_ms"], (
        "slot cost includes work after the first real whole-tree stop")
    assert settlements[0]["monotonic_elapsed_ms"] == interrupted["elapsed_ms"]
    assert settlements[0]["charged_ms"] == interrupted["elapsed_ms"]
    assert not any(event["event_kind"] == "ARM_CLOSED" for event in events)
    previous_cost = BudgetLedger(store).balance("affine")["committed_ms"]
    assert previous_cost == interrupted["elapsed_ms"] > 0

    reopened = ResearchStore(tmp_path / "resumed", "live-checkpoint")
    resumed = SharedRecovery(reopened, registry, adapters).resume(
        interrupted["attempt_id"], saved[0]["artifact_id"], authorization=grant, budget=BudgetSpec(6))
    assert resumed["state"] == "SUCCEEDED", resumed
    actual = json.loads((reopened.path / "artifacts" / resumed["artifact_id"]).read_bytes())
    assert actual["forecast"] == expected_output["forecast"]
    assert actual["metrics"] == expected_output["metrics"]
    assert reopened.attempts()[resumed["attempt_id"]]["parent_attempt_id"] == interrupted["attempt_id"]
    assert BudgetLedger(reopened).balance("affine")["committed_ms"] > previous_cost
