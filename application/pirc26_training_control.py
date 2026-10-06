"""Actual optimizer checkpoints over the existing owner/worker control channel.

This adapter is not a standalone research entry point. The enclosing worker
must already have passed shared source/input/admission/exposure/resource gates.
No budget balance is serialized and no checkpoint can extend an OS deadline.
"""

import time
import os

from estimation.phase_space import fit_o1
from estimation.phase_space_checkpoint import decode_state, encode_state
from estimation.phase_space_o2 import fit_o2
from infrastructure.research_control import WorkerControl
from infrastructure.research_store import ResearchError


def restore_training_state(state):
    if state is None:
        return None
    if (type(state) is not dict or set(state) != {"step", "data_position", "method_state", "rng_state"}
            or type(state["step"]) is not int or state["step"] < 1
            or state["data_position"] != state["step"]):
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "training position differs from the managed save")
    if type(state["method_state"]) is not dict or "state" not in state["method_state"]:
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "managed method state is absent")
    decoded = decode_state(state["method_state"]["state"])
    if type(decoded) is not dict or not {"step", "rng"} <= decoded.keys():
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "managed training state is incomplete")
    if state["step"] != decoded["step"] or state["rng_state"] != encode_state(decoded["rng"]):
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "training position/RNG differs from the managed save")
    return state["method_state"]


class ManagedTrainingControl:
    def __init__(self, total_steps):
        self.control = WorkerControl.from_environment()
        if self.control is None:
            raise ResearchError("CONTRACT_MISMATCH", "managed training requires the owner checkpoint channel")
        self.total_steps = total_steps
        self.started = time.perf_counter()
        self.first_step = None
        self.last_row = None

    def progress(self, row):
        if self.first_step is None:
            self.first_step = row["step"] - 1
        self.last_row = row

    def requested(self):
        return self.control.poll() is not None

    def save(self, state, row):
        decoded = decode_state(state["state"])
        completed = row["step"]
        elapsed = max(1e-9, time.perf_counter() - self.started)
        rate = (completed - self.first_step) / elapsed
        progress = {"completed_steps": completed, "total_steps": self.total_steps,
                    "throughput_per_second": rate, "eta_seconds": (self.total_steps - completed) / max(rate, 1e-9)}
        envelope = {"step": completed, "data_position": completed,
                    "method_state": state, "rng_state": encode_state(decoded["rng"])}
        self.control.save(envelope, progress)

    def arguments(self, state):
        return {"progress": self.progress, "checkpoint_requested": self.requested,
                "checkpoint_handler": self.save, "resume_state": restore_training_state(state)}


def managed_fit_o1(model, batches, plan, *, restored_state=None):
    control = ManagedTrainingControl(plan.max_steps)
    result = fit_o1(model, batches, plan, **control.arguments(restored_state))
    if result["status"] == "CHECKPOINTED":
        raise SystemExit(85)
    return result


def managed_fit_o2(model, examples, plan, o1_result, *, restored_state=None):
    control = ManagedTrainingControl(plan.max_steps)
    result = fit_o2(model, examples, plan, o1_result, **control.arguments(restored_state))
    if result["status"] == "CHECKPOINTED":
        raise SystemExit(85)
    return result


def exit_managed_worker(code):
    """Worker-only terminal exit, after closed output or acknowledged save.

    Windows ExitProcess/CRT _exit may deadlock in numerical DLL detach. Native
    self-termination avoids detach, but never replaces closing/flushing output
    or receiving the checkpoint ACK. The owner retains its process-tree check.
    https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-exitprocess
    """
    if type(code) is not int or code not in (0, 85) or WorkerControl.from_environment() is None:
        raise ResearchError("CONTRACT_MISMATCH", "terminal exit requires the managed worker channel and supported code")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes, kernel.GetCurrentProcess.restype = [], wintypes.HANDLE
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateProcess.restype = wintypes.BOOL
        if not kernel.TerminateProcess(kernel.GetCurrentProcess(), code):
            raise ctypes.WinError(ctypes.get_last_error())
        raise RuntimeError("native self-termination unexpectedly returned")
    os._exit(code)
