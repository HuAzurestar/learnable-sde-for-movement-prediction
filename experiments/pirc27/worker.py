"""Internal shared-runner worker; never creates a store or grants access."""

import sys
import time
from pathlib import Path

from application.propagation_execution import execute_propagation
from infrastructure.research_store import ResearchError, ResearchStore, atomic_write, digest, encode
from infrastructure.research_control import WorkerControl, read_frame, write_frame


def main():
    arguments = list(sys.argv)
    reference = None
    if len(arguments) >= 3 and arguments[-2] == "--admission-hash":
        reference = arguments[-1]
        arguments = arguments[:-2]
    if len(arguments) not in {6, 7}:
        raise ResearchError("CONTRACT_MISMATCH", "worker invocation differs")
    root, store_id, study_id, cell_hash, output = arguments[1:6]
    store = ResearchStore(Path(root), store_id)
    spec = store.manifest("study-"+study_id)["spec"]
    cells = [cell for cell in spec["cells"] if digest(cell) == cell_hash]
    if len(cells) != 1 or spec.get("runtime_binding") != {"root": str(Path(root).resolve()), "store_id": store_id}:
        raise ResearchError("CONTRACT_MISMATCH", "worker root/store or selected cell differs")
    from experiments.pirc27.mixture_plugin import PLUGIN_IDS
    mixture_ids = PLUGIN_IDS|{"affine-mixture-production-chunk"}
    state_limit = 65536 if cells[0]["plugin_id"] in mixture_ids else 16384
    restored = read_frame(Path(arguments[6]), state_limit) if len(arguments) == 7 else None
    if len(arguments) == 7 and restored is None:
        raise ResearchError("CONTRACT_MISMATCH", "worker resume state is absent or invocation differs")
    admission = None
    if reference is not None:
        from application.propagation_qualification_admission import load_worker_admission
        admission = load_worker_admission(store, reference, spec, cells[0], output)
    control = WorkerControl.from_environment()
    from application.propagation_resources import observation_enabled, attempt_from_output, PropagationResourceProbe
    probe = (PropagationResourceProbe(spec, cells[0], attempt_from_output(spec, output), resumed=restored is not None)
             if observation_enabled(cells[0]) else None)
    def observe(state):
        packet = probe.snapshot(state)
        write_frame(Path(output).with_name("propagation-resources.json"), packet, 16384,
                    deadline=control.descriptor["deadline"] if control is not None else None)
        return packet
    start = time.monotonic()
    def checkpoint(state, total):
        completed = state["step"]
        if control is not None and control.poll() is not None:
            elapsed = time.monotonic()-start
            advanced = completed-(restored["step"] if restored else 0)
            rate = advanced/max(elapsed, 1e-12)
            if probe is not None:
                observe("CHECKPOINTED")  # Before ACK/native stop, never inside resumable state.
            control.save(state, {"completed_steps": completed, "total_steps": total,
                "throughput_per_second": rate, "eta_seconds": (total-completed)/max(rate, 1e-12)})
            raise SystemExit(85)
    recoverable = cells[0]["plugin_id"] in {"affine-propagation-chunk", "synthetic-propagation-chunk", "affine-mlmc-qualification-chunk", "affine-mlmc-production-chunk", "affine-path-production-chunk"}|mixture_ids
    try:
        result = execute_propagation(spec, cells[0], resume_state=restored,
            checkpoint=checkpoint if recoverable else None, admission=admission)
    except BaseException as exc:
        if probe is not None and not (isinstance(exc, SystemExit) and exc.code == 85):
            try:
                observe("FAILED")
            except Exception:
                # Preserve the original numerical/control failure. Missing native
                # observations remain unavailable, never a successful zero value.
                print("Propagation resource observation unavailable before worker failure", file=sys.stderr)
        raise
    if probe is not None:
        result["resource_observation"] = observe("SUCCEEDED")
    atomic_write(Path(output), encode(result))


if __name__ == "__main__":
    main()
