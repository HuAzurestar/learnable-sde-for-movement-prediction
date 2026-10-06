"""Internal shared-runner worker; never creates a store or grants access."""

import sys
import time
from pathlib import Path

from application.propagation_execution import execute_propagation
from infrastructure.research_store import ResearchError, ResearchStore, atomic_write, digest, encode
from infrastructure.research_control import WorkerControl, read_frame


def main():
    arguments = list(sys.argv)
    reference = None
    if len(arguments) >= 3 and arguments[-2] == "--admission-hash":
        reference = arguments[-1]
        arguments = arguments[:-2]
    if len(arguments) not in {6, 7}:
        raise ResearchError("CONTRACT_MISMATCH", "worker invocation differs")
    root, store_id, study_id, cell_hash, output = arguments[1:6]
    restored = read_frame(Path(arguments[6]), 16384) if len(arguments) == 7 else None
    if len(arguments) == 7 and restored is None:
        raise ResearchError("CONTRACT_MISMATCH", "worker resume state is absent or invocation differs")
    store = ResearchStore(Path(root), store_id)
    spec = store.manifest("study-"+study_id)["spec"]
    cells = [cell for cell in spec["cells"] if digest(cell) == cell_hash]
    if len(cells) != 1 or spec.get("runtime_binding") != {"root": str(Path(root).resolve()), "store_id": store_id}:
        raise ResearchError("CONTRACT_MISMATCH", "worker root/store or selected cell differs")
    admission = None
    if reference is not None:
        from application.propagation_qualification_admission import load_worker_admission
        admission = load_worker_admission(store, reference, spec, cells[0], output)
    control = WorkerControl.from_environment()
    start = time.monotonic()
    def checkpoint(state, total):
        completed = state["step"]
        if control is not None and control.poll() is not None:
            elapsed = time.monotonic()-start
            advanced = completed-(restored["step"] if restored else 0)
            rate = advanced/max(elapsed, 1e-12)
            control.save(state, {"completed_steps": completed, "total_steps": total,
                "throughput_per_second": rate, "eta_seconds": (total-completed)/max(rate, 1e-12)})
            raise SystemExit(85)
    recoverable = cells[0]["plugin_id"] in {"affine-propagation-chunk", "synthetic-propagation-chunk", "affine-mlmc-qualification-chunk", "affine-mlmc-production-chunk"}
    atomic_write(Path(output), encode(execute_propagation(spec, cells[0], resume_state=restored,
        checkpoint=checkpoint if recoverable else None, admission=admission)))


if __name__ == "__main__":
    main()
