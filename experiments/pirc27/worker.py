"""Internal shared-runner worker; never creates a store or grants access."""

import sys
from pathlib import Path

from application.propagation_execution import execute_propagation
from infrastructure.research_store import ResearchError, ResearchStore, atomic_write, digest, encode


def main():
    root, store_id, study_id, cell_hash, output = sys.argv[1:]
    store = ResearchStore(Path(root), store_id)
    spec = store.manifest("study-"+study_id)["spec"]
    cells = [cell for cell in spec["cells"] if digest(cell) == cell_hash]
    if len(cells) != 1 or spec.get("runtime_binding") != {"root": str(Path(root).resolve()), "store_id": store_id}:
        raise ResearchError("CONTRACT_MISMATCH", "worker root/store or selected cell differs")
    atomic_write(Path(output), encode(execute_propagation(spec, cells[0])))


if __name__ == "__main__":
    main()
