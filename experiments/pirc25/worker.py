"""Internal adapter entry: only explicitly supported synthetic affine fixtures."""

import sys
from pathlib import Path

from infrastructure.research_store import ResearchStore, atomic_write, digest, encode
from .affine import execute


def main():
    root, store_id, study_id, cell_hash, output = sys.argv[1:]
    store = ResearchStore(Path(root), store_id)
    spec = store.manifest("study-" + study_id)["spec"]
    cells = [cell for cell in spec["cells"] if digest(cell) == cell_hash]
    if len(cells) != 1 or cells[0].get("visibility") != "synthetic":
        raise ValueError("affine fixture worker only accepts registered synthetic cells")
    atomic_write(Path(output), encode(execute(spec, cells[0])))


if __name__ == "__main__":
    main()
