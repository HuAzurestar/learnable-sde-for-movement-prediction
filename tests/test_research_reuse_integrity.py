"""Completed-cell reuse must validate the actual immutable output, not a flag."""

import pytest

from experiments.pirc25.affine import fixture_spec
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchError, ResearchStore, digest


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_reuse_rejects_damaged_completed_result(tmp_path, damage):
    store = ResearchStore(tmp_path, "reuse-integrity", initialize=True)
    value = fixture_spec("fixture", 1, (19,))
    store.register(value, digest(value))
    runner = SharedRunner(store)
    original = runner.run_cell("fixture", digest(value["cells"][0]))
    assert original["state"] == "SUCCEEDED"
    path = store.path / "artifacts" / original["artifact_id"]
    if damage == "corrupt":
        path.write_bytes(b"damaged synthetic fixture output")
    else:
        path.unlink()
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT|MISSING_INPUT"):
        runner.run_cell("fixture", digest(value["cells"][0]))
    assert len(store.attempts()) == 1, "bad output must not silently rerun or overwrite the completed cell"
