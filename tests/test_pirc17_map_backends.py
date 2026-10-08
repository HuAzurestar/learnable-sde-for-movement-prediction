from pathlib import Path

import pytest

from experiments.pirc17.development_rollout import resolve_map_backend


@pytest.mark.parametrize("name,expected", [("legacy", "RawMapQuery"), ("batched", "BatchedRawMapQuery"),
                                         ("cached", "CachedBatchedRawMapQuery"),
                                         ("multicell", "MultiCellRawMapQuery")])
def test_explicit_backends_bind_raster_and_geometry_sources(name, expected, tmp_path):
    pytest.importorskip("trajectory.online_terrain", reason="DSDE companion must be on PYTHONPATH")
    query_type, modules = resolve_map_backend(name)
    assert query_type.__name__ == expected
    names = [module.__name__ for module in modules]
    assert len(names) == len(set(names))
    assert {"trajectory.online_terrain", "trajectory.linear_materialization", "map_data.terrain_features"} <= set(names)
    assert ("trajectory.batched_terrain" in names) == (name != "legacy")
    assert ("trajectory.cached_terrain" in names) == (name in {"cached", "multicell"})
    assert ("trajectory.multicell_terrain" in names) == (name == "multicell")
    assert all(Path(module.__file__).is_file() for module in modules)
    query = query_type(tmp_path, [])
    try:
        if name in {"cached", "multicell"}:
            assert query.identity["raster_cache_limits"] == {"retained_array_bytes": 64*1024*1024, "open_readers": 4}
        if name == "multicell":
            assert query.identity["geometry_cache_limits"]["retained_cells"] == 4
    finally:
        query.close()


def test_unknown_backend_fails_before_loading_data():
    with pytest.raises(ValueError, match="unknown"):
        resolve_map_backend("approximate")
