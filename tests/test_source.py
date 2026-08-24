"""P3 多源数据标准化测试：DataSource ABC、声明式特征提取、三个数据源。

运行：python -m pytest tests/test_source.py
"""

import numpy as np
import pandas as pd
import pytest
import torch

from data.source import (
    DataSource,
    ConditionProvider,
    ConditionSource,
    CheckpointSource,
    TrajectorySource,
    extract_features,
)
from data.validation import DataValidationError


# -- DataSource ABC -----------------------------------------------------------
def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()


def test_data_ports_are_interface_segregated():
    assert issubclass(TrajectorySource, DataSource)
    assert issubclass(ConditionSource, ConditionProvider)
    assert not issubclass(ConditionSource, DataSource)
    assert not issubclass(CheckpointSource, DataSource)


# -- 声明式特征提取 -----------------------------------------------------------
def test_extract_features_mean():
    sub = pd.DataFrame({"solar_elev": [0.1, 0.2, 0.3], "day_fraction": [0.5, 0.6, 0.7]})
    f = extract_features(sub, "solar_elev")
    assert f is not None and f.shape == (2,)
    assert np.allclose(f, [0.2, 0.6])


def test_extract_features_terrain_mode_and_filter():
    sub = pd.DataFrame({
        "has_map": [1, 1, 0, 1],
        "dem_elev": [10.0, 20.0, 999.0, 30.0],
        "dem_slope": [1.0, 2.0, 999.0, 3.0],
        "landcover": [1, 2, 999, 2],
    })
    f = extract_features(sub, "terrain")
    assert f is not None and f.shape == (3,)
    assert f[0] == 20.0 and f[1] == 2.0 and f[2] == 2.0  # has_map==1 行 0/1/3


def test_extract_features_unknown_kind_raises():
    with pytest.raises(DataValidationError):
        extract_features(pd.DataFrame(), "no_such_kind")


def test_extract_features_missing_column_raises():
    with pytest.raises(DataValidationError):
        extract_features(pd.DataFrame({"a": [1]}), "solar_elev")


def test_extract_features_nan_returns_none():
    sub = pd.DataFrame({"solar_elev": [float("nan")], "day_fraction": [0.5]})
    assert extract_features(sub, "solar_elev") is None


# -- 三个数据源 -----------------------------------------------------------
def test_trajectory_source_loads_smoke(public_artifacts):
    segs = TrajectorySource(split="smoke").load()
    assert len(segs) > 0
    assert hasattr(segs[0], "x") and hasattr(segs[0], "t")


def test_checkpoint_source_loads(public_artifacts):
    src = CheckpointSource(coord="coord_x")
    src.validate()
    sde, cfg = src.load()
    assert cfg["n_modes"] == 3
    assert torch.all(torch.isfinite(sde.Gamma))
    assert torch.all(torch.isfinite(sde.g))
