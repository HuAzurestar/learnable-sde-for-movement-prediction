"""pytest 全局配置 + Windows BLAS 顺序护栏。

numpy/pandas 必须先于 torch 装载，否则 torch.linalg 与 pandas 的 OpenBLAS 符号冲突
→ 段错误（见 experiments/ablate.py:26-27）。conftest 在收集测试模块前被导入，故在此
置顶 import 即可保证全测试进程内的加载顺序。
"""

import json

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import pytest


@pytest.fixture
def public_artifacts(tmp_path, monkeypatch):
    """Provide small local-only trajectory and checkpoint artifacts."""
    data_root = tmp_path / "data"
    checkpoint_root = tmp_path / "checkpoint"
    data_root.mkdir()
    checkpoint_root.mkdir()

    rows = []
    for segment_index in range(64):
        file_id = f"synthetic-file-{segment_index // 4:03d}"
        segment_id = f"synthetic-segment-{segment_index:03d}"
        velocity = 0.03 + 0.005 * (segment_index % 3)
        for step in range(8):
            t = float(step * 60)
            rows.append(
                {
                    "file_id": file_id,
                    "segment_id": segment_id,
                    "t": t,
                    "x": float(segment_index) + velocity * t,
                    "y": -float(segment_index) + 0.5 * velocity * t,
                }
            )
    pd.DataFrame(rows).to_parquet(data_root / "smoke_fullleg.parquet", index=False)

    state = {
        "Gamma": [0.08, 0.04, 0.02],
        "a": [0.0, 0.0, 0.0],
        "c": [0.0, 0.01, -0.01],
        "g": [0.15, 0.25, 0.35],
        "prior_logits": [0.0, 0.0, 0.0],
    }
    checkpoint = {
        "n_modes": 3,
        "kappa": 0.0,
        "dt_ref": 60.0,
        "coord_x": {"state": state},
    }
    (checkpoint_root / "config.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )

    monkeypatch.setenv("LEARNABLE_SDE_DATA_ROOT", str(data_root))
    monkeypatch.setenv("LEARNABLE_SDE_CHECKPOINT", str(checkpoint_root))

    import data.loader as loader_module
    import experiments.backend as backend_module

    monkeypatch.setattr(loader_module, "DEFAULT_DATA_ROOT", data_root)
    monkeypatch.setattr(backend_module, "DEFAULT_DATA_ROOT", data_root)
    monkeypatch.setattr(backend_module, "_LEG_SMOKE", data_root / "smoke_fullleg.parquet")
    monkeypatch.setattr(backend_module, "_LEG_FULL", data_root / "unified_full_leg.parquet")
    monkeypatch.setattr(backend_module, "_ctx", backend_module._Context())

    return {"data_root": data_root, "checkpoint_root": checkpoint_root}
