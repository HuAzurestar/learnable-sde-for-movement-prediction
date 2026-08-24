"""Phase 0 安全网 —— backend 公开契约的特征化锚。

`fit_predict` / `load_eval_segments` 等此前零测试覆盖；拆 `backend.py`（Phase 3）前
先在此钉住公开 API 的形状 / 有限性 / 确定性 id，拆分前后输出一致即判过。

运行：python -m pytest tests/test_backend.py
"""

import numpy as np

from experiments.backend import (
    fit_predict,
    load_eval_segments,
    load_eval_segments_d2,
)


def test_load_eval_segments_shape(public_artifacts):
    ids, obs = load_eval_segments(n_seg=3)
    assert len(ids) == 3
    assert len(obs) == 3
    assert all(isinstance(i, str) for i in ids)
    assert all(np.isfinite(o) for o in obs)


def test_load_eval_segments_deterministic_ids(public_artifacts):
    ids1, _ = load_eval_segments(n_seg=3)
    ids2, _ = load_eval_segments(n_seg=3)
    assert ids1 == ids2


def test_load_eval_segments_d2_shape(public_artifacts):
    ids, obs = load_eval_segments_d2(n_seg=2)
    assert len(ids) == 2
    assert all(len(o) == 2 and np.isfinite(o).all() for o in obs)


def test_fit_predict_smoke(public_artifacts):
    # 默认 N_EVAL_SEG（smoke=40）：与 _load_eval_by_ids 的 max(len,40) 采样一致，避免子集被静默丢弃。
    ids, _ = load_eval_segments()
    config = {"condition": "none", "model": "seg_constant"}
    preds = fit_predict(config, ids, m=10)
    assert len(preds) == len(ids)
    for p in preds:
        assert len(p) == 10
        assert np.isfinite(p).all()
