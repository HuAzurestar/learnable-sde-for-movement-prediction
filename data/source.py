"""多源数据标准化与统一 ``DataSource`` 接口。

trajectory / condition / checkpoint 三类数据源收敛到同一接口：
  - load()：装载（段列表 / SDE）
  - validate()：坏数据 fail-fast
  - features()：条件特征（ConditionSource 专有，接口隔离）

声明式条件特征规格 CONDITION_SPECS 替代 backend.py 的 per-kind if/elif。backend 现有
cond 逻辑（含段偏移对齐）属正确性关键、暂不动，后续迁移到本模块的 extract_features。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

import numpy as np
import pandas as pd

from domain import TrajectorySegment
from .paths import resolve
from .validation import DataValidationError, ensure_finite


# ---------------------------------------------------------------------------
# 声明式条件特征规格（替代 per-kind if/elif）
# ---------------------------------------------------------------------------
CONDITION_SPECS: Dict[str, Dict[str, Any]] = {
    "solar_elev": {"columns": ["solar_elev", "day_fraction"], "agg": "mean"},
    "is_day": {"columns": ["is_day"], "agg": "mean"},
    "terrain": {
        "columns": ["dem_elev", "dem_slope", "landcover"],
        "agg": "mean",
        "per_column_agg": {"landcover": "mode"},
        "filter": {"column": "has_map", "value": 1},
    },
    "osm": {
        "columns": ["road_dist", "water_dist", "building_dist"],
        "agg": "mean",
        "filter": {"column": "has_osm", "value": 1},
    },
}


def extract_features(sub: pd.DataFrame, kind: str) -> Optional[np.ndarray]:
    """按 CONDITION_SPECS 提取段级条件特征向量 (D,)。

    - 未知 kind / 缺列 / 缺过滤列 → raise DataValidationError（配置/模式错误，fail-fast）；
    - 特征含 NaN（数据质量）→ 返回 None（诚实 skip）。
    """
    spec = CONDITION_SPECS.get(kind)
    if spec is None:
        raise DataValidationError(
            f"未知条件 kind {kind!r}（可用: {sorted(CONDITION_SPECS)}）"
        )
    flt = spec.get("filter")
    if flt is not None:
        if flt["column"] not in sub.columns:
            raise DataValidationError(
                f"条件 kind={kind} 缺过滤列 {flt['column']!r} —— schema 错误"
            )
        sub = sub[sub[flt["column"]] == flt["value"]]
        if len(sub) == 0:
            return None
    per_col = spec.get("per_column_agg", {})
    default_agg = spec.get("agg", "mean")
    vals = []
    for col in spec["columns"]:
        if col not in sub.columns:
            raise DataValidationError(f"条件 kind={kind} 缺列 {col!r} —— schema 错误")
        agg = per_col.get(col, default_agg)
        if agg == "mode":
            mode_vals = sub[col].mode()
            vals.append(float(mode_vals.iloc[0]) if len(mode_vals) else np.nan)
        else:  # mean
            vals.append(float(sub[col].mean()))
    if any(np.isnan(v) for v in vals):
        return None
    return np.array(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Typed data source and condition-provider ports
# ---------------------------------------------------------------------------
T = TypeVar("T")


class DataSource(Generic[T], ABC):
    """A source with one declared return type."""

    name: str = "base"

    @abstractmethod
    def load(self) -> T:
        raise NotImplementedError

    def validate(self) -> None:
        """校验数据源（路径存在/模式/有限性）。默认 no-op，子类按需覆盖。"""
        return None


class ConditionProvider(ABC):
    """Condition features are queried by segment; they are not trajectory data."""

    @abstractmethod
    def features_for(
        self,
        segment: TrajectorySegment,
        kind: str,
    ) -> Optional[np.ndarray]:
        ...


# ---------------------------------------------------------------------------
# TrajectorySource —— 轨迹腿
# ---------------------------------------------------------------------------
class TrajectorySource(DataSource[List[TrajectorySegment]]):
    """轨迹段数据源，包装 ``data.loader.SegmentLoader``。"""

    name = "trajectory"

    def __init__(self, split: str = "train", max_segments: Optional[int] = None,
                 seed: int = 20260814):
        self.split = split
        self.max_segments = max_segments
        self.seed = seed

    def load(self) -> List[TrajectorySegment]:
        from .loader import DEFAULT_DATA_ROOT, SegmentLoader

        loader = SegmentLoader(data_root=DEFAULT_DATA_ROOT, split=self.split, seed=self.seed,
                               max_segments=self.max_segments)
        loader.load()
        return loader.segments

    def validate(self) -> None:
        if not resolve("data_root").exists():
            raise DataValidationError(f"轨迹数据根目录不存在: {resolve('data_root')}")


# ---------------------------------------------------------------------------
# ConditionSource —— 条件特征
# ---------------------------------------------------------------------------
class ConditionSource(ConditionProvider):
    """条件特征源（solar/is_day/terrain/weather）。"""

    name = "condition"

    def __init__(self):
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def slice_for(self, file_id: str) -> Optional[pd.DataFrame]:
        """读 file_id 的 cond 切片（缓存）。切片不可得 → None。"""
        if file_id in self._cache:
            return self._cache[file_id]
        cond_root = resolve("cond_root")
        if not cond_root.exists():
            self._cache[file_id] = None
            return None
        import glob as _glob

        hits = _glob.glob(str(cond_root / "**" / f"{file_id}*_cond.parquet"), recursive=True)
        if not hits:
            self._cache[file_id] = None
            return None
        df = pd.read_parquet(hits[0])
        df["t_epoch"] = df["t"].astype("datetime64[s]").astype(np.int64)
        self._cache[file_id] = df
        return df

    def features(self, sub: pd.DataFrame, kind: str) -> Optional[np.ndarray]:
        return extract_features(sub, kind)

    def features_for(
        self,
        segment: TrajectorySegment,
        kind: str,
    ) -> Optional[np.ndarray]:
        """Extract aligned features when the segment declares an absolute start.

        The legacy backend owns historical offset reconstruction. New data
        adapters must place ``absolute_start_epoch`` in segment metadata so the
        condition provider remains independent of experiment call order.
        """
        file_id = str(segment.meta.get("file_id", ""))
        if not file_id:
            raise DataValidationError("segment metadata 缺 file_id")
        start = segment.meta.get("absolute_start_epoch")
        if start is None:
            raise DataValidationError("segment metadata 缺 absolute_start_epoch")
        data = self.slice_for(file_id)
        if data is None:
            return None
        duration = float(segment.t[-1] - segment.t[0])
        selected = data[
            (data["t_epoch"] >= float(start))
            & (data["t_epoch"] <= float(start) + duration)
        ]
        if len(selected) == 0:
            return None
        return extract_features(selected, kind)

    def validate(self) -> None:
        if not resolve("cond_root").exists():
            raise DataValidationError(f"条件数据根目录不存在: {resolve('cond_root')}")


# ---------------------------------------------------------------------------
# CheckpointSource —— checkpoint 复用
# ---------------------------------------------------------------------------
class CheckpointSource:
    """Load a compatible training checkpoint without retraining."""

    name = "checkpoint"

    def __init__(self, coord: str = "coord_x"):
        self.coord = coord

    def load(self) -> Any:
        import json

        import torch
        from models.segment_constant import SegmentConstantSDE

        ckpt_dir = resolve("checkpoint")
        cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
        for key in ("n_modes", "dt_ref", self.coord):
            if key not in cfg:
                raise DataValidationError(f"checkpoint config.json 缺键 {key!r}")
        sde = SegmentConstantSDE(n_modes=cfg["n_modes"], kappa=cfg.get("kappa", 0.0),
                                 dt_ref=cfg.get("dt_ref", 60.0))
        sd = cfg[self.coord].get("state", {})
        with torch.no_grad():
            for k in ("Gamma", "a", "c", "g", "prior_logits"):
                if k not in sd:
                    raise DataValidationError(f"checkpoint state 缺键 {k!r}")
                tensor = torch.tensor(sd[k], dtype=torch.float64)
                ensure_finite(tensor, name=f"checkpoint.{k}", source=self.coord)
                getattr(sde, k).copy_(tensor)
        return sde, cfg

    def validate(self) -> None:
        if not (resolve("checkpoint") / "config.json").exists():
            raise DataValidationError(f"checkpoint 缺失: {resolve('checkpoint') / 'config.json'}")
