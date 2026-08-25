"""统一数据管线与可辨识性护栏。

轨迹 → (t, x, cond) 统一批量; 异构对齐; 特征派生（solar_elev 等）。
按段携带 dt 元数据；Δt 可辨识诊断前置——细粒度段(Δt≤60s)→段级模型，
粗粒度段(Δt≥300s)→路由点态混合/退化基线（I-1 §4.2 规避路径工程化）。

本地研究数据目录预期包含：
  unified_full_leg.parquet  25.98M 点 / 96,830 段（train 18.2M / val 3.8M / eval 3.9M）
  zhejiang_holdout.parquet  80,159 点（finetune 60% / val 20% / eval 20%）
  global_splits.json        file_id 级 70/15/15 两两无重叠划分
  zhejiang_splits.json      track_id 级 60/20/20
schema: track_id, file_id, cluster_A, country, region, city, segment_id,
        t, x, y, z, vx, vy, speed   （x/y 为局部切平面米制；t 秒）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from domain import TrajectorySegment
from .geolife import (
    assert_cross_source_file_id_namespace_disjoint,
    geolife_file_id_splits,
)
from .paths import resolve
from .validation import (
    DataValidationError,
    ensure_finite,
    ensure_monotonic_increasing,
    validate_segment,
    validate_transitions,
)

DEFAULT_DATA_ROOT = resolve("data_root")

STATE_COLS = ["x", "y"]          # 局部切平面位置（2-D）


Segment = TrajectorySegment


def identifiability_diagnosis(dt: float) -> str:
    """I-1 可辨识诊断：返回模型路由。"""
    if dt <= 60.0:
        return "segment"            # 细粒度 → 段级常模式
    if dt < 300.0:
        return "segment_caution"    # 60<Δt<300: 可辨识性存疑，段级+诊断并行
    return "point_mixture"          # Δt≥300: 模式塌缩 → 路由点态混合/退化基线


def _seg_label(seg: "Segment") -> str:
    """段来源标签（fail-fast 消息定位用）。"""
    return f"seg:{seg.meta.get('segment_id', '?')}"


def to_phase_space_1d(seg: "Segment", coord: int = 0) -> torch.Tensor:
    """1D 相位空间 [X, V]（I-1 段级常模式消费）: 取 coord 坐标，V = ΔX/Δt（末点回填）。

    返回 (T+1, 2)。I-1 精确核在 (X,V) 上直接可辨识。
    """
    x = seg.x[:, coord].to(torch.float64)
    t = seg.t.to(torch.float64)
    label = _seg_label(seg)
    ensure_finite(x, name="x", source=label)
    ensure_finite(t, name="t", source=label)
    ensure_monotonic_increasing(t, name="t", source=label)
    dt = torch.diff(t).clamp(min=1e-6)
    v = torch.diff(x) / dt
    v = torch.cat([v, v[-1:]])
    return torch.stack([x, v], dim=-1)


class StateStats:
    """每维 mean/std 拟合于一个 split；训练/推理同一变换（可逆仿射，保持高斯核）。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float64)
        self.std = np.where(np.asarray(std, dtype=np.float64) < 1e-8, 1.0,
                            np.asarray(std, dtype=np.float64))

    @classmethod
    def fit(cls, x: np.ndarray) -> "StateStats":
        m = x.mean(axis=0)
        s = x.std(axis=0)
        return cls(m, s)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StateStats":
        return cls(np.array(d["mean"]), np.array(d["std"]))


class SegmentLoader:
    """轨迹段统一 loader（file_id 分区供给）。

    split ∈ {train, val, eval, geolife_train, geolife_val, geolife_eval,
             unified_train, unified_val, unified_eval,
             zhejiang_finetune, zhejiang_val, zhejiang_eval, smoke}
    train/val/eval 按 global_splits.json（file_id 级无重叠）；浙江留出仅微调/终评解锁。
    GeoLife 的 file_id 分区并非用户独立分区，不能据此声称跨用户泛化无泄漏。
    `max_segments` 是整个请求的总上限；对 unified_* 也最多载入 N 段，
    而不是对 OSM 和 GeoLife 各载入 N 段。
    """

    def __init__(self, data_root: str | Path = DEFAULT_DATA_ROOT,
                 split: str = "train", seed: int = 20260814,
                 max_segments: int | None = None,
                 geolife_root: str | Path | None = None):
        self.data_root = Path(data_root)
        self.geolife_root = (
            Path(geolife_root) if geolife_root is not None else resolve("geolife_data_root")
        )
        self.split = split
        self.seed = seed
        if max_segments is not None and max_segments < 0:
            raise ValueError("max_segments must be non-negative")
        self.max_segments = max_segments
        self.segments: List[Segment] = []
        self._geolife_splits: Optional[dict[str, set[str]]] = None

    def _split_parts(self) -> tuple[str, str]:
        if self.split == "smoke":
            return "smoke", ""
        if self.split.startswith("zhejiang_"):
            part = self.split.removeprefix("zhejiang_")
            if part not in {"finetune", "val", "eval"}:
                raise DataValidationError(f"unsupported split: {self.split!r}")
            return "zhejiang", part
        for domain in ("geolife", "unified"):
            prefix = f"{domain}_"
            if self.split.startswith(prefix):
                part = self.split.removeprefix(prefix)
                if part not in {"train", "val", "eval"}:
                    raise DataValidationError(f"unsupported split: {self.split!r}")
                return domain, part
        if self.split not in {"train", "val", "eval"}:
            raise DataValidationError(f"unsupported split: {self.split!r}")
        return "osm", self.split

    # -- split 供给 ----------------------------------------------------------
    def _file_ids_for_split(self) -> Optional[set]:
        if self.split.startswith("zhejiang"):
            sp = json.loads((self.data_root / "zhejiang_splits.json").read_text(encoding="utf-8"))
            key = self.split.removeprefix("zhejiang_")   # finetune / val / eval
            return set(sp.get(key, []))
        if self.split == "smoke":
            return None
        return self._osm_file_ids_for(self.split)

    def _osm_file_ids_for(self, part: str) -> set[str]:
        sp = json.loads((self.data_root / "global_splits.json").read_text(encoding="utf-8"))
        if part not in sp:
            raise DataValidationError(f"global_splits.json missing partition {part!r}")
        return {str(file_id) for file_id in sp[part]}

    def _geolife_file_ids_for(self, part: str) -> set[str]:
        if self._geolife_splits is None:
            self._geolife_splits = geolife_file_id_splits(self.geolife_root)
        return set(self._geolife_splits[part])

    def _load_holdout(self) -> int:
        """浙江留出 80k（唯一留出腿，训练期零接触）。"""
        df = pd.read_parquet(self.data_root / "zhejiang_holdout.parquet")
        keep = self._file_ids_for_split()
        df = df[df["file_id"].astype(str).isin(keep or set())]
        return self._build_segments(df, source="zhejiang_holdout")

    def _segment_keys(self, path: Path, keep: set[str]) -> pd.DataFrame:
        keys = pd.read_parquet(path, columns=["file_id", "segment_id"])
        return keys[keys["file_id"].astype(str).isin(keep)].drop_duplicates()

    def _limit_segment_keys(self, keys: pd.DataFrame) -> pd.DataFrame:
        if self.max_segments is not None and len(keys) > self.max_segments:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(keys), size=self.max_segments, replace=False)
            return keys.iloc[indices]
        return keys

    def _load_selected_table(
        self, path: Path, keys: pd.DataFrame, *, source: str
    ) -> int:
        if len(keys) == 0:
            return 0
        frame = pd.read_parquet(
            path, columns=["segment_id", "file_id", "t"] + STATE_COLS
        )
        frame = frame.merge(
            keys,
            on=["file_id", "segment_id"],
            how="inner",
            validate="many_to_one",
        )
        return self._build_segments(frame, source=source)

    def _load_table(self, path: Path, keep: set[str], *, source: str) -> int:
        """Load selected (file_id, segment_id) pairs without cross-file merging."""
        keys = self._limit_segment_keys(self._segment_keys(path, keep))
        return self._load_selected_table(path, keys, source=source)

    def _load_full_leg(self, part: str) -> int:
        """全量腿 25.98M，按 file_id split 过滤（不整表载入内存）。"""
        return self._load_table(
            self.data_root / "unified_full_leg.parquet",
            self._osm_file_ids_for(part),
            source="unified_full_leg",
        )

    def _load_geolife_leg(self, part: str) -> int:
        return self._load_table(
            self.geolife_root / "geolife_leg.parquet",
            self._geolife_file_ids_for(part),
            source="geolife_leg",
        )

    def _load_unified(self, part: str) -> int:
        """Load a deterministic, globally capped sample across both sources."""
        sources = (
            (
                "unified_full_leg",
                self.data_root / "unified_full_leg.parquet",
                self._osm_file_ids_for(part),
            ),
            (
                "geolife_leg",
                self.geolife_root / "geolife_leg.parquet",
                self._geolife_file_ids_for(part),
            ),
        )
        key_frames = []
        for source, path, keep in sources:
            keys = self._segment_keys(path, keep).copy()
            keys["__source"] = source
            key_frames.append(keys)
        selected = self._limit_segment_keys(
            pd.concat(key_frames, ignore_index=True)
        )
        total = 0
        for source, path, _ in sources:
            keys = selected[selected["__source"] == source].drop(
                columns=["__source"]
            )
            total += self._load_selected_table(path, keys, source=source)
        return total

    def _build_segments(self, df: pd.DataFrame, source: str) -> int:
        n = 0
        for (file_id, raw_segment_id), g in df.groupby(
            ["file_id", "segment_id"], sort=False
        ):
            g = g.sort_values("t")
            x = g[STATE_COLS].to_numpy(dtype=np.float64)
            t = g["t"].to_numpy(dtype=np.float64)
            if len(x) < 2:
                continue
            fid = str(file_id)
            canonical_segment_id = f"{source}:{fid}:{raw_segment_id}"
            seg = Segment(
                t=torch.tensor(t, dtype=torch.float64),
                x=torch.tensor(x, dtype=torch.float64),
                dt=float(np.median(np.diff(t))) if len(t) > 1 else 60.0,
                meta={"source": source, "segment_id": canonical_segment_id,
                      "raw_segment_id": str(raw_segment_id), "file_id": fid},
            )
            validate_segment(seg, source=f"{source}:{seg.meta['segment_id']}")
            self.segments.append(seg)
            n += 1
        return n

    # -- 主入口 ----------------------------------------------------------------
    def load(self) -> int:
        """装载当前 split，返回段数。"""
        self.segments.clear()
        domain, part = self._split_parts()
        if domain == "smoke":
            return self._build_segments(
                pd.read_parquet(self.data_root / "smoke_fullleg.parquet"),
                source="smoke_fullleg")
        if domain == "zhejiang":
            return self._load_holdout()
        if domain == "geolife":
            return self._load_geolife_leg(part)
        if domain == "unified":
            assert_cross_source_file_id_namespace_disjoint(
                self.geolife_root, self.data_root
            )
            return self._load_unified(part)
        return self._load_full_leg(part)

    def sample_segments(self, n: int) -> List[Segment]:
        if len(self.segments) <= n:
            return self.segments
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(self.segments), size=n, replace=False)
        return [self.segments[i] for i in idx]

    def stats(self) -> dict:
        n_seg = len(self.segments)
        return {"n_segments": n_seg, "split": self.split,
                "routed_to_segment": sum(1 for s in self.segments
                                         if identifiability_diagnosis(s.dt) == "segment"),
                "routed_to_segment_caution": sum(1 for s in self.segments
                                                 if identifiability_diagnosis(s.dt) == "segment_caution"),
                "routed_to_point_mixture": sum(1 for s in self.segments
                                               if identifiability_diagnosis(s.dt) == "point_mixture")}

    def transitions(self, dtype: torch.dtype = torch.float64
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """展平所有段为一步转移对 (x, y, dt)，供 NLL/CRPS 估计器消费。"""
        xs, ys, dts = [], [], []
        for s in self.segments:
            x = s.x
            if len(x) < 2:
                continue
            xs.append(x[:-1])
            ys.append(x[1:])
            dts.append(torch.diff(s.t))
        if not xs:
            raise ValueError("no transitions loaded")
        x, y, dt = (torch.cat(xs, dim=0).to(dtype),
                    torch.cat(ys, dim=0).to(dtype),
                    torch.cat(dts, dim=0).to(dtype))
        validate_transitions(x, y, dt, source=f"split:{self.split}")
        return x, y, dt
