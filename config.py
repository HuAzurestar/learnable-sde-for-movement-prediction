"""类型化配置：Config / Components 数据类 + fail-fast 校验。

替代裸 dict 透传：`run.py` 载入 `Config` 后构造注入组件；配错当场 raise，而非静默落基线。

校验依据：`components.*` 取值须在 config.yaml 已声明的 `ablation_matrix.*` 合法集内；
`seed`/`dtype`/`device` 各自校验类型与取值域。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from domain import ConfigurationError

ALLOWED_DTYPES = ("float32", "float64")
ALLOWED_DEVICES = ("cpu", "cuda")

# 消融开关轴（components 的五个维度），与 config.yaml `components` 一致。
_COMPONENT_AXES = ("model", "estimator", "transfer", "condition", "inference")


@dataclass
class Components:
    """消融开关（config.yaml `components` 一行）。"""

    model: str = "I1"
    estimator: str = "EM"
    transfer: str = "none"
    condition: str = "none"
    inference: str = "exact"


@dataclass
class Config:
    """config.yaml 的类型化视图。"""

    seed: int = 20260814
    dtype: str = "float64"
    device: str = "cpu"
    paths: Dict[str, Any] = field(default_factory=dict)
    components: Components = field(default_factory=Components)
    model: Dict[str, Any] = field(default_factory=dict)
    ablation_matrix: Dict[str, Any] = field(default_factory=dict)
    equivalence_regression: Dict[str, Any] = field(default_factory=dict)
    config_key_map: Dict[str, Any] = field(default_factory=dict)
    protocol: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        comp_raw = raw.get("components", {}) or {}
        comps = Components(
            model=comp_raw.get("model", "I1"),
            estimator=comp_raw.get("estimator", "EM"),
            transfer=comp_raw.get("transfer", "none"),
            condition=comp_raw.get("condition", "none"),
            inference=comp_raw.get("inference", "exact"),
        )
        return cls(
            seed=raw.get("seed", 20260814),
            dtype=raw.get("dtype", "float64"),
            device=raw.get("device", "cpu"),
            paths=raw.get("paths", {}) or {},
            components=comps,
            model=raw.get("model", {}) or {},
            ablation_matrix=raw.get("ablation_matrix", {}) or {},
            equivalence_regression=raw.get("equivalence_regression", {}) or {},
            config_key_map=raw.get("config_key_map", {}) or {},
            protocol=raw.get("protocol", {}) or {},
        )

    def validate(self) -> None:
        """fail-fast：非法配置当场 raise ValueError（含合法值提示）。"""
        if not isinstance(self.seed, int):
            raise ConfigurationError(f"seed 必须为 int，实得 {type(self.seed).__name__}")
        if self.dtype not in ALLOWED_DTYPES:
            raise ConfigurationError(f"dtype={self.dtype!r} 非法，允许 {ALLOWED_DTYPES}")
        if self.device not in ALLOWED_DEVICES:
            raise ConfigurationError(f"device={self.device!r} 非法，允许 {ALLOWED_DEVICES}")
        for axis in _COMPONENT_AXES:
            val = getattr(self.components, axis)
            legal = self.ablation_matrix.get(axis)
            if legal is not None and val not in legal:
                raise ConfigurationError(
                    f"components.{axis}={val!r} 非法（不在 ablation_matrix.{axis}={legal} 内）"
                )
