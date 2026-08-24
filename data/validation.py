"""数据 fail-fast 校验层 —— 坏数据「正确断开」，不进入实验。

统一入口：任何数据（段 / 转移对 / 相位空间张量）在进入估计器/实验前先过这里的谓词；
违反结构性不变量即 raise DataValidationError，避免坏数据静默毁掉实验。

校验分级（与 loader 的既有 filter 语义分工）：
  - 结构性损坏（NaN/Inf、形状不一致、t 非递增、dt≤0）→ 硬失败 raise；
  - 越界/空段（dt 超范围、段长<2）→ 由调用方按现有 filter+计数语义处理，不在此 raise。

消息统一含「来源 + 字段 + 违规点（首例索引/值）+ 建议动作」。
"""

from __future__ import annotations

from typing import Any

import torch

from domain import DataValidationError


def _first_bad(mask: torch.Tensor) -> int:
    """首个 True 的行索引（1D/2D 通用），无 True 返回 -1。"""
    idx = torch.nonzero(mask, as_tuple=False)
    return int(idx[0, 0]) if idx.numel() else -1


def ensure_finite(x: Any, *, name: str = "array", source: str = "") -> None:
    """x 必须全有限（无 NaN/Inf）。接受 torch 张量或可转为张量的数组。"""
    x = torch.as_tensor(x)
    if torch.isfinite(x).all():
        return
    bad = _first_bad(~torch.isfinite(x))
    raise DataValidationError(
        f"{source} {name} 含 NaN/Inf（首例行索引 {bad}）—— 错误数据，中止，勿进入实验"
    )


def ensure_monotonic_increasing(t: Any, *, name: str = "t", source: str = "") -> None:
    """t 严格递增（相邻差 > 0），否则时间戳非法。"""
    t = torch.as_tensor(t)
    dt = t[1:] - t[:-1]
    if torch.any(dt <= 0):
        bad = _first_bad(dt <= 0)
        raise DataValidationError(
            f"{source} {name} 非严格递增（首例非正差分索引 {bad}）—— 时间戳非法，中止"
        )


def ensure_shapes_match(a: Any, b: Any, *, name_a: str = "a", name_b: str = "b",
                        source: str = "") -> None:
    """a、b 首维长度一致。"""
    if len(a) != len(b):
        raise DataValidationError(
            f"{source} {name_a}({len(a)}) 与 {name_b}({len(b)}) 长度不一致—— 中止"
        )


def ensure_positive(v: Any, *, name: str = "v", source: str = "") -> None:
    """v 全 > 0（标量或 1D 张量）。"""
    v = torch.as_tensor(v)
    if torch.all(v > 0):
        return
    bad = _first_bad(v <= 0) if v.ndim >= 1 else 0
    raise DataValidationError(
        f"{source} {name} 含非正值（首例索引 {bad}）—— 中止"
    )


def validate_segment(seg: Any, *, source: str = "") -> None:
    """段级结构不变量：长度≥2、x/t 有限、t 严格递增、dt>0。

    seg 只需具备 .x / .t / .dt 属性（duck-typing，避免与 loader 循环导入）。
    """
    x, t = seg.x, seg.t
    ensure_shapes_match(x, t, name_a="x", name_b="t", source=source)
    if len(x) < 2:
        raise DataValidationError(f"{source} 段长 {len(x)} < 2 —— 不足以构成转移，中止")
    ensure_finite(x, name="x", source=source)
    ensure_finite(t, name="t", source=source)
    ensure_monotonic_increasing(t, name="t", source=source)
    if float(seg.dt) <= 0:
        raise DataValidationError(f"{source} dt={seg.dt} ≤ 0 —— 中止")


def validate_phase_space(z: Any, *, source: str = "") -> None:
    """相位空间张量 (T+1, d)：至少 2 行、有限。供 EM 等估计器入口校验。"""
    if z.ndim != 2 or z.shape[0] < 2:
        raise DataValidationError(
            f"{source} 形状 {tuple(z.shape)} 非法（需 (T+1, d), T≥1）—— 中止"
        )
    ensure_finite(z, name="z", source=source)


def validate_transitions(x: Any, y: Any, dt: Any, *, source: str = "") -> None:
    """一步转移对不变量：x/y/dt 首维一致、有限、dt>0。"""
    ensure_shapes_match(x, y, name_a="x", name_b="y", source=source)
    ensure_shapes_match(x, dt, name_a="x", name_b="dt", source=source)
    ensure_finite(x, name="x", source=source)
    ensure_finite(y, name="y", source=source)
    ensure_finite(dt, name="dt", source=source)
    ensure_positive(dt, name="dt", source=source)
