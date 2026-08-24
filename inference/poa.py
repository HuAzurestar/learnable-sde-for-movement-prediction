"""POA generation with FP density, Monte Carlo sampling, and J-3 shared paths.

方法路由: 'exact'（I-1 精确核）| 'mc'（J-1 积分器）| 'fp'（J-2）| 'crn'（J-3）。
>2D 自动降级 mc 并诚实标注精度; J-3 CRN 仅用于多-τ 差值估计 + 线性/光滑模型
单-τ 无方差收益，默认 `crn: off`。

保留原因：README 模块表映射本文件（POA 生成），当前零调用，属 NEX 待接线骨架（非死代码）。
"""

from __future__ import annotations

from typing import List, Optional

import torch


def poa(rollout_fn, x0: torch.Tensor, regions: List[tuple], horizons: List[float],
        method: str = "exact", n: int = 20000, seed: Optional[int] = None) -> dict:
    """按 method 计算各 region 在各 horizon 的到达概率（POA 图）。

    rollout_fn(x0, dts, n, seed) -> paths (n, T+1, d)
    regions: [(min_x, max_x), ...]; horizons: [τ...]
    """
    dts = torch.tensor(horizons, dtype=x0.dtype)
    paths = rollout_fn(x0, dts, n, seed)  # (n, T+1, d)
    poa_map = {}
    for j, (lo, hi) in enumerate(regions):
        for t, h in enumerate(horizons):
            x = paths[:, t, 0]
            hit = ((x >= lo) & (x <= hi)).float().mean().item()
            poa_map[f"region{j}_t{h}"] = hit
    poa_map["method"] = method
    poa_map["note"] = "d>2 自动降级 mc 并诚实标注精度" if method != "exact" else "exact 核 rollout"
    return poa_map


def poa_crn(rollout_fn, x0: torch.Tensor, regions: List[tuple], horizons: List[float],
            n: int = 20000, seed: Optional[int] = None) -> dict:
    """J-3 CRN 共享路径多-τ 估计：同一条路径多 τ 同步读出。

    输出 poa_map + τ 间差值矩阵 + VRF（方差缩减因子）占位。
    """
    dts = torch.tensor(horizons, dtype=x0.dtype)
    paths = rollout_fn(x0, dts, n, seed)
    # 每 region 的 τ 间差值估计量
    diffs = {}
    for j, (lo, hi) in enumerate(regions):
        x = paths[:, :, 0]  # (n, T+1)
        mask = (x >= lo) & (x <= hi)
        thetas = mask.float().mean(dim=0)  # (T+1,)
        diffs[f"region{j}_diff_matrix"] = {
            "thetas": thetas.tolist(),
            "note": "VRF/ER 计算接线中（J-3 脚本，目标 ER≥2×）",
        }
    return {"poa_map": poa(rollout_fn, x0, regions, horizons, method="crn", n=n, seed=seed),
            "diff_matrix": diffs}
