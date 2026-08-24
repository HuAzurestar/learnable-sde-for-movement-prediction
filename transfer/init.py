"""C-5 two-stage fine-tuning.

Euler 层漂移-扩散解耦: 冻结 Σ_g 只调 b → 不可约残差纯扩散失配；全量微调消两者。
param_groups{drift, diffusion} 冻结/放行（无梯度泄漏）。
已验（MLP 神经 SDE）: 浙江全量微调 energy 3.772 反超退化 3.828, ~1.5%, CI 不跨零。
I-1 multi-regime transfer updates parameters per regime and therefore requires
its own ablation validation.

保留原因：README 模块表映射本文件（C-5 两步微调），当前 C5FineTuner 零调用，
属 NEX 待接线骨架（非死代码）。
"""

from __future__ import annotations

from typing import List

import torch

from domain import CapabilityError
from models.base import ParameterGroupProvider, ParameterRole


class C5FineTuner:
    """冻结/放行参数组的微调控制器。

    mode: 'drift_only'（只调 drift 组）| 'full'（drift+diffusion 全调）| 'none'
    """

    def __init__(self, model: ParameterGroupProvider, mode: str = "full"):
        if not isinstance(model, ParameterGroupProvider):
            raise CapabilityError(f"{type(model).__name__} 不提供参数分组能力")
        self.mode = mode
        self.param_groups = model.parameter_groups()

    def optimizer_params(self, lr: float = 1e-2) -> List[dict]:
        if self.mode == "none":
            return []
        if self.mode == "drift_only":
            return [{"params": self.param_groups[ParameterRole.DRIFT], "lr": lr}]
        if self.mode == "full":
            return [{"params": self.param_groups[ParameterRole.DRIFT], "lr": lr},
                    {"params": self.param_groups[ParameterRole.DIFFUSION], "lr": lr}]
        raise ValueError(f"unknown mode {self.mode}")
