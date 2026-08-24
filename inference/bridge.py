"""I-2 bridge conditioning as an optional inference component.

h-transform 只改模拟漂移（加 a∇log h），接口层仍 (drift, diffusion)，
FP/积分器照常工作。Doob/SB/软终点三模式，P0#2 终点条件化部署。

保留原因：README 模块表映射本文件（I-2 桥条件化），当前零调用，属 NEX 待接线骨架（非死代码）。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from domain import ModelContext


class BridgeConditioning:
    """终点条件化桥。mode: 'doob' | 'soft'（SB 基扩散升级另立 issue）。"""

    def __init__(self, sde, mode: str = "doob", terminal: Optional[torch.Tensor] = None,
                 sigma_t: float = 1.0, sde_mode: int = 0):
        self.sde = sde
        self.mode = mode
        self.terminal = terminal
        self.sigma_t = sigma_t
        self.sde_mode = sde_mode  # 段模式 k（显式，非隐藏状态）

    def conditioned_drift(self, x: torch.Tensor, t_left: float) -> torch.Tensor:
        """加 a∇log h 的桥漂移（Doob h-transform）。"""
        context = ModelContext(regime=self.sde_mode)
        if self.terminal is None:
            return self.sde.drift(torch.zeros((), dtype=x.dtype), x, context)
        # 线性终点势: h ∝ exp(−‖x−x_T‖²/(2σ²τ)) 的梯度（τ = 剩余时间）
        tau = max(t_left, 1e-3)
        grad = -(x - self.terminal) / (self.sigma_t ** 2 * tau)
        return self.sde.drift(torch.zeros((), dtype=x.dtype), x, context) + grad
