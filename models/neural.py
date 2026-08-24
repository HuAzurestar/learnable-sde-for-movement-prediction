"""C-7 neural SDE skeleton with exogenous conditions in augmented state ``[x; e]``.

已验（神经参数化）: 连续 solar_elev 显著 −12.4% energy（配对 bootstrap CI 下界>0）；
二值 is_day 不显著；天气仅冒烟方向性。
本文件为接口骨架 —— EnvDriftNet + transition_nll + rollout 由后续实现
22 臂冒烟脚本接线时实现（condition: solar 臂）。
"""

from __future__ import annotations

import torch

from domain import ModelContext
from .base import ParameterGroupProvider, ParameterRole, SDEModel


class TimeVaryingNeuralSDE(SDEModel, ParameterGroupProvider):
    """C-7 神经 SDE 接口骨架；不声明精确转移能力。"""

    def __init__(self, d: int = 2, d_env: int = 1, hidden: list = (64, 64),
                 dt_scale: float = 1.0, drift_clip: float = 10.0):
        super().__init__()
        self.d = d
        self.d_env = d_env
        self.hidden = hidden
        self.dt_scale = dt_scale
        self.drift_clip = drift_clip
        self._trained = False

    @property
    def state_dim(self) -> int:
        return self.d

    @property
    def noise_dim(self) -> int:
        return self.d

    def drift(self, t, x, context: ModelContext):
        raise NotImplementedError("C-7 neural drift 由 22 臂冒烟脚本接线")

    def diffusion(self, t, x, context: ModelContext):
        raise NotImplementedError

    def transition_nll(self, x, y, dt, cond=None):
        """一步 EM 转移 p̃(y|x,e)=N(y; x+(dt/τ)b, (dt/τ)diag(σ²))（训练=评估，同口径）。"""
        raise NotImplementedError("接线后实现")

    def parameter_groups(self):
        return {ParameterRole.DRIFT: (), ParameterRole.DIFFUSION: ()}
