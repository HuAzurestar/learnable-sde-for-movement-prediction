"""C-7 neural SDE implementation placeholder.

NEX-381-v6 does not permit this skeleton to run.  Its future implementation is
frozen to normalized state ``[X,V]`` plus one normalized, manifest-aligned
window mean ``solar_elev`` input (never ``day_fraction``), two Tanh hidden layers, diagonal
``(ell_X, ell_V)`` diffusion, ``dt_scale=60 seconds`` and elementwise drift
clipping at 10.  See ``experiments/capacity_preregistration`` for the complete
optimizer, initialization, adapter and Euler-Maruyama contracts.
"""

from __future__ import annotations

import torch

from domain import ModelContext
from .base import ParameterGroupProvider, ParameterRole, SDEModel


class TimeVaryingNeuralSDE(SDEModel, ParameterGroupProvider):
    """C-7 神经 SDE 接口骨架；不声明精确转移能力。"""

    def __init__(self, d: int = 2, d_env: int = 1, hidden: list = (64, 64),
                 dt_scale: float = 60.0, drift_clip: float = 10.0):
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
        """Frozen future form: N(y; x+(dt/60)b, (dt/60)diag(exp(2 ell)))."""
        raise NotImplementedError("接线后实现")

    def parameter_groups(self):
        return {ParameterRole.DRIFT: (), ParameterRole.DIFFUSION: ()}
