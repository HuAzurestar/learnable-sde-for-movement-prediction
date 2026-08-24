"""C-5/C-7 one-step Gaussian NLL with a shared training/evaluation convention.

对 Euler 层一步 EM 转移 p̃(y|x,e)=N(y; x+Δt·b, Δt·diag(σ²)) 计算 NLL。
C-5 冻结 Σ 只调 b 的 drift-only 变体在 transfer/init.py 中通过 param_groups 实现。

保留原因：README「模块→契约→源论文」表映射本文件（C-5/C-7 一步高斯 NLL），当前零调用，
属 NEX 待接线骨架（非死代码）。
"""

from __future__ import annotations

import torch


def gaussian_transition_nll(x: torch.Tensor, y: torch.Tensor, drift: torch.Tensor,
                            logvar: torch.Tensor, dt: float) -> torch.Tensor:
    """一步条件高斯 NLL。drift/logvar: (...,d) 预测。"""
    mean = x + dt * drift
    var = torch.exp(logvar) * dt
    diff = y - mean
    nll = 0.5 * ((diff * diff / var).sum(-1) + (logvar + torch.log(dt)).sum(-1)
                 + mean.shape[-1] * torch.log(torch.tensor(2 * 3.141592653589793)))
    return nll.mean()
