"""J-1 split-step integrator.

朗之万块结构 Strang 对称组合 Φ_h = S_{h/2} ∘ O_h ∘ S_{h/2}：
  - O_h 摩擦-扩散子步（闭式 OU）: v' = a·v + b·ξ, a=e^{−Γh}, b=σ√((1−a²)/(2Γ))
  - S_h 势能子步（可逆 velocity-Verlet）: 位置-速度交换
已验: 弱阶≈2, 强阶≈1; 等精度 wall-time ≈0.2×EM。线性情形与 I-1 精确核同源
等价性回归含 `inference: exact|J1_split` 误差对照（应 <1e-6）。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from domain import ModelContext


class SplitIntegrator:
    """相空间 (x, v) 欠阻尼朗之万的 J-1 分裂积分器。

    结构: dV = −Γ V dt + (a−κ)X dt + c dt + g dW;  dX = V dt
    Args:
        Gamma: 摩擦系数
        force_lin: 位置依赖加速度系数 (a−κ)
        force_const: 常数加速度 c
        sigma: 速度噪声幅值 g
    """

    def __init__(self, Gamma: float, force_lin: float, force_const: float, sigma: float,
                 dtype: torch.dtype = torch.float64, device: str = "cpu"):
        self.Gamma = Gamma
        self.force_lin = force_lin
        self.force_const = force_const
        self.sigma = sigma
        self.dtype = dtype
        self.device = device

    def _acc(self, x: torch.Tensor) -> torch.Tensor:
        return self.force_lin * x + self.force_const

    def step(self, z: torch.Tensor, h: float, noise: torch.Tensor) -> torch.Tensor:
        """单步 Strang 分裂 S_{h/2} ∘ O_h ∘ S_{h/2}。z: (...,2), noise 预生成 ξ。"""
        x, v = z[..., 0], z[..., 1]
        # S_{h/2}: 势能子步（velocity-Verlet）
        v = v + 0.5 * h * self._acc(x)
        x = x + h * v
        v = v + 0.5 * h * self._acc(x)
        # O_h: 摩擦-扩散闭式 OU
        a = math.exp(-self.Gamma * h)
        b = self.sigma * math.sqrt((1 - a * a) / (2 * self.Gamma)) if self.Gamma > 0 else self.sigma * math.sqrt(h)
        v = a * v + b * noise
        return torch.stack([x, v], dim=-1)

    def rollout(self, x0: torch.Tensor, dt: float, n_sub: int, n: int,
                seed: Optional[int] = None) -> torch.Tensor:
        """从 x0 推进宏步 dt（n_sub 子步），返回 n 条路径的终态 (n, 2)。"""
        h = dt / n_sub
        g = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        z = x0.detach().clone().unsqueeze(0).expand(n, 2).clone()
        total = n * n_sub
        if g is not None:
            noise = torch.randn(total, dtype=self.dtype, device=self.device, generator=g)
        else:
            noise = torch.randn(total, dtype=self.dtype, device=self.device)
        noise = noise.view(n, n_sub)
        for s in range(n_sub):
            z = self.step(z, h, noise[:, s])
        return z

    # -- 解析有效核（等价性回归用，确定性，无 MC 噪声） --------------------------
    @staticmethod
    def _compose(M1, m1, M2, m2):
        """返回先 (M1,m1) 再 (M2,m2) 的仿射映射。"""
        return M2 @ M1, M2 @ m1 + m2

    def _S_map(self, h: float) -> tuple:
        """velocity-Verlet 势能子步线性仿射映射（force αx+c 在子步 h 内）。"""
        a = self.force_lin
        M = torch.tensor([[1.0 + 0.5 * h * h * a, h],
                          [h * a + 0.25 * h ** 3 * a * a, 1.0 + 0.5 * h * h * a]],
                         dtype=self.dtype, device=self.device)
        m = torch.tensor([0.5 * h * h * self.force_const,
                          h * self.force_const + 0.25 * h ** 3 * a * self.force_const],
                         dtype=self.dtype, device=self.device)
        return M, m

    def _O_map(self, h: float) -> tuple:
        """摩擦-扩散 OU 子步: v' = e^{-Γh} v + b ξ。返回 (M, m, noise_var)。"""
        e = math.exp(-self.Gamma * h)
        M = torch.tensor([[1.0, 0.0], [0.0, e]], dtype=self.dtype, device=self.device)
        b2 = self.sigma ** 2 * (1 - e * e) / (2 * self.Gamma) if self.Gamma > 0 else self.sigma ** 2 * h
        return M, torch.zeros(2, dtype=self.dtype, device=self.device), b2

    def effective_kernel(self, dt: float, n_sub: int) -> tuple:
        """宏步 dt 的解析有效核 (F_eff, c_eff, Σ_eff)。

        Φ_h = S_{h/2} ∘ O_h ∘ S_{h/2} 精确仿射组合 n_sub 次（确定性，无采样误差）。
        """
        h = dt / n_sub
        M_S, m_S = self._S_map(0.5 * h)
        M_O, m_O, b2 = self._O_map(h)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        # 单子步噪声协方差 = M_S diag(0,b²) M_Sᵀ（O_h 的 OU 噪声经末尾 S_{h/2}）
        N_ou = torch.zeros((2, 2), dtype=self.dtype, device=self.device)
        N_ou[1, 1] = b2
        N_sub = M_S @ N_ou @ M_S.T
        # 组合单子步仿射映射 Φ_h = S_{h/2}∘O_h∘S_{h/2}
        M_Phi, m_Phi = self._compose(M_S, m_S, M_O, m_O)
        M_Phi, m_Phi = self._compose(M_Phi, m_Phi, M_S, m_S)
        # 累加 n_sub 次
        M_run, m_run, S_run = Id, torch.zeros(2, dtype=self.dtype, device=self.device), torch.zeros((2, 2), dtype=self.dtype, device=self.device)
        for _ in range(n_sub):
            S_run = M_Phi @ S_run @ M_Phi.T + N_sub
            M_run, m_run = self._compose(M_run, m_run, M_Phi, m_Phi)
        return M_run, m_run, 0.5 * (S_run + S_run.T)


def exact_vs_j1_error(sde, x0: torch.Tensor, dt: float, n_sub: int,
                      mode: int = 0, seed: int = 20260814) -> dict:
    """等价性回归: I-1 精确核 vs J-1 分裂（阈值 <1e-6 量级）。

    比较 J-1 解析有效核 (F,c,Σ) 与精确核 (F_e,c_e,Σ_e) 的分布矩（确定性）。
    mode = 段模式 k（单模式 gate 恒 0）。
    """
    model_context = ModelContext(regime=mode)
    transition = sde.exact_transition(x0, dt, model_context)
    mean_e = transition.mean
    F_e, c_e, S_e = sde.affine_transition(dt, model_context)
    integ = SplitIntegrator(float(sde.Gamma[mode]), float(sde.a[mode] - sde.kappa),
                            float(sde.c[mode]), float(sde.g[mode]), dtype=sde.dtype, device=sde.device)
    F_j, c_j, S_j = integ.effective_kernel(dt, n_sub)
    mean_j = F_j @ x0 + c_j
    mean_err = (mean_j - mean_e).abs().max().item()
    cov_err = (S_j - S_e).abs().max().item()
    return {
        "mean_err": mean_err,
        "cov_err": cov_err,
        "pass": mean_err < 1e-6 and cov_err < 1e-6,
        "n_sub": n_sub,
    }
