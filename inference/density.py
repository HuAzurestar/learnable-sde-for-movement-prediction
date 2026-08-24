"""J-2 Fokker--Planck density evolution solver.

Fokker-Planck ∂ₜρ = L*ρ，守恒型有限体积：
  - 1D OU（收敛阶门用）: ∂ρ/∂t = −∂(aρ)/∂x + D∂²ρ/∂x²，a(x)=−Γx，D=σ²/2
    * 平流: Lax-Wendroff（二阶 O(h²)）; 扩散: 中心差分; 反射零通量边界
  - 2D 相空间（接口/骨架）: 欠阻尼朗之万生成元，Strang 分裂
已验（原论文）: O(h²) 收敛（实测 1.7–2.8）; 2D 相空间等 wall-time 优于 MC 一个量级。
约束: 生成元逐点正确是前提（推论 3）; 不变测度仅作长时程一致性诊断 gate。
"""

from __future__ import annotations

import torch


def _pad_reflect(u: torch.Tensor, n: int = 1) -> torch.Tensor:
    """1D 反射（零通量）边界填充: [u_0,u_1,...] → [u_n..u_1, u, u_{end-1}..u_{end-n}]。"""
    return torch.cat([u[:n].flip(0), u, u[-n:].flip(0)])


def solve_fp_1d(rho0: torch.Tensor, Gamma: float, sigma: float, T: float,
                xmin: float = -8.0, xmax: float = 8.0, n_sub: int = 1) -> torch.Tensor:
    """1D OU FP: 从 rho0 演化到 T。rho0: (nx,)。返回 (nx,) 密度（归一化）。

    守恒型 Lax-Wendroff（平流，二阶）+ 中心差分（扩散）+ 反射边界。
    时间步由稳定性驱动（平流 CFL + 扩散 CFL），n_sub 仅作最小时步数下限。
    """
    nx = rho0.shape[0]
    dx = (xmax - xmin) / (nx - 1)
    x = torch.linspace(xmin, xmax, nx, dtype=rho0.dtype)
    D = 0.5 * sigma ** 2
    a = -Gamma * x

    def _advect(rho: torch.Tensor, dt: float) -> torch.Tensor:
        """守恒型 Lax-Wendroff: ρ^{n+1}_i = ρ_i − (dt/2dx)(F_{i+1}−F_{i−1})
        + (dt²/2dx²)[a_{i+1/2}(F_{i+1}−F_i) − a_{i−1/2}(F_i−F_{i−1})], F=aρ
        界面速度 a_{i±1/2} 用反射填充的相邻平均（边界一致性）。"""
        F = a * rho
        F_p = _pad_reflect(F, 1)
        Fm = F_p[:-2]      # F_{i-1}
        F0 = F_p[1:-1]     # F_i
        Fp = F_p[2:]       # F_{i+1}
        a_p = _pad_reflect(a, 1)
        a_ph = 0.5 * (a_p[1:-1] + a_p[2:])   # a_{i+1/2}
        a_mh = 0.5 * (a_p[1:-1] + a_p[:-2])  # a_{i-1/2}
        rho_new = rho - (dt / (2 * dx)) * (Fp - Fm) \
            + (dt * dt / (2 * dx * dx)) * (a_ph * (Fp - F0) - a_mh * (F0 - Fm))
        return rho_new

    def _diffuse(rho: torch.Tensor, dt: float) -> torch.Tensor:
        lap = _pad_reflect(rho, 1)[:-2] - 2 * rho + _pad_reflect(rho, 1)[2:]
        return rho + D * dt / dx ** 2 * lap

    amax = Gamma * max(abs(xmin), abs(xmax))
    dt = min(0.5 * dx / (amax + 1e-9), 0.25 * dx * dx / (D + 1e-9))
    n_steps = max(int(T / dt), n_sub)
    dt = T / n_steps
    rho = rho0.clone()
    for _ in range(n_steps):
        rho = _advect(rho, dt)
        rho = _diffuse(rho, dt)
        rho = rho.clamp(min=0.0) * (rho0.sum() / (rho.sum() + 1e-15))
    return rho


class FPSolver:
    """2D 相空间 FP 求解器接口（欠阻尼朗之万生成元）。

    2D 全实现随 J-2 消融臂接线落地；当前提供接口 + 1D OU 收敛验证内核
    （solve_fp_1d），等价性回归门用 1D 内核测 O(h²) 收敛阶。
    """

    def __init__(self, bounds=(-8.0, 8.0, -6.0, 6.0), safety: float = 0.4):
        self.bounds = bounds
        self.safety = safety

    def solve(self, rho0: torch.Tensor, Gamma: float, omega2: float, sigma: float,
              T: float, n_steps: int = 200) -> torch.Tensor:
        """2D 占位实现：返回初始密度 + 标注。1D 收敛验证走 solve_fp_1d。"""
        return {"rho_T": rho0,
                "note": "2D FP 求解器随 J-2 消融臂接线；O(h²) 收敛验证见 gate_J2_fp_order (1D 内核)"}
