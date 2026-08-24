"""CPU smoke tests for the numerical kernels (target runtime: at most 10 minutes).

核级无损门（等价性回归的 kernel 层）:
  G1 精确高斯核 vs 标量 OU 闭式（F, Σ 逐一对齐）
  G2 discrete_to_continuous 往返（标量 OU 恢复 Γ, σ）
  G3 J-1 分裂积分器 vs 精确核（<1e-6）
  G4 CRPS d=1 闭式 vs MC 能量分
  G5 I-1 EM 合成双模式数据模式恢复（可辨识冒烟）
运行: python -m experiments.smoke_test
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.segment_constant import SegmentConstantSDE, _matrix_logm
from domain import ModelContext
from inference.base import ExactGaussianEngine
from inference.integrator import SplitIntegrator, exact_vs_j1_error
from estimation.score import crps_gaussian, energy_score_mc
from estimation.base import FitContext
from estimation.em import SegmentEMData


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def make_synthetic_segments(n_seg: int = 12, len_per_seg: int = 120, dt: float = 60.0,
                            seed: int = 20260814) -> tuple:
    """双模式段级欠阻尼朗之万: 模式 0 低扩散往返, 模式 1 高扩散慢漂移。"""
    torch.manual_seed(seed)
    params = [
        {"Gamma": 0.05, "a": -0.02, "c": 0.0, "g": 0.12},   # 低扩散，向原点
        {"Gamma": 0.01, "a": 0.0, "c": 0.0, "g": 0.35},     # 高扩散
    ]
    segments, true_modes = [], []
    for s in range(n_seg):
        k = s % 2
        p = params[k]
        sde = SegmentConstantSDE(n_modes=1, kappa=0.0, dt_ref=dt)
        sde.set_regime_parameters(
            0,
            gamma=p["Gamma"],
            linear_drift=p["a"],
            constant_drift=p["c"],
            diffusion=p["g"],
        )
        x0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        dts = torch.full((len_per_seg - 1,), dt, dtype=torch.float64)
        seg = ExactGaussianEngine().rollout(
            sde,
            x0,
            dts,
            n_samples=1,
            model_context=ModelContext(regime=0),
            generator=torch.Generator().manual_seed(seed + s),
        ).squeeze(0)
        segments.append(seg)
        true_modes.append(k)
    return segments, true_modes


# ---------------------------------------------------------------------------
# G1 精确核 vs 标量 OU 闭式
# ---------------------------------------------------------------------------
def gate_exact_kernel_vs_ou() -> dict:
    """标量 OU dX = −ΓX dt + σ dW 的 2×2 相空间嵌入。F, Σ 应逐一对齐闭式。"""
    Gamma, sigma, dt = 0.02, 0.15, 60.0
    sde = SegmentConstantSDE(n_modes=1, kappa=0.0)
    sde.set_regime_parameters(0, gamma=Gamma, linear_drift=0.0,
                              constant_drift=0.0, diffusion=sigma)
    x = torch.tensor([1.0, -0.5], dtype=torch.float64)
    transition = sde.exact_transition(x, dt, ModelContext(regime=0))
    mean, S = transition.mean, transition.covariance
    # 闭式（欠阻尼 OU 相空间）: F = [[1, (1-e^{-Γdt})/Γ],[0, e^{-Γdt}]]
    e = math.exp(-Gamma * dt)
    F_cl = torch.tensor([[1.0, (1 - e) / Gamma], [0.0, e]], dtype=torch.float64)
    mean_cl = F_cl @ x
    # Σ 闭式: Var(V)=σ²(1-e^{-2Γdt})/(2Γ); Cov(X,V)=σ²(1-e^{-Γdt})²/(2Γ²); Var(X)=σ²(2Γdt−3+4e−e²)/(2Γ³)
    varV = sigma ** 2 * (1 - e * e) / (2 * Gamma)
    covXV = sigma ** 2 * (1 - e) ** 2 / (2 * Gamma ** 2)
    varX = sigma ** 2 * (2 * Gamma * dt - 3 + 4 * e - e * e) / (2 * Gamma ** 3)
    S_cl = torch.tensor([[varX, covXV], [covXV, varV]], dtype=torch.float64)
    mean_err = (mean - mean_cl).abs().max().item()
    cov_err = (S - S_cl).abs().max().item()
    return {"gate": "G1 exact-kernel-vs-OU", "mean_err": mean_err, "cov_err": cov_err,
            "pass": mean_err < 1e-8 and cov_err < 1e-8}


# ---------------------------------------------------------------------------
# G2 discrete_to_continuous 往返
# ---------------------------------------------------------------------------
def gate_discrete_to_continuous() -> dict:
    Gamma, sigma, dt = 0.03, 0.2, 60.0
    sde = SegmentConstantSDE(n_modes=1)
    sde.set_regime_parameters(0, gamma=Gamma, diffusion=sigma)
    F, c, S = sde.affine_transition(dt, ModelContext(regime=0))
    A_rec, b_rec, B_rec = sde.discrete_to_continuous(F, c, S, dt)
    g_rec = B_rec[1, 0].item()
    Gamma_rec = -A_rec[1, 1].item()
    return {"gate": "G2 discrete-to-continuous", "Gamma_rec": Gamma_rec, "g_rec": g_rec,
            "Gamma_err": abs(Gamma_rec - Gamma), "g_err": abs(g_rec - sigma),
            "pass": abs(Gamma_rec - Gamma) < 0.05 and abs(g_rec - sigma) < 0.05}


# ---------------------------------------------------------------------------
# G3 J-1 split vs exact
# ---------------------------------------------------------------------------
def gate_j1_vs_exact() -> dict:
    sde = SegmentConstantSDE(n_modes=1)
    sde.set_regime_parameters(0, gamma=0.02, linear_drift=-0.01,
                              constant_drift=0.001, diffusion=0.15)
    x0 = torch.tensor([2.0, 1.0], dtype=torch.float64)
    r = exact_vs_j1_error(sde, x0, dt=60.0, n_sub=32768, seed=20260814)
    r["gate"] = "G3 J1-split-vs-exact"
    return r


# ---------------------------------------------------------------------------
# G4 CRPS 闭式 vs MC（标准 CRPS: E|Z−y| − ½E|Z−Z'|，与 C1_paper 口径一致）
# ---------------------------------------------------------------------------
def gate_crps_closed_vs_mc() -> dict:
    mu, sigma = torch.tensor(0.0), torch.tensor(1.0)
    y = torch.tensor(0.3)
    closed = crps_gaussian(mu, sigma, y)
    torch.manual_seed(0)
    m = 400000
    Z = torch.randn(m)
    Zp = torch.randn(m)
    mc = (Z - y).abs().mean() - 0.5 * (Z - Zp).abs().mean()
    return {"gate": "G4 CRPS-closed-vs-MC", "closed": closed.item(), "mc": mc.item(),
            "err": abs(closed.item() - mc.item()), "pass": abs(closed.item() - mc.item()) < 0.01}


# ---------------------------------------------------------------------------
# G5 EM 模式恢复
# ---------------------------------------------------------------------------
def gate_em_mode_recovery() -> dict:
    segments, true_modes = make_synthetic_segments(n_seg=12, len_per_seg=120, dt=60.0)
    sde = SegmentConstantSDE(n_modes=2)
    from estimation.em import SegmentEM
    em = SegmentEM(max_iter=30)
    summary = em.fit(
        sde,
        SegmentEMData.uniform(segments, 60.0),
        FitContext(torch.Generator().manual_seed(20260814), torch.device("cpu"), torch.float64),
    )
    # 段级后验主导模式 vs 真值（模式标签可置换，取最佳置换准确率）
    doms = []
    for i, z in enumerate(segments):
        post = sde.segment_posterior(z, 60.0)
        doms.append(int(post.argmax()))
    n_mode = sde.n_modes
    best = 0.0
    for perm in _permutations(n_mode):
        acc = sum(1 for d, t in zip(doms, true_modes) if perm[d] == t) / len(doms)
        best = max(best, acc)
    return {"gate": "G5 EM-mode-recovery", "acc": best, "converged": summary.converged,
            "pass": best >= 0.75}


def _permutations(n: int):
    """0..n-1 的全排列（n 小，本门限 n≤3）。"""
    import itertools
    return list(itertools.permutations(range(n)))


def main():
    gates = [
        gate_exact_kernel_vs_ou(),
        gate_discrete_to_continuous(),
        gate_j1_vs_exact(),
        gate_crps_closed_vs_mc(),
        gate_em_mode_recovery(),
    ]
    all_pass = True
    print("=== learnable_sde 冒烟 (kernel 层无损门) ===")
    for g in gates:
        ok = g["pass"]
        all_pass &= ok
        detail = {k: v for k, v in g.items() if k not in ("gate", "pass")}
        print(f"[{'PASS' if ok else 'FAIL'}] {g['gate']} {detail}")
    print("=== 结果:", "ALL PASS" if all_pass else "SOME FAILED", "===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
