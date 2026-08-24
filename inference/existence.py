"""Existence conditioning for positive information (E4--E7).

E4--E7 form the dual/generalization of the exclusion bridge (E1--E3/E0):

  失联人员运动 = R^d 上 Itô 扩散（I-1 段级常模式欠阻尼朗之万，状态 z=[X,V]）
    dX = V dt
    dV = [−Γ V + (a−κ)X + c] dt + g dW ,    a := σσᵀ = diag(0, g²)
  搜索域 D⊂R（位置），目标区域 A（位置区间之并），首达时 τ_A = inf{t≥0: X_t∈A}。

  存在事件（正信息「专家认定：营火/行走痕迹，人员有大概率出现过」）：
    E := {τ_A ∈ [t_1, t_2]} = {∃ s∈[t_1,t_2]: X_s ∈ A} 。

  E4   存在概率 h_exist(t,x) = P(τ_A ∈ [t,t_2] | X_t=x) 是 Dirichlet 1 后向方程解
         ∂_t h + L h = 0,  h|_{[t1,t2]×∂A}=1,  h(t_2,·)=1_A；先验存在概率 π := P(E)。
        本模块用 Feynman–Kac MC 数值解（从 z 前向模拟 I-1 精确核路径，数「窗内命中 A」比例）。
  E4b  存在加权密度 p_exist(t,x) = h_exist(t,x)·p(t,x)/π（硬存在桥，E2 的对偶）。
  E5   软证据 = 加权存在约束（mixture 定理）：观测 Y=1（误报率 α、漏报率 β）后
         Q = w·P(·|E) + (1−w)·P(·|E^c),   w = ρπ/(ρπ+1−π),  ρ=(1−β)/α。
        —— 单 h-变换（推论 E5c）仅「窗开启前/命中前」成立；mixture 才是全局正确描述。
  E6   误判鲁棒性：后验只依赖 λ=log ρ，TV-Lipschitz ‖Q(λ)−Q(λ')‖_TV ≤ ½|λ−λ'|。
  对偶  全时域 [0,T] 下 h_exist + h_excl ≡ 1；π·P(·|E)+(1−π)·P(·|E^c)=P（先验 2-点混合分解）。

**正确性口径（关键）**：终端时刻的条件边际密度 = 路径级拒绝采样（rejection）——
模拟 N 条路径，保留「窗内命中 A」者（存在桥）或「窗内未命中 A」者（非存在桥），
其终端经验分布 = 精确 MC 估计。硬存在基线（E0-对照）= 只用终端信息截断到 A 再重归一，
规范化常数 P(X_T∈A)；二者差 = 「命中 A 又离开」的路径穿越质量（与排除桥 M5 对偶）。

复用 `inference/exclusion.py` 的 `simulate_paths` / `survival_mc` / `AbsorbingExclusionConditioning`；
与 `bridge.BridgeConditioning`（doob/soft h-变换接口）同族，h 换为存在势（Dirichlet 1）数值解。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from models.segment_constant import SegmentConstantSDE
from numerics import safe_cholesky

from .exclusion import (
    AbsorbingExclusionConditioning,
    mixture_pdf,
    mode_kernel,
    path_occupancy,
    simulate_paths,
    terminal_gaussian_mixture,
)


# ---------------------------------------------------------------------------
# 命中窗内区域（E4 存在事件判定）
# ---------------------------------------------------------------------------
def _hit_in_window(paths: torch.Tensor, regions: Sequence[Tuple[float, float]],
                   t1: float, t2: float, dt: float) -> torch.Tensor:
    """(n_paths,) bool：路径在时间窗 [t1,t2] 内曾进入 regions 之并。

    paths: (n_paths, n_steps+1, 2)；索引 s 对应时刻 s·dt（traj[0]=z0 at t=0）。
    """
    n_steps = paths.shape[1] - 1
    X = paths[..., 0]  # (n_paths, n_steps+1)
    s_min = max(int(math.ceil(t1 / dt)), 0)
    s_max = min(int(math.floor(t2 / dt)), n_steps)
    if s_min > s_max:
        return torch.zeros(X.shape[0], dtype=torch.bool, device=paths.device)
    Xw = X[:, s_min:s_max + 1]
    hit = torch.zeros(X.shape[0], dtype=torch.bool, device=paths.device)
    for (lo, hi) in regions:
        hit = hit | ((Xw >= lo) & (Xw <= hi)).any(dim=1)
    return hit


def _terminal_hist(X: torch.Tensor, cell_edges: np.ndarray) -> np.ndarray:
    """终端位置经验直方图（归一化到 Σ=1）。"""
    hist, _ = np.histogram(X, bins=cell_edges)
    hist = hist.astype(np.float64)
    return hist / (hist.sum() + 1e-12)


# ---------------------------------------------------------------------------
# E4 存在概率（Feynman–Kac MC 数值解，Dirichlet 1）
# ---------------------------------------------------------------------------
def existence_mc(sde: SegmentConstantSDE, z0: torch.Tensor,
                 regions: Sequence[Tuple[float, float]], t1: float, t2: float,
                 horizon: float, dt: float, n_paths: int,
                 seed: Optional[int] = None, per_mode: bool = False):
    """P(τ_A ∈ [t1,t2] | Z_0=z0) 的 MC 估计（存在势 h_exist 的数值解）。

    与 survival_mc（E1 排除）对偶：full-horizon [0,T] 下 existence_mc + survival_mc = 1。
    """
    n_steps = max(int(round(horizon / dt)), 1)
    s_min = max(int(math.ceil(t1 / dt)), 0)
    s_max = min(int(math.floor(t2 / dt)), n_steps)
    hits = []
    for k in range(sde.n_modes):
        F, c, S = mode_kernel(sde, k, dt)
        L = safe_cholesky(S)
        g = torch.Generator(device=sde.device)
        if seed is not None:
            g.manual_seed(seed + k * 100003)
        z = z0.detach().clone().to(sde.dtype).unsqueeze(0).expand(n_paths, 2).clone()
        hit = torch.zeros(n_paths, dtype=torch.bool, device=sde.device)
        for s in range(n_steps):
            z = z @ F.T + c + torch.randn(n_paths, 2, dtype=z.dtype, device=sde.device, generator=g) @ L.T
            if s_min <= s <= s_max:
                x = z[:, 0]
                in_any = torch.zeros(n_paths, dtype=torch.bool, device=sde.device)
                for (lo, hi) in regions:
                    in_any = in_any | ((x >= lo) & (x <= hi))
                hit = hit | in_any
        hits.append(float(hit.float().mean()))
    if per_mode:
        return torch.tensor(hits, dtype=torch.float64)
    pi = torch.softmax(sde.prior_logits, dim=0)
    return float(sum(pi[k].item() * hits[k] for k in range(sde.n_modes)))


# ---------------------------------------------------------------------------
# 核心类：硬存在桥（E4）+ 软证据混合（E5/E6）
# ---------------------------------------------------------------------------
class ExistenceConditioning:
    """存在条件化（正信息）。E4 存在概率 + E4b 存在加权密度 + E0 对照基线。

    与 `exclusion.AbsorbingExclusionConditioning` 对偶：h 由吸收边界 Dirichlet 0 换为
    命中边界 Dirichlet 1；full-horizon [0,T] 时 h_exist + h_excl ≡ 1。
    """

    def __init__(self, sde: SegmentConstantSDE, region: Tuple[float, float],
                 horizon: float, dt: float = 60.0, t1: Optional[float] = None,
                 t2: Optional[float] = None, n_paths: int = 20000,
                 seed: Optional[int] = None):
        self.sde = sde
        self.region = (float(min(region)), float(max(region)))
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.t1 = 0.0 if t1 is None else float(t1)
        self.t2 = float(horizon) if t2 is None else float(t2)
        self.n_paths = int(n_paths)
        self.seed = seed

    # -- E4 -----------------------------------------------------------------
    def existence_prob(self, z0: torch.Tensor, n_paths: Optional[int] = None,
                       seed: Optional[int] = None) -> float:
        """π = P(E) = P(τ_A ∈ [t1,t2] | Z_0=z0)（存在势数值解，混合口径）。"""
        return existence_mc(self.sde, z0, [self.region], self.t1, self.t2,
                            self.horizon, self.dt, n_paths or self.n_paths,
                            seed if seed is not None else self.seed)

    # -- E4b 黄金口径：路径级拒绝采样终端密度 ---------------------------------
    def rejection_terminal(self, z0: torch.Tensor, x_grid: torch.Tensor,
                           n_paths: Optional[int] = None, seed: Optional[int] = None
                           ) -> Tuple[np.ndarray, float]:
        """P(·|E) 的终端经验密度（黄金口径）+ 存在概率 π。

        模拟 n_paths 条路径，保留窗内命中 A 者，其终端 X 归一化直方图 = 硬存在桥
        条件化终端密度的 MC 估计。返回 (密度数组, π)。
        """
        n_paths = n_paths or self.n_paths
        seed = seed if seed is not None else self.seed
        paths = simulate_paths(self.sde, z0, self.horizon, self.dt, n_paths, seed)
        hit = _hit_in_window(paths, [self.region], self.t1, self.t2, self.dt)
        pi = float(hit.float().mean())
        X = paths[:, -1, 0].detach().numpy()
        if hit.sum() == 0:
            return np.zeros(len(x_grid) - 1), pi
        return _terminal_hist(X[hit.numpy()], x_grid.numpy()), pi

    def non_existence_terminal(self, z0: torch.Tensor, x_grid: torch.Tensor,
                               n_paths: Optional[int] = None, seed: Optional[int] = None
                               ) -> np.ndarray:
        """P(·|E^c) 的终端经验密度（窗内未命中 A 的路径；full-horizon 时 = 排除桥）。"""
        n_paths = n_paths or self.n_paths
        seed = seed if seed is not None else self.seed
        paths = simulate_paths(self.sde, z0, self.horizon, self.dt, n_paths, seed)
        hit = _hit_in_window(paths, [self.region], self.t1, self.t2, self.dt)
        X = paths[:, -1, 0].detach().numpy()
        nonhit = ~hit.numpy()
        if nonhit.sum() == 0:
            return np.zeros(len(x_grid) - 1)
        return _terminal_hist(X[nonhit], x_grid.numpy())

    # -- E0 对照基线 ---------------------------------------------------------
    def hard_existence(self, z0: torch.Tensor, x_grid: torch.Tensor) -> np.ndarray:
        """E0 对照基线：只用终端信息截断到 A 再重归一（规范化常数 P(X_T∈A)）。

        忽略路径穿越（「命中 A 又离开」），与 E4b 路径级存在的差 = 路径穿越质量。
        """
        mus, sigs, pis = terminal_gaussian_mixture(self.sde, z0, self.horizon, self.dt)
        centers = 0.5 * (x_grid[:-1] + x_grid[1:])
        p = mixture_pdf(centers, mus, sigs, pis)
        inside = (centers >= self.region[0]) & (centers <= self.region[1])
        out = p.numpy().copy()
        out[~inside] = 0.0
        return out / (out.sum() + 1e-12)


class SoftExistenceConditioning:
    """软证据存在条件化（E5 mixture 定理）。

    alpha = 误报率 P(Y=1|E^c)（把他人/动物痕迹误认为失联人员痕迹）；
    beta  = 漏报率 P(Y=0|E)。观测 Y=1 后：
        Q = w·P(·|E) + (1−w)·P(·|E^c),  w = ρπ/(ρπ+1−π),  ρ = (1−β)/α。
    单标量 λ=log ρ 刻画（E6）：后验只依赖 λ，TV-Lipschitz ≤ ½|Δλ|。
    """

    def __init__(self, existence: ExistenceConditioning, alpha: float = 0.1,
                 beta: float = 0.1):
        assert 0.0 <= alpha < 1.0 and 0.0 <= beta < 1.0
        self.existence = existence
        self.alpha = float(alpha)
        self.beta = float(beta)

    @property
    def log_lr(self) -> float:
        """λ = log ρ = log(1−β) − log α。"""
        return math.log((1.0 - self.beta) / max(self.alpha, 1e-12))

    @staticmethod
    def weight_from_lr(pi: float, lam: float) -> float:
        """w(λ) = π e^λ / (π e^λ + 1 − π)（logistic 型，E6）。"""
        e = math.exp(lam)
        return (pi * e) / (pi * e + (1.0 - pi) + 1e-300)

    def weight(self, pi: float) -> float:
        return self.weight_from_lr(pi, self.log_lr)

    def soft_terminal(self, z0: torch.Tensor, x_grid: torch.Tensor,
                      n_paths: Optional[int] = None, seed: Optional[int] = None
                      ) -> Tuple[np.ndarray, float, float]:
        """软证据后验终端密度 = w·p_exist + (1−w)·p_non。返回 (密度, π, w)。"""
        p_exist, pi = self.existence.rejection_terminal(z0, x_grid, n_paths, seed)
        p_non = self.existence.non_existence_terminal(z0, x_grid, n_paths, seed)
        w = self.weight(pi)
        return w * p_exist + (1.0 - w) * p_non, pi, w


# ---------------------------------------------------------------------------
# 对偶验证：存在桥 vs 排除桥，同一区域 A
# ---------------------------------------------------------------------------
def existence_dual_check(sde: SegmentConstantSDE, z0: torch.Tensor,
                         region: Tuple[float, float], horizon: float, dt: float,
                         x_grid: torch.Tensor, n_paths: int,
                         seed: Optional[int] = None) -> dict:
    """full-horizon [0,T] 对偶验证。

    同一批路径上计算：
      π = P(τ_A ≤ T)（存在）、surv = P(τ_A > T)（排除）→ 对偶恒等式 π + surv ≡ 1；
      p_exist = P(·|τ_A≤T)、p_excl = P(·|τ_A>T) → 混合还原先验 π·p_exist+(1−π)·p_excl = p；
      质量方向对偶：p_exist 在 A 内质量 vs p_excl 在 A 内质量（一增一减）。
    """
    paths = simulate_paths(sde, z0, horizon, dt, n_paths, seed)
    hit = _hit_in_window(paths, [region], 0.0, horizon, dt)  # full-horizon 存在事件
    X = paths[:, -1, 0].detach().numpy()
    hit_np = hit.numpy()
    pi = float(hit_np.mean())
    surv = 1.0 - pi

    edges = x_grid.numpy()
    p_exist = _terminal_hist(X[hit_np], edges)
    p_excl = _terminal_hist(X[~hit_np], edges)
    p_prior_mc = _terminal_hist(X, edges)

    # 混合还原先验（同批路径下应精确成立，仅浮点误差）
    recon = pi * p_exist + (1.0 - pi) * p_excl
    recon_l1 = float(np.abs(recon - p_prior_mc).sum())

    # 解析先验（fp_mc 交叉核对，MC 噪声口径）
    mus, sigs, pis = terminal_gaussian_mixture(sde, z0, horizon, dt)
    centers_t = 0.5 * (x_grid[:-1] + x_grid[1:])
    centers = centers_t.numpy()
    widths = np.diff(edges)
    prior_mass = mixture_pdf(centers_t, mus, sigs, pis).numpy() * widths
    prior_mass /= (prior_mass.sum() + 1e-12)
    l1_fp_mc = float(np.abs(prior_mass - p_prior_mc).sum())

    # 质量方向对偶：A 内质量
    inside = (centers >= min(region)) & (centers <= max(region))
    mass_A_exist = float(p_exist[inside].sum())
    mass_A_excl = float(p_excl[inside].sum())
    mass_A_prior = float(prior_mass[inside].sum())

    return {
        "region": [float(min(region)), float(max(region))],
        "pi_exist": pi, "surv_excl": surv,
        "dual_identity_pi_plus_surv": pi + surv,
        "recon_l1": recon_l1,
        "fp_mc_l1": l1_fp_mc,
        "mass_in_A": {"exist": mass_A_exist, "excl": mass_A_excl, "prior": mass_A_prior},
        "path_crossing_note": ("存在桥在 A 内质量应显著高于先验，排除桥在 A 内质量≈0；"
                               "二者加权和还原先验（混合对偶，理论 §5 判据）"),
    }


# ---------------------------------------------------------------------------
# E6 误判鲁棒性（(α,β) 网格 → λ → w → 重采样混合）
# ---------------------------------------------------------------------------
def robustness_sweep(sde: SegmentConstantSDE, z0: torch.Tensor,
                     region: Tuple[float, float], horizon: float, dt: float,
                     x_grid: torch.Tensor, n_paths: int,
                     alpha_grid: Sequence[float], beta_grid: Sequence[float],
                     seed: Optional[int] = None) -> dict:
    """软证据对专家误判 (α,β) 的鲁棒性。

    桥分量 P(·|E)、P(·|E^c) 与 (α,β) 无关（只改权重 w），一次模拟即可扫全部网格：
      λ=log((1−β)/α) → w(λ) → p_soft = w·p_exist+(1−w)·p_excl。
    验证 E6：TV(p_soft(λ), p_soft(λ')) = |w−w'|·TV(p_exist,p_excl) ≤ 2|w−w'| ≤ ½|λ−λ'|。
    """
    exist = ExistenceConditioning(sde, region, horizon, dt, t1=0.0, t2=horizon,
                                  n_paths=n_paths, seed=seed)
    p_exist, pi = exist.rejection_terminal(z0, x_grid)
    p_non = exist.non_existence_terminal(z0, x_grid)
    tv_comp = 0.5 * float(np.abs(p_exist - p_non).sum())  # TV(P(·|E), P(·|E^c))

    rows = []
    lam_prev = w_prev = p_prev = None
    max_tv_vs_dlam = 0.0
    for alpha in alpha_grid:
        for beta in beta_grid:
            lam = math.log((1.0 - beta) / max(alpha, 1e-12))
            w = SoftExistenceConditioning.weight_from_lr(pi, lam)
            p_soft = w * p_exist + (1.0 - w) * p_non
            row = {
                "alpha": float(alpha), "beta": float(beta),
                "lambda": lam, "rho": math.exp(lam), "w": w,
                "tv_vs_prior": 0.5 * float(np.abs(p_soft - (pi * p_exist + (1 - pi) * p_non)).sum()),
                "tv_vs_hard_exist": 0.5 * float(np.abs(p_soft - p_exist).sum()),
            }
            if lam_prev is not None:
                dlam = abs(lam - lam_prev)
                tv = 0.5 * float(np.abs(p_soft - p_prev).sum())
                # E6 界：TV ≤ ½|Δλ|
                max_tv_vs_dlam = max(max_tv_vs_dlam, tv / (dlam + 1e-300))
                row["tv_vs_prev"] = tv
                row["e6_bound_0p5_dlam"] = 0.5 * dlam
                row["e6_satisfied"] = bool(tv <= 0.5 * dlam + 1e-9)
            rows.append(row)
            lam_prev, w_prev, p_prev = lam, w, p_soft

    # logistic 斜率 |dw/dλ| = w(1−w) ≤ ¼（解析），数值复验
    dw = [(rows[i + 1]["w"] - rows[i]["w"]) / (rows[i + 1]["lambda"] - rows[i]["lambda"])
          for i in range(len(rows) - 1)]
    return {
        "pi_exist": pi,
        "tv_components": tv_comp,
        "max_tv_over_dlam": max_tv_vs_dlam,
        "e6_lipschitz_bound_0p5": max_tv_vs_dlam <= 0.5 + 1e-9,
        "max_abs_dw_dlam": max(abs(d) for d in dw) if dw else 0.0,
        "dw_dlam_bound_0p25": (max(abs(d) for d in dw) if dw else 0.0) <= 0.25 + 1e-9,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 收敛主线：正信息条件化迭代下搜索优先级收敛（对偶 iterative_exclusion）
# ---------------------------------------------------------------------------
def iterative_existence(sde: SegmentConstantSDE, z0: torch.Tensor, x_true: float,
                        cell_edges: torch.Tensor, horizon: float, dt: float,
                        n_paths: int, n_rounds: int = 6,
                        seed: Optional[int] = None,
                        soft: Optional[Tuple[float, float]] = None) -> dict:
    """正信息条件化迭代：嵌套存在区域 A_1 ⊃ A_2 ⊃ … 收缩到真值格 → 优先级收敛。

    每轮以「真值格为中心」取 1/2^j 域宽的存在区域（专家逐步精确化「有大概率出现过」），
    施加（软）存在条件化，记录真值格 rank 与 CEP50（覆盖 50% 质量所需面积比例）。
    软证据 (α,β) 给定时按 E5 mixture 加权；否则用硬存在桥（无噪极限 ρ→∞）。
    """
    n_cells = len(cell_edges) - 1
    edges = cell_edges.numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    true_cell = min(max(int(np.searchsorted(edges, x_true)) - 1, 0), n_cells - 1)

    paths = simulate_paths(sde, z0, horizon, dt, n_paths, seed)
    X = paths[:, -1, 0].detach().numpy()
    p_prior = _terminal_hist(X, edges)

    # 先验真值格 rank（MC 口径 + 解析口径，共享 prior_true_cell_rank 定义）
    prior_rank = int((p_prior > p_prior[true_cell]).sum()) + 1
    prior_cep50 = _cep50(p_prior)
    mus, sigs, pis = terminal_gaussian_mixture(sde, z0, horizon, dt)
    centers_t = 0.5 * (cell_edges[:-1] + cell_edges[1:])
    prior_mass = mixture_pdf(centers_t, mus, sigs, pis).numpy() * widths
    prior_mass /= (prior_mass.sum() + 1e-12)
    prior_rank_analytic = int((prior_mass > prior_mass[true_cell]).sum()) + 1
    prior_cep50_analytic = _cep50(prior_mass)

    rounds = []
    for j in range(1, n_rounds + 1):
        # 嵌套区域：真值格为中心，域宽 = 2 * max(1, n_cells / 2^j) 格
        half = max(1, int(n_cells / (2 ** j)))
        lo_cell = max(0, true_cell - half)
        hi_cell = min(n_cells, true_cell + half + 1)
        region = (float(edges[lo_cell]), float(edges[hi_cell - 1] + (edges[1] - edges[0]) * 0.0))
        # 区域取到 hi_cell 的右边界
        region = (float(edges[lo_cell]), float(edges[hi_cell]))

        hit = _hit_in_window(paths, [region], 0.0, horizon, dt).numpy()
        p_exist = _terminal_hist(X[hit], edges)
        p_non = _terminal_hist(X[~hit], edges)
        pi = float(hit.mean())

        if soft is None:
            p_post = p_exist          # 硬存在桥（ρ→∞）
            w = 1.0
        else:
            alpha, beta = soft
            lam = math.log((1.0 - beta) / max(alpha, 1e-12))
            w = SoftExistenceConditioning.weight_from_lr(pi, lam)
            p_post = w * p_exist + (1.0 - w) * p_non

        rank = int((p_post > p_post[true_cell]).sum()) + 1
        cep50 = _cep50(p_post)
        mass_A = float(p_post[lo_cell:hi_cell].sum())
        rounds.append({
            "round": j, "region_cells": [lo_cell, hi_cell],
            "region_lo": region[0], "region_hi": region[1],
            "area_frac": float(hi_cell - lo_cell) / n_cells,
            "pi_exist": pi, "w": w,
            "true_cell_rank": rank, "cep50_area_frac": cep50,
            "mass_in_A": mass_A,
        })

    return {
        "x_true": float(x_true), "true_cell": true_cell,
        "prior_true_cell_rank": prior_rank,
        "prior_true_cell_rank_analytic": prior_rank_analytic,
        "prior_cep50_area_frac": prior_cep50,
        "prior_cep50_area_frac_analytic": prior_cep50_analytic,
        "soft": (None if soft is None else {"alpha": soft[0], "beta": soft[1]}),
        "n_paths": n_paths, "horizon": float(horizon), "dt": float(dt),
        "rounds": rounds,
    }


def _cep50(p: np.ndarray) -> float:
    """覆盖 50% 质量所需面积比例（搜索效率，与排除实验 CEP50 同口径）。"""
    n = len(p)
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    k50 = int(np.searchsorted(cum, 0.5)) + 1
    return float(min(k50, n)) / n
