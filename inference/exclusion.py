"""Exclusion conditioning for negative information (E1--E3/E0).

The mathematical contract is:

  失联人员运动 = R^d 上 Itô 扩散（I-1 段级常模式欠阻尼朗之万，状态 z=[X,V]）
    dX = V dt
    dV = [−Γ V + (a−κ)X + c] dt + g dW ,    a := σσᵀ = diag(0, g²)
  搜索域 D⊂R（位置），排除区 A（若干位置区间之并），首达时 τ_A = inf{t≥0: X_t∈A}。
  排除事件（负信息「搜过=没人」）= {τ_A > T}（全程不入 A）。

  E1  生存概率 h(t,z) = P(τ_A > T | Z_t=z) 是吸收边界后向方程唯一有界解
        ∂_t h + L h = 0 在 [0,T]×(D\\A),  h|_{[0,T]×∂A}=0 (Dirichlet 0),  h(T,·)=1_{D\\A}。
        本模块用 Feynman–Kac MC 数值解（从 z 前向模拟 I-1 精确核路径，数「永不入 A」比例）。
  E2  生存加权密度 p_excl(t,x) = h(t,x)·p(t,x)/P(τ_A>T)（t<T 时 h=前向生存）。
  E3  Doob h-变换：条件化律 P(·|τ_A>T) 是 P 关于 h 的 h-变换，漂移修正
        b^h = b + a∇log h；欠阻尼 a=diag(0,g²) ⇒ 修正只落 V 分量 Δb_V = g² ∂_V log h。
  E0  硬排除基线：只用终端信息截断 D\\A 再重归一（规范化常数 P(X_T∉A)，非路径生存）。

**正确性口径（关键）**：终端时刻 t=T 的条件边际密度 = P(X_T∈·, τ_A>T)/P(τ_A>T)，
含「路径中途穿过 A 又离开」的时间反演生存因子，故 E2 的 1_{x∉A}·p/P(τ_A>T) 仅在
A「进入即吸收」时可化为截断；一般情形的**黄金口径是路径级拒绝采样（rejection）**：
模拟 N 条路径，保留全程不入 A 者，其终端经验分布 = P(·|τ_A>T) 的精确 MC 估计。
E0（终端截断重归一）是忽略路径信息的朴素基线；二者差 = P(X_T∉A) − P(τ_A>T) ≥ 0，
即「穿过又离开 A」的质量。本模块以 rejection 为主引擎，E0 作对照。

复用 `inference/bridge.py` 的 `BridgeConditioning`（doob/soft h-变换接口），h 由终点势
换为吸收边界后向方程数值解（survival_mc）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from domain import ModelContext
from models.segment_constant import SegmentConstantSDE
from numerics import safe_cholesky


# ---------------------------------------------------------------------------
# 精确核 / 终端高斯（线性 SDE 闭式，供 E0 与 fp_mc_cross_check 交叉验证）
# ---------------------------------------------------------------------------
def mode_kernel(sde: SegmentConstantSDE, k: int, dt: float
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """模式 k 的精确转移核 (F, c, Σ)（F z + c, N(0,Σ)）。"""
    return sde.affine_transition(dt, ModelContext(regime=k))


def terminal_gaussian_mixture(sde: SegmentConstantSDE, z0: torch.Tensor,
                              horizon: float, dt: float
                              ) -> Tuple[List[float], List[float], List[float]]:
    """终端位置 X_T | mode k ~ N(mu_k, sig2_k) 的精确高斯混合。

    返回 (mu_x: [K], sig2_x: [K], pi: [K])。X 分量均值/方差由 F^n z0 + ΣF^i c、
    Σ F^i Σ (F^i)ᵀ 累加得到（I-1 线性 SDE 精确核，无离散化误差）。
    """
    n_steps = max(int(round(horizon / dt)), 1)
    z0 = z0.to(torch.float64)
    mus, sigs, pis = [], [], []
    pi = torch.softmax(sde.prior_logits, dim=0)
    Id = torch.eye(2, dtype=z0.dtype, device=z0.device)
    for k in range(sde.n_modes):
        F, c, S = mode_kernel(sde, k, dt)
        mu = z0.clone()
        Fi = Id.clone()
        Ssum = torch.zeros((2, 2), dtype=z0.dtype, device=z0.device)
        for _ in range(n_steps):
            mu = F @ mu + c
            Ssum = Ssum + Fi @ S @ Fi.T
            Fi = F @ Fi
        mus.append(float(mu[0]))
        sigs.append(max(float(Ssum[0, 0]), 1e-12))
        pis.append(float(pi[k]))
    return mus, sigs, pis


def _gaussian_pdf(x: torch.Tensor, mu: float, sig2: float) -> torch.Tensor:
    s = max(float(sig2), 1e-12) ** 0.5
    return torch.exp(-0.5 * ((x - mu) / s) ** 2) / (s * 2.5066282746310002)


def mixture_pdf(x: torch.Tensor, mus: Sequence[float], sigs: Sequence[float],
                pis: Sequence[float]) -> torch.Tensor:
    """终端位置密度 p(T, x) = Σ π_k N(x; mu_k, sig2_k)（x 可批）。"""
    x = x.to(torch.float64)
    out = torch.zeros_like(x)
    for mu, s, p in zip(mus, sigs, pis):
        if p <= 0:
            continue
        out = out + p * _gaussian_pdf(x, mu, s)
    return out


# ---------------------------------------------------------------------------
# 路径模拟（I-1 精确核，段级常模式；rejection 主引擎）
# ---------------------------------------------------------------------------
def simulate_paths(sde: SegmentConstantSDE, z0: torch.Tensor, horizon: float,
                   dt: float, n_paths: int, seed: Optional[int] = None,
                   mode: Optional[int] = None) -> torch.Tensor:
    """从 z0=(2,) 模拟 n_paths 条路径，返回 (n_paths, n_steps+1, 2)。

    mode=None：每路径从段先验 π 抽模式（段内常值，与 I-1 rollout 同口径）；
    mode=k：固定单模式。X 分量即位置（排除判定用）。
    """
    z0 = z0.to(torch.float64)
    n_steps = max(int(round(horizon / dt)), 1)
    g = torch.Generator(device=sde.device)
    if seed is not None:
        g.manual_seed(seed)
    pi = torch.softmax(sde.prior_logits, dim=0)
    if mode is None:
        modes = torch.multinomial(pi, n_paths, replacement=True, generator=g)
    else:
        modes = torch.full((n_paths,), mode, dtype=torch.long)

    # 逐模式预计算精确核
    kernels = {}
    for k in range(sde.n_modes):
        F, c, S = mode_kernel(sde, k, dt)
        L = safe_cholesky(S)
        kernels[k] = (F, c, L)

    z = z0.unsqueeze(0).expand(n_paths, 2).clone()
    traj = [z.clone()]
    for _ in range(n_steps):
        z_next = torch.empty_like(z)
        for k in range(sde.n_modes):
            m = modes == k
            if not m.any():
                continue
            F, c, L = kernels[k]
            nk = int(m.sum())
            eps = torch.randn(nk, 2, dtype=z.dtype, device=sde.device, generator=g)
            z_next[m] = z[m] @ F.T + c + eps @ L.T
        z = z_next
        traj.append(z.clone())
    return torch.stack(traj, dim=1).detach()  # (n_paths, n_steps+1, 2)


def path_occupancy(paths: torch.Tensor, cell_edges: torch.Tensor) -> torch.Tensor:
    """每条路径是否曾进入各格：返回 (n_paths, n_cells) bool。

    cell_edges: (n_cells+1,) 位置格边界。
    """
    X = paths[..., 0]  # (n_paths, n_steps+1)
    edges = cell_edges.to(torch.float64)
    n_cells = len(edges) - 1
    occ = torch.zeros(X.shape[0], n_cells, dtype=torch.bool)
    for c in range(n_cells):
        occ[:, c] = ((X >= edges[c]) & (X < edges[c + 1])).any(dim=1)
    # 末格闭右端
    occ[:, -1] = occ[:, -1] | (X >= edges[-2]).any(dim=1) & (X <= edges[-1]).any(dim=1)
    return occ


# ---------------------------------------------------------------------------
# E1 生存概率（Feynman–Kac MC 数值解）
# ---------------------------------------------------------------------------
def survival_mc(sde: SegmentConstantSDE, z0: torch.Tensor,
                regions: Sequence[Tuple[float, float]], horizon: float, dt: float,
                n_paths: int, seed: Optional[int] = None,
                per_mode: bool = False):
    """P(τ_A > T | Z_0=z0) 的 MC 估计（后向方程 Feynman–Kac 数值解）。

    regions: [(lo, hi), ...] 位置区间之并（A）。horizon=T；dt=推进步长（秒）。
    z0: (2,)。per_mode=True 返回 (K,) 逐模式生存概率；否则返回按段先验 π 混合的标量。
    """
    n_steps = max(int(round(horizon / dt)), 1)
    surv = []
    for k in range(sde.n_modes):
        F, c, S = mode_kernel(sde, k, dt)
        L = safe_cholesky(S)
        g = torch.Generator(device=sde.device)
        if seed is not None:
            g.manual_seed(seed + k * 100003)
        z = z0.detach().clone().to(sde.dtype).unsqueeze(0).expand(n_paths, 2).clone()
        alive = torch.ones(n_paths, dtype=torch.bool, device=sde.device)
        for _ in range(n_steps):
            z = z @ F.T + c + torch.randn(n_paths, 2, dtype=z.dtype, device=sde.device, generator=g) @ L.T
            x = z[:, 0]
            in_any = torch.zeros(n_paths, dtype=torch.bool, device=sde.device)
            for (lo, hi) in regions:
                in_any = in_any | ((x >= lo) & (x <= hi))
            alive = alive & (~in_any)
        surv.append(float(alive.float().mean()))
    if per_mode:
        return torch.tensor(surv, dtype=torch.float64)
    pi = torch.softmax(sde.prior_logits, dim=0)
    return float(sum(pi[k].item() * surv[k] for k in range(sde.n_modes)))


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------
class AbsorbingExclusionConditioning:
    """排除条件化（负信息）。E1 生存概率 + E2 生存加权密度 + E3 h-变换 + E0 基线。

    与 `bridge.BridgeConditioning` 同族的 doob/soft h-变换接口：`conditioned_drift`
    返回 b + a∇log h；h 由吸收边界后向方程数值解（survival_mc）提供，非终点势。
    """

    def __init__(self, sde: SegmentConstantSDE, region: Tuple[float, float],
                 horizon: float, dt: float = 60.0, n_paths: int = 20000,
                 seed: Optional[int] = None):
        self.sde = sde
        self.region = (float(min(region)), float(max(region)))
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.n_paths = int(n_paths)
        self.seed = seed

    # -- E1 -----------------------------------------------------------------
    def survival(self, z0: torch.Tensor, n_paths: Optional[int] = None,
                 seed: Optional[int] = None) -> float:
        """P(τ_A > T | Z_0=z0)（后向方程数值解，混合口径）。"""
        return survival_mc(self.sde, z0, [self.region], self.horizon, self.dt,
                           n_paths or self.n_paths,
                           seed if seed is not None else self.seed)

    # -- E2/E3 正确口径：路径级拒绝采样终端密度 ---------------------------------
    def rejection_terminal(self, z0: torch.Tensor, x_grid: torch.Tensor,
                           n_paths: Optional[int] = None, seed: Optional[int] = None
                           ) -> Tuple[np.ndarray, float]:
        """黄金口径：P(·|τ_A>T) 的终端经验密度 + 生存概率。

        模拟 n_paths 条路径，保留全程不入 A 者，其终端 X 落在 x_grid 格上的
        归一化直方图 = 条件化终端密度的 MC 估计。返回 (密度数组, 生存概率)。
        """
        n_paths = n_paths or self.n_paths
        seed = seed if seed is not None else self.seed
        paths = simulate_paths(self.sde, z0, self.horizon, self.dt, n_paths, seed)
        X = paths[:, -1, 0]
        in_A = (X >= self.region[0]) & (X <= self.region[1])
        # 路径级生存（全程不入 A）
        occ = (paths[..., 0] >= self.region[0]) & (paths[..., 0] <= self.region[1])
        survives = ~(occ.any(dim=1))
        surv = float(survives.float().mean())
        xg = x_grid.to(torch.float64).numpy()
        if survives.sum() == 0:
            return np.zeros_like(xg), surv
        term = X[survives].numpy()
        hist, _ = np.histogram(term, bins=xg, density=False)
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-12)
        return hist, surv

    # -- E0 -----------------------------------------------------------------
    def hard_exclusion(self, z0: torch.Tensor, x_grid: torch.Tensor) -> np.ndarray:
        """E0 硬排除基线：终端截断 D\\A 再重归一（规范化常数 P(X_T∉A)）。"""
        mus, sigs, pis = terminal_gaussian_mixture(self.sde, z0, self.horizon, self.dt)
        centers = 0.5 * (x_grid[:-1] + x_grid[1:])
        p = mixture_pdf(centers, mus, sigs, pis)
        inside = (centers >= self.region[0]) & (centers <= self.region[1])
        out = p.numpy().copy()
        out[inside] = 0.0
        return out / (out.sum() + 1e-12)

    # -- E3 -----------------------------------------------------------------
    def conditioned_drift(self, z: torch.Tensor, t_left: float, mode: int = 0,
                          dv: float = 0.05, n_paths: Optional[int] = None,
                          seed: Optional[int] = None) -> torch.Tensor:
        """Doob h-变换漂移 b + a∇log h（欠阻尼：修正只落 V 分量 Δb_V = g² ∂_V log h）。

        z: (2,) [X,V]；t_left=剩余时间 T−t；h(t_left,·)=survival_mc 数值解。
        ∂_V log h 用中心差分（V±dv 处 survival_mc 估计）。mode=段模式 k。
        """
        z = z.to(torch.float64)
        n_paths = n_paths or self.n_paths
        seed = seed if seed is not None else self.seed
        base = self.sde.drift(
            torch.zeros((), dtype=z.dtype, device=z.device),
            z,
            ModelContext(regime=mode),
        )
        h_plus = survival_mc(self.sde, torch.stack([z[0], z[1] + dv]), [self.region],
                             float(t_left), self.dt, n_paths, seed=seed)
        h_minus = survival_mc(self.sde, torch.stack([z[0], z[1] - dv]), [self.region],
                              float(t_left), self.dt, n_paths, seed=seed + 1)
        dlogh_dV = (np.log(max(h_plus, 1e-12)) - np.log(max(h_minus, 1e-12))) / (2.0 * dv)
        g2 = float(self.sde.g[mode]) ** 2
        corr = torch.zeros(2, dtype=z.dtype)
        corr[1] = g2 * dlogh_dV
        return base + corr

    def sample_excluded(self, z0: torch.Tensor, n: int, seed: Optional[int] = None,
                        dv: float = 0.05, n_paths: Optional[int] = None) -> torch.Tensor:
        """E3 Doob h-变换 rollout：从 z0 推进 horizon（dt 步），返回 n 条路径终态 (n,2)。

        每步漂移 = conditioned_drift(z, t_left)（b + g² ∂_V log h），噪声仍为精确核扩散。
        验证口径：h-变换路径的生存率应显著高于无条件路径（→1），终端密度应逼近
        rejection 黄金口径（Doob 定理：h-变换过程 = 条件化过程）。"""
        z0 = z0.to(torch.float64)
        n_paths = n_paths or self.n_paths
        n_steps = max(int(round(self.horizon / self.dt)), 1)
        g = torch.Generator(device=self.sde.device)
        if seed is not None:
            g.manual_seed(seed)
        # 主导模式噪声强度（段级常模式，按先验加权 g²）
        pi = torch.softmax(self.sde.prior_logits, dim=0)
        g2_eff = float(sum(pi[k].item() * float(self.sde.g[k]) ** 2 for k in range(self.sde.n_modes)))
        z = z0.unsqueeze(0).expand(n, 2).clone()
        for s in range(n_steps):
            t_left = self.horizon - s * self.dt
            # 逐条路径漂移（h-变换依赖当前 z，逐条算；冒烟级 n 取小）
            drift = torch.stack([
                self.conditioned_drift(z[i], t_left, dv=dv, n_paths=n_paths, seed=seed)
                for i in range(n)])
            # 精确核推进：z_next = z + drift*dt + L eps（欠阻尼噪声只落 V）
            L = torch.zeros(2, dtype=z.dtype)
            L[1] = float(max(g2_eff, 1e-12)) ** 0.5 * (self.dt ** 0.5)
            eps = torch.randn(n, 2, dtype=z.dtype, device=self.sde.device, generator=g)
            z = z + drift * self.dt + eps * L.unsqueeze(0)
        return z.detach()


# ---------------------------------------------------------------------------
# 迭代排除（收敛主线：负信息迭代下搜索区域优先级收敛）
# ---------------------------------------------------------------------------
@dataclass
class ExclusionStep:
    """单步排除结果。"""
    iter: int
    searched_cells: int
    searched_area_frac: float
    survival_mass: float                 # P(τ_{S_i} > T)（E1 路径生存，正确口径）
    terminal_not_in_frac: float          # P(X_T ∉ S_i)（E0 终端口径）
    true_cell_rank: int                  # 真值格优先级（1=最高，E2/E3 口径）
    true_cell_rank_e0: int               # E0 口径优先级（截断不改变排序）
    true_cell_mass: float                # E2/E3 条件密度在真值格的质量
    cep50_area_frac: float               # 覆盖 50% 条件质量所需面积比例（搜索效率）
    found: bool                          # 本步搜索格 = 真值格


def iterative_exclusion(sde: SegmentConstantSDE, z0: torch.Tensor, x_true: float,
                        cell_edges: torch.Tensor, horizon: float, dt: float,
                        n_paths: int, max_iters: Optional[int] = None,
                        seed: Optional[int] = None) -> Tuple[List[ExclusionStep], dict]:
    """迭代排除：每次排除「当前条件密度最高格」入已搜索集合 S_i，重算条件密度。

    黄金口径：一次模拟 n_paths 条路径 → 每格占用（曾否进入）→ 任意 S_i 的
    「路径级生存」= 未进 S_i 任一格的路径；其终端经验分布 = P(·|τ_{S_i}>T)。
    对照 E0：终端截断（prior 直方图零化 S_i 再重归一，不改变剩余格排序）。

    返回 (steps, summary)。summary 含 prior_mass（每格先验质量，供 E0）。
    """
    n_cells = len(cell_edges) - 1
    centers = 0.5 * (cell_edges[:-1] + cell_edges[1:])
    mus, sigs, pis = terminal_gaussian_mixture(sde, z0, horizon, dt)
    prior = mixture_pdf(centers, mus, sigs, pis).numpy()
    widths = np.diff(cell_edges.numpy())
    prior_mass = prior * widths
    prior_mass /= (prior_mass.sum() + 1e-12)

    true_cell = int(np.searchsorted(cell_edges.numpy(), x_true)) - 1
    true_cell = min(max(true_cell, 0), n_cells - 1)

    # 一次模拟 + 占用矩阵
    paths = simulate_paths(sde, z0, horizon, dt, n_paths, seed)
    occ = path_occupancy(paths, cell_edges)  # (n_paths, n_cells)
    X_term = paths[:, -1, 0].detach().numpy()

    searched = np.zeros(n_cells, dtype=bool)
    steps: List[ExclusionStep] = []
    max_iters = max_iters if max_iters is not None else n_cells

    for it in range(1, max_iters + 1):
        # 当前已搜索格 S_i 下的路径级生存
        if searched.any():
            hit = occ[:, searched].any(dim=1).numpy()
            survives = ~hit
        else:
            survives = np.ones(len(X_term), dtype=bool)
        if survives.sum() == 0:
            break
        hist, _ = np.histogram(X_term[survives], bins=cell_edges.numpy())
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-12)  # 条件终端密度（E2/E3 黄金口径）

        # 排除当前最高条件密度未搜索格
        cand = np.where(searched, -1.0, hist)
        pick = int(np.argmax(cand))
        searched[pick] = True

        # 排除后 S_i（含 pick）下的状态
        hit_i = occ[:, searched].any(dim=1).numpy()
        survives_i = ~hit_i
        surv_mass = float(survives_i.mean())
        hist_i, _ = np.histogram(X_term[survives_i], bins=cell_edges.numpy())
        hist_i = hist_i.astype(np.float64)
        hist_i /= (hist_i.sum() + 1e-12)

        # E0 终端口径：prior_mass 零化 S_i 再重归一
        p_e0 = np.where(searched, 0.0, prior_mass)
        p_e0 /= (p_e0.sum() + 1e-12)

        # 终端非命中（E0 规范化常数）
        searched_mass = prior_mass[searched].sum()
        terminal_not_in = 1.0 - searched_mass

        # 真值格优先级（1=最高）
        true_mass = hist_i[true_cell]
        rank = int((hist_i > true_mass).sum()) + 1
        rank_e0 = int((p_e0 > p_e0[true_cell]).sum()) + 1

        # CEP50：覆盖 50% 条件质量所需面积比例
        order = np.argsort(-hist_i)
        cum = np.cumsum(hist_i[order])
        k50 = int(np.searchsorted(cum, 0.5)) + 1
        cep50 = float(k50) / n_cells

        steps.append(ExclusionStep(
            iter=it, searched_cells=int(searched.sum()),
            searched_area_frac=float(searched.sum()) / n_cells,
            survival_mass=surv_mass, terminal_not_in_frac=float(terminal_not_in),
            true_cell_rank=rank, true_cell_rank_e0=rank_e0,
            true_cell_mass=float(true_mass), cep50_area_frac=cep50,
            found=(pick == true_cell),
        ))
        if pick == true_cell:
            break

    summary = {
        "n_cells": n_cells,
        "true_cell": true_cell,
        "true_cell_center": float(centers[true_cell]),
        "x_true": float(x_true),
        "prior_mass": prior_mass.tolist(),
        "prior_true_cell_mass": float(prior_mass[true_cell]),
        "prior_true_cell_rank": int((prior_mass > prior_mass[true_cell]).sum()) + 1,
        "n_paths": n_paths,
        "horizon": float(horizon),
        "dt": float(dt),
    }
    return steps, summary
