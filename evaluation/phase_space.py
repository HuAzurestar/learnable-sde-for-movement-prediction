"""Proper-score definitions shared by O2 gradients and held-out evaluation."""

import math

import torch

from models.phase_space import ModelContractError


def energy_score_value(samples, target, *, pair_seed=None, pair_count=65536):
    """Exact non-diagonal U-statistic up to 256 paths; explicitly sampled above."""
    if (samples.ndim != 2 or target.shape != samples.shape[1:] or len(samples) < 2
            or samples.dtype != target.dtype or samples.device != target.device
            or not torch.isfinite(samples).all() or not torch.isfinite(target).all()):
        raise ModelContractError("MODEL_CONTRACT_ERROR: finite samples [M,D], target [D] required")
    count = len(samples)
    first = (samples - target).norm(dim=-1).mean()
    if count <= 256:
        distances = torch.cdist(samples, samples)
        # Diagonal is exactly zero in the estimator, including cdist roundoff.
        distances = distances - torch.diag_embed(torch.diag(distances))
        value = first - distances.sum() / (2 * count * (count - 1))
        return value, {"estimator_id": "energy-u-exact-v1", "pair_count": count * (count - 1)}
    if type(pair_seed) is not int or not 0 <= pair_seed < 2**63 or type(pair_count) is not int or not 2 <= pair_count <= 65536:
        raise ModelContractError("RESOURCE_PLAN_REJECTED: registered pair seed/count required above 256 paths")
    generator = torch.Generator(device=samples.device).manual_seed(pair_seed)
    left = torch.randint(count, (pair_count,), device=samples.device, generator=generator)
    right = torch.randint(count - 1, (pair_count,), device=samples.device, generator=generator)
    right = right + (right >= left)
    distances = (samples[left] - samples[right]).norm(dim=-1)
    return first - .5 * distances.mean(), {"estimator_id": "energy-u-uniform-pairs-v1", "pair_count": pair_count,
        "pair_seed": pair_seed, "pair_sampling_se": .5 * distances.detach().std().item() / math.sqrt(pair_count)}


def evaluate_forecast(result, authorized_truth, *, pair_seed=None, pair_count=65536):
    """Truth is supplied only after the caller's exposure/authorization gate."""
    samples = result["samples"]
    if (samples.ndim != 3 or samples.shape[2] != 4 or authorized_truth.shape != samples.shape[1:]
            or samples.dtype != authorized_truth.dtype or samples.device != authorized_truth.device
            or not torch.isfinite(authorized_truth).all()):
        raise ModelContractError("MODEL_CONTRACT_ERROR: aligned authorized truth [T,4] required")
    if len(samples) < 2:
        return {"status": "UNAVAILABLE", "reason": "fewer than two valid paths", "metrics": None,
                "failure_rate": result["failure_rate"]}
    rows = []
    for index, timestamp in enumerate(result["time_grid"]):
        position, truth = samples[:, index, :2], authorized_truth[index, :2]
        value, estimator = energy_score_value(position, truth, pair_seed=pair_seed, pair_count=pair_count)
        score_se = None
        if 3 <= len(position) <= 256:
            count = len(position)
            distances = torch.cdist(position.detach(), position.detach())
            distances -= torch.diag_embed(torch.diag(distances))
            first = (position.detach() - truth).norm(dim=-1)
            leave_one = (first.sum() - first) / (count - 1)
            leave_one -= (distances.sum() - 2 * distances.sum(1)) / (2 * (count - 1) * (count - 2))
            score_se = ((count - 1) / count * (leave_one - leave_one.mean()).square().sum()).sqrt().item()
        order = position.sort(dim=0).values
        weights = (2 * torch.arange(1, len(position) + 1, dtype=position.dtype, device=position.device) - len(position) - 1)[:, None]
        crps = (position - truth).abs().mean(0) - (weights * order).sum(0) / len(position)**2
        metrics = {"energy_score": value.item(), "marginal_crps": crps.detach().cpu().tolist(),
            "position_error": (position.mean(0) - truth).norm().item(),
            "velocity_rmse": (samples[:, index, 2:].mean(0) - authorized_truth[index, 2:]).square().mean().sqrt().item(),
            "pit": ((position < truth).float().mean(0) + .5 * (position == truth).float().mean(0)).cpu().tolist(),
            "held_out_nll": None, "energy_score_jackknife_se": score_se}
        intervals = {}
        for probability in (.5, .8, .95):
            tail = (1 - probability) / 2
            lower, upper = torch.quantile(position, torch.tensor([tail, 1 - tail], dtype=position.dtype, device=position.device), dim=0)
            intervals[str(probability)] = {"marginal_coverage": ((truth >= lower) & (truth <= upper)).float().cpu().tolist(),
                                          "width": (upper - lower).detach().cpu().tolist()}
        rows.append({"time": timestamp, "metrics": metrics, "estimator": estimator, "intervals": intervals,
                     "sample_count": len(samples), "position_units": "m", "velocity_units": "m/s"})
    errors = [row["metrics"]["position_error"] for row in rows[1:]]
    return {"status": "PARTIAL" if result["failed_sample_ids"] else "SUCCEEDED",
            "evaluation_population": "surviving-paths", "rows": rows,
            "ade": sum(errors) / len(errors), "fde": errors[-1],
            "failure_rate": result["failure_rate"], "failed_paths": len(result["failed_sample_ids"]),
            "requested_paths": result["requested_paths"]}
