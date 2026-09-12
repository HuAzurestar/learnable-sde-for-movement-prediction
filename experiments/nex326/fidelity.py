"""Machine-readable implementation-fidelity audit for the critical NEX326 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .model import (
    REPTILE_INITIAL_STEP,
    REPTILE_INNER_RATE,
    REPTILE_INNER_STEPS,
    REPTILE_META_EPOCHS,
)
from .specification import ExperimentSpec, load_experiment_spec


FIDELITY_SCHEMA_VERSION = "nex326-fidelity-audit-v1"


ROUTE_EXPECTATIONS = (
    (7, "full", "estimator", "crps_energy"),
    (8, "qmle", "estimator", "qmle"),
    (9, "mixed", "estimator", "mixed"),
    (9, "pure_es", "estimator", "pure_es"),
    (11, "full", "transfer", "full_finetune"),
    (12, "scratch", "transfer", "scratch"),
    (13, "animal_pretrain", "transfer", "animal_pretrain"),
    (14, "reptile", "transfer", "meta_reptile"),
    (15, "drift_only", "finetune", "drift_only"),
    (15, "two_step", "finetune", "two_step"),
    (22, "doob", "bridge", "doob"),
    (22, "sb", "bridge", "gaussian_schrodinger"),
    (22, "soft_endpoint", "bridge", "soft_endpoint"),
)


def _execution_configs(spec: ExperimentSpec) -> dict[tuple[int, str], dict[str, object]]:
    return {
        (arm.arm_id, str(subconfig["subconfig_id"])): {
            **spec.full_components,
            **subconfig.get("components", {}),
        }
        for arm, subconfig in spec.executions
    }


def build_fidelity_report(spec: ExperimentSpec | None = None) -> dict[str, object]:
    """Audit routing and declare the scope of each bounded implementation."""
    selected = spec or load_experiment_spec()
    configs = _execution_configs(selected)
    route_checks: list[dict[str, object]] = []
    for arm_id, subconfig_id, component, expected in ROUTE_EXPECTATIONS:
        actual = configs.get((arm_id, subconfig_id), {}).get(component)
        route_checks.append(
            {
                "arm_id": arm_id,
                "subconfig_id": subconfig_id,
                "component": component,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )

    capabilities = [
        {
            "arms": list(range(1, 23)),
            "capability": "execution_environment_identity",
            "implementation": "source bundle plus observed Python and direct dependency versions",
            "fidelity": "exact_direct_environment_identity",
            "remaining_limit": "transitive dependencies, OS, BLAS, and hardware are recorded only indirectly",
        },
        {
            "arms": [7],
            "capability": "energy_objective",
            "implementation": "validation-selected covariance scale grid",
            "fidelity": "bounded_surrogate",
            "remaining_limit": "the objective optimizes covariance scale while retaining fitted drift",
        },
        {
            "arms": [9],
            "capability": "mixed_and_pure_energy_objectives",
            "implementation": "joint validation grid over drift interpolation and covariance scale",
            "fidelity": "bounded_joint_objective_optimization",
            "remaining_limit": "the bounded grid is not an unconstrained gradient optimizer over every model parameter",
        },
        {
            "arms": [8],
            "capability": "gaussian_qmle",
            "implementation": "ridge-stabilized linear Gaussian closed-form fit",
            "fidelity": "exact_for_bounded_linear_gaussian_model",
            "remaining_limit": "not a nonlinear or large-cohort estimator",
        },
        {
            "arms": [11, 12, 13, 15],
            "capability": "transfer_and_adaptation",
            "implementation": "distinct pretrain, scratch, drift-only, and two-step parameter paths",
            "fidelity": "implemented_for_bounded_model",
            "remaining_limit": "animal execution remains data-dependent on the licensed split",
        },
        {
            "arms": [14],
            "capability": "first_order_reptile",
            "implementation": {
                "inner_solution": "task-local gradient steps from current initialization",
                "outer_update": "sequential first-order interpolation",
                "meta_epochs": REPTILE_META_EPOCHS,
                "initial_step": REPTILE_INITIAL_STEP,
                "inner_steps": REPTILE_INNER_STEPS,
                "inner_rate": REPTILE_INNER_RATE,
            },
            "fidelity": "implemented_first_order_reptile_for_bounded_model",
            "remaining_limit": "the inner loss is the bounded linear-Gaussian task loss, not a large neural model",
        },
        {
            "arms": [17],
            "capability": "spatial_terrain_conditioning",
            "implementation": {
                "fit_lookup": "SRTM sampled at observed positions",
                "inference_lookup": "SRTM sampled at each simulated position",
                "future_route_point_index_used": False,
                "fp_propagation": "local_gaussian_mean_closure",
                "comparison_guard": "evaluation segment-ID fingerprint must match Full reference",
            },
            "fidelity": "implemented_spatial_condition_with_bounded_fp_closure",
            "remaining_limit": "weather is unavailable; nonlinear C(x,y) finite propagation uses a declared local Gaussian mean closure",
        },
        {
            "arms": [22],
            "capability": "endpoint_conditioning",
            "implementation": {
                "paths": ["doob", "finite_particle_schrodinger", "soft_endpoint"],
                "schrodinger_solver": "log-domain Sinkhorn/IPF endpoint coupling",
                "reference_process": "isotropic Brownian",
                "path_sampler": "conditional Brownian bridge",
                "prior_contract": "exact-coverage independent feed adapter",
                "failure_policy": "non-convergence raises RunError without fallback",
            },
            "fidelity": "implemented_finite_particle_schrodinger_bridge_extension",
            "remaining_limit": "expert-assisted endpoint priors are outside the current PIRC-19 core evidence; no mock feed is used in core receipts",
        },
    ]
    failures = [check for check in route_checks if not check["passed"]]
    return {
        "schema_version": FIDELITY_SCHEMA_VERSION,
        "experiment_id": selected.experiment_id,
        "spec_version": selected.spec_version,
        "overall_status": (
            "routing_failed" if failures else "bounded_reconstruction_with_declared_approximations"
        ),
        "paper_equivalent": False,
        "route_checks": route_checks,
        "capabilities": capabilities,
        "failed_route_count": len(failures),
    }


def write_fidelity_report(destination: Path, spec_path: Path | None = None) -> dict[str, object]:
    report = build_fidelity_report(load_experiment_spec(spec_path) if spec_path else None)
    if report["failed_route_count"]:
        raise ValueError("NEX326 fidelity routing audit failed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = write_fidelity_report(args.output, args.spec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_fidelity_report", "main", "write_fidelity_report"]
