"""Leakage-safe NEX326 execution overrides for PIRC-20 cohorts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Mapping, Sequence

import numpy as np

from . import runner as runner_module
from .cohort import Segment
from .model import ModelState
from .runner import ConditionResolver, NEX326Runner, Prediction


RUNTIME_VERSION = "pirc20-nex326-runtime-v1"
_CORE_PREDICT_SEGMENTS = runner_module.predict_segments
_PATCH_LOCK = threading.RLock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extend_implementation_identity(identity: Mapping[str, object]) -> dict[str, object]:
    extended = dict(identity)
    files = [dict(item) for item in identity["files"]]  # type: ignore[index]
    files.append({"path": Path(__file__).name, "sha256": _sha256(Path(__file__))})
    source_bundle = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    runtime = dict(identity["runtime"])  # type: ignore[arg-type]
    execution_identity = hashlib.sha256(
        json.dumps(
            {"source_bundle_sha256": source_bundle, "runtime": runtime},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    extended.update(
        {
            "source_bundle_sha256": source_bundle,
            "execution_identity_sha256": execution_identity,
            "files": files,
            "pirc20_runtime_version": RUNTIME_VERSION,
        }
    )
    return extended


def predict_segments_without_endpoint_start(
    model: ModelState,
    segments: Sequence[Segment],
    config: Mapping[str, object],
    *,
    seed: int,
    n_samples: int,
    condition_resolver: ConditionResolver | None = None,
) -> tuple[Prediction, ...]:
    """Predict without ever initializing a rollout from the evaluation endpoint."""

    rng = np.random.default_rng(seed)
    predictions: list[Prediction] = []
    for segment in segments:
        if len(segment.time) < 2:
            raise runner_module.RunError(
                f"prediction requires at least two time points: {segment.segment_id}"
            )
        condition_field = (
            condition_resolver.for_segment(segment)
            if condition_resolver is not None
            else None
        )
        start = min(max(1, len(segment.time) // 2 - 1), len(segment.time) - 2)
        poa = str(config.get("poa", "fp"))
        integrator = str(config.get("integrator", "split"))
        if poa == "fp":
            samples = runner_module._fp_rollout(
                model, segment, start, integrator, n_samples, rng, condition_field
            )
        else:
            samples = runner_module._mc_rollout(
                model, segment, start, integrator, n_samples, rng, condition_field
            )
        bridged, before, after, bridge_path, bridge_diagnostics = (
            runner_module._apply_bridge(
                samples,
                segment,
                str(config.get("bridge", "none")),
                rng,
                epsilon_scale=float(config.get("bridge_epsilon_scale", 0.5)),
                time_steps=int(config.get("bridge_time_steps", 8)),
                max_iterations=int(config.get("bridge_sinkhorn_iterations", 500)),
                tolerance=float(config.get("bridge_sinkhorn_tolerance", 1e-8)),
            )
        )
        predictions.append(
            Prediction(
                segment_id=segment.segment_id,
                target=segment.state[-1].copy(),
                samples=bridged,
                prior_mean=segment.endpoint_prior_mean,
                prior_source=segment.endpoint_prior_source,
                unbridged_prior_distance=before,
                bridged_prior_distance=after,
                bridge_path=bridge_path,
                bridge_diagnostics=bridge_diagnostics,
            )
        )
    return tuple(predictions)


class PIRC20NEX326Runner(NEX326Runner):
    """NEX326 runner whose prediction boundary cannot equal evaluation truth."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not str(self.cohort.data_version).startswith("pirc20-"):
            raise ValueError("PIRC20NEX326Runner requires a PIRC-20 cohort")
        self.implementation = _extend_implementation_identity(self.implementation)

    def run_one(self, arm, subconfig):
        with _PATCH_LOCK:
            if runner_module.predict_segments is not _CORE_PREDICT_SEGMENTS:
                raise runner_module.RunError("NEX326 prediction function is already overridden")
            runner_module.predict_segments = predict_segments_without_endpoint_start
            try:
                return super().run_one(arm, subconfig)
            finally:
                runner_module.predict_segments = _CORE_PREDICT_SEGMENTS


__all__ = [
    "PIRC20NEX326Runner",
    "RUNTIME_VERSION",
    "predict_segments_without_endpoint_start",
]
