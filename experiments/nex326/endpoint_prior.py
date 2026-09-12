"""Attach an independently versioned endpoint-prior feed to an NEX326 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .cohort import load_cohort


PRIOR_SCHEMA_VERSION = "nex326-endpoint-prior-v1"
REQUEST_SCHEMA_VERSION = "nex326-endpoint-prior-request-v1"


class EndpointPriorError(ValueError):
    """An endpoint-prior feed is incomplete, ambiguous, or not independent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EndpointPriorError(f"{label} must contain a JSON object")
    return payload


def _validated_records(
    feed: Mapping[str, object],
    expected_segment_ids: set[str],
) -> dict[str, dict[str, object]]:
    if feed.get("schema_version") != PRIOR_SCHEMA_VERSION:
        raise EndpointPriorError("unsupported endpoint-prior schema")
    if not feed.get("feed_id") or not feed.get("data_version"):
        raise EndpointPriorError("endpoint-prior feed requires identity and version")
    attestation = feed.get("independence_attestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("derived_from_evaluation_truth") is not False
        or not attestation.get("method")
        or not attestation.get("responsible_party")
    ):
        raise EndpointPriorError(
            "endpoint-prior feed requires a non-truth-derived independence attestation"
        )
    records = feed.get("records")
    if not isinstance(records, list) or not records:
        raise EndpointPriorError("endpoint-prior feed has no records")
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not record.get("segment_id"):
            raise EndpointPriorError("endpoint-prior record lacks segment identity")
        segment_id = str(record["segment_id"])
        if segment_id in indexed:
            raise EndpointPriorError(f"duplicate endpoint prior for {segment_id}")
        try:
            mean = np.asarray(record.get("mean"), dtype=float)
            covariance = np.asarray(record.get("covariance"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise EndpointPriorError(
                f"endpoint prior values are not numeric for {segment_id}"
            ) from exc
        if mean.shape != (2,) or not np.isfinite(mean).all():
            raise EndpointPriorError(f"endpoint prior mean is invalid for {segment_id}")
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise EndpointPriorError(f"endpoint prior covariance is invalid for {segment_id}")
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise EndpointPriorError(f"endpoint prior covariance is not symmetric for {segment_id}")
        if float(np.linalg.eigvalsh(covariance).min()) <= 0.0:
            raise EndpointPriorError(
                f"endpoint prior covariance is not positive definite for {segment_id}"
            )
        indexed[segment_id] = {
            "mean": mean.tolist(),
            "covariance": covariance.tolist(),
        }
    observed = set(indexed)
    if observed != expected_segment_ids:
        raise EndpointPriorError(
            "endpoint-prior coverage must exactly match evaluation segments; "
            f"missing={sorted(expected_segment_ids - observed)}, "
            f"unexpected={sorted(observed - expected_segment_ids)}"
        )
    return indexed


def attach_endpoint_priors(
    cohort_path: Path | str,
    prior_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Validate and attach one independent prior to every evaluation segment."""
    cohort_file = Path(cohort_path)
    prior_file = Path(prior_path)
    destination = Path(output_path)
    if not cohort_file.is_file() or not prior_file.is_file():
        raise EndpointPriorError("cohort and endpoint-prior feed must both exist")
    if destination.exists():
        raise EndpointPriorError(f"output already exists: {destination}")
    cohort = load_cohort(cohort_file)
    payload = _load_mapping(cohort_file, "cohort")
    if "generator" in payload or not isinstance(payload.get("splits"), Mapping):
        raise EndpointPriorError("endpoint priors require an explicit materialized cohort")
    evaluation = payload["splits"].get("evaluation")
    if not isinstance(evaluation, list) or not evaluation:
        raise EndpointPriorError("cohort has no explicit evaluation segments")
    if any(isinstance(item, Mapping) and item.get("endpoint_prior") for item in evaluation):
        raise EndpointPriorError("cohort already contains endpoint priors")
    expected_ids = {str(segment.segment_id) for segment in cohort.splits["evaluation"]}
    feed = _load_mapping(prior_file, "endpoint-prior feed")
    records = _validated_records(feed, expected_ids)
    source_name = f"{feed['feed_id']}:{feed['data_version']}"
    for segment in evaluation:
        segment_id = str(segment["segment_id"])
        segment["endpoint_prior"] = {
            **records[segment_id],
            "source": source_name,
            "derived_from_evaluation_truth": False,
        }

    payload["dataset_id"] = f"{payload['dataset_id']}+ENDPOINT-PRIOR"
    payload["data_version"] = (
        f"{payload['data_version']}+endpoint-{feed['data_version']}"
    )
    payload["endpoint_prior_status"] = {
        "status": "available",
        "feed_id": feed["feed_id"],
        "data_version": feed["data_version"],
        "record_count": len(records),
        "independence_attestation": feed["independence_attestation"],
    }
    source = payload.setdefault("source", {})
    if not isinstance(source, dict):
        raise EndpointPriorError("cohort source provenance must be an object")
    source["endpoint_prior"] = {
        "file": prior_file.name,
        "sha256": _sha256(prior_file),
        "base_cohort_file": cohort_file.name,
        "base_cohort_sha256": _sha256(cohort_file),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_cohort(destination)
    return payload


def write_endpoint_prior_request(
    cohort_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Export only the observed prefix needed by an independent prior provider."""
    cohort_file = Path(cohort_path)
    destination = Path(output_path)
    if not cohort_file.is_file():
        raise EndpointPriorError("cohort file does not exist")
    if destination.exists():
        raise EndpointPriorError(f"output already exists: {destination}")
    cohort = load_cohort(cohort_file)
    source = _load_mapping(cohort_file, "cohort")
    if "generator" in source or not isinstance(source.get("splits"), Mapping):
        raise EndpointPriorError("endpoint-prior requests require a materialized cohort")
    evaluation = cohort.splits["evaluation"]
    if not evaluation:
        raise EndpointPriorError("cohort has no evaluation segments")
    if any(segment.endpoint_prior_mean is not None for segment in evaluation):
        raise EndpointPriorError("cohort already contains endpoint priors")

    records: list[dict[str, object]] = []
    for segment in evaluation:
        cutoff = max(1, len(segment.time) // 2 - 1)
        records.append(
            {
                "segment_id": segment.segment_id,
                "source_domain": segment.source_domain,
                "region": segment.region,
                "observed_time": segment.time[: cutoff + 1].tolist(),
                "observed_state": segment.state[: cutoff + 1].tolist(),
                "observed_conditions": {
                    name: values[: cutoff + 1].tolist()
                    for name, values in sorted(segment.conditions.items())
                },
                "forecast_horizon": float(segment.time[-1] - segment.time[cutoff]),
            }
        )
    request: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "purpose": "independent_endpoint_prior_generation",
        "cohort": {
            "dataset_id": cohort.dataset_id,
            "data_version": cohort.data_version,
            "fingerprint": cohort.fingerprint,
            "file": cohort_file.name,
            "sha256": _sha256(cohort_file),
        },
        "state_space": "cohort_native_2d_xy",
        "inference_cut_rule": "max(1, segment_length // 2 - 1)",
        "record_count": len(records),
        "provider_requirements": {
            "output_schema": PRIOR_SCHEMA_VERSION,
            "one_record_per_segment": True,
            "required_distribution": "finite 2D mean and symmetric positive-definite 2x2 covariance",
            "prohibited_inputs": [
                "evaluation states after the final observed_time entry",
                "evaluation endpoint targets",
            ],
            "required_attestation": [
                "derived_from_evaluation_truth=false",
                "method",
                "responsible_party",
            ],
        },
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return request


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prior", type=Path)
    mode.add_argument("--emit-request", action="store_true")
    args = parser.parse_args(argv)
    if args.emit_request:
        request = write_endpoint_prior_request(args.cohort, args.output)
        print(
            json.dumps(
                {
                    "schema_version": request["schema_version"],
                    "record_count": request["record_count"],
                    "cohort": request["cohort"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    assert args.prior is not None
    payload = attach_endpoint_priors(args.cohort, args.prior, args.output)
    print(
        json.dumps(
            {
                "dataset_id": payload["dataset_id"],
                "data_version": payload["data_version"],
                "endpoint_prior_status": payload["endpoint_prior_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EndpointPriorError",
    "attach_endpoint_priors",
    "main",
    "write_endpoint_prior_request",
]
