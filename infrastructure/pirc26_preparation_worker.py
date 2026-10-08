"""Internal bounded DSDE conversion/normalizer worker, never a standalone job."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from application.pirc26_dsde import (SourcePair, ProjectionSpec, materialize_source_pair, bind_conversion, MAX_BYTES)
from application.pirc26_preparation import (VERSION, NORMALIZER_POLICY, MAX_SOURCE_BYTES, MAX_RESULT_BYTES,
    MAX_OBSERVATIONS, validate_settings, train_binding, require, _validate_result)
from experiments.pirc25.affine import code_hash
from infrastructure.pirc26_worker_control import require_owned_worker
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import atomic_write, digest, encode, ResearchError
from infrastructure.research_store import ResearchStore
from infrastructure.research_control import read_frame


def read_input(root, name, limit):
    with opened_regular_file(root, root / name, maximum_bytes=limit) as (stream, size, _):
        content = stream.read(size + 1)
        require(len(content) == size <= limit, "worker handoff changed or exceeded quota", "CORRUPT_ARTIFACT")
    return content


def execute(output):
    # No raw source, NumPy/Arrow decode or fitting before actual running
    # attempt, funded reservation, wrapper PID and native deadline are proven.
    control = require_owned_worker(output)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    root = output.parent
    request = json.loads(read_input(root, "preparation-request.json", 2 * 1024 * 1024))
    require(request["schema_version"] == VERSION and request["computation_ref"]["attempt_id"] == control.attempt_id,
            "preparation request/owned attempt differs")
    identity = read_frame(output.parents[2] / "store.json", 16384)
    store = ResearchStore(output.parents[3], identity["store_id"])
    require(request == store.manifest(request["computation_ref"]["request_manifest_id"]),
            "worker transport differs from actual frozen owner request", "UNAUTHORIZED_DATA")
    validate_settings(request["settings"])
    require(request["runtime_code_hash"] == code_hash()
        and request["train_binding"] == train_binding(request["sources"], request["settings"]),
        "preparation frozen code/train membership differs")
    settings, total_bytes, observations, materialized = request["settings"], 0, 0, []
    for i, source in enumerate(request["sources"]):
        require(not control.requested(), "preparation reached actual owner soft stop", "INTERRUPTED")
        feature = read_input(root, str(i) + "-features.parquet", MAX_BYTES)
        condition = read_input(root, str(i) + "-conditions.parquet", MAX_BYTES)
        total_bytes += len(feature) + len(condition)
        require(total_bytes <= MAX_SOURCE_BYTES, "joint worker source quota", "RESOURCE_PLAN_REJECTED")
        pair = SourcePair(source["protocol_hash"], source["authorization_hash"],
            encode(source["feature"]).decode(), encode(source["condition"]).decode(), feature, condition)
        item = materialize_source_pair(pair, settings["feature_spec"], settings["benchmark_binding"],
            ProjectionSpec(**settings["projection"]), block_id=source["selection"]["output_block_id"],
            duplicate_policy=settings["duplicate_policy"], max_observations=MAX_OBSERVATIONS)
        observations += sum(len(s["time"]) for s in item["document"]["segments"])
        require(observations <= MAX_OBSERVATIONS, "joint decoded preparation population quota", "RESOURCE_PLAN_REJECTED")
        materialized.append(item)
        del pair, feature, condition
    import numpy as np
    population = []
    for source, item in zip(request["sources"], materialized):
        if source["split_role"] != "train":
            continue
        for segment in item["document"]["segments"]:
            require(not control.requested(), "normalizer reached actual owner soft stop", "INTERRUPTED")
            positions = np.asarray(segment["position"], dtype=np.float64)
            times = np.asarray(segment["time"], dtype=np.float64)
            velocity = np.diff(positions, axis=0) / np.diff(times)[:, None]
            context = np.asarray(segment["condition"], dtype=np.float64)[1:]
            population.append(np.concatenate((positions[1:], velocity, context), axis=1))
    require(bool(population), "authorized causal training population absent", "UNAUTHORIZED_DATA")
    population = np.concatenate(population, axis=0)
    require(np.isfinite(population).all(), "normalizer train states nonfinite", "NONFINITE")
    means, scales = population.mean(axis=0), population.std(axis=0, ddof=0)
    require(np.isfinite(means).all() and np.isfinite(scales).all(), "normalizer statistics nonfinite", "NONFINITE")
    scales[scales == 0] = NORMALIZER_POLICY["zero_variance_scale"]
    normalizer = {"schema_version": "pirc26-fitted-normalizer-v1", "train_binding_hash": digest(request["train_binding"]),
        "policy": settings["normalizer_policy"], "context_hash": digest(settings["context_binding"]),
        "means": means.tolist(), "scales": scales.tolist(), "observations": len(population),
        "independent_block_ids": sorted({s["independent_block_id"] for s in request["sources"] if s["split_role"] == "train"})}
    blocks = []
    for item in materialized:
        bound = bind_conversion(item, train_binding_hash=normalizer["train_binding_hash"],
            normalizer_hash=digest(normalizer), context_hash=normalizer["context_hash"])
        blocks.append({k: bound[k] for k in ("document", "provenance")})
    require(not control.requested(), "preparation reached actual owner soft stop before publication", "INTERRUPTED")
    require(request["runtime_code_hash"] == code_hash(), "preparation implementation moved")
    result = {"schema_version": VERSION, "request_hash": digest(request), "computation_ref": request["computation_ref"],
        "train_binding": request["train_binding"], "normalizer": normalizer, "blocks": blocks,
        "scientific_qualification": "not-established"}
    from application.pirc26_population import training_population
    result["training_population"] = training_population(blocks, request["train_binding"], normalizer,
                                                       study_id=request["original_study_id"])
    _validate_result(result, request)
    content = encode(result)
    require(len(content) <= MAX_RESULT_BYTES, "worker preparation result byte quota", "RESOURCE_PLAN_REJECTED")
    atomic_write(output, content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(2)
    try:
        execute(Path(sys.argv[1]).absolute())
    except ResearchError as exc:
        # No scientific bytes, raw path, coordinates or private exception text.
        print(exc.code, file=sys.stderr)
        raise SystemExit(1)
