"""Selection-only transport: original units, no fitting or read permission.

Serialize already validated owned-worker geometry, never re-estimate a frame,
normalizer or model. Metadata lookup must not open the preparation result,
which contains coordinates from both roles. Final/test are not supported here.
"""

from copy import deepcopy
import hashlib

from application.pirc26_population import (population_source, require, MAX_BYTES,
    MAX_SEGMENTS, MAX_OBSERVATIONS)
from infrastructure.research_store import digest, encode

VERSION = "pirc26-selection-population-v1"


def selection_populations(blocks, normalizer, *, study_id):
    """All declared files/segments of each original selection unit, in order."""
    from application.pirc26_preparation import _validate_geometry
    require(type(blocks) is list and 1 <= len(blocks) <= 16, "bounded prepared population required")
    require(type(study_id) is str and 0 < len(study_id) <= 128, "selection publication study required")
    groups = {}
    for block in blocks:
        p = block["provenance"]
        if p["split_role"] == "train":
            continue
        require(p["split_role"] == "selection" and p["purpose"] == "select",
            "selection transport cannot admit other roles", "UNAUTHORIZED_DATA")
        unit = p["feature"]["independent_block_id"]
        require(unit not in normalizer["independent_block_ids"], "selection unit overlaps training", "UNAUTHORIZED_DATA")
        groups.setdefault(unit, []).append(block)
    populations = []
    for unit, members in groups.items():
        metadata_members, segments, mapping, seen = [], [], [], set()
        document = {k: deepcopy(v) for k, v in members[0]["document"].items() if k != "segments"}
        observations = 0
        for block in members:
            d, p = block["document"], block["provenance"]
            _validate_geometry(d, len(normalizer["means"]))
            require(all(d[k] == document[k] for k in document if k != "block_id")
                and d["train_binding_hash"] == normalizer["train_binding_hash"]
                and d["normalizer_hash"] == digest(normalizer) and d["context_hash"] == normalizer["context_hash"],
                "selection changed the train-fitted transform")
            require(d["block_id"] not in seen and len(d["segments"]) == len(p["membership"]),
                "selection file/segment identity differs")
            seen.add(d["block_id"])
            metadata_members.append({"source_output_block_id": d["block_id"],
                "source_document_sha256": hashlib.sha256(encode(d)).hexdigest(),
                "feature_sha256": p["feature"]["sha256"], "condition_sha256": p["condition"]["sha256"],
                "independent_block_id": unit, "source_membership_hash": digest(p["membership"])})
            for segment, membership in zip(d["segments"], p["membership"]):
                segment_id = "select-" + digest([d["block_id"], segment["segment_id"]])
                n = len(segment["time"])
                require(membership["segment_id"] == segment["segment_id"] and membership["independent_block_id"] == unit
                    and len(membership["point_ids"]) == len(membership["source_point_indices"]) == n,
                    "selection original point membership differs")
                segments.append({**deepcopy(segment), "segment_id": segment_id})
                mapping.append({"pooled_segment_id": segment_id, "source_output_block_id": d["block_id"],
                    "source_segment_id": segment["segment_id"], "independent_block_id": unit,
                    "point_membership_hash": digest(membership), "observations": n})
                observations += n
                require(len(segments) <= MAX_SEGMENTS and observations <= MAX_OBSERVATIONS
                    and observations * (len(normalizer["means"]) + 2) <= 1048576,
                    "complete selection unit exceeds decoded quota", "RESOURCE_PLAN_REJECTED")
        identity = digest([study_id, unit, metadata_members, digest(normalizer)])
        document.update(block_id="selection-population-" + identity, segments=segments)
        content = encode(document)
        require(len(content) <= MAX_BYTES, "complete selection unit exceeds transport quota", "RESOURCE_PLAN_REJECTED")
        metadata = {"schema_version": VERSION, "source_study_id": study_id, "block_id": document["block_id"],
            "independent_block_id": unit, "population_kind": "all-declared-selection-files-one-original-unit",
            "purpose": "select", "coordinate_frame": document["coordinate_frame"],
            "train_binding_hash": normalizer["train_binding_hash"], "normalizer_hash": digest(normalizer),
            "context_hash": normalizer["context_hash"], "members": metadata_members, "segments": mapping,
            "observations": observations, "content_sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content), "scientific_qualification": "not-established"}
        populations.append({"document": document, "provenance": metadata})
    return populations


def selection_source(store, reference, independent_block_id, *, consumer_study_id=None, arm=None):
    """Fresh source metadata only; an actual consumer needs its own grant/read."""
    from application.pirc26_preparation import _sources, _completion, _expiry
    with store._read_transaction():
        train = population_source(store, reference)  # Actual preparation, cost, train-only transform.
        proof = store._manifest(reference["manifest_id"])
        request = store._manifest(reference["request_manifest_id"])
        original = store._manifest("study-" + request["original_study_id"])["spec"]
        run = store._manifest("run-" + reference["run_id"])
        source_arm = next(a for a in original["arms"] if a["arm_id"] == run["arm_id"])
        arm = source_arm if arm is None else arm
        require(arm in original["arms"], "selection consumer created another arm", "CONTRACT_MISMATCH")
        raw = [s for s in request["sources"] if s["split_role"] == "selection"
               and s["independent_block_id"] == independent_block_id]
        require(bool(raw), "selection unit absent from actual frozen preparation", "UNAUTHORIZED_DATA")
        consumer = original["study_id"] if consumer_study_id is None else consumer_study_id
        def authority():
            require(_sources(store, original, arm, [s["selection"] for s in raw], require_train=False) == raw,
                "original selection source authority moved", "UNAUTHORIZED_DATA")
            for item in raw:
                selection = item["selection"]
                grant = store.authorization(selection["authorization_id"], version=selection["authorization_version"])
                require(consumer == original["study_id"] or consumer in grant.get("consumer_study_ids", []),
                    "raw selection grant does not permit this consumer", "UNAUTHORIZED_DATA")
        authority()
        candidates = [p for p in proof["selection_populations"]
                      if p["provenance"]["independent_block_id"] == independent_block_id]
        require(len(candidates) == 1 and set(candidates[0]) == {"artifact_id", "provenance"},
            "unique closed selection unit receipt absent", "CORRUPT_ARTIFACT")
        selected = candidates[0]
        metadata = selected["provenance"]
        normalizer = train["normalizer"]
        artifact = store._manifest("artifact-" + selected["artifact_id"])
        require(metadata["schema_version"] == VERSION and metadata["source_study_id"] == original["study_id"]
            and metadata["purpose"] == "select" and metadata["scientific_qualification"] == "not-established"
            and metadata["population_kind"] == "all-declared-selection-files-one-original-unit"
            and metadata["independent_block_id"] not in normalizer["independent_block_ids"]
            and all(metadata[k] == train["provenance"][k] for k in (
                "coordinate_frame", "train_binding_hash", "normalizer_hash", "context_hash"))
            and [(m["feature_sha256"], m["condition_sha256"], m["independent_block_id"], m["source_output_block_id"])
                 for m in metadata["members"]] == [(s["feature"]["sha256"], s["condition"]["sha256"],
                    independent_block_id, s["selection"]["output_block_id"]) for s in raw]
            and metadata["block_id"] == "selection-population-" + digest([
                original["study_id"], independent_block_id, metadata["members"], digest(normalizer)])
            and artifact["artifact_id"] == artifact["sha256"] == selected["artifact_id"] == metadata["content_sha256"]
            and artifact["size_bytes"] == metadata["size_bytes"] <= MAX_BYTES
            and artifact["role"] == "selection-population" and artifact["study_id"] == original["study_id"]
            and artifact["block_ids"] == [independent_block_id],
            "complete original selection unit/artifact binding differs", "CORRUPT_ARTIFACT")
        grants = _completion(store, original, arm, raw, require_train=False)
        store._read_completion(authority, lambda: _expiry(grants))
    _expiry(grants)
    entry = {"block_id": metadata["block_id"], "dataset_id": "pirc26-derived-selection-population",
        "release_id": VERSION, "source_block_id": independent_block_id, "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"], "path": artifact["artifact_id"], "split_role": "selection", "fit_scope": False,
        "selection_population_ref": deepcopy(reference), "selection_population_hash": digest(metadata),
        "independent_block_ids": [independent_block_id], "population_kind": metadata["population_kind"]}
    return {"protocol_entry": entry, "data_root": str(store.path / "artifacts"),
            "provenance": deepcopy(metadata), "normalizer": deepcopy(normalizer)}
