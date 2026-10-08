"""Complete bounded train transport; a computational pool is not a sample.

Construction belongs to the managed preparation worker. Owner validation is
structural/integrity checking only. Metadata lookup never reads coordinates,
grants a consumer permission or qualifies a fitted/scientific model.
"""

from copy import deepcopy
import hashlib

from infrastructure.research_store import ResearchError, digest, encode
from application.pirc26_fragment_contract import fragment_summary, population_fragments, validate_population_fragments

VERSION = "pirc26-training-population-v1"
MAX_BYTES = 8 * 1024 * 1024
MAX_SEGMENTS = 256
MAX_OBSERVATIONS = 16_912


def require(ok, message, code="CONTRACT_MISMATCH"):
    if not ok:
        raise ResearchError(code, message)


def training_population(blocks, train_binding, normalizer, *, study_id):
    """Preserve every ordered train file/segment, never concatenate paths."""
    from application.pirc26_preparation import _validate_geometry
    require(type(blocks) is list and 1 <= len(blocks) <= 16, "bounded prepared population required")
    require(type(study_id) is str and 0 < len(study_id) <= 128, "explicit population publication study required")
    train = [b for b in blocks if b["provenance"]["split_role"] == "train"]
    require(bool(train), "complete train population absent", "UNAUTHORIZED_DATA")
    expected = train_binding["members"]
    require(len(train) == len(expected) and all(
        b["provenance"]["feature"] == member["feature"]
        and b["provenance"]["condition"] == member["condition"]
        and b["provenance"]["feature"]["independent_block_id"] == member["independent_block_id"]
        and b["provenance"]["purpose"] == "fit" for b, member in zip(train, expected)),
        "prepared train files differ from complete frozen membership", "UNAUTHORIZED_DATA")
    width = len(normalizer["means"])
    document = {k: deepcopy(v) for k, v in train[0]["document"].items() if k != "segments"}
    # Publication scope is separate from numerical fit identity. Same-study
    # preparations may share an artifact; different studies must not collide
    # with content-addressed, immutable artifact ownership metadata.
    document["block_id"] = "train-population-" + digest([study_id, digest(train_binding)])
    document["segments"] = []
    members, segment_members, seen = [], [], set()
    observations, transitions = 0, 0
    for block in train:
        source, provenance = block["document"], block["provenance"]
        _validate_geometry(source, width)
        summary = fragment_summary(block, expected_policy=train_binding.get("fragment_policy", "reject"))
        require(all(source[k] == document[k] for k in document if k not in {"block_id", "segments"})
            and source["train_binding_hash"] == digest(train_binding)
            and source["normalizer_hash"] == digest(normalizer)
            and source["context_hash"] == normalizer["context_hash"], "population transform binding differs")
        source_id = source["block_id"]
        require(source_id not in seen, "duplicate prepared train file")
        seen.add(source_id)
        members.append({"source_output_block_id": source_id,
            "source_document_sha256": hashlib.sha256(encode(source)).hexdigest(),
            "feature_sha256": provenance["feature"]["sha256"],
            "condition_sha256": provenance["condition"]["sha256"],
            "independent_block_id": provenance["feature"]["independent_block_id"],
            "source_membership_hash": digest(provenance["membership"])})
        if summary is not None:
            members[-1]["fragment_disposition"] = summary
        require(len(source["segments"]) == len(provenance["membership"]), "source segment membership differs")
        for segment, membership in zip(source["segments"], provenance["membership"]):
            # Explicit map makes this namespace reversible without guessing or
            # dropping original identities. Each causal path remains separate.
            segment_id = "pool-" + digest([source_id, segment["segment_id"]])
            document["segments"].append({**deepcopy(segment), "segment_id": segment_id})
            require(len(document["segments"]) <= MAX_SEGMENTS, "complete train segment quota", "RESOURCE_PLAN_REJECTED")
            n = len(segment["time"])
            require(membership["segment_id"] == segment["segment_id"]
                and membership["independent_block_id"] == members[-1]["independent_block_id"]
                and len(membership["point_ids"]) == len(membership["source_point_indices"]) == n,
                "source points or independent unit differ")
            segment_members.append({"pooled_segment_id": segment_id, "source_output_block_id": source_id,
                "source_segment_id": segment["segment_id"], "independent_block_id": membership["independent_block_id"],
                "point_membership_hash": digest(membership), "observations": n, "transitions": n - 2})
            observations += n
            transitions += n - 2  # backward velocity drops the first point
            require(observations <= MAX_OBSERVATIONS and observations * (width + 2) <= 1048576,
                "complete train decoded population quota", "RESOURCE_PLAN_REJECTED")
    require(observations - len(segment_members) == normalizer["observations"], "normalizer causal population differs")
    content = encode(document)
    require(len(content) <= MAX_BYTES, "complete train transport byte quota", "RESOURCE_PLAN_REJECTED")
    metadata = {"schema_version": VERSION, "block_id": document["block_id"], "source_study_id": study_id,
        "population_kind": "computational-train-only-not-independent-sample", "purpose": "fit",
        "coordinate_frame": document["coordinate_frame"], "train_binding_hash": digest(train_binding),
        "normalizer_hash": digest(normalizer), "context_hash": normalizer["context_hash"],
        "members": members, "segments": segment_members,
        "independent_block_ids": sorted({m["independent_block_id"] for m in members}),
        "observations": observations, "transitions": transitions,
        "content_sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
        "scientific_qualification": "not-established"}
    fragments = population_fragments(members)
    if fragments is not None:
        metadata["fragment_eligibility"] = fragments
    return {"document": document, "provenance": metadata}


def validate_population(value, blocks, train_binding, normalizer, *, study_id):
    """Exact structural equality includes all geometry and membership order."""
    require(value == training_population(blocks, train_binding, normalizer, study_id=study_id),
            "complete training population geometry/membership substituted", "CORRUPT_ARTIFACT")


def population_source(store, reference):
    """Fresh metadata-only lookup of the separately published train artifact.

    Caller must independently register a consumer protocol and lawful grant,
    then read through the original exposure ledger. No source payload or the
    preparation's selection-bearing result is read here.
    """
    from application.pirc26_preparation import _sources, _verify_reads, _completion, _expiry
    from application.research_computation import _verify_job_cost
    from experiments.pirc25.affine import code_hash
    with store._read_transaction():
        proof = store._manifest(reference["manifest_id"])
        request = store._manifest(reference["request_manifest_id"])
        attempt = store._attempts().get(reference["attempt_id"])
        require(attempt and attempt["state"] == "SUCCEEDED" and attempt["run_id"] == reference["run_id"]
            and proof["schema_version"] == "pirc26-preparation-receipt-v2"
            and proof["computation_ref"] == request["computation_ref"] == reference
            and proof["request_hash"] == digest(request) and proof["result_artifact_id"] == attempt["artifact_id"],
            "training population preparation receipt differs", "CORRUPT_ARTIFACT")
        run = store._manifest("run-" + attempt["run_id"])
        derived = store._manifest("study-" + run["study_id"])["spec"]
        original = store._manifest("study-" + request["original_study_id"])["spec"]
        arm = next(a for a in original["arms"] if a["arm_id"] == run["arm_id"])
        require(derived["arms"] == [arm] and derived["preparation_origin"] == request["preparation_origin"]
            and request["preparation_origin"]["spec_hash"] == digest(original)
            and run["spec_hash"] == reference["computation_spec_hash"]
            and derived["code_hash"] == request["runtime_code_hash"] == code_hash(),
            "training population original arm/code binding differs", "CORRUPT_ARTIFACT")
        _verify_reads(store, request)
        _verify_job_cost(store, reference, run, proof)
        sources = [s for s in request["sources"] if s["split_role"] == "train"]
        require(_sources(store, original, arm, [s["selection"] for s in sources]) == sources,
            "training population source authority moved", "UNAUTHORIZED_DATA")
        population = proof["training_population"]
        require(type(population) is dict and set(population) == {"artifact_id", "provenance", "normalizer"},
                "closed training population receipt required")
        metadata = population["provenance"]
        normalizer = population["normalizer"]
        artifact = store._manifest("artifact-" + population["artifact_id"])
        validate_population_fragments(metadata, sources, request["settings"].get("fragment_policy", "reject"))
        require(metadata["schema_version"] == VERSION and metadata["train_binding_hash"] == digest(request["train_binding"])
            and metadata["source_study_id"] == original["study_id"]
            and metadata["block_id"] == "train-population-" + digest([original["study_id"], digest(request["train_binding"])])
            and metadata["population_kind"] == "computational-train-only-not-independent-sample"
            and metadata["purpose"] == "fit" and metadata["scientific_qualification"] == "not-established"
            and [(m["feature_sha256"], m["condition_sha256"], m["independent_block_id"], m["source_output_block_id"])
                 for m in metadata["members"]] == [(s["feature"]["sha256"], s["condition"]["sha256"],
                    s["independent_block_id"], s["selection"]["output_block_id"]) for s in sources]
            and metadata["independent_block_ids"] == sorted({s["independent_block_id"] for s in sources})
            and artifact["sha256"] == artifact["artifact_id"] == metadata["content_sha256"] == population["artifact_id"]
            and artifact["size_bytes"] == metadata["size_bytes"] <= MAX_BYTES
            and artifact["role"] == "training-population" and artifact["study_id"] == original["study_id"]
            and artifact["block_ids"] == metadata["independent_block_ids"],
            "training population artifact/member receipt differs", "CORRUPT_ARTIFACT")
        require(digest(normalizer) == metadata["normalizer_hash"]
            and normalizer["policy"] == request["settings"]["normalizer_policy"]
            and normalizer["train_binding_hash"] == metadata["train_binding_hash"]
            and normalizer["context_hash"] == metadata["context_hash"]
            and normalizer["independent_block_ids"] == metadata["independent_block_ids"],
            "actual fitted normalizer receipt differs", "CORRUPT_ARTIFACT")
        entry = {"block_id": metadata["block_id"], "dataset_id": "pirc26-derived-training-population",
            "release_id": VERSION, "source_block_id": metadata["block_id"], "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"], "path": artifact["artifact_id"], "split_role": "train", "fit_scope": True,
            "training_population_ref": deepcopy(reference), "training_population_hash": digest(metadata),
            "independent_block_ids": metadata["independent_block_ids"],
            "population_kind": metadata["population_kind"]}
        grants = _completion(store, original, arm, sources)
    _expiry(grants)
    return {"protocol_entry": entry, "data_root": str(store.path / "artifacts"),
            "provenance": deepcopy(metadata), "normalizer": deepcopy(normalizer)}
