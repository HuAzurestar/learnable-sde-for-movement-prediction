"""Explicit read bridge for a frozen historic model under renewed permissions.

The original strict reader is unchanged. This separate API verifies historic
admission/cost/read receipts without claiming current execution qualification.
Only the model-only attachment is opened; no mixed result or raw coordinates.
"""
from copy import deepcopy
import json

from application.pirc26_fitted_model import VERSION, MAX_BYTES, _origin
from application.pirc26_population import population_source, require
from application.pirc26_preparation import _sources, _completion, _expiry
from application.research_continuation import approval, renewed_source_grant
from application.research_reuse import verified_historical_success_metadata
from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, digest, encode
from infrastructure.research_visibility import combine_visibility, study_visibility


def continuation_context(store, reference):
    require(type(reference) is dict and set(reference) == {"manifest_id", "sha256"},
            "exact model continuation reference required")
    value = store._manifest(reference["manifest_id"])
    require(digest(value) == reference["sha256"]
            and value.get("schema_version") == "pirc26-model-continuation-v1",
            "model continuation receipt differs")
    decision = approval(store, value["approval"])
    require(value["historical_code_hash"] == decision.get("historical_code_hash")
            and value["current_code_hash"] == code_hash()
            and type(value["renewals"]) is dict and value["renewals"],
            "registered model code epoch/renewals differ", "UNAUTHORIZED_DATA")
    # Every linked renewal must stem from the SAME approved decision.
    for renewal in value["renewals"].values():
        receipt = store._manifest(renewal["manifest_id"])
        require(digest(receipt) == renewal["sha256"] and receipt["approval"] == value["approval"],
                "model continuation has unapproved renewal", "UNAUTHORIZED_DATA")
    return deepcopy(value)


def continued_model_source(store, reference, *, continuation, consumer_study_id, authorization):
    """Metadata-only, exact old proof plus explicit current upstream authority."""
    try:
        with store._read_transaction():
            bridge = continuation_context(store, continuation)
            require(reference == bridge["model_reference"], "continued model reference differs")
            receipt = store._manifest(reference["manifest_id"])
            require(reference["manifest_id"] == "pirc26-fitted-model-" + digest(receipt)
                    and digest(receipt) == reference["receipt_hash"]
                    and receipt["schema_version"] == VERSION
                    and receipt["producer_attempt_id"] == reference["producer_attempt_id"],
                    "historic model receipt differs", "CORRUPT_ARTIFACT")
            attempt = store._attempts().get(reference["producer_attempt_id"])
            require(attempt and attempt["state"] == "SUCCEEDED", "historic fit not successful", "UNQUALIFIED")
            run = store._manifest("run-" + attempt["run_id"])
            spec = store._manifest("study-" + run["study_id"])["spec"]
            cell = run["cell"]
            decision = approval(store, bridge["approval"])
            require(spec["code_hash"] == bridge["historical_code_hash"]
                    and cell["arm_id"] in decision["arm_ids"]
                    and cell["plugin_id"].startswith("pirc26-population-")
                    and cell["execution"]["config"]["objective"] == "O1",
                    "historic model family/code/arm differs", "UNQUALIFIED")
            proof = verified_historical_success_metadata(store, attempt, spec, cell,
                admission_hash=receipt["producer_admission_hash"])
            admitted = proof["receipt"]
            binding = admitted["documents"]["package"]["payload"]["pirc26_population"]
            population = population_source(store, binding["preparation_ref"], continuation=continuation)
            metadata, normalizer = population["provenance"], population["normalizer"]
            require(binding["schema_version"] == "pirc26-owned-population-binding-v1"
                    and binding["population_hash"] == digest(metadata)
                    and [b for b in admitted["documents"]["protocol"]["blocks"]
                         if b["block_id"] == cell["block_id"]] == [population["protocol_entry"]],
                    "historic fit population differs", "CORRUPT_ARTIFACT")
            job = admitted["documents"]["package"]["payload"]["pirc26_job"]
            batches = sum((s["transitions"] + job["batch_size"] - 1) // job["batch_size"]
                          for s in metadata["segments"])
            expected = {"schema_version": binding["schema_version"],
                        "preparation_ref": binding["preparation_ref"], "population_hash": binding["population_hash"],
                        "observations": metadata["observations"], "transitions": metadata["transitions"],
                        "batch_count": batches, "independent_block_ids": metadata["independent_block_ids"],
                        "use": "training-diagnostics-not-held-out"}
            require(receipt["training_population"] == expected, "historic complete population binding differs")
            model_spec = receipt["model_card"]["spec"]
            require(all(model_spec[k] == metadata[k] for k in
                        ("coordinate_frame", "train_binding_hash", "normalizer_hash", "context_hash"))
                    and model_spec["means"] == normalizer["means"] and model_spec["scales"] == normalizer["scales"],
                    "historic model transform differs")
            request = store._manifest(binding["preparation_ref"]["request_manifest_id"])
            original = store._manifest("study-" + request["original_study_id"])["spec"]
            arm = next(a for a in spec["arms"] if a["arm_id"] == cell["arm_id"])
            require(arm in original["arms"], "historic fit relabeled budget arm")
            sources = [s for s in request["sources"] if s["split_role"] == "train"]
            current_grants = []
            for source in sources:
                selection = source["selection"]
                old = store.authorization(selection["authorization_id"], version=selection["authorization_version"])
                current = renewed_source_grant(store, old, bridge["renewals"])
                require(consumer_study_id == original["study_id"]
                        or consumer_study_id in current.get("consumer_study_ids", []),
                        "renewed raw source does not permit consumer", "UNAUTHORIZED_DATA")
                current_grants.append(current)
            old_execution = store.authorization(spec["admission"]["authorization_id"],
                                                version=spec["admission"].get("authorization_version"))
            execution = renewed_source_grant(store, old_execution, bridge["renewals"])
            require(execution["study_id"] == spec["study_id"] and execution["protocol_hash"] == spec["protocol_hash"]
                    and cell["block_id"] in execution["block_ids"] and {"execute", "fit"} <= set(execution["purposes"])
                    and cell["visibility"] in execution["visibilities"]
                    and execution.get("data_root") == population["data_root"], "renewed execution scope differs")
            current_grants.append(execution)
            source_artifact = store._manifest("artifact-" + population["protocol_entry"]["path"])
            visibility = combine_visibility([proof["result_artifact"]["visibility"], source_artifact["visibility"],
                study_visibility(store._manifest, spec), study_visibility(store._manifest, original)])
            require(all(receipt.get(k) == v for k, v in
                        _origin(attempt, run, spec, proof, expected, visibility).items()),
                    "historic model success/cost/origin differs", "CORRUPT_ARTIFACT")
            artifact = store._manifest("artifact-" + reference["artifact_id"])
            require(receipt["artifact_id"] == artifact["artifact_id"] == artifact["sha256"] == reference["artifact_id"]
                    and receipt["artifact_manifest_hash"] == digest(artifact) and artifact["role"] == "fitted-model"
                    and artifact["media_type"] == "application/json" and artifact["study_id"] == spec["study_id"]
                    and artifact["block_ids"] == receipt["block_ids"] and artifact["visibility"] == visibility
                    and 0 < artifact["size_bytes"] <= MAX_BYTES, "historic model attachment differs", "CORRUPT_ARTIFACT")
            # Current model grant must be an exact registered scope-preserving renewal.
            renewed_models = []
            for renewal in bridge["renewals"].values():
                renewal_receipt = store._manifest(renewal["manifest_id"])
                if renewal_receipt["new_hash"] == digest(authorization):
                    old = store.authorization(renewal_receipt["authorization_id"], version=renewal_receipt["old_version"])
                    renewed_models.append(renewed_source_grant(store, old, bridge["renewals"]))
            require(renewed_models == [authorization]
                    and authorization["study_id"] == spec["study_id"] and authorization["protocol_hash"] == spec["protocol_hash"]
                    and (consumer_study_id == spec["study_id"] or consumer_study_id in authorization.get("consumer_study_ids", [])),
                    "model renewal/consumer differs", "UNAUTHORIZED_DATA")
            def authority():
                require(continuation_context(store, continuation) == bridge, "model continuation moved")
                require(_sources(store, original, arm, [s["selection"] for s in sources],
                                 renewals=bridge["renewals"]) == sources, "renewed raw authority differs")
                store.verify_artifact_read(reference["artifact_id"], purpose="evaluate", authorization=authorization)
            authority()
            _completion(store, original, arm, sources, renewals=bridge["renewals"])
            store._read_completion(authority, lambda: _expiry([*current_grants, authorization]))
        _expiry([*current_grants, authorization])
        return deepcopy(receipt)
    except ResearchError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ResearchError("CORRUPT_ARTIFACT", "historic model continuation is malformed") from exc


def read_continued_model(store, reference, *, continuation, consumer_study_id, authorization):
    """Read only model attachment, with fresh checks before and after the I/O."""
    with store._read_transaction():
        proof = continued_model_source(store, reference, continuation=continuation,
                                      consumer_study_id=consumer_study_id, authorization=authorization)
        content = store.read_artifact(reference["artifact_id"], purpose="evaluate", authorization=authorization)
        try:
            body = json.loads(content)
            require(type(body) is dict and set(body) == {"schema_version", "checkpoint", "training_population",
                        "producer_attempt_id", "scientific_qualification"} and content == encode(body)
                    and body["schema_version"] == VERSION and body["scientific_qualification"] == "not-established"
                    and body["producer_attempt_id"] == proof["producer_attempt_id"]
                    and body["training_population"] == proof["training_population"]
                    and body["checkpoint"]["model_card"] == proof["model_card"]
                    and body["checkpoint"]["sha256"] == proof["checkpoint_hash"]
                    == digest({k: v for k, v in body["checkpoint"].items() if k != "sha256"}),
                    "continued model-only body differs", "CORRUPT_ARTIFACT")
        except ResearchError:
            raise
        except (KeyError, ValueError, TypeError) as exc:
            raise ResearchError("CORRUPT_ARTIFACT", "continued model-only body malformed") from exc
        require(continued_model_source(store, reference, continuation=continuation,
                    consumer_study_id=consumer_study_id, authorization=authorization) == proof,
                "continued model authority moved", "UNAUTHORIZED_DATA")
    _expiry([authorization])
    return body
