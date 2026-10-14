"""Model-only attachment from a successful, settled population fit.

No qualification, authorization, fitting, runtime or budget is created here.
Metadata preflight never opens the mixed training result or any attachment.
"""

from copy import deepcopy
import json

from application.pirc26_population import require
from application.pirc26_population_runtime import execution_plugin, read_fitted, source, PREFIX
from application.pirc26_preparation import _expiry
from application.research_reuse import verified_success_metadata
from infrastructure.research_store import ResearchError, digest, encode
from infrastructure.research_visibility import combine_visibility, study_visibility

VERSION = "pirc26-fitted-model-v1"
MAX_BYTES = 2 * 1024 * 1024


def _producer(store, attempt_id, consumer_study_id=None):
    attempt = store._attempts().get(attempt_id)
    require(attempt and attempt["state"] == "SUCCEEDED", "successful population fit absent", "UNQUALIFIED")
    run = store._manifest("run-" + attempt["run_id"])
    spec = store._manifest("study-" + run["study_id"])["spec"]
    cell = run["cell"]
    plugin = execution_plugin(cell["execution"]["config"]["family"])
    require(cell["plugin_id"].startswith(PREFIX) and cell["plugin_id"] == plugin.plugin_id,
        "model producer is not the complete-population adapter", "UNQUALIFIED")
    proof = verified_success_metadata(store, attempt, spec, cell, plugin)
    population, evidence = source(proof["receipt"], consumer_study_id=consumer_study_id, store=store)
    artifact = store._manifest("artifact-" + population["protocol_entry"]["path"])
    original = store._manifest("study-" + population["provenance"]["source_study_id"])["spec"]
    visibility = combine_visibility([proof["result_artifact"]["visibility"], artifact["visibility"],
        study_visibility(store._manifest, spec), study_visibility(store._manifest, original)])
    return attempt, run, spec, proof, evidence, visibility


def _origin(attempt, run, spec, proof, evidence, visibility):
    settlement = proof["settlement"]
    return {"producer_attempt_id": attempt["attempt_id"], "producer_run_id": run["run_id"],
        "producer_study_id": spec["study_id"], "producer_spec_hash": digest(spec),
        "producer_admission_hash": proof["receipt"]["admission_hash"],
        "source_result_artifact_id": attempt["artifact_id"],
        "source_result_manifest_hash": digest(proof["result_artifact"]),
        "settlement_event_hash": settlement["hash"], "cost": deepcopy(settlement["payload"]),
        "training_population": deepcopy(evidence), "visibility": visibility,
        "block_ids": proof["result_artifact"]["block_ids"],
        "scientific_qualification": "not-established"}


def publish_fitted(store, attempt_id, *, authorization):
    """Owner serialization only, AFTER actual fit validation and settlement.

    The provided grant must permit the original result read. Publishing the
    separate attachment neither grants prospective consumers access nor marks
    it qualified. Retry is immutable/content-addressed, not a new fit job.
    """
    with store._read_transaction():
        attempt, run, spec, proof, evidence, visibility = _producer(store, attempt_id)
        actual = read_fitted(store, attempt_id, consumer_study_id=spec["study_id"], authorization=authorization)
        require(actual["training_population"] == evidence, "validated fitted membership differs", "CORRUPT_ARTIFACT")
        body = {"schema_version": VERSION, **actual}
        content = encode(body)
        require(len(content) <= MAX_BYTES, "complete model attachment exceeds quota", "RESOURCE_PLAN_REJECTED")
        metadata = store.artifact(content, role="fitted-model", visibility=visibility,
            block_ids=proof["result_artifact"]["block_ids"], study_id=spec["study_id"])
        receipt = {"schema_version": VERSION, **_origin(attempt, run, spec, proof, evidence, visibility),
            "artifact_id": metadata["artifact_id"], "artifact_manifest_hash": digest(metadata),
            "checkpoint_hash": actual["checkpoint"]["sha256"],
            "model_card": deepcopy(actual["checkpoint"]["model_card"])}
        receipt_hash = digest(receipt)
        manifest_id = "pirc26-fitted-model-" + receipt_hash
        store.publish(manifest_id, receipt)
        store._read_completion(
            lambda: store.verify_artifact_read(attempt["artifact_id"], purpose="evaluate", authorization=authorization),
            lambda: _expiry([authorization]))
    _expiry([authorization])
    return {"manifest_id": manifest_id, "receipt_hash": receipt_hash,
        "artifact_id": metadata["artifact_id"], "producer_attempt_id": attempt_id}


def model_source(store, reference, *, consumer_study_id, authorization):
    """Fresh metadata and original source/disclosure authority; ZERO payload I/O."""
    try:
        require(type(reference) is dict and set(reference) == {
            "manifest_id", "receipt_hash", "artifact_id", "producer_attempt_id"}, "closed model reference required")
        with store._read_transaction():
            attempt, run, spec, proof, evidence, visibility = _producer(
                store, reference["producer_attempt_id"], consumer_study_id)
            receipt = store._manifest(reference["manifest_id"])
            require(receipt["schema_version"] == VERSION and digest(receipt) == reference["receipt_hash"]
                and reference["manifest_id"] == "pirc26-fitted-model-" + digest(receipt)
                and all(receipt[key] == value for key, value in _origin(attempt, run, spec, proof, evidence, visibility).items()),
                "model attachment success/admission/cost/population lineage differs", "CORRUPT_ARTIFACT")
            metadata = store._manifest("artifact-" + reference["artifact_id"])
            require(receipt["artifact_id"] == metadata["artifact_id"] == metadata["sha256"] == reference["artifact_id"]
                and receipt["artifact_manifest_hash"] == digest(metadata)
                and metadata["role"] == "fitted-model" and metadata["media_type"] == "application/json"
                and metadata["study_id"] == spec["study_id"] and metadata["block_ids"] == receipt["block_ids"]
                and metadata["visibility"] == visibility and 0 < metadata["size_bytes"] <= MAX_BYTES,
                "model-only artifact ownership or visibility differs", "CORRUPT_ARTIFACT")
            def authority():
                require(authorization["study_id"] == spec["study_id"]
                    and authorization["protocol_hash"] == spec["protocol_hash"]
                    and (consumer_study_id == spec["study_id"]
                         or consumer_study_id in authorization.get("consumer_study_ids", [])),
                    "model-only grant does not permit this consumer", "UNAUTHORIZED_DATA")
                store.verify_artifact_read(reference["artifact_id"], purpose="evaluate", authorization=authorization)
            authority()
            store._read_completion(authority, lambda: _expiry([authorization]))
        _expiry([authorization])
        return deepcopy(receipt)
    except ResearchError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchError("CORRUPT_ARTIFACT", "model attachment provenance is malformed") from exc


def read_model(store, reference, *, consumer_study_id, authorization):
    """Open ONLY the published model attachment through original read journal."""
    with store._read_transaction():
        proof = model_source(store, reference, consumer_study_id=consumer_study_id, authorization=authorization)
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
                == digest({k:v for k,v in body["checkpoint"].items() if k != "sha256"}),
                "model-only attachment body differs", "CORRUPT_ARTIFACT")
        except ResearchError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchError("CORRUPT_ARTIFACT", "model-only attachment body is malformed") from exc
        require(model_source(store, reference, consumer_study_id=consumer_study_id, authorization=authorization) == proof,
            "model-only attachment authority moved", "UNAUTHORIZED_DATA")
    _expiry([authorization])
    return body
