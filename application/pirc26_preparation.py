"""Role-safe paired-source preprocessing on the existing shared supervisor.

The owner admits bounded bytes, without numerical decoding. A genuinely funded
worker converts them and fits one normalizer on the complete frozen train
population. No data grants, model qualification, blind-test access, new budget
arms, global model fitting or scientific acceptance are conferred here.
"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from application.pirc26_dsde import (SourcePair, ProjectionSpec, ROLES, MAX_BYTES,
    _pair_contract, context_binding, read_source_pair)
from application.research_budget import BudgetSpec
from application.research_computation import _verify_job_cost
from application.research_data import EvaluationExposureLedger
from application.research_supervisor import ResearchSupervisor
from experiments.pirc25.affine import ROOT, code_hash
from infrastructure.research_store import ResearchError, atomic_write, digest, encode
from application.pirc26_fragment_contract import SHORT_POLICY, POLICIES as FRAGMENT_POLICIES, fragment_summary


VERSION = "pirc26-managed-dsde-preparation-v2"
MAX_PAIRS = 16
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
MAX_OBSERVATIONS = 16_912
NORMALIZER_POLICY = {
    "schema_version": "pirc26-normalizer-policy-v1",
    "fit_role": "train", "purpose": "fit", "dtype": "float64",
    "population": "all-causal-states-in-frozen-train-membership",
    "weighting": "per-observation", "variance": "population-ddof0",
    "zero_variance_scale": 1.0, "velocity": "backward-difference-v1",
}


def require(ok, message, code="CONTRACT_MISMATCH"):
    if not ok:
        raise ResearchError(code, message)


def preparation_settings(feature_spec, benchmark_binding, projection, *, duplicate_policy="reject", fragment_policy="reject"):
    """Metadata-only configuration; no source inspection or automatic frame fit."""
    require(isinstance(projection, ProjectionSpec), "explicit projection required")
    require(duplicate_policy in {"reject", "keep-first-exact-time-v1"}, "explicit duplicate policy required")
    require(type(fragment_policy) is str and fragment_policy in FRAGMENT_POLICIES, "explicit causal fragment policy required")
    settings = {"schema_version": VERSION, "feature_spec": deepcopy(feature_spec),
            "benchmark_binding": deepcopy(benchmark_binding), "projection": asdict(projection),
            "context_binding": context_binding(feature_spec, benchmark_binding),
            "duplicate_policy": duplicate_policy, "normalizer_policy": deepcopy(NORMALIZER_POLICY)}
    if fragment_policy == SHORT_POLICY:
        settings["fragment_policy"] = fragment_policy
        settings["normalizer_policy"]["fragment_policy"] = fragment_policy
    return settings


def validate_settings(settings):
    require(type(settings) is dict and set(settings) - {"fragment_policy"} == {
        "schema_version", "feature_spec", "benchmark_binding", "projection", "context_binding",
        "duplicate_policy", "normalizer_policy"}, "closed preparation settings required")
    require(settings == preparation_settings(settings["feature_spec"], settings["benchmark_binding"],
        ProjectionSpec(**settings["projection"]), duplicate_policy=settings["duplicate_policy"],
        fragment_policy=settings.get("fragment_policy", "reject")),
        "preparation settings/context/normalizer policy differs")
    require(len(encode(settings)) <= 1024 * 1024, "preparation settings byte quota", "RESOURCE_PLAN_REJECTED")


def _sources(store, original, arm, selections, *, require_train=True):
    """Fresh live authority for all pairs before opening any scientific bytes."""
    require(type(selections) is list and 1 <= len(selections) <= MAX_PAIRS,
            "bounded explicit source population required", "RESOURCE_PLAN_REJECTED")
    ledger, selected, seen, roles, total = EvaluationExposureLedger(store), [], set(), {}, 0
    for selection in selections:
        require(type(selection) is dict and set(selection) == {"protocol_id", "feature_block_id",
            "condition_block_id", "authorization_id", "authorization_version", "output_block_id", "purpose"},
            "closed explicit source selection required")
        protocol = store.manifest("protocol-" + selection["protocol_id"])
        require(protocol["study_id"] == original["study_id"] and digest(protocol) == original["protocol_hash"],
                "raw source protocol is not the original registered study", "UNAUTHORIZED_DATA")
        blocks = []
        for key in ("feature_block_id", "condition_block_id"):
            matches = [b for b in protocol["blocks"] if b["block_id"] == selection[key]]
            require(len(matches) == 1, "explicit raw source absent", "UNAUTHORIZED_DATA")
            blocks.append(matches[0])
        feature, condition = blocks
        role = _pair_contract(feature, condition, selection["purpose"])
        unit = feature["independent_block_id"]
        cells = [c for c in original["cells"] if c["arm_id"] == arm["arm_id"] and c["block_id"] == unit]
        require(bool(cells), "source independent unit absent from original arm", "UNAUTHORIZED_DATA")
        require(roles.setdefault(unit, role) == role, "independent unit occurs across source roles", "UNAUTHORIZED_DATA")
        grant = store.authorization(selection["authorization_id"], version=selection["authorization_version"])
        require(type(grant.get("data_root")) is str and Path(grant["data_root"]).is_absolute(),
                "explicit authorized preparation root required", "UNAUTHORIZED_DATA")
        root = Path(grant["data_root"])
        for block in blocks:
            relative = Path(block["path"])
            require(not relative.is_absolute() and (root / relative).resolve().is_relative_to(root.resolve()),
                    "paired source escapes authorized root", "UNAUTHORIZED_DATA")
            allowed, _, _ = ledger._read_authority(protocol, block, selection["purpose"],
                selection["authorization_id"], expected=grant, authorization_version=selection["authorization_version"])
            require(allowed and all(c.get("visibility", "restricted") in grant["visibilities"] for c in cells),
                    "paired-source authority/visibility differs or expired", "UNAUTHORIZED_DATA")
        output_id = selection["output_block_id"]
        require(type(output_id) is str and 0 < len(output_id) <= 128 and output_id not in seen,
                "bounded unique computational output blocks required")
        seen.add(output_id)
        identity = (digest(protocol), feature["file_id"], feature["sha256"], condition["sha256"])
        require(not any(tuple(d["source_identity"]) == identity for d in selected), "duplicate source file population")
        total += feature["size_bytes"] + condition["size_bytes"]
        require(total <= MAX_SOURCE_BYTES, "joint paired-source byte quota", "RESOURCE_PLAN_REJECTED")
        selected.append({"selection": deepcopy(selection), "feature": deepcopy(feature),
            "condition": deepcopy(condition), "protocol_hash": digest(protocol),
            "authorization_hash": digest(grant), "source_identity": list(identity),
            "split_role": role, "independent_block_id": unit})
    require(not require_train or any(d["split_role"] == "train" for d in selected),
            "train population required to fit normalizer", "UNAUTHORIZED_DATA")
    return selected


def train_binding(descriptors, settings):
    """Freeze actual file/unit membership and preprocessing before reads."""
    binding = {"schema_version": "pirc26-train-population-v1",
            # Fit identity binds only the actual train population. The full
            # protocol (including selection metadata) remains in request/source
            # receipts, but is not a training-only numerical fit dependency.
            "members": [{k: deepcopy(d[k]) for k in ("feature", "condition", "independent_block_id")}
                        for d in descriptors if d["split_role"] == "train"],
            "projection": settings["projection"], "context_binding": settings["context_binding"],
            "duplicate_policy": settings["duplicate_policy"], "normalizer_policy": settings["normalizer_policy"]}
    if settings.get("fragment_policy") == SHORT_POLICY:
        binding["fragment_policy"] = SHORT_POLICY
    return binding


def _validate_geometry(document, width):
    require(type(document) is dict and set(document) == {"schema_version", "block_id", "coordinate_frame", "state_units",
        "time_unit", "velocity_source", "train_binding_hash", "normalizer_hash", "context_hash", "segments"}
        and document["state_units"] == ["m", "m", "m/s", "m/s"] and document["time_unit"] == "s"
        and document["velocity_source"] == "backward-difference-v1" and type(document["segments"]) is list
        and 1 <= len(document["segments"]) <= 256, "closed physical geometry required")
    seen = set()
    for segment in document["segments"]:
        require(type(segment) is dict and set(segment) == {"segment_id", "time", "position", "condition", "condition_available_at"},
                "closed prepared segment required")
        require(type(segment["segment_id"]) is str and 0 < len(segment["segment_id"]) <= 128
                and segment["segment_id"] not in seen, "bounded unique prepared segments required")
        seen.add(segment["segment_id"])
        t = segment["time"]
        require(type(t) is list and 3 <= len(t) <= 4098
            and all(type(x) in (int, float) and math.isfinite(x) for x in t)
            and all(a < b for a, b in zip(t, t[1:])), "prepared chronology differs")
        for field, columns in (("position", 2), ("condition", width - 4)):
            require(type(segment[field]) is list and len(segment[field]) == len(t)
                and all(type(row) is list and len(row) == columns
                    and all(type(x) in (int, float) and math.isfinite(x) for x in row) for row in segment[field]),
                "prepared finite matrix dimensions differ")
        available = segment["condition_available_at"]
        require(type(available) is list and len(available) == len(t)
                and all(type(a) in (int, float) and math.isfinite(a) and a <= time for a, time in zip(available, t)),
                "prepared context contains future information")


def _validate_result(value, request):
    """Owner structural/integrity checks; do not rerun numerical preparation."""
    validate_settings(request["settings"])
    require(request["train_binding"] == train_binding(request["sources"], request["settings"]),
            "preparation eligibility/train binding differs", "CORRUPT_ARTIFACT")
    require(type(value) is dict and set(value) == {"schema_version", "request_hash", "computation_ref",
        "train_binding", "normalizer", "blocks", "training_population", "scientific_qualification"}, "closed preparation result required")
    require(value["schema_version"] == VERSION and value["request_hash"] == digest(request)
        and value["computation_ref"] == request["computation_ref"]
        and value["train_binding"] == request["train_binding"]
        and value["scientific_qualification"] == "not-established", "preparation result/source binding differs")
    normalizer = value["normalizer"]
    require(type(normalizer) is dict and set(normalizer) == {"schema_version", "train_binding_hash",
        "policy", "context_hash", "means", "scales", "observations", "independent_block_ids"}, "closed normalizer required")
    width = 4 + request["settings"]["context_binding"]["context_dim"]
    require(normalizer["schema_version"] == "pirc26-fitted-normalizer-v1"
        and normalizer["train_binding_hash"] == digest(request["train_binding"])
        and normalizer["policy"] == request["settings"]["normalizer_policy"]
        and normalizer["context_hash"] == digest(request["settings"]["context_binding"])
        and type(normalizer["observations"]) is int and 0 < normalizer["observations"] <= MAX_OBSERVATIONS
        and normalizer["independent_block_ids"] == sorted({d["independent_block_id"] for d in request["sources"] if d["split_role"] == "train"}),
        "normalizer population/context/policy differs")
    for key in ("means", "scales"):
        require(type(normalizer[key]) is list and len(normalizer[key]) == width
            and all(type(x) in (int, float) and math.isfinite(x) and (key != "scales" or x > 0) for x in normalizer[key]),
            "normalizer finite dimensions/scales differ")
    require(type(value["blocks"]) is list and len(value["blocks"]) == len(request["sources"]), "prepared population differs")
    observations, train_observations = 0, 0
    for block, source in zip(value["blocks"], request["sources"]):
        require(type(block) is dict and set(block) == {"document", "provenance"}, "closed prepared block required")
        document, provenance = block["document"], block["provenance"]
        _validate_geometry(document, width)
        fragment_summary(block, expected_policy=request["settings"].get("fragment_policy", "reject"))
        require(document["schema_version"] == "pirc26-observed-block-v1"
            and document["block_id"] == source["selection"]["output_block_id"]
            and document["coordinate_frame"] == request["settings"]["projection"]["coordinate_frame"]
            and document["train_binding_hash"] == digest(request["train_binding"])
            and document["normalizer_hash"] == digest(normalizer)
            and document["context_hash"] == normalizer["context_hash"]
            and provenance["feature"] == source["feature"] and provenance["condition"] == source["condition"]
            and provenance["source_protocol_hash"] == source["protocol_hash"]
            and provenance["authorization_hash"] == source["authorization_hash"]
            and provenance["train_binding_hash"] == document["train_binding_hash"]
            and provenance["normalizer_hash"] == document["normalizer_hash"]
            and provenance["split_role"] == source["split_role"]
            and provenance["purpose"] == source["selection"]["purpose"]
            and provenance["scientific_qualification"] == "not-established"
            and provenance["projection"] == request["settings"]["projection"]
            and provenance["context_binding"] == request["settings"]["context_binding"]
            and provenance["content_sha256"] == hashlib.sha256(encode(document)).hexdigest()
            and provenance["size_bytes"] == len(encode(document)) <= MAX_BYTES,
            "prepared geometry/provenance differs")
        require(len(document["segments"]) == len(provenance["membership"]) and all(
            m["independent_block_id"] == source["independent_block_id"] and m["segment_id"] == s["segment_id"]
            and len(m["source_point_indices"]) == len(m["point_ids"]) == len(s["time"])
            for m, s in zip(provenance["membership"], document["segments"])), "prepared independent-unit membership differs")
        observations += sum(len(s["time"]) for s in document["segments"])
        if source["split_role"] == "train":
            train_observations += sum(len(s["time"]) - 1 for s in document["segments"])
    require(observations <= MAX_OBSERVATIONS and train_observations == normalizer["observations"],
            "joint prepared/training observation quota differs", "RESOURCE_PLAN_REJECTED")
    from application.pirc26_population import validate_population
    validate_population(value["training_population"], value["blocks"], value["train_binding"], normalizer,
                        study_id=request["original_study_id"])
    require(len(encode(value)) <= MAX_RESULT_BYTES, "preparation result byte quota", "RESOURCE_PLAN_REJECTED")


def _verify_reads(store, request):
    events = {e["hash"]: e for e in store._events()}
    reference = request["computation_ref"]
    reserved = [e for e in events.values() if e["event_kind"] == "RESERVE"
        and e["payload"].get("reservation_id") == reference["reservation_id"]
        and e["payload"].get("attempt_id") == reference["attempt_id"]]
    workers = [e for e in events.values() if e["event_kind"] == "WORKER_STARTED"
        and e["payload"].get("reservation_id") == reference["reservation_id"]
        and e["payload"].get("attempt_id") == reference["attempt_id"]]
    require(len(reserved) == len(workers) == 1, "paired read funded worker boundary missing", "CORRUPT_ARTIFACT")
    require(len(request["read_event_hashes"]) == 2 * len(request["sources"]), "paired read receipts missing")
    require(len(set(request["read_event_hashes"])) == len(request["read_event_hashes"]), "duplicate paired read receipt", "CORRUPT_ARTIFACT")
    for i, source in enumerate(request["sources"]):
        for j, block in enumerate((source["feature"], source["condition"])):
            event = events.get(request["read_event_hashes"][2 * i + j], {})
            p = event.get("payload", {})
            require(event.get("event_kind") == "READ_COMPLETED"
                and reserved[0]["sequence"] < event.get("sequence", 0) < workers[0]["sequence"]
                and p.get("protocol_hash") == source["protocol_hash"] and p.get("block_id") == block["block_id"]
                and p.get("authorization_hash") == source["authorization_hash"]
                and p.get("purpose") == source["selection"]["purpose"] and p.get("sha256") == block["sha256"],
                "actual paired source read receipt differs", "CORRUPT_ARTIFACT")


def _expiry(grants):
    require(all(datetime.fromisoformat(g["expires_at"]) > datetime.now(timezone.utc) for g in grants),
            "preparation source permission expired at disclosure", "UNAUTHORIZED_DATA")


def _completion(store, original, arm, sources, *, require_train=True):
    """Preserve shared final physical-scope authority and post-I/O expiry."""
    selections = [s["selection"] for s in sources]
    grants = [store.authorization(s["authorization_id"], version=s["authorization_version"]) for s in selections]
    store._read_completion(
        lambda: require(_sources(store, original, arm, selections, require_train=require_train) == sources,
                        "preparation source authority moved at disclosure", "UNAUTHORIZED_DATA"),
        lambda: _expiry(grants))
    return grants


def load_prepared(store, reference):
    """Verify actual success, original source authority, receipts and charges."""
    with store._read_transaction():
        proof = store._manifest(reference["manifest_id"])
        request = store._manifest(reference["request_manifest_id"])
        attempt = store._attempts().get(reference["attempt_id"])
        require(attempt and attempt["state"] == "SUCCEEDED" and attempt["run_id"] == reference["run_id"]
            and proof["schema_version"] == "pirc26-preparation-receipt-v2"
            and proof["computation_ref"] == request["computation_ref"] == reference
            and proof["request_hash"] == digest(request) and proof["result_artifact_id"] == attempt["artifact_id"],
            "preparation success/receipt changed", "CORRUPT_ARTIFACT")
        run = store._manifest("run-" + attempt["run_id"])
        derived = store._manifest("study-" + run["study_id"])["spec"]
        original = store._manifest("study-" + request["original_study_id"])["spec"]
        arm = next(a for a in original["arms"] if a["arm_id"] == run["arm_id"])
        require(derived["arms"] == [arm] and derived["preparation_origin"] == request["preparation_origin"]
            and request["preparation_origin"]["spec_hash"] == digest(original)
            and run["spec_hash"] == reference["computation_spec_hash"]
            and derived["code_hash"] == request["runtime_code_hash"] == code_hash(),
            "original preparation arm/source/code changed", "CORRUPT_ARTIFACT")
        _verify_reads(store, request)
        _verify_job_cost(store, reference, run, proof)
        require(_sources(store, original, arm, [s["selection"] for s in request["sources"]]) == request["sources"],
                "preparation source authority moved", "UNAUTHORIZED_DATA")
        metadata = store._manifest("artifact-" + attempt["artifact_id"])
        require(metadata["study_id"] == run["study_id"] and metadata["role"] == "result"
                and digest(metadata) == attempt["artifact_manifest_hash"], "prepared result metadata changed", "CORRUPT_ARTIFACT")
        value = json.loads(store._verified_artifact_content(metadata))
        _validate_result(value, request)
        grants = _completion(store, original, arm, request["sources"])
    _expiry(grants)
    return value


class PreparationRunner:
    def __init__(self, store):
        self.store = store

    def run(self, study_id, cell_hash, selections, settings, *, budget=BudgetSpec(), parent_attempt_id=None, reason=None):
        budget.validate()
        validate_settings(settings)
        with self.store._read_transaction():
            original = self.store._manifest("study-" + study_id)["spec"]
            cells = [c for c in original["cells"] if digest(c) == cell_hash]
            require(len(cells) == 1, "original preparation reference cell absent")
            cell = cells[0]
            arm = next(a for a in original["arms"] if a["arm_id"] == cell["arm_id"])
            descriptors = _sources(self.store, original, arm, selections)
            require(all(d["feature"]["feature_spec_sha256"] == settings["context_binding"]["feature_spec_sha256"]
                        for d in descriptors), "source feature context differs before read")
            if settings.get("fragment_policy") == SHORT_POLICY:
                require(all(type(d["feature"].get("aligned_row_count")) is int
                    and 3 <= d["feature"]["aligned_row_count"] <= MAX_OBSERVATIONS for d in descriptors),
                    "frozen aligned row count required before fragment preparation")
        runtime_hash = code_hash()
        population = train_binding(descriptors, settings)
        origin = {"study_id": study_id, "spec_hash": digest(original), "cell_hash": cell_hash,
                  "settings_hash": digest(settings), "source_population_hash": digest(descriptors),
                  "runtime_code_hash": runtime_hash, "allocation": "shared-job-on-original-arm"}
        identity = digest(origin)
        derived = {**{k: original[k] for k in ("schema_version", "comparison_family", "protocol_hash", "feature_hash", "selection_hash")},
            "study_id": "preparejob-" + identity, "experiment_id": VERSION, "data_hash": digest(descriptors),
            "code_hash": runtime_hash, "arms": [arm], "cells": [{"arm_id": arm["arm_id"], "block_id": cell["block_id"],
                "seed": cell["seed"], "plugin_id": VERSION, "resource_class": "cpu", "visibility":
                    "synthetic" if all(c.get("visibility") == "synthetic" for c in original["cells"]
                        if c["arm_id"] == arm["arm_id"] and c["block_id"] in {d["independent_block_id"] for d in descriptors}) else "restricted"}],
            "preparation_origin": origin}
        self.store.register(derived, digest(derived))
        run_id = self.store.register_run(derived["study_id"], derived["cells"][0])
        prior = [a for a in self.store.attempts().values() if a["run_id"] == run_id]
        success = next((a for a in prior if a["state"] == "SUCCEEDED"), None)
        if success:
            request = self.store.manifest("prepare-request-" + success["attempt_id"])
            reference = request["computation_ref"]
            value = load_prepared(self.store, reference)
            return {"state": "SUCCEEDED", "attempt_id": success["attempt_id"], "artifact_id": success["artifact_id"],
                    "exit_code": 0, "reused": True, "preparation_ref": reference, "prepared": value}
        attempt_id = self.store.new_attempt(run_id, parent_attempt_id=parent_attempt_id, reason=reason)
        reference = {"manifest_id": "prepare-receipt-" + attempt_id, "request_manifest_id": "prepare-request-" + attempt_id,
            "attempt_id": attempt_id, "run_id": run_id, "computation_spec_hash": digest(derived),
            "reservation_id": digest([self.store.store_id, attempt_id]), "allocation": "shared-job-on-original-arm"}
        request = {"schema_version": VERSION, "computation_ref": reference, "original_study_id": study_id,
            "preparation_origin": origin, "runtime_code_hash": runtime_hash, "settings": deepcopy(settings),
            "sources": descriptors, "train_binding": population, "read_event_hashes": []}

        def command(output):
            # Existing supervisor has already reserved and claimed the original
            # arm. This builder only reads/serializes bytes, never decodes them.
            with self.store._read_transaction():
                require(_sources(self.store, original, arm, selections) == descriptors, "source authority moved before handoff", "UNAUTHORIZED_DATA")
            for i, source in enumerate(descriptors):
                selection = source["selection"]
                before = self.store.events()[-1]["sequence"]
                pair = read_source_pair(self.store, selection["protocol_id"], selection["feature_block_id"], selection["condition_block_id"],
                    authorization_id=selection["authorization_id"], authorization_version=selection["authorization_version"], purpose=selection["purpose"])
                require(pair.protocol_hash == source["protocol_hash"] and pair.authorization_hash == source["authorization_hash"], "handoff authority moved", "UNAUTHORIZED_DATA")
                completed = [e["hash"] for e in self.store.events() if e["sequence"] > before and e["event_kind"] == "READ_COMPLETED"
                    and e["payload"].get("protocol_hash") == pair.protocol_hash
                    and e["payload"].get("block_id") in {source["feature"]["block_id"], source["condition"]["block_id"]}]
                require(len(completed) == 2, "explicit pair read receipts ambiguous")
                request["read_event_hashes"].extend(completed)
                atomic_write(output.parent / (str(i) + "-features.parquet"), pair.feature_bytes)
                atomic_write(output.parent / (str(i) + "-conditions.parquet"), pair.condition_bytes)
            require(len(encode(request)) <= 2 * 1024 * 1024, "frozen worker request byte quota", "RESOURCE_PLAN_REJECTED")
            self.store.publish(reference["request_manifest_id"], request)
            atomic_write(output.parent / "preparation-request.json", encode(request))
            return [sys.executable, "-B", str(ROOT / "infrastructure/pirc26_preparation_worker.py"), str(output)]

        def validate(value):
            _validate_result(value, request)
            require(code_hash() == runtime_hash, "preparation implementation moved during execution")
            with self.store._read_transaction():
                _verify_reads(self.store, request)
                require(_sources(self.store, original, arm, selections) == descriptors, "source authority expired during preparation", "UNAUTHORIZED_DATA")
                _completion(self.store, original, arm, descriptors)

        outcome = ResearchSupervisor(self.store).run(attempt_id, command, budget, result_validator=validate,
            resource_plan={"maximum_result_bytes": MAX_RESULT_BYTES})
        outcome["reused"] = False
        if outcome["state"] != "SUCCEEDED":
            return outcome
        with self.store._read_transaction():
            settles = [e for e in self.store._events() if e["event_kind"] == "SETTLE" and e["payload"].get("reservation_id") == reference["reservation_id"]]
            require(len(settles) == 1, "preparation settlement missing", "CORRUPT_ARTIFACT")
            settled = settles[0]
            # Serialize the already validated worker document, not numerical
            # conversion/fitting. Publish a train-only artifact so a training
            # consumer need not open the selection-bearing preparation result.
            require(_sources(self.store, original, arm, selections) == descriptors, "source authority expired before result read", "UNAUTHORIZED_DATA")
            value = json.loads(self.store._verified_artifact_content(self.store._manifest("artifact-" + outcome["artifact_id"])))
            _validate_result(value, request)
            population = value["training_population"]
            require(_sources(self.store, original, arm, selections) == descriptors, "source authority expired before train publication", "UNAUTHORIZED_DATA")
            artifact = self.store.artifact(encode(population["document"]), role="training-population",
                visibility="synthetic" if all(c.get("visibility") == "synthetic" for c in original["cells"]
                    if c["block_id"] in population["provenance"]["independent_block_ids"]) else "restricted",
                block_ids=population["provenance"]["independent_block_ids"], study_id=original["study_id"])
            from application.pirc26_selection import selection_populations
            selection_artifacts = []
            for selected in selection_populations(value["blocks"], value["normalizer"], study_id=original["study_id"]):
                unit = selected["provenance"]["independent_block_id"]
                selected_artifact = self.store.artifact(encode(selected["document"]), role="selection-population",
                    visibility="synthetic" if all(c.get("visibility") == "synthetic" for c in original["cells"]
                        if c["block_id"] == unit) else "restricted",
                    block_ids=[unit], study_id=original["study_id"])
                selection_artifacts.append({"artifact_id": selected_artifact["artifact_id"],
                                            "provenance": selected["provenance"]})
            proof = {"schema_version": "pirc26-preparation-receipt-v2", "computation_ref": reference,
                "training_population": {"artifact_id": artifact["artifact_id"], "provenance": population["provenance"],
                                        "normalizer": value["normalizer"]},
                "selection_populations": selection_artifacts,
                "request_hash": digest(request), "result_artifact_id": outcome["artifact_id"], "settlement_event_hash": settled["hash"],
                "cost": {"arm_id": arm["arm_id"], "charged_ms": settled["payload"]["charged_ms"], "unit": "slot-ms",
                    "scope": "whole-shared-computation-job", "basis": "measured-monotonic"}}
            require(_sources(self.store, original, arm, selections) == descriptors, "source authority expired before receipt", "UNAUTHORIZED_DATA")
            self.store.publish(reference["manifest_id"], proof)
            _completion(self.store, original, arm, descriptors)
        value = load_prepared(self.store, reference)
        return {**outcome, "preparation_ref": reference, "prepared": value}
