"""Real published selection bytes in synthetic operator/admission controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from application.research_contracts import CapabilityRegistry
from experiments.pirc25.runner import SharedRunner
from infrastructure.research_store import ResearchStore, digest, encode
from tests.research_admission_fixtures import admit_fixture, synthetic_plugin
from tests.test_research_admission_chain import fixture_command
from tests.test_research_store import spec


PUBLIC_ROOT = Path(__file__).resolve().parents[1] / "experiments/pirc22"
PUBLIC_FILES = {"selection": "benchmark_selection.consumer.json", "matrix": "representation_matrix.json",
                "matrix-lock": "representation_matrix.lock.json"}


def public_fingerprints():
    return {name: hashlib.sha256((PUBLIC_ROOT / path).read_bytes()).hexdigest() for name, path in PUBLIC_FILES.items()}


def conditioner_binding(consumer):
    return {"selection_hash": consumer["source_selection_identity_sha256"],
            "consumer_identity_sha256": consumer["consumer_identity_sha256"],
            "matrix_identity_sha256": consumer["matrix_identity_sha256"],
            "configuration": deepcopy(consumer["selected_configuration"])}


def selection_fixture(root, mutate=None):
    documents = {name: json.loads((PUBLIC_ROOT / path).read_bytes()) for name, path in PUBLIC_FILES.items()}
    if mutate:
        mutate(documents)
    records = []
    for name, payload in documents.items():
        path = root / (name + ".json")
        content = encode(payload)
        path.write_bytes(content)
        checks = {"schema_version": payload["schema_version"]}
        if name == "selection":
            checks.update({key: payload[key] for key in ("status", "final_eval_read_count",
                "immutability_policy", "source_selection_identity_sha256")})
        record = {"object_id": name, "issue": "PIRC-22", "schema_version": payload["schema_version"],
            "acceptance_commit": "02e35638e621a7bef91a773b543a51b480a7f417",
            "code_sha": "c0d4283b77be230a311b8064578c3c2e07a6dd23",
            "artifact_id": "synthetic-control-" + name, "artifact_hash": hashlib.sha256(content).hexdigest(),
            "artifact_size_bytes": len(content), "canonical_hash": digest(payload), "path": path.name,
            "format": "json-metadata", "role": "metadata-only", "kind": "terrain-selection" if name == "selection" else "metadata",
            "metadata_checks": checks, "license": {"id": "explicit-synthetic-test-control", "scope": "metadata-only"},
            **{key: {"not_applicable": "public metadata control, no raw data license or new acceptance asserted"}
               for key in ("data_hash", "split_hash", "fold_hash", "feature_hash", "selection_hash")}}
        if name == "selection":
            record.update(selection_hash=payload["source_selection_identity_sha256"], benchmark_binding={
                "matrix_object_id": "matrix", "matrix_lock_object_id": "matrix-lock"})
        records.append(record)
    # Mutated copies are adversarial operator controls, NOT newly accepted
    # PIRC-22 versions. Referenced Git objects are real, no benchmark is run.
    accepted = [{"status": "accepted", "input": deepcopy(row)} for row in records]
    common = conditioner_binding(documents["selection"])
    cutover = {"mode": "adopted_primary", "selection_hash": common["selection_hash"], "conditioner_binding": common}
    manifest = {"schema_version": "pirc25-upstream-snapshot-v1", "inputs": records,
        "studies": [{"study_id": "synthetic-study", "pirc22_cutover": cutover},
                    {"study_id": "independent-study", "pirc22_cutover": {"mode": "not_applicable"}}],
        "cells": [{"cell_id": "dependent", "study_id": "synthetic-study", "role": "primary",
                   "conditioner_binding": deepcopy(common), "upstream_ids": list(PUBLIC_FILES)},
                  {"cell_id": "independent", "study_id": "independent-study", "role": "primary", "upstream_ids": []}]}
    return documents, manifest, accepted


def selection_admission(root, *, mutate=None, cell_change=None):
    documents, manifest, accepted = selection_fixture(root, mutate)
    store = ResearchStore(root, "selection-control", initialize=True)
    value = spec()
    value["cells"][0].update(plugin_id="admitted-fixture", capability="generic-rollout", visibility="synthetic",
                              conditioner_binding=conditioner_binding(documents["selection"]))
    value["arms"].append({**value["arms"][0], "arm_id": "candidate", "model_family_id": "candidate"})
    value["cells"].append({**deepcopy(value["cells"][0]), "arm_id": "candidate"})
    if cell_change:
        cell_change(value["cells"])
    plugin = synthetic_plugin("admitted-fixture", frozenset({"generic-rollout"}),
        ("x", "y", "vx", "vy"), ("m", "m", "m/s", "m/s"), "restart-only", fixture_command)
    registry = CapabilityRegistry()
    registry.register(plugin)
    grant = admit_fixture(store, value, plugin, root, formal=True, upstream_inputs=manifest["inputs"],
        accepted_versions=accepted, pirc22_cutover=manifest["studies"][0]["pirc22_cutover"])
    store.register(value, digest(value))
    return store, value, SharedRunner(store, registry), grant
