"""Metadata-only pilot registration with exact per-cell package bindings."""
from pathlib import Path

from application.research_admission import command_binding, data_binding, plugin_binding
from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate, protocol_binding
from application.research_upstream import register_acceptance_catalog, study_binding
from experiments.pirc25.affine import code_hash
from experiments.pirc25.snapshot import UpstreamSnapshot
from infrastructure.research_store import digest, encode


def register_pilot(store, *, study, authorization, arms, cells, plugins, recipe,
                   decision, source_hash, selection_rule, stopping_rule, policies, recovery_commands=None):
    """No workers, timers, grants for real data, or automatic qualification."""
    artifact = store.artifact(encode(recipe), role="pilot-input", visibility="synthetic",
                              block_ids=[recipe["source_family"]], study_id=study)
    protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": study + "-inputs",
        "study_id": study, "blocks": [{"block_id": recipe["source_family"], "dataset_id": recipe["source_family"],
        "release_id": "affine-oracle-suite-v1", "source_block_id": recipe["source_family"],
        "sha256": artifact["artifact_id"], "path": artifact["artifact_id"],
        "split_role": "validation", "fit_scope": False}]}
    evidence = {"entries": [], "reason": "public fixed affine generator; no terrain or real-data dependency"}
    catalog = {"schema_version": "pirc25-upstream-acceptance-v1", "visibility": "synthetic",
        "source": "PIRC-27 method qualification pilot", "entries": [],
        "source_evidence": evidence, "source_evidence_hash": digest(evidence)}
    catalog_hash = register_acceptance_catalog(store, catalog, digest(catalog))
    cutover = {"study_id": study, "pirc22_cutover": {"mode": "not_applicable"}}
    snapshot = UpstreamSnapshot({"schema_version": "pirc25-upstream-snapshot-v1", "visibility": "synthetic",
        "inputs": [], "studies": [cutover], "cells": [{"cell_id": digest(c), "study_id": study,
        "role": c["study_role"], "upstream_ids": []} for c in cells]})
    snapshot_hash = store.publish("upstream-snapshot-" + snapshot.snapshot_hash, snapshot.definition)
    plan = {"schema_version": "pirc25-preregistration-v1", "test_mode": "exploratory", "study_ids": [study],
        "protocol_bindings": [protocol_binding(protocol)],
        "primary_metrics": ["functional-absolute-error", "sampling-uncertainty", "charged-slot-ms", "failure-rate"],
        "selection_rule": selection_rule, "stopping_rule": stopping_rule,
        "comparisons": [{"kind": "per-cell-bounded-functional-error",
            "reference": "declared-continuous-affine-law-computed-in-each-funded-worker",
            "candidates": [a["arm_id"] for a in arms], "cross-method-superiority_claim": False}],
        "upstream_bindings": {study: study_binding(snapshot_hash, catalog_hash, cutover)},
        "orchestrator_source_sha256": source_hash, "qualification_policies": policies,
        "user_execution_authority": decision, "scientific_independent_n": 0}
    protocol["preregistration_hash"] = PreregistrationGate(store).register_preregistration(plan, digest(plan))
    EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
    authority = store.publish("pirc27-method-pilot-authority-" + digest(decision), decision)
    store.authorize({"authorization_id": authorization, "study_id": study,
        "expires_at": "2026-11-13T00:00:00+08:00", "evidence_hash": authority,
        "protocol_hash": digest(protocol), "data_root": str(store.path / "artifacts"),
        "test_authorization": False, "purposes": ["execute", "validate", "evaluate", "preview", "export"],
        "visibilities": ["synthetic"], "block_ids": [recipe["source_family"]]})
    packages, bindings = {}, []
    for cell in cells:
        plugin = plugins[cell["plugin_id"]]
        if plugin.plugin_id not in packages:
            payload = {"scope": "affine method qualification pilot, not formal scientific admission",
                       "source_artifact_id": artifact["artifact_id"], "plan_hash": digest(plan)}
            package = {"schema_version": "pirc25-package-v1", "kind": "PropagationResult", "study_id": study,
                "visibility": "synthetic", "state_order": list(plugin.state_order), "units": list(plugin.units),
                "resume_level": plugin.resume_level, "capabilities": sorted(plugin.capabilities), "code_hash": code_hash(),
                "data_hash": data_binding(protocol), "input_hash": data_binding(protocol), "protocol_hash": digest(protocol),
                "output_hash": digest(payload), "payload": payload, "plugin_hash": plugin_binding(plugin),
                "command_hash": command_binding(plugin.command_builder),
                "recovery_command_hash": command_binding((recovery_commands or {}).get(plugin.plugin_id, plugin.command_builder)),
                "upstream_snapshot_hash": snapshot_hash, "upstream_acceptance_hash": catalog_hash, "qualification": "fixture"}
            packages[plugin.plugin_id] = store.publish("package-" + digest(package), package)
        bindings.append({"cell_hash": digest(cell), "package_hash": packages[plugin.plugin_id]})
    spec = {"schema_version": "pirc25-contract-v1", "study_id": study, "experiment_id": study,
        "comparison_family": "pirc27-affine-method-qualification-pilot", "code_hash": code_hash(),
        "protocol_hash": digest(protocol), "data_hash": data_binding(protocol), "feature_hash": digest({"terrain": "none"}),
        "selection_hash": digest(plan), "arms": arms, "cells": cells,
        "runtime_binding": {"root": str(store.path.parent), "store_id": store.store_id},
        "orchestrator_source_sha256": source_hash, "user_authority_hash": authority,
        "maximum_new_research_slot_seconds": 14400, "scientific_independent_n": 0,
        "admission": {"mode": "pilot", "purpose": "validate", "protocol_id": protocol["protocol_id"],
        "authorization_id": authorization, "cell_packages": {"schema_version": "pirc25-cell-packages-v1", "bindings": bindings},
        "upstream_snapshot_hash": snapshot_hash, "upstream_acceptance_hash": catalog_hash,
        "upstream_root": str(store.path / "artifacts")}}
    store.register(spec, digest(spec))
    return spec
