"""Per-cell references select inputs, never create permission or qualification."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from application.research_admission import AdmissionGate, command_binding, plugin_binding
from application.research_budget import BudgetLedger, BudgetSpec
from application.research_contracts import CapabilityRegistry
from application.research_evidence import export_evidence
from application.research_execution import execution_binding
from domain.propagation import PropagationRequest
from experiments.pirc25.affine import code_hash
from experiments.pirc25.runner import SharedRunner
from experiments.pirc27.nonlinear import nonlinear_package
from experiments.pirc27.oracles import oracle_suite
from experiments.pirc27.plugin import (execution_config, execution_inputs, propagation_plugin,
                                      propagation_resume_command)
from infrastructure.research_admission_selection import (admission_package_references,
    cell_package_bindings, select_admission_package, verify_admission_selection)
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from infrastructure.research_visibility import study_visibility
from tests.research_admission_fixtures import admit_fixture


def table_spec():
    cells = [{"cell": "a"}, {"cell": "b"}, {"cell": "unavailable",
        "execution_disposition": {"schema_version": "pirc25-execution-disposition-v1",
            "status": "NOT_IMPLEMENTED", "reason": "frozen unavailable row"}}]
    return {"cells": cells, "admission": {"mode": "fixture", "protocol_id": "shared",
        "authorization_id": "shared-grant", "upstream_root": "unchanged-root",
        "cell_packages": {"schema_version": "pirc25-cell-packages-v1", "bindings": [
            {"cell_hash": digest(cell), "package_hash": digest({"source": index})}
            for index, cell in enumerate(cells[:2])]}}}


def test_selection_is_detached_exact_coverage_and_keeps_common_authority():
    spec = table_spec()
    entry = spec["admission"]["cell_packages"]["bindings"][1]
    entry.update(model_authorization_id="model-owner", model_protocol_id="model-inputs",
                 model_authorization_version="v1")
    before = deepcopy(spec)
    settings, selection = select_admission_package(spec, spec["cells"][1])
    assert settings == {**{k: v for k, v in spec["admission"].items() if k != "cell_packages"},
                        **{k: v for k, v in entry.items() if k != "cell_hash"}}
    assert selection["binding_hash"] == digest(entry)
    assert selection["table_hash"] == digest(spec["admission"]["cell_packages"])
    settings["authorization_id"] = "not-registered"
    bindings = cell_package_bindings(spec)
    bindings[digest(spec["cells"][0])]["package_hash"] = "changed"
    assert spec == before
    assert set(admission_package_references(spec)) == {b["package_hash"]
        for b in spec["admission"]["cell_packages"]["bindings"]}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "blocked", "default", "model-default",
    "mode", "protocol", "grant", "upstream", "version-without-id", "bad-hash", "bad-schema",
    "extra-table-key", "duplicate-cell", "oversized", "malformed-disposition"])
def test_invalid_or_ambiguous_tables_fail_closed_without_fallback(mutation):
    spec = table_spec()
    settings = spec["admission"]
    table = settings["cell_packages"]
    entry = table["bindings"][0]
    if mutation == "missing":
        table["bindings"].pop()
    elif mutation == "duplicate":
        table["bindings"].append(deepcopy(entry))
    elif mutation in {"foreign", "blocked"}:
        entry["cell_hash"] = digest(spec["cells"][2]) if mutation == "blocked" else digest("foreign")
    elif mutation in {"default", "model-default"}:
        settings["package_hash" if mutation == "default" else "model_protocol_id"] = digest("fallback")
    elif mutation in {"mode", "protocol", "grant", "upstream"}:
        entry[{"mode": "mode", "protocol": "protocol_id", "grant": "authorization_id",
               "upstream": "upstream_root"}[mutation]] = "override"
    elif mutation == "version-without-id":
        entry["model_authorization_version"] = "v1"
    elif mutation == "bad-hash":
        entry["package_hash"] = "NOT-A-HASH"
    elif mutation == "bad-schema":
        table["schema_version"] = "unknown"
    elif mutation == "extra-table-key":
        table["fallback"] = "forbidden"
    elif mutation == "duplicate-cell":
        spec["cells"].append(deepcopy(spec["cells"][0]))
    elif mutation == "oversized":
        table["bindings"] = [entry] * 10001
    else:
        spec["cells"][2]["execution_disposition"]["status"] = "SUCCEEDED"
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        select_admission_package(spec, spec["cells"][0])


def test_legacy_single_package_is_not_a_table_fallback_and_blocked_cells_cannot_select():
    spec = table_spec()
    with pytest.raises(ResearchError, match="NOT_IMPLEMENTED"):
        select_admission_package(spec, spec["cells"][2])
    with pytest.raises(ResearchError, match="CONTRACT_MISMATCH"):
        select_admission_package(spec, {"cell": "foreign"})
    spec["admission"] = {"mode": "fixture", "package_hash": digest("legacy")}
    settings, selection = select_admission_package(spec, spec["cells"][0])
    assert settings == spec["admission"] and selection is None
    assert admission_package_references(spec) == (settings["package_hash"],)
    verify_admission_selection(spec, spec["cells"][0], {})
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        verify_admission_selection(spec, spec["cells"][0], {"admission_selection": {"forged": True}})


@pytest.mark.parametrize("mutation", ["missing", "table", "binding", "cell", "package", "mode", "body"])
def test_receipt_selection_must_match_the_entire_frozen_table(mutation):
    spec = table_spec()
    cell = spec["cells"][0]
    settings, selection = select_admission_package(spec, cell)
    receipt = {"admission_selection": selection, "mode": "fixture", "documents": {"package": {"source": 0}}}
    assert verify_admission_selection(spec, cell, receipt) == settings
    if mutation == "missing":
        receipt.pop("admission_selection")
    elif mutation in {"table", "binding", "cell", "package"}:
        selection[mutation + "_hash"] = digest("substitution")
    elif mutation == "mode":
        receipt["mode"] = "formal"
    else:
        receipt["documents"]["package"]["source"] = 1
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        verify_admission_selection(spec, cell, receipt)


@pytest.mark.parametrize("input_kind", ["registered-protocol", "builtin-affine-generator", None])
def test_result_validator_cannot_skip_selection_by_changing_input_kind(input_kind):
    spec = table_spec()
    receipt = {"input_kind": input_kind}
    with pytest.raises(ResearchError, match="UNQUALIFIED"):
        AdmissionGate(None).result_validator(receipt, spec, spec["cells"][0], None)({})


def test_table_byte_quota_is_independent_of_the_cell_count_limit():
    cells = [{"cell": index} for index in range(10000)]
    settings = {"cell_packages": {"schema_version": "pirc25-cell-packages-v1", "bindings": [
        {"cell_hash": digest(cell), "package_hash": digest("source"),
         "model_authorization_id": "a"*120, "model_authorization_version": "v"*120,
         "model_protocol_id": "p"*120} for cell in cells]}}
    with pytest.raises(ResearchError, match="TOO_LARGE"):
        cell_package_bindings({"cells": cells, "admission": settings})


@pytest.mark.parametrize("restricted_source", ["package", "model", "qualification"])
def test_unexecuted_mapped_sources_keep_study_visibility_conservative(restricted_source):
    spec = table_spec()
    for cell in spec["cells"]:
        cell["visibility"] = "synthetic"
    # Rebind changed cell identities before freezing the complete table.
    spec["admission"]["cell_packages"]["bindings"] = []
    documents = {}
    for index, cell in enumerate(spec["cells"][:2]):
        package = {"visibility": "synthetic"}
        if index == 1:
            if restricted_source == "package":
                package["visibility"] = "restricted"
            elif restricted_source == "model":
                model = {"visibility": "restricted"}
                documents["package-" + digest(model)] = model
                package.update(requires_frozen_model=True, model_hash=digest(model))
            else:
                report = {"checks": [{"artifact_id": "secret-check"}]}
                documents["qualification-" + digest(report)] = report
                documents["artifact-secret-check"] = {"visibility": "restricted"}
                package["qualification_hash"] = digest(report)
        documents["package-" + digest(package)] = package
        spec["admission"]["cell_packages"]["bindings"].append(
            {"cell_hash": digest(cell), "package_hash": digest(package)})
    assert study_visibility(documents.__getitem__, spec) == "restricted"


def mixed_fixture(tmp_path, *, second_visibility="synthetic"):
    """One explicit disposable ledger; no scientific grants or data."""
    store = ResearchStore(tmp_path, "mixed-unit", initialize=True)
    case = oracle_suite()[0]
    cells, plugins, configs, inputs = [], [], [], []
    for synthetic in (False, True):
        package = nonlinear_package() if synthetic else case.package
        plugin = propagation_plugin(recovery=not synthetic, synthetic=synthetic)
        arm = "nonlinear-method" if synthetic else "affine-method"
        request = PropagationRequest("endpoint-fixture", package.package_hash, case.initial_mean,
            case.initial_covariance, 0.0, 0.0, (1.0,), "endpoint-x", 11, "paired-root", arm,
            samples=16, steps=4, chunk_size=8)
        config = execution_config(request, "cubature" if synthetic else "euler",
                                  recovery=not synthetic, synthetic=synthetic)
        input_binding = execution_inputs(request)
        cell = {"arm_id": arm, "block_id": "generator-v1", "seed": 11, "horizon": 1.0,
            "plugin_id": plugin.plugin_id, "capability": "generic-rollout", "visibility": "synthetic",
            "frozen_dynamics": package.manifest(), "propagation_request": json.loads(encode(asdict(request))),
            "resource_class": "cpu", "execution": execution_binding(plugin.registry_entry, config,
                input_binding, matrix_cells=2)}
        cells.append(cell)
        plugins.append(plugin)
        configs.append(config)
        inputs.append(input_binding)
    spec = {"schema_version": "pirc25-contract-v1", "study_id": "mixed-unit", "experiment_id": "oracle-unit",
        "comparison_family": "synthetic-engineering", "code_hash": code_hash(),
        "protocol_hash": digest("placeholder"), "data_hash": digest("placeholder"),
        "feature_hash": digest("no-terrain"), "selection_hash": digest("none"),
        "arms": [{"arm_id": cell["arm_id"], "model_family_id": "tanh-stress-v1" if index else "affine-stable-v1",
            "method_family_id": "cubature" if index else "euler", "objective_id": "endpoint-x",
            "budget_seconds": 86400} for index, cell in enumerate(cells)],
        "cells": cells, "runtime_binding": {"root": str(tmp_path.resolve()), "store_id": store.store_id}}
    grant = admit_fixture(store, spec, plugins[0], tmp_path, execution_config=configs[0], execution_inputs=inputs[0],
                          recovery_command_builder=propagation_resume_command, legacy_upstream=False)
    first_hash = spec["admission"].pop("package_hash")
    second = store.manifest("package-" + first_hash)
    payload = {"synthetic_adapter": plugins[1].plugin_id}
    second.update(plugin_hash=plugin_binding(plugins[1]), command_hash=command_binding(plugins[1].command_builder),
        recovery_command_hash=command_binding(plugins[1].command_builder), resume_level=plugins[1].resume_level,
        capabilities=sorted(plugins[1].capabilities), payload=payload, output_hash=digest(payload),
        visibility=second_visibility)
    second_hash = store.publish("package-" + digest(second), second)
    spec["admission"]["cell_packages"] = {"schema_version": "pirc25-cell-packages-v1", "bindings": [
        {"cell_hash": digest(cell), "package_hash": reference}
        for cell, reference in zip(cells, (first_hash, second_hash))]}
    registry = CapabilityRegistry()
    for plugin in plugins:
        registry.register(plugin)
    return store, spec, registry, grant, plugins


def test_heterogeneous_shared_matrix_runs_reuses_charges_and_exports_bound_selections(tmp_path):
    store, spec, registry, grant, _ = mixed_fixture(tmp_path)
    store.register(spec, digest(spec))
    runner = SharedRunner(store, registry)
    outcomes = [runner.run_cell(spec["study_id"], digest(cell), budget=BudgetSpec(60, category="smoke"))
                for cell in spec["cells"]]
    assert all(result["state"] == "SUCCEEDED" for result in outcomes), outcomes
    balances = {cell["arm_id"]: BudgetLedger(store).balance(cell["arm_id"])["committed_ms"]
                for cell in spec["cells"]}
    assert all(value > 0 for value in balances.values())
    for cell, result in zip(spec["cells"], outcomes):
        reused = runner.run_cell(spec["study_id"], digest(cell))
        assert reused["reused"] and reused["artifact_id"] == result["artifact_id"]
        assert BudgetLedger(store).balance(cell["arm_id"])["committed_ms"] == balances[cell["arm_id"]]
    bundle = export_evidence(store, spec["study_id"], grant)
    assert len(bundle["cells"]) == 2 and bundle["visibility"] == "synthetic"
    for row, cell in zip(bundle["cells"], spec["cells"]):
        receipt = row["admission"]
        verify_admission_selection(spec, cell, receipt)
        assert receipt["qualification"] == "fixture" and row["cost"]["charged_ms"] > 0
    assert len([event for event in store.events() if event["event_kind"] == "RESERVE"]) == 2


@pytest.mark.parametrize("fault", ["wrong-plugin-package", "missing-grant", "closed-arm"])
def test_mapped_admission_does_not_bypass_normal_shared_gates(tmp_path, fault):
    store, spec, registry, _, _ = mixed_fixture(tmp_path)
    cell = spec["cells"][1]
    if fault == "wrong-plugin-package":
        bindings = spec["admission"]["cell_packages"]["bindings"]
        bindings[1]["package_hash"] = bindings[0]["package_hash"]
    elif fault == "missing-grant":
        spec["admission"]["authorization_id"] = "not-registered"
    store.register(spec, digest(spec))
    if fault == "closed-arm":
        store.append("ARM_CLOSED", {"arm_id": cell["arm_id"], "reason": "synthetic closure control"})
    with pytest.raises(ResearchError):
        SharedRunner(store, registry).run_cell(spec["study_id"], digest(cell))
    assert not any(event["event_kind"] == "WORKER_STARTED" for event in store.events())
    balance = BudgetLedger(store).balance(cell["arm_id"])
    assert balance["committed_ms"] == 0 and balance["remaining_ms"] == 86400000


def test_resume_admission_selects_the_same_cell_package_without_rebinding_authority(tmp_path):
    store, spec, _, _, plugins = mixed_fixture(tmp_path)
    store.register(spec, digest(spec))
    cell = spec["cells"][0]
    attempt = store.new_attempt(store.register_run(spec["study_id"], cell))
    gate = AdmissionGate(store)
    ordinary = gate.prepare(spec, cell, plugins[0], attempt)
    resumed = gate.prepare(spec, cell, plugins[0], attempt, recovery_builder=propagation_resume_command)
    assert ordinary["admission_selection"] == resumed["admission_selection"]
    for name in ("package", "protocol", "authorization"):
        assert ordinary["documents"][name] == resumed["documents"][name]
    for name in ("snapshot", "acceptance_catalog"):
        assert ordinary["documents"]["upstream_snapshot"][name] == resumed["documents"]["upstream_snapshot"][name]
    assert ordinary["command_hash"] != resumed["command_hash"]
    assert resumed["execution_kind"] == "resume"
    verify_admission_selection(spec, cell, resumed)


def test_unrun_restricted_mapped_package_refuses_export_before_any_artifact_read(tmp_path, monkeypatch):
    store, spec, _, grant, _ = mixed_fixture(tmp_path, second_visibility="restricted")
    store.register(spec, digest(spec))
    def forbidden_read(*args, **kwargs):
        pytest.fail("restricted mapped source must be refused before artifact reads")
    monkeypatch.setattr(store, "read_artifact", forbidden_read)
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"):
        export_evidence(store, spec["study_id"], grant)
    assert not any(event["payload"].get("object_id", "").startswith("bundle-") for event in store.events())
