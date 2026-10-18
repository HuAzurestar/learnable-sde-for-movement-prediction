"""Current-code source renewal reuses only immutable synthetic recipes."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from application.research_admission import data_binding
from application.research_budget import BudgetLedger
from application.research_restart_adjudication import resolve_source_bridge
from experiments.pirc25.affine import code_hash
from experiments.pirc28.source_epoch import register_source_epoch
from experiments.pirc28.study import SyntheticStudyDefinition
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from tests.test_pirc28_study import definition_value
from tests.test_research_execution_policy import policy


def historical_source(tmp_path):
    store = ResearchStore(tmp_path, "source-epoch-fixture", initialize=True)
    current = SyntheticStudyDefinition.from_dict(definition_value())
    old_value = current.to_dict()
    old_value["runtime_code_hash"] = "b" * 64
    historical = SyntheticStudyDefinition.from_dict(old_value)
    store.publish("pirc28-study-definition-" + historical.definition_hash, old_value)
    binding = {"root": str(store.path.parent), "store_id": store.store_id}
    protocols = {}
    for proposal in historical.proposals(binding):
        content = encode(proposal["switching_recipe"])
        relative = "source-" + proposal["study_id"] + ".json"
        (store.path.parent / relative).write_bytes(content)
        protocol = {"schema_version": "pirc25-data-protocol-v1",
            "protocol_id": "input-" + proposal["study_id"], "study_id": proposal["study_id"],
            "blocks": [{"block_id": replicate, "dataset_id": "synthetic-pirc28",
                "release_id": historical.definition_hash, "source_block_id": replicate,
                "sha256": hashlib.sha256(content).hexdigest(), "path": relative,
                "split_role": "train", "fit_scope": True}
                for replicate in sorted({cell["block_id"] for cell in proposal["cells"]})]}
        protocols[proposal["study_id"]] = protocol
    prereg = historical.preregistration_fields(protocols)
    prereg_hash = digest(prereg)
    store.publish("preregistration-" + prereg_hash, prereg)
    for protocol in protocols.values():
        protocol["preregistration_hash"] = prereg_hash
        store.publish("protocol-" + protocol["protocol_id"], protocol)
    for proposal in historical.proposals(binding):
        study_id = proposal["study_id"]
        protocol = protocols[study_id]
        grant = {"authorization_id": "execute-" + study_id, "study_id": study_id,
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),
            "evidence_hash": digest(["historical-synthetic-scope", study_id]),
            "protocol_hash": digest(protocol), "data_root": str(store.path.parent),
            "test_authorization": False, "purposes": ["fit", "execute", "evaluate", "resume"],
            "visibilities": ["synthetic"],
            "block_ids": [block["block_id"] for block in protocol["blocks"]]}
        store.authorize(grant)
        proposal.update(schema_version=store.SCHEMA, protocol_hash=digest(protocol),
            data_hash=data_binding(protocol), admission={"mode": "pilot", "purpose": "fit",
                "protocol_id": protocol["protocol_id"], "authorization_id": grant["authorization_id"],
                "package_hash": digest(["historical-package", study_id])},
            comparison_plan={"reference_arm_id": "pirc28-k1", "candidate_arm_ids": ["pirc28-k2"],
                "preregistration_hash": prereg_hash,
                "failure_policy": "retain-and-exclude-incomplete-blocks"})
        store.register(proposal, digest(proposal))
    decision = policy(store.store_id)
    decision["source_code_hash"] = code_hash()
    reference = {"manifest_id": "policy-" + digest(decision), "sha256": digest(decision)}
    store.publish(reference["manifest_id"], decision)
    return store, historical, current, reference


def test_source_epoch_changes_only_runtime_binding_and_creates_no_work(tmp_path):
    store, historical, current, policy_ref = historical_source(tmp_path)
    before = store.events()
    balances = {arm: BudgetLedger(store).balance(arm) for arm in ("pirc28-k1", "pirc28-k2")}
    result = register_source_epoch(store, historical.definition_hash, current,
                                   execution_policy=policy_ref)
    bridge = resolve_source_bridge(store, result["source_bridge"],
        execution_policy=policy_ref, source_code_hash=code_hash())
    assert bridge["current_definition"].definition_hash == current.definition_hash
    assert result["physical_recipe_count"] == len(historical.study_ids)
    assert result["scientific_experiments_executed"] == 0
    assert not store.attempts()
    assert not any(event["event_kind"] in {"RESERVE", "ADMISSION", "WORKER_STARTED"}
                   for event in store.events()[len(before):])
    assert {arm: BudgetLedger(store).balance(arm) for arm in balances} == balances
    replay_before = store.events()
    assert register_source_epoch(store, historical.definition_hash, current,
        execution_policy=policy_ref)["source_bridge"] == result["source_bridge"]
    assert store.events() == replay_before


def test_source_epoch_refuses_any_scientific_plan_change(tmp_path):
    store, historical, current, policy_ref = historical_source(tmp_path)
    changed = current.to_dict()
    name = next(iter(changed["recipes"]))
    changed["recipes"][name]["inference"]["max_iterations"] += 1
    candidate = SyntheticStudyDefinition.from_dict(changed)
    before = store.events()
    with pytest.raises(ResearchError):
        register_source_epoch(store, historical.definition_hash, candidate,
                              execution_policy=policy_ref)
    assert store.events() == before
