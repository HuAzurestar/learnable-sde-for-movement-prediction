"""Register a current-code PIRC-28 synthetic source epoch without new science.

Only ``runtime_code_hash`` may change.  Original recipe files are reverified in
place; protocols, grants, packages and studies receive new immutable identities.
No attempt, reservation, artifact or scientific result is created.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib

from application.research_admission import command_binding, data_binding, plugin_binding
from application.research_contracts import validate_package
from application.research_data import EvaluationExposureLedger
from application.research_preregistration import PreregistrationGate
from application.research_restart_adjudication import scientific_cell_identity
from application.research_upstream import register_acceptance_catalog, validate_acceptance_catalog
from experiments.pirc25.affine import code_hash
from experiments.pirc25.snapshot import UpstreamSnapshot
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, digest, encode
from .runtime import resume_command, switching_plugin
from .study import SyntheticStudyDefinition, register_definition_studies, stage_definition


def _require(condition, message, code="CONTRACT_MISMATCH"):
    if not condition:
        raise ResearchError(code, message)


def _protocols(store, historical, current, proposals):
    result, physical = {}, {}
    for old_id, proposal in zip(historical.study_ids, proposals):
        original = store._manifest("protocol-input-" + old_id)
        protocol = deepcopy(original)
        protocol.pop("preregistration_hash", None)
        new_id = proposal["study_id"]
        protocol.update(protocol_id="input-" + new_id, study_id=new_id)
        content = encode(proposal["switching_recipe"])
        expected = hashlib.sha256(content).hexdigest()
        for block in protocol.get("blocks", []):
            _require(block.get("dataset_id") == "synthetic-pirc28"
                     and block.get("sha256") == expected
                     and block.get("split_role") == "train" and block.get("fit_scope") is True,
                     "historical synthetic protocol differs", "UNAUTHORIZED_DATA")
            relative = block.get("path")
            _require(type(relative) is str and relative not in physical
                     or physical.get(relative) == expected,
                     "one recipe path maps to different content", "IDENTITY_CONFLICT")
            if relative not in physical:
                with opened_regular_file(store.path.parent, store.path.parent / relative,
                        maximum_bytes=4 * 1024 * 1024, expected_size=len(content)) as (stream, _, verify):
                    _require(stream.read(len(content) + 1) == content,
                             "historical physical synthetic recipe differs", "CORRUPT_ARTIFACT")
                    verify()
                physical[relative] = expected
        result[new_id] = protocol
    _require(len(physical) == len(historical.study_ids),
             "one physical recipe per source study required")
    preregistration = current.preregistration_fields(result)
    preregistration_hash = digest(preregistration)
    for protocol in result.values():
        protocol["preregistration_hash"] = preregistration_hash
    return result, preregistration, physical


def register_source_epoch(store, historical_definition_hash, current_definition,
                          *, execution_policy, synthetic_grant_expires_at=None):
    """Publish a scope-preserving current-code source matrix and bridge."""
    from application.research_execution_policy import resolve_policy

    if synthetic_grant_expires_at is not None:
        try:
            _require(type(synthetic_grant_expires_at) is str and len(synthetic_grant_expires_at) <= 64,
                     'explicit finite synthetic renewal expiry required')
            expiry = datetime.fromisoformat(synthetic_grant_expires_at)
            now = datetime.now(timezone.utc)
            _require(expiry.tzinfo is not None and now < expiry <= now + timedelta(days=31),
                     'synthetic renewal expiry must be future and bounded to 31 days')
        except (ValueError, TypeError) as error:
            if isinstance(error, ResearchError):
                raise
            raise ResearchError('CONTRACT_MISMATCH', 'invalid synthetic renewal expiry') from error
    _require(isinstance(current_definition, SyntheticStudyDefinition),
             "canonical current source definition required")
    current_value = current_definition.to_dict()
    _require(current_value["runtime_code_hash"] == code_hash(),
             "current source definition code differs", "IDENTITY_CONFLICT")
    # Cold numerical/resource structure validation is pure and may take longer
    # than a peer writer's lock timeout. Prime canonical bytes before the
    # actual transaction, which still rereads their authoritative manifest.
    SyntheticStudyDefinition.from_dict(store.manifest(
        "pirc28-study-definition-" + historical_definition_hash))
    with store._read_transaction():
        historical = SyntheticStudyDefinition.from_dict(store._manifest(
            "pirc28-study-definition-" + historical_definition_hash))
        old_value = historical.to_dict()
        changed = [key for key in sorted(old_value) if old_value.get(key) != current_value.get(key)]
        _require(historical.definition_hash == historical_definition_hash
                 and set(old_value) == set(current_value) and changed == ["runtime_code_hash"],
                 "source renewal changes frozen scientific design", "IDENTITY_CONFLICT")
        resolve_policy(store, execution_policy, arm_id="pirc28-k1",
                       source_code_hash=current_value["runtime_code_hash"])
        binding = {"root": str(store.path.parent), "store_id": store.store_id}
        proposals = current_definition.proposals(binding)
        _require(len(proposals) == len(historical.study_ids),
                 "complete source study matrix required")
        existing = []
        for event in store._events():
            payload = event["payload"]
            if (event["event_kind"] == "MANIFEST"
                    and payload.get("object_id", "").startswith("pirc28-source-epoch-bridge-")):
                value = store._manifest(payload["object_id"])
                if (value.get("historical_definition_hash") == historical.definition_hash
                        and value.get("current_definition_hash") == current_definition.definition_hash
                        and value.get("execution_policy") == execution_policy):
                    existing.append({"manifest_id": payload["object_id"], "sha256": payload["sha256"]})
        _require(len(existing) <= 1, "multiple source epoch bridges differ", "IDENTITY_CONFLICT")
        if existing:
            from application.research_restart_adjudication import resolve_source_bridge
            resolve_source_bridge(store, existing[0], execution_policy=execution_policy,
                                  source_code_hash=current_value["runtime_code_hash"])
            bridge = store._manifest(existing[0]["manifest_id"])
            if synthetic_grant_expires_at is not None:
                scope = store._manifest('pirc28-synthetic-source-renewal-' + bridge['renewal_scope_hash'])
                _require(scope.get('expires_at') == synthetic_grant_expires_at,
                         'registered synthetic expiry cannot be changed in place', 'IDENTITY_CONFLICT')
            return {"source_bridge": existing[0],
                "study_definition_hash": current_definition.definition_hash,
                "study_ids": current_definition.study_ids,
                "preregistration_hash": bridge["preregistration_hash"],
                "renewal_scope_hash": bridge["renewal_scope_hash"],
                "physical_recipe_count": bridge["physical_recipe_count"],
                "scientific_experiments_executed": 0, "real_data_authorized": False}
        protocols, preregistration, physical = _protocols(
            store, historical, current_definition, proposals)

        old_specs = [store._manifest("study-" + study_id)["spec"]
                     for study_id in historical.study_ids]
        old_grants = [store.authorization(spec["admission"]["authorization_id"],
            version=spec["admission"].get("authorization_version")) for spec in old_specs]
        stable = ("purposes", "visibilities", "block_ids", "data_root", "test_authorization")
        baseline = old_grants[0]
        _require(all(all(grant.get(field) == baseline.get(field) for field in stable)
                     and grant.get("study_id") == spec["study_id"]
                     and grant.get("protocol_hash") == spec["protocol_hash"]
                     for grant, spec in zip(old_grants, old_specs))
                 and baseline.get("visibilities") == ["synthetic"]
                 and baseline.get("test_authorization") is False,
                 "historical synthetic grants do not share one exact scope", "UNAUTHORIZED_DATA")

        applicability = {"scope": "frozen standalone synthetic matrix only",
            "historical_definition_hash": historical.definition_hash,
            "current_definition_hash": current_definition.definition_hash,
            "runtime_code_hash": current_value["runtime_code_hash"],
            "execution_policy": deepcopy(execution_policy),
            "reason": "PIRC-28 source epoch renewal; no external scientific input accepted",
            "entries": []}
        catalog = {"schema_version": "pirc25-upstream-acceptance-v1",
            "source": "Explicit local synthetic applicability; no external scientific input accepted",
            "visibility": "synthetic", "entries": [], "source_evidence": applicability,
            "source_evidence_hash": digest(applicability)}
        validate_acceptance_catalog(catalog)
        catalog_hash = digest(catalog)
        snapshot = UpstreamSnapshot({"schema_version": "pirc25-upstream-snapshot-v1",
            "inputs": [], "visibility": "synthetic", "applicability": applicability,
            "studies": [{"study_id": proposal["study_id"], "pirc22_cutover": {
                "mode": "not_applicable", "reason": "Frozen no-terrain synthetic generator, no PIRC-22 conditioner"}}
                for proposal in proposals],
            "cells": [{"study_id": proposal["study_id"], "cell_id": digest(cell),
                "role": "primary", "upstream_ids": []}
                for proposal in proposals for cell in proposal["cells"]]})
        validation = snapshot.resolve(root=store.path.parent, accepted_versions=[])
        _require(not validation["resolved"] and not validation["rejected_inputs"]
                 and all(cell["status"] == "ready" for cell in validation["cells"]),
                 "current synthetic applicability refused", "UNAUTHORIZED_DATA")

        renewal_scope = {"schema_version": "pirc28-synthetic-source-renewal-v1",
            "execution_policy": deepcopy(execution_policy),
            "historical_definition_hash": historical.definition_hash,
            "current_definition_hash": current_definition.definition_hash,
            "historical_code_hash": old_value["runtime_code_hash"],
            "current_code_hash": current_value["runtime_code_hash"],
            "changed_fields": ["runtime_code_hash"],
            "purposes": deepcopy(baseline["purposes"]),
            "visibilities": deepcopy(baseline["visibilities"]),
            "test_authorization": False, "real_data_authorized": False,
            "public_export_authorized": False,
            "expires_at": synthetic_grant_expires_at or baseline["expires_at"]}
        scope_hash = digest(renewal_scope)
        plugin = switching_plugin()
        grants, packages, admissions = {}, {}, {}
        for proposal, old_grant in zip(proposals, old_grants):
            study_id = proposal["study_id"]
            protocol = protocols[study_id]
            grant = {"authorization_id": "execute-" + study_id, "study_id": study_id,
                "expires_at": synthetic_grant_expires_at or old_grant["expires_at"], "evidence_hash": scope_hash,
                "protocol_hash": digest(protocol), "data_root": old_grant["data_root"],
                "test_authorization": False, "purposes": deepcopy(old_grant["purposes"]),
                "visibilities": deepcopy(old_grant["visibilities"]),
                "block_ids": deepcopy(old_grant["block_ids"])}
            payload = {"schema_version": "pirc28-synthetic-source-package-v1",
                "input_kind": "complete frozen synthetic generator recipe",
                "study_definition_hash": current_definition.definition_hash,
                "recipe_hash": digest(proposal["switching_recipe"]),
                "scientific_outcome": "not_run", "real_data_authorized": False}
            package = {"schema_version": "pirc25-package-v1", "kind": "SwitchingResult",
                "study_id": study_id, "visibility": "synthetic", "state_order": list(plugin.state_order),
                "units": list(plugin.units), "resume_level": plugin.resume_level,
                "capabilities": sorted(plugin.capabilities), "code_hash": current_value["runtime_code_hash"],
                "data_hash": data_binding(protocol), "input_hash": data_binding(protocol),
                "protocol_hash": digest(protocol), "payload": payload, "output_hash": digest(payload),
                "plugin_hash": plugin_binding(plugin), "command_hash": command_binding(plugin.command_builder),
                "recovery_command_hash": command_binding(resume_command),
                "upstream_snapshot_hash": snapshot.snapshot_hash,
                "upstream_acceptance_hash": catalog_hash, "qualification": "fixture"}
            validate_package(package, kind="SwitchingResult", formal=False)
            grants[study_id], packages[study_id] = grant, package
            admissions[study_id] = {"mode": "pilot", "purpose": "fit",
                "protocol_id": protocol["protocol_id"], "authorization_id": grant["authorization_id"],
                "package_hash": digest(package), "upstream_snapshot_hash": snapshot.snapshot_hash,
                "upstream_acceptance_hash": catalog_hash, "upstream_root": str(store.path.parent)}

        # Every operation is immutable/idempotent. A crash can be replayed but
        # never yields an executable child until the final bridge exists.
        stage_definition(store, current_definition)
        PreregistrationGate(store).register_preregistration(preregistration, digest(preregistration))
        for protocol in protocols.values():
            EvaluationExposureLedger(store).register_protocol(protocol, digest(protocol))
        store.publish("pirc28-synthetic-source-renewal-" + scope_hash, renewal_scope)
        register_acceptance_catalog(store, catalog, catalog_hash)
        store.publish("upstream-snapshot-" + snapshot.snapshot_hash, snapshot.definition)
        for study_id in current_definition.study_ids:
            store.authorize(grants[study_id])
            store.publish("package-" + digest(packages[study_id]), packages[study_id])
        register_definition_studies(store, current_definition, admissions)

        pairs = []
        for old_id, new_id in zip(historical.study_ids, current_definition.study_ids):
            old_spec = store._manifest("study-" + old_id)["spec"]
            new_spec = store._manifest("study-" + new_id)["spec"]
            by_identity = {digest(scientific_cell_identity(cell)): cell for cell in new_spec["cells"]}
            cells = []
            for old_cell in old_spec["cells"]:
                identity_hash = digest(scientific_cell_identity(old_cell))
                new_cell = by_identity.get(identity_hash)
                _require(new_cell is not None, "renewed scientific cell absent", "IDENTITY_CONFLICT")
                cells.append({"scientific_cell_hash": identity_hash,
                    "historical_cell_hash": digest(old_cell), "current_cell_hash": digest(new_cell)})
            pairs.append({"historical_study_id": old_id, "current_study_id": new_id,
                "historical_spec_hash": digest(old_spec), "current_spec_hash": digest(new_spec),
                "historical_grant_hash": digest(store.authorization(old_spec["admission"]["authorization_id"],
                    version=old_spec["admission"].get("authorization_version"))),
                "current_grant_hash": digest(grants[new_id]), "cell_pairs": cells})
        bridge = {"schema_version": "pirc28-source-epoch-bridge-v1",
            "execution_policy": deepcopy(execution_policy),
            "historical_definition_hash": historical.definition_hash,
            "current_definition_hash": current_definition.definition_hash,
            "historical_code_hash": old_value["runtime_code_hash"],
            "current_code_hash": current_value["runtime_code_hash"],
            "changed_fields": ["runtime_code_hash"],
            "preregistration_hash": digest(preregistration),
            "renewal_scope_hash": scope_hash, "physical_recipe_count": len(physical),
            "study_pairs": pairs}
        reference = {"manifest_id": "pirc28-source-epoch-bridge-" + digest(bridge),
                     "sha256": digest(bridge)}
        store.publish(reference["manifest_id"], bridge)
        return {"source_bridge": reference, "study_definition_hash": current_definition.definition_hash,
                "study_ids": current_definition.study_ids, "preregistration_hash": digest(preregistration),
                "renewal_scope_hash": scope_hash, "physical_recipe_count": len(physical),
                "scientific_experiments_executed": 0, "real_data_authorized": False}
