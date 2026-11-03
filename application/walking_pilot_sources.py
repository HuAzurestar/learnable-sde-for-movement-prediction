"""Explicit old-study artifact grants and original-source segment lineage.

These helpers inspect no payload during configuration or preflight. The actual
reader is invoked only inside a funded, qualified supervisor worker.
"""
import hashlib
import json
import math
from datetime import datetime,timezone
from pathlib import Path

from infrastructure.research_store import ResearchError,digest


def consumer_scope(pilot_id, *, internal=False):
    """Machine-readable narrowing of the existing owner permission."""
    return {"schema_version":"walking-pilot-consumers-v1","pilot_id":pilot_id,
            "entrypoint":"walking-pilot-v1",
            "worker_stages":["prepare","train","predict"] if internal else ["source"],
            "controller":internal}


def validate_consumer(store,grant,consumer,metadata):
    """Re-read actual attempt/run/cell/config/policy at each disclosure guard."""
    from .research_budget import BudgetLedger
    from .walking_pilot_budget import policy,registered_binding
    from .walking_pilot import cells
    scope=grant.get("consumer_scope")
    if (not isinstance(scope,dict) or set(scope)!={"schema_version","pilot_id","entrypoint","worker_stages","controller"}
            or scope.get("schema_version")!="walking-pilot-consumers-v1"
            or scope.get("entrypoint")!="walking-pilot-v1"
            or type(scope.get("controller")) is not bool
            or not isinstance(scope.get("worker_stages"),list)
            or not set(scope["worker_stages"])<={"source","prepare","train","predict"}
            or not isinstance(consumer,dict) or consumer.get("entrypoint")!=scope["entrypoint"]
            or not isinstance(grant.get("consumer_arm_ids"),list) or not grant["consumer_arm_ids"]):
        raise ResearchError("UNAUTHORIZED_DATA","machine-bound pilot consumer scope required")
    p=policy(BudgetLedger(store),scope["pilot_id"])
    from .walking_execution import execution_config
    config=execution_config(store,scope["pilot_id"])
    study=store._manifest("study-"+p["study_id"])
    if (grant.get("pilot_study_id")!=p["study_id"] or config["study_id"]!=p["study_id"]
            or config["pilot_id"]!=p["pilot_id"] or p.get("config_hash")!=digest(config)
            or config["arms"]!=p["arms"] or study["spec"]["cells"]!=cells(config)
            or grant.get("source_inventory_sha256")!=config["source_inventory"]["sha256"]
            or grant.get("movement_scope")!="walking" or grant.get("test_authorization") is not False
            or grant.get("public_export") is not False):
        raise ResearchError("UNAUTHORIZED_DATA","pilot consumer authority/config differs")
    internal=metadata["study_id"]==p["study_id"]
    reference={"authorization_id":grant["authorization_id"],"version":grant.get("version")}
    if internal and reference!=config["artifact_grant"]:
        raise ResearchError("UNAUTHORIZED_DATA","internal grant differs from registered config")
    if consumer.get("controller") is True:
        if (set(consumer)!={"controller","pilot_id","entrypoint"} or not internal or not scope["controller"]
                or consumer["pilot_id"]!=p["pilot_id"] or not set(config["arms"])<=set(grant["consumer_arm_ids"])):
            raise ResearchError("UNAUTHORIZED_DATA","controller consumer outside scope")
        return
    if set(consumer)!={"attempt_id","run_id","entrypoint"}:
        raise ResearchError("UNAUTHORIZED_DATA","exact registered worker consumer required")
    attempt=store._attempts().get(consumer["attempt_id"])
    if not attempt or attempt["state"]!="RUNNING" or attempt["run_id"]!=consumer["run_id"]:
        raise ResearchError("UNAUTHORIZED_DATA","consumer has no active registered attempt")
    run=store._manifest("run-"+consumer["run_id"])
    actual=registered_binding(BudgetLedger(store),run)
    cell=run["cell"]
    if (actual is None or actual["pilot_id"]!=p["pilot_id"] or cell not in cells(config)
            or cell["pilot_stage"] not in scope["worker_stages"] or cell["pilot_arm"] not in grant["consumer_arm_ids"]):
        raise ResearchError("UNAUTHORIZED_DATA","consumer registered pilot/arm/stage outside scope")
    if not internal:
        source=next((s for s in config["sources"] if s["source_id"]==cell.get("source_id")),None)
        if (cell["pilot_stage"]!="source" or not source or source["artifact"]!=metadata
                or source["block_id"]!=cell["block_id"]
                or reference!={"authorization_id":source["authorization_id"],"version":source.get("authorization_version")}):
            raise ResearchError("UNAUTHORIZED_DATA","source consumer does not match registered source cell")


def checked_metadata(reference):
    content=Path(reference["path"]).read_bytes()
    if len(content)>1024**2 or hashlib.sha256(content).hexdigest()!=reference["sha256"]:
        raise ResearchError("CONTRACT_MISMATCH","metadata input hash or size differs")
    return json.loads(content)


def sources_from_metadata(inventory,grants,geometry):
    if (inventory.get("schema_version")!="pirc26-walking-pilot-source-inventory-v1"
            or geometry.get("schema_version")!="pirc26-pilot-original-segment-mapping-v1"
            or geometry["source_inventory_sha256"]!=grants["source_inventory_sha256"]
            or geometry["training_artifact_id"]!=inventory["training_artifact"]["artifact_id"]):
        raise ResearchError("CONTRACT_MISMATCH","original inventory/geometry/grant metadata differ")
    train_grant=next(g for g in grants["grants"] if g["purposes"]==["resume"])
    dev_grant=next(g for g in grants["grants"] if g["purposes"]==["evaluate"])
    rows=[]
    for source_id in inventory["training_source_ids"]:
        segments=[m["pooled_segment_id"] for m in geometry["training_segment_source_map"] if m["independent_block_id"]==source_id]
        if not segments:
            raise ResearchError("CONTRACT_MISMATCH","training source has no original segment mapping")
        rows.append({"source_id":source_id,"block_id":source_id,"role":"train","artifact":inventory["training_artifact"],
                     "segment_ids":segments,"authorization_id":train_grant["authorization_id"],"authorization_version":train_grant.get("version"),
                     "purpose":"resume","declared_status":"AVAILABLE"})
    for candidate in inventory["development_candidates"]:
        rows.append({"source_id":candidate["source_block_id"],"block_id":candidate["source_block_id"],"role":"selection",
                     "artifact":candidate["artifact"],"authorization_id":dev_grant["authorization_id"],"authorization_version":dev_grant.get("version"),
                     "purpose":"evaluate","declared_status":candidate["historical_source_status"]})
    return rows


def validate_source_metadata(config):
    inventory=checked_metadata(config["source_inventory"])
    grants=checked_metadata(config["source_grants"])
    geometry=checked_metadata(config["source_geometry"])
    if (grants["source_inventory_sha256"]!=config["source_inventory"]["sha256"]
            or geometry["source_inventory_sha256"]!=config["source_inventory"]["sha256"]
            or (not config.get("execution_id") and inventory["study_id"]!=config["study_id"]) or grants["pilot_study_id"]!=config["study_id"]
            or config["speed_limit"]!=inventory["walking_speed_assumption"]["U_m_s"]
            or config["speed_limit"]!=inventory["walking_speed_assumption"]["V_m_s"]
            or config["sources"]!=sources_from_metadata(inventory,grants,geometry)):
        raise ResearchError("CONTRACT_MISMATCH","frozen configuration differs from original metadata")
    return inventory,grants,geometry


def source_authority(store,config,source):
    with store._read_transaction():
        grant=_source_authority_in_scope(store,config,source)
        def expiry():
            if grant is not None and datetime.fromisoformat(grant["expires_at"])<=datetime.now(timezone.utc):
                raise ResearchError("UNAUTHORIZED_DATA","source authorization expired")
        store._read_completion(lambda: _source_authority_in_scope(store,config,source),expiry)
        return grant


def _source_authority_in_scope(store,config,source):
    if source["artifact"] is None:
        if source["role"]!="selection" or source["declared_status"]!="CONTRACT_REJECTED":
            raise ResearchError("UNAUTHORIZED_DATA","missing source cannot be substituted")
        return None
    metadata=store.manifest("artifact-"+source["artifact"]["artifact_id"])
    grant=store.authorization(source["authorization_id"],version=source.get("authorization_version"))
    expected_role="training-population" if source["role"]=="train" else "selection-population"
    if (metadata!=source["artifact"] or metadata["role"]!=expected_role or metadata["size_bytes"]>4*1024**2
            or metadata["study_id"]!=grant["study_id"] or source["purpose"] not in grant["purposes"]
            or source["artifact"]["artifact_id"] not in grant.get("artifact_ids",[])
            or not set(metadata["block_ids"])<=set(grant["block_ids"])
            or grant.get("pilot_study_id")!=config["study_id"]
            or grant.get("source_inventory_sha256")!=config["source_inventory"]["sha256"]
            or grant.get("movement_scope")!="walking" or grant.get("test_authorization") is not False
            or grant.get("public_export") is not False
            or datetime.fromisoformat(grant["expires_at"])<datetime.fromisoformat("2026-11-03T23:59:00+08:00")
            or datetime.fromisoformat(grant["expires_at"])<=datetime.now(timezone.utc)):
        raise ResearchError("UNAUTHORIZED_DATA","old-study source grant, artifact or consumer differs")
    return grant


def read_population(store,config,source,consumer):
    disclosure=_read_population_bytes(store,config,source,consumer)
    return _finish_population(store,config,source,consumer,disclosure)


def _read_population_bytes(store,config,source,consumer):
    """Protected bytes only; reusable inside the worker's short admission."""
    with store._read_transaction():
        from .walking_execution import execution_config
        if (config!=execution_config(store,config["pilot_id"],config.get("execution_id")) or source not in config["sources"]):
            raise ResearchError("UNAUTHORIZED_DATA","source differs from registered pilot config")
        grant=source_authority(store,config,source)
        if grant is None:
            return None
        artifact=source["artifact"]["artifact_id"]
        validate_consumer(store,grant,consumer,source["artifact"])
        run=store._manifest("run-"+consumer["run_id"])
        if run["cell"].get("source_id")!=source["source_id"]:
            raise ResearchError("UNAUTHORIZED_DATA","source differs from actual consumer cell")
        store.append("PILOT_SOURCE_CONSUMER",{"pilot_id":config["pilot_id"],"source_id":source["source_id"],
                     "artifact_id":artifact,"authorization_hash":digest(grant),**consumer})
        raw=store.read_artifact(artifact,purpose=source["purpose"],authorization=grant,consumer=consumer)
        return raw,grant,artifact


def _finish_population(store,config,source,consumer,disclosure):
    # JSON geometry and scientific work never own the admission lock. Final
    # source/consumer/permission verification still follows actual parsing.
    if disclosure is None:
        return {"source_id":source["source_id"],"status":"CONTRACT_REJECTED","segments":[]}
    raw,grant,artifact=disclosure
    from application.pirc26_data import _bounded_document
    document=_bounded_document(raw.decode("utf-8"))
    if (document.get("schema_version")!="pirc26-observed-block-v1" or document.get("time_unit")!="s"
            or document.get("state_units")!=["m","m","m/s","m/s"] or not isinstance(document.get("segments"),list)
            or not 1<=len(document["segments"])<=256):
        raise ResearchError("CONTRACT_MISMATCH","original physical population schema differs")
    allowed=set(source.get("segment_ids",[])); selected=[]; seen=set()
    for segment in document["segments"]:
        sid=segment["segment_id"]
        if sid in seen:
            raise ResearchError("CONTRACT_MISMATCH","duplicate original segment")
        seen.add(sid)
        if source["role"]=="train" and sid not in allowed:
            continue
        t,p=segment["time"],segment["position"]
        if (not 3<=len(t)<=4098 or len(t)!=len(p)
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in t)
                or any(a>=b for a,b in zip(t,t[1:])) or
                any(not isinstance(row,list) or len(row)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in row) for row in p)):
            raise ResearchError("CONTRACT_MISMATCH","finite native segment geometry required")
        selected.append({"segment_id":sid,"times":t,"positions":p})
    if source["role"]=="train" and {s["segment_id"] for s in selected}!=allowed:
        raise ResearchError("CONTRACT_MISMATCH","pooled artifact is missing original source segments")
    with store._read_transaction():
        source_authority(store,config,source)
        store.verify_artifact_read(artifact,purpose=source["purpose"],authorization=grant,consumer=consumer)
    return {"source_id":source["source_id"],"status":"AVAILABLE","segments":selected,
            "lineage":{"artifact_id":artifact,"original_study_id":grant["study_id"],"authorization_hash":digest(grant)}}
