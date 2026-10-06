"""Recorded per-cell upstream facts; never resolves/open upstream source paths."""
from infrastructure.research_store import ResearchError, digest
from .research_upstream import _document

_CODES = {"MISSING_ARTIFACT", "IDENTITY_MISMATCH", "UNACCEPTED_VERSION"}


def require(condition):
    if not condition:
        raise ResearchError("CONTRACT_MISMATCH", "recorded upstream view evidence differs from frozen study")


def recorded_cells(store, spec, events, selected, *, spec_hash):
    settings = spec.get("admission", {})
    configured = "upstream_snapshot_hash" in settings or "upstream_acceptance_hash" in settings
    definition, definition_error = None, None
    if configured:
        try:
            definition = _document(store, "snapshot", settings.get("upstream_snapshot_hash"))
        except ResearchError as exc:
            if exc.code not in _CODES:
                raise
            definition_error = exc.code
    frozen = {}
    if definition is not None:
        require(definition.get("schema_version") == "pirc25-upstream-snapshot-v1"
                and isinstance(definition.get("cells"), list))
        for cell in definition["cells"]:
            require(isinstance(cell, dict))
            if cell.get("study_id") == spec["study_id"]:
                key = cell.get("cell_id")
                require(isinstance(key, str) and key not in frozen
                        and isinstance(cell.get("upstream_ids"), list)
                        and all(isinstance(name, str) for name in cell["upstream_ids"]))
                frozen[key] = cell
        require(set(frozen) == {digest(cell) for cell in spec["cells"]})
    latest, validations, attempt_runs, runs = {}, {}, {}, {}
    for event in events:
        payload = event["payload"]
        if event["event_kind"] == "ATTEMPT":
            attempt_runs[payload["attempt_id"]] = payload["run_id"]
        if (event["event_kind"] in {"UPSTREAM_VALIDATION", "UPSTREAM_REFUSED"}
                and payload.get("study_id") == spec["study_id"]
                and payload.get("snapshot_hash") == settings.get("upstream_snapshot_hash")
                and payload.get("acceptance_catalog_hash") == settings.get("upstream_acceptance_hash")):
            key = payload.get("cell_hash")
            latest[key] = event
            if event["event_kind"] == "UPSTREAM_VALIDATION":
                validations[key] = event

    def consumer(payload, key):
        run_id, attempt_id = payload.get("run_id"), payload.get("attempt_id")
        require(isinstance(run_id, str) and isinstance(attempt_id, str) and attempt_runs.get(attempt_id) == run_id)
        if run_id not in runs:
            runs[run_id] = store.manifest("run-" + run_id)
        run = runs[run_id]
        require(run.get("run_id") == run_id and run.get("study_id") == spec["study_id"]
                and run.get("spec_hash") == spec_hash and run.get("cell_hash") == key)
        return {"study_id": spec["study_id"], "cell_hash": key, "spec_hash": spec_hash,
                "attempt_id": attempt_id, "run_id": run_id}

    rows = []
    for cell in selected:
        key = digest(cell)
        row = {"cell_id": key, "study_id": spec["study_id"], "arm_id": cell["arm_id"],
               "block_id": cell["block_id"], "seed": cell.get("seed"),
               "upstream_ids": frozen.get(key, {}).get("upstream_ids", []),
               "status": "NOT_CHECKED" if configured else "NOT_CONFIGURED",
               "rejected_inputs": [], "validation_ref": None, "refusal_ref": None}
        if definition_error:
            row.update(status="REFUSED", rejected_inputs=[{"code": definition_error}])
        event = validations.get(key)
        if event is not None:
            payload = event["payload"]
            validation = _document(store, "validation", payload.get("validation_hash"))
            expected = consumer(payload, key)
            require(validation.get("schema_version") == "pirc25-upstream-validation-v1"
                    and validation.get("consumer") == expected and validation.get("data_authorization") == "none"
                    and validation.get("snapshot_hash") == settings.get("upstream_snapshot_hash")
                    and validation.get("acceptance_catalog_hash") == settings.get("upstream_acceptance_hash")
                    and isinstance(validation.get("cells"), list) and len(validation["cells"]) == 1)
            checked = validation["cells"][0]
            require(isinstance(checked, dict) and checked.get("cell_id") == key
                    and checked.get("study_id") == spec["study_id"]
                    and checked.get("upstream_ids") == row["upstream_ids"]
                    and checked.get("status") == payload.get("status") in {"ready", "rejected"}
                    and checked.get("rejected_inputs") == payload.get("rejected_inputs")
                    and isinstance(checked["rejected_inputs"], list)
                    and isinstance(validation.get("validation_finished_at"), str))
            errors = []
            for error in checked["rejected_inputs"]:
                require(isinstance(error, dict) and error.get("code") in _CODES
                        and ("object_id" not in error or error["object_id"] in row["upstream_ids"]))
                errors.append({name: error[name] for name in ("object_id", "code") if name in error})
            row.update(status=checked["status"].upper(), rejected_inputs=errors,
                       validation_ref={"validation_hash": payload["validation_hash"], "event_hash": event["hash"],
                           "sequence": event["sequence"], "checked_at": validation["validation_finished_at"],
                           "attempt_id": expected["attempt_id"], "run_id": expected["run_id"]})
        event = latest.get(key)
        if event is not None and event["event_kind"] == "UPSTREAM_REFUSED":
            payload = event["payload"]
            consumer(payload, key)
            require(payload.get("error_code") in _CODES)
            # A selected rejection has both validation and refusal events. A
            # newer early failure must not inherit an older ready claim.
            same_check = row["validation_ref"] is not None and all(
                payload.get(name) == row["validation_ref"][name] for name in ("attempt_id", "run_id"))
            if not same_check or row["status"] != "REJECTED":
                row.update(status="REFUSED", rejected_inputs=[{"code": payload["error_code"]}])
            row["refusal_ref"] = {"event_hash": event["hash"], "sequence": event["sequence"],
                                  "attempt_id": payload.get("attempt_id"), "run_id": payload.get("run_id")}
        rows.append(row)
    return rows
