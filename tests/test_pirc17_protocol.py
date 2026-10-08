"""Software-only fixtures. No test here grants or performs real final-eval access."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.pirc17 import final_eval_guard as guard
from experiments.pirc17 import protocol as proto
from experiments.pirc17 import protocol_inputs as inputs
from experiments.pirc17 import protocol_core as core


@pytest.fixture(scope="module")
def candidate():
    return proto.build_protocol()


def execution_for(candidate):
    """Schema fixture, NOT the real DEV04 execution manifest or runner."""
    p = core.unpack(candidate)
    return core.envelope({"schema_version": proto.EXECUTION_VERSION,
        "protocol_sha256": candidate["sha256"], "source_sha256": proto.source_catalog(),
        "matrix_sha256": core.digest("synthetic-matrix"), "runtime_manifest_sha256": core.digest("synthetic-runtime"),
        "budget_ledger_schema_version": "pirc17-cumulative-phase-budget-v1",
        "entrypoint": "PSDE-SDE/experiments/pirc17/protocol.py",
        "phase_caps_seconds": p["resource_contract"]["phase_caps_seconds"],
        "max_generated_forecasts": 11513, "scientific_forecasts": 11020,
        "method_required_slots": sorted(k for k, v in p["components"]["method_mechanisms"]["slots"].items() if v["disposition"] == "REQUIRED"),
        "terrain_configurations": sorted(p["components"]["terrain_configurations"]), "final_eval_authorized": False})


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{"a":-1e999}'])
def test_strict_json_refuses_duplicate_and_nonfinite(raw):
    with pytest.raises(ValueError):
        core.decode(raw)


@pytest.mark.parametrize("path", ["/etc/passwd", "../x", "a/../x", "C:/x", "a\\b", "a//b", "a/./b", "a/", "a\x00b"])
def test_relative_path_boundary(path, tmp_path):
    with pytest.raises(ValueError):
        core.under(tmp_path, path)


def test_immutable_publication_and_content_hash(tmp_path):
    path, value = core.publish(tmp_path, {"hello": "数据"})
    assert path.name == value["sha256"]+".json"
    assert core.unpack(core.read_json(path), expected_sha256=value["sha256"]) == {"hello": "数据"}
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        core.publish(tmp_path, {"hello": "数据"})
    assert path.read_bytes() == before
    value["payload"]["hello"] = "changed"
    with pytest.raises(ValueError):
        core.unpack(value)


def test_partial_overlarge_and_wrong_file_identity_rejected(tmp_path):
    p = tmp_path/"partial.json"
    p.write_bytes(b'{"payload":')
    with pytest.raises(ValueError):
        core.read_json(p)
    p.write_bytes(b'{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        core.read_json(p, expected_file_sha256="0"*64)
    with pytest.raises(ValueError, match="bounded"):
        core.read_json(p, max_bytes=1)


def test_symlink_escape_is_not_a_bound_source(tmp_path):
    root = tmp_path/"root"
    root.mkdir()
    target = tmp_path/"outside.py"
    target.write_text("x=1", encoding="utf-8")
    link = root/"link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("host lacks permission for symlink fixture")
    with pytest.raises(ValueError, match="escapes"):
        core.under(root, "link.py")


def test_resolved_ancestor_escape_rejected_even_without_host_symlink_privilege(tmp_path, monkeypatch):
    root, outside = tmp_path/"root", tmp_path/"outside"
    root.mkdir()
    actual_resolve = Path.resolve
    def resolved(self, *args, **kwargs):
        if self == root/"linked"/"source.py":
            return outside/"source.py"
        return actual_resolve(self, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolved)
    with pytest.raises(ValueError, match="escapes"):
        core.under(root, "linked/source.py")


def test_protocol_semantics_scope_and_roundtrip(candidate):
    p = proto.validate_protocol(core.decode(core.canonical(candidate)), expected_sha256=candidate["sha256"])
    assert p["dataset_inputs"]["partitions"]["counts"] == inputs.SPLIT_COUNTS
    assert len(p["dataset_inputs"]["online_maps"]["admitted_receipt_assets_sha256"]) == 1354
    assert all(n == 0 for pair in p["dataset_inputs"]["partitions"]["intersections"].values() for n in pair.values())
    assert p["dataset_inputs"]["history"]["prior_final_eval_exposed"] is True
    assert not p["dataset_inputs"]["history"]["new_untouched_holdout_required"]
    assert not p["dataset_inputs"]["history"]["forced_exploratory_downgrade"]
    assert p["components"]["method_mechanisms"]["counts"] == {"slots": 36, "required": 28, "excluded": 8}
    assert len(p["components"]["terrain_configurations"]) == 10
    assert p["generation_counts"]["total_stochastic_forecasts_including_audits"] == 11513
    assert p["generation_counts"]["scientific_forecasts_unchanged"] == 11020
    assert p["generation_counts"]["additional_scoring_only_Gaussian_draws"] == 593920
    assert sum(p["resource_contract"]["phase_caps_seconds"].values()) == 48*3600
    assert not p["final_eval_authorized"] and p["final_eval_label_prediction_metric_reads"] == 0
    assert not p["global_numerical_qualification_claimed"]
    assert p["component_precedence"]["allowed_new_precheck_forecasts"] == 0


@pytest.mark.parametrize("path,value", [
    (("forecast_contract", "forecast", "particles"), 1024),
    (("forecast_contract", "forecast", "maximum_step_seconds"), .1),
    (("forecast_contract", "forecast", "nominal_horizon_seconds"), 300.),
    (("forecast_contract", "primary_origin_mode"), "point_only"),
    (("resource_contract", "max_attempts_per_work_item"), 2),
    (("dataset_inputs", "history", "prior_final_eval_exposed"), False),
    (("dataset_inputs", "history", "new_untouched_holdout_required"), True),
    (("dataset_inputs", "history", "forced_exploratory_downgrade"), True),
    (("components", "finite_delivery", "statistics", "inference_config", "delta_m"), 1.),
    (("components", "finite_delivery", "statistics", "inference_config", "bootstrap_iterations"), 4000),
    (("final_eval_authorized",), True),
    (("final_eval_label_prediction_metric_reads",), False),
])
def test_rehashed_scientific_mutation_cannot_masquerade_as_fixed_protocol(candidate, path, value):
    payload = copy.deepcopy(core.unpack(candidate))
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="scientific/input/authority"):
        proto.validate_protocol(core.envelope(payload))


def test_source_edit_missing_binding_and_expected_identity_rejected(candidate, tmp_path):
    path = tmp_path/"source.py"
    path.write_text("x=1", encoding="utf-8")
    binding = {"FIXTURE/source.py": core.file_hash(path)}
    proto.verify_sources(binding, {"FIXTURE": tmp_path})
    path.write_text("x=2", encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        proto.verify_sources(binding, {"FIXTURE": tmp_path})
    bad = copy.deepcopy(core.unpack(candidate))
    del bad["source_sha256"]["PSDE-SDE/experiments/pirc17/final_eval_guard.py"]
    with pytest.raises(ValueError, match="incomplete"):
        proto.validate_protocol(core.envelope(bad))
    with pytest.raises(ValueError, match="identity mismatch"):
        proto.validate_protocol(candidate, expected_sha256="0"*64)


def test_builder_never_opens_raw_data_or_adapters(monkeypatch):
    from experiments.nex326.pirc21_adapter import FeatureSnapshotAdapter
    def trap(*args, **kwargs):
        pytest.fail("real feature adapter must not be instantiated to seal metadata")
    monkeypatch.setattr(FeatureSnapshotAdapter, "__init__", trap)
    original = Path.open
    def public_sources_only(self, *args, **kwargs):
        assert self.suffix in {".json", ".py", ".md"}, f"unexpected raw input {self}"
        assert "artifacts" not in self.parts, f"private empirical artifact {self}"
        assert "final_eval" not in self.parts
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", public_sources_only)
    p = proto.validate_protocol(proto.build_protocol())
    assert p["final_eval_label_prediction_metric_reads"] == 0


@pytest.mark.parametrize("mutation", ["count", "scope", "phase", "slot", "terrain", "source", "entrypoint", "ledger", "authorized"])
def test_execution_requires_complete_matching_scope(candidate, mutation):
    value = execution_for(candidate)
    proto.validate_execution(value, candidate)
    p = copy.deepcopy(value["payload"])
    if mutation == "count": p["max_generated_forecasts"] = 11223
    if mutation == "scope": p["protocol_sha256"] = "0"*64
    if mutation == "phase": p["phase_caps_seconds"]["method_forecasts"] += 1
    if mutation == "slot": p["method_required_slots"].pop()
    if mutation == "terrain": p["terrain_configurations"].pop()
    if mutation == "source": p["source_sha256"].pop(next(iter(p["source_sha256"])))
    if mutation == "entrypoint": p["entrypoint"] = "../runner.py"
    if mutation == "ledger": p["budget_ledger_schema_version"] = "reset-on-resume"
    if mutation == "authorized": p["final_eval_authorized"] = True
    with pytest.raises(ValueError):
        proto.validate_execution(core.envelope(p), candidate)


def identity_fixture():
    return [{"split": role, "sample_id": core.digest(role), "segment_id": "segment-"+role,
             "independent_block_id": "block-"+role} for role in inputs.SPLIT_COUNTS]


def test_identity_metadata_accepts_real_style_strings_not_assumed_sha_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(inputs, "SPLIT_COUNTS", {k: 1 for k in inputs.SPLIT_COUNTS})
    p = tmp_path/"samples.jsonl"
    p.write_bytes(b"\n".join(core.canonical(r) for r in identity_fixture()))
    result = inputs.identity_partition_audit(p, core.file_hash(p))
    assert result["label_prediction_metric_reads"] == 0
    assert all(v == 0 for pair in result["intersections"].values() for v in pair.values())
    assert result["split_row_identity_sha256"]["final_eval"] == core.digest([
        [r[k] for k in ("sample_id", "segment_id", "independent_block_id")]
        for r in identity_fixture() if r["split"] == "final_eval"])


@pytest.mark.parametrize("mutation", ["sample_id", "segment_id", "independent_block_id", "duplicate", "missing", "extra", "wrong_hash"])
def test_identity_partition_intersection_and_denominator_fail_closed(monkeypatch, tmp_path, mutation):
    monkeypatch.setattr(inputs, "SPLIT_COUNTS", {k: 1 for k in inputs.SPLIT_COUNTS})
    rows = identity_fixture()
    if mutation in {"sample_id", "segment_id", "independent_block_id"}: rows[-1][mutation] = rows[0][mutation]
    if mutation == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    if mutation == "missing": rows.pop()
    if mutation == "extra": rows.append(copy.deepcopy(rows[-1]))
    p = tmp_path/"samples.jsonl"
    p.write_bytes(b"\n".join(core.canonical(r) for r in rows))
    with pytest.raises(ValueError):
        inputs.identity_partition_audit(p, "0"*64 if mutation == "wrong_hash" else core.file_hash(p))


def snapshot_fixture():
    spec = {"feature_spec_id": "software-fixture"}
    row = {"split": "validation", "file_id": "file", "path": "features/a.parquet", "sha256": "1"*64, "row_count": 3}
    inv = hashlib.sha256(f"{row['path']}\0{row['sha256']}\0{row['row_count']}\n".encode()).hexdigest()
    return {"feature_spec_sha256": core.digest(spec), "files": [row], "content_inventory_sha256": inv}, spec, inv


def test_snapshot_spec_uses_canonical_content_not_formatted_file_hash():
    manifest, spec, inv = snapshot_fixture()
    assert inputs.validate_snapshot_metadata(manifest, spec, expected_inventory_sha256=inv, expected_files=1) == 1
    assert core.digest(spec) != hashlib.sha256(json.dumps(spec, indent=2).encode()).hexdigest()
    manifest["feature_spec_sha256"] = hashlib.sha256(json.dumps(spec, indent=2).encode()).hexdigest()
    with pytest.raises(ValueError, match="fingerprint"):
        inputs.validate_snapshot_metadata(manifest, spec, expected_inventory_sha256=inv, expected_files=1)


@pytest.mark.parametrize("mutation", ["row_count", "path", "sha256", "split", "duplicate"])
def test_snapshot_inventory_change_is_rejected(mutation):
    manifest, spec, inv = snapshot_fixture()
    if mutation == "duplicate": manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    else: manifest["files"][0][mutation] = {"row_count": True, "path": "../outside", "sha256": "no", "split": "test"}[mutation]
    with pytest.raises(ValueError):
        inputs.validate_snapshot_metadata(manifest, spec, expected_inventory_sha256=inv, expected_files=1)


@pytest.fixture
def authorization_fixture(candidate, tmp_path, monkeypatch):
    """Simulated approval only for unit testing; private tempfile never admitted."""
    evidence = tmp_path/"synthetic-evidence.txt"
    evidence.write_text("SOFTWARE FIXTURE - NO REAL HUMAN AUTHORIZATION", encoding="utf-8")
    bindings = {"FIXTURE/synthetic-evidence.txt": core.file_hash(evidence)}
    monkeypatch.setattr(guard, "verify_sources", lambda b: proto.verify_sources(b, {"FIXTURE": tmp_path}))
    execution = execution_for(candidate)
    paths, checks = {}, {}
    for task in ("TEST-01", "REVIEW-01"):
        payload = {"schema_version": guard.CHECK_VERSION, "task": task, "protocol_sha256": candidate["sha256"],
            "execution_sha256": execution["sha256"], "result": "PASS", "checks": {k: {"result": "PASS",
                "evidence_sha256": bindings, "command_or_review": "unit-fixture", "actual_result": "synthetic-only"} for k in guard.CHECK_IDS}}
        paths[task], checks[task] = core.publish(tmp_path/task, payload)
    approval = {"schema_version": guard.APPROVAL_VERSION, "task": "ACCEPT-01", "decision": "CONFIRMED",
        "protocol_sha256": candidate["sha256"], "execution_sha256": execution["sha256"],
        "test_sha256": checks["TEST-01"]["sha256"], "review_sha256": checks["REVIEW-01"]["sha256"],
        "human_confirmation": {"kind": "user-message", "reference": "SIMULATED-UNIT-TEST",
            "quoted_decision": "SIMULATED, NOT A USER APPROVAL", "recorded_at_utc": "2026-09-30T00:00:00Z", "recorded_by": "unit-test-fixture"},
        "acknowledged_limits": list(guard.LIMITS), "evidence_sha256": bindings}
    approval_path, value = core.publish(tmp_path/"approval", approval)
    return {"access_kind": "final_eval_eligibility", "protocol": candidate, "execution": execution,
        "approval_path": approval_path, "approval_sha256": value["sha256"], "test_path": paths["TEST-01"],
        "review_path": paths["REVIEW-01"], "journal_directory": tmp_path/"journal"}


@pytest.mark.parametrize("missing", ["approval_path", "approval_sha256", "test_path", "review_path"])
def test_missing_authority_never_invokes_loader(authorization_fixture, missing):
    args = dict(authorization_fixture)
    args[missing] = None
    with pytest.raises(ValueError):
        guard.guarded_call(**args, operation=lambda _: pytest.fail("unapproved loader invoked"))
    assert not args["journal_directory"].exists()


def test_dataset_id_alone_is_not_authority(authorization_fixture):
    args = dict(authorization_fixture)
    args["approval_sha256"] = inputs.DATASET_ID
    with pytest.raises(ValueError):
        guard.guarded_call(**args, operation=lambda _: pytest.fail("legacy self-unlock"))


@pytest.mark.parametrize("mutation", ["stale", "protocol", "execution", "decision", "agent", "limits", "review", "evidence", "time"])
def test_wrong_approval_refused_before_callback(authorization_fixture, tmp_path, mutation):
    args = dict(authorization_fixture)
    record = copy.deepcopy(core.read_json(args["approval_path"])["payload"])
    if mutation in {"stale", "protocol"}: record["protocol_sha256"] = "0"*64
    if mutation == "execution": record["execution_sha256"] = "0"*64
    if mutation == "decision": record["decision"] = "REJECTED"
    if mutation == "agent": record["human_confirmation"]["kind"] = "agent-assertion"
    if mutation == "limits": record["acknowledged_limits"].pop()
    if mutation == "review": record["review_sha256"] = "0"*64
    if mutation == "evidence": record["evidence_sha256"] = {}
    if mutation == "time": record["human_confirmation"]["recorded_at_utc"] = "2026-09-30T00:00:00"
    args["approval_path"], bad = core.publish(tmp_path/"changed", record)
    if mutation != "stale": args["approval_sha256"] = bad["sha256"]
    with pytest.raises(ValueError):
        guard.guarded_call(**args, operation=lambda _: pytest.fail("unapproved callback"))
    assert not args["journal_directory"].exists()


@pytest.mark.parametrize("mutation", ["missing", "failed", "unknown", "empty_evidence", "scope", "task"])
def test_test_review_require_every_bound_check(authorization_fixture, mutation):
    args = authorization_fixture
    record = copy.deepcopy(core.read_json(args["test_path"])["payload"])
    if mutation == "missing": del record["checks"]["EC-12"]
    if mutation == "failed": record["checks"]["HC-01"]["result"] = "FAIL"
    if mutation == "unknown": record["checks"]["HC-01"]["result"] = "UNKNOWN"
    if mutation == "empty_evidence": record["checks"]["HC-01"]["evidence_sha256"] = {}
    if mutation == "scope": record["execution_sha256"] = "0"*64
    if mutation == "task": record["task"] = "TEST-02"
    with pytest.raises(ValueError):
        guard.validate_check(core.envelope(record), task="TEST-01", protocol=args["protocol"], execution=args["execution"])


@pytest.mark.parametrize("access", guard.ACCESS_KINDS[1:])
def test_approval_still_needs_population_before_any_outcomes(authorization_fixture, access):
    args = dict(authorization_fixture, access_kind=access)
    with pytest.raises(ValueError, match="population must be sealed"):
        guard.guarded_call(**args, operation=lambda _: pytest.fail("outcome callback"))


def test_success_is_durably_logged_before_synthetic_callback(authorization_fixture):
    args = authorization_fixture
    def operation(context):
        entries = [core.read_json(p) for p in args["journal_directory"].glob("*.json")]
        assert len(entries) == 1 and entries[0]["payload"]["event"] == "started"
        assert entries[0]["sha256"] == context.access_started_sha256
        assert context.legacy_cohort_ack == inputs.DATASET_ID
        assert context.population_sha256 is None
        return "fixture-only"
    assert guard.guarded_call(**args, operation=operation) == "fixture-only"
    events = [core.read_json(p)["payload"] for p in args["journal_directory"].glob("*.json")]
    assert {e["event"] for e in events} == {"started", "returned"}


def test_failed_callback_remains_possible_read_not_reset_to_zero(authorization_fixture):
    def failure(context):
        raise RuntimeError("fixture")
    with pytest.raises(RuntimeError, match="fixture"):
        guard.guarded_call(**authorization_fixture, operation=failure)
    events = [core.read_json(p)["payload"] for p in authorization_fixture["journal_directory"].glob("*.json")]
    failed = next(e for e in events if e["event"] == "failed")
    assert failed["possible_reads_retained"] is True and failed["exception_type"] == "RuntimeError"
    assert len(events) == 2


def test_journal_failure_prevents_legacy_ack_delivery(authorization_fixture, monkeypatch):
    def failure(*args, **kwargs):
        raise OSError("disk-full-fixture")
    monkeypatch.setattr(guard, "publish", failure)
    with pytest.raises(OSError, match="disk-full"):
        guard.guarded_call(**authorization_fixture, operation=lambda _: pytest.fail("unlogged access"))


def population_fixture(n=60):
    """No real IDs/data. Tiny protocol envelope only for pure population tests."""
    rows = [{"sample_id": core.digest(["unit-test", i]), "segment_id": f"segment-{i}", "independent_block_id": f"block-{i}",
        "split": "final_eval", "eligible": True, "reasons": [], "window_sha256": core.digest(["window", i])} for i in range(n)]
    identity = core.digest({k: sorted({r[k] for r in rows}) for k in ("sample_id", "segment_id", "independent_block_id")})
    row_identity = core.digest(sorted([r[k] for k in ("sample_id", "segment_id", "independent_block_id")] for r in rows))
    protocol = core.envelope({"dataset_inputs": {"dataset_id": "fixture", "sha256": "1"*64,
        "partitions": {"counts": {"final_eval": n}, "split_identity_sha256": {"final_eval": identity},
                       "split_row_identity_sha256": {"final_eval": row_identity}}},
        "eligibility_contract": {"synthetic": True}})
    execution = core.envelope({"synthetic": True})
    eligibility = core.envelope({"schema_version": guard.ELIGIBILITY_VERSION, "protocol_sha256": protocol["sha256"],
        "execution_sha256": execution["sha256"], "dataset_id": "fixture",
        "eligibility_rule_sha256": core.digest(protocol["payload"]["eligibility_contract"]), "prior_performance_reads": 0, "rows": rows})
    return protocol, execution, eligibility


@pytest.mark.parametrize("n", [0, 1, 29, 30, 46, 60])
def test_population_full_denominator_and_shortfall(n):
    protocol, execution, eligibility = population_fixture(n)
    payload = guard.population_contract(eligibility, protocol=protocol, execution=execution)
    population = core.envelope(payload)
    guard.validate_population(population, eligibility, expected_sha256=population["sha256"], protocol=protocol, execution=execution)
    assert len(payload["selection"]["selected"]) == min(n, 46)
    assert payload["selection"]["shortfall_blocks"] == max(0, 46-n)
    assert payload["selection"]["minimum_30_blocks_met"] == (n >= 30)
    assert payload["selection"]["execution_admitted"] is False


@pytest.mark.parametrize("mutation", ["omitted", "duplicate", "sample", "segment", "block", "swap_blocks", "swap_segments", "outcome", "validation", "unexplained", "window", "prior", "bool_prior", "scope"])
def test_population_cannot_drop_failed_cases_or_substitute_results(mutation):
    protocol, execution, eligibility = population_fixture()
    p = eligibility["payload"]
    row = p["rows"][0]
    if mutation == "omitted": p["rows"].pop()
    if mutation == "duplicate": p["rows"][0] = copy.deepcopy(p["rows"][1])
    if mutation == "sample": row["sample_id"] = "0"*64
    if mutation == "segment": row["segment_id"] = "another"
    if mutation == "block": row["independent_block_id"] = "another"
    if mutation in {"swap_blocks", "swap_segments"}:
        key = "independent_block_id" if mutation == "swap_blocks" else "segment_id"
        row[key], p["rows"][1][key] = p["rows"][1][key], row[key]
    if mutation == "outcome": row["score_m"] = 0
    if mutation == "validation": row["split"] = "validation"
    if mutation == "unexplained": row.update(eligible=False, window_sha256=None)
    if mutation == "window": row["window_sha256"] = None
    if mutation == "prior": p["prior_performance_reads"] = 1
    if mutation == "bool_prior": p["prior_performance_reads"] = False
    if mutation == "scope": p["execution_sha256"] = "0"*64
    with pytest.raises(ValueError):
        guard.population_contract(core.envelope(p), protocol=protocol, execution=execution)


def test_population_reasons_retained_and_selection_not_caller_chosen():
    protocol, execution, eligibility = population_fixture()
    eligibility["payload"]["rows"][0].update(eligible=False, reasons=["gap_over_60s"], window_sha256=None)
    eligibility = core.envelope(eligibility["payload"])
    payload = guard.population_contract(eligibility, protocol=protocol, execution=execution)
    assert payload["selection"]["eligible_block_count"] == 59
    payload["selection"]["selected"].pop()
    forged = core.envelope(payload)
    with pytest.raises(ValueError, match="differs"):
        guard.validate_population(forged, eligibility, expected_sha256=forged["sha256"], protocol=protocol, execution=execution)


def test_added_executor_sources_need_new_execution_binding(candidate, monkeypatch):
    execution = execution_for(candidate)
    extended = {**proto.source_catalog(), "PSDE-SDE/experiments/pirc17/new_formal_executor.py": "9"*64}
    monkeypatch.setattr(proto, "source_catalog", lambda: extended)
    # A new file does not alter an old core seal; it must enter the later exact
    # execution manifest, which in turn requires new test/review/human receipts.
    proto.validate_protocol(candidate)
    with pytest.raises(ValueError, match="execution source catalog incomplete"):
        proto.validate_execution(execution, candidate)
