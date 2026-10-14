"""Synthetic files and simulated approvals ONLY; no real evaluation access."""
import copy
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.pirc20 import _sample_id
from experiments.pirc17 import formal_eligibility as eligible
from experiments.pirc17 import final_eval_guard as guard
from experiments.pirc17 import protocol as proto
from experiments.pirc17 import protocol_core as core


def metadata_fixture(name="a", *, split="final_eval"):
    sample = dict(data_version="synthetic-only", file_id="file-"+name, segment_id="segment-"+name,
                  independent_block_id="block-"+name, split=split,
                  history_start=0, history_end=2, target_start=3, target_end=63)
    sample["sample_id"] = _sample_id(sample)
    rows = [dict(segment_point_index=i, source_point_index=42+2*i,
                 absolute_epoch_ns=1_700_000_000*eligible.NS+i*30*eligible.NS,
                 **{k: sample[k] for k in ("file_id", "segment_id", "split")},
                 **{k+"_valid": True for k in eligible.STATUS_COLUMNS}) for i in range(64)]
    return sample, rows


def check(sample, rows):
    return eligible.qualify_window(sample, rows, input_sha256="1"*64, rule_sha256="2"*64)


def test_fixed_window_original_source_indexes_and_elapsed_time():
    sample, rows = metadata_fixture()
    disposition, window = check(sample, list(reversed(rows)))
    assert disposition["eligible"] and not disposition["reasons"]
    p = core.unpack(window)
    assert p["score_elapsed_ns"] == list(eligible.SCORE_NS)
    assert [r["segment_point_index"] for r in p["targets"]] == [4, 12, 32, 62]
    assert [r["source_point_index"] for r in p["prefix"]] == [42, 44, 46]
    assert disposition["window_sha256"] == window["sha256"]
    assert "positions" not in p and len(p["prefix"]) == 3
    assert check(sample, rows) == check(sample, list(reversed(rows)))


@pytest.mark.parametrize("change,reason", [
    ("empty", "incomplete-or-duplicate-original-indexes"),
    ("missing", "incomplete-or-duplicate-original-indexes"),
    ("duplicate", "incomplete-or-duplicate-original-indexes"),
    ("file", "alignment-identity-mismatch"),
    ("role", "alignment-identity-mismatch"),
    ("source", "invalid-or-duplicate-source-indexes"),
    ("negative_source", "invalid-or-duplicate-source-indexes"),
    ("float_time", "invalid-integer-point-metadata"),
    ("overflow", "timestamp-outside-int64"),
    ("equal_time", "nonpositive-observation-gap"),
    ("gap", "observation-gap-exceeds-60s"),
    ("short", "followup-shorter-than-1800s"),
])
def test_invalid_original_windows_keep_explicit_disposition(change, reason):
    sample, rows = metadata_fixture()
    if change == "empty": rows = []
    elif change == "missing": rows.pop(8)
    elif change == "duplicate": rows.append(dict(rows[8]))
    elif change == "file": rows[1]["file_id"] = "wrong"
    elif change == "role": rows[1]["split"] = "train"
    elif change == "source": rows[1]["source_point_index"] = rows[0]["source_point_index"]
    elif change == "negative_source": rows[1]["source_point_index"] = -1
    elif change == "float_time": rows[1]["absolute_epoch_ns"] = float(rows[1]["absolute_epoch_ns"])
    elif change == "overflow": rows[-1]["absolute_epoch_ns"] = 2**63
    elif change == "equal_time": rows[1]["absolute_epoch_ns"] = rows[0]["absolute_epoch_ns"]
    elif change == "gap":
        for r in rows[4:]: r["absolute_epoch_ns"] += 31*eligible.NS
    elif change == "short":
        for i, r in enumerate(rows): r["absolute_epoch_ns"] = 1_700_000_000*eligible.NS+i*29*eligible.NS
    disposition, window = check(sample, rows)
    assert not disposition["eligible"] and window is None
    assert disposition["window_sha256"] is None and reason in disposition["reasons"]


@pytest.mark.parametrize("group", eligible.STATUS_COLUMNS)
@pytest.mark.parametrize("index,passes", [(0, True), (2, False), (63, False)])
def test_coverage_includes_origin_and_registered_end_not_only_last_score(group, index, passes):
    sample, rows = metadata_fixture()
    rows[index][group+"_valid"] = False
    disposition, _ = check(sample, rows)
    assert disposition["eligible"] is passes
    if not passes:
        assert f"invalid-{group}-coverage-origin-through-target-end" in disposition["reasons"]


def test_integer_tie_chooses_earlier_observation_at_30_second_boundary():
    sample, rows = metadata_fixture()
    # One-minute slot ties between +30 and +90. No timestamp rounding at UTC scale.
    for i, r in enumerate(rows[4:], start=4):
        r["absolute_epoch_ns"] += 30*eligible.NS
    disposition, window = check(sample, rows)
    assert disposition["eligible"]
    assert window["payload"]["score_elapsed_ns"][0] == 30*eligible.NS
    assert window["payload"]["targets"][0]["segment_point_index"] == 3
    rows[3]["absolute_epoch_ns"] -= 1
    rows[4]["absolute_epoch_ns"] -= 1  # preserve the independently required <=60s gap
    _, window = check(sample, rows)
    assert window["payload"]["score_elapsed_ns"][0] == 90*eligible.NS-1


@pytest.mark.parametrize("where,key", [("sample", "score"), ("row", "lon"), ("row", "future_error")])
def test_performance_or_position_columns_are_not_qualification_inputs(where, key):
    sample, rows = metadata_fixture()
    (sample if where == "sample" else rows[0])[key] = 123
    with pytest.raises(ValueError, match="metadata|columns"):
        check(sample, rows)


def disk_fixture(root, *, mutation=None, future_shift=0.):
    release, snapshot = root/"release", root/"snapshot"
    release.mkdir(); snapshot.mkdir()
    condition_root = root/"data"/"cond_slices"
    condition_root.mkdir(parents=True)
    records = [metadata_fixture(str(i), split=role) for i, role in enumerate(("train", "validation", "final_eval", "final_eval", "final_eval"))]
    samples = [{**s, "factor_availability": {}} for s, _ in records]
    alignment, entries, conditions = [], [], []
    for ordinal, (s, rows) in enumerate(records):
        if not ((mutation == "missing-window" and ordinal == 3) or mutation == "all-windows-missing"):
            alignment.extend({k: v for k, v in r.items() if not k.endswith("_valid")} for r in rows)
        if s["split"] != "final_eval":
            continue
        coverage = [dict(file_id=r["file_id"], point_index=r["source_point_index"],
                        segment_id=r["segment_id"], independent_block_id=s["independent_block_id"],
                        absolute_epoch_ns=r["absolute_epoch_ns"], split=r["split"],
                        **{c: "valid" for c in eligible.STATUS_COLUMNS.values()},
                        numeric_feature="MUST NOT BE DECODED") for r in rows]
        if ordinal == 4 and mutation == "invalid-end": coverage[-1]["overture_road_status"] = "missing"
        if ordinal == 2:
            if mutation == "duplicate-feature": coverage.append(dict(coverage[0]))
            if mutation == "wrong-file": coverage[0]["file_id"] = "wrong"
            if mutation == "wrong-split": coverage[0]["split"] = "train"
            if mutation == "wrong-block": coverage[-1]["independent_block_id"] = "wrong"
            if mutation == "wrong-time": coverage[-1]["absolute_epoch_ns"] += 1
        path = snapshot/(s["file_id"]+".parquet")
        pq.write_table(pa.Table.from_pylist(coverage), path)
        entries.append(dict(path=path.name, sha256=core.file_hash(path), row_count=len(coverage),
                            split=s["split"], file_id=s["file_id"]))
        coordinates = [dict(file_id=s["file_id"], lon=110.+i*.00001+(future_shift if i > 46 else 0.),
                            lat=35.+i*.000005, solar_elev=-999.) for i in range(190)]
        if ordinal == 2:
            if mutation == "condition-short": coordinates = coordinates[:47]
            if mutation == "condition-bad-file": coordinates[0]["file_id"] = "wrong"
            if mutation == "condition-bad-truth": coordinates[166]["lat"] = 90.
        cp = condition_root/(s["file_id"]+".parquet")
        times = [1_700_000_000*eligible.NS+(i-42)*15*eligible.NS for i in range(len(coordinates))]
        if ordinal == 2 and mutation == "condition-origin-time": times[46] += 1
        if ordinal == 2 and mutation == "condition-target-time": times[166] += 1
        time_column = (pa.array(["not-UTC"]*len(times)) if ordinal == 2 and mutation == "condition-nontime"
                       else pa.array(times, type=pa.timestamp("ns", tz="UTC")))
        pq.write_table(pa.Table.from_pylist(coordinates).append_column("t", time_column), cp)
        conditions.append(dict(file_id=s["file_id"], relative_path=cp.name, sha256=core.file_hash(cp)))
    # No development feature file may be opened by final qualification.
    entries.append(dict(path="never-open-development.parquet", sha256="0"*64, row_count=1,
                        split="train", file_id="development-trap"))
    def write(path, value): path.write_bytes(core.canonical(value))
    for name, rows in (("samples.jsonl", samples), ("alignment.jsonl", alignment), ("condition_file_manifest.jsonl", conditions)):
        (release/name).write_bytes(b"\n".join(core.canonical(r) for r in rows)+b"\n")
    counts = {role: sum(s["split"] == role for s in samples) for role in ("train", "validation", "final_eval")}
    write(release/"cohort.json", dict(schema_version="pirc20-cohort-v1", cohort_id="synthetic-dataset",
        data_version="synthetic-only", sample_manifest="samples.jsonl", sample_manifest_sha256=core.file_hash(release/"samples.jsonl"),
        sample_count=len(samples), sample_counts_by_split=counts, final_eval_access="sealed_identity_only",
        ordered_sample_ids_hash_encoding="utf8_sample_id_newline_in_manifest_order",
        ordered_sample_ids_sha256=hashlib.sha256("".join(s["sample_id"]+"\n" for s in samples).encode()).hexdigest(),
        window=dict(mode="fixed_points", history_points=3, target_points=61)))
    artifacts = {name: core.file_hash(release/name) for name in ("cohort.json", "samples.jsonl", "alignment.jsonl", "condition_file_manifest.jsonl")}
    write(release/"dataset.json", dict(dataset_id="synthetic-dataset", artifacts={k: {"sha256": v} for k, v in artifacts.items()}))
    spec = {"feature_spec_id": "synthetic-spec"}
    inventory = hashlib.sha256()
    for e in sorted(entries, key=lambda r: r["path"]):
        inventory.update(f"{e['path']}\0{e['sha256']}\0{e['row_count']}\n".encode())
    manifest = dict(status="valid", dataset_id="synthetic-dataset", snapshot_id="synthetic-snapshot",
                    feature_spec_id="synthetic-spec", feature_spec_sha256=core.digest(spec), files=entries,
                    content_inventory_sha256=inventory.hexdigest())
    write(snapshot/"manifest.json", manifest); write(snapshot/"feature_spec.json", spec)
    names = ("sample_id", "segment_id", "independent_block_id")
    binding = dict(dataset_id="synthetic-dataset", dataset_file_sha256=core.file_hash(release/"dataset.json"),
        release_artifact_sha256=artifacts,
        partitions=dict(counts=counts,
            split_identity_sha256={role: core.digest({k: sorted({s[k] for s in samples if s["split"] == role}) for k in names}) for role in counts},
            split_row_identity_sha256={role: core.digest(sorted([s[k] for k in names] for s in samples if s["split"] == role)) for role in counts}),
        snapshot=dict(snapshot_id="synthetic-snapshot", feature_spec_id="synthetic-spec", manifest_sha256=core.file_hash(snapshot/"manifest.json"),
            feature_spec_file_sha256=core.file_hash(snapshot/"feature_spec.json"), feature_spec_content_sha256=core.digest(spec),
            content_inventory_sha256=inventory.hexdigest(), files=len(entries)))
    binding["sha256"] = core.digest(binding)
    return binding, release, snapshot


def authorize_fixture(root, binding, monkeypatch):
    # Replace only public input metadata in the scientific-protocol builder.
    # Protocol/execution/check/approval validation and journaling run unmocked.
    monkeypatch.setattr(proto, "input_binding", lambda: copy.deepcopy(binding))
    protocol = proto.build_protocol()
    p = core.unpack(protocol)
    execution = core.envelope(dict(schema_version=proto.EXECUTION_VERSION, protocol_sha256=protocol["sha256"],
        source_sha256=proto.source_catalog(), matrix_sha256=core.digest("SYNTHETIC MATRIX"),
        runtime_manifest_sha256=core.digest("SYNTHETIC RUNTIME"), budget_ledger_schema_version="pirc17-cumulative-phase-budget-v1",
        entrypoint="PSDE-SDE/experiments/pirc17/formal_eligibility.py", phase_caps_seconds=p["resource_contract"]["phase_caps_seconds"],
        max_generated_forecasts=11513, scientific_forecasts=11020,
        method_required_slots=sorted(k for k, v in p["components"]["method_mechanisms"]["slots"].items() if v["disposition"] == "REQUIRED"),
        terrain_configurations=sorted(p["components"]["terrain_configurations"]), final_eval_authorized=False))
    evidence = root/"fixture-evidence.txt"
    evidence.write_text("SIMULATED SOFTWARE TEST; NOT HUMAN AUTHORIZATION", encoding="utf-8")
    bindings = {"FIXTURE/fixture-evidence.txt": core.file_hash(evidence)}
    monkeypatch.setattr(guard, "verify_sources", lambda b: proto.verify_sources(b, {"FIXTURE": root}))
    paths, checks = {}, {}
    for task in ("TEST-01", "REVIEW-01"):
        paths[task], checks[task] = core.publish(root/task, dict(schema_version=guard.CHECK_VERSION, task=task,
            protocol_sha256=protocol["sha256"], execution_sha256=execution["sha256"], result="PASS",
            checks={k: dict(result="PASS", evidence_sha256=bindings, command_or_review="synthetic-fixture",
                            actual_result="NO REAL DATA OR APPROVAL") for k in guard.CHECK_IDS}))
    path, approval = core.publish(root/"approval", dict(schema_version=guard.APPROVAL_VERSION,
        task="ACCEPT-01", decision="CONFIRMED", protocol_sha256=protocol["sha256"], execution_sha256=execution["sha256"],
        test_sha256=checks["TEST-01"]["sha256"], review_sha256=checks["REVIEW-01"]["sha256"],
        human_confirmation=dict(kind="user-message", reference="SIMULATED UNIT TEST", quoted_decision="NOT AN ACTUAL APPROVAL",
                                recorded_at_utc="2026-09-30T00:00:00Z", recorded_by="software-test"),
        acknowledged_limits=list(guard.LIMITS), evidence_sha256=bindings))
    return dict(protocol=protocol, execution=execution, approval_path=path, approval_sha256=approval["sha256"],
                test_path=paths["TEST-01"], review_path=paths["REVIEW-01"], journal_directory=root/"journal")


@pytest.mark.parametrize("mutation,count", [(None, 3), ("missing-window", 2), ("invalid-end", 2),
    ("wrong-block", 2), ("wrong-time", 2), ("all-windows-missing", 0)])
def test_real_guard_raw_metadata_consumer_and_full_population_seal(tmp_path, monkeypatch, mutation, count):
    binding, release, snapshot = disk_fixture(tmp_path, mutation=mutation)
    authority = authorize_fixture(tmp_path, binding, monkeypatch)
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: pytest.fail("qualification must not decode numeric feature/position tables"))
    result = eligible.qualify_final_inputs(**authority, release=release, snapshot=snapshot, output_directory=tmp_path/"output")
    summary = core.unpack(core.read_json(result["result_path"]), expected_sha256=result["result_sha256"])
    report = core.read_json(tmp_path/"output"/summary["eligibility_path"])
    population = core.read_json(tmp_path/"output"/summary["population_path"])
    guard.validate_population(population, report, expected_sha256=result["population_sha256"],
                              protocol=authority["protocol"], execution=authority["execution"])
    assert result["denominator_samples"] == 3 and result["eligible_samples"] == count
    assert len(report["payload"]["rows"]) == 3 and len(population["payload"]["selection"]["selected"]) == count
    assert population["payload"]["selection"]["shortfall_blocks"] == 46-count
    assert summary["position_feature_value_prediction_metric_reads"] == 0
    assert not population["payload"]["selection"]["execution_admitted"]
    for row in report["payload"]["rows"]:
        if row["eligible"]:
            window = core.read_json(tmp_path/"output"/summary["windows"][row["sample_id"]])
            assert window["sha256"] == row["window_sha256"]
    events = [core.unpack(core.read_json(f)) for f in (tmp_path/"journal").glob("*.json")]
    assert sorted(e["event"] for e in events) == ["returned", "started"]


@pytest.mark.parametrize("mutation", ["duplicate-feature", "wrong-file", "wrong-split"])
def test_feature_inventory_corruption_cannot_publish_population(tmp_path, monkeypatch, mutation):
    binding, release, snapshot = disk_fixture(tmp_path, mutation=mutation)
    authority = authorize_fixture(tmp_path, binding, monkeypatch)
    with pytest.raises(ValueError, match="inventory identity|duplicate point"):
        eligible.qualify_final_inputs(**authority, release=release, snapshot=snapshot, output_directory=tmp_path/"output")
    assert not (tmp_path/"output"/"population").exists()
    events = [core.unpack(core.read_json(f)) for f in (tmp_path/"journal").glob("*.json")]
    assert any(e["event"] == "failed" and e["possible_reads_retained"] for e in events)


@pytest.mark.parametrize("missing", ["approval_path", "approval_sha256", "test_path", "review_path"])
def test_actual_entrypoint_rejects_missing_approval_before_source_or_output_access(tmp_path, monkeypatch, missing):
    binding, release, snapshot = disk_fixture(tmp_path)
    authority = authorize_fixture(tmp_path, binding, monkeypatch)
    authority[missing] = None
    monkeypatch.setattr(eligible, "_release_sources", lambda *a: pytest.fail("unapproved raw loader invoked"))
    with pytest.raises(ValueError):
        eligible.qualify_final_inputs(**authority, release=release, snapshot=snapshot, output_directory=tmp_path/"output")
    assert not (tmp_path/"output").exists() and not (tmp_path/"journal").exists()


@pytest.mark.parametrize("which", ["approval", "execution", "review", "evidence"])
def test_actual_entrypoint_rejects_stale_authority_before_source_access(tmp_path, monkeypatch, which):
    binding, release, snapshot = disk_fixture(tmp_path)
    authority = authorize_fixture(tmp_path, binding, monkeypatch)
    if which == "approval": authority["approval_sha256"] = "0"*64
    elif which == "execution":
        e = dict(core.unpack(authority["execution"]))
        e["matrix_sha256"] = "0"*64
        authority["execution"] = core.envelope(e)
    elif which == "review": authority["review_path"] = authority["test_path"]
    elif which == "evidence": (tmp_path/"fixture-evidence.txt").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(eligible, "_release_sources", lambda *a: pytest.fail("unapproved raw loader invoked"))
    with pytest.raises(ValueError):
        eligible.qualify_final_inputs(**authority, release=release, snapshot=snapshot, output_directory=tmp_path/"output")
    assert not (tmp_path/"output").exists() and not (tmp_path/"journal").exists()


@pytest.mark.parametrize("which", ["alignment.jsonl", "samples.jsonl", "cohort.json", "feature"])
def test_used_input_hash_mismatch_is_terminal_not_requalification(tmp_path, monkeypatch, which):
    binding, release, snapshot = disk_fixture(tmp_path)
    authority = authorize_fixture(tmp_path, binding, monkeypatch)
    path = next(snapshot.glob("*.parquet")) if which == "feature" else release/which
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        eligible.qualify_final_inputs(**authority, release=release, snapshot=snapshot, output_directory=tmp_path/"output")
    assert not (tmp_path/"output"/"population").exists()
