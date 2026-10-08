"""Source-bound final population qualification, behind the exact ACCEPT01 guard.

Only original identities, timestamps and feature-validity columns are decoded.
No positions, feature values, models or performance measures enter selection.
The formal controller must charge this whole operation to its input work item;
this module is a guarded domain consumer, not a standalone run authorization.
"""
from __future__ import annotations

from itertools import groupby
from pathlib import Path

from data.pirc20 import load_pirc20_cohort
from . import final_eval_guard as guard
from .protocol_core import digest, envelope, file_hash, publish, read_json, under, unpack
from .protocol_inputs import validate_snapshot_metadata

WINDOW_VERSION = "pirc17-final-observed-window-v1"
RESULT_VERSION = "pirc17-final-input-qualification-v1"
NS = 1_000_000_000
SCORE_NS = (60*NS, 300*NS, 900*NS, 1800*NS)
STATUS_COLUMNS = {"surface": "dem_surface_status", "road": "overture_road_status",
                  "river": "hydrorivers_river_status", "worldcover": "worldcover_status",
                  "history": "historical_motion_status"}
SAMPLE_FIELDS = ("sample_id", "data_version", "file_id", "segment_id", "independent_block_id",
                 "split", "history_start", "history_end", "target_start", "target_end")
POINT_FIELDS = ("segment_point_index", "source_point_index", "absolute_epoch_ns")


def sample_record(sample):
    return {k: getattr(sample, k) for k in SAMPLE_FIELDS}


def _bound_file(root, relative, expected):
    path = under(root, relative)
    if not path.is_file() or file_hash(path) != expected:
        raise ValueError("bound input file missing or changed: "+relative)
    return path


def _release_sources(release, binding):
    """Private consumer: call only within an authorized, measured operation."""
    dataset = read_json(under(release, "dataset.json"),
                        expected_file_sha256=binding["dataset_file_sha256"])
    if (dataset["dataset_id"] != binding["dataset_id"] or
            {k: v["sha256"] for k, v in dataset["artifacts"].items()} != binding["release_artifact_sha256"]):
        raise ValueError("release differs from sealed input chain")
    hashes = binding["release_artifact_sha256"]
    cohort_path = _bound_file(release, "cohort.json", hashes["cohort.json"])
    _bound_file(release, "samples.jsonl", hashes["samples.jsonl"])
    cohort = load_pirc20_cohort(cohort_path)
    if (cohort.cohort_id != binding["dataset_id"] or
            dict(cohort.sample_counts_by_split) != binding["partitions"]["counts"]):
        raise ValueError("cohort count or identity differs from sealed release")
    samples = list(cohort.iter_samples())
    names = ("sample_id", "segment_id", "independent_block_id")
    for split, count in binding["partitions"]["counts"].items():
        group = [s for s in samples if s.split == split]
        identities = {k: sorted({getattr(s, k) for s in group}) for k in names}
        rows = sorted([getattr(s, k) for k in names] for s in group)
        if (len(group) != count or digest(identities) != binding["partitions"]["split_identity_sha256"][split]
                or digest(rows) != binding["partitions"]["split_row_identity_sha256"][split]):
            raise ValueError("original partition identities changed")
    return cohort, [sample_record(s) for s in samples if s.split == "final_eval"]


def _snapshot_sources(snapshot, binding):
    """Verify metadata chain and used file bytes BEFORE any Parquet decode."""
    import pyarrow.parquet as pq

    frozen = binding["snapshot"]
    manifest = read_json(under(snapshot, "manifest.json"), expected_file_sha256=frozen["manifest_sha256"])
    spec = read_json(under(snapshot, "feature_spec.json"), expected_file_sha256=frozen["feature_spec_file_sha256"])
    if (manifest["dataset_id"] != binding["dataset_id"] or manifest["snapshot_id"] != frozen["snapshot_id"]
            or manifest["feature_spec_id"] != frozen["feature_spec_id"] or spec["feature_spec_id"] != frozen["feature_spec_id"]
            or digest(spec) != frozen["feature_spec_content_sha256"] or manifest["status"] != "valid"):
        raise ValueError("snapshot identity/specification changed")
    validate_snapshot_metadata(manifest, spec, expected_inventory_sha256=frozen["content_inventory_sha256"],
                               expected_files=frozen["files"])
    entries = []
    for entry in manifest["files"]:
        if entry["split"] != "final_eval":
            continue
        path = _bound_file(snapshot, entry["path"], entry["sha256"])
        if pq.ParquetFile(path).metadata.num_rows != entry["row_count"]:
            raise ValueError("bound feature row count changed")
        entries.append({**entry, "absolute_path": path.as_posix()})
    # An empty final inventory is not a valid all-ineligible outcome.
    if not entries:
        raise ValueError("snapshot lacks final-eval inventory")
    return entries


def qualify_window(sample, rows, *, input_sha256, rule_sha256):
    """Pure fixed-rule qualification of one complete denominator member.

    Rows contain alignment metadata and already joined boolean validity only.
    No caller-provided eligible flag, score or position is accepted or needed.
    Integer-nanosecond comparisons preserve exact tolerance and tie boundaries.
    """
    if set(sample) != set(SAMPLE_FIELDS) or sample["split"] != "final_eval":
        raise ValueError("original final sample metadata required")
    bounds = [sample[k] for k in ("history_start", "history_end", "target_start", "target_end")]
    if (any(type(x) is not int for x in bounds) or not 0 <= bounds[0] <= bounds[1] < bounds[2] <= bounds[3]
            or bounds[2] != bounds[1]+1):
        raise ValueError("invalid registered original bounds")
    disposition = {k: sample[k] for k in ("sample_id", "segment_id", "independent_block_id", "split")}
    reasons = []
    expected = set(POINT_FIELDS) | {"file_id", "segment_id", "split"} | {k+"_valid" for k in STATUS_COLUMNS}
    if any(set(r) != expected for r in rows):
        raise ValueError("alignment/validity metadata only; exact columns required")
    if any(type(r[k]) is not int for r in rows for k in POINT_FIELDS):
        reasons.append("invalid-integer-point-metadata")
    if not reasons:
        rows = sorted(rows, key=lambda r: r["segment_point_index"])
        if [r["segment_point_index"] for r in rows] != list(range(bounds[0], bounds[3]+1)):
            reasons.append("incomplete-or-duplicate-original-indexes")
        if any(any(r[k] != sample[k] for k in ("file_id", "segment_id", "split")) for r in rows):
            reasons.append("alignment-identity-mismatch")
        source_indexes = [r["source_point_index"] for r in rows]
        if any(x < 0 for x in source_indexes) or len(source_indexes) != len(set(source_indexes)):
            reasons.append("invalid-or-duplicate-source-indexes")
        if any(not -(2**63) <= r["absolute_epoch_ns"] < 2**63 for r in rows):
            reasons.append("timestamp-outside-int64")
    if not reasons:
        n_prefix = bounds[1]-bounds[0]+1
        prefix, future = rows[:n_prefix], rows[n_prefix:]
        if n_prefix < 2:
            reasons.append("insufficient-visible-prefix")
        epoch = prefix[-1]["absolute_epoch_ns"]
        gaps = [b["absolute_epoch_ns"]-a["absolute_epoch_ns"] for a, b in zip(rows, rows[1:])]
        if any(g <= 0 for g in gaps):
            reasons.append("nonpositive-observation-gap")
        if any(g > 60*NS for g in gaps):
            reasons.append("observation-gap-exceeds-60s")
        if future[-1]["absolute_epoch_ns"]-epoch < 1800*NS:
            reasons.append("followup-shorter-than-1800s")
        for group in STATUS_COLUMNS:
            if any(r[group+"_valid"] is not True for r in rows[n_prefix-1:]):
                reasons.append("invalid-"+group+"-coverage-origin-through-target-end")
        chosen = [min(future, key=lambda r: (abs(r["absolute_epoch_ns"]-epoch-t), r["absolute_epoch_ns"]))
                  for t in SCORE_NS]
        if (any(abs(r["absolute_epoch_ns"]-epoch-t) > 30*NS for r, t in zip(chosen, SCORE_NS))
                or len({r["segment_point_index"] for r in chosen}) != len(SCORE_NS)):
            reasons.append("missing-distinct-original-score-targets")
    window = None
    if not reasons:
        point = lambda r: {k: r[k] for k in POINT_FIELDS}
        window = envelope({"schema_version": WINDOW_VERSION, "input_binding_sha256": input_sha256,
            "eligibility_rule_sha256": rule_sha256, "sample": dict(sample),
            "alignment_validity_sha256": digest(rows), "prefix": [point(r) for r in prefix],
            "targets": [point(r) for r in chosen], "origin_epoch_ns": epoch,
            "score_elapsed_ns": [r["absolute_epoch_ns"]-epoch for r in chosen]})
    return {**disposition, "eligible": not reasons, "reasons": reasons,
            "window_sha256": None if window is None else window["sha256"]}, window


def _joined_metadata(release, entries, samples, binding):
    """Stream sample groups; never materialize every final point in Python."""
    import duckdb
    import pyarrow as pa

    alignment = _bound_file(release, "alignment.jsonl", binding["release_artifact_sha256"]["alignment.jsonl"])
    db = duckdb.connect()
    try:
        db.execute("SET threads=4")
        db.execute("SET memory_limit='2GB'")
        db.register("samples", pa.Table.from_pylist(samples))
        db.register("inventory", pa.Table.from_pylist([
            {k: e[k] for k in ("absolute_path", "file_id", "split", "row_count")} for e in entries]))
        statuses = ", ".join(f"{column}='valid' AS {group}_valid" for group, column in STATUS_COLUMNS.items())
        db.execute(f"""CREATE TEMP TABLE coverage AS SELECT filename, file_id, point_index,
            segment_id, independent_block_id, absolute_epoch_ns, split, {statuses}
            FROM read_parquet(?, filename=true)""", [[e["absolute_path"] for e in entries]])
        bad = db.execute("""SELECT count(*) FROM coverage c LEFT JOIN inventory i ON c.filename=i.absolute_path
            WHERE i.absolute_path IS NULL OR c.file_id IS DISTINCT FROM i.file_id OR c.split IS DISTINCT FROM i.split
               OR c.point_index IS NULL OR c.absolute_epoch_ns IS NULL OR c.segment_id IS NULL
               OR c.independent_block_id IS NULL""").fetchone()[0]
        duplicate = db.execute("""SELECT count(*) FROM (SELECT 1 FROM coverage
            GROUP BY file_id, point_index, absolute_epoch_ns, split HAVING count(*)>1)""").fetchone()[0]
        if bad or duplicate:
            raise ValueError("feature inventory identity mismatch or duplicate point identity")
        flags = ", ".join(f"coalesce(c.{k}_valid AND c.segment_id=a.segment_id "
                           f"AND c.independent_block_id=s.independent_block_id, false) AS {k}_valid" for k in STATUS_COLUMNS)
        query = f"""SELECT s.sample_id, a.segment_point_index, a.source_point_index, a.absolute_epoch_ns,
            a.file_id, a.segment_id, a.split, {flags}
            FROM read_json(?, columns={{segment_id:'VARCHAR', split:'VARCHAR', segment_point_index:'BIGINT',
                absolute_epoch_ns:'BIGINT', file_id:'VARCHAR', source_point_index:'BIGINT'}}) a
            JOIN samples s ON a.segment_id=s.segment_id AND a.split=s.split
            LEFT JOIN coverage c ON a.file_id=c.file_id AND a.source_point_index=c.point_index
                AND a.absolute_epoch_ns=c.absolute_epoch_ns AND a.split=c.split
            WHERE a.segment_point_index BETWEEN s.history_start AND s.target_end
            ORDER BY s.sample_id, a.segment_point_index"""
        reader = db.execute(query, [str(alignment)]).to_arrow_reader(batch_size=8192)
        stream = (row for batch in reader for row in batch.to_pylist())
        for sample_id, group in groupby(stream, key=lambda r: r["sample_id"]):
            yield sample_id, [{k: v for k, v in r.items() if k != "sample_id"} for r in group]
    finally:
        db.close()


def _qualify(access, *, protocol, execution, release, snapshot, output_directory):
    sealed = unpack(protocol)
    binding = sealed["dataset_inputs"]
    if access.access_kind != "final_eval_eligibility" or access.legacy_cohort_ack != binding["dataset_id"]:
        raise ValueError("verified final eligibility access required")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    _, samples = _release_sources(release, binding)
    entries = _snapshot_sources(snapshot, binding)
    by_id = {s["sample_id"]: s for s in samples}
    rule_sha = digest(sealed["eligibility_contract"])
    rows, seen, windows = [], set(), {}
    for sample_id, metadata in _joined_metadata(release, entries, samples, binding):
        if sample_id in seen or sample_id not in by_id:
            raise ValueError("unexpected or repeated eligibility sample group")
        seen.add(sample_id)
        row, window = qualify_window(by_id[sample_id], metadata, input_sha256=binding["sha256"], rule_sha256=rule_sha)
        rows.append(row)
        if window is not None:
            path, _ = publish(output/"windows", unpack(window))
            windows[sample_id] = path.relative_to(output).as_posix()
    # Missing whole windows remain in the denominator, never disappear in joins.
    for sample_id in sorted(by_id.keys()-seen):
        row, _ = qualify_window(by_id[sample_id], [], input_sha256=binding["sha256"], rule_sha256=rule_sha)
        rows.append(row)
    eligibility_path, eligibility = publish(output/"eligibility", {
        "schema_version": guard.ELIGIBILITY_VERSION, "protocol_sha256": protocol["sha256"],
        "execution_sha256": execution["sha256"], "dataset_id": binding["dataset_id"],
        "eligibility_rule_sha256": rule_sha, "prior_performance_reads": 0,
        "rows": sorted(rows, key=lambda r: r["sample_id"])})
    population_path, population = publish(output/"population",
        guard.population_contract(eligibility, protocol=protocol, execution=execution))
    result_path, result = publish(output, {"schema_version": RESULT_VERSION,
        "protocol_sha256": protocol["sha256"], "execution_sha256": execution["sha256"],
        "approval_sha256": access.approval_sha256, "access_started_sha256": access.access_started_sha256,
        "eligibility_path": eligibility_path.relative_to(output).as_posix(), "eligibility_sha256": eligibility["sha256"],
        "population_path": population_path.relative_to(output).as_posix(), "population_sha256": population["sha256"],
        "windows": windows, "denominator_samples": len(rows), "eligible_samples": len(windows),
        "position_feature_value_prediction_metric_reads": 0})
    return {"result_path": str(result_path), "result_sha256": result["sha256"],
            "population_sha256": population["sha256"], "denominator_samples": len(rows), "eligible_samples": len(windows)}


def qualify_final_inputs(*, protocol, execution, approval_path, approval_sha256, test_path,
                         review_path, journal_directory, release, snapshot, output_directory):
    """The only public raw eligibility entry; no raw path is opened before guard."""
    return guard.guarded_call(access_kind="final_eval_eligibility", protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, operation=lambda access: _qualify(access, protocol=protocol,
            execution=execution, release=release, snapshot=snapshot, output_directory=output_directory))
