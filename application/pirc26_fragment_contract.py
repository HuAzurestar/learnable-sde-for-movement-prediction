"""Structural source-point dispositions, never a grant or numerical decode.

Only an explicitly frozen policy may exclude short refined paths. All raw
feature points, including duplicates and excluded paths, remain auditable.
The owned converter checks their joins/context/geometry before publication.
"""

from infrastructure.research_store import ResearchError, digest

SHORT_POLICY = "exclude-short-causal-fragments-v1"
POLICIES = {"reject", SHORT_POLICY}
VERSION = "pirc26-causal-fragment-disposition-v1"
MAX_ROWS = 16_912


def require(ok):
    if not ok:
        raise ResearchError("CORRUPT_ARTIFACT", "causal fragment disposition differs")


def fragment_summary(block, *, expected_policy=None):
    """Check a bounded closed receipt against geometry and original membership.

    This does not re-open/decode source data or certify numerical geometry.
    Actual source hashes/row count and trusted worker ownership are separately
    bound by the original preparation request, read receipts and supervisor.
    """
    require(type(block) is dict and {"provenance", "document"} <= set(block))
    p, document = block["provenance"], block["document"]
    require(type(p) is dict and type(document) is dict)
    policy = p.get("fragment_policy", "reject")
    require(type(policy) is str and policy in POLICIES and (expected_policy is None or policy == expected_policy))
    if policy == "reject":
        require("fragment_disposition" not in p)
        return None
    require(type(p.get("feature")) is dict and type(p.get("membership")) is list
        and type(p["feature"].get("independent_block_id")) is str
        and 0 < len(p["feature"]["independent_block_id"]) <= 128
        and type(document.get("segments")) is list and 1 <= len(document["segments"]) <= 256
        and all(type(segment) is dict and {"segment_id", "time"} <= set(segment) for segment in document["segments"]))
    receipt = p.get("fragment_disposition")
    require(type(receipt) is dict and set(receipt) == {"schema_version", "policy", "source_rows", "segments"}
        and receipt["schema_version"] == VERSION and receipt["policy"] == policy)
    rows = p["feature"].get("aligned_row_count")
    require(type(rows) is int and 3 <= rows <= MAX_ROWS and type(receipt["source_rows"]) is int
        and receipt["source_rows"] == rows)
    records = receipt["segments"]
    require(type(records) is list and 1 <= len(records) <= 256)
    require(type(p.get("duplicate_policy")) is str and p["duplicate_policy"] in {"reject", "keep-first-exact-time-v1"})
    expected_members, expected_times, seen_segments, seen_points = [], [], set(), set()
    previous_index, source_rows, duplicates, excluded_points, excluded_segments = -1, 0, 0, 0, 0
    for record in records:
        require(type(record) is dict and set(record) == {"segment_id", "disposition", "reason_code", "points"})
        segment_id, points = record["segment_id"], record["points"]
        require(type(segment_id) is str and 0 < len(segment_id) <= 128 and segment_id not in seen_segments)
        seen_segments.add(segment_id)
        require(type(points) is list and 0 < len(points) <= rows)
        source_rows += len(points)
        require(source_rows <= rows)
        kept, previous_epoch = [], None
        for point in points:
            require(type(point) is dict and set(point) == {"point_id", "source_point_index", "absolute_epoch_ns"})
            point_id, index, epoch = point["point_id"], point["source_point_index"], point["absolute_epoch_ns"]
            require(type(point_id) is str and 0 < len(point_id) <= 128 and point_id not in seen_points
                and type(index) is int and previous_index < index < 1_048_576
                and type(epoch) is int and -(2 ** 63) <= epoch < 2 ** 63)
            seen_points.add(point_id)
            previous_index = index
            if previous_epoch is not None:
                require(0 <= epoch - previous_epoch <= 60_000_000_000)
                if epoch == previous_epoch:
                    require(p["duplicate_policy"] == "keep-first-exact-time-v1")
                    duplicates += 1
                    continue
            previous_epoch = epoch
            kept.append(point)
        require(len(kept) <= 4098)  # Never clip or split an oversized path.
        if len(kept) < 3:
            require(record["disposition"] == "excluded-short-causal-path"
                and record["reason_code"] == "INSUFFICIENT_CAUSAL_POINTS")
            excluded_segments += 1
            excluded_points += len(kept)
            continue
        require(record["disposition"] == "included-causal-path" and record["reason_code"] is None)
        first = kept[0]["absolute_epoch_ns"]
        expected_times.append([(point["absolute_epoch_ns"] - first) / 1_000_000_000 for point in kept])
        expected_members.append({"segment_id": segment_id, "independent_block_id": p["feature"]["independent_block_id"],
            "point_ids": [point["point_id"] for point in kept],
            "source_point_indices": [point["source_point_index"] for point in kept], "absolute_start_epoch_ns": first})
    require(source_rows == rows and bool(expected_members)  # No whole-file deletion.
        and p["membership"] == expected_members
        and type(p.get("removed_duplicate_timestamps")) is int and p["removed_duplicate_timestamps"] == duplicates
        and len(document["segments"]) == len(expected_members)
        and all(segment["segment_id"] == member["segment_id"] and segment["time"] == times
            for segment, member, times in zip(document["segments"], expected_members, expected_times)))
    return {"policy": policy, "receipt_hash": digest(receipt), "source_rows": rows,
        "included_points": rows - duplicates - excluded_points, "duplicate_points": duplicates,
        "excluded_points": excluded_points, "included_segments": len(expected_members),
        "excluded_segments": excluded_segments}


def population_fragments(members):
    """Train/selection-only bounded counts and hashes, never new sample IDs."""
    require(type(members) is list and 1 <= len(members) <= 16 and all(type(member) is dict for member in members))
    summaries = [member.get("fragment_disposition") for member in members]
    if not any(summary is not None for summary in summaries):
        return None
    require(all(type(summary) is dict and set(summary) == {"policy", "receipt_hash", "source_rows",
        "included_points", "duplicate_points", "excluded_points", "included_segments", "excluded_segments"}
        and summary["policy"] == SHORT_POLICY for summary in summaries))
    for summary in summaries:
        require(type(summary["receipt_hash"]) is str and len(summary["receipt_hash"]) == 64
            and all(c in "0123456789abcdef" for c in summary["receipt_hash"])
            and all(type(summary[key]) is int and summary[key] >= 0 for key in
                ("source_rows", "included_points", "duplicate_points", "excluded_points", "included_segments", "excluded_segments"))
            and 3 <= summary["source_rows"] <= MAX_ROWS
            and summary["source_rows"] == summary["included_points"] + summary["duplicate_points"] + summary["excluded_points"]
            and 1 <= summary["included_segments"] <= 256
            and summary["included_segments"] + summary["excluded_segments"] <= 256
            and 3 * summary["included_segments"] <= summary["included_points"] <= 4098 * summary["included_segments"]
            and summary["excluded_segments"] <= summary["excluded_points"] <= 2 * summary["excluded_segments"])
    return {"policy": SHORT_POLICY, "receipt_hashes": [summary["receipt_hash"] for summary in summaries],
        **{key: sum(summary[key] for summary in summaries) for key in
            ("source_rows", "included_points", "duplicate_points", "excluded_points", "included_segments", "excluded_segments")}}


def validate_population_fragments(metadata, sources, policy):
    """Bind role-only summary to every original source's frozen row count."""
    members = metadata["members"]
    require(type(members) is list and len(members) == len(sources))
    summary = population_fragments(members)
    require(metadata.get("fragment_eligibility") == summary and (summary is None) == (policy == "reject"))
    if summary is not None:
        require(policy == SHORT_POLICY and all(member["fragment_disposition"]["source_rows"]
            == source["feature"].get("aligned_row_count") for member, source in zip(members, sources)))
