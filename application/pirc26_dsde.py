"""Bounded DSDE source conversion, separate from permission and qualification.

The owner reads explicit same-split files through the original exposure ledger.
The numerical converter consumes those admitted bytes only; it never scans a
snapshot, opens a trajectory, grants access, dispatches work or resets budgets.
Production preprocessing must itself run under the shared supervisor/budget.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchError, digest, encode
from application.pirc26_fragment_contract import SHORT_POLICY, POLICIES as FRAGMENT_POLICIES


VERSION = "pirc26-dsde-observed-conversion-v1"
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 1_048_576
MAX_DECODED_BYTES = 128 * 1024 * 1024
ROLES = {"train": ("train", "fit"), "selection": ("validation", "select")}


def _require(ok, code, message):
    if not ok:
        raise ResearchError(code, message)


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class SourcePair:
    """Hash-bound owner transport, not a permission or scientific verdict."""

    protocol_hash: str
    authorization_hash: str
    feature_metadata_utf8: str
    condition_metadata_utf8: str
    feature_bytes: bytes
    condition_bytes: bytes

    def metadata(self):
        try:
            def unique(pairs):
                value = {}
                for key, item in pairs:
                    _require(key not in value, "CONTRACT_MISMATCH", "duplicate source metadata key")
                    value[key] = item
                return value
            values = tuple(json.loads(v, object_pairs_hook=unique) for v in
                           (self.feature_metadata_utf8, self.condition_metadata_utf8))
            _require(all(type(v) is dict for v in values), "CONTRACT_MISMATCH", "source metadata objects required")
            return values
        except (ValueError, TypeError) as exc:
            raise ResearchError("CONTRACT_MISMATCH", "invalid source metadata") from exc


def _pair_contract(feature, condition, purpose):
    _require(type(feature) is dict and type(condition) is dict,
             "CONTRACT_MISMATCH", "source metadata objects required")
    _require(feature.get("source_kind") == "dsde-feature-parquet"
             and condition.get("source_kind") == "dsde-condition-parquet",
             "CONTRACT_MISMATCH", "explicit DSDE feature/condition kinds required")
    role = feature.get("split_role")
    _require(role in ROLES and condition.get("split_role") == role and purpose == ROLES[role][1],
             "UNAUTHORIZED_DATA", "DSDE preparation is restricted to train/fit or selection/select")
    for key in ("file_id", "independent_block_id", "dataset_id", "source_split"):
        _require(type(feature.get(key)) is str and 0 < len(feature[key]) <= 128
                 and condition.get(key) == feature[key],
                 "CONTRACT_MISMATCH", "DSDE source file/unit/dataset/split bindings differ")
    _require(feature["source_split"] == ROLES[role][0], "UNAUTHORIZED_DATA", "DSDE split mapping differs")
    _require(_hash(feature.get("feature_spec_sha256")), "CONTRACT_MISMATCH", "frozen feature spec required")
    for block in (feature, condition):
        _require(all(type(block.get(key)) is str and 0 < len(block[key]) <= 128
                     for key in ("block_id", "source_block_id", "release_id"))
                 and type(block.get("path")) is str and 0 < len(block["path"]) <= 4096,
                 "CONTRACT_MISMATCH", "explicit source identity/path required")
        _require(block.get("fit_scope") is (role == "train"),
                 "CONTRACT_MISMATCH", "DSDE fitting scope differs")
        _require(type(block.get("size_bytes")) is int and 0 < block["size_bytes"] <= MAX_BYTES,
                 "RESOURCE_PLAN_REJECTED", "bounded frozen DSDE file size required")
        _require(_hash(block.get("sha256")), "CONTRACT_MISMATCH", "frozen DSDE file digest required")
    return role


def read_source_pair(store, protocol_id, feature_block_id, condition_block_id, *,
                     authorization_id, purpose, authorization_version=None, consumer=None):
    """Owner-only explicit reads; shared ledger records starts before hashing.

    This is input preparation, not permission to execute/disclose, or a claim
    that a source's license has been reviewed. No test/final-eval path exists.
    """
    protocol = store.manifest("protocol-" + protocol_id)
    selected = []
    for block_id in (feature_block_id, condition_block_id):
        matches = [b for b in protocol["blocks"] if b["block_id"] == block_id]
        _require(len(matches) == 1, "UNAUTHORIZED_DATA", "DSDE source absent from frozen protocol")
        selected.append(matches[0])
    feature, condition = selected
    _pair_contract(feature, condition, purpose)
    grant = store.authorization(authorization_id, version=authorization_version)
    _require(type(grant.get("data_root")) is str and Path(grant["data_root"]).is_absolute(),
             "UNAUTHORIZED_DATA", "explicit authorized DSDE root required")
    ledger = EvaluationExposureLedger(store)
    # Check both scopes before any scientific byte is opened. The public read
    # rechecks live authority/expiry and immutable identity at the boundary.
    for block in selected:
        allowed, _, _ = ledger._read_authority(protocol, block, purpose, authorization_id,
                                              authorization_version=authorization_version)
        if not allowed:
            store.append("EXPOSURE_DENIED", {"study_id": protocol["study_id"],
                "protocol_hash": digest(protocol), "block_id": block["block_id"],
                "purpose": purpose, "authorization_id": authorization_id,
                "authorization_version": authorization_version, "authorization_hash": digest(grant),
                "allowed": False, "boundary": VERSION})
        _require(allowed, "UNAUTHORIZED_DATA", "DSDE pair authority differs or expired")
    root = Path(grant["data_root"])
    for block in selected:
        relative = Path(block["path"])
        path = root / relative
        _require(not relative.is_absolute() and path.resolve().is_relative_to(root.resolve()),
                 "UNAUTHORIZED_DATA", "DSDE path escapes authorized root")
        _require(path.stat().st_size == block["size_bytes"], "CORRUPT_ARTIFACT", "DSDE source size changed")
    contents = []
    for block in selected:
        content = ledger.read(protocol_id, block["block_id"], purpose=purpose,
            authorization_id=authorization_id, authorization_version=authorization_version,
            data_root=root, consumer=consumer)
        _require(len(content) == block["size_bytes"], "CORRUPT_ARTIFACT", "DSDE admitted size differs")
        contents.append(content)
    return SourcePair(digest(protocol), digest(grant), encode(feature).decode("utf-8"),
                      encode(condition).decode("utf-8"), *contents)


@dataclass(frozen=True)
class ProjectionSpec:
    """Explicit registered local frame; no automatic data-fitted frame."""

    coordinate_frame: str
    origin_latitude_deg: float
    origin_longitude_deg: float
    max_offset_degrees: float
    method: str = "local-east-north-equirectangular-v1"
    earth_radius_m: float = 6_371_000.0

    def __post_init__(self):
        _require(type(self.coordinate_frame) is str and 0 < len(self.coordinate_frame) <= 128,
                 "CONTRACT_MISMATCH", "explicit metric coordinate frame required")
        _require(self.method == "local-east-north-equirectangular-v1" and self.earth_radius_m == 6_371_000.0,
                 "CONTRACT_MISMATCH", "registered projection method differs")
        values = (self.origin_latitude_deg, self.origin_longitude_deg, self.max_offset_degrees)
        _require(all(type(v) in (int, float) and math.isfinite(v) for v in values)
                 and abs(self.origin_latitude_deg) < 89 and abs(self.origin_longitude_deg) <= 180
                 and 0 < self.max_offset_degrees <= 1,
                 "CONTRACT_MISMATCH", "bounded local projection parameters required")


def context_binding(feature_spec, benchmark_binding):
    """Freeze the actual accepted representation, not a new terrain search."""
    from experiments.pirc22.consumer import validate_benchmark_selection_binding
    from experiments.pirc22.representations import load_representation_matrix
    binding = validate_benchmark_selection_binding(deepcopy(benchmark_binding))
    matrix = load_representation_matrix()
    matrix.validate_feature_spec(feature_spec)
    candidate = matrix.candidate(binding["selected_configuration"]["candidate_id"])
    return {"schema_version": "pirc26-frozen-context-v1", "converter_version": VERSION,
            "feature_spec_sha256": matrix.source_feature_spec_sha256,
            "consumer_identity_sha256": binding["consumer_identity_sha256"],
            "selection_identity_sha256": binding["source_selection_identity_sha256"],
            "variant_ids": list(candidate.variant_ids), "composition_ids": list(candidate.composition_ids),
            "context_dim": candidate.model_input_dim, "include_validity_indicators": True}


def _parquet(content, columns, *, max_rows=MAX_ROWS):
    import pyarrow as pa
    import pyarrow.parquet as pq
    _require(type(content) is bytes and 0 < len(content) <= MAX_BYTES,
             "RESOURCE_PLAN_REJECTED", "bounded admitted Parquet bytes required")
    try:
        source = pq.ParquetFile(pa.BufferReader(content))
        meta = source.metadata
        _require(0 < meta.num_rows <= max_rows and meta.num_columns <= 256
                 and sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)) <= MAX_DECODED_BYTES,
                 "RESOURCE_PLAN_REJECTED", "Parquet decoded source quota")
        _require(set(columns) <= set(source.schema_arrow.names), "CONTRACT_MISMATCH", "DSDE source columns missing")
        _require(all(pa.types.is_integer(source.schema_arrow.field(name).type)
                     or pa.types.is_floating(source.schema_arrow.field(name).type)
                     or pa.types.is_string(source.schema_arrow.field(name).type)
                     or pa.types.is_large_string(source.schema_arrow.field(name).type)
                     or pa.types.is_timestamp(source.schema_arrow.field(name).type)
                     for name in columns), "CONTRACT_MISMATCH", "scalar DSDE source columns required")
        table = source.read(columns=columns, use_threads=False)
        _require(table.nbytes <= MAX_DECODED_BYTES, "RESOURCE_PLAN_REJECTED", "decoded source byte quota")
        return table
    except ResearchError:
        raise
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "invalid admitted DSDE Parquet") from exc


def _context_adapter(spec, binding, entry):
    from experiments.nex326.pirc21_adapter import FeatureSnapshotAdapter
    from experiments.pirc22.representations import load_representation_matrix

    class AdmittedRows(FeatureSnapshotAdapter):
        """Reuse pure registered transforms, never the legacy filesystem loader."""

        def __init__(self):
            self.spec = deepcopy(spec)
            self.selection = load_representation_matrix().candidate(
                binding["selected_configuration"]["candidate_id"]).feature_selection()
            self.selection.validate()
            self._factor_by_id = {str(v["factor_id"]): v for v in self.spec["factors"]}
            self._variant_by_id = {str(v["variant_id"]): v for v in self.spec["variants"]}
            self._composition_by_id = {str(v["composition_id"]): v for v in self.spec["compositions"]}
            self._column_factor = {str(c["name"]): k for k, v in self._factor_by_id.items() for c in v["value_columns"]}
            self._validate_selection()
            _require(all(self._variant_by_id[v]["fit_scope"] == "none" for v in self.selection.variant_ids),
                     "CONTRACT_MISMATCH", "context requires separately frozen train-fitted transforms")
            self._fit_state, self._fit_scope = {}, None
            self.manifest = {"dataset_id": entry["dataset_id"], "snapshot_id": entry["release_id"],
                "content_inventory_sha256": entry["sha256"], "feature_spec_sha256": entry["feature_spec_sha256"],
                "files": [{"file_id": entry["file_id"], "split": entry["source_split"],
                           "sha256": entry["sha256"], "path": entry["path"]}]}
            self.table = None

        def _load(self, split, *, final_eval_unlock=None, file_ids=None, segment_ids=None):
            _require(split == entry["source_split"] and file_ids == [entry["file_id"]]
                     and final_eval_unlock is None and segment_ids is None and self.table is not None,
                     "UNAUTHORIZED_DATA", "context is restricted to the explicit admitted source table")
            return self.table

    return AdmittedRows()


def materialize_source_pair(pair, feature_spec, benchmark_binding, projection, *, block_id,
                            duplicate_policy="reject", fragment_policy="reject",
                            max_observations=16_912, max_output_bytes=MAX_BYTES):
    """Numerical geometry/context stage, before a train normalizer exists.

    No placeholder normalizer identity is assigned to this intermediate value.
    Managed production callers execute this stage inside the owned worker.
    It is not an observed-input artifact until bind_conversion adds actual
    frozen training/normalizer/context identities and checks its final size.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    _require(isinstance(pair, SourcePair) and isinstance(projection, ProjectionSpec)
             and all(_hash(v) for v in (pair.protocol_hash, pair.authorization_hash)),
             "CONTRACT_MISMATCH", "frozen conversion transport/provenance required")
    _require(type(block_id) is str and 0 < len(block_id) <= 128
             and type(duplicate_policy) is str and duplicate_policy in {"reject", "keep-first-exact-time-v1"}
             and type(fragment_policy) is str and fragment_policy in FRAGMENT_POLICIES,
             "CONTRACT_MISMATCH", "explicit block and timestamp policy required")
    _require(type(max_observations) is int and 3 <= max_observations <= MAX_ROWS
             and type(max_output_bytes) is int and 0 < max_output_bytes <= MAX_BYTES,
             "RESOURCE_PLAN_REJECTED", "bounded conversion quotas required")
    feature, condition = pair.metadata()
    if fragment_policy == SHORT_POLICY:
        _require(type(feature.get("aligned_row_count")) is int and 3 <= feature["aligned_row_count"] <= max_observations,
                 "CONTRACT_MISMATCH", "frozen aligned row count required for fragment eligibility")
    role = feature.get("split_role")
    _pair_contract(feature, condition, ROLES.get(role, (None, None))[1])
    for metadata, content in ((feature, pair.feature_bytes), (condition, pair.condition_bytes)):
        _require(type(content) is bytes and len(content) == metadata["size_bytes"]
                 and hashlib.sha256(content).hexdigest() == metadata["sha256"],
                 "CORRUPT_ARTIFACT", "admitted DSDE source bytes differ")
    frozen_context = context_binding(feature_spec, benchmark_binding)
    _require(feature["feature_spec_sha256"] == frozen_context["feature_spec_sha256"],
             "CONTRACT_MISMATCH", "conversion context identity differs")
    adapter = _context_adapter(feature_spec, benchmark_binding, feature)
    table = _parquet(pair.feature_bytes, adapter._required_columns(),
                     max_rows=min(max_observations, MAX_ROWS // (6 + frozen_context["context_dim"])))
    _require(3 <= table.num_rows <= max_observations and table.num_rows * (6 + frozen_context["context_dim"]) <= MAX_ROWS,
             "RESOURCE_PLAN_REJECTED", "observed-state decoded element/row quota")
    if "aligned_row_count" in feature:
        _require(type(feature["aligned_row_count"]) is int and feature["aligned_row_count"] == table.num_rows,
                 "CONTRACT_MISMATCH", "decoded aligned rows differ from frozen source inventory")
    adapter.table = table
    raw = _parquet(pair.condition_bytes, ["file_id", "t", "lat", "lon"])
    _require(pa.types.is_timestamp(raw["t"].type) and raw["t"].null_count == 0,
             "CONTRACT_MISMATCH", "exact condition timestamps required")
    epochs = pc.cast(pc.cast(raw["t"], pa.timestamp("ns")), pa.int64()).to_pylist()
    lat, lon = raw["lat"].to_pylist(), raw["lon"].to_pylist()
    _require(all(v == feature["file_id"] for v in raw["file_id"].to_pylist()),
             "CONTRACT_MISMATCH", "condition file identity differs")
    names = ("dataset_version", "point_id", "file_id", "point_index", "absolute_epoch_ns", "segment_id", "split", "independent_block_id")
    identity = table.select(names).to_pylist()
    seen_points, seen_segments, grouped = set(), set(), []
    previous_index = -1
    for offset, row in enumerate(identity):
        _require(row["dataset_version"] == feature["dataset_id"] and row["file_id"] == feature["file_id"]
                 and row["split"] == feature["source_split"] and row["independent_block_id"] == feature["independent_block_id"],
                 "CONTRACT_MISMATCH", "FeatureRow source/split/unit identity differs")
        _require(all(type(row[k]) is str and 0 < len(row[k]) <= 128 for k in ("point_id", "segment_id"))
                 and row["point_id"] not in seen_points, "CONTRACT_MISMATCH", "bounded unique point/segment identities required")
        index = row["point_index"]
        _require(type(index) is int and previous_index < index < len(epochs)
                 and type(row["absolute_epoch_ns"]) is int and epochs[index] == row["absolute_epoch_ns"],
                 "CONTRACT_MISMATCH", "exact source point/time/order join differs")
        previous_index = index
        seen_points.add(row["point_id"])
        if not grouped or grouped[-1][0] != row["segment_id"]:
            _require(row["segment_id"] not in seen_segments, "CONTRACT_MISMATCH", "source segment rows are not contiguous")
            seen_segments.add(row["segment_id"])
            grouped.append((row["segment_id"], []))
        grouped[-1][1].append((offset, index, row["absolute_epoch_ns"]))
    _require(len(grouped) <= 256, "RESOURCE_PLAN_REJECTED", "observed-state segment quota")
    context = adapter.transform(feature["source_split"], file_ids=[feature["file_id"]]).model_matrix()
    _require(context.shape == (table.num_rows, frozen_context["context_dim"]) and np.isfinite(context).all(),
             "CONTRACT_MISMATCH", "frozen context shape/finite contract differs")
    segments, membership, duplicate_count, dispositions = [], [], 0, []
    for segment_id, points in grouped:
        kept, previous_epoch = [], None
        for offset, index, epoch in points:
            if previous_epoch is not None:
                _require(0 <= epoch - previous_epoch <= 60_000_000_000,
                         "CONTRACT_MISMATCH", "source refined segment time boundary differs")
                if epoch == previous_epoch:
                    _require(duplicate_policy == "keep-first-exact-time-v1", "CONTRACT_MISMATCH", "duplicate source timestamp requires frozen policy")
                    duplicate_count += 1
                    continue
            previous_epoch = epoch
            kept.append((offset, index, epoch))
        _require(len(kept) <= 4098 and (len(kept) >= 3 or fragment_policy == SHORT_POLICY),
                 "RESOURCE_PLAN_REJECTED", "source segment needs explicit bounded window policy")
        first_epoch = kept[0][2]
        time = [(epoch - first_epoch) / 1_000_000_000 for _, _, epoch in kept]
        _require(all(a < b for a, b in zip(time, time[1:])), "CONTRACT_MISMATCH", "timestamps collapsed during seconds conversion")
        position_by_offset = {}
        # Opt-in exclusion must not conceal invalid points or fixed-frame
        # violations, including duplicates and short fragments. Strict legacy
        # mode retains its existing validation/rejection order and geometry.
        for offset, index, _ in (points if fragment_policy == SHORT_POLICY else kept):
            latitude, longitude = lat[index], lon[index]
            _require(type(latitude) in (int, float) and type(longitude) in (int, float)
                     and math.isfinite(latitude) and math.isfinite(longitude)
                     and abs(latitude) <= 90 and abs(longitude) <= 180,
                     "CONTRACT_MISMATCH", "finite valid source coordinates required")
            dy = latitude - projection.origin_latitude_deg
            dx = (longitude - projection.origin_longitude_deg + 180) % 360 - 180
            _require(max(abs(dx), abs(dy)) <= projection.max_offset_degrees,
                     "CONTRACT_MISMATCH", "source lies outside registered local projection")
            factor = projection.earth_radius_m * math.pi / 180
            position_by_offset[offset] = [dx * factor * math.cos(math.radians(projection.origin_latitude_deg)), dy * factor]
        if fragment_policy == SHORT_POLICY:
            excluded = len(kept) < 3
            dispositions.append({"segment_id": segment_id,
                "disposition": "excluded-short-causal-path" if excluded else "included-causal-path",
                "reason_code": "INSUFFICIENT_CAUSAL_POINTS" if excluded else None,
                "points": [{"point_id": identity[offset]["point_id"], "source_point_index": index,
                    "absolute_epoch_ns": epoch} for offset, index, epoch in points]})
            if excluded:
                continue
        position = [position_by_offset[offset] for offset, _, _ in kept]
        segments.append({"segment_id": segment_id, "time": time, "position": position,
                         "condition": [context[offset].tolist() for offset, _, _ in kept],
                         "condition_available_at": list(time)})
        membership.append({"segment_id": segment_id, "independent_block_id": feature["independent_block_id"],
                           "point_ids": [identity[offset]["point_id"] for offset, _, _ in kept],
                           "source_point_indices": [index for _, index, _ in kept], "absolute_start_epoch_ns": first_epoch})
    _require(bool(segments), "RESOURCE_PLAN_REJECTED", "source file has no eligible causal path; whole-file exclusion forbidden")
    document = {"schema_version": "pirc26-observed-block-v1", "block_id": block_id,
        "coordinate_frame": projection.coordinate_frame, "state_units": ["m", "m", "m/s", "m/s"], "time_unit": "s",
        "velocity_source": "backward-difference-v1", "segments": segments}
    content = encode(document)
    _require(len(content) <= max_output_bytes, "RESOURCE_PLAN_REJECTED", "converted JSON exceeds frozen byte quota")
    provenance = {"schema_version": "pirc26-dsde-conversion-provenance-v1", "converter_version": VERSION,
        "source_protocol_hash": pair.protocol_hash, "authorization_hash": pair.authorization_hash,
        "feature": feature, "condition": condition, "split_role": role, "purpose": ROLES[role][1],
        "projection": deepcopy(projection.__dict__), "context_binding": frozen_context,
        "duplicate_policy": duplicate_policy, "removed_duplicate_timestamps": duplicate_count,
        "membership": membership, "content_sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
        "scientific_qualification": "not-established"}
    if fragment_policy == SHORT_POLICY:
        from application.pirc26_fragment_contract import VERSION as DISPOSITION_VERSION, fragment_summary
        provenance.update(fragment_policy=fragment_policy, fragment_disposition={
            "schema_version": DISPOSITION_VERSION, "policy": fragment_policy,
            "source_rows": table.num_rows, "segments": dispositions})
        fragment_summary({"document": document, "provenance": provenance}, expected_policy=fragment_policy)
    return {"document": document, "content": content, "provenance": provenance}


def bind_conversion(materialized, *, train_binding_hash, normalizer_hash, context_hash, max_output_bytes=MAX_BYTES):
    """Finalize geometry with real frozen identities; never fit a normalizer."""
    _require(all(_hash(v) for v in (train_binding_hash, normalizer_hash, context_hash))
             and context_hash == digest(materialized["provenance"]["context_binding"]),
             "CONTRACT_MISMATCH", "conversion context/normalizer bindings differ")
    _require(type(max_output_bytes) is int and 0 < max_output_bytes <= MAX_BYTES,
             "RESOURCE_PLAN_REJECTED", "bounded final conversion quota required")
    document = {**materialized["document"], "train_binding_hash": train_binding_hash,
                "normalizer_hash": normalizer_hash, "context_hash": context_hash}
    content = encode(document)
    _require(len(content) <= max_output_bytes, "RESOURCE_PLAN_REJECTED", "converted JSON exceeds frozen byte quota")
    provenance = {**materialized["provenance"], "train_binding_hash": train_binding_hash,
                  "normalizer_hash": normalizer_hash, "content_sha256": hashlib.sha256(content).hexdigest(),
                  "size_bytes": len(content)}
    return {"document": document, "content": content, "provenance": provenance}


def convert_source_pair(pair, feature_spec, benchmark_binding, projection, *, block_id,
                        train_binding_hash, normalizer_hash, context_hash,
                        duplicate_policy="reject", fragment_policy="reject",
                        max_observations=16_912, max_output_bytes=MAX_BYTES):
    """Convert one admitted file with already frozen external bindings.

    This compatibility entry does not fit or qualify a normalizer. Managed
    multi-file preparation uses materialize_source_pair then bind_conversion.
    """
    _require(all(_hash(v) for v in (train_binding_hash, normalizer_hash, context_hash))
             and context_hash == digest(context_binding(feature_spec, benchmark_binding)),
             "CONTRACT_MISMATCH", "frozen conversion provenance required")
    materialized = materialize_source_pair(pair, feature_spec, benchmark_binding, projection,
        block_id=block_id, duplicate_policy=duplicate_policy, fragment_policy=fragment_policy, max_observations=max_observations,
        max_output_bytes=max_output_bytes)
    return bind_conversion(materialized, train_binding_hash=train_binding_hash, normalizer_hash=normalizer_hash,
                           context_hash=context_hash, max_output_bytes=max_output_bytes)
