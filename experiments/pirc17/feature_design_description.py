"""Describe original terrain-origin design columns from existing snapshot rows.

Only train/validation origin values are read. No map queries, models, fitting,
forecasts or final-evaluation numeric data. Source hashes bind all selected
files; this narrow description is not a new whole-snapshot admission gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
from .checkpoint_resume import load
from .configurations import subset_matrix, terrain_configurations
from .features import CanonicalEncoder
from .protocol_core import canonical, digest, file_hash, read_json, under, unpack


INVENTORY_SHA = "675b755e7b234698cc2954e5d743b36d41e37284c4abff200ac888da212fe250"
ORIGINAL_CONTEXT_FILE_SHA = "921311395ea79212e4d1a7303f5ad37f15617880db490139a2ccebf08dd5e7d1"
ROLES = {"train": 404, "validation": 81}


class _OriginTransformReader(FeatureSnapshotAdapter):
    """Use unchanged pointwise kernels, never the full-snapshot loader.

Metadata is pinned before construction; selected files are individually verified
by the caller. Normal formal consumers and their validation remain untouched.
"""
    def __init__(self, root, selection, manifest, spec):
        self._pinned_metadata = manifest, spec
        super().__init__(root, selection)

    def _validate_snapshot(self):
        manifest, spec = self._pinned_metadata
        if (self.manifest != manifest or self.spec != spec
                or manifest["schema_version"] != "pirc21-feature-snapshot-v1"
                or spec["schema_version"] != "pirc21-feature-spec-v1"
                or manifest["status"] != "valid"
                or manifest["feature_spec_sha256"] != digest(spec)):
            raise ValueError("pinned point-transform metadata differs")

    def _load(self, *args, **kwargs):
        raise RuntimeError("origin diagnostics cannot use the general snapshot loader")

    def fit(self, *args, **kwargs):
        raise RuntimeError("origin diagnostics cannot fit transformations")


def describe(values, valid, columns):
    x, masks = np.asarray(values, dtype=float), np.asarray(valid)
    if (x.ndim != 2 or masks.shape != x.shape or masks.dtype != np.dtype(bool)
            or x.shape[1] != len(columns) or len(set(columns)) != len(columns)
            or not len(x) or not np.isfinite(x[masks]).all()):
        raise ValueError("finite valid features and aligned Boolean masks required")
    numeric = np.where(masks, x, 0.)
    raw = np.column_stack((numeric, masks.astype(float), np.ones(len(x))))
    spectrum = np.linalg.svd(raw, compute_uv=False)
    tolerance = max(raw.shape) * np.finfo(float).eps * spectrum[0]
    rank = int(np.count_nonzero(spectrum > tolerance))
    stats = []
    for index, name in enumerate(columns):
        column = numeric[:, index]
        stats.append({"column": name, "valid_count": int(masks[:, index].sum()),
                      "valid_fraction": float(masks[:, index].mean()),
                      "constant_numeric_column": bool(np.all(column == column[0])),
                      "numeric_minimum": float(column.min()), "numeric_maximum": float(column.max()),
                      "numeric_mean": float(column.mean()), "numeric_population_std": float(column.std())})
    return {"rows": len(x), "numeric_feature_columns": len(columns),
            "raw_columns_with_masks_and_intercept": raw.shape[1],
            "raw_design_rank": rank, "svd_rank_tolerance": float(tolerance),
            "raw_design_condition_number": float(spectrum[0] / spectrum[-1]) if rank == raw.shape[1] else None,
            "nonzero_spectrum_condition_number": float(spectrum[0] / spectrum[rank-1]),
            "constant_one_mask_columns": int(np.all(masks, axis=0).sum()),
            "constant_zero_mask_columns": int(np.all(~masks, axis=0).sum()),
            "constant_numeric_columns": sum(v["constant_numeric_column"] for v in stats),
            "columns": stats}


def project(runtime, context_path, inventory_path):
    started = time.perf_counter()
    settings, _ = load(runtime)
    original = unpack(read_json(context_path, expected_file_sha256=ORIGINAL_CONTEXT_FILE_SHA))
    inputs = unpack(original["input_identity"])
    current = unpack(read_json(settings["context"]), expected_sha256=settings["context_sha256"])
    restored_inputs = unpack(current["input_identity"])
    if any(inputs[k] != restored_inputs[k] for k in ("terrain_development_identity", "configuration_columns")):
        raise ValueError("restored training population/columns differ from original")
    bundle = unpack(read_json(settings["bundle"]), expected_sha256=settings["bundle_sha256"])
    protocol = unpack(bundle["protocol"])
    if original["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("original training protocol differs")
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    fits = [m for m in inventory["models"] if m["matrix"] == "terrain"]
    if len(fits) != 10:
        raise ValueError("ten original terrain fits required")
    for m in fits:
        record = unpack(read_json(m["original_model_record_path"],
                                  expected_file_sha256=m["original_model_record_file_sha256"]),
                        expected_sha256=m["original_model_record_content_sha256"])
        if record["input_sha256"] != original["input_identity"]["sha256"]:
            raise ValueError("original fits not bound to this training input identity")
    binding = protocol["dataset_inputs"]
    frozen = binding["snapshot"]
    locations = {k: Path(v) for k, v in settings["input_paths"].items()}
    root, release = locations["snapshot"], locations["release"]
    manifest = read_json(root / "manifest.json", expected_file_sha256=frozen["manifest_sha256"])
    spec = read_json(root / "feature_spec.json", expected_file_sha256=frozen["feature_spec_file_sha256"])
    if digest(spec) != frozen["feature_spec_content_sha256"] or manifest["dataset_id"] != binding["dataset_id"]:
        raise ValueError("original snapshot metadata identity differs")
    configs = terrain_configurations()
    all_config = configs["all-terrain"]
    adapter = _OriginTransformReader(root, FeatureSelection(
        variant_ids=tuple(all_config["variant_ids"]), composition_ids=tuple(all_config["composition_ids"])), manifest, spec)
    encoder = CanonicalEncoder(adapter)
    if list(encoder.columns) != inputs["configuration_columns"]["all-terrain"]:
        raise ValueError("original full-column order differs")
    development = inputs["terrain_development_identity"]
    wanted = {s for role in ROLES for s in development["sample_ids"][role]}
    if len(wanted) != sum(ROLES.values()):
        raise ValueError("entire original development sample grid required")
    sample_path = release / "samples.jsonl"
    if file_hash(sample_path) != binding["release_artifact_sha256"]["samples.jsonl"]:
        raise ValueError("original sample manifest differs")
    samples = {}
    with sample_path.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["sample_id"] in wanted:
                if sample["sample_id"] in samples:
                    raise ValueError("duplicate original sample")
                samples[sample["sample_id"]] = sample
    if set(samples) != wanted:
        raise ValueError("missing original development samples")
    entries = {(e["split"], e["file_id"]): e for e in manifest["files"]}
    groups = defaultdict(list)
    for role, number in ROLES.items():
        ids = development["sample_ids"][role]
        if len(ids) != number:
            raise ValueError("original development count differs")
        for sid in ids:
            sample = samples[sid]
            if sample["split"] != role:
                raise ValueError("development role differs; no final-eval numeric reads")
            groups[role, sample["file_id"]].append(sample)
    needed = {"dataset_version", "file_id", "segment_id", "split", "independent_block_id", "point_index"}
    for vid in adapter.selection.variant_ids:
        params = adapter._variant_by_id[vid]["parameters"]
        names = list(params["input_columns"]) + list(params.get("reference_columns", ()))
        needed.update(names)
        needed.update(adapter._factor_by_id[adapter._column_factor[n]]["status_column"] for n in names)
    origins, source_hashes = {}, {}
    for (role, fid), group in groups.items():
        entry = entries[role, fid]
        bound = development["sources"][role + ":" + fid]
        if entry["sha256"] != bound["feature_sha256"] or entry["condition_sha256"] != bound["condition_sha256"]:
            raise ValueError("original feature/condition binding differs")
        path = under(root, entry["path"])
        if file_hash(path) != entry["sha256"] or pq.ParquetFile(path).metadata.num_rows != entry["row_count"]:
            raise ValueError("selected feature bytes/row count differ")
        table = pq.read_table(path, columns=sorted(needed),
                              filters=[("split", "=", role),
                                       ("segment_id", "in", sorted({s["segment_id"] for s in group}))])
        segments = defaultdict(list)
        for row in table.to_pylist():
            segments[row["segment_id"]].append(row)
        for sample in group:
            rows = sorted(segments[sample["segment_id"]], key=lambda r: r["point_index"])
            if len(rows) != sample["target_end"] + 1 or len({r["point_index"] for r in rows}) != len(rows):
                raise ValueError("original segment boundaries differ")
            row = rows[sample["history_end"]]
            if any(row[k] != v for k, v in {"dataset_version": binding["dataset_id"], "file_id": fid,
                    "split": role, "segment_id": sample["segment_id"],
                    "independent_block_id": sample["independent_block_id"]}.items()):
                raise ValueError("origin row identity differs")
            origins[sample["sample_id"]] = row
        source_hashes[role + ":" + fid] = entry["sha256"]
    if set(origins) != wanted or len(source_hashes) != len(development["sources"]):
        raise ValueError("incomplete original origin/source projection")
    summaries = {}
    for role in ROLES:
        ids = development["sample_ids"][role]
        features = encoder.encode([origins[sid] for sid in ids])
        matrix = features.model_matrix(len(ids))
        summaries[role] = {}
        for config in configs:
            columns = inputs["configuration_columns"][config]
            selected = subset_matrix(matrix, encoder.columns, columns)
            k = len(columns)
            summaries[role][config] = describe(selected[:, :k], selected[:, k:].astype(bool), columns)
    result = {"schema_version": "pirc17-original-feature-design-description-v1",
              "original_context_file_sha256": ORIGINAL_CONTEXT_FILE_SHA,
              "original_input_identity_sha256": original["input_identity"]["sha256"],
              "original_inventory_sha256": INVENTORY_SHA,
              "snapshot_manifest_sha256": frozen["manifest_sha256"],
              "feature_spec_sha256": frozen["feature_spec_content_sha256"],
              "selected_feature_files": len(source_hashes), "selected_sources_sha256": digest(source_hashes),
              "roles": summaries,
              "scope": {"new_fits": 0, "new_forecasts": 0, "new_particle_scores": 0,
                        "map_queries": 0, "final_eval_numeric_reads": 0,
                        "private_origins_rows_paths_or_ids_exported": False,
                        "whole_snapshot_revalidated": False,
                        "independent_saved_forecast_audit": False,
                        "description": "Exact original train/validation origin columns via unchanged frozen pointwise kernels. SVD rank uses max(n,p)*float64-epsilon*smax; deficient raw condition numbers are null, not finite. Descriptive source/design evidence, not an effect-size or final qualification claim."}}
    canonical(result)
    return result, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "original-context", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result, seconds = project(args.runtime, args.original_context, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"selected_files": result["selected_feature_files"], "seconds": seconds,
                      "train_origins": 404, "validation_origins": 81,
                      "new_fits_forecasts_or_map_queries": 0}))


if __name__ == "__main__":
    main()
