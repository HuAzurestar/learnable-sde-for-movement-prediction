"""Describe existing recording/split identities; never fit or forecast."""
from __future__ import annotations

import argparse
import ast
import hashlib
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from .method_algorithm_description import INVENTORY_SHA
from .protocol_core import canonical, decode, digest, file_hash, read_json, sha256, unpack

ROLES = ("train", "validation", "final_eval")
IDENTITIES = ("sample_id", "segment_id", "independent_block_id")


def metadata_functions(builder_source, adapter_source):
    """Extract only hash assignment functions, without importing data/fitting code."""
    namespace = {"hashlib": hashlib, "PIRC20AdapterError": ValueError}
    wanted = ({"_split_for_block"}, {"_rank", "_roles"})
    for source, names in zip((builder_source, adapter_source), wanted):
        tree = ast.parse(source)
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise ValueError("complete source assignment functions required")
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"ADAPT_SEED", "ADAPT_FRACTION"}:
                        namespace[target.id] = ast.literal_eval(node.value)
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<bound-identity-functions>", "exec"), namespace)
    if namespace["ADAPT_SEED"] != 20260912 or namespace["ADAPT_FRACTION"] != 0.20:
        raise ValueError("original adaptation assignment differs")
    return namespace["_split_for_block"], namespace["_roles"]


def audit_rows(assignments, samples, *, seed, split_function, role_function):
    """Join metadata fields, keeping all private identifiers inside this function."""
    assigned = {}
    assignment_sets = {role: {name: set() for name in ("file_id", "independent_block_id")}
                       for role in ROLES}
    for row in assignments:
        file_id, block, role = row["file_id"], sha256(row["independent_block_id"]), row["split"]
        if (not isinstance(file_id, str) or not file_id or file_id in assigned
                or role not in ROLES or split_function(block, seed) != role):
            raise ValueError("duplicate file or mismatched source-hash assignment")
        assigned[file_id] = (block, role)
        assignment_sets[role]["file_id"].add(file_id)
        assignment_sets[role]["independent_block_id"].add(block)
    seen = {role: {name: set() for name in ("file_id", *IDENTITIES)} for role in ROLES}
    identities = {role: [] for role in ROLES}
    segment_rows, ordered = {}, hashlib.sha256()
    for row in samples:
        role, file_id, block = row["split"], row["file_id"], row["independent_block_id"]
        if role not in ROLES or assigned.get(file_id) != (block, role):
            raise ValueError("sample file/block/split join differs")
        identity = {name: row[name] for name in ("data_version", "file_id", "segment_id",
                     "history_start", "history_end", "target_start", "target_end")}
        bounds = [row[name] for name in ("history_start", "history_end", "target_start", "target_end")]
        if any(type(value) is not int for value in bounds):
            raise ValueError("integer window bounds required")
        start, end, future, last = bounds
        if start != 0 or last < 3 or end != max(1, (last + 1) // 2 - 1) or future != end + 1:
            raise ValueError("one-segment midpoint history/target bounds differ")
        if sha256(row["sample_id"]) != digest(identity):
            raise ValueError("sample identity does not bind exact window metadata")
        segment = row["segment_id"]
        if not isinstance(segment, str) or not segment or segment in segment_rows:
            raise ValueError("one unique midpoint sample per segment required")
        for name in ("file_id", *IDENTITIES):
            seen[role][name].add(row[name])
        identities[role].append([row[name] for name in IDENTITIES])
        segment_rows[segment] = SimpleNamespace(segment_id=segment, split=role,
                                                independent_block_id=block, file_id=file_id)
        ordered.update((row["sample_id"] + "\n").encode("utf-8"))
    intersections = {f"{a}:{b}": {name: len(seen[a][name] & seen[b][name])
                     for name in ("file_id", *IDENTITIES)} for a, b in combinations(ROLES, 2)}
    assignment_intersections = {f"{a}:{b}": {name: len(assignment_sets[a][name] & assignment_sets[b][name])
                               for name in assignment_sets[a]} for a, b in combinations(ROLES, 2)}
    if any(n for table in (intersections, assignment_intersections)
           for fields in table.values() for n in fields.values()):
        raise ValueError("source-file/recording/segment/window split overlap")
    legacy_partition = {
        "counts": {role: len(identities[role]) for role in ROLES},
        "intersections": {pair: {name: fields[name] for name in IDENTITIES}
                          for pair, fields in intersections.items()},
        "split_identity_sha256": {role: digest({name: sorted(seen[role][name]) for name in IDENTITIES})
                                  for role in ROLES},
        "split_row_identity_sha256": {role: digest(sorted(identities[role])) for role in ROLES}}
    adapter_roles = role_function(list(segment_rows.values()), False)
    role_sets = {role: {name: set() for name in ("file_id", "segment_id", "independent_block_id")}
                 for role in ("train", "adapt", "validation")}
    for segment, role in adapter_roles.items():
        row = segment_rows[segment]
        if role not in role_sets or row.split == "final_eval":
            raise ValueError("development assignment includes final evaluation")
        for name in role_sets[role]:
            role_sets[role][name].add(getattr(row, name))
    inner_intersections = {f"{a}:{b}": {name: len(role_sets[a][name] & role_sets[b][name])
                          for name in role_sets[a]} for a, b in combinations(role_sets, 2)}
    if any(n for fields in inner_intersections.values() for n in fields.values()):
        raise ValueError("development role identity overlap")
    public = {
        "release_counts": {role: {"assigned_files": len(assignment_sets[role]["file_id"]),
                         "assigned_recording_hash_blocks": len(assignment_sets[role]["independent_block_id"]),
                         "sampled_files": len(seen[role]["file_id"]),
                         "sampled_recording_hash_blocks": len(seen[role]["independent_block_id"]),
                         "midpoint_windows": len(identities[role]),
                         "sampled_segments": len(seen[role]["segment_id"])} for role in ROLES},
        "release_intersections": intersections,
        "assignment_intersections": assignment_intersections,
        "global_development_roles": {role: {name: len(values) for name, values in fields.items()}
                                     for role, fields in role_sets.items()},
        "global_development_role_intersections": inner_intersections,
        "ordered_sample_ids_sha256": ordered.hexdigest()}
    return public, legacy_partition, segment_rows, adapter_roles


def project(release, bundle_path, inventory_path):
    bundle = unpack(read_json(bundle_path))
    protocol, execution = unpack(bundle["protocol"]), unpack(bundle["execution"])
    if execution["protocol_sha256"] != bundle["protocol"]["sha256"]:
        raise ValueError("protocol/execution identity differs")
    binding = protocol["dataset_inputs"]
    release = Path(release)
    dataset = read_json(release / "dataset.json", expected_file_sha256=binding["dataset_file_sha256"])
    if dataset["dataset_id"] != binding["dataset_id"]:
        raise ValueError("release identity differs")
    names = ("split.json", "cohort.json", "samples.jsonl", "leakage_report.json")
    hashes = {name: binding["release_artifact_sha256"][name] for name in names}
    for name, expected in hashes.items():
        if dataset["artifacts"][name]["sha256"] != expected:
            raise ValueError("release artifact binding differs")
    split = read_json(release / "split.json", expected_file_sha256=hashes["split.json"])
    cohort = read_json(release / "cohort.json", expected_file_sha256=hashes["cohort.json"])
    leakage = read_json(release / "leakage_report.json", expected_file_sha256=hashes["leakage_report.json"])
    if (split["seed"] != 20260912 or split["ratios"] != {"train": .70, "validation": .15, "final_eval": .15}
            or split["user_group"]["status"] != "unavailable"
            or cohort["window"]["mode"] != "nex326_midpoint"
            or cohort["sample_manifest_sha256"] != hashes["samples.jsonl"]):
        raise ValueError("original grouping/window contract differs")
    psde = Path(__file__).resolve().parents[2]
    source_paths = {"DSDE-SDE/trajectory/cohort_release.py": psde.parent / "DSDE-SDE/trajectory/cohort_release.py",
                    "PSDE-SDE/experiments/nex326/pirc20_adapter.py": psde / "experiments/nex326/pirc20_adapter.py"}
    sources = {}
    for relative, path in source_paths.items():
        expected = protocol["source_sha256"][relative]
        if file_hash(path) != expected or execution["source_sha256"][relative] != expected:
            raise ValueError("frozen identity assignment source differs")
        sources[relative] = expected
    functions = metadata_functions(*(path.read_text(encoding="utf-8") for path in source_paths.values()))
    if file_hash(release / "samples.jsonl") != hashes["samples.jsonl"]:
        raise ValueError("sample manifest bytes differ")
    def rows():
        with (release / "samples.jsonl").open("rb") as stream:
            while raw := stream.readline(16385):
                if len(raw) > 16384:
                    raise ValueError("bounded sample metadata row required")
                row = decode(raw)
                if row["data_version"] != dataset["dataset_id"]:
                    raise ValueError("sample release identity differs")
                yield row
    public, partition, segments, roles = audit_rows(split["assignments"], rows(), seed=split["seed"],
                                             split_function=functions[0], role_function=functions[1])
    if (any(partition[name] != binding["partitions"][name] for name in partition)
            or partition["counts"] != cohort["sample_counts_by_split"]
            or sum(partition["counts"].values()) != cohort["sample_count"]
            or public["ordered_sample_ids_sha256"] != cohort["ordered_sample_ids_sha256"]):
        raise ValueError("complete original sample population identity differs")
    for role in ROLES:
        if public["release_counts"][role]["assigned_files"] != dataset["split_counts"][role]["files"]:
            raise ValueError("complete original assigned file population differs")
        public["release_counts"][role]["released_refined_segments"] = dataset["split_counts"][role]["segments"]
        public["release_counts"][role]["aligned_points_reported"] = dataset["split_counts"][role]["points"]
    inventory = read_json(inventory_path, expected_file_sha256=INVENTORY_SHA)
    selected, fits, slots = None, set(), set()
    for entry in inventory["models"]:
        if entry["matrix"] != "NEX326-methods":
            continue
        record = unpack(read_json(entry["original_model_record_path"], expected_file_sha256=entry["original_model_record_file_sha256"]),
                        expected_sha256=entry["original_model_record_content_sha256"])
        training = record["artifact"]["training"]
        if (record["protocol_sha256"] != bundle["protocol"]["sha256"]
                or record["matrix_sha256"] != inventory["matrix_sha256"]
                or record["fit_identity"] != entry["fit_identity"]
                or record["parameter_identity"] != entry["parameter_identity"]
                or training["input_sha256"] != binding["development"]["method_input_sha256"]
                or record["fit_identity"] in fits):
            raise ValueError("saved fit population binding differs")
        fits.add(record["fit_identity"])
        if slots.intersection(entry["prediction_configs"]):
            raise ValueError("duplicate method slot")
        slots.update(entry["prediction_configs"])
        current = {role: set() for role in ("train", "adapt", "validation")}
        for row in training["per_segment"]:
            segment, role = row["segment_id"], row["role"]
            if role not in current or roles.get(segment) != role or segment in current[role]:
                raise ValueError("saved selected fit roles disagree with original split rule")
            current[role].add(segment)
        if ({role: len(values) for role, values in current.items()} != binding["development"]["sample_counts"]
                or training["sample_counts"] != binding["development"]["sample_counts"]
                or selected is not None and current != selected):
            raise ValueError("all saved method fits must use the same complete original windows")
        selected = current
    if len(fits) != 16 or len(slots) != 28 or selected is None:
        raise ValueError("complete original 16-fit/28-slot method inventory required")
    public["selected_method_roles"] = {role: {"windows": len(values),
        "recording_hash_blocks": len({segments[s].independent_block_id for s in values}),
        "files": len({segments[s].file_id for s in values})} for role, values in selected.items()}
    public.update({"schema_version": "pirc17-recording-partition-description-v1",
        "dataset_sha256": binding["dataset_file_sha256"], "release_artifact_sha256": hashes,
        "protocol_sha256": bundle["protocol"]["sha256"], "source_sha256": sources,
        "inventory_sha256": INVENTORY_SHA, "sample_identity_partition_matches_original_protocol": True,
        "release_assignment_seed": split["seed"], "adaptation_assignment_seed": 20260912,
        "adaptation_block_fraction": .20,
        "block_definition": "SHA256 of UTF-8 literal coords string, NUL, literal speeds_kmh string, NUL; no participant or geographic identity",
        "legacy_directory_split_report": leakage["legacy_split_audit"],
        "historical_exposure": binding["history"],
        "scope": {"new_fits": 0, "new_forecasts": 0, "new_particle_scores": 0,
            "raw_coordinate_speed_or_clock_values_read": False, "raw_recording_hashes_recomputed": False,
            "source_point_alignment_rows_reread": False, "participant_identifiers_available": False,
            "participant_or_near_route_independence_established": False,
            "physical_timestamp_provenance_established": False, "independent_saved_forecast_audit": False,
            "private_file_segment_sample_recording_ids_exported": False,
            "legacy_directory_split_report_recomputed": False,
            "global_role_counts_are_task_qualified_fit_population": False}})
    canonical(public)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "bundle", "inventory", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = project(args.release, args.bundle, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print("Described all original recording/split/window metadata; no fits or forecasts.")


if __name__ == "__main__":
    main()
