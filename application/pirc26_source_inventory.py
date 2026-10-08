"""Compile frozen source metadata, never permission or scientific bytes.

Only four explicitly hash-bound inventory documents are opened. Selected
Parquet paths are inspected for size/containment, never opened or decoded.
Whole independent units are preserved; no implicit first-file truncation,
final-evaluation source, production grant, registration or budget mutation.
Returned private metadata is for local operator preparation, not public export.
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from application.pirc26_dsde import _pair_contract
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, digest, identifier


VERSION = "pirc26-source-inventory-v1"
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_TOTAL_METADATA_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100_000
MAX_PAIRS = 16
MAX_SOURCE_BYTES = 32 * 1024 * 1024
NAMES = {"dataset": "dataset.json", "split": "split.json",
         "features": "manifest.json", "conditions": "condition_file_manifest.jsonl"}


def require(ok, message, code="CONTRACT_MISMATCH"):
    if not ok:
        raise ResearchError(code, message)


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _name(value):
    require(type(value) is str and 0 < len(value) <= 128, "bounded source identity required")
    return value


def _relative(value):
    require(type(value) is str and 0 < len(value) <= 4096
            and "\\" not in value and ":" not in value and "\x00" not in value,
            "portable relative source path required", "UNAUTHORIZED_DATA")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(p not in {".", ".."} for p in value.split("/"))
            and all(value.split("/")), "source path escapes lexical root", "UNAUTHORIZED_DATA")
    return Path(*path.parts)


def _root(value):
    value = Path(value)
    require(value.is_absolute(), "explicit absolute source root required", "UNAUTHORIZED_DATA")
    root = Path(os.path.abspath(value))
    require(root.is_dir() and root.resolve() == root,
            "source root must retain its lexical identity", "UNAUTHORIZED_DATA")
    return root


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate inventory JSON key")
        result[key] = value
    return result


def _json(content):
    try:
        return json.loads(content, object_pairs_hook=_unique_object,
                          parse_constant=lambda _: require(False, "nonfinite inventory JSON"))
    except ResearchError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ResearchError("CONTRACT_MISMATCH", "invalid bounded inventory JSON") from exc


def load_inventory(root, metadata):
    """Read exact detached inventory versions, with all handles verified jointly."""
    root = _root(root)
    require(type(metadata) is dict and set(metadata) == set(NAMES), "four explicit inventory references required")
    values, references, checks, total = {}, {}, [], 0
    with ExitStack() as stack:
        for key, name in NAMES.items():
            ref = metadata[key]
            require(type(ref) is dict and set(ref) == {"path", "sha256"} and _hash(ref["sha256"]),
                    "closed hash-bound inventory reference required")
            relative = _relative(ref["path"])
            require(relative.name == name, "only declared inventory filenames may be opened")
            stream, size, verify = stack.enter_context(opened_regular_file(
                root, root / relative, maximum_bytes=MAX_METADATA_BYTES))
            total += size
            require(total <= MAX_TOTAL_METADATA_BYTES, "joint inventory byte quota", "RESOURCE_PLAN_REJECTED")
            content = stream.read(size + 1)
            require(len(content) == size and hashlib.sha256(content).hexdigest() == ref["sha256"],
                    "inventory content differs from frozen reference", "IDENTITY_MISMATCH")
            if key == "conditions":
                lines = content.splitlines()
                require(0 < len(lines) <= MAX_ROWS and all(lines), "bounded condition inventory required")
                values[key] = [_json(line) for line in lines]
            else:
                values[key] = _json(content)
                require(type(values[key]) is dict, "inventory JSON object required")
            references[key] = dict(ref)
            checks.append(verify)
        # Earlier metadata cannot change while a later input is being consumed.
        for verify in checks:
            verify()
    values["references"] = references
    return values


def _index(rows, fields):
    require(type(rows) is list and 0 < len(rows) <= MAX_ROWS, "bounded inventory rows required")
    result = {}
    for row in rows:
        require(type(row) is dict and set(row) == set(fields), "closed inventory row required")
        file_id = _name(row["file_id"])
        require(file_id not in result, "duplicate inventory file identity")
        result[file_id] = row
    return result


def _source_size(root, relative):
    """Filename/stat metadata only. No source-file handle is opened."""
    path = root / _relative(relative)
    try:
        information = path.lstat()
        require(stat.S_ISREG(information.st_mode) and path.resolve().is_relative_to(root),
                "source must be a regular contained file", "UNAUTHORIZED_DATA")
        require(0 < information.st_size <= MAX_METADATA_BYTES,
                "source exceeds existing paired-file quota", "RESOURCE_PLAN_REJECTED")
        return information.st_size
    except OSError as exc:
        raise ResearchError("MISSING_ARTIFACT", "selected source metadata unavailable") from exc


def compile_source_protocol(root, metadata, *, feature_directory, condition_directory,
                            study_id, protocol_id, selected_units):
    """Compile every selected unit file; no grant, payload read or ready verdict."""
    root = _root(root)
    identifier(study_id)
    identifier(protocol_id)
    feature_dir, condition_dir = _relative(feature_directory), _relative(condition_directory)
    inventory = load_inventory(root, metadata)
    dataset, split, features = (inventory[k] for k in ("dataset", "split", "features"))
    dataset_id = _name(dataset.get("dataset_id"))
    require(dataset.get("schema_version") == "pirc20-release-v1" and dataset.get("status") == "complete"
            and split.get("schema_version") == "pirc20-release-v1" and split.get("release_id") == dataset_id
            and features.get("schema_version") == "pirc21-feature-snapshot-v1"
            and features.get("status") == "valid" and features.get("dataset_id") == dataset_id
            and features.get("cohort_id") == dataset_id, "source release/snapshot identities differ", "IDENTITY_MISMATCH")
    require(dataset.get("artifacts", {}).get("split.json", {}).get("sha256") == metadata["split"]["sha256"]
            and dataset.get("artifacts", {}).get("condition_file_manifest.jsonl", {}).get("sha256")
                == metadata["conditions"]["sha256"]
            and dataset.get("source", {}).get("condition_inventory_sha256") == metadata["conditions"]["sha256"],
            "release inventory bindings differ", "IDENTITY_MISMATCH")
    require(_hash(features.get("feature_spec_sha256")), "frozen feature specification required")
    release_id = _name(features.get("snapshot_id"))
    require((root / feature_dir / "manifest.json") == (root / _relative(metadata["features"]["path"])),
            "feature source directory differs from frozen manifest", "IDENTITY_MISMATCH")
    assignments = _index(split.get("assignments"), ("file_id", "independent_block_id", "split"))
    feature_files = _index(features.get("files"), ("file_id", "path", "sha256", "condition_sha256", "row_count", "split"))
    conditions = _index(inventory["conditions"], ("file_id", "relative_path", "sha256"))
    require(set(assignments) == set(feature_files) and set(assignments) <= set(conditions),
            "complete feature/source assignment inventory differs", "IDENTITY_MISMATCH")
    units, roles = {}, {}
    for file_id, assignment in assignments.items():
        unit, role = _name(assignment["independent_block_id"]), assignment["split"]
        require(role in {"train", "validation", "final_eval"}, "unknown original source split")
        require(roles.setdefault(unit, role) == role, "original unit crosses source splits", "UNAUTHORIZED_DATA")
        units.setdefault(unit, []).append(file_id)
        feature, condition = feature_files[file_id], conditions[file_id]
        require(feature["split"] == role and _hash(feature["sha256"]) and _hash(condition["sha256"])
                and feature["condition_sha256"] == condition["sha256"]
                and type(feature["row_count"]) is int and feature["row_count"] >= 0,
                "feature/condition/source role identities differ", "IDENTITY_MISMATCH")
    require(type(selected_units) is list and 0 < len(selected_units) <= MAX_PAIRS
            and all(type(unit) is str for unit in selected_units)
            and len(set(selected_units)) == len(selected_units), "explicit distinct bounded source units required")
    require(all(unit in units and roles[unit] in {"train", "validation"} for unit in selected_units),
            "selected units absent or protected", "UNAUTHORIZED_DATA")
    require(any(roles[unit] == "train" for unit in selected_units), "complete train population required", "UNAUTHORIZED_DATA")
    ordered = [(unit, file_id) for unit in selected_units for file_id in sorted(units[unit])]
    require(len(ordered) <= MAX_PAIRS, "complete unit population exceeds pair quota", "RESOURCE_PLAN_REJECTED")
    blocks, sources, members, total, source_paths = [], [], [], 0, set()
    for unit, file_id in ordered:
        f, c = feature_files[file_id], conditions[file_id]
        role = "train" if roles[unit] == "train" else "selection"
        purpose = "fit" if role == "train" else "select"
        require(f["row_count"] > 0, "selected file has no aligned features", "RESOURCE_PLAN_REJECTED")
        feature_path = (feature_dir / _relative(f["path"])).as_posix()
        condition_path = (condition_dir / _relative(c["relative_path"])).as_posix()
        require(PurePosixPath(f["path"]).parts[:2] == ("features", roles[unit]),
                "source filename split differs from declared role", "UNAUTHORIZED_DATA")
        # Condition inventories retain historical container names. Only the
        # frozen current assignment and exact per-file hash bind source role;
        # a legacy 'eval' directory is neither a grant nor a current test split.
        require(feature_path not in source_paths and condition_path not in source_paths
                and feature_path != condition_path, "source paths alias distinct members", "IDENTITY_MISMATCH")
        source_paths.update((feature_path, condition_path))
        feature_size, condition_size = _source_size(root, feature_path), _source_size(root, condition_path)
        total += feature_size + condition_size
        require(total <= MAX_SOURCE_BYTES, "complete source byte quota", "RESOURCE_PLAN_REJECTED")
        key = digest([dataset_id, release_id, file_id])
        common = {"dataset_id": dataset_id, "source_block_id": unit, "file_id": file_id,
            "independent_block_id": unit, "source_split": roles[unit], "split_role": role, "fit_scope": role == "train"}
        feature = {**common, "block_id": "features-" + key, "source_kind": "dsde-feature-parquet",
            "release_id": release_id, "path": feature_path, "sha256": f["sha256"],
            "size_bytes": feature_size, "feature_spec_sha256": features["feature_spec_sha256"]}
        condition = {**common, "block_id": "conditions-" + key, "source_kind": "dsde-condition-parquet",
            "release_id": dataset_id, "path": condition_path, "sha256": c["sha256"], "size_bytes": condition_size}
        _pair_contract(feature, condition, purpose)
        blocks.extend((feature, condition))
        sources.append({"feature_block_id": feature["block_id"], "condition_block_id": condition["block_id"],
                        "output_block_id": "prepared-" + key, "purpose": purpose})
        members.append({"independent_block_id": unit, "file_id": file_id, "source_split": roles[unit],
                        "feature_sha256": f["sha256"], "condition_sha256": c["sha256"],
                        "declared_aligned_rows": f["row_count"]})
    require(root.resolve() == root, "source root changed during metadata compilation", "UNAUTHORIZED_DATA")
    proof = {"schema_version": VERSION, "metadata": inventory["references"],
             "selected_units": list(selected_units), "members": members,
             "source_size_bytes": total, "payload_hashes": "declared-not-read",
             "condition_path_role": "legacy-container-not-read-authority",
             "execution_readiness": "not-established"}
    return {"protocol": {"schema_version": "pirc25-data-protocol-v1", "protocol_id": protocol_id,
                         "study_id": study_id, "blocks": blocks, "source_inventory_hash": digest(proof)},
            "source_selections": sources, "membership": proof}
