"""Disposable metadata compiler controls; no production source or grants."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from application.pirc26_source_inventory import compile_source_protocol, load_inventory, MAX_PAIRS
from application.research_data import EvaluationExposureLedger
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode


@pytest.fixture
def inventory(tmp_path):
    assignments, features, conditions = [], [], []
    for index, (unit, split) in enumerate((("train-unit", "train"), ("train-unit", "train"),
                                         ("selection-unit", "validation"), ("sealed-unit", "final_eval"))):
        file_id = "file-" + str(index)
        fp, cp = f"snapshot/features/{split}/{file_id}.parquet", f"conditions/{split}/{file_id}.parquet"
        for path in (fp, cp):
            p = tmp_path / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(("synthetic-not-parquet:" + path).encode())
        condition_hash = hashlib.sha256((tmp_path / cp).read_bytes()).hexdigest()
        assignments.append({"file_id": file_id, "independent_block_id": unit, "split": split})
        features.append({"file_id": file_id, "path": fp.removeprefix("snapshot/"), "split": split,
                         "sha256": hashlib.sha256((tmp_path / fp).read_bytes()).hexdigest(),
                         "condition_sha256": condition_hash, "row_count": 5})
        conditions.append({"file_id": file_id, "relative_path": cp.removeprefix("conditions/"), "sha256": condition_hash})
    docs = {"dataset": {"schema_version": "pirc20-release-v1", "dataset_id": "disposable-dataset", "status": "complete"},
            "split": {"schema_version": "pirc20-release-v1", "release_id": "disposable-dataset", "assignments": assignments},
            "features": {"schema_version": "pirc21-feature-snapshot-v1", "status": "valid",
                "dataset_id": "disposable-dataset", "cohort_id": "disposable-dataset", "snapshot_id": "disposable-snapshot",
                "feature_spec_sha256": "a" * 64, "files": features}, "conditions": conditions}
    paths = {"dataset": "release/dataset.json", "split": "release/split.json",
             "features": "snapshot/manifest.json", "conditions": "release/condition_file_manifest.jsonl"}

    def seal():
        references = {}
        for key in ("split", "features", "conditions", "dataset"):
            if key == "dataset":
                docs[key]["artifacts"] = {"split.json": {"sha256": references["split"]["sha256"]},
                    "condition_file_manifest.jsonl": {"sha256": references["conditions"]["sha256"]}}
                docs[key]["source"] = {"condition_inventory_sha256": references["conditions"]["sha256"]}
            data = b"".join(encode(r) + b"\n" for r in docs[key]) if key == "conditions" else encode(docs[key])
            path = tmp_path / paths[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            references[key] = {"path": paths[key], "sha256": hashlib.sha256(data).hexdigest()}
        return references

    return tmp_path, docs, seal


def compile(inventory, **kwargs):
    root, _, seal = inventory
    return compile_source_protocol(root, seal(), feature_directory="snapshot", condition_directory="conditions",
        study_id="disposable-preparation", protocol_id="disposable-sources",
        selected_units=kwargs.pop("selected_units", ["train-unit", "selection-unit"]), **kwargs)


def test_complete_unit_membership_roles_and_original_protocol_compatibility(inventory):
    result = compile(inventory)
    blocks = result["protocol"]["blocks"]
    assert len(blocks) == 6  # both train files, not just the first
    assert [m["file_id"] for m in result["membership"]["members"]] == ["file-0", "file-1", "file-2"]
    assert [s["purpose"] for s in result["source_selections"]] == ["fit", "fit", "select"]
    assert all(b["source_block_id"] == b["independent_block_id"] for b in blocks)
    assert all(b["fit_scope"] is (b["split_role"] == "train") for b in blocks)
    assert all(b["aligned_row_count"] == 5 for b in blocks if b["source_kind"] == "dsde-feature-parquet")
    assert result["membership"]["payload_hashes"] == "declared-not-read"
    assert result["membership"]["execution_readiness"] == "not-established"
    assert result["protocol"]["source_inventory_hash"] == digest(result["membership"])
    assert not any("authorization" in k for k in result)
    store = ResearchStore(inventory[0] / "disposable-ledger", "inventory-fixture", initialize=True)
    EvaluationExposureLedger(store).register_protocol(result["protocol"], digest(result["protocol"]))
    assert store.manifest("protocol-disposable-sources") == result["protocol"]
    assert not store.attempts()


def test_compiler_never_opens_selected_or_final_scientific_payloads(inventory, monkeypatch):
    refs = inventory[2]()
    import application.pirc26_source_inventory as module
    actual = module.opened_regular_file
    opened = []

    def metadata_only(root, path, **kwargs):
        assert path.suffix != ".parquet"
        opened.append(path.name)
        return actual(root, path, **kwargs)

    monkeypatch.setattr(module, "opened_regular_file", metadata_only)
    (inventory[0] / "snapshot/features/final_eval/file-3.parquet").unlink()
    (inventory[0] / "conditions/final_eval/file-3.parquet").unlink()
    result = module.compile_source_protocol(inventory[0], refs, feature_directory="snapshot", condition_directory="conditions",
        study_id="disposable-preparation", protocol_id="disposable-sources", selected_units=["train-unit", "selection-unit"])
    assert set(opened) == {"dataset.json", "split.json", "manifest.json", "condition_file_manifest.jsonl"}
    assert len(result["protocol"]["blocks"]) == 6


def test_legacy_condition_folders_do_not_relabel_current_frozen_source_roles(inventory):
    root, docs, _ = inventory
    old = root / "conditions/train/file-0.parquet"
    new = root / "conditions/eval/file-0.parquet"
    new.parent.mkdir()
    old.rename(new)
    docs["conditions"][0]["relative_path"] = "eval/file-0.parquet"
    result = compile(inventory)
    block = next(b for b in result["protocol"]["blocks"] if b["path"] == "conditions/eval/file-0.parquet")
    assert block["source_split"] == block["split_role"] == "train" and block["fit_scope"] is True
    assert result["membership"]["condition_path_role"] == "legacy-container-not-read-authority"


def test_selected_source_aliases_cannot_be_counted_as_distinct_files(inventory):
    docs = inventory[1]
    docs["features"]["files"][1]["path"] = docs["features"]["files"][0]["path"]
    with pytest.raises(ResearchError, match="IDENTITY_MISMATCH"):
        compile(inventory)


@pytest.mark.parametrize("units", [["sealed-unit"], ["train-unit", "sealed-unit"], ["missing"],
    ["selection-unit"], ["train-unit", "train-unit"], [], [True]])
def test_protected_unknown_duplicate_or_trainless_units_refused(inventory, units):
    with pytest.raises(ResearchError):
        compile(inventory, selected_units=units)


@pytest.mark.parametrize("change", ["missing-feature", "missing-condition", "duplicate-feature", "duplicate-condition",
    "duplicate-assignment", "cross-split-unit", "wrong-feature-split", "wrong-condition-hash", "wrong-dataset",
    "wrong-cohort", "zero-rows", "boolean-rows", "feature-traversal", "condition-traversal", "feature-split-path",
    "condition-split-path", "windows-absolute"])
def test_forged_or_incomplete_inventory_refused(inventory, change):
    docs = inventory[1]
    if change == "missing-feature": docs["features"]["files"].pop(0)
    elif change == "missing-condition": docs["conditions"].pop(0)
    elif change == "duplicate-feature": docs["features"]["files"].append(deepcopy(docs["features"]["files"][0]))
    elif change == "duplicate-condition": docs["conditions"].append(deepcopy(docs["conditions"][0]))
    elif change == "duplicate-assignment": docs["split"]["assignments"].append(deepcopy(docs["split"]["assignments"][0]))
    elif change == "cross-split-unit": docs["split"]["assignments"][2]["independent_block_id"] = "train-unit"
    elif change == "wrong-feature-split": docs["features"]["files"][0]["split"] = "validation"
    elif change == "wrong-condition-hash": docs["features"]["files"][0]["condition_sha256"] = "b" * 64
    elif change == "wrong-dataset": docs["features"]["dataset_id"] = "other-dataset"
    elif change == "wrong-cohort": docs["features"]["cohort_id"] = "other-cohort"
    elif change == "zero-rows": docs["features"]["files"][0]["row_count"] = 0
    elif change == "boolean-rows": docs["features"]["files"][0]["row_count"] = True
    elif change == "feature-traversal": docs["features"]["files"][0]["path"] = "features/train/../../../outside.parquet"
    elif change == "condition-traversal": docs["conditions"][0]["relative_path"] = "train/../../outside.parquet"
    elif change == "feature-split-path": docs["features"]["files"][0]["path"] = "features/final_eval/file-0.parquet"
    elif change == "condition-split-path": docs["conditions"][0]["relative_path"] = "final_eval/file-0.parquet"
    elif change == "windows-absolute": docs["features"]["files"][0]["path"] = "Q:/outside.parquet"
    with pytest.raises(ResearchError):
        compile(inventory)


def test_whole_population_quota_refuses_without_silent_truncation(inventory, monkeypatch):
    import application.pirc26_source_inventory as module
    monkeypatch.setattr(module, "MAX_PAIRS", 2)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        compile(inventory, selected_units=["train-unit", "selection-unit"])
    monkeypatch.setattr(module, "MAX_PAIRS", MAX_PAIRS)
    monkeypatch.setattr(module, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        compile(inventory)


@pytest.mark.parametrize("boundary", ["frozen-hash", "release-binding", "duplicate-key", "nonfinite", "directory-binding"])
def test_inventory_read_and_release_content_bindings_refused(inventory, boundary):
    root, _, seal = inventory
    refs = seal()
    if boundary == "directory-binding":
        with pytest.raises(ResearchError, match="IDENTITY_MISMATCH"):
            compile_source_protocol(root, refs, feature_directory="other", condition_directory="conditions",
                study_id="disposable-preparation", protocol_id="disposable-sources", selected_units=["train-unit"])
        return
    if boundary == "frozen-hash": refs["split"]["sha256"] = "f" * 64
    elif boundary == "release-binding": refs["conditions"]["sha256"] = "f" * 64
    else:
        data = b'{"duplicate":1,"duplicate":2}' if boundary == "duplicate-key" else b'{"bad":NaN}'
        (root / refs["features"]["path"]).write_bytes(data)
        refs["features"]["sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(ResearchError): load_inventory(root, refs)


def test_selected_missing_and_nonregular_sources_refused(inventory):
    p = inventory[0] / "conditions/train/file-0.parquet"
    p.unlink()
    with pytest.raises(ResearchError, match="MISSING_ARTIFACT"): compile(inventory)
    p.mkdir()
    with pytest.raises(ResearchError, match="UNAUTHORIZED_DATA"): compile(inventory)


def test_joint_open_scope_detects_earlier_metadata_replacement(inventory, monkeypatch):
    import application.pirc26_source_inventory as module
    refs = inventory[2]()
    actual = module.opened_regular_file

    def mutate_earlier(root, path, **kwargs):
        if path.name == "condition_file_manifest.jsonl":
            target = root / refs["dataset"]["path"]
            target.write_bytes(target.read_bytes() + b" ")
        return actual(root, path, **kwargs)

    monkeypatch.setattr(module, "opened_regular_file", mutate_earlier)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        load_inventory(inventory[0], refs)
