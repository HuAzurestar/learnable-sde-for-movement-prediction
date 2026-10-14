"""File-based four-state preparation, never a registration or research launch."""

from dataclasses import replace
import copy
import hashlib
import json

import pytest

from domain.frozen_dynamics import content_hash
from domain.mixture import MixtureSettings
from experiments.pirc27.design import StudyMethod, freeze_design
from experiments.pirc27.preparation import prepare_study
from infrastructure.research_store import ResearchError, ResearchStore, encode
from tests.test_propagation_study_design import fixture


def documents(tmp_path, *, nonlinear=False, labelled=False):
    methods = (StudyMethod("euler", samples=8, steps=2, chunk_size=4, recovery=True),
               StudyMethod("mixture", samples=8, steps=2, recovery=True,
                   mixture_settings=MixtureSettings(2, .1, 0., (1.,)*4, 0., 100000, 60.)),
               StudyMethod("pde", samples=8, steps=2))
    design = fixture(nonlinear=nonlinear, methods=methods)
    if labelled:
        design = replace(design,
            models=tuple(replace(m, configuration_id="physical") for m in design.models),
            methods=tuple(replace(m, configuration_id="registered") for m in design.methods))
    frozen = freeze_design(design)
    path = tmp_path / "design.json"
    path.write_bytes(encode(frozen.manifest()))
    root = tmp_path / "runtime"
    # Disposable engineering identity only; no registrations or execution.
    store = ResearchStore(root, "preparation-unit", initialize=True)
    config = {"schema_version": "pirc25-local-runtime-config-v1",
        "runtime_root": str(root), "store_id": store.store_id,
        "contract_version": "pirc25-contract-v1", "participants": ["PIRC-26", "PIRC-27", "PIRC-28"],
        "data_source_root": str(tmp_path / "data-not-opened"),
        "data_binding_status": "source_location_only_not_study_registration_or_read_authority",
        "pirc38_runtime_baseline": "a"*40, "initialized_using_code_sha": "b"*40}
    connection = root / "runtime.json"
    connection.write_bytes(encode(config))
    return path, frozen, connection, config, store


def prepare(path, frozen, connection):
    return prepare_study(path, expected_hash=frozen.manifest_hash, runtime_config=connection,
        expected_runtime_config_hash=hashlib.sha256(connection.read_bytes()).hexdigest())


@pytest.mark.parametrize("nonlinear,labelled", [(False, False), (True, False), (False, True), (True, True)])
def test_roundtrip_full_matrix_without_mutating_existing_runtime(tmp_path, nonlinear, labelled):
    path, frozen, connection, config, store = documents(tmp_path, nonlinear=nonlinear, labelled=labelled)
    before = {p.relative_to(store.path).as_posix(): p.read_bytes()
              for p in store.path.rglob("*") if p.is_file()}
    result = prepare(path, frozen, connection)
    spec = result["study_spec"]
    original = frozen.study_spec(expected_hash=frozen.manifest_hash)
    assert spec == {**original, "runtime_binding": {"root": config["runtime_root"], "store_id": store.store_id}}
    assert result["study_spec_hash"] == content_hash(spec)
    assert result["design_hash"] == frozen.manifest_hash
    assert result["preparation_status"] == "NOT_REGISTERED_NOT_ADMITTED"
    assert result["motion_space_dimension"] == 2 and result["state_dimension"] == 4
    assert result["terrain_dependency"] == "only-terrain-dependent-comparisons-wait-for-PIRC-17"
    assert "admission" not in spec
    assert len(spec["cells"]) == frozen.manifest()["expected_cells"]
    assert any(c.get("execution_disposition", {}).get("status") == "NOT_IMPLEMENTED" for c in spec["cells"])
    assert before == {p.relative_to(store.path).as_posix(): p.read_bytes()
                      for p in store.path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["row", "arms", "count", "state", "time-unit", "model-dimension",
    "unknown", "configuration-label", "mixture-settings", "source"])
def test_even_rehashed_modified_manifest_cannot_be_a_compiler_output(tmp_path, change):
    path, frozen, connection, _, _ = documents(tmp_path, labelled=True)
    doc = copy.deepcopy(frozen.manifest())
    if change == "row":
        doc["matrix"][0]["cell"]["visibility"] = "test"
    elif change == "arms":
        doc["arms"][0]["budget_seconds"] = 172800
    elif change == "count":
        doc["expected_cells"] += 1
    elif change == "state":
        doc["axis_manifest"]["state_names"].append("z")
    elif change == "time-unit":
        doc["axis_manifest"]["time_unit"] = "days"
    elif change == "model-dimension":
        doc["axis_manifest"]["models"][0]["initial_mean"].append(0.)
    elif change == "unknown":
        doc["authorization"] = "invented"
    elif change == "configuration-label":
        doc["matrix"][0]["configuration_id"] = "foreign"
    elif change == "mixture-settings":
        doc["axis_manifest"]["methods"][1]["mixture_settings"]["state_scales"].append(1.)
    elif change == "source":
        doc["code_hash"] = "0"*64
    path.write_bytes(encode(doc))
    with pytest.raises(ResearchError):
        prepare_study(path, expected_hash=content_hash(doc), runtime_config=connection,
            expected_runtime_config_hash=hashlib.sha256(connection.read_bytes()).hexdigest())


@pytest.mark.parametrize("field,value", [("schema_version", "unknown"), ("runtime_root", "relative"),
    ("store_id", "another-store"), ("contract_version", "unknown"), ("participants", ["PIRC-28"]),
    ("participants", ["PIRC-27", "PIRC-27"]), ("data_source_root", "relative"),
    ("data_binding_status", "authorized-for-everything"), ("pirc38_runtime_baseline", "missing")])
def test_connection_metadata_is_not_permission_or_a_store_alias(tmp_path, field, value):
    path, frozen, connection, config, _ = documents(tmp_path)
    config[field] = value
    connection.write_bytes(encode(config))
    with pytest.raises(ResearchError):
        prepare(path, frozen, connection)


def test_missing_store_is_not_initialized(tmp_path):
    path, frozen, connection, config, _ = documents(tmp_path)
    root = tmp_path / "not-created"
    config["runtime_root"] = str(root)
    connection.write_bytes(encode(config))
    with pytest.raises(ResearchError, match="STORE_MISSING"):
        prepare(path, frozen, connection)
    assert not root.exists()


def test_runtime_in_git_is_not_accepted(tmp_path):
    path, frozen, connection, config, store = documents(tmp_path)
    (store.path.parent / ".git").mkdir()
    with pytest.raises(ResearchError, match="outside Git"):
        prepare(path, frozen, connection)


@pytest.mark.parametrize("target", ["design", "connection", "identity"])
def test_duplicate_json_keys_are_rejected(tmp_path, target):
    path, frozen, connection, _, store = documents(tmp_path)
    selected = {"design": path, "connection": connection, "identity": store.path / "store.json"}[target]
    selected.write_bytes(b'{"schema_version":"first","schema_version":"second"}')
    with pytest.raises(ResearchError):
        prepare(path, frozen, connection)


@pytest.mark.parametrize("target", ["design", "connection"])
def test_wrong_expected_content_hash_is_rejected(tmp_path, target):
    path, frozen, connection, _, _ = documents(tmp_path)
    with pytest.raises(ResearchError, match="hash"):
        prepare_study(path, expected_hash="0"*64 if target == "design" else frozen.manifest_hash,
            runtime_config=connection, expected_runtime_config_hash="0"*64 if target == "connection"
            else hashlib.sha256(connection.read_bytes()).hexdigest())


@pytest.mark.parametrize("value", [None, "", "F"*64])
def test_connection_hash_cannot_be_omitted_or_noncanonical(tmp_path, value):
    path, frozen, connection, _, _ = documents(tmp_path)
    with pytest.raises(ResearchError, match="hash"):
        prepare_study(path, expected_hash=frozen.manifest_hash, runtime_config=connection,
            expected_runtime_config_hash=value)


@pytest.mark.parametrize("target", ["design", "connection", "identity"])
def test_byte_quota_refuses_before_json_decode(tmp_path, monkeypatch, target):
    from experiments.pirc27 import preparation as module
    path, frozen, connection, _, store = documents(tmp_path)
    selected = {"design": path, "connection": connection, "identity": store.path / "store.json"}[target]
    selected.write_bytes(b" "*(module.MAX_CONNECTION_BYTES + 1))
    if target == "design":
        monkeypatch.setattr(module, "MAX_MANIFEST_BYTES", module.MAX_CONNECTION_BYTES)
    with pytest.raises(ResearchError, match="byte quota"):
        prepare(path, frozen, connection)


@pytest.mark.parametrize("raw", [b'{"invalid": NaN}', b'{"invalid": Infinity}', b'\xff', b'['*2000])
def test_invalid_json_is_a_safe_contract_error(tmp_path, raw):
    path, frozen, connection, _, _ = documents(tmp_path)
    path.write_bytes(raw)
    with pytest.raises(ResearchError, match="invalid bounded preparation JSON"):
        prepare(path, frozen, connection)


def test_cli_only_emits_preparation_and_safe_failure_envelopes(tmp_path, capsys):
    from experiments.pirc27.__main__ import main
    path, frozen, connection, _, store = documents(tmp_path)
    arguments = ["prepare", str(path), "--expected-hash", frozen.manifest_hash,
        "--runtime-config", str(connection), "--expected-runtime-config-hash",
        hashlib.sha256(connection.read_bytes()).hexdigest()]
    assert main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["preparation_status"] == "NOT_REGISTERED_NOT_ADMITTED"
    assert not list((store.path / "manifests").iterdir())
    arguments[3] = "0"*64
    assert main(arguments) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["error"]["code"] == "CONTRACT_MISMATCH"
    assert str(path) not in json.dumps(failure)
