"""Typed state refusal before allocation, and actual frozen buffer semantics."""

from copy import deepcopy
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from estimation.phase_space_checkpoint import decode_state, encode_state, rng_state
from infrastructure.pirc26_checkpoint_contract import CheckpointContractError, inspect_checkpoint, normalizer_buffers
from models.phase_space import (AffineAccelerationDrift, ModelContractError, NeuralResidualAccelerationDrift,
                               PhaseSpaceSDE, RBFResidualDrift, SplineResidualDrift)
from tests.test_pirc26_checkpoint_preflight import seal
from tests.test_pirc26_dynamics import model, spec


@pytest.mark.parametrize("kind,dtype,shape,data", [
    ("tensor", "float64", [2, 2], [1., 2., 3., 4.]),
    ("tensor", "float64", [2, 2], [[1., 2., 3.], [4.]]),
    ("tensor", "float64", [2, 0], []),
    ("tensor", "float64", [], [1.]),
    ("array", "uint32", [2], [-1, 1]),
    ("array", "uint32", [2], [1, 1 << 32]),
    ("tensor", "uint8", [1], [256]),
    ("tensor", "int32", [1], [1 << 31]),
    ("tensor", "int64", [1], [1 << 63]),
    ("tensor", "int64", [1], [1.]),
    ("array", "int64", [1], [True]),
    ("tensor", "bool", [1], [1]),
    ("tensor", "float64", [1], [True]),
    ("tensor", "float32", [1], [1e40]),
    ("tensor", "float32", [1], [1e-100]),
    ("array", "float32", [1], [.1]),
    ("tensor", "float64", [1], [(1 << 53)+1]),
    ("tensor", "float64", [1], [1]),
    ("tensor", "float64", [1], ["1"]),
    ("array", "float64", [1], [float("inf")]),
    ("tensor", "float16", [0], []),
])
def test_typed_semantic_refusals_precede_both_numerical_array_engines(monkeypatch, kind, dtype, shape, data):
    def allocation(*args, **kwargs):
        pytest.fail("malformed checkpoint reached numerical allocation")
    monkeypatch.setattr(torch, "tensor", allocation)
    monkeypatch.setattr(np, "asarray", allocation)
    with pytest.raises(ModelContractError):
        decode_state({kind: {"dtype": dtype, "shape": shape, "data": data}})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.int32, torch.uint8, torch.bool])
@pytest.mark.parametrize("shape", [(), (2, 0), (0, 3), (2, 2)])
def test_actual_tensor_dtypes_scalar_and_empty_nesting_round_trip(dtype, shape):
    source = torch.zeros(shape, dtype=dtype)
    encoded = encode_state(source)
    restored = decode_state(json.loads(json.dumps(encoded)))
    assert restored.shape == source.shape and restored.dtype == dtype and torch.equal(source, restored)


def test_actual_rng_and_numpy_signed_unsigned_extremes_remain_exact():
    state = rng_state()
    assert encode_state(decode_state(encode_state(state))) == encode_state(state)
    for dtype, values in [("uint32", [0, (1 << 32)-1]), ("int64", [-(1 << 63), (1 << 63)-1]),
                          ("float32", [np.float32(.1), -0.]), ("float64", [.1, -0.])]:
        source = np.asarray(values, dtype=dtype)
        restored = decode_state(encode_state(source))
        assert source.dtype == restored.dtype and source.tobytes() == restored.tobytes()


@pytest.mark.parametrize("family", ["M1-R", "M1-S", "M2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_resigned_zero_normalization_refused_before_model_or_tensor(monkeypatch, family, dtype):
    cp = deepcopy(model(family).to(dtype=dtype).checkpoint())
    cp["model_card"]["spec"]["scales"][0] = 1e-100
    cp["state"]["acceleration_model.scales"]["data"][0] = 0.
    seal(cp)
    def allocation(*args, **kwargs):
        pytest.fail("zero normalization reached constructor/tensor")
    monkeypatch.setattr(AffineAccelerationDrift, "__init__", allocation)
    monkeypatch.setattr(torch, "tensor", allocation)
    with pytest.raises(CheckpointContractError, match="positive normalization"):
        inspect_checkpoint(cp)
    with pytest.raises(ModelContractError, match="positive normalization"):
        PhaseSpaceSDE.from_checkpoint(cp)


@pytest.mark.parametrize("family", ["M1-R", "M1-S", "M2"])
@pytest.mark.parametrize("fault", ["zero-scale", "overflow-scale", "overflow-mean"])
def test_direct_residual_constructor_refuses_before_buffer_allocation(monkeypatch, family, fault):
    values = [1.]*4 if fault != "overflow-mean" else [0.]*4
    values[0] = 1e-100 if fault == "zero-scale" else 1e100
    recipe = replace(spec(), **{"means" if fault == "overflow-mean" else "scales": values})
    affine = AffineAccelerationDrift()
    def allocation(*args, **kwargs):
        pytest.fail("invalid normalizer allocated a residual buffer/layer")
    monkeypatch.setattr(torch, "tensor", allocation)
    with pytest.raises(ModelContractError):
        if family == "M1-R":
            RBFResidualDrift(affine, recipe, [0], [[0.]], [1.])
        elif family == "M1-S":
            SplineResidualDrift(affine, recipe, [0], [[0., 0., 1., 1.]], degree=1)
        else:
            NeuralResidualAccelerationDrift(affine, recipe, [0], hidden=(16,), seed=12)


def test_frozen_float32_representation_is_explicit_under_double_default():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        recipe = replace(spec(), means=(.1,)*4, scales=(.3,)*4)
        drift = NeuralResidualAccelerationDrift(AffineAccelerationDrift(), recipe, [0], hidden=(16,), seed=12)
        m = PhaseSpaceSDE(drift, [[.4, 0.], [0., .3]], recipe).double()
        buffers = normalizer_buffers(recipe.means, recipe.scales)
        assert m.acceleration_model.scales.tolist() == buffers["scales"]
        assert m.acceleration_model.means.tolist() == buffers["means"]
        assert PhaseSpaceSDE.from_checkpoint(m.checkpoint()).checkpoint() == m.checkpoint()
    finally:
        torch.set_default_dtype(previous)


def test_cold_inspector_normalizer_underflow_refusal_does_not_import_numerical_engines():
    cp = deepcopy(model("M2").double().checkpoint())
    cp["model_card"]["spec"]["scales"][0] = 1e-100
    cp["state"]["acceleration_model.scales"]["data"][0] = 0.
    seal(cp)
    program = '''
import json,sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy'}:
            raise AssertionError('numerical import during refusal')
sys.meta_path.insert(0, Block())
from infrastructure.pirc26_checkpoint_contract import inspect_checkpoint, CheckpointContractError
try:
    inspect_checkpoint(json.load(sys.stdin))
except CheckpointContractError as error:
    assert 'positive normalization' in str(error)
else:
    raise AssertionError('zero normalizer accepted')
'''
    completed = subprocess.run([sys.executable, "-c", program], input=json.dumps(cp), capture_output=True,
                               text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_training_structure_refused_before_json_materialization(monkeypatch):
    import infrastructure.research_control as control
    def materialize(*args, **kwargs):
        pytest.fail("oversized declaration reached json.dumps")
    monkeypatch.setattr(control.json, "dumps", materialize)
    for value in ({"k"*129: 1}, [None]*65537, {"map": [["x", "x"*65537]]}):
        with pytest.raises(ModelContractError, match="RESOURCE_PLAN_REJECTED"):
            decode_state(value)


@pytest.mark.parametrize("fault", ["array", "key", "tag", "tuple"])
def test_entire_state_preflight_refuses_late_fault_before_first_valid_array(monkeypatch, fault):
    value = {"map": [["first", encode_state(torch.ones(2))], ["last", 1]]}
    if fault == "array":
        value["map"][1][1] = {"tensor": {"dtype": "int64", "shape": [1], "data": [1.5]}}
    elif fault == "key":
        value["map"][1][0] = True
    elif fault == "tag":
        value["map"][1][1] = {"unknown": [], "tensor": {}}
    else:
        value["map"][1][1] = {"tuple": 1}
    def allocation(*args, **kwargs):
        pytest.fail("late invalid field caused an earlier array allocation")
    monkeypatch.setattr(torch, "tensor", allocation)
    monkeypatch.setattr(np, "asarray", allocation)
    with pytest.raises(ModelContractError):
        decode_state(value)


@pytest.mark.parametrize("key", [True, .5, (1, 2)])
def test_encoder_refuses_map_keys_outside_closed_decoder_contract(key):
    with pytest.raises(ModelContractError, match="map key"):
        encode_state({key: 1})


@pytest.mark.parametrize("dtype,content", [(torch.float32, 1e-100), (torch.float32, .1),
                                          (torch.float64, (1 << 53)+1)])
def test_resigned_model_tensor_refuses_lossy_representation_before_allocation(monkeypatch, dtype, content):
    cp = deepcopy(model("M0").to(dtype=dtype).checkpoint())
    cp["state"]["velocity_factor"]["data"][0][0] = content
    seal(cp)
    def allocation(*args, **kwargs):
        pytest.fail("lossy model representation reached constructor")
    monkeypatch.setattr(AffineAccelerationDrift, "__init__", allocation)
    monkeypatch.setattr(torch, "tensor", allocation)
    with pytest.raises(ModelContractError, match="dtype representation"):
        PhaseSpaceSDE.from_checkpoint(cp)
