"""Actual codec shapes/capacity and refusal order, using synthetic models."""

from copy import deepcopy
import json
import subprocess
import sys

import pytest
import torch

from application.pirc26_components import model_factory, construct_components, preflight_model
from application.pirc26_runtime import validate_job
from infrastructure.pirc26_checkpoint_contract import inspect_checkpoint, identity, CheckpointContractError
from infrastructure.research_store import ResearchError
from models.phase_space import (AffineAccelerationDrift, DynamicsSpec, ModelContractError,
    PhaseSpaceSDE, RBFResidualDrift, SplineResidualDrift, NeuralResidualAccelerationDrift)
from tests.test_pirc26_components_data import declarations
from tests.test_pirc26_dynamics import model


def seal(cp):
    for item in cp["state"].values():
        item["sha256"] = identity({k: v for k, v in item.items() if k != "sha256"})
    cp["sha256"] = identity({k: v for k, v in cp.items() if k != "sha256"})
    return cp


def maximum_model(family, dtype):
    c = 128
    spec = DynamicsSpec("synthetic-max-local", "a"*64, "b"*64, "c"*64, c, (0.,)*(4+c), (1.,)*(4+c))
    affine, features = AffineAccelerationDrift(c), list(range(8))
    if family == "M1-R":
        drift = RBFResidualDrift(affine, spec, features, [[i/128.]*8 for i in range(128)], [1.]*128)
    elif family == "M1-S":
        knots = [0.]*4 + [i/13 for i in range(1, 13)] + [1.]*4
        drift = SplineResidualDrift(affine, spec, features, [knots]*8, degree=3)
    elif family == "M2":
        drift = NeuralResidualAccelerationDrift(affine, spec, features, hidden=(32, 32), seed=12)
    else:
        drift = affine
    return PhaseSpaceSDE(drift, [[.4, 0.], [.1, .3]], spec).to(dtype=dtype)


@pytest.mark.parametrize("family", ["M0", "M1-R", "M1-S", "M2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_maximum_registered_architecture_is_derived_without_trusting_tensor_shapes(family, dtype):
    m = maximum_model(family, dtype)
    cp = m.checkpoint()
    checked = inspect_checkpoint(cp, component_limit=128)
    assert checked["tensor_elements"] == sum(t.numel() for t in m.state_dict().values())
    assert checked["component_capacity"] == 128
    assert checked["shapes"] == {k: list(t.shape) for k, t in m.state_dict().items()}
    assert PhaseSpaceSDE.from_checkpoint(cp).checkpoint() == cp
    with pytest.raises(CheckpointContractError, match="RESOURCE_PLAN_REJECTED"):
        inspect_checkpoint(cp, component_limit=127)


@pytest.mark.parametrize("fault", ["large-shape", "ragged", "extra-elements", "empty-row", "dtype", "tensor-extra-key",
    "unknown-parameter", "scalar-bool", "mlp-width", "feature-range", "bandwidth-zero", "basis-mismatch", "spline-quota", "nonfinite"])
def test_resigned_malformed_recipes_are_rejected_before_constructor_or_tensor(monkeypatch, fault):
    family = "M1-R" if fault in ("bandwidth-zero", "basis-mismatch") else "M1-S" if fault == "spline-quota" else "M2" if fault in ("mlp-width", "feature-range") else "M0"
    cp = deepcopy(model(family).checkpoint())
    item = cp["state"]["velocity_factor"]
    if fault == "large-shape":
        item["shape"] = [100000, 100000]
    elif fault == "ragged":
        item["data"][0].pop()
    elif fault == "extra-elements":
        item["data"][0].append(0.)
    elif fault == "empty-row":
        cp["state"]["acceleration_model.context_weight"]["data"] = []
    elif fault == "dtype":
        item["dtype"] = "torch.int64"
    elif fault == "tensor-extra-key":
        item["extra"] = 1
    elif fault == "unknown-parameter":
        cp["state"]["undeclared"] = deepcopy(item)
    elif fault == "scalar-bool":
        item["data"][0][0] = True
    elif fault == "mlp-width":
        cp["configuration"]["hidden"] = [100000, 100000]
    elif fault == "feature-range":
        cp["configuration"]["features"] = [128]
    elif fault == "bandwidth-zero":
        cp["configuration"]["bandwidth"][0] = 0.
    elif fault == "basis-mismatch":
        cp["state"]["acceleration_model.centers"]["data"][0][0] = .5
    elif fault == "spline-quota":
        cp["configuration"]["knots"] = [[-2.]*4 + [0.]*128 + [2.]*4]
    else:
        item["data"][0][0] = float("nan")
    if fault != "nonfinite":
        seal(cp)
    def allocation(*args, **kwargs):
        pytest.fail("refusal allocated a model or tensor")
    monkeypatch.setattr(AffineAccelerationDrift, "__init__", allocation)
    for name in ("tensor", "as_tensor", "zeros"):
        monkeypatch.setattr(torch, name, allocation)
    with pytest.raises(ModelContractError):
        PhaseSpaceSDE.from_checkpoint(cp)


def test_actual_factories_and_whole_composition_refuse_declared_workspace_before_first_factory(monkeypatch):
    m = model("M2")
    config, profile, registries, bindings, adapter, _ = declarations(m)
    # Make valid bindings for an undersized declared component workspace.
    from application.pirc26_components import component_bindings
    profile["components"] = 4
    bindings = component_bindings(config, profile, matrix_cells=1, registries=registries)
    def allocated(*args, **kwargs):
        pytest.fail("undersized workspace invoked a factory")
    for registry in registries.values():
        monkeypatch.setattr(registry, "create_bound", allocated)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        construct_components(adapter, bindings, matrix_cells=1, seed=12, registries=registries, initial_checkpoint=m.checkpoint())
    monkeypatch.setattr(PhaseSpaceSDE, "from_checkpoint", allocated)
    with pytest.raises(ResearchError, match="RESOURCE_PLAN_REJECTED"):
        model_factory(config, profile, {"initial_checkpoint": m.checkpoint()})


@pytest.mark.parametrize("fault", ["capacity", "configuration", "dtype", "initializer"])
def test_owner_job_refuses_model_recipe_before_any_physical_read(tmp_path, monkeypatch, fault):
    from tests.test_pirc26_runtime import prepare, owner_admit
    store, value, plugin, job, _, _, _ = prepare(tmp_path)
    _, receipt = owner_admit(store, value, plugin)
    cp = deepcopy(job["initial_checkpoint"])
    cfg = receipt["cell"]["execution"]["config"]
    if fault == "capacity":
        receipt["cell"]["execution"]["inputs"]["components"] = 1
    elif fault == "configuration":
        cfg["configuration_hash"] = "f"*64
    elif fault == "dtype":
        receipt["cell"]["execution"]["inputs"]["dtype"] = "float32"
    else:
        # M0 job changes family to M2 without matching the actual recipe.
        cfg["family"] = "M2"
    job["initial_checkpoint"] = cp
    before = list(store.events())
    monkeypatch.setattr(PhaseSpaceSDE, "from_checkpoint", lambda *a, **kw: pytest.fail("owner preflight allocated model"))
    with pytest.raises(ResearchError):
        validate_job(job, receipt)
    assert list(store.events()) == before


def test_actual_factory_workspace_refusal_is_cold_and_engine_free():
    m = model("M2")
    config, profile, _, _, _, _ = declarations(m)
    profile["components"] = 1
    program = r'''
import importlib.abc, json, sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy'}:
            raise AssertionError('unneeded numerical import: ' + fullname)
sys.meta_path.insert(0, NoEngine())
from application.pirc26_components import model_factory
from infrastructure.research_store import ResearchError
data = json.load(sys.stdin)
try:
    model_factory(data['config'], data['profile'], {'initial_checkpoint': data['checkpoint']})
except ResearchError as exc:
    assert exc.code == 'RESOURCE_PLAN_REJECTED'
else:
    raise AssertionError('undersized model admitted')
assert not {'torch','numpy','scipy'}.intersection(sys.modules)
print('preflight-ok')
'''
    process = subprocess.run([sys.executable, "-B", "-c", program],
        input=json.dumps({"config": config, "profile": profile, "checkpoint": m.checkpoint()}), text=True,
        capture_output=True, timeout=40)
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == "preflight-ok"


def test_structural_quota_precedes_json_materialization(monkeypatch):
    import infrastructure.research_control as control
    bad = {"oversized": [0]*65536}
    monkeypatch.setattr(control.json, "dumps", lambda *a, **kw: pytest.fail("oversized JSON was materialized"))
    with pytest.raises(CheckpointContractError, match="RESOURCE_PLAN_REJECTED"):
        inspect_checkpoint(bad)


def test_model_and_training_source_identities_bind_the_allocation_free_codec():
    from application.pirc26_dynamics import dynamics_identity
    from estimation.phase_space_checkpoint import training_scope
    from estimation.phase_space import O1Plan
    files = dynamics_identity()["files"]
    for name in ("infrastructure/pirc26_checkpoint_contract.py", "infrastructure/research_control.py"):
        assert name in files
    # The training source list is the one actually used by training_scope.
    import estimation.phase_space_checkpoint as checkpoint
    assert set(files).intersection(checkpoint.SOURCE_FILES) >= {
        "infrastructure/pirc26_checkpoint_contract.py", "infrastructure/research_control.py"}
    assert training_scope(model("M0"), O1Plan(), [], "O1")["code_hash"]
