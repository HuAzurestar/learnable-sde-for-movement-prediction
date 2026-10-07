"""Primitive design refusal precedes copying, tensors and package evaluation."""

from dataclasses import replace

import pytest

import experiments.pirc27.design as module
import inference.affine_oracle as oracle
from domain.errors import DataValidationError
from infrastructure.research_store import ResearchError
from experiments.pirc27.design import freeze_design
from tests.test_propagation_study_design import fixture


def forbidden(*args, **kwargs):
    pytest.fail("invalid design reached copying, tensor allocation or package evaluation")


MODEL_CHANGES = [
    {"initial_mean": (0.,)*5}, {"initial_mean": [0.]*4},
    {"initial_mean": ((0.,)*100,)*4}, {"initial_mean": (True, 0., 0., 0.)},
    {"initial_covariance": ((0.,)*4,)*5},
    {"initial_covariance": ((0.,)*100,)*4},
    {"initial_covariance": [[0.]*4]*4},
    {"initial_covariance": ((float("nan"), 0., 0., 0.),)+( (0.,)*4,)*3},
]
METHOD_CHANGES = [
    {"steps": True}, {"steps": 8193}, {"steps": [1]*100},
    {"samples": 1}, {"samples": 1_000_001}, {"samples": {"nested": [1]*100}},
    {"chunk_size": True}, {"chunk_size": 257}, {"chunk_size": (1,)*100},
]
FUNCTIONAL_CHANGES = [
    {"kind": ["endpoint-x"]}, {"kind": "unknown"},
    {"normal": (1.,)*5}, {"normal": [1.]*4}, {"normal": ((1.,)*100,)*4},
    {"normal": (0.,)*4}, {"threshold": {"nested": [1]*100}},
    {"closed": 1}, {"tolerance": float("inf")}, {"tolerance": 0.},
]


@pytest.mark.parametrize("group,changes", [
    *(('models', changes) for changes in MODEL_CHANGES),
    *(('methods', changes) for changes in METHOD_CHANGES),
    *(('functionals', changes) for changes in FUNCTIONAL_CHANGES),
])
def test_invalid_primitives_refused_before_any_copy_or_numeric_allocation(monkeypatch, group, changes):
    design = fixture()
    values = getattr(design, group)
    design = replace(design, **{group: (replace(values[0], **changes),)+values[1:]})
    monkeypatch.setattr(module, "asdict", forbidden)
    monkeypatch.setattr(oracle, "_covariance", forbidden)
    monkeypatch.setattr(module, "validate_oracle_input", forbidden)
    with pytest.raises(DataValidationError):
        freeze_design(design)


@pytest.mark.parametrize("field", ["arm_id", "model_family_id", "method_family_id", "objective_id"])
def test_invalid_later_arm_refused_before_copying_any_arm(monkeypatch, field):
    design = fixture()
    arms = design.arms[:-1]+(replace(design.arms[-1], **{field: ["invalid"]*100}),)
    monkeypatch.setattr(module, "asdict", forbidden)
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(replace(design, arms=arms))


@pytest.mark.parametrize("group,field", [("models", "initial_mean"),
    ("models", "initial_covariance"), ("methods", "steps"),
    ("functionals", "normal"), ("functionals", "kind"), ("arms", "objective_id")])
def test_cyclic_values_produce_contract_errors_not_recursion_or_type_errors(group, field):
    cyclic = []
    cyclic.append(cyclic)
    design = fixture()
    values = getattr(design, group)
    design = replace(design, **{group: (replace(values[0], **{field: cyclic}),)+values[1:]})
    with pytest.raises((DataValidationError, ResearchError)):
        freeze_design(design)
