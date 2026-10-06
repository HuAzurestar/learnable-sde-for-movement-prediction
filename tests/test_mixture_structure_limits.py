"""Refuse invalid structure before copying/serializing, not after allocation."""

from dataclasses import replace

import pytest

import domain.mixture as mixture_module
import experiments.pirc27.design as design_module
from domain.errors import DataValidationError
from domain.mixture import MixturePolicy, MixtureSettings
from experiments.pirc27.design import StudyMethod, freeze_design
from tests.test_propagation_study_design import fixture
from tests.test_propagation_methods import inputs
from experiments.pirc25.affine import code_hash


def forbid_copy(*args, **kwargs):
    pytest.fail("invalid mixture structure reached dataclass copy/serialization")


@pytest.mark.parametrize("changes", [{"state_scales": (1.,)*5},
    {"state_scales": [1.,]*4}, {"state_scales": ((1.,)*100,)*4},
    {"component_cap": True}, {"maximum_work_units": 1_000_001},
    {"merge_distance": {"unexpected": [0.]*100}}])
def test_invalid_design_settings_refused_before_any_dataclass_serialization(monkeypatch, changes):
    settings = replace(MixtureSettings(8, 0., 0., (1.,)*4, 0., 1_000_000, 60.), **changes)
    design = fixture(methods=(StudyMethod("mixture", steps=2, samples=8,
        recovery=True, mixture_settings=settings),))
    monkeypatch.setattr(design_module, "asdict", forbid_copy)
    with pytest.raises(DataValidationError):
        freeze_design(design)


@pytest.mark.parametrize("field,value", [("state_scales", [1.]*5),
    ("state_scales", [[1.]*100]*4), ("component_cap", True),
    ("merge_distance", {"unexpected": [0.]*100}), ("maximum_job_seconds", 7201)])
def test_invalid_policy_manifest_refused_before_canonical_copy(monkeypatch, field, value):
    document = MixturePolicy("a"*64, "b"*64, "c"*64,
        8, 0., 0., (1.,)*4, 0., 1_000_000, 60.).manifest()
    document[field] = value
    monkeypatch.setattr(mixture_module, "asdict", forbid_copy)
    with pytest.raises(DataValidationError):
        MixturePolicy.from_manifest(document)


def test_invalid_live_policy_manifest_refused_before_dataclass_copy(monkeypatch):
    policy = MixturePolicy("a"*64, "b"*64, "c"*64,
        8, 0., 0., (1.,)*5, 0., 1_000_000, 60.)
    monkeypatch.setattr(mixture_module, "asdict", forbid_copy)
    with pytest.raises(DataValidationError):
        policy.manifest()


@pytest.mark.parametrize("changes", [{"state_scales": (1.,)*5},
    {"state_scales": ((1.,)*100,)*4}, {"component_cap": True}, {"maximum_job_seconds": 7201}])
def test_direct_settings_binding_refuses_before_copy(monkeypatch, changes):
    package, request = inputs(steps=2)
    source = code_hash()
    settings = replace(MixtureSettings(8, 0., 0., (1.,)*4, 0., 1_000_000, 60.), **changes)
    monkeypatch.setattr(mixture_module, "asdict", forbid_copy)
    with pytest.raises(DataValidationError):
        settings.bind(package, request, source)
