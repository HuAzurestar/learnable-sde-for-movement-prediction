"""Literal-source and saved-metadata fixtures only: no fitting or SDE simulation."""
import pytest

from experiments.pirc17.method_algorithm_description import CONSTANTS, MODEL_FIELDS, describe_model, source_constants


def test_literal_algorithm_defaults_without_importing_fitter():
    source = "\n".join(f"{name} = {i + 1}" for i, name in enumerate(CONSTANTS))
    source += "\ndef _blend_models(a, b, *, finetune, weight=0.65):\n raise RuntimeError('never called')\n"
    values = source_constants(source)
    assert values["target_adaptation_weight"] == .65
    assert values["REPTILE_INNER_STEPS"] == CONSTANTS.index("REPTILE_INNER_STEPS")+1


def test_missing_literal_default_rejected():
    with pytest.raises(ValueError, match="complete literal"):
        source_constants("RIDGE = 1e-6")


def test_projection_excludes_parameters_and_private_fields():
    source = {name: 0 for name in MODEL_FIELDS}
    source.update(weights="private", covariances="private", private_path="private")
    assert set(describe_model(source)) == set(MODEL_FIELDS)


@pytest.mark.parametrize("value", [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_saved_diagnostic_rejected(value):
    source = {name: 0 for name in MODEL_FIELDS}
    source['meta_inner_objective_before'] = value
    with pytest.raises(ValueError, match="finite saved"):
        describe_model(source)
