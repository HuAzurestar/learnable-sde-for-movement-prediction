"""Saved-number fixtures only; no models fitted or forecast constructors used."""
import copy

import numpy as np
import pytest

from experiments.pirc17.method_noise_description import covariance_description, describe_method, EMBEDDING
from experiments.pirc17.protocol_core import digest


def fixture(kind="seg_constant_mode"):
    n = 1 if kind in {"single_gaussian", "explicit_decomp"} else 3
    columns = 7 if kind == "explicit_decomp" else 4
    model = {"model_kind": kind, "condition_names": ["solar"],
             "weights": [np.zeros((columns, 2)).tolist() for _ in range(n)],
             "covariances": [[[2., 0.], [0., 1.]] for _ in range(n)],
             "mode_probabilities": [1/n] * n, "covariance_scale": 1.7,
             "estimator_drift_fraction": 0.}
    binding = {"fit_identity": "fit", "reference_interval_seconds": 60., "mode_count": n,
               "representative_slot": "arm-01/full", "prediction_configs": ["arm-01/full"],
               "original_model_record_file_sha256": "a" * 64}
    parameter_id = digest({"training_identity_sha256": "training", "model": model})
    binding["parameter_identity"] = parameter_id
    artifact = {"model": model, "training": {"reference_interval_seconds": 60.,
                                             "training_identity_sha256": "training"},
                "dynamics": {"reference_interval_seconds": 60., "noise_embedding": EMBEDDING,
                             "embedding_is_continuous_time_fit_qualification": False,
                             "fit_identity": parameter_id, "model_sha256": digest(model)}}
    return artifact, binding


def rebind(a):
    a["dynamics"]["model_sha256"] = digest(a["model"])
    a["dynamics"]["fit_identity"] = digest({
        "training_identity_sha256": a["training"]["training_identity_sha256"], "model": a["model"]})


def test_tau_embedding_does_not_rescale_saved_covariance_again():
    a, b = fixture()
    d = describe_method(a, b)
    assert d["modes"][0]["saved_rate_eigenvalues_m2_per_s2"] == [1., 2.]
    assert d["modes"][0]["embedded_diffusion_eigenvalues_m2_per_s"] == [60., 120.]
    assert d["selected_covariance_scale_already_in_saved_R"] == 1.7
    assert d["diffusion_eigenvalue_envelope_m2_per_s"] == [60., 120.]
    assert not d["per_role_mode_counts_retained_in_saved_artifact"]
    assert not d["final_probabilities_multiplied_by_exposure_label_to_invent_counts"]
    assert "weights" not in str(d)


@pytest.mark.parametrize("kind,count", [("seg_constant_mode", 3), ("pointwise_mixture", 3),
                                       ("gmm_kernel", 3), ("single_gaussian", 1), ("explicit_decomp", 1)])
def test_each_registered_mode_family(kind, count):
    a, b = fixture(kind)
    assert len(describe_method(a, b)["modes"]) == count


@pytest.mark.parametrize("q", [[[1.]], [[-1., 0.], [0., 1.]],
                             [[1., .01], [0., 1.]], [[float("nan"), 0.], [0., 1.]]])
def test_bad_covariances_rejected(q):
    with pytest.raises(ValueError):
        covariance_description(q, 60.)


@pytest.mark.parametrize("tau", [0, -1, True, float("inf")])
def test_invalid_tau_rejected(tau):
    with pytest.raises(ValueError):
        covariance_description([[1., 0.], [0., 2.]], tau)


def test_negative_roundoff_description_is_not_a_positive_floor():
    d = covariance_description([[-1e-15, 0.], [0., 1.]], 60.)
    assert d["bind_time_negative_roundoff_repair_required"]
    assert d["saved_rate_eigenvalues_m2_per_s2"][0] < 0
    assert d["bound_rate_eigenvalues_m2_per_s2"] == [0., 1.]
    assert d["embedded_diffusion_eigenvalues_m2_per_s"] == [0., 60.]


@pytest.mark.parametrize("field,value", [("reference_interval_seconds", 5.), ("fit_identity", "other"),
                                       ("model_sha256", "b" * 64), ("noise_embedding", "Q=hR"),
                                       ("embedding_is_continuous_time_fit_qualification", True)])
def test_bad_dynamics_binding(field, value):
    a, b = fixture()
    a["dynamics"][field] = value
    with pytest.raises(ValueError, match="binding"):
        describe_method(a, b)


@pytest.mark.parametrize("probabilities", [[.3, .3, .3], [-.1, .5, .6], [1.], [float("inf"), 0., 0.]])
def test_invalid_probability_vectors(probabilities):
    a, b = fixture()
    a["model"]["mode_probabilities"] = probabilities
    # Infinite JSON is itself forbidden before a numerical report is made.
    with pytest.raises(ValueError):
        rebind(a)
        b["parameter_identity"] = a["dynamics"]["fit_identity"]
        describe_method(a, b)


def test_inventory_mode_shape_mismatch():
    a, b = fixture()
    b["mode_count"] = 1
    with pytest.raises(ValueError, match="mode count"):
        describe_method(a, b)


def test_saved_coefficient_shape_is_checked_but_not_exported():
    a, b = fixture()
    a["model"]["weights"][0] = [[0., 0.]]
    rebind(a)
    b["parameter_identity"] = a["dynamics"]["fit_identity"]
    with pytest.raises(ValueError, match="coefficient shape"):
        describe_method(a, b)


def test_input_numbers_are_not_mutated():
    a, b = fixture()
    original = copy.deepcopy(a)
    describe_method(a, b)
    assert a == original
