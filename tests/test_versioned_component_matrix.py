"""Execute every declared CPU single-axis sealed engine/dtype combination.

These synthetic engineering fixtures do not qualify research algorithms. The
managed one/four-axis affine adapters have separate fixed float64 contracts.
The existing exact/float64 regression and its tolerances remain unchanged.
"""

from dataclasses import asdict
import json
import math

import pytest
import torch

from application.experiment import ExperimentApplication
from application.synthetic import make_synthetic_em_data
from domain import ForecastRequest, ModelContext
from infrastructure.research_store import digest
import registry as root
from tests.test_versioned_component_factories import config, profile


ENGINES = ('exact', 'euler', 'J1_split', 'J3_CRN')
# Fixed before the first execution, not fitted to observed results. Float64
# retains the original legacy-comparison tolerance; the newly covered float32
# fixture uses single-precision comparison bounds, not float64's precision.
POST_FIT_TOLERANCES = {
    'float64': {'rtol': 1e-10, 'atol': 1e-12},
    'float32': {'rtol': 1e-5, 'atol': 1e-6},
}


def assert_forecast(actual, expected, dtype, *, tolerance):
    # The existing exact/CRN API publishes terminal sample covariance, not a
    # horizon-indexed covariance tensor. Euler/split leave this field absent.
    shapes = {'samples': (8, 2, 2), 'mean': (2, 2), 'covariance': (2, 2)}
    assert actual.metadata == expected.metadata
    for name, shape in shapes.items():
        value, reference = getattr(actual, name), getattr(expected, name)
        if reference is None:
            assert value is None
            continue
        assert value is not None and value.shape == reference.shape == shape
        assert value.dtype == reference.dtype == dtype
        assert value.device.type == reference.device.type == 'cpu'
        assert torch.isfinite(value).all() and torch.isfinite(reference).all()
        torch.testing.assert_close(value, reference, **tolerance)


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_supported_sealed_engine_dtype_preserves_actual_training_and_prediction(
        tmp_path, engine, dtype_name):
    cfg, inputs = config(), profile()
    cfg.components.inference = engine
    cfg.dtype = inputs['dtype'] = dtype_name
    dtype = {'float32': torch.float32, 'float64': torch.float64}[dtype_name]
    tolerance = POST_FIT_TOLERANCES[dtype_name]
    bindings = root.component_bindings(cfg, inputs, matrix_cells=1)
    plan = root.plan_components(cfg, bindings, matrix_cells=1)
    data, _ = make_synthetic_em_data(n_segments=4, length=10, dt=1.0, seed=cfg.seed)
    request = ForecastRequest(torch.tensor([1.0, 0.25], dtype=dtype),
        torch.tensor([1.0, 2.0], dtype=dtype), 8, ModelContext(regime=0))
    fixture = {'config': asdict(cfg), 'profile': inputs,
        'segments': [segment.tolist() for segment in data.segments],
        'dts': data.dts, 'initial_state': request.initial_state.tolist(),
        'horizons': request.horizons.tolist(), 'n_samples': request.n_samples,
        'regime': 0, 'post_fit_tolerance': tolerance}
    (tmp_path / 'matrix-fixture.json').write_text(
        json.dumps({'fixture_hash': digest(fixture), 'fixture': fixture}, indent=2),
        encoding='utf-8')
    assert {role: binding['component_id'] for role, binding in bindings.items()} == {
        'model': 'I1', 'trainer': 'EM', 'predictor': engine}
    assert all(binding['component_version'] == '1.0.0' for binding in bindings.values())
    for role, entry in plan['entries'].items():
        assert entry['state_order'] == ['x', 'vx'] and entry['units'] == ['m', 'm/s']
        assert entry['resource_class'] == 'cpu'
        assert bindings[role]['registry_entry_hash'] == digest(entry)
    versioned = ExperimentApplication.from_config(cfg, component_bindings=bindings, matrix_cells=1)
    legacy = ExperimentApplication.from_config(cfg)
    initial_parameters = {name: value.detach().clone() for name, value in versioned.model.state_dict().items()}
    assert versioned.model.state_dim == legacy.model.state_dim == 2
    assert versioned.model.noise_dim == legacy.model.noise_dim == 1
    assert_forecast(versioned.predict(versioned.model, request),
        legacy.predict(legacy.model, request), dtype, tolerance={'rtol': 0, 'atol': 0})
    # Run BOTH real trainers. Neither parameters nor fit results are borrowed
    # from the other pipeline, and no engine/factory/RNG result is substituted.
    actual_fit, legacy_fit = versioned.train(data).fit, legacy.train(data).fit
    assert actual_fit.iterations == legacy_fit.iterations and 0 < actual_fit.iterations <= 2
    assert actual_fit.converged == legacy_fit.converged
    assert len(actual_fit.objective_history) == len(legacy_fit.objective_history) == actual_fit.iterations
    assert all(math.isfinite(value) for value in actual_fit.objective_history + legacy_fit.objective_history)
    actual_state, legacy_state = versioned.model.state_dict(), legacy.model.state_dict()
    assert actual_state.keys() == legacy_state.keys() == initial_parameters.keys()
    assert any(not torch.equal(value, initial_parameters[name]) for name, value in actual_state.items())
    for name, value in actual_state.items():
        assert value.dtype == legacy_state[name].dtype == dtype
        assert torch.isfinite(value).all() and torch.isfinite(legacy_state[name]).all()
        torch.testing.assert_close(value, legacy_state[name], **tolerance)
    torch.testing.assert_close(torch.tensor(actual_fit.objective_history, dtype=dtype),
        torch.tensor(legacy_fit.objective_history, dtype=dtype), **tolerance)
    actual, expected = versioned.predict(versioned.model, request), legacy.predict(legacy.model, request)
    assert_forecast(actual, expected, dtype, tolerance=tolerance)
    assert versioned.component_plan == plan
    (tmp_path / 'matrix-result.json').write_text(json.dumps({
        'schema_version': 'pirc38-versioned-fixture-matrix-v1', 'fixture_hash': digest(fixture),
        'component_plan_hash': plan['component_plan_hash'], 'bindings': bindings,
        'actual_fit': actual_fit.to_dict(), 'legacy_fit': legacy_fit.to_dict(),
        'sample_max_abs_difference': float((actual.samples - expected.samples).abs().max().detach()),
        'actual_samples_hash': digest(actual.samples.tolist()),
        'legacy_samples_hash': digest(expected.samples.tolist()),
        'shape': list(actual.samples.shape), 'dtype': dtype_name, 'device': 'cpu',
        'engine_metadata': dict(actual.metadata)}, indent=2), encoding='utf-8')
