"""Bounded one-mode analytic diagnostics; no MC sampling or source authority.

The producer and verifier use separate Kalman update implementations. They share
the admitted conditional-affine transition, so this is a filtering boundary
control, not an independent proof of that transition or scientific qualification.
"""

import math

import numpy as np

from infrastructure.research_store import digest, encode
from inference.switching import InferenceSpec
from models.switching import SwitchingError, SwitchingSDEModel
from .approximation import ApproximationSpec
from .approximation_checks import _initial, _known_mode_checks, observation_prefix
from .approximation_evidence import _boundary, _keys, _map, _require, _same_declaration

SCHEMA = 'pirc28-k1-affine-boundary-v1'
FLAGS = {'reference_sampling_verified': False, 'managed_evidence_bound': False,
         'qualified_identifiability': False, 'real_data_authorized': False}


def boundary_plan(inference, times, origin_index, paths, observation_dimension):
    """Additive standalone envelope, including retained report/owner copies."""
    if (not isinstance(inference, InferenceSpec) or type(paths) is not int or not 1 <= paths <= 64
            or type(observation_dimension) is not int or not 1 <= observation_dimension <= 4
            or type(origin_index) is not int or not 1 <= origin_index < len(times)
            or origin_index >= 100_000):
        raise SwitchingError('CONTRACT_MISMATCH', 'bounded K1 analytic prefix required')
    inference.validate()
    prefix_times = np.asarray(times[:origin_index+1], dtype=float)
    if (prefix_times.ndim != 1 or not np.isfinite(prefix_times).all()
            or prefix_times[0] != 0 or np.any(np.diff(prefix_times) <= 0)):
        raise SwitchingError('CONTRACT_MISMATCH', 'finite increasing K1 prefix required')
    steps = sum(math.ceil(float(dt)/inference.max_step_seconds) for dt in np.diff(prefix_times))
    if steps > inference.max_substeps:
        raise SwitchingError('INFERENCE_INAPPLICABLE', 'K1 prefix exceeds its original step quota')
    n = len(prefix_times)
    report = 96+paths*16
    elements = 2*steps*16*24+8*paths*n*(observation_dimension+4+16)+3*report
    work = 2*paths*steps
    if elements > 20_000_000 or work > 50_000_000:
        raise SwitchingError('RESOURCE_LIMIT', 'K1 boundary exceeds original allocation/work caps')
    return {'schema_version': 'pirc28-k1-boundary-plan-v1', 'tensor_elements': elements,
        'tensor_bytes': elements*8, 'maximum_filter_steps': work,
        'retained_report_numeric_elements': report, 'owner_report_copy_elements': 2*report,
        'reference_particles': 0, 'independent_batches': 0,
        'resource_admitted': False, 'managed_executor_implemented': False}


def _inputs(observations, model, probability, means, covariances, inference, spec, origin_index, poll):
    _require(callable(poll), 'explicit K1 boundary budget poll required')
    poll()
    _require(isinstance(model, SwitchingSDEModel) and model.n_modes == 1,
             'K1 boundary requires the original one-mode model')
    _require(isinstance(spec, ApproximationSpec), 'canonical boundary specification required')
    # Enforce metadata quota BEFORE copying prefix observations or filtering.
    from .synthetic import ObservationBatch
    _require(type(observations) is ObservationBatch and observations.values.ndim == 3,
             'canonical bounded observation batch required')
    plan = boundary_plan(inference, observations.times, origin_index,
                         len(observations.values), observations.values.shape[-1])
    prefix = observation_prefix(observations, origin_index)
    p, m, c = _initial(model, probability, means, covariances)
    poll()
    return prefix, p, m, c, plan


def _metadata(prefix, model, p, m, c, inference, spec, origin_index, plan):
    return {'schema_version': SCHEMA, 'model_hash': digest(model.to_dict()),
        'observation_prefix_hash': digest(prefix.to_dict()), 'inference_hash': digest(inference.to_dict()),
        'spec_hash': spec.spec_hash, 'origin_index': origin_index,
        'initial_state_hash': digest({'probability': p.tolist(), 'means': m.tolist(), 'covariances': c.tolist()}),
        'resource_plan': plan, 'accuracy': 'one-mode-conditional-affine-Kalman-boundary; not-switching-accuracy',
        **FLAGS}


def check_k1_boundary(observations, model, probability, means, covariances, inference, spec,
                      *, origin_index, check_budget):
    prefix, p, m, c, plan = _inputs(observations, model, probability, means, covariances,
                                  inference, spec, origin_index, check_budget)
    rows = []
    for path in range(len(prefix.values)):
        check_budget()
        rows.append({'path_index': path, 'boundary': _known_mode_checks(prefix, model, m, c,
            inference, spec.to_dict()['known_mode_boundary'], path, check_budget)[0]})
    check_budget()
    return {**_metadata(prefix, model, p, m, c, inference, spec, origin_index, plan),
        'state': 'NUMERICAL_CHECKS_PASSED' if all(row['boundary']['passed'] for row in rows)
                 else 'INFERENCE_INAPPLICABLE', 'paths': rows}


def validate_k1_boundary(report, observations, model, probability, means, covariances, inference, spec,
                         *, origin_index, check_budget):
    """Independent arithmetic; caller supplies original inputs, grants and budget."""
    prefix, p, m, c, plan = _inputs(observations, model, probability, means, covariances,
                                  inference, spec, origin_index, check_budget)
    metadata = _metadata(prefix, model, p, m, c, inference, spec, origin_index, plan)
    _keys(report, {*metadata, 'state', 'paths'}, 'K1 boundary report')
    for name, expected in metadata.items():
        _same_declaration(report[name], expected, check_budget)
    _require(type(report['paths']) is list and len(report['paths']) == len(prefix.values),
             'all original K1 prefix paths required')
    passed = True
    for path, row in enumerate(report['paths']):
        check_budget()
        _keys(row, {'path_index', 'boundary'}, 'K1 path')
        _require(type(row['path_index']) is int and row['path_index'] == path, 'K1 path order differs')
        expected = _boundary(prefix, model, m, c, inference,
                             spec.to_dict()['known_mode_boundary'], path, check_budget)[0]
        _keys(row['boundary'], expected, 'K1 boundary')
        _require(type(row['boundary']['mode']) is int and row['boundary']['mode'] == 0
                 and type(row['boundary']['passed']) is bool
                 and row['boundary']['passed'] is expected['passed'], 'K1 boundary disposition differs')
        _map({name: row['boundary'][name] for name in ('mean_max_abs', 'covariance_max_abs', 'log_likelihood_abs')},
             {name: expected[name] for name in ('mean_max_abs', 'covariance_max_abs', 'log_likelihood_abs')})
        passed &= expected['passed']
    state = 'NUMERICAL_CHECKS_PASSED' if passed else 'INFERENCE_INAPPLICABLE'
    _require(type(report['state']) is str and report['state'] == state, 'K1 overall boundary disposition differs')
    # Every scalar/container is now bounded and typed before serialization.
    _require(len(encode(report)) <= 16*1024+1024*len(prefix.values), 'K1 report byte quota exceeded')
    check_budget()
    return {'schema_version': 'pirc28-k1-boundary-consistency-v1', 'report_hash': digest(report),
        'state': state, 'input_identity_bound': True, 'numerical_diagnostics_consistent': True, **FLAGS}
