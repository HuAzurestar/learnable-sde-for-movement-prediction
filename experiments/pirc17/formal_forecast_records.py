"""Saved-forecast validation; never fit, load targets, generate paths or authorize.

The controller separately validates the entire owned directory and authority.
Here array bytes, scientific routing, stream/model scope and diagnostic meaning
are checked. A valid failure record is NOT a successful scientific observation.
"""
import hashlib
import math
import zipfile

import numpy as np

from .method_mechanisms import EXACT_REFERENCE, forecast_stream_binding
from .method_rollout import MethodForecast, VERSION as METHOD_VERSION
from .method_training import required_slot
from .origins import frozen_array
from .protocol_core import digest, file_hash, sha256, under, unpack
from .rollout import Forecast, ROLLOUT_VERSION
from .formal_timing import validate_timing
from .formal_scope_cache import matches_work
from .formal_pinned_metadata import record_payload

VERSION = 'pirc17-registered-forecast-result-v2'
KINDS = {'scientific_forecast', 'same_grid_reference', 'forecast_replay',
         'runtime_cold', 'runtime_warmup', 'runtime_warm', 'inertial_path'}
ARRAY_LIMIT = 2*1024*1024  # Fixed N512/four slots, including conditional moments.


def origin_stream_id(case):
    # Visible semantic identity only, never input/target/window/file hashes.
    return digest(['pirc17-formal-origin-stream-v1', case.sample_id, case.mode])


def forecast_scope(work, *, protocol, execution, matrix, input_identity, case, fit_receipt):
    scope = record_payload(input_identity)
    fit = record_payload(fit_receipt) if fit_receipt is not None else None
    return dict(schema_version=VERSION, protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'],
        matrix_sha256=matrix['sha256'], input_sha256=input_identity['sha256'], approval_sha256=scope['approval_sha256'],
        population_sha256=scope['population_sha256'], work_id=work['work_id'],
        origin_id=None if case is None else origin_stream_id(case), case_sha256=None if case is None else digest(case.identity()),
        sample_id=None if case is None else case.sample_id, independent_block_id=None if case is None else case.independent_block_id,
        origin_mode=work['origin_mode'], fit_identity=work['fit_identity'],
        fit_receipt_sha256=None if fit is None else fit_receipt['sha256'],
        parameter_identity=None if fit is None else fit['parameter_identity'],
        scientific_work=work['scientific'], numerically_qualified=False, scientific_claim_authorized=False)


def _maps(payload, *, work, case, model, catalog):
    """Check static admission and explicitly cumulative actual-use counters."""
    value = payload['maps']
    if work['matrix'] != 'terrain' or case is None or model is None:
        if value is not None:
            raise ValueError('nonterrain/unavailable work cannot claim map queries')
        return
    fixed = record_payload(catalog)
    if not isinstance(value, dict) or set(value) != {'query_rows_this_forecast', 'observation'}:
        raise ValueError('actual map observation required')
    observation = value['observation']
    identity = observation['identity']
    counts = [value['query_rows_this_forecast'], observation['attempted_query_rows'], observation['completed_query_rows']]
    if (observation['catalog_sha256'] != catalog['sha256']
            or any(fixed[k] != payload[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256'))
            or {k: v for k, v in identity.items() if k != 'verified_assets'} != fixed['static_identity']
            or any(fixed['asset_sha256'].get(k) != v for k, v in identity['verified_assets'].items())
            or any(type(n) is not int or n < 0 for n in counts) or counts[0] > counts[1] or counts[2] > counts[1]):
        raise ValueError('saved raw-map identity/counts differ from admitted catalog')


def restore_forecast(record, *, directory, work, protocol, execution, matrix, input_identity, case, fit_receipt, model, map_catalog,
                     import_bridge=None):
    """Return (scoring Forecast, native MethodForecast or None), or (None,None).

No successful-subset filtering: callers must retain every returned disposition.
The expected case/model is supplied by the independently bound input/fit owner.
"""
    from .formal_import_scope import FORECAST_VERSION, bridge_for
    p = record_payload(record)
    if p.get('schema_version') == FORECAST_VERSION:
        bridge = bridge_for(record, protocol=protocol, execution=execution, matrix=matrix,
                            input_identity=input_identity, bridge=import_bridge)
        return bridge.restore_forecast(record, work=work, case=case, fit_receipt=fit_receipt,
                                       model=model, map_catalog=map_catalog)
    if work['kind'] not in KINDS or not matches_work(matrix, work):
        raise ValueError('saved forecast is not its exact registered work')
    expected = forecast_scope(work, protocol=protocol, execution=execution, matrix=matrix,
                              input_identity=input_identity, case=case, fit_receipt=fit_receipt)
    fields = {'status', 'reason', 'generated_forecasts_attempted', 'prediction_seconds', 'arrays',
              'diagnostics', 'forecast_version', 'invalid_feature_rows', 'feature_query_rows', 'maps', 'runtime'}
    if set(p) != set(expected) | fields or any(p[k] != v for k, v in expected.items()):
        raise ValueError('saved forecast scope/model/case differs')
    if type(p['prediction_seconds']) not in (int, float) or not np.isfinite(p['prediction_seconds']) or p['prediction_seconds'] < 0:
        raise ValueError('finite measured prediction time required')
    if case is None:
        status = 'NOT_ADMITTED'
    elif work['kind'] != 'inertial_path' and model is None:
        status = 'DEPENDENCY_UNAVAILABLE'
    else:
        status = p['status']
        if status not in {'success', 'failed'}:
            raise ValueError('admitted forecast needs an actual success/failure disposition')
    attempted = int(case is not None and model is not None and work['kind'] != 'inertial_path')
    if (p['status'] != status or type(p['generated_forecasts_attempted']) is not int
            or p['generated_forecasts_attempted'] != attempted):
        raise ValueError('forecast disposition/actual generation count differs')
    _maps(p, work=work, case=case, model=model, catalog=map_catalog)
    validate_timing(p['runtime'], work=work, case=case, attempted=attempted, prediction_seconds=p['prediction_seconds'])
    if status != 'success':
        if (not isinstance(p['reason'], str) or not p['reason'] or len(p['reason']) > 500
                or any(p[k] is not None for k in ('arrays', 'diagnostics', 'forecast_version', 'invalid_feature_rows', 'feature_query_rows'))
                or (status != 'failed' and p['prediction_seconds'] != 0)):
            raise ValueError('failed/unavailable forecast must not contain successful arrays or counters')
        return None, None
    if p['reason'] is not None:
        raise ValueError('successful forecast cannot carry a failure reason')
    for name in ('invalid_feature_rows', 'feature_query_rows'):
        if type(p[name]) is not int or p[name] < 0:
            raise ValueError('actual nonnegative feature counters required')
    if p['invalid_feature_rows'] > p['feature_query_rows']:
        raise ValueError('invalid feature rows exceed actual queries')
    spec = p['arrays']
    if not isinstance(spec, dict) or set(spec) != {'path', 'sha256', 'bytes', 'shapes'} or spec['path'] != 'forecast.npz':
        raise ValueError('one fixed saved-array artifact required')
    path = under(directory, spec['path'])
    if (not path.is_file() or path.stat().st_size != spec['bytes'] or not 0 < spec['bytes'] <= ARRAY_LIMIT
            or file_hash(path) != spec['sha256']):
        raise ValueError('saved forecast array bytes changed')
    is_method = work['matrix'] in {'NEX326-methods', 'NEX326-diagnostic'}
    slot = required_slot('arm-01/full' if work['subject'] == EXACT_REFERENCE else work['subject']) if is_method else None
    fp = is_method and slot['components']['poa'] == 'fp'
    keys = {'elapsed_seconds', 'positions_m'} | ({'method_positions_m'} if is_method else set())
    if fp: keys |= {'conditional_means_m', 'conditional_covariances_m2'}
    n = 1 if work['kind'] == 'inertial_path' else record_payload(protocol)['forecast_contract']['forecast']['particles']
    shapes = dict(elapsed_seconds=[4], positions_m=[n, 4, 2])
    if is_method: shapes['method_positions_m'] = [n, 4, 2]
    if fp: shapes.update(conditional_means_m=[n, 4, 2], conditional_covariances_m2=[n, 4, 2, 2])
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (len(entries) != len(keys) or {x.filename for x in entries} != {k+'.npy' for k in keys}
                or sum(x.file_size for x in entries) > ARRAY_LIMIT):
            raise ValueError('saved forecast archive has extra/duplicate/unbounded arrays')
        # Check each NPY header BEFORE np.load can allocate its declared shape.
        for entry in entries:
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in {(1, 0), (2, 0)}:
                    raise ValueError('unsupported saved-array header')
                read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = read_header(stream)
                if (list(shape) != shapes[entry.filename[:-4]] or dtype != np.dtype('float64') or fortran
                        or entry.file_size-stream.tell() != math.prod(shape)*8):
                    raise ValueError('saved-array header shape/dtype/byte count differs')
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in keys}
    if (spec['shapes'] != shapes or any(list(a.shape) != shapes[k] or a.dtype != np.dtype('float64')
            or not np.isfinite(a).all() for k, a in arrays.items()) or not np.array_equal(arrays['elapsed_seconds'], case.score_seconds)):
        raise ValueError('saved forecast shape/dtype/time/value differs')
    d = p['diagnostics']
    version = METHOD_VERSION if is_method else ('pirc17-formal-deterministic-inertial-v1' if work['kind'] == 'inertial_path' else ROLLOUT_VERSION)
    if p['forecast_version'] != version:
        raise ValueError('saved forecast kernel version differs')
    if is_method:
        components = slot['components']
        binding = forecast_stream_binding(work['subject'], case.mode)
        key = ([origin_stream_id(case), 'paired', binding['crn_pair_id']] if binding['crn_pair_id']
               else [origin_stream_id(case), 'independent', components['poa'], binding['run_id']])
        settings = record_payload(protocol)['forecast_contract']['forecast']
        check = dict(version=METHOD_VERSION, dynamics=model.dynamics.identity(), propagation=components['poa'],
            model_kind=model.dynamics.model.model_kind, condition_names=list(model.dynamics.model.condition_names),
            origin_epoch_seconds=case.method_origin.epoch_seconds, velocity_source=case.method_origin.velocity_source,
            velocity_observed_at_seconds=case.method_origin.velocity_observed_at_seconds, velocity_error_mps=case.method_origin.velocity_error_mps,
            integrator='exact' if work['subject'] == EXACT_REFERENCE else components['integrator'], particles=n,
            seed=work['seed'], origin_id=origin_stream_id(case), origin_mode=case.mode, run_id=binding['run_id'],
            crn_pair_id=binding['crn_pair_id'], random_stream_sha256=digest(key),
            max_step_seconds=settings['maximum_step_seconds'], history_step_seconds=components['dt_seconds'],
            mode_step_seconds=model.dynamics.reference_interval_seconds, numerically_qualified=False,
            scientific_claim_authorized=False, nonlinear_exactness_claimed=False, variance_reduction_qualified=False)
        if any(d.get(k) != v for k, v in check.items()) or not np.array_equal(case.to_scoring_frame(arrays['method_positions_m']), arrays['positions_m']):
            raise ValueError('saved method diagnostics/frame/stream differs')
        end = float(case.score_seconds[-1])
        maximum = math.ceil(end/settings['maximum_step_seconds'])+math.ceil(end/components['dt_seconds'])+math.ceil(end/model.dynamics.reference_interval_seconds)+4
        if (d['registered_max_steps'] != maximum or d['registered_max_particle_steps'] != maximum*n
                or type(d['integration_steps']) is not int or not 0 < d['integration_steps'] <= maximum
                or p['feature_query_rows'] != (n*d['integration_steps'] if model.dynamics.model.condition_names else 0)
                or p['invalid_feature_rows'] != 0):
            raise ValueError('saved method step/feature accounting differs')
        if fp:
            covariance = arrays['conditional_covariances_m2']
            tolerance = 128*np.finfo(float).eps*np.maximum(1., np.max(np.abs(covariance), axis=(-1, -2)))
            if (np.any(np.max(np.abs(covariance-covariance.swapaxes(-1, -2)), axis=(-1, -2)) > tolerance)
                    or np.any(np.linalg.eigvalsh(covariance)[..., 0] < -tolerance)):
                raise ValueError('saved conditional covariance is not positive semidefinite')
    elif work['kind'] == 'inertial_path':
        inertial = case.terrain_origin.position_m[None, None, :]+case.terrain_origin.velocity_mps[None, None, :]*case.score_seconds[None, :, None]
        if not np.array_equal(arrays['positions_m'], inertial) or d != {'deterministic': True, 'velocity': 'causal velocity or train-prior mean'}:
            raise ValueError('inertial path must be the exact deterministic mean path')
    else:
        settings = record_payload(protocol)['forecast_contract']['forecast']
        driver = d['brownian_identity']
        times = np.array(integration_grid(case.score_seconds, settings['maximum_step_seconds'], settings['terrain_history_clock_seconds']), dtype='<f8')
        if (d['configuration'] != work['subject'] or driver['stream_id'] != origin_stream_id(case)
                or driver['seed'] != work['seed'] or driver['max_particles'] != n
                or driver['version'] != BROWNIAN_VERSION or driver['noise_dimensions'] != 2 or driver['atomic_intervals'] != len(times)-1
                or driver['stream_id_sha256'] != hashlib.sha256(origin_stream_id(case).encode()).hexdigest()
                or driver['time_grid_sha256'] != hashlib.sha256(times.tobytes()).hexdigest()
                or p['feature_query_rows'] != n*(len(times)-1)
                or d['maximum_step_seconds'] != settings['maximum_step_seconds']
                or d['history_step_seconds'] != settings['terrain_history_clock_seconds']):
            raise ValueError('terrain settings/paired stream changed')
        sha256(driver['path_sha256'])
        maps = p['maps']
        if (not isinstance(maps, dict) or maps['query_rows_this_forecast'] < 0
                or (work['subject'] == 'base' and maps['query_rows_this_forecast'] != 0)
                or (work['subject'] != 'base' and maps['query_rows_this_forecast'] != p['feature_query_rows'])):
            raise ValueError('terrain raw-map demand differs from actual kernel feature rows')
    scoring = Forecast(frozen_array(arrays['elapsed_seconds']), frozen_array(arrays['positions_m']),
                       p['invalid_feature_rows'], p['feature_query_rows'], p['forecast_version'])
    native = None
    if is_method:
        native = MethodForecast(Forecast(scoring.elapsed_seconds, frozen_array(arrays['method_positions_m']),
            scoring.invalid_feature_rows, scoring.feature_query_rows, scoring.version),
            frozen_array(arrays['conditional_means_m']) if fp else None,
            frozen_array(arrays['conditional_covariances_m2']) if fp else None, d)
    return scoring, native
from .brownian import BROWNIAN_VERSION, integration_grid
