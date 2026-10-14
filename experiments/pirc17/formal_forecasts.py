"""Actual retained, target-free formal forecast and saved-array consumers.

Not a CLI, approval gate or durable-attempt ledger. Instantiate INSIDE the
measured, guarded worker after exact matrix/input/fit validation. No target
container is accepted. The controller owns reservation/deadline/closure and
the independent result callback; no time measured here replaces its ledger.
"""
from copy import deepcopy
from contextlib import nullcontext
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .brownian import BrownianPath
from .comparison_registry import ORIGIN_MODES
from .features import PredictedPositionFeatures
from .formal_fit_records import restore_registered_fit
from .formal_forecast_records import KINDS, ARRAY_LIMIT, forecast_scope, origin_stream_id, restore_forecast
from .formal_maps import RegisteredMaps
from .formal_origins import OriginCase
from .formal_training import FitConsumers
from .formal_timing import RuntimeTrial, SCOPE as TIMING_SCOPE, VERSION as TIMING_VERSION
from .inference import SEEDS
from .method_mechanisms import EXACT_REFERENCE, forecast_stream_binding
from .method_rollout import forecast_method
from .method_training import required_slot
from .protocol_core import digest, envelope, file_hash, publish, unpack
from .rollout import Forecast, rollout
from .runtime import hardware_manifest


class ForecastConsumers:
    def __init__(self, *, fits, cases, population, maps):
        if not isinstance(fits, FitConsumers) or not isinstance(maps, RegisteredMaps):
            raise ValueError('registered retained fit/map consumers required')
        self.protocol, self.execution, self.matrix = (deepcopy(x) for x in (fits.protocol, fits.execution, fits.matrix))
        self.input_identity, self.encoders = deepcopy(fits.inputs.identity), dict(fits.inputs.encoders)
        p, m, scope, pop, catalog = (unpack(x) for x in (self.protocol, self.matrix, self.input_identity, population, maps.catalog))
        e = unpack(self.execution)
        if (e['protocol_sha256'] != self.protocol['sha256'] or e['matrix_sha256'] != self.matrix['sha256']
                or m['protocol_sha256'] != self.protocol['sha256'] or scope['protocol_sha256'] != self.protocol['sha256']
                or scope['execution_sha256'] != self.execution['sha256']):
            raise ValueError('forecast protocol/execution/input bindings differ')
        self.settings = p['forecast_contract']['forecast']
        # The sealed protocol has no free precision/horizon override here.
        if (self.settings['particles'] != 512 or self.settings['maximum_step_seconds'] != 5.
                or self.settings['nominal_horizon_seconds'] != 1800. or self.settings['nominal_score_seconds'] != [60., 300., 900., 1800.]
                or self.settings['terrain_history_clock_seconds'] != 5. or m['seeds'] != list(SEEDS)
                or m['rank_counts'] != {'causal_prefix': 46, 'known_velocity': 6, 'point_only': 6}
                or scope['population_sha256'] != population['sha256']
                or any(catalog[k] != scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256'))):
            raise ValueError('fixed forecast/map/population scope changed')
        if set(cases) != set(ORIGIN_MODES):
            raise ValueError('all three origin modes required')
        self.cases = {k: tuple(v) for k, v in cases.items()}
        for mode, group in self.cases.items():
            selected = pop['selection']['selected' if mode == 'causal_prefix' else 'secondary_selected']
            if (len(group) > m['rank_counts'][mode] or any(not isinstance(c, OriginCase) for c in group)
                    or [dict(sample_id=c.sample_id, independent_block_id=c.independent_block_id, split='final_eval') for c in group] != selected
                    or any(c.mode != mode or c.population_sha256 != population['sha256'] or c.score_seconds.shape != (4,)
                        or not np.isfinite(c.score_seconds).all() or np.any(np.diff(c.score_seconds) <= 0)
                        or np.any(np.abs(c.score_seconds-np.array([60., 300., 900., 1800.])) > 30.) for c in group)):
                raise ValueError('forecast cases differ from original frozen population/time slots')
        if pop['selection']['secondary_selected'] != pop['selection']['selected'][:6]:
            raise ValueError('secondary modes must use the first six primary origins')
        work = [w for w in m['workloads'] if w['kind'] in KINDS]
        self.work = {w['work_id']: w for w in work}
        if len(self.work) != len(work):
            raise ValueError('duplicate forecast descriptors')
        fit_by_slot = {s: 'method-fit:'+g['training_components_sha256'] for g in m['training_groups'] for s in g['slots']}
        fit_work = {w['fit_identity']: w for w in fits.work.values()}
        for w in work:
            mode = w['origin_mode']
            inertial = w['kind'] == 'inertial_path'
            reference = w['kind'] == 'same_grid_reference'
            method = w['matrix'] in {'NEX326-methods', 'NEX326-diagnostic'}
            fit_id = None if inertial else (fit_by_slot.get('arm-01/full' if reference else w['subject']) if method else 'terrain-fit:'+w['subject'])
            if (w['work_id'] != digest({k: v for k, v in w.items() if k != 'work_id'})
                    or mode not in self.cases or type(w['origin_rank']) is not int or not 0 <= w['origin_rank'] < m['rank_counts'][mode]
                    or w['fit_identity'] != fit_id or (not inertial and fit_id not in fit_work)):
                raise ValueError('forecast work descriptor identity/rank/fit differs')
            if (w['generated_forecasts'] != int(not inertial) or w['scientific'] != (w['kind'] == 'scientific_forecast')
                    or (not inertial and not reference and w['matrix'] not in {'NEX326-methods', 'terrain'})
                    or (w['seed'] is not None if inertial else w['seed'] not in SEEDS)
                    or (reference and (w['subject'] != EXACT_REFERENCE or w['matrix'] != 'NEX326-diagnostic'))
                    or (inertial and (w['subject'] != 'all' or w['matrix'] != 'inertial'))
                    or (not method and not inertial and w['subject'] not in self.encoders)):
                raise ValueError('forecast work routing/generation differs')
            if method and m['paired_streams'][mode][w['subject']] != forecast_stream_binding(w['subject'], mode):
                raise ValueError('method paired stream policy changed')
            runtime = w['kind'].startswith('runtime_')
            replay = w['kind'] == 'forecast_replay'
            phase = ('runtime_and_forecast_replay' if runtime or replay else 'offline_common_scores' if inertial
                     else 'method_forecasts' if method else 'terrain_forecasts')
            cap = (p['resource_contract']['phase_caps_seconds'][phase] if inertial else
                   p['resource_contract']['per_method_forecast_seconds' if method else 'per_terrain_forecast_seconds'])
            if (w['phase'] != phase or w['max_active_seconds'] != cap
                    or (runtime and (type(w['repetition']) is not int or not 0 <= w['repetition'] < (1 if w['kind'] == 'runtime_warmup' else 5)))
                    or (not runtime and w['repetition'] is not None)
                    or ((runtime or replay) and (mode != 'causal_prefix' or w['origin_rank'] != 0 or w['seed'] != SEEDS[0]))):
                raise ValueError('forecast phase/cap/repetition differs')
        self.receipts, self.models = deepcopy(fits.receipts), {}
        self.import_bridge = getattr(fits, 'import_bridge', None)
        if set(self.receipts)-set(fit_work):
            raise ValueError('unregistered fitted dependency')
        # Restore each verified fitted model ONCE, not per forecast. No refit.
        for key, receipt in self.receipts.items():
            self.models[key] = restore_registered_fit(receipt, work=fit_work[key], protocol=self.protocol,
                execution=self.execution, matrix=self.matrix, input_identity=self.input_identity, import_bridge=self.import_bridge)
        self.maps, self.attempted = maps, set()
        if self.import_bridge is not None:
            self.attempted.update(set(self.work) & fits.imported_forecast_ids)
        self.driver_origin, self.drivers = None, {}
        self.runtime_hardware, self.runtime_maps, self.runtime_state = None, None, None
        self.runtime_session = uuid.uuid4().hex  # Provider identity only; never a scientific random stream.

    def close(self):
        """Release any unfinished benchmark provider; the outer owner closes maps."""
        if self.runtime_maps is not None:
            self.runtime_maps.close()
            self.runtime_maps = None

    def _runtime_identity(self, work):
        group = work['matrix'], work['subject']
        if work['kind'] == 'runtime_warm':
            state = self.runtime_state
            if (state is None or state['group'] != group or state['result_sha256'] is None
                    or (work['matrix'] == 'terrain' and (self.runtime_maps is None or self.runtime_maps.closed))):
                raise ValueError('same-subject registered warmup must precede warm trials')
            return state['provider_id'], state['result_sha256'], state['status']
        return digest(['pirc17-runtime-provider-v1', self.runtime_session, work['work_id']]), None, None

    def _method(self, work, case, model):
        c = required_slot('arm-01/full' if work['subject'] == EXACT_REFERENCE else work['subject'])['components']
        s, h, end = self.settings, self.settings['maximum_step_seconds'], float(case.score_seconds[-1])
        # Bound the union of the unchanged numerical/history/mode/output clocks;
        # this guard adds no time point, particle or experiment to that grid.
        maximum = math.ceil(end/h)+math.ceil(end/c['dt_seconds'])+math.ceil(end/model.dynamics.reference_interval_seconds)+len(case.score_seconds)
        binding = forecast_stream_binding(work['subject'], case.mode)
        result = forecast_method(model.dynamics, case.method_origin, case.score_seconds,
            propagation=c['poa'], integrator='exact' if work['subject'] == EXACT_REFERENCE else c['integrator'],
            particles=s['particles'], seed=work['seed'], max_step_seconds=h, history_step_seconds=c['dt_seconds'],
            max_steps=maximum, max_particle_steps=maximum*s['particles'], origin_id=origin_stream_id(case),
            run_id=binding['run_id'], crn_pair_id=binding['crn_pair_id'], condition_names=model.dynamics.model.condition_names,
            condition_at=case.condition_at if model.dynamics.model.condition_names else None)
        arrays = {'elapsed_seconds': result.forecast.elapsed_seconds, 'positions_m': case.to_scoring_frame(result.forecast.positions_m),
                  'method_positions_m': result.forecast.positions_m}
        if result.conditional_means_m is not None:
            arrays.update(conditional_means_m=result.conditional_means_m, conditional_covariances_m2=result.conditional_covariances_m2)
        return result.forecast, arrays, result.diagnostics

    def _terrain(self, work, case, model, maps, stages=None):
        stream, s = origin_stream_id(case), self.settings
        if self.driver_origin != stream:
            self.driver_origin, self.drivers = stream, {}
        # At most one origin/mode's five small fixed-grid drivers are retained.
        # Benchmarks recreate the driver within their timed inference operation.
        benchmark = work['kind'].startswith('runtime_')
        driver = None if benchmark else self.drivers.get(work['seed'])
        if driver is None:
            driver = BrownianPath(case.score_seconds, [s['maximum_step_seconds']], particles=s['particles'],
                seed=work['seed'], stream_id=stream, history_step_seconds=s['terrain_history_clock_seconds'])
            if not benchmark: self.drivers[work['seed']] = driver
        def query(xy):
            if work['subject'] == 'base': return [{} for _ in xy]
            # Map source/identity/backend failures are fatal integrity errors,
            # not recoverable numerical failures or a missing-feature fallback.
            try:
                with stages.span('terrain_io_and_query') if stages is not None else nullcontext():
                    return maps(xy)
            except Exception as exc: raise RuntimeError('formal raw-map query failed') from exc
        provider = PredictedPositionFeatures(self.encoders[work['subject']], query, case.scoring_frame)
        result = rollout(case.terrain_origin, case.score_seconds, particles=s['particles'], seed=work['seed'],
            max_step_seconds=s['maximum_step_seconds'], history_step_seconds=s['terrain_history_clock_seconds'],
            base_drift=model.base_drift, diffusion=model.diffusion, terrain=provider, conditioner=model.correction,
            brownian_increments=driver)
        return result, {'elapsed_seconds': result.elapsed_seconds, 'positions_m': result.positions_m}, {
            'configuration': work['subject'], 'brownian_identity': driver.identity,
            'maximum_step_seconds': s['maximum_step_seconds'], 'history_step_seconds': s['terrain_history_clock_seconds']}

    def execute(self, work, *, output_directory):
        if not isinstance(work, dict) or self.work.get(work.get('work_id')) != work:
            raise ValueError('exact registered forecast work required')
        wid = work['work_id']
        if wid in self.attempted:
            raise ValueError('forecast already attempted; no automatic retry')
        self.attempted.add(wid)
        group = self.cases[work['origin_mode']]
        case = group[work['origin_rank']] if work['origin_rank'] < len(group) else None
        model, fit = self.models.get(work['fit_identity']), self.receipts.get(work['fit_identity'])
        payload = forecast_scope(work, protocol=self.protocol, execution=self.execution, matrix=self.matrix,
                                 input_identity=self.input_identity, case=case, fit_receipt=fit)
        payload.update(status='NOT_ADMITTED', reason='rank absent from frozen eligible population', generated_forecasts_attempted=0,
            prediction_seconds=0., arrays=None, diagnostics=None, forecast_version=None, invalid_feature_rows=None,
            feature_query_rows=None, maps=None, runtime=None)
        arrays = None
        if case is not None:
            if model is None and work['kind'] != 'inertial_path':
                payload.update(status='DEPENDENCY_UNAVAILABLE', reason='registered fitted dependency unavailable; no fallback/refit')
            else:
                maps = self.maps
                benchmark = work['kind'].startswith('runtime_')
                trial = RuntimeTrial() if benchmark else None
                if benchmark:
                    provider_id, warmup_sha, warmup_status = self._runtime_identity(work)
                    if self.runtime_hardware is None:
                        self.runtime_hardware = envelope(hardware_manifest())
                    if work['kind'] == 'runtime_warmup': self.close()
                started = time.perf_counter()
                before = maps.attempted_query_rows
                payload['generated_forecasts_attempted'] = int(work['kind'] != 'inertial_path')
                try:
                    with trial if benchmark else nullcontext():
                        stages = trial.stages if benchmark else None
                        if work['kind'] == 'runtime_warmup':
                            self.runtime_state = dict(group=(work['matrix'], work['subject']), provider_id=provider_id,
                                result_sha256=None, status=None)
                        if work['matrix'] == 'terrain' and work['kind'] in {'runtime_cold', 'runtime_warmup'}:
                            with stages.span('checkpoint_and_input_io'):
                                try: maps = self.maps.fresh_provider()
                                except Exception as exc: raise RuntimeError('formal cold-map initialization failed') from exc
                                if work['kind'] == 'runtime_warmup': self.runtime_maps = maps
                            before = 0
                        elif work['matrix'] == 'terrain' and work['kind'] == 'runtime_warm':
                            maps = self.runtime_maps
                            before = maps.attempted_query_rows
                        with stages.span('rollout') if stages is not None else nullcontext():
                            if work['kind'] == 'inertial_path':
                                origin = case.terrain_origin
                                positions = origin.position_m[None, None, :]+origin.velocity_mps[None, None, :]*case.score_seconds[None, :, None]
                                result = Forecast(case.score_seconds, positions, 0, 0, 'pirc17-formal-deterministic-inertial-v1')
                                arrays, diagnostics = {'elapsed_seconds': result.elapsed_seconds, 'positions_m': positions}, {
                                    'deterministic': True, 'velocity': 'causal velocity or train-prior mean'}
                            elif work['matrix'] == 'terrain': result, arrays, diagnostics = self._terrain(work, case, model, maps, stages)
                            else: result, arrays, diagnostics = self._method(work, case, model)
                    payload.update(status='success', reason=None, diagnostics=diagnostics, forecast_version=result.version,
                        invalid_feature_rows=result.invalid_feature_rows, feature_query_rows=result.feature_query_rows)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    arrays = None
                    payload.update(status='failed', reason=(type(exc).__name__+': '+str(exc))[:500])
                finally:
                    payload['prediction_seconds'] = time.perf_counter()-started
                    if benchmark and trial.measurement is not None:
                        payload['prediction_seconds'] = trial.measurement['end_to_end_ms']/1000
                        payload['runtime'] = dict(schema_version=TIMING_VERSION, condition=work['kind'],
                            hardware=self.runtime_hardware, provider_instance_id=provider_id,
                            provider_created=work['matrix'] == 'terrain' and work['kind'] in {'runtime_cold', 'runtime_warmup'},
                            warmup_result_sha256=warmup_sha, warmup_status=warmup_status,
                            actual_horizon_seconds=float(case.score_seconds[-1]), scope=deepcopy(TIMING_SCOPE), trial=trial.measurement)
                    try:
                        if work['matrix'] == 'terrain':
                            payload['maps'] = {'query_rows_this_forecast': maps.attempted_query_rows-before,
                                               'observation': maps.observation()}
                    finally:
                        if maps is not self.maps and maps is not self.runtime_maps: maps.close()
                        if work['kind'] == 'runtime_warm' and work['repetition'] == 4: self.close()
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if arrays is not None:
            if sum(np.asarray(a).nbytes for a in arrays.values()) > ARRAY_LIMIT:
                raise ValueError('registered saved-array budget exceeded')
            path = directory/'forecast.npz'
            with path.open('xb') as stream:
                np.savez(stream, **{k: np.ascontiguousarray(v, dtype='<f8') for k, v in arrays.items()})
                stream.flush(); os.fsync(stream.fileno())
            payload['arrays'] = {'path': path.name, 'sha256': file_hash(path), 'bytes': path.stat().st_size,
                                 'shapes': {k: list(v.shape) for k, v in arrays.items()}}
        record = envelope(payload)
        restore_forecast(record, directory=directory, work=work, protocol=self.protocol, execution=self.execution,
            matrix=self.matrix, input_identity=self.input_identity, case=case, fit_receipt=fit, model=model, map_catalog=self.maps.catalog)
        path, record = publish(directory, payload)
        if work['kind'] == 'runtime_warmup' and payload['generated_forecasts_attempted']:
            self.runtime_state.update(result_sha256=record['sha256'], status=payload['status'])
        return {'artifact_path': str(path), 'artifact_sha256': record['sha256'], 'status': payload['status'],
                'generated_forecasts_attempted': payload['generated_forecasts_attempted']}
