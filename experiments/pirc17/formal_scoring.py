"""Actual common scores from closed saved outputs, with full denominators.

No fit, forecast, raw map query, resampling of trajectories or new horizon.
Public execution is guarded. Pure compute/verify methods are for measured
worker callbacks and independent saved-output reanalysis, not authorization.
"""
from collections import Counter

import numpy as np

from . import final_eval_guard as guard
from .formal_forecast_records import origin_stream_id
from .formal_saved import SavedForecasts, ScoringInputs
from .metrics import COVERAGE_LEVELS, METRICS_VERSION, EntropyGrid, score_path
from .protocol_core import publish, sha256, unpack

VERSION = 'pirc17-formal-common-scores-v1'


def point_mass_scores(path, target, elapsed, *, weights, grid):
    """Exact degenerate-distribution scores, NOT fake replicated particles."""
    x, y, times, w = (np.asarray(a, dtype=float) for a in (path, target, elapsed, weights))
    if (x.shape != (1, 4, 2) or y.shape != (4, 2) or times.shape != (4,) or w.shape != (4,)
            or not all(np.isfinite(a).all() for a in (x, y, times, w)) or times[0] <= 0 or np.any(np.diff(times) <= 0)
            or np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('one finite deterministic path and original four targets required')
    error = np.linalg.norm(x[0]-y, axis=1)
    by_time = [dict(elapsed_seconds=float(t), energy_score_m=float(error[i]),
        marginal_crps_m=np.abs(x[0, i]-y[i]).tolist(),
        region=dict(region='deterministic_point_mass', center_m=x[0, i].tolist(),
            levels=[dict(level=float(level), radius_m=0., area_m2=0., covered=bool(error[i] == 0.), empirical_mass=1.)
                    for level in COVERAGE_LEVELS]), position_entropy=grid.entropy(x[:, i])) for i, t in enumerate(times)]
    return dict(schema_version=METRICS_VERSION, particle_count=1, by_time=by_time, time_weights=w.tolist(),
        time_weighted_energy_score_m=float(w@error), point_estimator='deterministic_inertial_path',
        ade_grid_mean_m=float(error.mean()), time_weighted_displacement_error_m=float(w@error), fde_m=float(error[-1]),
        particle_endpoint_error_quantiles_m={str(q): float(error[-1]) for q in (.5, .9, .95)},
        path_energy_score_m=float(np.linalg.norm((x[0]-y)*np.sqrt(w)[:, None])),
        path_score_role='supplementary_weighted_joint_path_energy_not_sum_of_marginals',
        entropy_change_from_first_scoring_time_nats=[0.]*4,
        nll=dict(status='unavailable', reason='deterministic point mass; no qualified predictive density'),
        mode_entropy=dict(status='unavailable', reason='deterministic inertial baseline has no labelled modes'),
        joint_path_entropy=dict(status='unavailable', reason='not a registered joint path entropy estimator'))


class ScoringConsumers:
    def __init__(self, *, saved, positions):
        if not isinstance(saved, SavedForecasts):
            raise ValueError('bound saved forecast reader required')
        self.saved, self.inputs = saved, ScoringInputs(positions, saved)
        p = unpack(saved.protocol)
        policy = p['components']['finite_delivery']['metrics']
        g = policy['entropy_grid']
        if g != dict(east_min_m=-10000, east_max_m=10000, north_min_m=-10000, north_max_m=10000, cell_width_m=250, overflow_bins=1):
            raise ValueError('registered entropy grid changed')
        self.grid = EntropyGrid(tuple(np.arange(-10000., 10000.+250., 250.)), tuple(np.arange(-10000., 10000.+250., 250.)))
        self.weights = np.asarray(p['forecast_contract']['forecast']['time_weights'], dtype=float)
        if self.weights.tolist() != [.25]*4 or p['forecast_contract']['forecast']['particles'] != 512:
            raise ValueError('fixed common scorer particle/time policy changed')
        self.work = {k: w for k, w in saved.work.items() if w['kind'] == 'common_scores'}
        self.dependencies = {}
        for wid, w in self.work.items():
            dependencies = [f for f in saved.forecasts.values() if f['origin_mode'] == w['origin_mode']
                and f['origin_rank'] == w['origin_rank'] and f['kind'] in {'scientific_forecast', 'same_grid_reference', 'inertial_path'}]
            counts = Counter(f['kind'] for f in dependencies)
            if (counts != {'scientific_forecast': 190, 'same_grid_reference': 5, 'inertial_path': 1}
                    or w['phase'] != 'offline_common_scores' or w['generated_forecasts'] != 0 or w['scientific']):
                raise ValueError('complete registered scoring denominator required')
            self.dependencies[wid] = dependencies
        self.attempted = set()

    def row(self, work, saved=None):
        prediction = self.saved.read(work) if saved is None else saved
        case = self.saved.case(work)
        row = dict(forecast_work_id=work['work_id'], matrix=work['matrix'], configuration=work['subject'], seed=work['seed'],
            origin_mode=work['origin_mode'], origin_rank=work['origin_rank'], partition='final_eval', scientific=work['scientific'],
            origin_id=None if case is None else origin_stream_id(case), sample_id=None if case is None else case.sample_id,
            independent_block_id=None if case is None else case.independent_block_id,
            context_sha256=None if case is None else self.inputs.context_sha256(case),
            target_sha256=None if case is None else self.inputs.target_sha256[case.sample_id],
            saved_forecast_sha256=None if prediction.record is None else prediction.record['sha256'],
            source_status=prediction.status, status=prediction.status if prediction.status in {'success', 'failed', 'NOT_ADMITTED'} else 'unavailable',
            reason=prediction.reason, scores=None, score_m=None)
        if prediction.forecast is None:
            return row
        f = prediction.forecast
        target = self.inputs.targets[case.sample_id]
        if not np.array_equal(f.elapsed_seconds, target.elapsed_seconds):
            raise ValueError('saved prediction and original scoring clock differ')
        if work['kind'] == 'inertial_path':
            scores = point_mass_scores(f.positions_m, target.positions_m, f.elapsed_seconds, weights=self.weights, grid=self.grid)
        else:
            scores = score_path(f.positions_m, target.positions_m, f.elapsed_seconds, time_weights=self.weights, entropy_grid=self.grid)
            if prediction.native is not None:
                model = self.saved.models[work['fit_identity']].dynamics.model
                probabilities = np.asarray(model.mode_probabilities)
                positive = probabilities[probabilities > 0]
                scores['mode_entropy'] = dict(status='computed', entropy_nats=float(-np.sum(positive*np.log(positive))),
                    mode_probabilities=probabilities.tolist(), source='fitted labelled iid mode-redraw probabilities',
                    fit_receipt_sha256=self.saved.receipts[work['fit_identity']]['sha256'],
                    scope='Analytic entropy of mode identity at a redraw, not observed switches, position entropy or path entropy.')
        row.update(scores=scores, score_m=scores['time_weighted_energy_score_m'])
        return row

    def compute(self, work):
        if not isinstance(work, dict) or self.work.get(work.get('work_id')) != work:
            raise ValueError('exact registered common scoring work required')
        rows = [self.row(w) for w in self.dependencies[work['work_id']]]
        return dict(schema_version=VERSION, work_id=work['work_id'], protocol_sha256=self.saved.protocol['sha256'],
            execution_sha256=self.saved.execution['sha256'], matrix_sha256=self.saved.matrix['sha256'],
            saved_forecast_index_sha256=self.saved.index_identity([w['work_id'] for w in self.dependencies[work['work_id']]])['sha256'],
            scoring_inputs_sha256=self.inputs.identity['sha256'],
            population_sha256=self.saved.population['sha256'], origin_mode=work['origin_mode'], origin_rank=work['origin_rank'],
            expected_rows=196, counts_by_status=dict(Counter(r['status'] for r in rows)), rows=rows,
            new_forecasts=0, new_fits=0, new_scoring_draws=0, scientific_claim_authorized=False,
            numerically_qualified=False, aggregation='No successful-subset aggregation; reference and inertial rows are not scientific replications.')

    def _execute(self, access, work, output_directory):
        scope = unpack(self.saved.input_identity)
        if (not isinstance(access, guard.VerifiedAccess) or access.access_kind != 'final_eval_metrics'
                or any(getattr(access, k) != scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256'))):
            raise ValueError('matching guarded metric access required')
        if work['work_id'] in self.attempted:
            raise ValueError('common scoring work already attempted')
        self.attempted.add(work['work_id'])
        payload = self.compute(work)
        payload['metrics_access_started_sha256'] = access.access_started_sha256
        path, result = publish(output_directory, payload)
        return dict(artifact_path=str(path), artifact_sha256=result['sha256'], status='computed', generated_forecasts_attempted=0)

    def verify(self, record, *, work, access_started_sha256):
        """Independent domain callback: recompute actual scores from saved bytes."""
        expected = self.compute(work)
        expected['metrics_access_started_sha256'] = sha256(access_started_sha256)
        if unpack(record) != expected:
            raise ValueError('saved common scores differ from actual prediction/target recomputation')
        return dict(expected_rows=expected['expected_rows'], counts_by_status=expected['counts_by_status'],
                    scores_recomputed=True, new_forecasts=0, new_fits=0)


def score_formal_work(*, scorer, work, output_directory, protocol, execution, approval_path, approval_sha256,
                      test_path, review_path, journal_directory, population_path, population_sha256, eligibility_path):
    return guard.guarded_call(access_kind='final_eval_metrics', protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, population_path=population_path, population_sha256=population_sha256,
        eligibility_path=eligibility_path, operation=lambda access: scorer._execute(access, work, output_directory))
