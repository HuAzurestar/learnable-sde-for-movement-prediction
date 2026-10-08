"""Guarded public aggregates from already closed and independently audited work.

No private positions, raw model parameters, raw map rows or host paths leave
this projection. Public replay repeats statistics from block/seed scores and
numeric mechanism summaries; it does not claim to repeat raw-source science
or confer human acceptance. Export is the one existing measured matrix item.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np

from . import final_eval_guard as guard
from .comparison_registry import ORIGIN_MODES
from .decision_policy import decision_policy, registered_contrasts
from .formal_analysis import _comparison_gates
from .formal_budget import contract_for_matrix
from .formal_closed import ClosedOutputs
from .formal_paired import FAMILIES, paired_family, terrain_conclusions
from .formal_reanalysis import _access, _artifact
from .inference import SEEDS
from .method_mechanisms import EXACT_REFERENCE, NUMERICAL_SLOTS, SCORE_SLOTS, VARIANCE_SLOTS, mechanism_registry
from .metrics import COVERAGE_LEVELS
from .protocol_core import canonical, digest, publish, read_json, sha256, unpack

VERSION = 'pirc17-public-aggregate-export-v1'
MAX_BYTES = 128*1024*1024  # Full11368 public score rows exceed the default32MiB JSON bound.
SCORE_SCALARS = ('time_weighted_energy_score_m', 'ade_grid_mean_m',
    'time_weighted_displacement_error_m', 'fde_m', 'path_energy_score_m')
LIMITS = (
    'Exposed evaluation cohort; not an untouched holdout.',
    'N512, h5 seconds, nominal 30-minute forecast; no 72-hour extrapolation or global numerical convergence claim.',
    'Five forecast seeds are not five fitted models or additional independent blocks.',
    'Secondary origin modes and scoring times are descriptive; no extra hypotheses.',
    'Shared interactions remain jointly owned; no unique/additive/synergy attribution.',
    'Public replay uses audited aggregates, not raw trajectories, refitting or new prediction.',
    'Missing/failed required observations are retained; no successful-subset accuracy estimates.',
    'Complete evidence cards, manuscript/PDF, scientific review and human acceptance remain separate deliverables.',
)


def _fields(value, names):
    guard._fields(value, names)


def _rename(value, names):
    if isinstance(value, dict): return {names.get(k, k): _rename(v, names) for k, v in value.items()}
    if isinstance(value, list): return [_rename(v, names) for v in value]
    return names.get(value, value) if isinstance(value, str) else value


def _statistical_view(value):
    """Public source references differ, numeric inference must not differ."""
    value = deepcopy(value)
    for result in value['results'].values(): result.pop('mechanism_evidence', None)
    for key in ('missing_rows', 'failed_rows'): value[key] = sorted(value[key])
    return value


def _public_scores(value):
    if value is None: return None
    # Explicit allowlist, including nested levels: never copy region.center_m.
    output = {k: deepcopy(value[k]) for k in (*SCORE_SCALARS, 'particle_count', 'time_weights',
        'point_estimator', 'particle_endpoint_error_quantiles_m', 'path_score_role',
        'entropy_change_from_first_scoring_time_nats')}
    output['by_time'] = [dict(elapsed_seconds=r['elapsed_seconds'], energy_score_m=r['energy_score_m'],
        marginal_crps_m=deepcopy(r['marginal_crps_m']), region=dict(region=r['region']['region'],
            levels=[{k: v[k] for k in ('level', 'radius_m', 'area_m2', 'covered', 'empirical_mass')} for v in r['region']['levels']]),
        position_entropy={k: r['position_entropy'][k] for k in ('entropy_nats', 'overflow_mass', 'total_mass', 'grid_identity', 'particle_count')})
        for r in value['by_time']]
    output['mode_entropy'] = dict(status=value['mode_entropy']['status'],
        entropy_nats=value['mode_entropy'].get('entropy_nats'), scope='Fitted labelled redraw entropy, not path entropy.')
    output['nll'] = dict(status=value['nll']['status'], reason='No registered qualified predictive density.')
    output['joint_path_entropy'] = dict(status=value['joint_path_entropy']['status'], reason='Not identified by marginal grid entropy.')
    return output


def _public_mechanisms(analysis):
    raw = unpack(analysis['raw_mechanisms'])
    return {mode: {slot: {k: deepcopy(g[k]) for k in ('slot_id', 'definition_sha256', 'statistic', 'operator',
        'threshold', 'units', 'status', 'value', 'expected_count', 'available_count', 'rows_sha256')}
        for slot, g in raw['modes'][mode]['gates'].items()} for mode in ORIGIN_MODES}


def _diagnostic_projection(analysis, names):
    """Public numeric witnesses, never integration dynamics or target arrays."""
    raw, output = unpack(analysis['raw_mechanisms']), {}
    for mode in ORIGIN_MODES:
        source = raw['modes'][mode]
        diagnostics = {}
        for slot, rows in source['raw_diagnostics'].items():
            values = []
            for row in rows:
                gate = row['gate']
                fields = ('actual_horizons_seconds', 'per_time_rms_conditional_mean_error_m',
                    'per_time_rms_covariance_difference_m2', 'per_time_gaussian_coupling_rms_m') if slot in NUMERICAL_SLOTS else (
                    'per_time', 'diagnostic_draws_per_time', 'quadrature_order')
                values.append(dict(origin_id=names[row['origin_id']], independent_block_id=names[row['independent_block_id']],
                    seed=row['seed'], gate_sha256=None if gate is None else gate['sha256'],
                    measurements=None if gate is None else {k: deepcopy(gate['evidence'][k]) for k in fields}))
            diagnostics[slot] = values
        variances = source['variance']['gates']
        evidence = None if variances['arm-21/mc'] is None else variances['arm-21/mc']['evidence']
        variance = None if evidence is None else {k: deepcopy(evidence[k]) for k in ('partition', 'blocks',
            'independent_block_count', 'seed_count', 'variance_ddof', 'numerator_m2', 'denominator_m2',
            'descriptive_block_bootstrap_interval', 'undefined_bootstrap_replicates', 'bootstrap_iterations',
            'bootstrap_seed', 'uncertainty_scope', 'same_joint_statistic_shared_by_two_slots')}
        output[mode] = dict(per_origin_diagnostics=diagnostics, variance=_rename(variance, names))
    return output


def _verify_numeric_witnesses(gates, details, population):
    """Repeat max-error and seed-variance arithmetic from public numbers."""
    _fields(details, ('per_origin_diagnostics', 'variance'))
    if set(details['per_origin_diagnostics']) != NUMERICAL_SLOTS | SCORE_SLOTS:
        raise ValueError('all five per-origin numerical/score diagnostics required')
    expected = {(o, b, seed) for b, origins in population.items() for o in origins for seed in SEEDS}
    for slot, rows in details['per_origin_diagnostics'].items():
        keys = [(r['origin_id'], r['independent_block_id'], r['seed']) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('public diagnostic changed its full origin/seed denominator')
        values = []
        for row in rows:
            data = row['measurements']
            if data is None: continue
            sha256(row['gate_sha256'])
            if slot in NUMERICAL_SLOTS:
                errors = data['per_time_gaussian_coupling_rms_m']
                if len(errors) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in errors):
                    raise ValueError('four finite nonnegative integration diagnostic errors required')
            else:
                if data['diagnostic_draws_per_time'] != 256 or data['quadrature_order'] != 12 or len(data['per_time']) != 4:
                    raise ValueError('fixed scoring-only diagnostic budget required')
                errors = []
                for r in data['per_time']:
                    a, b = r['moment_gaussian_quadrature_es_m'], r['gaussian_mc_es_m']
                    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (a, b)):
                        raise ValueError('finite scoring diagnostic values required')
                    error = abs(a-b)/max(abs(a), 1e-9)
                    if error != r['relative_error']: raise ValueError('public scoring error arithmetic differs')
                    errors.append(error)
            values.append(max(errors))
        gate = gates[slot]
        if len(values) != gate['available_count'] or gate['value'] != (max(values) if values and len(values) == len(expected) else None):
            raise ValueError('public worst-case mechanism differs from all required numeric witnesses')
    v = details['variance']
    if v is None:
        if any(gates[slot]['available_count'] for slot in VARIANCE_SLOTS):
            raise ValueError('missing variance cannot have a computed ratio')
        return
    blocks = v['blocks']
    if (v['partition'] != 'final_eval' or v['seed_count'] != 5 or v['variance_ddof'] != 1
            or v['bootstrap_iterations'] != 2000 or v['bootstrap_seed'] != 20260926
            or v['independent_block_count'] != len(population) or not population
            or [r['block_id'] for r in blocks] != sorted(population)):
        raise ValueError('complete fixed public seed-variance population required')
    values = []
    for row in blocks:
        differences = np.asarray(row['paired_seed_differences_m'], dtype=float)
        if differences.shape != (5, 2) or not np.isfinite(differences).all() or row['origin_count'] != 1:
            raise ValueError('complete five-seed paired differences required')
        actual = np.var(differences, axis=0, ddof=1).tolist()
        if actual != [row['mc_minus_fp_variance_m2'], row['crn_minus_fp_variance_m2']]:
            raise ValueError('public absolute seed variances differ')
        values.append(actual)
    values = np.asarray(values); averages = values.mean(axis=0)
    if averages.tolist() != [v['numerator_m2'], v['denominator_m2']]: raise ValueError('public variance means differ')
    rng = np.random.default_rng(20260926)
    draws = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    defined = draws[:, 1] > 0
    interval = np.quantile(draws[:, 0]/draws[:, 1], [.025, .975], method='inverted_cdf').tolist() if defined.all() else None
    if v['undefined_bootstrap_replicates'] != int(np.count_nonzero(~defined)) or v['descriptive_block_bootstrap_interval'] != interval:
        raise ValueError('public variance bootstrap differs')
    ratio = float(averages[0]/averages[1]) if averages[1] > 0 else None
    if any(gates[slot]['value'] != ratio for slot in VARIANCE_SLOTS): raise ValueError('public variance ratio differs')


def _replay_gates(values, block_count):
    definitions = mechanism_registry()['slots']
    required = {k for k, d in definitions.items() if d['disposition'] == 'REQUIRED'}
    if set(values) != required: raise ValueError('all28 public numeric mechanism summaries required')
    result = {}
    for slot, g in values.items():
        _fields(g, ('slot_id', 'definition_sha256', 'statistic', 'operator', 'threshold', 'units',
            'status', 'value', 'expected_count', 'available_count', 'rows_sha256'))
        d = definitions[slot]
        expected = block_count*len(SEEDS) if d['source'] in {
            'same-grid-coupled-full-horizon-reference', 'saved-forecast-and-scoring-only-Gaussian-draws'} else 1
        if (any(g[k] != d[k] for k in ('slot_id', 'statistic', 'operator', 'threshold', 'units'))
                or g['definition_sha256'] != digest(d) or type(g['expected_count']) is not int
                or g['expected_count'] != expected or type(g['available_count']) is not int
                or not 0 <= g['available_count'] <= expected):
            raise ValueError('public mechanism changed its registered definition/denominator')
        sha256(g['rows_sha256'])
        complete = expected > 0 and g['available_count'] == expected
        if g['status'] != ('computed' if complete else 'unavailable'):
            raise ValueError('public mechanism availability differs from full denominator')
        v = g['value']
        if complete:
            if type(v) not in (int, float) or not math.isfinite(v): raise ValueError('finite mechanism statistic required')
            passed = v <= d['threshold'] if d['operator'] == 'le' else v >= d['threshold']
        else:
            if v is not None: raise ValueError('missing mechanism is not an observed statistic')
            passed = None
        result[slot] = dict(g, passed=passed, source_scope='Audited numeric summary; raw source replay is separate.')
    return result


def public_tables(inputs):
    """Descriptive accuracy/coverage table, with no successful intersection."""
    slots = mechanism_registry()['slots']
    subjects = [('NEX326-methods', k) for k, v in slots.items() if v['disposition'] == 'REQUIRED']
    subjects += [('terrain', k) for k in unpack(inputs['terrain_ownership'])['configurations']]
    subjects += [('NEX326-diagnostic', EXACT_REFERENCE), ('inertial', 'all')]
    allowed = set(subjects)
    if any(r['origin_mode'] not in ORIGIN_MODES or (r['matrix'], r['configuration']) not in allowed
           for r in inputs['score_rows']):
        raise ValueError('public metric table contains an unregistered mode or subject')
    tables = []
    for mode in ORIGIN_MODES:
        population = inputs['populations'][mode]
        block_of = {o: b for b, origins in population.items() for o in origins}
        for matrix, subject in subjects:
            seeds = [None] if matrix == 'inertial' else SEEDS
            expected = {(o, seed) for o in block_of for seed in seeds}
            rows = [r for r in inputs['score_rows'] if r['origin_mode'] == mode and r['matrix'] == matrix
                    and r['configuration'] == subject and r['origin_id'] is not None]
            keys = [(r['origin_id'], r['seed']) for r in rows]
            if (len(keys) != len(set(keys)) or not set(keys) <= expected
                    or any(r['independent_block_id'] != block_of[r['origin_id']] or r['partition'] != 'final_eval'
                        or r['scientific'] != (matrix in {'NEX326-methods', 'terrain'}) for r in rows)):
                raise ValueError('public metric table changed its registered population/matrix/seed keys')
            missing = len(expected)-len(rows)
            complete = bool(expected) and not missing and all(r['status'] == 'success' for r in rows)
            measurements = None
            if complete:
                for r in rows:
                    if r['scores'] is None or r['score_m'] != r['scores']['time_weighted_energy_score_m']:
                        raise ValueError('public scalar score differs from saved score metrics')
                    if len(r['scores']['by_time']) != 4 or any(tuple(v['level'] for v in t['region']['levels']) != COVERAGE_LEVELS
                                                               for t in r['scores']['by_time']):
                        raise ValueError('all registered scoring times and coverage levels required')
                measurements = {key: float(np.mean([r['scores'][key] for r in rows])) for key in SCORE_SCALARS}
                measurements['by_time'] = [dict(nominal_seconds=t,
                    actual_elapsed_seconds_range=[min(r['scores']['by_time'][i]['elapsed_seconds'] for r in rows),
                        max(r['scores']['by_time'][i]['elapsed_seconds'] for r in rows)],
                    energy_score_m=float(np.mean([r['scores']['by_time'][i]['energy_score_m'] for r in rows])),
                    position_entropy_nats=float(np.mean([r['scores']['by_time'][i]['position_entropy']['entropy_nats'] for r in rows])),
                    coverage=[dict(level=level, coverage_rate=float(np.mean([r['scores']['by_time'][i]['region']['levels'][j]['covered'] for r in rows])),
                        mean_area_m2=float(np.mean([r['scores']['by_time'][i]['region']['levels'][j]['area_m2'] for r in rows])))
                        for j, level in enumerate(COVERAGE_LEVELS)]) for i, t in enumerate((60, 300, 900, 1800))]
            tables.append(dict(origin_mode=mode, matrix=matrix, configuration=subject,
                expected_forecasts=len(expected), available_score_rows=len(rows), missing_score_rows=missing,
                counts_by_status=dict(Counter(r['status'] for r in rows)), status='computed' if complete else 'unavailable',
                means=measurements, role='descriptive equal blocks and five matched seeds; inertial has one deterministic path per block',
                hypothesis_tests_performed=False))
    return tables


def replay_public(inputs):
    """Offline arithmetic over public aggregates; no raw files or access guard.

The caller must pin the export hash for source authenticity. This deliberately
cannot establish authenticity merely because self-consistent numbers replay.
"""
    _fields(inputs, ('decision_policy_sha256', 'populations', 'mechanisms', 'mechanism_details', 'terrain_ownership', 'score_rows'))
    if inputs['decision_policy_sha256'] != decision_policy()['sha256']:
        raise ValueError('public replay requires the exact sealed decision policy')
    if set(inputs['populations']) != set(ORIGIN_MODES) or set(inputs['mechanisms']) != set(ORIGIN_MODES):
        raise ValueError('all three registered origin modes required')
    owners = unpack(inputs['terrain_ownership'])
    modes = {}
    for mode in ORIGIN_MODES:
        population = inputs['populations'][mode]
        gates = _replay_gates(inputs['mechanisms'][mode], len(population))
        _verify_numeric_witnesses(gates, inputs['mechanism_details'][mode], population)
        source_sha = digest(inputs['mechanisms'][mode])
        rows = [r for r in inputs['score_rows'] if r['origin_mode'] == mode and r['scientific'] and r['origin_id'] is not None]
        families = {}
        for family in FAMILIES:
            configs = {c for pair in registered_contrasts(family).values() for c in pair}
            matrix = 'terrain' if family.startswith('weighted-es-') else 'NEX326-methods'
            selected = [dict(r, reason=None if r['status'] == 'success' else 'Required audited score not successful.')
                        for r in rows if r['matrix'] == matrix and r['configuration'] in configs]
            present = {(r['origin_id'], r['seed'], r['configuration']) for r in selected}
            # Missing score work stays explicitly unavailable, never replacement
            # scoring. Preserve the frozen population even with no saved rows.
            for block, origins in population.items():
                for origin in origins:
                    for seed in SEEDS:
                        for config in sorted(configs):
                            if (origin, seed, config) not in present:
                                selected.append(dict(origin_id=origin, independent_block_id=block, seed=seed,
                                    configuration=config, matrix=matrix, origin_mode=mode, partition='final_eval',
                                    status='unavailable', score_m=None, reason='Required closed score unavailable.'))
            families[family] = paired_family(selected, family_id=family, expected_origins_by_block=population,
                origin_mode=mode, mechanisms=_comparison_gates(family, method_gates=gates,
                    terrain_configs=owners['configurations'], mechanism_sha256=source_sha,
                    ownership_sha256=inputs['terrain_ownership']['sha256']))
        modes[mode] = dict(families=families, mechanism_gates=gates)
    primary = modes['causal_prefix']['families']
    return dict(modes=modes, metric_tables=public_tables(inputs), factor_conclusions=terrain_conclusions(primary['weighted-es-primary'],
        primary['weighted-es-lio'], ownership=inputs['terrain_ownership']), scientific_claim_authorized=False,
        numerically_qualified=False, new_forecasts=0, new_fits=0)


def _runtime_projection(record):
    """Keep measured cost evidence, not absolute DLL paths or free-form errors."""
    value = unpack(record)
    result = {k: deepcopy(value[k]) for k in ('registered_counts', 'settings', 'timing_scope', 'replay_counts_by_status',
        'replay_scope', 'verification_scope', 'latency_scope')}
    result['hardware'] = {key: {k: deepcopy(v) for k, v in unpack(item).items() if k != 'loaded_threadpools'}
                          for key, item in value['hardware'].items()}
    result['replay_rows'] = [{k: deepcopy(row[k]) for k in ('matrix', 'subject', 'original_work_id', 'replay_work_id',
        'original_sha256', 'replay_sha256', 'original_status', 'replay_status', 'original_prediction_identity',
        'replay_prediction_identity', 'status', 'output_identity_equal', 'successful_prediction_reproduced')}
        for row in value['replay_rows']]
    result['runtime_subjects'] = []
    for subject in value['runtime_subjects']:
        rows = []
        for row in subject['rows']:
            r = {k: deepcopy(row[k]) for k in ('work_id', 'kind', 'repetition', 'status', 'forecast_sha256', 'warm_condition_verified')}
            t = row['runtime']
            r['measurement'] = None if t is None else dict(hardware_sha256=t['hardware']['sha256'],
                actual_horizon_seconds=t['actual_horizon_seconds'], trial=deepcopy(t['trial']))
            rows.append(r)
        result['runtime_subjects'].append(dict(matrix=subject['matrix'], subject=subject['subject'], rows=rows,
            conditions=deepcopy(subject['conditions']), warmup_excluded_from_latency_quantiles=True, memory_scope=subject['memory_scope']))
    return result


class ExportConsumers:
    def __init__(self, *, closed, protocol, execution, matrix, scope, access_journal):
        if not isinstance(closed, ClosedOutputs): raise ValueError('actual closed output reader required')
        self.closed, self.protocol, self.execution, self.matrix = closed, protocol, execution, matrix
        self.scope = {k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')}
        self.access_journal = access_journal
        expected = contract_for_matrix(matrix, protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'],
            runtime_manifest_sha256=closed.contract['runtime_manifest_sha256'], approval_sha256=self.scope['approval_sha256'],
            ledger_directory=closed.directory, resource_policy=closed.contract.get('resource_policy'))
        if canonical(expected) != canonical(closed.contract): raise ValueError('export requires the exact complete matrix contract')
        self.works = {w['work_id']: w for w in unpack(matrix)['workloads']}
        self.work = {k: w for k, w in self.works.items() if w['kind'] == 'aggregate_export'}
        if len(self.work) != 1: raise ValueError('one registered export work required')
        self.attempted = set()

    def _record(self, work):
        item = self.closed.read(work)
        if item is None: raise ValueError('required closed export source unavailable: '+work['kind'])
        record, _ = _artifact(self.closed, item)
        payload = unpack(record)
        _access(self.access_journal, payload['metrics_access_started_sha256'], scope=self.scope, kind='final_eval_metrics')
        for key in ('protocol_sha256', 'execution_sha256', 'population_sha256'):
            if payload[key] != self.scope[key]: raise ValueError('export source changed its scientific scope')
        if payload['matrix_sha256'] != self.matrix['sha256']: raise ValueError('export source changed its matrix')
        return record

    def compute(self, work):
        if self.work.get(work.get('work_id')) != work: raise ValueError('exact registered export work required')
        audit_work = next(w for w in self.works.values() if w['kind'] == 'independent_reanalysis')
        analysis_work = next(w for w in self.works.values() if w['kind'] == 'mechanisms_and_inference')
        audit_record, analysis_record = self._record(audit_work), self._record(analysis_work)
        audit, analysis = unpack(audit_record), unpack(analysis_record)
        prefix = ClosedOutputs(self.closed.directory, contract=self.closed.contract, tip=unpack(audit['closed_prefix'])['tip'])
        if prefix.identity != audit['closed_prefix']: raise ValueError('export audit prefix is not the actual committed prefix')
        expected_analysis = {k: v for k, v in analysis.items() if k != 'metrics_access_started_sha256'}
        if (audit['analysis']['status'] != 'verified' or audit['analysis']['original_sha256'] != analysis_record['sha256']
                or audit['analysis']['recomputed_payload_sha256'] != digest(expected_analysis)
                or audit['analysis']['factor_conclusions'] != analysis['factor_conclusions']):
            raise ValueError('export analysis is not the independently verified source')
        inventory = {r['work_id']: r for r in audit['work_inventory']}
        if len(inventory) != len(audit['work_inventory']) or set(inventory) != set(self.works):
            raise ValueError('export cannot omit work from the full audit denominator')
        for wid, row in inventory.items():
            if row['role'] != 'predecessor': continue
            entry = self.closed.entries.get(wid)
            actual_settlement = None if entry is None else entry['settlement_sha256']
            if row['ledger_status'] != self.closed.disposition(wid) or row['settlement_sha256'] != actual_settlement:
                raise ValueError('scientific predecessors changed after the independent audit')
        populations = {m: analysis['modes'][m]['expected_origins_by_block'] for m in ORIGIN_MODES}
        blocks = sorted({b for p in populations.values() for b in p})
        names = {b: f'block-{i+1:03d}' for i, b in enumerate(blocks)}
        for mode, population in populations.items():
            for block, origins in population.items():
                for i, origin in enumerate(origins): names[origin] = f'{mode}:{names[block]}:origin-{i+1:03d}'
        dispositions = {r['work_id']: r for r in analysis['score_work_dispositions']}
        score_works = {k: w for k, w in self.works.items() if w['kind'] == 'common_scores'}
        if set(dispositions) != set(score_works): raise ValueError('all58 score dispositions required')
        rows = []
        for wid, w in score_works.items():
            d = dispositions[wid]
            if d['score_artifact_sha256'] is None:
                if self.closed.disposition(wid) == 'success': raise ValueError('analysis omitted a successful closed score')
                continue
            record = self._record(w)
            if (record['sha256'] != d['score_artifact_sha256'] or record['sha256'] != inventory[wid]['artifact_sha256']
                    or inventory[wid]['domain_status'] != 'verified_scores'):
                raise ValueError('score file differs from independently audited analysis inputs')
            scored = unpack(record)
            if scored['expected_rows'] != 196 or len(scored['rows']) != 196:
                raise ValueError('complete per-origin score denominator required')
            for r in scored['rows']:
                row = {k: deepcopy(r[k]) for k in ('forecast_work_id', 'matrix', 'configuration', 'seed', 'origin_mode',
                    'origin_rank', 'partition', 'scientific', 'status', 'score_m', 'saved_forecast_sha256')}
                row.update(origin_id=None if r['origin_id'] is None else names[r['origin_id']],
                    independent_block_id=None if r['independent_block_id'] is None else names[r['independent_block_id']],
                    source_score_sha256=record['sha256'], source_reason_sha256=digest(r['reason']), scores=_public_scores(r['scores']))
                rows.append(row)
        inputs = dict(decision_policy_sha256=analysis['decision_policy_sha256'], populations=_rename(populations, names),
            mechanisms=_public_mechanisms(analysis), mechanism_details=_diagnostic_projection(analysis, names),
            terrain_ownership=deepcopy(analysis['terrain_ownership']), score_rows=rows)
        replayed = replay_public(inputs)
        for mode in ORIGIN_MODES:
            for family in FAMILIES:
                original = _rename(_statistical_view(analysis['modes'][mode]['families'][family]), names)
                actual = _statistical_view(replayed['modes'][mode]['families'][family])
                # Relabeling preserves the original sorted block order and hence
                # fixed bootstrap draws; only failed-row presentation is sorted.
                for key in ('missing_rows', 'failed_rows'): original[key] = sorted(original[key])
                if canonical(original) != canonical(actual): raise ValueError('public aggregates do not reproduce the saved inference')
        if replayed['factor_conclusions'] != analysis['factor_conclusions']:
            raise ValueError('public aggregates do not reproduce the saved factor conclusions')
        costs = unpack(self.closed.identity)
        result = dict(schema_version=VERSION, work_id=work['work_id'], **self.scope, matrix_sha256=self.matrix['sha256'],
            audit_sha256=audit_record['sha256'], analysis_sha256=analysis_record['sha256'],
            scope='Public block/seed score aggregates and numeric mechanism summaries; original identifiers replaced by ordinal labels.',
            inputs=inputs, recomputed=replayed, method_ledger=deepcopy(analysis['method_ledger']),
            history=dict(role='baseline-only', terrain_verdict=None), runtime_replay=_runtime_projection(audit['runtime_replay']),
            audit_summary=dict(required_predecessors=audit['required_predecessors'],
                all_predecessors_verified=audit['all_predecessors_verified'],
                counts_by_ledger_status=deepcopy(audit['counts_by_ledger_status']),
                input_evidence=deepcopy(audit['input_evidence']), work_inventory=deepcopy(audit['work_inventory'])),
            score_work_dispositions=deepcopy(analysis['score_work_dispositions']),
            budget_snapshot=dict(ledger_root_sha256=self.closed.root_sha256, tip=deepcopy(self.closed.tip),
                charged_ns_by_phase=costs['charged_ns_by_phase'], measured_ns_by_phase=costs['measured_ns_by_phase'],
                conservatively_charged_ns_by_phase=costs['conservatively_charged_ns_by_phase'],
                generated_forecasts_reserved=costs['generated_forecasts_reserved'], includes_export_completion=False,
                scope='Snapshot before export completes, not the final closed execution cost.'),
            limitations=list(LIMITS), raw_trajectories_exported=False, original_identifiers_exported=False,
            scientific_claim_authorized=False, numerically_qualified=False, human_accepted=False, new_forecasts=0, new_fits=0)
        if len(canonical(result)) > MAX_BYTES-1024: raise ValueError('public aggregate bundle exceeds its fixed file bound')
        return result

    def _execute(self, access, work, output_directory):
        if (not isinstance(access, guard.VerifiedAccess) or access.access_kind != 'final_eval_metrics'
                or any(getattr(access, k) != v for k, v in self.scope.items())):
            raise ValueError('matching guarded aggregate-export access required')
        if work['work_id'] in self.attempted: raise ValueError('aggregate export already attempted')
        self.attempted.add(work['work_id'])
        _access(self.access_journal, access.access_started_sha256, scope=self.scope, kind='final_eval_metrics')
        payload = dict(self.compute(work), metrics_access_started_sha256=access.access_started_sha256)
        path, record = publish(output_directory, payload)
        return dict(artifact_path=str(path), artifact_sha256=record['sha256'], status='computed', generated_forecasts_attempted=0)

    def verify(self, record, *, work, access_started_sha256):
        _access(self.access_journal, access_started_sha256, scope=self.scope, kind='final_eval_metrics')
        expected = dict(self.compute(work), metrics_access_started_sha256=sha256(access_started_sha256))
        if canonical(unpack(record)) != canonical(expected): raise ValueError('aggregate export differs from actual audited sources')
        return dict(public_aggregate_replayed=True, all_predecessors_verified=expected['audit_summary']['all_predecessors_verified'],
            raw_trajectories_exported=False, new_forecasts=0, new_fits=0, scientific_claim_authorized=False)


def export_formal_work(*, exporter, work, output_directory, protocol, execution, approval_path, approval_sha256,
                       test_path, review_path, journal_directory, population_path, population_sha256, eligibility_path):
    return guard.guarded_call(access_kind='final_eval_metrics', protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, population_path=population_path, population_sha256=population_sha256,
        eligibility_path=eligibility_path, operation=lambda access: exporter._execute(access, work, output_directory))


def main(argv=None):
    """Verify/rebuild only a caller-pinned public aggregate, never run science."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Replay a hash-pinned public PIRC-17 aggregate without private data.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--sha256', required=True, help='Independently pinned export content SHA256.')
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args(argv)
    record = read_json(args.input, max_bytes=MAX_BYTES)
    value = unpack(record, expected_sha256=args.sha256)
    if (value.get('schema_version') != VERSION or any(value.get(k) is not False for k in (
            'raw_trajectories_exported', 'original_identifiers_exported', 'scientific_claim_authorized',
            'numerically_qualified', 'human_accepted'))):
        raise ValueError('exact unaccepted public aggregate required; replay cannot authorize claims')
    result = replay_public(value['inputs'])
    if canonical(result) != canonical(value['recomputed']): raise ValueError('public aggregate replay differs')
    path, output = publish(args.output_directory, dict(schema_version=VERSION+'-offline-replay',
        source_sha256=record['sha256'], result=result, raw_data_read=False, new_forecasts=0, new_fits=0,
        human_accepted=False, scope='Arithmetic reproducibility of a pinned aggregate, not new raw-source validation.'))
    print(json.dumps(dict(status='verified', source_sha256=record['sha256'], replay_sha256=output['sha256'], path=str(path))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
