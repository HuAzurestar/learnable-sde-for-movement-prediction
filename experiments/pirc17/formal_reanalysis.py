"""Independent saved-output domain replay inside the fixed audit work item.

The input owner supplies its already guarded typed positions/prior and exact
input/map identities. Compare these with the closed input context; never use
the producer's fitted objects, forecast cache, score arrays or verdict flags.
This checks saved evidence, not raw-data reacquisition, historical OS/latency
remeasurement, fresh fitting or empirical replication. The public entry still
requires the actual approval guard and the controller owns the outer budget.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from . import final_eval_guard as guard
from .formal_analysis import AnalysisConsumers, SavedScores
from .formal_budget import contract_for_matrix
from .formal_closed import ClosedOutputs
from .formal_eligibility import RESULT_VERSION
from .formal_forecast_records import KINDS
from .formal_inputs import validate_window
from .formal_origins import build_origin_cases, validate_prior
from .formal_runtime import RuntimeEvidence
from .formal_saved import SavedForecasts
from .formal_scoring import ScoringConsumers
from .protocol_core import canonical, digest, envelope, publish, read_json, sha256, under, unpack

VERSION = 'pirc17-independent-saved-output-reanalysis-v1'
CONTEXT_VERSION = 'pirc17-closed-input-context-v1'


def input_context(*, protocol, execution, matrix, input_identity, population, positions, prior, map_catalog):
    """Pure projection for publication by the measured, guarded input owner.

    Contains private scoring truth: keep in owned artifacts, never hand this
    bundle to a predictor or export it as a public paper artifact.
    """
    prior_scope = validate_prior(prior)
    if (prior_scope['input_sha256'] != unpack(input_identity)['method_input_sha256']
            or unpack(input_identity)['prior_evidence_sha256'] != prior.evidence['sha256']):
        raise ValueError('closed input prior differs from registered training inputs')
    cases = build_origin_cases(positions.prefixes, population, prior=prior)
    saved = SavedForecasts(protocol=protocol, execution=execution, matrix=matrix, input_identity=input_identity,
        population=population, cases=cases, fit_receipts={}, map_catalog=map_catalog, root=Path.cwd(), index={})
    scorer = ScoringConsumers(saved=saved, positions=positions)
    return envelope(dict(schema_version=CONTEXT_VERSION, protocol_sha256=protocol['sha256'],
        execution_sha256=execution['sha256'], matrix_sha256=matrix['sha256'], input_identity=input_identity,
        population=population, prior_evidence=prior.evidence, map_catalog=map_catalog,
        causal_prefixes=[p.identity() for p in positions.prefixes], scoring_inputs=scorer.inputs.identity,
        truth_passed_to_predictors=False, raw_sources_independently_reloaded=False))


def _access(journal, key, *, scope, kind, population_required=True):
    event = unpack(read_json(under(journal, sha256(key)+'.json')), expected_sha256=key)
    expected = dict(schema_version='pirc17-final-access-event-v1', event='started', access_kind=kind,
        **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256')},
        population_sha256=scope['population_sha256'] if population_required else None)
    if (set(event) != set(expected) | {'attempt_id', 'at_utc'}
            or any(event[k] != v for k, v in expected.items())
            or not isinstance(event['attempt_id'], str) or len(event['attempt_id']) != 32
            or any(c not in '0123456789abcdef' for c in event['attempt_id'])
            or not isinstance(event['at_utc'], str)
            or datetime.fromisoformat(event['at_utc']).utcoffset() != timedelta(0)):
        raise ValueError('matching actual guarded access-start event required')


def _artifact(closed, item, *, path_key='artifact_path', hash_key='artifact_sha256'):
    """Resolve only an actually owned, byte-verified closed manifest file."""
    manifest = item['manifest']
    path = Path(manifest[path_key])
    directory = item['directory'].resolve()
    if not path.is_absolute() or not path.resolve().is_relative_to(directory):
        raise ValueError('manifest must name its actual owned output artifact')
    relative = path.relative_to(directory).as_posix()
    if str(under(directory, relative)) != str(path) or relative not in item['artifacts']:
        raise ValueError('manifest artifact is not in its closed file inventory')
    record = read_json(path, expected_file_sha256=item['artifacts'][relative]['file_sha256'])
    unpack(record, expected_sha256=manifest[hash_key])
    binding = dict(path=path.relative_to(closed.directory).as_posix(),
        file_sha256=item['artifacts'][relative]['file_sha256'], content_sha256=record['sha256'])
    return record, binding


def qualification_evidence(record, *, directory, protocol, execution, approval_sha256, population, prefixes):
    """Recheck actual full eligibility/window files and outcome-blind selection.

    Raw source-file qualification was performed by the guarded input owner;
    this does not claim that reading its report repeats those original reads.
    """
    q = unpack(record)
    guard._fields(q, ('schema_version', 'protocol_sha256', 'execution_sha256', 'approval_sha256',
        'access_started_sha256', 'eligibility_path', 'eligibility_sha256', 'population_path', 'population_sha256',
        'windows', 'denominator_samples', 'eligible_samples', 'position_feature_value_prediction_metric_reads'))
    if (q['schema_version'] != RESULT_VERSION or q['protocol_sha256'] != protocol['sha256']
            or q['execution_sha256'] != execution['sha256'] or q['approval_sha256'] != approval_sha256
            or q['population_sha256'] != population['sha256']
            or type(q['position_feature_value_prediction_metric_reads']) is not int
            or q['position_feature_value_prediction_metric_reads'] != 0):
        raise ValueError('closed qualification differs from exact approved inputs')
    report = read_json(under(directory, q['eligibility_path']))
    unpack(report, expected_sha256=q['eligibility_sha256'])
    actual_population = read_json(under(directory, q['population_path']))
    if actual_population != population:
        raise ValueError('closed qualification changed its selected population')
    guard.validate_population(actual_population, report, expected_sha256=population['sha256'], protocol=protocol, execution=execution)
    rows = unpack(report)['rows']
    windows = {r['sample_id']: 'windows/'+r['window_sha256']+'.json' for r in rows if r['eligible']}
    if (q['windows'] != windows or type(q['denominator_samples']) is not int or q['denominator_samples'] != len(rows)
            or type(q['eligible_samples']) is not int or q['eligible_samples'] != len(windows)):
        raise ValueError('closed qualification omitted original eligibility/window population')
    by_prefix = {p.sample_id: p for p in prefixes}
    if [dict(sample_id=p.sample_id, independent_block_id=p.independent_block_id, split=p.split) for p in prefixes] != unpack(population)['selection']['selected']:
        raise ValueError('all selected guarded prefixes in original order required')
    p = unpack(protocol)
    for row in rows:
        if not row['eligible']: continue
        window = read_json(under(directory, windows[row['sample_id']]))
        w = validate_window(window, expected_sha256=row['window_sha256'])
        if (any(w['sample'][k] != row[k] for k in ('sample_id', 'segment_id', 'independent_block_id', 'split'))
                or w['input_binding_sha256'] != p['dataset_inputs']['sha256']
                or w['eligibility_rule_sha256'] != digest(p['eligibility_contract'])):
            raise ValueError('actual qualified window differs from original input identity')
        prefix = by_prefix.get(row['sample_id'])
        if prefix is not None and (prefix.window_sha256 != window['sha256'] or prefix.origin_epoch_ns != w['origin_epoch_ns']
                or prefix.score_seconds.tolist() != [t/1_000_000_000 for t in w['score_elapsed_ns']]):
            raise ValueError('guarded scoring inputs differ from actual selected qualified window')
    return dict(qualification_sha256=record['sha256'], eligibility_sha256=report['sha256'],
        population_sha256=population['sha256'], denominator_samples=len(rows), eligible_samples=len(windows),
        selected_samples=len(by_prefix), original_window_files_verified=True, outcome_blind_selection_recomputed=True,
        raw_sources_independently_reloaded=False)


class ReanalysisConsumers:
    def __init__(self, *, closed, protocol, execution, matrix, input_identity, population, positions, prior,
                 map_catalog, access_journal):
        if not isinstance(closed, ClosedOutputs):
            raise ValueError('actual pinned closed-output reader required')
        self.closed = closed
        self.args = dict(protocol=deepcopy(protocol), execution=deepcopy(execution), matrix=deepcopy(matrix),
            input_identity=deepcopy(input_identity), population=deepcopy(population), positions=deepcopy(positions),
            prior=deepcopy(prior), map_catalog=deepcopy(map_catalog))
        self.access_journal = Path(access_journal).resolve()
        scope = unpack(input_identity)
        expected = contract_for_matrix(matrix, protocol_sha256=protocol['sha256'], execution_sha256=execution['sha256'],
            runtime_manifest_sha256=closed.contract['runtime_manifest_sha256'], approval_sha256=scope['approval_sha256'],
            ledger_directory=closed.directory, resource_policy=closed.contract.get('resource_policy'))
        if canonical(expected) != canonical(closed.contract):
            raise ValueError('audit requires the entire exact approved matrix and budget ledger')
        matrix_payload = unpack(matrix)
        self.works = {w['work_id']: w for w in matrix_payload['workloads']}
        counts = Counter(w['kind'] for w in self.works.values())
        required = dict(input_qualification_and_population=1, method_fit=16, terrain_fit=10,
            scientific_forecast=11020, same_grid_reference=290, inertial_path=58, common_scores=58,
            forecast_replay=38, runtime_cold=75, runtime_warmup=15, runtime_warm=75,
            mechanisms_and_inference=1, independent_reanalysis=1, aggregate_export=1)
        if counts != required or matrix_payload['counts_by_kind'] != required:
            raise ValueError('full 11659-work audit denominator required')
        self.work = {key: w for key, w in self.works.items() if w['kind'] == 'independent_reanalysis'}
        if len(self.work) != 1:
            raise ValueError('one fixed independent-reanalysis work required')
        self.attempted = set()

    def sources(self):
        """Rebuild readers from actual completed files, never caller indexes."""
        args, closed = self.args, self.closed
        cases = build_origin_cases(args['positions'].prefixes, args['population'], prior=args['prior'])
        receipts, forecasts, scores, analyses, inventory = {}, {}, {}, {}, []
        inputs = None
        expected_context = input_context(**args)
        allowed = set(KINDS) | {'input_qualification_and_population', 'method_fit', 'terrain_fit', 'common_scores',
                              'mechanisms_and_inference', 'independent_reanalysis', 'aggregate_export'}
        for wid, work in self.works.items():
            kind = work['kind']
            if kind not in allowed:
                raise ValueError('audit must account for every registered work kind')
            role = 'current_audit' if kind == 'independent_reanalysis' else 'future_export' if kind == 'aggregate_export' else 'predecessor'
            row = dict(work_id=wid, kind=kind, phase=work['phase'], ledger_status=closed.disposition(wid),
                role=role, domain_status='not_verified', artifact_sha256=None, settlement_sha256=None)
            inventory.append(row)
            if role != 'predecessor': continue
            item = closed.read(work)
            if item is None:
                row['domain_status'] = 'unavailable'
                continue
            row['settlement_sha256'] = item['settlement_sha256']
            if kind == 'input_qualification_and_population':
                record, binding = _artifact(closed, item, path_key='context_path', hash_key='context_sha256')
                if record != expected_context:
                    raise ValueError('closed input context differs from actual guarded positions/prior/map/input identities')
                qualification, qbinding = _artifact(closed, item, path_key='qualification_path', hash_key='qualification_sha256')
                inputs = qualification_evidence(qualification, directory=Path(closed.directory/qbinding['path']).parent,
                    protocol=args['protocol'], execution=args['execution'], approval_sha256=unpack(args['input_identity'])['approval_sha256'],
                    population=args['population'], prefixes=args['positions'].prefixes)
                scope = unpack(args['input_identity'])
                _access(self.access_journal, unpack(qualification)['access_started_sha256'], scope=scope,
                        kind='final_eval_eligibility', population_required=False)
                _access(self.access_journal, args['positions'].access_started_sha256, scope=scope, kind='final_eval_positions')
                _access(self.access_journal, scope['access_started_sha256'], scope=scope, kind='final_eval_features')
                _access(self.access_journal, unpack(args['map_catalog'])['access_started_sha256'], scope=scope, kind='final_eval_features')
                row.update(domain_status='verified_guarded_context_and_qualification', artifact_sha256=record['sha256'])
                continue
            record, binding = _artifact(closed, item)
            row['artifact_sha256'] = record['sha256']
            payload = unpack(record)
            if kind in {'method_fit', 'terrain_fit'}:
                receipts[work['fit_identity']] = record
                row['domain_status'] = 'fit_restoration_pending'
            elif kind in KINDS:
                forecasts[wid] = binding
                row['domain_status'] = 'forecast_restoration_pending'
            elif kind in {'common_scores', 'mechanisms_and_inference'}:
                key = payload['metrics_access_started_sha256']
                _access(self.access_journal, key, scope=unpack(args['input_identity']), kind='final_eval_metrics')
                if kind == 'common_scores':
                    scores[wid] = dict(binding, metrics_access_started_sha256=key)
                    row['domain_status'] = 'score_recomputation_pending'
                else:
                    analyses[wid] = record
                    row['domain_status'] = 'analysis_recomputation_pending'
        import_bridge = None
        if closed._state.imported_success:
            # One independent audit bridge, never producer-owned fitted state
            # or a fresh full bridge/context reconstruction for each forecast.
            from .formal_import_scope import ScopeBridge
            if closed.imports is None:
                from .formal_partial_imports import ImportedOutputs
                closed.imports = ImportedOutputs(closed)
            import_bridge = ScopeBridge(closed.imports.manifest['import_scope'],
                **{k: args[k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})
        saved = SavedForecasts(**{k: args[k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'population', 'map_catalog')},
            cases=cases, fit_receipts=receipts, root=closed.directory, index=forecasts, import_bridge=import_bridge)
        scorer = ScoringConsumers(saved=saved, positions=args['positions'])
        sources = SavedScores(scorer=scorer, root=closed.directory, index=scores, access_journal=self.access_journal)
        return saved, scorer, sources, analyses, inventory, inputs, expected_context['sha256']

    def compute(self, work):
        if not isinstance(work, dict) or self.work.get(work.get('work_id')) != work:
            raise ValueError('exact independent-reanalysis work required')
        saved, scorer, scores, analyses, inventory, inputs, context_sha = self.sources()
        predictions = Counter()
        for row in inventory:
            wid = row['work_id']
            if row['domain_status'] == 'fit_restoration_pending':
                row['domain_status'] = 'verified_fit'
            elif row['domain_status'] == 'forecast_restoration_pending':
                prediction = saved.read(self.works[wid])
                predictions[prediction.status] += 1
                row.update(domain_status='verified_forecast_record', forecast_status=prediction.status)
        analysis = AnalysisConsumers(scores=scores)
        analysis_work = next(iter(analysis.work.values()))
        # Compute once: SavedScores.load verifies every available common score,
        # while the actual raw mechanisms and all registered families are replayed.
        recomputed = analysis.compute(analysis_work)
        for row in inventory:
            if row['domain_status'] == 'score_recomputation_pending': row['domain_status'] = 'verified_scores'
        original = analyses.get(analysis_work['work_id'])
        if original is not None:
            expected = dict(recomputed, metrics_access_started_sha256=unpack(original)['metrics_access_started_sha256'])
            if canonical(unpack(original)) != canonical(expected):
                raise ValueError('closed analysis differs from independent raw-score/mechanism/statistical recomputation')
            next(r for r in inventory if r['work_id'] == analysis_work['work_id'])['domain_status'] = 'verified_analysis'
        runtime = RuntimeEvidence(saved).compute()
        predecessors = [r for r in inventory if r['role'] == 'predecessor']
        complete = all(r['ledger_status'] == 'success' and r['domain_status'].startswith('verified_') for r in predecessors)
        return dict(schema_version=VERSION, work_id=work['work_id'], protocol_sha256=saved.protocol['sha256'],
            execution_sha256=saved.execution['sha256'], matrix_sha256=saved.matrix['sha256'],
            population_sha256=saved.population['sha256'], input_context_sha256=context_sha,
            closed_prefix=deepcopy(self.closed.identity), input_evidence=inputs,
            work_inventory=inventory, required_predecessors=len(predecessors), all_predecessors_verified=complete,
            counts_by_ledger_status=dict(Counter(r['ledger_status'] for r in predecessors)),
            restored_forecast_counts=dict(predictions), saved_forecast_context_sha256=saved.identity['sha256'],
            saved_score_index_sha256=scores.identity['sha256'],
            analysis=dict(status='verified' if original is not None else 'unavailable',
                original_sha256=None if original is None else original['sha256'], recomputed_payload_sha256=digest(recomputed),
                missing_saved_analysis_replaced=False, factor_conclusions=recomputed['factor_conclusions']),
            runtime_replay=runtime, new_forecasts=0, new_fits=0, scientific_claim_authorized=False,
            numerically_qualified=False, raw_sources_independently_reloaded=False,
            historical_observations_independently_remeasured=False,
            scope='Complete fixed inventory; missing evidence remains unavailable. Domain replay of saved outputs against the guarded input owner, not new raw-data qualification or empirical replication.')

    def _execute(self, access, work, output_directory):
        scope = unpack(self.args['input_identity'])
        if (not isinstance(access, guard.VerifiedAccess) or access.access_kind != 'final_eval_metrics'
                or any(getattr(access, k) != scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256'))):
            raise ValueError('matching guarded independent-reanalysis access required')
        if work['work_id'] in self.attempted:
            raise ValueError('independent reanalysis already attempted')
        self.attempted.add(work['work_id'])
        _access(self.access_journal, access.access_started_sha256, scope=scope, kind='final_eval_metrics')
        payload = self.compute(work)
        payload['metrics_access_started_sha256'] = access.access_started_sha256
        path, record = publish(output_directory, payload)
        return dict(artifact_path=str(path), artifact_sha256=record['sha256'], status='computed', generated_forecasts_attempted=0)

    def verify(self, record, *, work, access_started_sha256):
        _access(self.access_journal, access_started_sha256, scope=unpack(self.args['input_identity']), kind='final_eval_metrics')
        expected = dict(self.compute(work), metrics_access_started_sha256=sha256(access_started_sha256))
        if canonical(unpack(record)) != canonical(expected):
            raise ValueError('saved independent audit differs from actual closed outputs and domain replay')
        return dict(all_predecessors_verified=expected['all_predecessors_verified'], new_forecasts=0, new_fits=0,
                    actual_saved_output_reanalysis=True)


def reanalyze_formal_work(*, audit, work, output_directory, protocol, execution, approval_path, approval_sha256,
                         test_path, review_path, journal_directory, population_path, population_sha256, eligibility_path):
    return guard.guarded_call(access_kind='final_eval_metrics', protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, population_path=population_path, population_sha256=population_sha256,
        eligibility_path=eligibility_path, operation=lambda access: audit._execute(access, work, output_directory))
