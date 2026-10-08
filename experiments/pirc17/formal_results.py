"""Actual controller authorization and saved-science result callbacks.

Instantiate cheaply before Controller; initialization, receipt checks, domain
replay and publication are inside its existing measured work interval. No raw
source acquisition, refit, forecast or additional work is dispatched here.
Tentative artifacts are admitted only after their actual successful settlement.
"""
from copy import deepcopy
from pathlib import Path

from . import final_eval_guard as guard, formal_budget as budget, formal_controller as control, formal_session as native
from .formal_analysis import AnalysisConsumers, SavedScores
from .formal_closed import ClosedOutputs
from .formal_export import ExportConsumers, MAX_BYTES as EXPORT_MAX_BYTES
from .formal_fit_records import restore_registered_fit
from .formal_forecast_records import KINDS, restore_forecast
from .formal_input_work import _owned_record, input_verification
from .formal_origins import build_origin_cases
from .formal_reanalysis import ReanalysisConsumers, _access
from .formal_saved import SavedForecasts
from .formal_scoring import ScoringConsumers
from .formal_scope_cache import matrix_payload, own_matrix
from .protocol_core import canonical, file_hash, read_json, sha256, under, unpack


class ScientificResults:
    def __init__(self, ledger, *, protocol, execution, matrix, authority):
        if not isinstance(ledger, budget.Ledger):
            raise ValueError('actual held controller ledger required')
        self.ledger = ledger
        self._inputs = (protocol, execution, matrix, authority)
        self.works = self.active = self.pending = self.context = self.population_paths = None
        self.saved = self.scorer = None
        self.completed, self.receipts, self.score_index, self.phase_authority = set(), {}, {}, {}
        self.failed = False
        self.import_reference = None

    def _start(self):
        if self.works is not None: return
        self.protocol, self.execution, self.matrix, self.authority = (deepcopy(x) for x in self._inputs)
        guard._fields(self.authority, ('approval_path', 'approval_sha256', 'test_path', 'review_path', 'journal_directory'))
        expected = budget.contract_for_matrix(self.matrix, protocol_sha256=self.protocol['sha256'], execution_sha256=self.execution['sha256'],
            runtime_manifest_sha256=self.ledger._state.contract['runtime_manifest_sha256'],
            approval_sha256=self.authority['approval_sha256'], ledger_directory=self.ledger.directory,
            resource_policy=self.ledger._state.contract.get('resource_policy'))
        if canonical(expected) != canonical(self.ledger._state.contract):
            raise ValueError('controller scientific callbacks differ from the complete registered contract')
        self.matrix = own_matrix(self.matrix)
        self.works = {w['work_id']: w for w in matrix_payload(self.matrix)['workloads']}
        self.input_ids = {wid for wid, w in self.works.items() if w['kind'] == 'input_qualification_and_population'}
        self.fit_ids = {wid for wid, w in self.works.items() if w['kind'] in {'method_fit', 'terrain_fit'}}
        if len(self.input_ids) != 1 or len(self.fit_ids) != 26:
            raise ValueError('one guarded input and all26 independent fit owners required')

    def _current(self, work):
        self._start()
        state = self.ledger._state
        if (self.failed or self.ledger._poisoned or self.ledger._lock.stream.closed or state.halted is not None
                or state.pending is None or state.pending['work_id'] != work.get('work_id')
                or canonical(state.work.get(work.get('work_id'))) != canonical(work)):
            raise ValueError('exact live unhalted controller reservation required')
        return self.works[work['work_id']], deepcopy(state.pending), deepcopy(self.ledger.tip)

    def _check_authority(self, phase):
        """Full version/source/approval check at phase boundaries; pinned receipts
        are reread on every item. Public protected loaders also retain their own
        original full guards. Cached phase checks are not renewable approval.
        """
        auth = self.authority
        sha256(auth['approval_sha256'])
        if any(auth[k] is None for k in ('approval_path', 'test_path', 'review_path')):
            raise ValueError('human approval and exact test/review receipt paths required')
        approval, test, review = (read_json(auth[k]) for k in ('approval_path', 'test_path', 'review_path'))
        unpack(approval, expected_sha256=auth['approval_sha256'])
        identities = (approval['sha256'], test['sha256'], review['sha256'])
        if phase not in self.phase_authority:
            guard.validate_protocol(self.protocol)
            guard.validate_execution(self.execution, self.protocol)
            guard.validate_approval(approval, expected_sha256=auth['approval_sha256'],
                protocol=self.protocol, execution=self.execution, test=test, review=review)
            self.phase_authority[phase] = identities
        if identities != self.phase_authority[phase]:
            raise ValueError('approved test/review/decision changed within the phase')
        unpack(test, expected_sha256=identities[1]); unpack(review, expected_sha256=identities[2])

    def _admit_previous(self, tip):
        if self.pending is None:
            if self.import_reference is not None:
                state = self.ledger._state
                if (state.partial_imports is None or state.partial_imports['manifest_reference'] != self.import_reference
                        or self.completed != set(state.imported_success)
                        or set(state.status) != self.completed | {state.pending['work_id']}
                        or any(state.status[k] != 'success' for k in self.completed)):
                    raise ValueError('controller imported context lacks actual committed admission')
                return
            if self.completed or set(self.ledger._state.status) != {self.ledger._state.pending['work_id']}:
                raise ValueError('callbacks cannot silently adopt old or unfinished execution')
            return
        prior = self.pending
        work = prior['work']
        if self.ledger._state.status.get(work['work_id']) != 'success':
            raise ValueError('previous domain-verified result was not successfully settled')
        # The real writer already applied these events. Rebind the saved prior
        # result to its actual settlement and current committed head, consuming
        # only this small suffix rather than rescanning the entire ledger.
        previous, first = prior['reservation_sha256'], prior['reservation_event_index']+1
        for index in range(first, tip['event_count']):
            record = native._read(under(self.ledger.directory, f'events/{index:06d}.json'))
            event = unpack(record)
            budget._fields(event, ('schema_version', 'index', 'root_sha256', 'previous_sha256', 'type', 'row'))
            if (event['schema_version'] != budget.VERSION+'-event' or type(event['index']) is not int or event['index'] != index
                    or event['root_sha256'] != self.ledger.root_sha256 or event['previous_sha256'] != previous):
                raise ValueError('saved scientific admission lost its actual committed event chain')
            if index == first:
                row = event['row']
                if (event['type'] != 'settle' or row['reservation_sha256'] != prior['reservation_sha256']
                        or row['status'] != 'success' or row['result_sha256'] != prior['result_sha256']):
                    raise ValueError('actual settlement does not admit the validated scientific result')
            previous = record['sha256']
        if previous != tip['last_event_sha256']:
            raise ValueError('scientific result admission is not anchored by the committed controller head')
        # The saved result itself must still match the settled transport ID.
        native._read(prior['directory']/'result.json', prior['result_sha256'])
        if work['kind'] in KINDS:
            self.saved.admit(work, prior['binding'])
        elif work['kind'] == 'common_scores':
            self.score_index[work['work_id']] = {**prior['binding'],
                'metrics_access_started_sha256': prior['record']['payload']['metrics_access_started_sha256']}
        self.completed.add(work['work_id'])
        self.pending = None

    def authorize(self, work):
        try:
            full, reservation, tip = self._current(work)
            if self.active is not None:
                raise ValueError('previous authorized request has no verified result; no retry')
            self._check_authority(full['phase'])  # Before any protected saved-input read.
            self._admit_previous(tip)
            if not self.completed and full['kind'] != 'input_qualification_and_population':
                raise ValueError('guarded input must be the first scientific work')
            self.active = dict(work=full, reservation=reservation, tip=tip)
            return dict(schema_version=control.VERSION+'-work-authority', ledger_root_sha256=self.ledger.root_sha256,
                execution_sha256=self.execution['sha256'], approval_sha256=self.authority['approval_sha256'], work_id=full['work_id'], granted=True)
        except BaseException:
            self.failed = True
            raise

    def restore_imports(self, *, reference,context,population_paths,receipts,index,bridge):
        """Adopt independently checked current inputs, never producer objects.

        Still tentative until the controller journals this exact reference;
        _admit_previous requires that admission before the first unfinished
        reservation. Every original successful input/fit/forecast is retained.
        """
        self._start()
        state = self.ledger._state
        if (self.context is not None or self.completed or self.pending is not None or self.active is not None
                or self.import_reference is not None or state.partial_imports is not None
                or state.pending is not None or not state.imported_success
                or not (self.input_ids | self.fit_ids) <= set(state.imported_success)
                or any(context[k] != getattr(self,k) for k in ('protocol','execution','matrix'))):
            raise ValueError('fresh independently verified complete partial context required')
        if (set(receipts) != {self.works[k]['fit_identity'] for k in self.fit_ids}
                or set(index) != {k for k in state.imported_success if self.works[k]['kind'] in KINDS}):
            raise ValueError('partial parent readers must retain every original fit and forecast')
        self.context,self.population_paths,self.receipts = context,deepcopy(population_paths),deepcopy(receipts)
        self.import_reference = deepcopy(reference)
        self._readers(import_bridge=bridge,index=index)
        self.completed = set(state.imported_success)

    def _readers(self, *, import_bridge=None,index=None):
        context = self.context
        cases = build_origin_cases(context['positions'].prefixes, context['population'], prior=context['prior'])
        self.saved = SavedForecasts(**{k: context[k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'population', 'map_catalog')},
            cases=cases, fit_receipts=self.receipts, root=self.ledger.directory,
            index={} if index is None else index,import_bridge=import_bridge)
        self.scorer = ScoringConsumers(saved=self.saved, positions=context['positions'])

    def _scientific(self, work, manifest, directory):
        """Domain replay with cached guarded inputs, not a producer PASS flag."""
        kind = work['kind']
        if kind == 'input_qualification_and_population':
            if self.context is not None:
                raise ValueError('input context cannot be replaced')
            validation, context, paths = input_verification(work, manifest, directory, protocol=self.protocol,
                execution=self.execution, matrix=self.matrix, approval_sha256=self.authority['approval_sha256'],
                access_journal=self.authority['journal_directory'])
            self.context, self.population_paths = context, paths
            return validation, None, None
        if self.context is None or not self.input_ids <= self.completed:
            raise ValueError('actual guarded input must settle before scientific validation')
        path, record = _owned_record(manifest, directory, 'artifact_path', 'artifact_sha256',
            max_bytes=EXPORT_MAX_BYTES if kind == 'aggregate_export' else 32*1024*1024)
        payload = unpack(record)
        expected = dict(artifact_path=str(path), artifact_sha256=record['sha256'])
        files = {path.relative_to(directory).as_posix()}
        if kind in {'method_fit', 'terrain_fit'}:
            restore_registered_fit(record, work=work, protocol=self.protocol, execution=self.execution,
                                   matrix=self.matrix, input_identity=self.context['input_identity'])
            expected.update(fit_identity=work['fit_identity'], parameter_identity=payload['parameter_identity'])
            details = dict(saved_fit_restored=True, fit_identity=work['fit_identity'], new_fits=0)
            self.receipts[work['fit_identity']] = record
            if len(self.receipts) == len(self.fit_ids):
                self._readers()  # Once, still inside the last fit's measured work.
        else:
            if self.saved is None or not self.fit_ids <= self.completed:
                raise ValueError('all26 fit owners must settle before forecast/score validation')
            if kind in KINDS:
                restore_forecast(record, directory=directory, work=work, protocol=self.protocol, execution=self.execution,
                    matrix=self.matrix, input_identity=self.context['input_identity'], case=self.saved.case(work),
                    fit_receipt=self.saved.receipts.get(work['fit_identity']), model=self.saved.models.get(work['fit_identity']),
                    map_catalog=self.context['map_catalog'])
                expected.update(status=payload['status'], generated_forecasts_attempted=payload['generated_forecasts_attempted'])
                if payload['arrays'] is not None: files.add(payload['arrays']['path'])
                details = dict(saved_forecast_restored=True, forecast_status=payload['status'], new_forecasts=0,
                               scientific_claim_authorized=False, numerically_qualified=False)
            else:
                key = payload['metrics_access_started_sha256']
                _access(self.authority['journal_directory'], key, scope=unpack(self.context['input_identity']), kind='final_eval_metrics')
                expected.update(status='computed', generated_forecasts_attempted=0)
                if kind == 'common_scores':
                    details = self.scorer.verify(record, work=work, access_started_sha256=key)
                elif kind == 'mechanisms_and_inference':
                    scores = SavedScores(scorer=self.scorer, root=self.ledger.directory, index=self.score_index,
                                         access_journal=self.authority['journal_directory'])
                    details = AnalysisConsumers(scores=scores).verify(record, work=work, access_started_sha256=key)
                elif kind == 'independent_reanalysis':
                    closed = ClosedOutputs(self.ledger.directory, contract=self.ledger._state.contract, tip=self.ledger.tip)
                    audit = ReanalysisConsumers(closed=closed, **self.context, access_journal=self.authority['journal_directory'])
                    details = audit.verify(record, work=work, access_started_sha256=key)
                elif kind == 'aggregate_export':
                    closed = ClosedOutputs(self.ledger.directory, contract=self.ledger._state.contract, tip=self.ledger.tip)
                    exporter = ExportConsumers(closed=closed, protocol=self.protocol, execution=self.execution, matrix=self.matrix,
                        scope=unpack(self.context['input_identity']), access_journal=self.authority['journal_directory'])
                    details = exporter.verify(record, work=work, access_started_sha256=key)
                else:
                    raise ValueError('unregistered scientific work kind')
        if canonical(manifest) != canonical(expected):
            raise ValueError('scientific manifest does not match its verified actual artifact')
        artifacts = {relative: dict(file_sha256=file_hash(under(directory, relative)), bytes=under(directory, relative).stat().st_size)
                     for relative in files}
        validation = dict(schema_version=control.VERSION+'-artifact-verification', work_id=work['work_id'], verified=True,
                          artifacts=artifacts, details=details)
        control.verify_artifacts(validation, work, directory)
        binding = dict(path=path.relative_to(self.ledger.directory).as_posix(), file_sha256=artifacts[path.relative_to(directory).as_posix()]['file_sha256'],
                       content_sha256=record['sha256'])
        return validation, binding, record

    def validate(self, work, manifest, directory):
        try:
            full, reservation, tip = self._current(work)
            if self.active != dict(work=full, reservation=reservation, tip=tip):
                raise ValueError('this exact pending work must be authorized before result validation')
            directory = Path(directory)
            span = control._control_record(self.ledger, self.ledger._state.active_control)
            session_dir = Path(span['session_directory'])
            if (directory.parent != under(session_dir, 'outputs') or len(directory.name) != 6 or not directory.name.isdecimal()
                    or str(directory.resolve()) != str(directory)):
                raise ValueError('result must be in the current owned controller session directory')
            result = native._read(under(directory, 'result.json'))
            payload = unpack(result)
            budget._fields(payload, ('schema_version', 'session_sha256', 'sequence', 'reservation_sha256', 'work_id', 'value'))
            session = native._read(under(session_dir, 'session.json'))
            if (payload['schema_version'] != native.VERSION+'-result' or payload['session_sha256'] != session['sha256']
                    or payload['reservation_sha256'] != reservation['reservation_sha256'] or payload['work_id'] != work['work_id']
                    or type(payload['sequence']) is not int or payload['sequence'] != int(directory.name)
                    or canonical(payload['value']) != canonical(manifest)):
                raise ValueError('actual transport result differs from the pending scientific work')
            validation, binding, record = self._scientific(full, manifest, directory)
            self.pending = dict(work=full, directory=directory, result_sha256=result['sha256'],
                reservation_sha256=reservation['reservation_sha256'], reservation_event_index=tip['event_count']-1,
                binding=binding, record=record)
            self.active = None
            return validation
        except BaseException:
            self.failed = True
            raise
