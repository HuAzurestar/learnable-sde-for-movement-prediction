"""Actual retained scientific handler for the owned native session.

No public launch or approval shortcut. InputWork performs guarded acquisition;
the caller must supply the real controller/domain validators and exact seal.
Only newly committed ledger events are read between requests. The complete
closed-output reader is instantiated for independent reanalysis, not for each
forecast. The final entrypoint/runtime seal remains a separate integration.
"""
from copy import deepcopy
import os
from pathlib import Path

from . import formal_budget as budget, formal_session as native
from .formal_analysis import AnalysisConsumers, SavedScores, analyze_formal_work
from .formal_closed import ClosedOutputs
from .formal_export import ExportConsumers, export_formal_work
from .formal_forecast_records import KINDS
from .formal_forecasts import ForecastConsumers
from .formal_input_work import InputWork, _owned_record
from .formal_origins import build_origin_cases
from .formal_reanalysis import ReanalysisConsumers, reanalyze_formal_work
from .formal_saved import SavedForecasts
from .formal_scoring import ScoringConsumers, score_formal_work
from .formal_training import FitConsumers
from .protocol_core import canonical, digest, file_hash, publish, read_json, under, unpack


class FormalWorker:
    """One native session, one input population and no implicit recovery/refit."""
    def __init__(self, session, contract, *, input_options):
        self.session = deepcopy(session)
        self.directory = Path(session['directory']).resolve()
        self.root = Path(session['ledger_directory']).resolve()
        self.session_sha256 = digest(session)
        self.contract = deepcopy(budget.validate_contract(contract))
        self.inputs = InputWork(**input_options)
        self.protocol, self.execution, self.matrix = self.inputs.protocol, self.inputs.execution, self.inputs.matrix
        expected = budget.contract_for_matrix(self.matrix, protocol_sha256=self.protocol['sha256'],
            execution_sha256=self.execution['sha256'], runtime_manifest_sha256=contract['runtime_manifest_sha256'],
            approval_sha256=self.inputs.authority['approval_sha256'], ledger_directory=self.root,
            resource_policy=contract.get('resource_policy'))
        actual_root = read_json(under(self.root, 'ledger.json'))
        if (canonical(expected) != canonical(contract) or canonical(unpack(actual_root)) != canonical(contract)
                or actual_root['sha256'] != session['ledger_root_sha256']
                or session['execution_sha256'] != self.execution['sha256']):
            raise ValueError('retained worker requires the exact actual registered ledger and input scope')
        self.state = budget._State(self.contract, actual_root['sha256'])
        self.works = {w['work_id']: w for w in unpack(self.matrix)['workloads']}
        self.sequence, self.previous_barrier = 0, self.session_sha256
        self.pending, self.prepared, self.fits, self.forecasts, self.saved, self.scorer = (None,)*6
        self.completed, self.score_index = set(), {}
        self.failed = self.closed = False
        self.import_manifest = None

    def bootstrap(self, output_directory):
        """Restore qualified saved science under the native input deadline.

        Native transport already requires the exact partial-predecessor input
        control. No fit or forecast kernel is invoked. This prepares owned
        import records; controller/closed-output admission is separately bound.
        """
        from . import formal_partial_predecessor as predecessor, formal_import_scope as imports
        from .formal_metadata_batch import metadata_batch
        ref = self.inputs.partial_predecessor_reference
        if (self.failed or self.closed or self.prepared is not None or self.import_manifest is not None
                or self.sequence != 0 or ref is None):
            raise ValueError('one explicit partial-restoration bootstrap required')
        from .formal_restoration_costs import RestorationCosts
        costs = RestorationCosts(self.directory, 'worker')
        try:
            costs.mark('bootstrap-metadata')
            request = native._read(under(self.directory, 'bootstrap-request.json'))
            native._bootstrap_request(request, session_sha256=self.session_sha256,
                session=self.session, contract=self.contract)
            head = unpack(read_json(under(self.root, 'head.json')))
            while self.state.count < head['event_count']:
                record = native._read(under(self.root, f'events/{self.state.count:06d}.json'))
                self.state.apply(unpack(record), record['sha256'])
            binding, _ = predecessor._load(ref)
            if (self.tip != head or self.state.predecessor_floor is None
                    or self.state.predecessor_floor['binding_sha256'] != binding['sha256']
                    or self.state.pending is not None or not self.state.imported_success):
                raise ValueError('worker must restore its actual imported predecessor')
            output = Path(output_directory)
            if output != under(self.directory, 'bootstrap') or not output.is_dir() or any(output.iterdir()):
                raise ValueError('empty owned native bootstrap output required')
            costs.mark('input-restoration')
            self.prepared = self.inputs.restore_partial(binding, output_directory=output/'input')
            costs.mark('scope-restoration')
            scope = imports.build_scope(predecessor_reference=ref,
                source_context_reference=self.prepared.source_context_reference,
                successor_context_reference=predecessor.reference(Path(self.prepared.manifest['context_path'])),
                protocol=self.protocol, execution=self.execution, matrix=self.matrix)
            path, _ = publish(output/'scope', unpack(scope))
            scope_ref = predecessor.reference(path)
            bridge = imports.ScopeBridge(scope_ref, protocol=self.protocol, execution=self.execution,
                matrix=self.matrix, input_identity=self.prepared.context['input_identity'])
            with metadata_batch(bridge):
                costs.mark('model-restoration', total=26)
                self.fits = FitConsumers.restore_all(bridge=bridge, encoders=self.prepared.training.encoders)
                costs.progress(len(self.fits.receipts))
                entries, index = {}, {}
                costs.mark('saved-result-restoration', total=len(self.state.imported_success))
                for wid in self.state.imported_success:
                    work = self.works[wid]
                    if work['kind'] == 'input_qualification_and_population':
                        directory, manifest = output/'input', self.prepared.manifest
                    else:
                        if work['kind'] in {'method_fit', 'terrain_fit'}:
                            record = self.fits.receipts[work['fit_identity']]
                        else:
                            cases = bridge.new_cases[work['origin_mode']]
                            case = cases[work['origin_rank']] if work['origin_rank'] < len(cases) else None
                            record = bridge.forecast_record(work, case=case,
                                fit_receipt=self.fits.receipts.get(work['fit_identity']))
                        directory = output/'imports'/wid
                        path, record = publish(directory, unpack(record))
                        manifest = dict(artifact_path=str(path), artifact_sha256=record['sha256'])
                        if work['kind'] in KINDS:
                            index[wid] = dict(path=path.relative_to(self.root).as_posix(),
                                content_sha256=record['sha256'], file_sha256=file_hash(path))
                    artifacts = {p.relative_to(directory).as_posix():dict(file_sha256=file_hash(p), bytes=p.stat().st_size)
                                 for p in directory.rglob('*') if p.is_file()}
                    entries[wid] = dict(directory=str(directory), manifest=deepcopy(manifest), artifacts=artifacts,
                        source_completion=deepcopy(self.state.imported_success[wid]))
                    costs.progress(len(entries))
                self.completed = set(self.state.imported_success)
                costs.mark('predictor-reader-restoration')
                self._prepare_predictors(import_bridge=bridge, index=index)
            # No transport SUCCESS can escape an unclosed metadata batch.
            costs.mark('closing-source-verification')
            bridge.check_references()
            record = dict(schema_version='pirc17-partial-worker-import-manifest-v1',
                ledger_root_sha256=self.session['ledger_root_sha256'], protocol_sha256=self.protocol['sha256'],
                execution_sha256=self.execution['sha256'], matrix_sha256=self.matrix['sha256'],
                approval_sha256=self.inputs.authority['approval_sha256'], predecessor_reference=ref,
                import_scope=scope_ref, imported_entries=entries, new_fits=0, new_forecasts=0,
                raw_sources_independently_reloaded=False)
            path, _ = publish(output/'manifest', record)
            self.import_manifest = predecessor.reference(path)
            costs.close('returned')
            return deepcopy(self.import_manifest)
        except BaseException:
            costs.close('failed')
            self.failed = True
            self.close()
            raise

    @property
    def tip(self):
        return dict(root_sha256=self.session['ledger_root_sha256'], event_count=self.state.count,
                    last_event_sha256=self.state.tip)

    def _admit_previous(self, settlement):
        if self.pending is None:
            if settlement is not None:
                raise ValueError('new worker cannot silently resume or reconstruct previous scientific work')
            return
        work, manifest, directory, request = self.pending
        if (settlement is None or settlement['status'] != 'success'
                or self.state.status.get(work['work_id']) != 'success'):
            raise ValueError('previous scientific output has not been successfully settled')
        barrier = native._read(under(self.directory, f'barriers/{self.sequence-1:06d}.json'))
        result = native._barrier_result(barrier, request, directory=self.directory,
            session_sha256=self.session_sha256, sequence=self.sequence-1,
            previous_sha256=self.previous_barrier, worker_pid=os.getpid())
        if (result is None or barrier['payload']['result_sha256'] != settlement['result_sha256']
                or canonical(result['value']) != canonical(manifest)):
            raise ValueError('actual prior barrier/result differs from its successful settlement')
        if work['kind'] != 'input_qualification_and_population':
            path, record = _owned_record(manifest, directory, 'artifact_path', 'artifact_sha256')
            binding = dict(path=path.relative_to(self.root).as_posix(), file_sha256=file_hash(path), content_sha256=record['sha256'])
            if work['kind'] in {'method_fit', 'terrain_fit'}:
                if canonical(record) != canonical(self.fits.receipts[work['fit_identity']]):
                    raise ValueError('committed fit differs from the retained fit receipt')
            elif work['kind'] in KINDS:
                self.saved.admit(work, binding)
            elif work['kind'] == 'common_scores':
                binding['metrics_access_started_sha256'] = unpack(record)['metrics_access_started_sha256']
                self.score_index[work['work_id']] = binding
        self.completed.add(work['work_id'])
        self.previous_barrier, self.pending = barrier['sha256'], None

    def _request(self, work, directory):
        expected_output = under(self.directory, f'outputs/{self.sequence:06d}')
        if (Path(directory) != expected_output or not expected_output.is_dir() or any(expected_output.iterdir())
                or self.state.work.get(work.get('work_id')) != work):
            raise ValueError('exact native work projection and empty owned output directory required')
        request = native._read(under(self.directory, f'requests/{self.sequence:06d}.json'))
        current = unpack(request)
        index = budget._integer(current['reservation_event_index'])
        head = unpack(read_json(under(self.root, 'head.json')))
        expected_head = dict(root_sha256=self.session['ledger_root_sha256'], event_count=index+1,
                             last_event_sha256=current['reservation_sha256'])
        if canonical(head) != canonical(expected_head) or not self.state.count <= index < 1000000:
            raise ValueError('request must be the current committed ledger reservation')
        settlement = None
        while self.state.count <= index:
            record = native._read(under(self.root, f'events/{self.state.count:06d}.json'))
            event = unpack(record)
            if event['root_sha256'] != self.session['ledger_root_sha256']:
                raise ValueError('incremental event belongs to another ledger')
            if event['type'] == 'settle':
                pending = self.state.pending
                if (settlement is not None or self.pending is None or pending is None
                        or pending['work_id'] != self.pending[0]['work_id']):
                    raise ValueError('unexpected previous work in native result admission')
                settlement = event['row']
            self.state.apply(event, record['sha256'])
        if (self.tip != head or self.state.pending is None or self.state.halted is not None
                or self.state.pending['work_id'] != work['work_id']):
            raise ValueError('exact unhalted current reservation required')
        self._admit_previous(settlement)
        p, registered = native._request(request, session_sha256=self.session_sha256,
            previous_sha256=self.previous_barrier, sequence=self.sequence, session=self.session, work=self.state.work)
        claim = unpack(native._dispatch_claim(self.root, p, self.session['ledger_root_sha256']))
        if (registered != work or claim['session_directory'] != str(self.directory)
                or claim['job_name'] != self.session['job_name']):
            raise ValueError('scientific dispatch belongs to another owned session')
        return request

    def _prepare_predictors(self, *, import_bridge=None, index=None):
        # Initialize once under the LAST fit's 90s reservation, not outside the
        # measured scope or as repeated setup within each 30s forecast. These
        # provisional models cannot be USED before that fit actually settles.
        context = self.prepared.context
        cases = build_origin_cases(context['positions'].prefixes, context['population'], prior=context['prior'])
        self.forecasts = ForecastConsumers(fits=self.fits, cases=cases, population=context['population'], maps=self.prepared.maps)
        self.saved = SavedForecasts(**{k: context[k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'population', 'map_catalog')},
            cases=cases, fit_receipts=self.fits.receipts, root=self.root, index={} if index is None else index,
            import_bridge=import_bridge)
        self.scorer = ScoringConsumers(saved=self.saved, positions=context['positions'])

    def _metrics(self):
        return dict(protocol=self.protocol, execution=self.execution, **self.inputs.authority, **self.prepared.population_paths)

    def _execute(self, work, output):
        kind = work['kind']
        if kind == 'input_qualification_and_population':
            if self.prepared is not None or self.sequence != 0:
                raise ValueError('one original guarded input work must start this session')
            self.prepared = self.inputs.execute(work, output_directory=output)
            self.fits = FitConsumers(protocol=self.protocol, execution=self.execution, matrix=self.matrix, inputs=self.prepared.training)
            return self.prepared.manifest
        if self.prepared is None or not set(self.inputs.work) <= self.completed:
            raise ValueError('guarded input work must be successfully settled before computation')
        if kind in {'method_fit', 'terrain_fit'}:
            result = self.fits.execute(work, output_directory=output)
            if len(self.fits.receipts) == len(self.fits.work):
                self._prepare_predictors()
            return result
        if not set(self.fits.work) <= self.completed or self.forecasts is None:
            raise ValueError('all registered fit owners must settle before using retained predictors')
        if kind in KINDS:
            return self.forecasts.execute(work, output_directory=output)
        if kind == 'common_scores':
            return score_formal_work(scorer=self.scorer, work=work, output_directory=output, **self._metrics())
        if kind == 'mechanisms_and_inference':
            scores = SavedScores(scorer=self.scorer, root=self.root, index=self.score_index,
                                 access_journal=self.inputs.authority['journal_directory'])
            return analyze_formal_work(analysis=AnalysisConsumers(scores=scores), work=work, output_directory=output, **self._metrics())
        if kind == 'independent_reanalysis':
            closed = ClosedOutputs(self.root, contract=self.contract, tip=self.tip)
            audit = ReanalysisConsumers(closed=closed, **self.prepared.context,
                                        access_journal=self.inputs.authority['journal_directory'])
            return reanalyze_formal_work(audit=audit, work=work, output_directory=output, **self._metrics())
        if kind == 'aggregate_export':
            closed = ClosedOutputs(self.root, contract=self.contract, tip=self.tip)
            exporter = ExportConsumers(closed=closed, protocol=self.protocol, execution=self.execution, matrix=self.matrix,
                scope=unpack(self.prepared.context['input_identity']), access_journal=self.inputs.authority['journal_directory'])
            return export_formal_work(exporter=exporter, work=work, output_directory=output, **self._metrics())
        raise ValueError('unregistered scientific work kind')

    def __call__(self, work, output_directory):
        if self.failed or self.closed:
            raise ValueError('failed/closed retained handler cannot restart')
        try:
            output = Path(output_directory)
            request = self._request(work, output)
            full = self.works[work['work_id']]
            manifest = self._execute(full, output)
            self.pending = (full, deepcopy(manifest), output, request)
            self.sequence += 1
            return manifest
        except BaseException:
            self.failed = True
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            if self.forecasts is not None:
                self.forecasts.close()
            if self.prepared is not None:
                self.prepared.close()
