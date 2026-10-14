"""Read a pinned committed ledger prefix and its actual closed output files.

Never acquire the writer lock, repair a head, settle a reservation, dispatch a
process, or trust a caller's output index. Later committed events do not alter
the pinned audit population. Historical OS observations are checked, not
pretended to be newly measured. Scientific domain replay is a separate layer.
"""
from copy import deepcopy
from pathlib import Path

from . import formal_budget as budget, formal_controller as control, formal_session as native
from .protocol_core import canonical, digest, envelope, read_json, sha256, under, unpack

VERSION = 'pirc17-closed-output-prefix-v1'


class ClosedOutputs:
    def __init__(self, directory, *, contract, tip):
        self.directory = Path(directory).resolve()
        budget.validate_contract(contract)
        root = read_json(under(self.directory, 'ledger.json'))
        if (canonical(unpack(root)) != canonical(contract) or str(self.directory) != contract['ledger_directory']):
            raise ValueError('closed output ledger differs from exact expected contract')
        self.root_sha256, self.contract, self.tip = root['sha256'], deepcopy(contract), deepcopy(tip)
        budget._fields(tip, ('root_sha256', 'event_count', 'last_event_sha256'))
        count = budget._integer(tip['event_count'])
        if count > 1000000 or tip['root_sha256'] != root['sha256']:
            raise ValueError('bounded independently pinned ledger prefix required')
        sha256(tip['last_event_sha256'])
        head = unpack(read_json(under(self.directory, 'head.json')))
        budget._fields(head, ('root_sha256', 'event_count', 'last_event_sha256'))
        head_count = budget._integer(head['event_count'])
        if head['root_sha256'] != root['sha256'] or not count <= head_count <= 1000000:
            raise ValueError('audit prefix is not committed in the current ledger')
        sha256(head['last_event_sha256'])
        self._state = state = budget._State(contract, root['sha256'])
        self.entries, self.sessions = {}, {}
        self.imports = None
        self.phase_first_work = {}
        for work in contract['workloads']:
            self.phase_first_work.setdefault(work['phase'], work['work_id'])
        pending, reservation_index = None, None
        for index in range(count):
            record = read_json(under(self.directory, f'events/{index:06d}.json'), max_bytes=64*1024)
            event = unpack(record)
            if event['root_sha256'] != root['sha256']:
                raise ValueError('audit event belongs to another ledger')
            if event['type'] == 'reserve':
                pending, reservation_index = {**event['row'], 'reservation_sha256': record['sha256']}, index
            elif event['type'] == 'settle':
                if pending is None:
                    raise ValueError('closed result lacks its actual reservation')
                self.entries[pending['work_id']] = dict(reservation=deepcopy(pending), reservation_event_index=reservation_index,
                    settlement=deepcopy(event['row']), settlement_sha256=record['sha256'], control_sha256=state.active_control)
            state.apply(event, record['sha256'])
            if event['type'] == 'settle': pending = None
        if state.tip != tip['last_event_sha256']:
            raise ValueError('pinned ledger prefix hash changed')
        for wid, source in state.imported_success.items():
            # This remains the OLD actual settlement. Do not invent a native
            # reservation/result/observation in the successor ledger.
            self.entries[wid] = dict(settlement_sha256=source['settlement_sha256'],
                                     imported_source=deepcopy(source))
        if state.imported_success:
            self.phase_first_work = {}
            for work in contract['workloads']:
                if work['work_id'] not in state.imported_success:
                    self.phase_first_work.setdefault(work['phase'],work['work_id'])
        # Prove that the pinned prefix actually leads to the committed head.
        # Later events are only hash-chain checked, not adopted as audit results.
        previous = state.tip
        for index in range(count, head_count):
            record = native._read(under(self.directory, f'events/{index:06d}.json'))
            event = unpack(record)
            budget._fields(event, ('schema_version', 'index', 'root_sha256', 'previous_sha256', 'type', 'row'))
            if (event['schema_version'] != budget.VERSION+'-event' or type(event['index']) is not int
                    or event['index'] != index or event['root_sha256'] != self.root_sha256
                    or event['previous_sha256'] != previous):
                raise ValueError('pinned prefix is disconnected from committed head')
            previous = record['sha256']
        if previous != head['last_event_sha256']:
            raise ValueError('committed head does not anchor this ledger chain')
        # Reading while its owner appends is allowed only for the already
        # committed pinned prefix; no recovery of an uncommitted suffix here.
        current = unpack(read_json(under(self.directory, 'head.json')))
        budget._fields(current, ('root_sha256', 'event_count', 'last_event_sha256'))
        if (current['root_sha256'] != root['sha256'] or budget._integer(current['event_count']) < head_count
                or current['event_count'] == head_count and current['last_event_sha256'] != head['last_event_sha256']):
            raise ValueError('committed audit prefix disappeared during reading')
        sha256(current['last_event_sha256'])
        self.identity = envelope(dict(schema_version=VERSION, ledger_root_sha256=root['sha256'], tip=deepcopy(tip),
            contract_sha256=digest(contract), work_dispositions={k: self.disposition(k) for k in state.work},
            charged_ns_by_phase=deepcopy(state.charged), measured_ns_by_phase=deepcopy(state.measured),
            conservatively_charged_ns_by_phase=deepcopy(state.conservative), generated_forecasts_reserved=state.generated,
            pending=deepcopy(state.pending), halted_reason=state.halted, historical_observations_remeasured=False,
            **(dict(partial_imports=deepcopy(state.partial_imports)) if state.partial_imports is not None else {})))

    def disposition(self, work_id):
        if work_id not in self._state.work:
            raise ValueError('unregistered output work')
        return self._state.status.get(work_id, 'unattempted')

    def _session(self, directory, checksum):
        key = str(directory), checksum
        if key not in self.sessions:
            session, contract = native._load_session(directory, checksum)
            if contract != self.contract:
                raise ValueError('closed output session switched its execution ledger')
            ready = unpack(native._read(under(directory, 'ready.json')))
            budget._fields(ready, ('schema_version', 'session_sha256', 'worker_pid'))
            if ready['schema_version'] != native.VERSION+'-ready' or ready['session_sha256'] != checksum:
                raise ValueError('actual worker ready evidence differs')
            budget._integer(ready['worker_pid'], positive=True)
            self.sessions[key] = session, ready
        return self.sessions[key]

    def read(self, work):
        """Verify actual completion chain and all owned bytes; no domain PASS."""
        return self._read(work, verify_owned_bytes=True)

    def read_metadata(self, work):
        """Completion/index metadata ONLY; never current bytes or domain PASS.

        This distinct result is for sealing a future supervised restoration,
        not for scoring, model construction, or cross-execution admission.
        Historical file hashes are preserved, not newly measured here.
        """
        return self._read(work, verify_owned_bytes=False)

    def _read(self, work, *, verify_owned_bytes):
        wid = work['work_id']
        registered = self._state.work.get(wid)
        if registered is None or (work.get('phase') != registered['phase']):
            raise ValueError('registered closed output work required')
        if self.disposition(wid) != 'success':
            return None
        if wid in self._state.imported_success:
            if self.imports is None:
                from .formal_partial_imports import ImportedOutputs
                self.imports = ImportedOutputs(self)
            from .formal_metadata_batch import share_batch
            share_batch(self, self.imports)
            return (self.imports.read(work) if verify_owned_bytes else self.imports.read_metadata(work))
        row = self.entries[wid]; reservation, settlement = row['reservation'], row['settlement']
        observer = unpack(native._read(under(self.directory, 'controls/'+settlement['completion_evidence_sha256']+'.json'),
                                      settlement['completion_evidence_sha256']))
        fields = ('schema_version', 'ledger_root_sha256', 'control_sha256', 'reservation_sha256', 'work_id',
            'started_ns', 'checked_through_ns', 'native_observation', 'work_authority', 'scientific_manifest_validation',
            'closure', 'candidate_status', 'error_type', 'stop_reason')
        budget._fields(observer, fields)
        if (observer['schema_version'] != control.VERSION+'-work-observation' or observer['ledger_root_sha256'] != self.root_sha256
                or observer['work_id'] != wid or observer['reservation_sha256'] != reservation['reservation_sha256']
                or observer['control_sha256'] != row['control_sha256'] or observer['candidate_status'] != 'success'
                or any(observer[k] is not None for k in ('closure', 'error_type', 'stop_reason'))):
            raise ValueError('settlement lacks matching successful controller completion')
        start = control._control_record(self, observer['control_sha256'])
        control._authority(observer['work_authority'], self, work)
        observation = unpack(observer['native_observation'])
        budget._fields(observation, ('schema_version', 'session_sha256', 'ledger_root_sha256', 'reservation_sha256',
            'work_id', 'sequence', 'status', 'started_ns', 'ended_ns', 'elapsed_ns', 'deadline_ns', 'inter_call_gap',
            'barrier_sha256', 'result_sha256', 'worker_pid', 'stop_observation', 'error_type', 'closure', 'accounting'))
        expected = dict(schema_version=native.VERSION+'-observation', ledger_root_sha256=self.root_sha256,
            reservation_sha256=reservation['reservation_sha256'], work_id=wid, status='success',
            started_ns=observer['started_ns'], deadline_ns=observer['started_ns']+reservation['reserved_ns'],
            result_sha256=settlement['result_sha256'])
        if (any(type(observation.get(k)) is not type(v) or observation.get(k) != v for k, v in expected.items())
                or any(observation.get(k) is not None for k in ('stop_observation', 'error_type', 'closure'))):
            raise ValueError('actual native completion differs from settled work')
        begin, end, checked = (budget._integer(x) for x in (observer['started_ns'], observation['ended_ns'], observer['checked_through_ns']))
        elapsed = budget._integer(settlement['elapsed_ns'])
        if (not start['started_ns'] <= begin <= end <= checked <= begin+elapsed < observation['deadline_ns']
                or observation['elapsed_ns'] != end-begin):
            raise ValueError('closed native/controller timing exceeds its actual settlement')
        session_dir = Path(start['session_directory'])
        session, ready = self._session(session_dir, observation['session_sha256'])
        if session['job_name'] != start['worker_job_name'] or session['worker_command'] != start['worker_command']:
            raise ValueError('controller and actual session command/job differ')
        claim = unpack(native._dispatch_claim(self.directory, reservation, self.root_sha256))
        if claim['session_directory'] != str(session_dir) or claim['job_name'] != session['job_name']:
            raise ValueError('actual dispatch belongs to another session')
        sequence = budget._integer(observation['sequence'])
        previous = observation['session_sha256'] if sequence == 0 else native._read(under(session_dir, f'barriers/{sequence-1:06d}.json'))['sha256']
        request = native._read(under(session_dir, f'requests/{sequence:06d}.json'))
        req, _ = native._request(request, session_sha256=observation['session_sha256'], previous_sha256=previous,
            sequence=sequence, session=session, work=self._state.work)
        if (req['reservation_event_index'] != row['reservation_event_index'] or req['work_id'] != wid
                or req['started_ns'] != begin or req['deadline_ns'] != observation['deadline_ns']):
            raise ValueError('closed request is not its settled reservation')
        # Only check recorded historical accounting; never impersonate a live
        # OS Job or claim that saved observations independently remeasure it.
        accounting = observation['accounting']
        if (not isinstance(accounting, dict) or budget._integer(accounting['active_processes']) < 2
                or budget._integer(accounting['total_processes']) < accounting['active_processes']):
            raise ValueError('historical owned-worker accounting unavailable')
        if observation['worker_pid'] != ready['worker_pid']:
            raise ValueError('observed worker changed its ready PID')
        barrier = native._read(under(session_dir, f'barriers/{sequence:06d}.json'), observation['barrier_sha256'])
        if unpack(barrier)['result_sha256'] != settlement['result_sha256']:
            raise ValueError('barrier output differs from actual settled result')
        manifest = native._barrier_result(barrier, request, directory=session_dir,
            session_sha256=observation['session_sha256'], sequence=sequence, previous_sha256=previous,
            worker_pid=ready['worker_pid'])
        if manifest is None:
            raise ValueError('a failed barrier cannot close a successful output')
        directory = under(session_dir, f'outputs/{sequence:06d}')
        validation = control.read_verification(self, work, observer['scientific_manifest_validation'])
        if verify_owned_bytes:
            validation = control.verify_artifacts(validation, work, directory)
        runtime = validation['details'].get('effective_runtime')
        from .formal_entrypoint import MODULE
        from .formal_runtime_batch import read_closed_runtime
        if runtime is not None or session['worker_command'][1:5] == ['-u', '-m', MODULE, 'worker']:
            actual = read_closed_runtime(self, session_dir, contract=self.contract,
                session_sha256=observation['session_sha256'], ledger_root_sha256=self.root_sha256,
                worker_pid=ready['worker_pid'], phase=work['phase'], first_work_id=self.phase_first_work[work['phase']])
            if canonical(actual) != canonical(runtime):
                raise ValueError('closed runtime bytes differ from actual controller-verified proof')
        return dict(directory=directory, manifest=manifest['value'], artifacts=deepcopy(validation['artifacts']),
            settlement_sha256=row['settlement_sha256'], observation_sha256=settlement['completion_evidence_sha256'],
            result_sha256=settlement['result_sha256'], controller_elapsed_ns=elapsed,
            **({} if verify_owned_bytes else dict(metadata_only=True, owned_source_bytes_rechecked=False)))
