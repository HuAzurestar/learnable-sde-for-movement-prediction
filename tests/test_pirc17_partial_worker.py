"""Full typed synthetic science import, with explicit authority/native seams.

No new empirical run/approval, fitting or forecast generation. All real domain
readers and budget/owned-file publication paths below remain unmodified.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_pirc17_import_scope import source, case
from tests.test_pirc17_formal_forecasts import prepared, fixture_maps
from experiments.pirc17 import formal_budget as budget, formal_partial_predecessor as predecessor
from experiments.pirc17 import formal_session as native, formal_input_work as inputs
from experiments.pirc17 import formal_partial_imports as admission, formal_controller as control, formal_closed
from experiments.pirc17.formal_training import RestoredFitInputs
from experiments.pirc17.formal_worker import FormalWorker
from experiments.pirc17.protocol_core import digest, envelope, publish, read_json, unpack


def test_worker_restores_all_models_and_saved_predictions_as_owned_records_without_generation(case,monkeypatch):
    args, source_case = case['args'], case['source']
    binding, _ = predecessor._load(source_case['predecessor_ref'])
    root = (case['root']/'successor-SOFTWARE-ledger').resolve()
    runtime = envelope(dict(protocol_sha256=args['protocol']['sha256'],matrix_sha256=args['matrix']['sha256'],
        ledger_directory=str(root),predecessor=binding,software_only=True))
    approval = unpack(args['input_identity'])['approval_sha256']
    contract = budget.contract_for_matrix(args['matrix'],protocol_sha256=args['protocol']['sha256'],
        execution_sha256=args['execution']['sha256'],runtime_manifest_sha256=runtime['sha256'],
        approval_sha256=approval,ledger_directory=root)
    # Explicit original-native closure seam: source is synthetic, not a real
    # prior experiment. Existing native lifecycle suites cover actual Jobs.
    monkeypatch.setattr(predecessor,'verify_partial_predecessor',lambda value:value)
    monkeypatch.setattr(native,'_bootstrap_request',lambda *a,**kw:dict(software_only=True))
    from experiments.pirc17 import formal_training, formal_forecasts
    monkeypatch.setattr(formal_training,'fit_development_method',lambda *a,**kw:pytest.fail('re-fit method'))
    monkeypatch.setattr(formal_training,'fit_direct_dynamics',lambda *a,**kw:pytest.fail('re-fit terrain'))
    original_forecast_method=formal_forecasts.forecast_method
    monkeypatch.setattr(formal_forecasts,'forecast_method',lambda *a,**kw:pytest.fail('regenerated forecast'))
    with budget.Ledger.create(root,contract) as ledger:
        ledger.import_partial_floor(runtime)
        directory = root/'session-000001'; directory.mkdir()
        config = dict(schema_version=native.VERSION,directory=str(directory),job_name='PIRC17-FORMAL-'+'e'*32,
            worker_command=['SOFTWARE-ONLY'],ledger_directory=str(root),ledger_root_sha256=ledger.root_sha256,
            execution_sha256=args['execution']['sha256'])
        descriptor = native._write(directory/'session.json',config)
        start = 1_000_000_000  # Explicit synthetic clock, NOT actual restoration timing.
        control_record = envelope(dict(schema_version=control.VERSION+'-control',
            ledger_root_sha256=ledger.root_sha256,phase='input_qualification_and_binding',
            started_ns=start,phase_deadline_ns=start+contract['phase_caps_ns']['input_qualification_and_binding'],
            worker_job_name=config['job_name'],session_directory=str(directory),worker_command=config['worker_command']))
        key = control_record['sha256']
        ledger.open_control(key,phase='input_qualification_and_binding',credit_ns=5*budget.NANOSECONDS)
        (root/'controls').mkdir()
        control._publish(root/'controls',unpack(control_record))
        request = native._write(directory/'bootstrap-request.json',dict(schema_version=native.VERSION+'-bootstrap-request',
            session_sha256=descriptor['sha256'],control_sha256=key,started_ns=start,
            deadline_ns=unpack(control_record)['phase_deadline_ns']))
        native._write(directory/'ready.json',dict(schema_version=native.VERSION+'-ready',
            session_sha256=descriptor['sha256'],worker_pid=777))  # NOT native Job membership.
        options = dict(**{k:args[k] for k in ('protocol','execution','matrix')},
            authority=dict(approval_path=None,approval_sha256=approval,test_path=None,review_path=None,
                           journal_directory=root/'access'),release=root,snapshot=root,data_root=root,
            trajectory_path=root/'NO-RAW-TRAJECTORY',development_eligibility_path=root/'NO-TRAINING',
            partial_predecessor_reference=source_case['predecessor_ref'])
        worker = FormalWorker(config,contract,input_options=options)
        def software_current_inputs(self,value,*,output_directory):
            # Real typed CURRENT context but explicit synthetic guard seam;
            # test_partial_inputs separately exercises actual guard denial and
            # current access journals/full qualification reconstruction.
            assert value == binding
            self.attempted.update(self.work)
            output_directory.mkdir()
            path, context = publish(output_directory,unpack(case['context']))
            qpath, q = publish(output_directory/'qualification',dict(software_only=True,
                access_started_sha256=digest('SOFTWARE eligibility access seam')))
            maps = fixture_maps(unpack(args['input_identity']), {})
            # No query is made; preserve the saved complete static catalog.
            maps.catalog = deepcopy(args['map_catalog'])
            fit_inputs = RestoredFitInputs(source_case['fits'].inputs.encoders,args['prior'],args['input_identity'])
            manifest = dict(context_path=str(path),context_sha256=context['sha256'],qualification_path=str(qpath),qualification_sha256=q['sha256'])
            return inputs.PreparedInputWork(deepcopy(args),fit_inputs,maps,{},manifest,source_case['context_ref'])
        monkeypatch.setattr(inputs.InputWork,'restore_partial',software_current_inputs)
        output = directory/'bootstrap'; output.mkdir()
        before = ledger.summary()
        try:
            result = worker.bootstrap(output)
            record, manifest = predecessor._load(result)
            assert len(manifest['imported_entries']) == 28
            assert worker.completed == set(ledger._state.imported_success)
            assert len(worker.fits.models) == 26 and len(worker.fits.attempted) == 26
            assert source_case['work']['work_id'] in worker.forecasts.attempted
            assert source_case['work']['work_id'] in worker.saved.index
            restored = worker.saved.read(source_case['work'])
            assert restored.status == 'success' and restored.forecast.positions_m.shape == (512,4,2)
            assert manifest['new_fits'] == manifest['new_forecasts'] == 0
            assert not manifest['raw_sources_independently_reloaded']
            assert ledger.summary() == before  # Restoration cannot reserve/settle science.
            assert all(entry['source_completion'] == ledger._state.imported_success[wid]
                       for wid,entry in manifest['imported_entries'].items())
            barrier = native._write(directory/'bootstrap-barrier.json',dict(schema_version=native.VERSION+'-bootstrap-barrier',
                session_sha256=descriptor['sha256'],request_sha256=request['sha256'],control_sha256=key,
                worker_pid=777,status='success',value=result,error_type=None))
            observation = envelope(dict(schema_version=native.VERSION+'-bootstrap-observation',
                ledger_root_sha256=ledger.root_sha256,session_sha256=descriptor['sha256'],control_sha256=key,
                request_sha256=request['sha256'],barrier=barrier,started_ns=start,ended_ns=start+1,elapsed_ns=1,
                deadline_ns=unpack(control_record)['phase_deadline_ns'],worker_pid=777,
                accounting=dict(active_processes=2,total_processes=2)))
            # Original native-chain and CURRENT guard adapters are explicit
            # software seams. Actual closed/native/guard suites cover them;
            # this test integrates unmodified domain admission and cold readers.
            real_closed = formal_closed.ClosedOutputs
            source_items = {}
            for wid, entry in manifest['imported_entries'].items():
                artifact = entry['source_completion']['artifact']
                source_dir = source_case['root'] if artifact is None else (source_case['root']/artifact['path']).parent
                source_items[wid] = dict(directory=source_dir,manifest=dict(software_only=True),
                    **{k: entry['source_completion'][k] for k in
                    ('result_sha256','settlement_sha256','observation_sha256','controller_elapsed_ns')})
            class SoftwareOriginal:
                def read(self,work):return deepcopy(source_items[work['work_id']])
            def software_original(directory,*,contract,tip):
                if Path(directory) == source_case['root']:return SoftwareOriginal()
                return real_closed(directory,contract=contract,tip=tip)
            monkeypatch.setattr(formal_closed,'ClosedOutputs',software_original)
            monkeypatch.setattr(inputs,'read_input_work',lambda *a,**kw:(deepcopy(args),{},dict(software_guard_seam=True)))
            ticks=[]
            from experiments.pirc17.formal_results import ScientificResults
            parent=ScientificResults(ledger,**{k:args[k] for k in ('protocol','execution','matrix')},authority=options['authority'])
            from experiments.pirc17 import formal_metadata_batch
            assert getattr(worker.saved.import_bridge, '_metadata_batch', None) is None
            # A closing metadata failure must not call the parent handoff or
            # bind admission, even AFTER every per-item domain loop succeeds.
            original_finish = formal_metadata_batch._Batch.finish
            def closing_failure(self):
                original_finish(self)
                raise ValueError('SOFTWARE closing metadata rejection')
            with monkeypatch.context() as closing_patch:
                closing_patch.setattr(formal_metadata_batch._Batch, 'finish', closing_failure)
                with pytest.raises(ValueError, match='closing metadata rejection'):
                    admission.verify_manifest(unpack(observation),output,lambda:ticks.append(1),ledger=ledger,
                        **{k:args[k] for k in ('protocol','execution','matrix')},access_journal=root/'access',
                        on_verified=lambda **kw:pytest.fail('unclosed batch handed to parent'))
            assert ledger._state.partial_imports is None and parent.completed == set()
            verification=admission.verify_manifest(unpack(observation),output,lambda:ticks.append(1),ledger=ledger,
                **{k:args[k] for k in ('protocol','execution','matrix')},access_journal=root/'access',on_verified=parent.restore_imports)
            assert len(ticks)>=28
            assert parent.completed==set(ledger._state.imported_success) and len(parent.receipts)==26
            assert parent.saved is not worker.saved and len(parent.saved.index)==1
            proof_payload=dict(schema_version=control.VERSION+'-bootstrap-observation',ledger_root_sha256=ledger.root_sha256,
                control_sha256=key,native_observation=observation,restoration_verification=verification,checked_through_ns=start+2)
            proof=control._publish(root/'controls',proof_payload)
            row=dict(manifest_reference=result,bootstrap_observation_sha256=proof['sha256'])
            # Negative historical-proof mutations are rehashed; rejection is
            # semantic/provenance/timing, not just stale content digests.
            for mutation in ('root','control','pid','elapsed','credit','domain','barrier'):
                bad=deepcopy(proof_payload)
                if mutation=='root':bad['ledger_root_sha256']=digest('other root')
                elif mutation=='control':bad['control_sha256']=digest('other control')
                elif mutation=='domain':bad['restoration_verification']['imported_success_count']-=1
                elif mutation=='credit':bad['checked_through_ns']=start+5*budget.NANOSECONDS
                else:
                    obs=deepcopy(unpack(bad['native_observation']))
                    if mutation=='pid':obs['worker_pid']+=1
                    elif mutation=='elapsed':obs['elapsed_ns']+=1
                    else:
                        bar=deepcopy(unpack(obs['barrier']));bar['value']['content_sha256']=digest('other manifest')
                        obs['barrier']=envelope(bar)
                    bad['native_observation']=envelope(obs)
                bad_proof=control._publish(root/'controls',bad)
                with pytest.raises(ValueError):
                    admission.validate_binding(ledger._state,dict(row,bootstrap_observation_sha256=bad_proof['sha256']))
                assert ledger._state.partial_imports is None
            before_binding=ledger.summary()
            ledger.bind_partial_imports(result,bootstrap_observation_sha256=proof['sha256'])
            bound=ledger.summary()
            assert bound['generated_forecasts_reserved']==before_binding['generated_forecasts_reserved']
            assert bound['charged_ns_by_phase']==before_binding['charged_ns_by_phase']
            assert ledger.dispositions()==dict.fromkeys(manifest['imported_entries'],'success') | {
                w['work_id']:'unattempted' for w in contract['workloads'] if w['work_id'] not in manifest['imported_entries']}
            with pytest.raises(ValueError,match='once-only'):
                ledger.bind_partial_imports(result,bootstrap_observation_sha256=proof['sha256'])
            cold=real_closed(root,contract=contract,tip=ledger.tip)
            imported=cold.read(source_case['work'])
            assert imported['imported'] and imported['directory'].is_relative_to(root)
            assert imported['result_sha256']==source_items[source_case['work']['work_id']]['result_sha256']
            assert cold.entries[source_case['work']['work_id']]['settlement_sha256']==imported['settlement_sha256']
            assert imported['original_directory']==str(source_items[source_case['work']['work_id']]['directory'])
            assert not (root/'dispatches').exists()  # No new scientific native claims.
            from experiments.pirc17 import formal_reanalysis
            # This fixture's guard/qualification is explicit SOFTWARE ONLY;
            # exercise independent source reconstruction/cache, not authority.
            monkeypatch.setattr(formal_reanalysis,'qualification_evidence',lambda *a,**kw:dict(software_only=True))
            monkeypatch.setattr(formal_reanalysis,'_access',lambda *a,**kw:None)
            audit=formal_reanalysis.ReanalysisConsumers(closed=cold,**args,access_journal=root/'access')
            saved,scorer,scores,analyses,inventory,qualified,context_sha=audit.sources()
            assert saved.import_bridge is not None and saved.import_bridge is not worker.saved.import_bridge
            assert len(saved.receipts)==26 and len(saved.index)==1
            assert context_sha==case['context']['sha256']
            assert saved.read(source_case['work']).forecast.positions_m.shape==(512,4,2)
            assert sum(row['ledger_status']=='success' for row in inventory)==28
            # Forward ONE unfinished SYNTHETIC forecast, never a real final-eval
            # or redo of an imported success. Explicit authority seam; real
            # matrix/reservation/token/result/typed-domain paths are unchanged.
            ledger.close_control(key,observed_ns=0,evidence_sha256=proof['sha256'],reason='phase_complete')
            next_id=next(iter(ledger._state.carried_generation_reservations))
            work=worker.works[next_id]
            method_control=control._publish(root/'controls',dict(unpack(control_record),phase=work['phase']))
            ledger.open_control(method_control['sha256'],phase=work['phase'],credit_ns=5*budget.NANOSECONDS)
            reserved=ledger.reserve(next_id)
            monkeypatch.setattr(parent,'_check_authority',lambda phase:None) # NOT human authority.
            parent.authorize(ledger._state.work[next_id])
            monkeypatch.setattr(formal_forecasts,'forecast_method',original_forecast_method)
            (directory/'outputs').mkdir()
            next_output=directory/'outputs/000000'
            next_manifest=worker.forecasts.execute(work,output_directory=next_output)
            native._write(next_output/'result.json',dict(schema_version=native.VERSION+'-result',session_sha256=descriptor['sha256'],
                sequence=0,reservation_sha256=reserved['reservation_sha256'],work_id=next_id,value=next_manifest))
            validation=parent.validate(ledger._state.work[next_id],next_manifest,next_output)
            assert validation['details']['saved_forecast_restored'] and parent.pending['work']==work
            assert ledger.summary()['generated_forecasts_reserved']==2 # Old undispatched token used once.
            assert parent.completed==set(ledger._state.imported_success) # New item NOT falsely settled.
            entry=manifest['imported_entries'][source_case['work']['work_id']]
            path=Path(entry['manifest']['artifact_path'])
            # An extra owned file is not silently ignored by cold scoring.
            extra,_=publish(path.parent,dict(software_only_extra=True))
            with pytest.raises(ValueError,match='missing/extra'):cold.read(source_case['work'])
            extra.unlink()  # Only this known synthetic fixture file, not science.
            source_items[source_case['work']['work_id']]['settlement_sha256']=digest('changed original completion')
            with pytest.raises(ValueError,match='original closed completion'):cold.read(source_case['work'])
            with pytest.raises(ValueError,match='one explicit'):worker.bootstrap(output)
        finally: worker.close()
