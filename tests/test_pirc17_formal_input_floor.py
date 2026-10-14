"""Software budget wiring, not private input evidence or human authority.

The cost content pin and live consumer are explicitly replaced. The real
570.736s amount/categories are retained; no fixture is a production candidate.
"""
from copy import deepcopy

import pytest

from experiments.pirc17 import formal_budget as budget, formal_input_interruption as module
from experiments.pirc17.protocol_core import digest,envelope,unpack


@pytest.fixture
def post_read(tmp_path,monkeypatch):
    directory=tmp_path/'NEW-SOFTWARE-LEDGER'
    old=str(tmp_path/'ABSENT-NOT-ACTUAL-HISTORY')
    caps={module.PHASE:3600_000_000_000,'science':169200_000_000_000}
    works=[dict(work_id=digest('input-fixture'),phase=module.PHASE,max_active_ns=caps[module.PHASE],generated_forecasts=0),
        dict(work_id=digest('science-fixture'),phase='science',max_active_ns=100,generated_forecasts=1)]
    columns=lambda amount:{p:amount if p==module.PHASE else 0 for p in caps}
    c=dict(schema_version=module.costs.VERSION,ledger_directory=old,
        ledger_root_sha256=module.REGISTERED['expected_root_sha256'],head_sha256=module.REGISTERED['expected_head_sha256'],
        terminal_proof_sha256=module.REGISTERED['expected_terminal_proof_sha256'],
        protocol_sha256=digest('SOFTWARE PROTOCOL'),matrix_sha256=digest('SOFTWARE MATRIX'),
        execution_sha256=digest('SOFTWARE OLD EXECUTION'),approval_sha256=digest('NOT HUMAN OLD APPROVAL'),
        phase_caps_ns=caps,total_cap_ns=sum(caps.values()),charged_ns_by_phase=columns(570_736_000_000),
        measured_ns_by_phase=columns(536_204_000_000),conservatively_charged_ns_by_phase=columns(34_532_000_000),
        control_charged_ns_by_phase=columns(34_532_000_000),control_observed_ns_by_phase=columns(24_671_000_000),
        charged_total_ns=570_736_000_000,generated_forecasts_reserved=0,work_inventory_count=2,
        work_dispositions={works[0]['work_id']:'failure'},halted_reason='supervision_error',
        final_eval_reads=0,read_only=True,authorizes_execution=False,software_fixture_only=True)
    cost=envelope(c)
    monkeypatch.setattr(module,'REGISTERED_COST_SNAPSHOT',cost['sha256'])  # Not production proof.
    binding=envelope(dict(schema_version=module.VERSION,ledger_directory=old,cost_snapshot=cost,
        eligible_closed_input_interruption=True,process_tree_closed=True,read_only=True,authorizes_execution=False,
        generated_forecasts=0,predictive_model_fits=0,raw_sources_independently_reloaded=False,
        worker_training_state_retained=False,inventory_sha256=module.REGISTERED_INVENTORY,
        claim_inventory_sha256=module.REGISTERED_CLAIM_INVENTORY,bundle_sha256=module.REGISTERED_BUNDLE,
        original_predecessor_sha256=module.REGISTERED_PREDECESSOR,input_context_sha256=module.REGISTERED_CONTEXT,
        input_qualification_sha256=module.REGISTERED_QUALIFICATION,selected_samples=46,
        historical_access=deepcopy(module.REGISTERED_ACCESS),software_fixture_only=True))
    runtime=envelope(dict(protocol_sha256=c['protocol_sha256'],matrix_sha256=c['matrix_sha256'],
        ledger_directory=str(directory),predecessor=binding,software_fixture_only=True))
    contract=dict(schema_version=budget.VERSION,protocol_sha256=c['protocol_sha256'],matrix_sha256=c['matrix_sha256'],
        runtime_manifest_sha256=runtime['sha256'],execution_sha256=digest('SOFTWARE NEW EXECUTION'),
        approval_sha256=digest('NOT HUMAN NEW APPROVAL'),ledger_directory=str(directory),phase_caps_ns=caps,
        total_cap_ns=sum(caps.values()),max_generated_forecasts=1,max_attempts_per_item=1,workloads=works)
    def software_only(binding):
        assert unpack(binding)['software_fixture_only'] is True
        return unpack(binding)
    monkeypatch.setattr(module,'verify_input_interruption',software_only)
    return directory,contract,runtime


def test_full_cost_debit_once_and_original_balance_before_any_work(post_read,monkeypatch):
    directory,contract,runtime=post_read
    with budget.Ledger.create(directory,contract) as ledger:
        event=ledger.import_startup_floor(runtime)
        assert unpack(event)['index']==0
        s=ledger.summary()
        assert s['charged_ns_by_phase'][module.PHASE]==570_736_000_000
        assert s['measured_ns_by_phase'][module.PHASE]==536_204_000_000
        assert s['conservatively_charged_ns_by_phase'][module.PHASE]==34_532_000_000
        assert s['control_observed_ns_by_phase'][module.PHASE]==24_671_000_000
        assert s['remaining_ns_by_phase'][module.PHASE]==3029_264_000_000
        assert s['remaining_total_ns']==172229_264_000_000
        assert s['attempted_work_items']==s['generated_forecasts_reserved']==0
        with pytest.raises(ValueError,match='once-only'): ledger.import_startup_floor(runtime)
        root=ledger.root_sha256
    def forbidden(*a,**k): raise AssertionError('replay must not reinspect historical data/processes')
    monkeypatch.setattr(module,'verify_input_interruption',forbidden)
    monkeypatch.setattr(module.startup.native,'_query_job',forbidden)
    with budget.Ledger.open(directory,expected_root_sha256=root) as ledger:
        assert ledger.summary()==s
        reserve=ledger.reserve(contract['workloads'][0]['work_id'])
        assert reserve['reserved_ns']==3029_264_000_000
        ledger.settle(reserve['reservation_sha256'],status='failure',elapsed_ns=1_000_000_000,
            completion_evidence_sha256=digest('SOFTWARE RESULT'),result_sha256=None,reason='software fixture')
        assert ledger.summary()['charged_ns_by_phase'][module.PHASE]==571_736_000_000


@pytest.mark.parametrize('fault',['zero_cost','first_only','double_first','categories','zero_exposure','sample_count',
    'inventory','context','fit','forecast','permission','retained_training','raw_reload','old_execution','old_approval'])
def test_malformed_floor_rejected_before_any_debit(post_read,fault):
    directory,contract,runtime=post_read
    r=deepcopy(unpack(runtime));b=unpack(r['predecessor']);c=unpack(b['cost_snapshot'])
    if fault=='zero_cost':c['charged_total_ns']=0
    elif fault=='first_only':c['charged_total_ns']=40_423_000_000
    elif fault=='double_first':c['charged_total_ns']+=40_423_000_000
    elif fault=='categories':c['measured_ns_by_phase'][module.PHASE]+=1
    elif fault=='zero_exposure':b['historical_access']['historically_exposed']=False
    elif fault=='sample_count':b['selected_samples']=47
    elif fault=='inventory':b['inventory_sha256']=digest('other inventory')
    elif fault=='context':b['input_context_sha256']=digest('other context')
    elif fault=='fit':b['predictive_model_fits']=1
    elif fault=='forecast':b['generated_forecasts']=1
    elif fault=='permission':b['authorizes_execution']=True
    elif fault=='retained_training':b['worker_training_state_retained']=True
    elif fault=='raw_reload':b['raw_sources_independently_reloaded']=True
    elif fault=='old_execution':contract['execution_sha256']=c['execution_sha256']
    elif fault=='old_approval':contract['approval_sha256']=c['approval_sha256']
    b['cost_snapshot']=envelope(c);r['predecessor']=envelope(b);runtime=envelope(r)
    contract['runtime_manifest_sha256']=runtime['sha256']
    with budget.Ledger.create(directory,contract) as ledger:
        before=ledger.summary()
        with pytest.raises(ValueError):ledger.import_startup_floor(runtime)
        assert ledger.summary()==before and ledger.tip['event_count']==0


def test_unverified_live_history_never_debits_or_dispatches(post_read,monkeypatch):
    directory,contract,runtime=post_read
    def denied(*a,**k):raise ValueError('actual history closure not verified')
    monkeypatch.setattr(module,'verify_input_interruption',denied)
    with budget.Ledger.create(directory,contract) as ledger:
        before=ledger.summary()
        with pytest.raises(ValueError,match='closure not verified'):ledger.import_startup_floor(runtime)
        assert ledger.summary()==before and ledger.tip['event_count']==0
