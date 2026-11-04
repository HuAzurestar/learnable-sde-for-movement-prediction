"""Pilot sublimits in the existing BudgetLedger's authoritative event chain.

No second store or replenishable private balance. Reservations still debit the
original arm account. Every new-arm reservation requires the bound pilot.
"""
from datetime import datetime, timezone
import math

from infrastructure.research_store import ResearchError, digest, identifier


def policy(ledger, pilot_id, *, live=True):
    with ledger.store._read_transaction():
        result=_policy_in_scope(ledger,pilot_id,live=live)
        from .walking_execution import resolve_execution, activations
        if activations(ledger.store,pilot_id):
            result,_=resolve_execution(ledger.store,result)
        ledger.store._read_completion(
            lambda: _policy_in_scope(ledger,pilot_id,live=False),
            lambda: _policy_expiry(result,live=live))
        return result


def _policy_expiry(p, *, live=True):
    expiry=datetime.fromisoformat(p["expires_at"])
    ceiling=datetime.fromisoformat("2026-11-03T23:59:00+08:00")
    if expiry.tzinfo is None or expiry>ceiling or (live and expiry<=datetime.now(timezone.utc)):
        raise ResearchError("UNAUTHORIZED_DATA","pilot authorization expired or extended")


def _policy_in_scope(ledger, pilot_id, *, live=True):
    p=ledger.store._manifest("walking-pilot-"+identifier(pilot_id))
    if (p.get("schema_version")!="pirc26-walking-budget-v1" or p.get("pilot_id")!=pilot_id
            or p.get("global_ms")!=1800000 or p.get("training_ms")!=120000
            or p.get("final_test") is not False or p.get("motion")!="walking"
            or len(p.get("arms",{}))!=10 or len(set(p["arms"].values()))!=10
            or p["arms"].get("M0-UO2")!="pirc26-m0-o1" or p["arms"].get("M2-UO2")!="pirc26-m2-o1"
            or set(p["arms"])!={"M0-UO2","M2-UO2","A-S","A-N","B-S","B-N","C-T-S","C-T-N","C-R-S","C-R-N"}
            or type(p.get("speed_limit")) not in (int,float) or not math.isfinite(p["speed_limit"]) or p["speed_limit"]<=0):
        raise ResearchError("CONTRACT_MISMATCH","invalid frozen walking policy")
    approval=ledger.store._manifest(p["approval"]["manifest_id"])
    current_receipt=(approval.get("schema_version")=="pirc26-user-scoped-pilot-permission-v1"
        and approval.get("user")=="azurestarhu" and approval.get("decision_record")=="PIRC-26/DECISION-01@v1"
        and approval.get("pilot_study_id")==p["study_id"] and approval.get("final_test_authorization") is False
        and approval.get("public_export") is False and approval.get("movement_scope")=="walking"
        and approval.get("walking_speed_assumption",{}).get("U_m_s")==p["speed_limit"]
        and approval.get("walking_speed_assumption",{}).get("V_m_s")==p["speed_limit"]
        and approval.get("budget")=={"deadline_reset":False,"global_scientific_seconds":1800,"per_arm_training_seconds":120,"timeout_policy":"partial-results"})
    synthetic_receipt=(approval.get("decider")=="azurestarhu" and approval.get("decision")=="DECISION-01@v1"
        and approval.get("feature")=="PIRC-26"
        and approval.get("source_wording_sha256")=="ed9e71a39ca450e244851f1ab20ac24e6c1207132d8ee1b93ee9daedcc9ff00e"
        and approval.get("allowed_pilot")==pilot_id and p["study_id"]=="synthetic")
    if digest(approval)!=p["approval"]["sha256"] or not (current_receipt or synthetic_receipt):
        raise ResearchError("UNAUTHORIZED_DATA","pilot approval receipt differs")
    _policy_expiry(p,live=live)
    return p


def usage(ledger,pilot_id,reservations=None):
    with ledger.store._read_transaction():
        return _usage_in_scope(ledger,pilot_id,reservations)


def _usage_in_scope(ledger,pilot_id,reservations=None):
    reservations=ledger._state()[0] if reservations is None else reservations
    rows=[r for r in reservations.values() if r.get("pilot_id")==pilot_id]
    owner=sum(e["payload"]["elapsed_ms"] for e in ledger.store._events()
              if e["event_kind"] in {"PILOT_OWNER_COST","PILOT_OWNER_CHECKPOINT","PILOT_OWNER_TERMINAL"} and e["payload"]["pilot_id"]==pilot_id)
    def cost(r):
        return r["charged_ms"] if r["settled"] else r["reserved_ms"]
    result={"global_ms":owner+sum(cost(r) for r in rows),
            "training_ms":{arm:sum(cost(r) for r in rows if r["arm_id"]==arm and r["pilot_phase"]=="train")
                           for arm in {r["arm_id"] for r in rows}},
            "pending":any(not r["settled"] or r.get("monotonic_elapsed_ms") is None for r in rows) or bool(owner_opens(ledger.store,pilot_id)),
            "stage_pending":any(not r["settled"] or r.get("monotonic_elapsed_ms") is None for r in rows),"owner_ms":owner}
    states=_grid_state_in_scope(ledger,pilot_id)
    if states:
        limits={}
        for inv,st in states.items():
            receipts=st["receipts"]
            if st["terminal"]:
                limits[inv]=receipts[-1]["payload"]["billing"]["limits_snapshot"]
            else:
                live=getattr(_owner,"grid",None)
                x=live.sample() if live is not None and live.scope==(ledger.store,pilot_id) and live.invocation_id==inv else (receipts[-1]["payload"]["billing"]["sample_ns"] if receipts else 0)
                limits[inv]=_limits(ledger.store._events(),st["open"],x,arms=st["arms"])
                if live is None: limits[inv]["unknown"]=True
        result["billing_limits"]=limits
    return result


def registered_binding(ledger,run):
    """Resolve authority from the registered run, never a caller's phase label."""
    with ledger.store._read_transaction():
        return _registered_binding_in_scope(ledger,run)


def _registered_binding_in_scope(ledger,run):
    cell=run["cell"]
    if any(k in cell for k in ("pilot_id","pilot_arm","pilot_stage")) and not cell.get("pilot_id"):
        raise ResearchError("UNAUTHORIZED_DATA","registered pilot markers require a pilot identity")
    study=ledger.store._manifest("study-"+run["study_id"])
    if (cell not in study["spec"]["cells"] or run["cell_hash"]!=digest(cell)
            or run["spec_hash"]!=study["spec_hash"] or run["arm_id"]!=cell["arm_id"]):
        raise ResearchError("UNAUTHORIZED_DATA","registered run/cell binding differs")
    bound=[]
    for event in ledger.store._events():
        if event["event_kind"]=="MANIFEST" and event["payload"]["object_id"].startswith("walking-pilot-"):
            p=ledger.store._manifest(event["payload"]["object_id"])
            new={v for k,v in p.get("arms",{}).items() if k not in {"M0-UO2","M2-UO2"}}
            if run["study_id"]==p.get("study_id") or run["arm_id"] in new:
                bound.append(p["pilot_id"])
    if cell.get("pilot_id") is not None:
        bound.append(cell["pilot_id"])
    if not bound:
        return None
    if len(set(bound))!=1:
        raise ResearchError("UNAUTHORIZED_DATA","registered pilot binding is ambiguous")
    p=policy(ledger,bound[0])
    if (cell.get("pilot_id")!=p["pilot_id"] or run["study_id"]!=p["study_id"]
            or cell.get("pilot_arm") not in p["arms"]
            or p["arms"][cell["pilot_arm"]]!=run["arm_id"]
            or cell.get("execution_id")!=p.get("execution_id")
            or cell.get("pilot_stage") not in {"source","prepare","train","predict","h2"}):
        raise ResearchError("UNAUTHORIZED_DATA","registered pilot study/arm/stage is restricted")
    return p


def check_reservation(ledger,run,budget,requested,reservations,*,walking_clock=None,walking_allocation_ms=None):
    p=registered_binding(ledger,run)
    if p is None:
        if walking_clock is not None or walking_allocation_ms is not None:
            raise ResearchError("CONTRACT_MISMATCH","nonwalking grid clock")
        if budget.pilot_id is not None:
            raise ResearchError("UNAUTHORIZED_DATA","nonpilot cell cannot claim a pilot")
        return {}
    is_grid=p.get("billing_basis")==GRID_BASIS
    if is_grid!=(walking_clock is not None) or is_grid!=(walking_allocation_ms is not None):
        raise ResearchError("CONTRACT_MISMATCH","registered billing basis differs")
    if is_grid:
        walking_clock.check(ledger,p["pilot_id"])
        _require(type(walking_allocation_ms) is int and walking_allocation_ms==requested)
    phase="prepare" if run["cell"]["pilot_stage"]=="source" else run["cell"]["pilot_stage"]
    if budget.pilot_id!=p["pilot_id"] or budget.pilot_phase!=phase or budget.category!="pilot" or budget.allocation_only:
        raise ResearchError("UNAUTHORIZED_DATA","budget differs from registered restricted pilot phase")
    u=usage(ledger,p["pilot_id"],reservations)
    check_owner_admission(ledger.store,p["pilot_id"])
    if u["stage_pending"]:
        raise ResearchError("BUDGET_BUSY","pilot is serial; unresolved costs remain")
    if requested>p["global_ms"]-u["global_ms"]:
        raise ResearchError("BUDGET_EXHAUSTED","pilot global limit")
    if phase=="train" and requested>p["training_ms"]-u["training_ms"].get(run["arm_id"],0):
        raise ResearchError("BUDGET_EXHAUSTED","pilot cumulative training limit")
    grid_fields={}
    if is_grid:
        available=remaining_ns(ledger,p["pilot_id"],run["arm_id"],phase,grid=walking_clock)
        if requested*U>available: raise ResearchError("BUDGET_EXHAUSTED","exact grid allocation exceeded")
        st=walking_clock.state(ledger); _require(st["receipts"],"checkpoint required before reserve")
        x=walking_clock.sample()
        grid_fields["billing"]={**walking_clock.common(ledger),"attempt_id":run["_grid_attempt_id"],"run_id":run["run_id"],
            "arm_id":run["arm_id"],"pilot_phase":phase,"allocation_ms":requested,"admission_sample_ns":x,
            "admission_head":_ref(ledger.store._events()[-1]),"owner_checkpoint_ref":_ref(st["receipts"][-1]),
            "limits_snapshot":_limit_usage_in_scope(ledger,p["pilot_id"],sample_ns=x,grid=walking_clock)}
    return {**grid_fields,"pilot_id":p["pilot_id"],"pilot_phase":phase,"execution_deadline":p["expires_at"],
            **({"execution_id":p["execution_id"],"original_study_id":p["original_study_id"]} if p.get("execution_id") else {})}


def remaining(ledger,pilot_id,arm_id,phase):
    with ledger.store._read_transaction():
        p=policy(ledger,pilot_id); u=usage(ledger,pilot_id)
        check_owner_admission(ledger.store,pilot_id)
        if (ledger.store.path/"recovery-hold.json").exists() or u["stage_pending"]:
            raise ResearchError("RECOVERY_REQUIRED","pilot has unknown/held costs")
        available=p["global_ms"]-u["global_ms"]
        if phase=="train":
            available=min(available,p["training_ms"]-u["training_ms"].get(arm_id,0))
        available=min(available,ledger._balance(arm_id)["remaining_ms"])
        if ledger._balance(arm_id)["closed"]:
            available=0
        return max(0,available)/1000


def charge_owner(ledger,pilot_id,elapsed_ms,event_id):
    """Charge preparation/control/report work excluding already billed workers.

    Overruns are retained, never truncated to the remaining allowance. Once a
    metadata operation consumes the remainder, no later worker can be funded.
    """
    if type(elapsed_ms) is not int or elapsed_ms<0:
        raise ResearchError("CONTRACT_MISMATCH","invalid measured owner cost")
    with ledger.store._read_transaction():
        current=policy(ledger,pilot_id,live=False)
        if current.get("billing_basis")==GRID_BASIS:
            raise ResearchError("CONTRACT_MISMATCH","grid owner costs require invocation receipts")
        return ledger.store._append("PILOT_OWNER_COST",{"pilot_id":pilot_id,"elapsed_ms":elapsed_ms},event_id)


# The capability exists only in the controller thread while it owns the OS
# pilot lock. It cannot be supplied through config, a reservation or a worker.
import os
import threading
from contextlib import contextmanager
_owner = threading.local()


def owner_opens(store,pilot_id):
    _grid_state_in_scope(__import__("application.research_budget",fromlist=["BudgetLedger"]).BudgetLedger(store),pilot_id)
    opened={}
    for e in store._events():
        p=e["payload"]
        if p.get("pilot_id")!=pilot_id:
            continue
        if e["event_kind"]=="PILOT_OWNER_OPEN":
            opened[p["invocation_id"]]=p
        elif e["event_kind"]=="PILOT_OWNER_TERMINAL":
            opened.pop(p["invocation_id"],None)
    return opened


def check_owner_admission(store,pilot_id):
    opened=owner_opens(store,pilot_id)
    capability=getattr(_owner,"capability",None)
    if opened and (capability is None or capability[:3]!=(os.getpid(),store,pilot_id)
                   or set(opened)!={capability[3]}):
        raise ResearchError("RECOVERY_REQUIRED","unresolved controller costs require exact recovery evidence")


@contextmanager
def _owner_session_unlocked(ledger,pilot_id,execution_id,invocation_id,*,config=None,grid=None):
    store=ledger.store
    with store._read_transaction():
        _policy_in_scope(ledger,pilot_id,live=False)
        if config is not None:
            # Establish a trusted bucket/config identity before opening costs.
            # Grant/code/expiry preflight remains inside the metered session.
            from .walking_execution import activations,reference
            active=activations(store,pilot_id)
            registered=(reference(store,reference(store,active[-1]["execution_ref"])["config_ref"])
                        if active else store._manifest("walking-config-"+pilot_id))
            if registered!=config or registered.get("execution_id")!=execution_id:
                raise ResearchError("UNAUTHORIZED_DATA","controller configuration is not the registered pilot identity")
        if grid is not None and _usage_in_scope(ledger,pilot_id)["stage_pending"]:
            raise ResearchError("RECOVERY_REQUIRED","prior stage costs unresolved")
        if owner_opens(store,pilot_id):
            raise ResearchError("RECOVERY_REQUIRED","previous controller has unknown costs")
        is_grid=config is not None and config.get("schema_version")=="pirc26-walking-config-grid-v1"
        if is_grid != (grid is not None):
            raise ResearchError("CONTRACT_MISMATCH","config and live grid must agree")
        payload={"pilot_id":pilot_id,"execution_id":execution_id,"invocation_id":invocation_id}
        if grid is not None:
            _require(isinstance(grid,GridClock) and grid.invocation_id==invocation_id and grid.scope is None)
            _require(active and registered==config and reference(store,active[-1]["execution_ref"])["schema"]=="walking-execution-grid-v1")
            grid.scope=(store,pilot_id)
            payload["billing"]={"schema_version":1,"basis":GRID_BASIS,"store_id":store.store_id,"pilot_id":pilot_id,
                "execution_id":execution_id,"study_id":config["study_id"],"invocation_id":invocation_id,"owner_open_ref":None,
                "clock_kind":"python-monotonic-ns","unit_ns_per_ms":U,"origin_ns":grid.origin_ns,"sample_ns":0,
                "code_hash":reference(store,active[-1]["execution_ref"])["code_hash"],"config_sha256":digest(config),
                "execution_ref":active[-1]["execution_ref"],"retained_prefix":_ref(store._events()[-1]),"limit_basis":LIMIT_BASIS}
        store._append("PILOT_OWNER_OPEN",payload,"owner-open-"+invocation_id)
    _owner.capability=(os.getpid(),store,pilot_id,invocation_id)
    _owner.grid=grid
    try:
        yield
    finally:
        del _owner.capability
        del _owner.grid
        if grid is not None: grid.scope=None


class OwnerMeter:
    def __new__(cls,*args,grid=None,**kwargs):
        return object.__new__(GridOwnerMeter if cls is OwnerMeter and grid is not None else cls)

    def __init__(self,ledger,config,start,clock,invocation_id,*,grid=None):
        self.ledger=ledger; self.config=config; self.start=start; self.clock=clock
        self.invocation_id=invocation_id; self.billed=0; self.previous=None; self.sequence=0
        self.stages={}; self.pending=None; self.unknown=False

    def sample(self):
        return max(0,math.ceil((self.clock()-self.start)*1000)-sum(self.stages.values()))

    def stage(self,attempt_id):
        with self.ledger.store._read_transaction():
            paid=[r for r in self.ledger._state()[0].values() if r["attempt_id"]==attempt_id]
            if any(not r["settled"] or r.get("monotonic_elapsed_ms") is None for r in paid):
                self.unknown=True
                raise ResearchError("RECOVERY_REQUIRED","native stage cost is unknown")
            for r in paid:
                self.stages[r["reservation_id"]]=r["charged_ms"]

    def checkpoint(self):
        if self.unknown:
            raise ResearchError("RECOVERY_REQUIRED","owner difference is unknown")
        measured=self.sample()
        event_id="owner-checkpoint-"+digest([self.config["pilot_id"],self.config.get("execution_id"),self.invocation_id,self.sequence])
        if self.pending is None:
            self.pending=(measured,self.payload(measured))
        measured,payload=self.pending
        with self.ledger.store._read_transaction():
            _policy_in_scope(self.ledger,self.config["pilot_id"],live=False)
            event=self.ledger.store._append("PILOT_OWNER_CHECKPOINT",payload,event_id)
        self.billed=measured; self.previous=event["event_id"]; self.sequence+=1; self.pending=None

    def payload(self,measured):
        if measured<self.billed:
            raise ResearchError("RECOVERY_REQUIRED","owner cumulative clock regressed")
        return {"pilot_id":self.config["pilot_id"],"execution_id":self.config.get("execution_id"),
            "invocation_id":self.invocation_id,"elapsed_ms":measured-self.billed,
            "cumulative_owner_ms":measured,"previous_receipt":self.previous,
            "stage_deductions_ms":dict(self.stages)}

    def terminal(self,report,save):
        store=self.ledger.store
        terminal_id="owner-terminal-"+self.invocation_id
        with store._read_transaction():
            _policy_in_scope(self.ledger,self.config["pilot_id"],live=False)
            u=_usage_in_scope(self.ledger,self.config["pilot_id"])
            if u["stage_pending"] or self.unknown or self.pending is not None:
                raise ResearchError("RECOVERY_REQUIRED","cannot finalize unknown stage costs")
            report.update(settlement_pending=True,terminal_receipt_id=terminal_id,
                billing_boundary="SOL-015-v1",budget=u,
                budget_as_of={"sequence":len(store._events()),"hash":store._events()[-1]["hash"]},
                settlement_tail={"pilot_charged":False,"included_in_qualification_wall":True,
                    "operations":["terminal append","exit full-chain verification/unlock","return/preconstructed CLI output"]})
            report["cli_output"]=__import__("json").dumps({"status":report["status"],"report":report["report_path"],
                "terminal_receipt_id":terminal_id,"arms":len(report["rows"])})
            raw=save()
            report_hash=__import__("hashlib").sha256(raw).hexdigest()
            # This is t*: after ALL ordinary business I/O and encoding.
            measured=self.sample()
            payload=self.payload(measured)
            payload.update(report_sha256=report_hash,billing_boundary="SOL-015-v1",
                global_ms=u["global_ms"]+payload["elapsed_ms"],training_ms=u["training_ms"])
            event=store._append("PILOT_OWNER_TERMINAL",payload,terminal_id)
        # Successful return means the transaction's physical exit check passed.
        return event


def verified_report(ledger,pilot_id,report_path):
    """A provisional file alone is never a final-cost authority."""
    import json,hashlib
    from pathlib import Path
    with ledger.store._read_transaction():
        _policy_in_scope(ledger,pilot_id,live=False)
        raw=Path(report_path).read_bytes(); report=json.loads(raw)
        matches=[e for e in ledger.store._events() if e["event_id"]==report.get("terminal_receipt_id")]
        if (len(matches)!=1 or matches[0]["event_kind"]!="PILOT_OWNER_TERMINAL"
                or matches[0]["payload"]["pilot_id"]!=pilot_id
                or matches[0]["payload"]["report_sha256"]!=hashlib.sha256(raw).hexdigest()):
            raise ResearchError("RECOVERY_REQUIRED","provisional report lacks verified terminal receipt")
        _grid_state_in_scope(ledger,pilot_id)
        b=matches[0]["payload"].get("billing")
        if b and (report.get("billing_basis")!=GRID_BASIS or report.get("billing_schema_version")!=1 or report.get("owner_open_ref")!=b["owner_open_ref"]):
            raise ResearchError("CONTRACT_MISMATCH","report grid identity differs")
        return {"report":report,"receipt":matches[0]}

# SOL-018: integer invocation partition. All durations use this live clock.
U=1000000
GRID_BASIS='walking-invocation-grid-v1'
LIMIT_BASIS='retained-upper-plus-exact-ns-v1'
GRID_FIELDS=('billing_basis','limit_basis','billing_decision_ref')


def _integer(value):
    if type(value) is not int or not 0<=value<=2**63-1:
        raise ResearchError('CONTRACT_MISMATCH','invalid grid integer')
    return value


def _q(value):
    return (_integer(value)+U-1)//U


def deadline_float(value):
    _integer(value)
    result=value/1000000000
    while result.as_integer_ratio()[0]*1000000000>value*result.as_integer_ratio()[1]:
        result=math.nextafter(result,-math.inf)
    return result


def _ref(event):
    return {'event_id':event['event_id'],'hash':event['hash']}


_COMMON={'schema_version','basis','store_id','pilot_id','execution_id','study_id','invocation_id','owner_open_ref'}
_IDENT={'attempt_id','run_id','arm_id','pilot_phase'}
_INTERVAL={'s_ns','r_ns','duration_ns','q_s_ms','q_r_ms','allocated_ms'}


def _require(ok,message='grid evidence differs'):
    if not ok: raise ResearchError('CONTRACT_MISMATCH',message)


class GridClock:
    def __init__(self,origin_ns,clock_ns,invocation_id):
        self.origin_ns=_integer(origin_ns); self.clock_ns=clock_ns
        self.invocation_id=identifier(invocation_id); self.last=0
        self.pid=os.getpid(); self.thread=threading.get_ident(); self.scope=None

    def sample(self):
        _require(self.pid==os.getpid() and self.thread==threading.get_ident(),'grid thread differs')
        value=_integer(self.clock_ns()-self.origin_ns)
        _require(value>=self.last,'grid clock regressed'); self.last=value
        return value

    def check(self,ledger,pilot_id):
        _require(self.scope==(ledger.store,pilot_id) and getattr(_owner,'grid',None) is self
            and getattr(_owner,'capability',None)==(os.getpid(),ledger.store,pilot_id,self.invocation_id),
            'live owner capability required')
        self.sample()

    def state(self,ledger):
        self.check(ledger,self.scope[1])
        st=_grid_state_in_scope(ledger,self.scope[1])[self.invocation_id]
        _require(not st['terminal'],'owner invocation is terminal')
        return st

    def common(self,ledger):
        st=self.state(ledger)
        return {k:v for k,v in st['open']['payload']['billing'].items() if k in _COMMON} | {'owner_open_ref':_ref(st['open'])}

    def no_stage(self,ledger,reservation,reason):
        st=self.state(ledger); rid=reservation['reservation_id']
        _require(rid not in st['stages'] and rid not in st['stops'])
        return {**self.common(ledger),**{k:reservation[k] for k in _IDENT},
            **dict.fromkeys(_INTERVAL), 'allocated_ms':0,'stage_open_ref':None,'stop_ref':None,
            'interval_kind':'no-stage','no_interval_reason':reason}

    def interval(self,ledger,reservation):
        st=self.state(ledger); rid=reservation['reservation_id']
        if rid not in st['stops']:
            return {**self.common(ledger),**{k:reservation[k] for k in _IDENT},**dict.fromkeys(_INTERVAL),
                'stage_open_ref':_ref(st['stages'][rid]) if rid in st['stages'] else None,
                'stop_ref':None,'interval_kind':'unknown','no_interval_reason':None}
        stop=st['stops'][rid]; b=stop['payload']['billing']
        return {**self.common(ledger),**{k:reservation[k] for k in _IDENT},**{k:b[k] for k in _INTERVAL},
            'stage_open_ref':b['stage_open_ref'],'stop_ref':_ref(stop),'interval_kind':'confirmed','no_interval_reason':None}

    def stop(self,ledger,reservation,r_ns,*,native):
        st=self.state(ledger); rid=reservation['reservation_id']
        if rid in st['stops']: return st['stops'][rid]
        sb=st['stages'][rid]['payload']['billing']; s=sb['s_ns']; _integer(r_ns); _require(r_ns>=s)
        b={**self.common(ledger),**{k:reservation[k] for k in _IDENT},'reservation_id':rid,
            'stage_open_ref':_ref(st['stages'][rid]),'s_ns':s,'r_ns':r_ns,'duration_ns':r_ns-s,
            'q_s_ms':_q(s),'q_r_ms':_q(r_ns),'allocated_ms':_q(r_ns)-_q(s),
            'r_kind':'native-stop-confirmed' if native else 'no-tree-launched'}
        return ledger.store._append('WORKER_TREE_STOPPED' if native else 'WALKING_STAGE_STOPPED_NO_LAUNCH',
            {'attempt_id':reservation['attempt_id'],'reservation_id':rid,'observed_elapsed_ms':_q(r_ns-s),
             'confirmation':'native-job-or-process-group-no-running-descendants' if native else 'no-tree-launched','billing':b},
            'walking-stage-stop-'+rid)


def _grid_state_in_scope(ledger,pilot_id):
    """Reconstruct grid intervals from the same physically verified event chain."""
    states={}; seen={}; reserve={}; grid_executions=set(); grid_pilots=set()
    def reference(ref,kind=None):
        _require(type(ref) is dict and set(ref)=={'event_id','hash'})
        event=seen.get(ref['event_id']); _require(event is not None and event['hash']==ref['hash']
            and (kind is None or event['event_kind']==kind),'invalid grid reference')
        return event
    for event in ledger.store._events():
        p=event['payload']; kind=event['event_kind']; b=p.get('billing')
        if kind=='WALKING_EXECUTION_ACTIVATED':
            from .walking_execution import reference as manifest
            binding=manifest(ledger.store,p['execution_ref'])
            if binding.get('schema')=='walking-execution-grid-v1':
                grid_executions.add(binding['execution_id']); grid_pilots.add(binding['budget_pilot_id'])
        if kind=='RESERVE': reserve[p['reservation_id']]=event
        if b is None:
            _require(not(kind=='PILOT_OWNER_COST' and p.get('pilot_id') in grid_pilots),'legacy owner cost after grid activation')
            _require(not(kind in {'RESERVE','SETTLE','PILOT_OWNER_OPEN','PILOT_OWNER_CHECKPOINT','PILOT_OWNER_TERMINAL'} and p.get('execution_id') in grid_executions),'registered grid basis removed')
            _require(not(kind in {'WALKING_STAGE_OPEN','WORKER_TREE_STOPPED','WALKING_STAGE_STOPPED_NO_LAUNCH'} and reserve.get(p.get('reservation_id'),{}).get('payload',{}).get('billing')),'stage billing removed')
            _require(not(kind=='SETTLE' and reserve.get(p['reservation_id'],{}).get('payload',{}).get('billing')),'grid basis removed')
            _require(not(kind.startswith('PILOT_OWNER_') and p.get('invocation_id') in states),'owner basis removed')
            seen[event['event_id']]=event; continue
        _require(type(b) is dict and _COMMON<=set(b) and type(b['schema_version']) is int and b['schema_version']==1
            and b['basis']==GRID_BASIS and b['store_id']==ledger.store.store_id)
        for key in ('store_id','pilot_id','execution_id','study_id','invocation_id'): identifier(b[key])
        inv=b['invocation_id']
        for key in ('elapsed_ms','cumulative_owner_ms','global_ms','reserved_ms','charged_ms','monotonic_elapsed_ms'):
            if key in p and p[key] is not None: _integer(p[key])
        for key in ('schema_version','unit_ns_per_ms','origin_ns','sample_ns','allocation_ms','admission_sample_ns','s_ns','r_ns','duration_ns','q_s_ms','q_r_ms','allocated_ms','deadline_ns','sequence','q_sample_ms','stage_sum_ms','owner_exact_ns'):
            if key in b and b[key] is not None: _integer(b[key])
        if kind.startswith('PILOT_OWNER_'):
            _require(p.get('pilot_id')==b['pilot_id'] and p.get('execution_id')==b['execution_id'] and p.get('invocation_id')==inv)
        if kind=='PILOT_OWNER_OPEN':
            _require(event['event_id']=='owner-open-'+inv)
            _require(set(b)==_COMMON|{'clock_kind','unit_ns_per_ms','origin_ns','sample_ns','code_hash','config_sha256','execution_ref','retained_prefix','limit_basis'})
            _require(inv not in states and b['owner_open_ref'] is None and b['sample_ns']==0
                and b['unit_ns_per_ms']==U and b['clock_kind']=='python-monotonic-ns' and b['limit_basis']==LIMIT_BASIS)
            _integer(b['origin_ns']); _require(seen and b['retained_prefix']==_ref(list(seen.values())[-1]))
            from .walking_execution import reference as manifest
            binding=manifest(ledger.store,b['execution_ref']); cfg=manifest(ledger.store,binding['config_ref'])
            _require(binding['schema']=='walking-execution-grid-v1' and binding['execution_id']==b['execution_id']
                and binding['study_id']==b['study_id'] and binding['budget_pilot_id']==b['pilot_id']
                and binding['code_hash']==b['code_hash'] and digest(cfg)==b['config_sha256'])
            states[inv]={'open':event,'res':{},'stages':{},'stops':{},'settles':{},'receipts':[],'terminal':False,'arms':set(cfg['arms'].values())}
        else:
            _require(inv in states); st=states[inv]; ob=st['open']['payload']['billing']
            _require(not st['terminal'] and b['owner_open_ref']==_ref(st['open']) and all(b[k]==ob[k] for k in _COMMON-{'owner_open_ref'}))
            if kind=='RESERVE':
                _require(p['reservation_id']==digest([ledger.store.store_id,p['attempt_id']]) and event['event_id']=='reserve-'+p['reservation_id'])
                _require(set(b)==_COMMON|_IDENT|{'allocation_ms','admission_sample_ns','admission_head','owner_checkpoint_ref','limits_snapshot'})
                _require(all(b[k]==p[k] for k in _IDENT) and _integer(b['allocation_ms'])>0 and b['allocation_ms']==p['reserved_ms'])
                _integer(b['admission_sample_ns']); reference(b['admission_head'])
                cp=reference(b['owner_checkpoint_ref'],'PILOT_OWNER_CHECKPOINT')
                _require(st['receipts'] and cp==st['receipts'][-1] and set(st['res'])==set(st['settles']))
                _require(digest(b['limits_snapshot'])==digest(_limits(list(seen.values()),st['open'],b['admission_sample_ns'],arms=st['arms'])))
                st['res'][p['reservation_id']]=event
            elif kind in {'WALKING_STAGE_OPEN','WORKER_TREE_STOPPED','WALKING_STAGE_STOPPED_NO_LAUNCH','SETTLE'}:
                rid=p['reservation_id']; _require(rid in st['res']); r=st['res'][rid]['payload']
                if 'reservation_id' in b: _require(b['reservation_id']==rid)
                _require(all(b[k]==r[k] for k in _IDENT))
                if kind=='WALKING_STAGE_OPEN':
                    _require(event['event_id']=='walking-stage-open-'+rid)
                    _require(set(b)==_COMMON|_IDENT|{'reservation_id','s_ns','q_s_ms','allocation_ms','deadline_ns','reservation_ref','previous_stage_ref'})
                    _require(rid not in st['stages'] and rid not in st['settles'] and all(k in st['settles'] for k in st['stages']))
                    s=_integer(b['s_ns']); _require(not st['receipts'] or s>=st['receipts'][-1]['payload']['billing']['sample_ns'])
                    _require(b['q_s_ms']==_q(s) and b['allocation_ms']==r['reserved_ms']
                        and s<=_integer(b['deadline_ns'])<=s+r['reserved_ms']*U and b['reservation_ref']==_ref(st['res'][rid]))
                    prior=list(st['settles'].values()); _require(b['previous_stage_ref']==(_ref(prior[-1]) if prior else None))
                    _require(not prior or s>=max(v['payload']['billing']['r_ns'] or 0 for v in prior))
                    st['stages'][rid]=event
                elif kind!='SETTLE':
                    _require(set(b)==_COMMON|_IDENT|_INTERVAL|{'reservation_id','stage_open_ref','r_kind'})
                    _require(rid in st['stages'] and rid not in st['stops'])
                    sb=st['stages'][rid]['payload']['billing']; _require(b['stage_open_ref']==_ref(st['stages'][rid]) and b['s_ns']==sb['s_ns'])
                    for k in _INTERVAL: _integer(b[k])
                    _require(b['r_ns']>=b['s_ns'] and b['duration_ns']==b['r_ns']-b['s_ns'] and b['q_s_ms']==_q(b['s_ns'])
                        and b['q_r_ms']==_q(b['r_ns']) and b['allocated_ms']==b['q_r_ms']-b['q_s_ms'])
                    _require(b['r_kind']==('native-stop-confirmed' if kind=='WORKER_TREE_STOPPED' else 'no-tree-launched'))
                    if kind=='WALKING_STAGE_STOPPED_NO_LAUNCH':
                        _require(not any(v['event_kind']=='WORKER_STARTED' and v['payload'].get('reservation_id')==rid for v in seen.values()))
                    st['stops'][rid]=event
                else:
                    _require(event['event_id']=='settle-'+rid)
                    _require(set(b)==_COMMON|_IDENT|_INTERVAL|{'stage_open_ref','stop_ref','interval_kind','no_interval_reason'})
                    _require(rid not in st['settles'])
                    if b['interval_kind']=='confirmed':
                        _require(rid in st['stops'] and b['stop_ref']==_ref(st['stops'][rid]))
                        sb=st['stops'][rid]['payload']['billing']
                        _require(all(b[k]==sb[k] for k in _INTERVAL|{'stage_open_ref'}) and b['no_interval_reason'] is None
                            and p['charged_ms']==b['allocated_ms']==p['monotonic_elapsed_ms'])
                    elif b['interval_kind']=='no-stage':
                        _require(not any(e['event_kind']=='WORKER_STARTED' and e['payload'].get('reservation_id')==rid for e in seen.values()) and rid not in st['stages'] and b['allocated_ms']==p['charged_ms']==p['monotonic_elapsed_ms']==0
                            and b['no_interval_reason'] in {'queue-rejected-before-s','queue-expired-before-s','command-failed-before-s'}
                            and all(b[k] is None for k in (_INTERVAL-{'allocated_ms'})|{'stage_open_ref','stop_ref'}))
                    else:
                        _require(b['interval_kind']=='unknown' and p['monotonic_elapsed_ms'] is None and b['allocated_ms'] is None
                            and all(b[k] is None for k in (_INTERVAL-{'allocated_ms'})|{'stop_ref','no_interval_reason'}))
                    st['settles'][rid]=event
            elif kind in {'PILOT_OWNER_CHECKPOINT','PILOT_OWNER_TERMINAL'}:
                extra={'sequence','sample_ns','q_sample_ms','stage_refs','stage_sum_ms','owner_exact_ns','previous_receipt_ref','checkpoint_kind'}
                if kind.endswith('TERMINAL'): extra|={'limits_snapshot','sample_role'}
                _require(set(b)==_COMMON|extra and set(st['res'])==set(st['settles'])
                    and all(v['payload']['monotonic_elapsed_ms'] is not None for v in st['settles'].values()))
                expected_id=('owner-terminal-'+inv if kind.endswith('TERMINAL') else 'owner-checkpoint-'+digest([b['pilot_id'],b['execution_id'],inv,b['sequence']]))
                _require(event['event_id']==expected_id)
                x=_integer(b['sample_ns']); _require(not st['receipts'] or x>=st['receipts'][-1]['payload']['billing']['sample_ns'])
                paid=list(st['settles'].values()); charges={k:v['payload']['charged_ms'] for k,v in st['settles'].items()}
                duration=sum(v['payload']['billing']['duration_ns'] or 0 for v in paid); total=sum(charges.values())
                _require(b['stage_refs']==[_ref(v) for v in paid] and b['sequence']==len(st['receipts'])
                    and all(x>=(v['payload']['billing']['r_ns'] or 0) for v in paid))
                prev=st['receipts'][-1] if st['receipts'] else None; old=prev['payload']['cumulative_owner_ms'] if prev else 0
                _require(b['q_sample_ms']==_q(x) and b['stage_sum_ms']==total and b['owner_exact_ns']==x-duration
                    and p['stage_deductions_ms']==charges and p['cumulative_owner_ms']==_q(x)-total
                    and p['elapsed_ms']==_q(x)-total-old and p['elapsed_ms']>=0
                    and b['previous_receipt_ref']==(_ref(prev) if prev else None)
                    and p['previous_receipt']==(prev['event_id'] if prev else None)
                    and b['checkpoint_kind']==('terminal' if kind.endswith('TERMINAL') else 'checkpoint'))
                if kind.endswith('TERMINAL'):
                    expected_global=sum(v['payload']['elapsed_ms'] for v in seen.values() if v['event_kind'] in {'PILOT_OWNER_COST','PILOT_OWNER_CHECKPOINT','PILOT_OWNER_TERMINAL'} and v['payload']['pilot_id']==b['pilot_id'])+p['elapsed_ms']
                    final_rows={v['payload']['reservation_id']:v['payload'] for v in seen.values() if v['event_kind'] in {'RESERVE','SETTLE'}}
                    expected_global+=sum(v['charged_ms'] if v['settled'] else v['reserved_ms'] for v in final_rows.values() if v.get('pilot_id')==b['pilot_id'])
                    expected_train={a:sum(v['charged_ms'] if v['settled'] else v['reserved_ms'] for v in final_rows.values() if v.get('pilot_id')==b['pilot_id'] and v['arm_id']==a and v['pilot_phase']=='train') for a in {v['arm_id'] for v in final_rows.values() if v.get('pilot_id')==b['pilot_id']}}
                    _require(p['global_ms']==expected_global and p['training_ms']==expected_train and p['billing_boundary']=='SOL-015-v1')
                    _require(b['sample_role']=='t-star-after-report' and digest(b['limits_snapshot'])==digest(_limits(list(seen.values()),st['open'],x,arms=st['arms'])))
                    st['terminal']=True
                st['receipts'].append(event)
            else: _require(False,'unsupported grid event')
        seen[event['event_id']]=event
    return {k:v for k,v in states.items() if v['open']['payload']['pilot_id']==pilot_id}

def _limits(events,opened,x,exclude=None,arms=None):
    _integer(x); ob=opened['payload']['billing']; rows={}; historical={}; owner=0; stages={}
    prefix=ob['retained_prefix']; reached=False
    for e in events:
        p=e['payload']; k=e['event_kind']
        if k in {'RESERVE','SETTLE'}: rows[p['reservation_id']]=p
        if not reached:
            if k in {'PILOT_OWNER_COST','PILOT_OWNER_CHECKPOINT','PILOT_OWNER_TERMINAL'} and p['pilot_id']==ob['pilot_id']: owner+=p['elapsed_ms']
            if k in {'RESERVE','SETTLE'}: historical[p['reservation_id']]=p
        if _ref(e)==prefix: reached=True
        if k=='WALKING_STAGE_OPEN' and p['billing']['invocation_id']==ob['invocation_id']: stages[p['reservation_id']]=p['billing']
    _require(reached)
    base=(owner+sum(r['charged_ms'] if r['settled'] else r['reserved_ms'] for r in historical.values() if r.get('pilot_id')==ob['pilot_id']))*U
    train={}; arm={}; unknown=False; active=0
    for rid,r in rows.items():
        if arms is not None and r['arm_id'] not in arms: continue
        a=r['arm_id']; arm.setdefault(a,0)
        if r.get('pilot_id')==ob['pilot_id']: train.setdefault(a,0)
        if not r['settled']:
            if rid!=exclude: unknown=True
            if rid in stages and rid==exclude:
                active=x-stages[rid]['s_ns']; _integer(active)
            continue
        if type(r.get('monotonic_elapsed_ms')) is not int or r['monotonic_elapsed_ms']<0 or r.get('charged_ms')!=r['monotonic_elapsed_ms']: unknown=True; continue
        b=r.get('billing'); amount=b['duration_ns'] or 0 if b else r['charged_ms']*U
        if b and b.get('interval_kind')=='unknown': unknown=True; continue
        _integer(amount); arm[a]+=amount
        if r.get('pilot_id')==ob['pilot_id'] and r.get('pilot_phase')=='train': train[a]=train.get(a,0)+amount
    return {'as_of':_ref(events[-1]),'retained_prefix':prefix,'global_upper_ns':_integer(base+x),
        'training_upper_ns':{k:_integer(v) for k,v in train.items()},'arm_upper_ns':{k:_integer(v) for k,v in arm.items()},
        'unknown':unknown,'excluded_reservation_id':exclude,'active_stage_elapsed_ns':active,'sample_ns':x}


def _limit_usage_in_scope(ledger,pilot_id,*,sample_ns,grid,exclude_reservation_id=None):
    st=grid.state(ledger)
    return _limits(ledger.store._events(),st['open'],sample_ns,exclude_reservation_id,st['arms'])


def remaining_ns(ledger,pilot_id,arm_id,phase,*,grid,stage_cap=True):
    with ledger.store._read_transaction():
        grid.check(ledger,pilot_id); p=policy(ledger,pilot_id); u=usage(ledger,pilot_id)
        check_owner_admission(ledger.store,pilot_id)
        balance=ledger._balance(arm_id); x=grid.sample()
        limits=_limit_usage_in_scope(ledger,pilot_id,sample_ns=x,grid=grid)
        if limits['unknown'] or (ledger.store.path/'recovery-hold.json').exists():
            raise ResearchError('RECOVERY_REQUIRED','unknown/held grid usage')
        if balance['closed']: return 0
        st=grid.state(ledger)
        _require(all(k in st['settles'] for k in st['stages']),'active stage cannot reserve')
        values=[(p['global_ms']-u['global_ms'])*U,1800000000000-limits['global_upper_ns'],
            balance['remaining_ms']*U,86400000000000-limits['arm_upper_ns'].get(arm_id,0)]
        if stage_cap: values.append(120000000000)
        if phase=='train': values += [(p['training_ms']-u['training_ms'].get(arm_id,0))*U,120000000000-limits['training_upper_ns'].get(arm_id,0)]
        return max(0,min(values))


def _grid_stage_open(ledger,reservation,grid,deadline_ns,*,s_ns=None):
    with ledger.store._read_transaction():
        st=grid.state(ledger); rid=reservation['reservation_id']; s=grid.sample() if s_ns is None else _integer(s_ns)
        lim=_limit_usage_in_scope(ledger,reservation['pilot_id'],sample_ns=s,grid=grid,exclude_reservation_id=rid)
        if lim['unknown']: raise ResearchError('RECOVERY_REQUIRED','unknown grid limits')
        a=reservation['arm_id']; windows=[reservation['reserved_ms']*U,120000000000,
            86400000000000-lim['arm_upper_ns'].get(a,0),1800000000000-lim['global_upper_ns']]
        if reservation['pilot_phase']=='train': windows.append(120000000000-lim['training_upper_ns'].get(a,0))
        end=min(deadline_ns-grid.origin_ns,s+min(windows))
        if end<=s: raise ResearchError('BUDGET_EXHAUSTED','stage window exhausted at S')
        previous=list(st['settles'].values())
        b={**grid.common(ledger),**{k:reservation[k] for k in _IDENT},'reservation_id':rid,'s_ns':s,'q_s_ms':_q(s),
            'allocation_ms':reservation['reserved_ms'],'deadline_ns':end,'reservation_ref':_ref(st['res'][rid]),
            'previous_stage_ref':_ref(previous[-1]) if previous else None}
        event=ledger.store._append('WALKING_STAGE_OPEN',{'reservation_id':rid,'billing':b},'walking-stage-open-'+rid)
        return event


def _grid_go_in_scope(ledger,reservation,grid,stage_open,deadline_ns):
    grid.check(ledger,reservation['pilot_id']); st=grid.state(ledger); rid=reservation['reservation_id']
    _require(rid in st['stages'] and rid not in st['settles'] and st['stages'][rid]==stage_open)
    policy(ledger,reservation['pilot_id']); balance=ledger._balance(reservation['arm_id'])
    if balance['closed']: raise ResearchError('BUDGET_EXHAUSTED','arm closed before GO')
    x=grid.sample(); sb=stage_open['payload']['billing']; elapsed=x-sb['s_ns']
    lim=_limit_usage_in_scope(ledger,reservation['pilot_id'],sample_ns=x,grid=grid,exclude_reservation_id=rid)
    if lim['unknown']: raise ResearchError('RECOVERY_REQUIRED','unknown before GO')
    a=reservation['arm_id']; u=_usage_in_scope(ledger,reservation['pilot_id'])
    values=[deadline_ns-grid.origin_ns-x,sb['deadline_ns']-x,120000000000-elapsed,
        86400000000000-lim['arm_upper_ns'].get(a,0)-elapsed,1800000000000-lim['global_upper_ns'],
        (1800000-u['global_ms']+reservation['reserved_ms'])*U-elapsed,
        (balance['remaining_ms']+reservation['reserved_ms'])*U-elapsed]
    if reservation['pilot_phase']=='train': values += [120000000000-lim['training_upper_ns'].get(a,0)-elapsed,
        (120000-u['training_ms'].get(a,0)+reservation['reserved_ms'])*U-elapsed]
    if min(values)<=0: raise ResearchError('BUDGET_EXHAUSTED','grid GO window exhausted')
    # Any newly earlier cap requires new fuses; this minimal adapter rejects GO.
    if x+min(values)<deadline_ns-grid.origin_ns:
        raise ResearchError('BUDGET_EXHAUSTED','current cap precedes installed fuse')


class GridOwnerMeter(OwnerMeter):
    def __init__(self,ledger,config,start,clock,invocation_id,*,grid):
        super().__init__(ledger,config,start,clock,invocation_id); self.grid=grid; self.terminal_pending=None

    def sample(self):
        with self.ledger.store._read_transaction():
            st=self.grid.state(self.ledger)
            _require(set(st['res'])==set(st['settles']),'unresolved stage at owner sample')
            paid=list(st['settles'].values())
            if any(v['payload']['monotonic_elapsed_ms'] is None for v in paid):
                raise ResearchError('RECOVERY_REQUIRED','unknown grid stage')
            self.stages={k:v['payload']['charged_ms'] for k,v in st['settles'].items()}
            common={k:v for k,v in st['open']['payload']['billing'].items() if k in _COMMON}
            common['owner_open_ref']=_ref(st['open'])
            self.sample_context=(common,_ref(st['receipts'][-1]) if st['receipts'] else None,
                self.ledger.store._events(),st['open'],st['arms'])
            # No authority/file I/O follows this sample during payload construction.
            x=self.grid.sample(); total=sum(self.stages.values()); exact=x-sum(v['payload']['billing']['duration_ns'] or 0 for v in paid)
            _require(exact>=0 and all(x>=(v['payload']['billing']['r_ns'] or 0) for v in paid))
            return (x,_q(x)-total,tuple(_ref(v) for v in paid),exact)

    def payload(self,sample):
        x,measured,refs,exact=sample
        p=super().payload(measured); common,previous,events,opened,arms=self.sample_context
        p['billing']={**common,'sequence':self.sequence,'sample_ns':x,'q_sample_ms':_q(x),
            'stage_refs':list(refs),'stage_sum_ms':sum(self.stages.values()),'owner_exact_ns':exact,
            'previous_receipt_ref':previous,'checkpoint_kind':'checkpoint'}
        return p

    def checkpoint(self):
        with self.ledger.store._read_transaction():
            if self.pending is None:
                sample=self.sample(); self.pending=(sample,self.payload(sample))
            sample,payload=self.pending
            event_id='owner-checkpoint-'+digest([self.config['pilot_id'],self.config.get('execution_id'),self.invocation_id,self.sequence])
            event=self.ledger.store._append('PILOT_OWNER_CHECKPOINT',payload,event_id)
            _grid_state_in_scope(self.ledger,self.config['pilot_id'])
        self.billed=sample[1]; self.previous=event['event_id']; self.sequence+=1; self.pending=None

    def terminal(self,report,save):
        store=self.ledger.store; terminal_id='owner-terminal-'+self.invocation_id
        with store._read_transaction():
            created=self.terminal_pending is None
            if created:
                if self.pending is not None: raise ResearchError('RECOVERY_REQUIRED','checkpoint uncertain')
                u=_usage_in_scope(self.ledger,self.config['pilot_id']); st=self.grid.state(self.ledger)
                report.update(settlement_pending=True,terminal_receipt_id=terminal_id,billing_boundary='SOL-015-v1',budget=u,
                    billing_basis=GRID_BASIS,billing_schema_version=1,owner_open_ref=_ref(st['open']),
                    budget_as_of={'sequence':len(store._events()),'hash':store._events()[-1]['hash']},
                    settlement_tail={'pilot_charged':False,'included_in_qualification_wall':True,
                    'operations':['terminal append','exit full-chain verification/unlock','return/preconstructed CLI output']})
                report['cli_output']=__import__('json').dumps({'status':report['status'],'report':report['report_path'],'terminal_receipt_id':terminal_id,'arms':len(report['rows'])})
                raw=save(); report_hash=__import__('hashlib').sha256(raw).hexdigest()
                sample=self.sample(); p=self.payload(sample); p['billing'].update(checkpoint_kind='terminal',sample_role='t-star-after-report',
                    limits_snapshot=_limits(self.sample_context[2],self.sample_context[3],sample[0],arms=self.sample_context[4]))
                p.update(report_sha256=report_hash,billing_boundary='SOL-015-v1',global_ms=u['global_ms']+p['elapsed_ms'],training_ms=u['training_ms'])
                self.terminal_pending=(raw,p)
            raw,p=self.terminal_pending
            if not created and __import__('pathlib').Path(report['report_path']).read_bytes()!=raw:
                raise ResearchError('IDENTITY_CONFLICT','pending terminal report changed')
            event=store._append('PILOT_OWNER_TERMINAL',p,terminal_id)
            _grid_state_in_scope(self.ledger,self.config['pilot_id'])
        return event

@contextmanager
def owner_session(ledger,pilot_id,execution_id,invocation_id,*,config=None,grid=None):
    # Own the actual OS lock throughout the capability, including terminal exit.
    from contextlib import nullcontext
    from infrastructure.research_store import file_lock
    lock=file_lock(ledger.store.path/(".walking-pilot-"+identifier(pilot_id)+".lock"),timeout=0) if grid else nullcontext()
    with lock:
        with _owner_session_unlocked(ledger,pilot_id,execution_id,invocation_id,config=config,grid=grid):
            yield
