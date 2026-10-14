"""Explicitly unverified software metadata; cannot pass production inspection.

Source pins/numbers alone are NOT eligibility or human permission. The absent
directory/fixture-only bytes cannot equal the actual linked historical files.
Positive wiring tests must explicitly replace the live eligibility consumer.
"""
from pathlib import Path
import pytest

from experiments.pirc17 import formal_budget as budget, formal_carryover as costs, formal_predecessor as pred
from experiments.pirc17.protocol_core import digest,envelope,unpack


def software_predecessor(protocol,matrix,*,directory,phase_caps_ns=None,workloads=None):
    m = unpack(matrix)
    caps = phase_caps_ns or {k:v*budget.NANOSECONDS for k,v in m['phase_caps_seconds'].items()}
    works = workloads or m['workloads']
    phase = 'input_qualification_and_binding'
    total = pred.REGISTERED_CHARGED_NS
    measured = 24_344_000_000 if total == 40_423_000_000 else total*3//4
    control = total-measured
    def columns(value): return {k:value if k == phase else 0 for k in caps}
    directory = str(Path(directory).resolve())
    c = dict(schema_version=costs.VERSION,ledger_directory=directory,
        ledger_root_sha256=pred.REGISTERED['expected_root_sha256'],head_sha256=pred.REGISTERED['expected_head_sha256'],
        terminal_proof_sha256=pred.REGISTERED['expected_terminal_proof_sha256'],
        protocol_sha256=protocol['sha256'],matrix_sha256=matrix['sha256'],
        execution_sha256=digest('NOT HISTORICAL EXECUTION'),approval_sha256=digest('NOT HUMAN PREDECESSOR APPROVAL'),
        runtime_manifest_sha256=digest('NOT HISTORICAL RUNTIME'),phase_caps_ns=caps,total_cap_ns=sum(caps.values()),
        charged_ns_by_phase=columns(total),measured_ns_by_phase=columns(measured),
        conservatively_charged_ns_by_phase=columns(control),control_charged_ns_by_phase=columns(control),
        control_observed_ns_by_phase=columns(11_156_000_000 if total == 40_423_000_000 else control*9//10),
        charged_total_ns=total,generated_forecasts_reserved=0,work_inventory_count=len(works),
        work_dispositions={works[0]['work_id']:'failure'},halted_reason='supervision_error',
        final_eval_reads=0,read_only=True,authorizes_execution=False,software_fixture_only=True)
    return envelope(dict(schema_version=pred.VERSION,ledger_directory=directory,cost_snapshot=envelope(c),
        source_file_sha256={str(Path(directory)/'ABSENT-NOT-HISTORICAL.json'):digest('SYNTHETIC BYTES')},
        eligible_closed_input_startup=True,process_tree_closed=True,read_only=True,generated_forecasts=0,
        final_eval_reads=0,authorizes_execution=False,software_fixture_only=True))


def permit_software_eligibility(monkeypatch):
    def explicit_fixture(binding):
        value = unpack(binding)
        assert value.get('software_fixture_only') is True
        assert value['authorizes_execution'] is False
        return value
    monkeypatch.setattr(pred,'verify_startup_predecessor',explicit_fixture)
    from experiments.pirc17 import formal_input_interruption as interruption
    monkeypatch.setattr(interruption,'verify_input_interruption',explicit_fixture)


_SOFTWARE_INPUT_COST_PINS = set()


@pytest.fixture(scope='module',autouse=True)
def software_frozen_metadata():
    """Reuse full immutable definitions in SOFTWARE tests, not production.

    The real validators still compare every supplied value against the complete
    11659-item definition. Only rebuilding unchanged metadata is memoized.
    Source bytes and scientific specifications are independently checked before
    and after this module; native subprocesses do not inherit these test seams.
    This is not empirical or full runtime qualification evidence.
    """
    from copy import deepcopy
    from experiments.pirc17 import protocol,formal_matrix
    from experiments.pirc17.protocol_core import canonical
    catalog,semantics,definition=protocol.source_catalog,protocol.protocol_semantics,formal_matrix.definition
    sources,spec=catalog(),semantics()
    definitions={}
    def full_definition(record):
        key=canonical(record)
        if key not in definitions:definitions[key]=definition(record)
        return deepcopy(definitions[key])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(protocol,'source_catalog',lambda:dict(sources))
        patch.setattr(protocol,'protocol_semantics',lambda:deepcopy(spec))
        patch.setattr(formal_matrix,'definition',full_definition)
        yield
    assert catalog()==sources and canonical(semantics())==canonical(spec), 'metadata/source changed during software scope'


@pytest.fixture(autouse=True)
def software_input_pins(monkeypatch):
    """Explicit test-only cost content pin replacement, NEVER live eligibility.

    Register fixture costs at construction, NOT at validation: a rehashed
    changed cost cannot become accepted merely by calling this software seam.
    """
    from experiments.pirc17 import formal_input_interruption as interruption
    monkeypatch.setattr(__import__(__name__,fromlist=['_SOFTWARE_INPUT_COST_PINS']),
                        '_SOFTWARE_INPUT_COST_PINS',set())
    actual = interruption.validate_binding_scope
    def synthetic_scope(value):
        if value.get('software_fixture_only') is not True:
            return actual(value)
        if value['cost_snapshot']['sha256'] not in _SOFTWARE_INPUT_COST_PINS:
            raise ValueError('unregistered software cost content;not actual history')
        with monkeypatch.context() as patch:
            patch.setattr(interruption,'REGISTERED_COST_SNAPSHOT',value['cost_snapshot']['sha256'])
            return actual(value)
    monkeypatch.setattr(interruption,'validate_binding_scope',synthetic_scope)


def software_input_interruption(protocol,matrix,*,directory,phase_caps_ns=None,workloads=None):
    from copy import deepcopy
    from experiments.pirc17 import formal_input_interruption as interruption
    old=unpack(software_predecessor(protocol,matrix,directory=directory,phase_caps_ns=phase_caps_ns,workloads=workloads))
    c=unpack(old['cost_snapshot']);phase='input_qualification_and_binding'
    def columns(amount):return {p:amount if p==phase else 0 for p in c['phase_caps_ns']}
    c.update(ledger_root_sha256=interruption.REGISTERED['expected_root_sha256'],
        head_sha256=interruption.REGISTERED['expected_head_sha256'],
        terminal_proof_sha256=interruption.REGISTERED['expected_terminal_proof_sha256'],
        charged_ns_by_phase=columns(570_736_000_000),measured_ns_by_phase=columns(536_204_000_000),
        conservatively_charged_ns_by_phase=columns(34_532_000_000),control_charged_ns_by_phase=columns(34_532_000_000),
        control_observed_ns_by_phase=columns(24_671_000_000),charged_total_ns=570_736_000_000)
    cost=envelope(c);_SOFTWARE_INPUT_COST_PINS.add(cost['sha256'])
    return envelope(dict(schema_version=interruption.VERSION,ledger_directory=old['ledger_directory'],cost_snapshot=cost,
        eligible_closed_input_interruption=True,process_tree_closed=True,read_only=True,authorizes_execution=False,
        generated_forecasts=0,predictive_model_fits=0,raw_sources_independently_reloaded=False,worker_training_state_retained=False,
        inventory_sha256=interruption.REGISTERED_INVENTORY,claim_inventory_sha256=interruption.REGISTERED_CLAIM_INVENTORY,
        bundle_sha256=interruption.REGISTERED_BUNDLE,original_predecessor_sha256=interruption.REGISTERED_PREDECESSOR,
        input_context_sha256=interruption.REGISTERED_CONTEXT,input_qualification_sha256=interruption.REGISTERED_QUALIFICATION,
        selected_samples=46,historical_access=deepcopy(interruption.REGISTERED_ACCESS),software_fixture_only=True))
