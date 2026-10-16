"""Bounded aggregate dispositions, NOT standalone source or data authority.

Call only after canonical authorized reads, native source/cost verification and
funded numeric checks (worker), or their metadata-only owner binding. A caller's
invented rows/receipts/binding cannot authorize publication through this module.
"""

from collections import Counter
from functools import lru_cache
import json

from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, ResearchStore, digest, encode
from .approximation import CHECKS, pending_qualification
from .plan import REFERENCE_PLAN, load_plan, prepare_definition

REGISTERED_PLAN_HASH = 'cfd7bcfd702b9c42f10f0414dfd3783c0738b9caa06729dc1e19af508699ebaf'
SCHEMA = 'pirc28-aggregate-approximation-qualification-v2'


def _require(condition, description):
    if not condition:
        raise ResearchError('CONTRACT_MISMATCH', description)


@lru_cache(maxsize=1)
def _registered_definition(content, source_hash):
    # Immutable metadata only; no generation, fitting, filtering or bootstrap.
    result = prepare_definition(json.loads(content))
    _require(result.to_dict()['runtime_code_hash'] == source_hash,
             'registered master preparation source changed')
    return result


def approved_master_binding(store, studies, reference):
    """A supplied complete master is not necessarily the approved full master.

    Caller first checks study_definition_binding, including every pre-fit source.
    Comparing complete canonical definitions binds all recipes, seeds, priors,
    initializers, thresholds, roles and resources, NOT only matrix dimensions.
    """
    plan = load_plan(REFERENCE_PLAN)
    _require(digest(plan) == REGISTERED_PLAN_HASH, 'registered scientific plan changed without reapproval')
    expected, matched = None, False
    if reference is not None:
        master = store.manifest('pirc28-study-definition-'+reference)
        _require(digest(master) == reference, 'supplied master canonical hash differs')
        # Cheap refusal for engineering masters. Counts never establish MATCH.
        if (len(master['recipes']) == 64
                and len(master['identifiability_spec']['replicate_ids']) == 16):
            expected = _registered_definition(encode(plan), code_hash()).definition_hash
            matched = reference == expected
    return {'schema_version': 'pirc28-approved-scientific-master-binding-v1',
        'registered_plan_hash': REGISTERED_PLAN_HASH, 'supplied_definition_hash': reference,
        'expected_definition_hash': expected,
        'state': 'REGISTERED_MASTER_MATCHED' if matched else 'NEEDS_APPROVED_SCIENTIFIC_DEFINITION',
        'registered_scenarios': 64, 'registered_replicates_per_scenario': 16,
        'registered_paired_cells': 2048, 'real_data_authorized': False}


def aggregate_approximation(studies, rows, lineage, proofs, binding, *, fixture):
    """Derive EVERY selected cell/check disposition; never trust a PASS flag.

    All earlier failed/retried attempts remain bound in lineage and roster hashes.
    A later forecast/evaluation failure may retain PASSED prefix checks, but its
    outcome/absent metrics remain failed in the original statistical denominator.
    """
    _require(type(fixture) is bool and type(rows) is list and 0 < len(rows) <= 10000
             and type(lineage) is list and len(rows) == len(lineage)
             and type(proofs) is list, 'bounded complete aggregate metadata required')
    pending = pending_qualification(studies)
    registered = {(study['switching_recipe']['scenario']['scenario_id'], cell['replicate_id'], cell['fit_K']):
                  (study, cell) for study in studies for cell in study['cells']}
    _require(len(registered) == len(rows), 'aggregate registered cell roster differs')
    if binding['state'] == 'REGISTERED_MASTER_MATCHED':
        _require(len(rows) == 2048 and len(studies) == 64
            and binding.get('registered_plan_hash') == REGISTERED_PLAN_HASH
            and binding.get('expected_definition_hash') == binding.get('supplied_definition_hash')
            and all(study.get('pirc28_study_definition_hash') == binding['expected_definition_hash']
                    and len(study['cells']) == 32 for study in studies),
            'approved full scientific master cannot describe a smaller matrix')
    entries = {}
    for entry in lineage:
        key = (entry['scenario_id'], entry['replicate_id'], entry['K'])
        _require(key in registered and key not in entries, 'duplicate or foreign lineage cell')
        entries[key] = entry
    proof_index = {}
    for proof in proofs:
        identity = (proof['study_id'], proof['cell_hash'], proof['attempt_id'])
        _require(identity not in proof_index, 'duplicate approximation attempt proof')
        proof_index[identity] = proof
    dispositions, seen, used_proofs = [], set(), set()
    for row in rows:
        key = (row['scenario_id'], row['replicate_id'], row['K'])
        _require(key in registered and key not in seen, 'duplicate or foreign aggregate row')
        seen.add(key)
        study, cell = registered[key]
        entry = entries[key]
        _require(entry['study_id'] == study['study_id'] and entry['cell_hash'] == digest(cell),
                 'aggregate source cell lineage differs')
        attempts = {item['attempt_id']: item for item in entry['attempts']}
        _require(len(attempts) == len(entry['attempts']), 'duplicate original attempt')
        selected = entry['selected_attempt_id']
        _require(selected is None or selected in attempts, 'selected original attempt is missing')
        # Retain and bind every failed/partial check, not only the selected one.
        for attempt in attempts.values():
            identity = (study['study_id'], digest(cell), attempt['attempt_id'])
            if identity in proof_index:
                proof = proof_index[identity]
                _require(proof['artifact_id'] == attempt['artifact_id'], 'attempt approximation artifact differs')
                used_proofs.add(identity)
        proof = proof_index.get((study['study_id'], digest(cell), selected))
        check_state = 'NO_COMPLETE_CHECK'
        if proof is not None:
            _require(proof.get('authorized_native_source_and_cost_bound') is True
                and proof.get('selection_scores_recomputed') is True
                and proof.get('full_generated_input_hashes_verified') is True
                and proof.get('real_data_authorized') is False,
                'selected native approximation source is unbound')
            check_state = proof['state']
            if check_state == 'NUMERICAL_CHECKS_PASSED':
                _require(proof.get('prefix_inputs_replayed') is True
                    and proof.get('prefix_diagnostics_consistent') is True
                    and type(proof.get('reference_report_hash')) is str,
                    'selected complete approximation report is missing')
                if row['K'] == 1:
                    _require(proof.get('k1_boundary_verified') is True
                        and proof.get('reference_sampling_verified') is False
                        and proof.get('reference_seed') is None
                        and proof.get('reference_execution_hash') is None
                        and type(proof.get('k1_boundary_execution_hash')) is str,
                        'K1 analytic boundary cannot borrow K2 sampling authority')
                else:
                    _require(proof.get('reference_sampling_verified') is True
                        and type(proof.get('reference_execution_hash')) is str
                        and type(proof.get('reference_seed')) is int,
                        'K2 actual fixed reference completion is missing')
        elif row['state'] in ('NOT_RUN', 'TIMEOUT', 'BUDGET_EXHAUSTED'):
            check_state = row['state']
        dispositions.append({'scenario_id': key[0], 'replicate_id': key[1], 'K': key[2],
            'study_id': study['study_id'], 'cell_hash': digest(cell), 'outcome_state': row['state'],
            'selected_attempt_id': selected, 'check_state': check_state,
            'selected_check_hash': None if proof is None else digest(proof),
            'all_original_attempts_hash': digest(entry['attempts']),
            'failed_attempt_ids': [item['attempt_id'] for item in entry['attempts']
                                   if item['state'] in ResearchStore.TERMINAL and item['state'] != 'SUCCEEDED'],
            'unresolved_attempt_ids': [item['attempt_id'] for item in entry['attempts']
                                       if item['state'] not in ResearchStore.TERMINAL]})
    _require(used_proofs == set(proof_index), 'foreign approximation attempt proof')
    complete = all(item['check_state'] == 'NUMERICAL_CHECKS_PASSED' for item in dispositions)
    approved = binding['state'] == 'REGISTERED_MASTER_MATCHED'
    pilot = all(study['admission']['mode'] == 'pilot' for study in studies)
    unresolved = sum(len(item['unresolved_attempt_ids']) for item in dispositions)
    qualified = not fixture and approved and pilot and complete and not unresolved and pending['spec_hash'] is not None
    states = {item['check_state'] for item in dispositions}
    state = ('FIXTURE_ONLY' if fixture else 'NEEDS_APPROVED_SCIENTIFIC_DEFINITION' if not approved
             else 'UNSUPPORTED_SOURCE_STAGE' if not pilot else 'SOURCE_ATTEMPTS_NOT_TERMINAL' if unresolved
             else 'PASSED' if qualified
             else 'BUDGET_EXHAUSTED' if states & {'TIMEOUT', 'BUDGET_EXHAUSTED'}
             else 'INCOMPLETE' if 'NOT_RUN' in states
             else 'INFERENCE_INAPPLICABLE' if 'INFERENCE_INAPPLICABLE' in states
             else 'NEEDS_APPROXIMATION_QUALIFICATION')
    return {'schema_version': SCHEMA, 'spec_hash': pending['spec_hash'], 'state': state,
        'required_checks': list(CHECKS), 'qualified': qualified,
        'managed_evidence_bound': qualified, 'real_data_authorized': False,
        'approved_master_binding': binding, 'aggregate_hash': digest(rows),
        'complete_lineage_hash': digest(lineage), 'all_attempt_proof_roster_hash': digest(proofs),
        'registered_cells': len(rows), 'unresolved_original_attempts': unresolved,
        'selected_check_counts': dict(sorted(Counter(
            item['check_state'] for item in dispositions).items())), 'cell_dispositions': dispositions,
        'sampling_unit': 'independent-synthetic-replicate',
        'failure_rule': 'retain-all-original-outcomes-and-costs; no-missing-check-as-pass',
        'accuracy': 'checked-approximation; finite-particle-reference; not-exact-switching-posterior',
        'behavioral_interpretation_authorized': False}


def apply_disposition(gate, approximation, *, fixture):
    """Keep statistical negative results; withhold only unsupported promotion."""
    if not fixture and gate['state'] == 'PASSED' and approximation['qualified'] is not True:
        gate['identifiability_only_state'] = 'PASSED'
        gate['state'] = 'NEEDS_APPROXIMATION_QUALIFICATION'
        if 'applicable_ranges' in gate:
            gate['applicable_scenarios'], gate['applicable_ranges'] = [], []
    return not fixture and gate['state'] == 'PASSED' and approximation['qualified'] is True
