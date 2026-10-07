"""Fabricated metadata controls, NOT native execution or scientific evidence.

Full registered geometry exercises derivation; owner publication still requires
authorized real canonical artifacts/native costs, covered by managed controls.
No production store, grant, experiment, fit or sampler is created here.
"""

import copy

import pytest

from experiments.pirc25.affine import code_hash
from experiments.pirc28 import scientific_release as release
from experiments.pirc28.failure_disclosure import disclose_failure
from experiments.pirc28.plan import REFERENCE_PLAN, load_plan, describe_plan
from experiments.pirc28.qualification import IdentifiabilitySpec, METRICS, qualification_disposition
from infrastructure.research_store import ResearchError, digest, encode


class MetadataStore:
    def __init__(self, master):
        self.master = master

    def manifest(self, name):
        assert name == 'pirc28-study-definition-'+digest(self.master)
        return copy.deepcopy(self.master)


@pytest.fixture(scope='module')
def full_metadata():
    plan = load_plan(REFERENCE_PLAN)
    frozen = release._registered_definition(encode(plan), code_hash())
    master = frozen.to_dict()
    studies = frozen.proposals({'root': 'C:/engineering-metadata-only', 'store_id': 'not-registered'})
    rows, lineage, proofs = [], [], []
    for study in studies:
        study['admission'] = {'mode': 'pilot'}
        scenario = study['switching_recipe']['scenario']['scenario_id']
        negative = master['identifiability_spec']['scenario_roles'][scenario] == 'required-unidentifiable'
        for cell in study['cells']:
            identity = [study['study_id'], cell['replicate_id'], cell['fit_K']]
            attempt, artifact = digest(identity)[:32], digest(['FAKE-artifact', identity])
            k = cell['fit_K']
            rows.append({'scenario_id': scenario, 'replicate_id': cell['replicate_id'], 'K': k,
                'state': 'SUCCEEDED', 'artifact_hash': artifact,
                'input_hash': digest([scenario, cell['replicate_id']]),
                'metrics': {**({name: 0.001 for name in METRICS} if k == 2 else {}),
                            'position_energy_score_d2': 1/k},
                'posterior_occupancy': [1.0] if k == 1 else [0.5, 0.5], 'collapsed': negative and k == 2})
            lineage.append({'scenario_id': scenario, 'replicate_id': cell['replicate_id'], 'K': k,
                'study_id': study['study_id'], 'cell_hash': digest(cell), 'selected_attempt_id': attempt,
                'attempts': [{'attempt_id': attempt, 'artifact_id': artifact, 'state': 'SUCCEEDED',
                              'cost': {'charged_ms': 1, 'fixture': 'FABRICATED-METADATA-NOT-NATIVE-PROOF'}}]})
            proof = {'study_id': study['study_id'], 'cell_hash': digest(cell), 'attempt_id': attempt,
                'artifact_id': artifact, 'state': 'NUMERICAL_CHECKS_PASSED',
                'authorized_native_source_and_cost_bound': True, 'selection_scores_recomputed': True,
                'full_generated_input_hashes_verified': True, 'prefix_inputs_replayed': True,
                'prefix_diagnostics_consistent': True, 'reference_report_hash': digest(['FAKE-report', identity]),
                'real_data_authorized': False, 'reference_sampling_verified': k == 2,
                'reference_seed': 42 if k == 2 else None,
                'reference_execution_hash': digest('FAKE-reference') if k == 2 else None}
            if k == 1:
                proof.update(k1_boundary_verified=True, k1_boundary_execution_hash=digest('FAKE-boundary'))
            proofs.append(proof)
    binding = release.approved_master_binding(MetadataStore(master), studies, digest(master))
    return master, studies, rows, lineage, proofs, binding


def derive(data, fixture=False):
    _, studies, rows, lineage, proofs, binding = data
    return release.aggregate_approximation(studies, rows, lineage, proofs, binding, fixture=fixture)


def test_exact_registered_master_and_full_metadata_derivation_without_numerics(full_metadata, monkeypatch):
    from experiments.pirc28 import approximation_checks, pipeline
    monkeypatch.setattr(approximation_checks, 'check_prefix_approximation', lambda *a, **k: pytest.fail('sampling'))
    monkeypatch.setattr(pipeline, 'execute_recipe', lambda *a, **k: pytest.fail('fit'))
    result = derive(full_metadata)
    assert result['registered_cells'] == 2048 and result['selected_check_counts'] == {'NUMERICAL_CHECKS_PASSED': 2048}
    assert result['qualified'] is True  # Derivative only, NOT publication authority.
    assert result['real_data_authorized'] is result['behavioral_interpretation_authorized'] is False
    definition = IdentifiabilitySpec.from_dict(full_metadata[0]['identifiability_spec'])
    gate = qualification_disposition(definition, full_metadata[2], digest(full_metadata[2]))
    assert gate['state'] == 'PASSED' and len(gate['applicable_scenarios']) == 2
    assert release.apply_disposition(gate, result, fixture=False) is True


@pytest.mark.parametrize('damage', ['prior', 'seed', 'threshold', 'role', 'initialization', 'sampling', 'source'])
def test_same_sized_rehashed_engineering_master_is_not_approved(full_metadata, damage):
    master = copy.deepcopy(full_metadata[0])
    name = next(iter(master['recipes']))
    if damage == 'prior': master['recipes'][name]['prior_mean'][0] += 0.1
    elif damage == 'seed': master['replicate_seeds'][name][next(iter(master['replicate_seeds'][name]))] += 1
    elif damage == 'threshold': master['identifiability_spec']['thresholds']['mode_ece'] += 0.01
    elif damage == 'role': master['identifiability_spec']['scenario_roles'][name] = (
        'required-qualified' if master['identifiability_spec']['scenario_roles'][name] == 'phase-diagram-only'
        else 'phase-diagram-only')
    elif damage == 'initialization': master['recipes'][name]['initializations']['1'][0]['a'][0][0] += 0.1
    elif damage == 'sampling': master['recipes'][name]['test_paths'] += 1
    else: master['runtime_code_hash'] = digest('different-code')
    assert digest(master) != digest(full_metadata[0])  # Never test a no-op forgery.
    result = release.approved_master_binding(MetadataStore(master), full_metadata[1], digest(master))
    assert result['state'] == 'NEEDS_APPROVED_SCIENTIFIC_DEFINITION'


def test_smaller_matrix_cannot_borrow_full_master_binding(full_metadata):
    data = copy.deepcopy(full_metadata)
    data[1][:] = data[1][:1]
    data[2][:] = data[2][:32]
    data[3][:] = data[3][:32]
    data[4][:] = data[4][:32]
    with pytest.raises(ResearchError, match='smaller matrix'):
        derive(data)


@pytest.mark.parametrize('state', ['NOT_RUN', 'TIMEOUT', 'BUDGET_EXHAUSTED', 'FAILED', 'INFERENCE_INAPPLICABLE'])
def test_missing_or_failed_check_never_counts_as_pass(full_metadata, state):
    data = copy.deepcopy(full_metadata)
    data[2][0].update(state=state, metrics=None, posterior_occupancy=None, collapsed=None)
    data[4].pop(0)
    result = derive(data)
    assert result['qualified'] is False and len(result['cell_dispositions']) == 2048
    assert result['cell_dispositions'][0]['outcome_state'] == state
    assert result['selected_check_counts']['NUMERICAL_CHECKS_PASSED'] == 2047


def test_later_failure_keeps_passed_check_and_failed_statistical_denominator(full_metadata):
    data = copy.deepcopy(full_metadata)
    data[2][0].update(state='FAILED', metrics=None, posterior_occupancy=None, collapsed=None)
    data[3][0]['attempts'][0]['state'] = 'FAILED'
    result = derive(data)
    assert result['cell_dispositions'][0]['outcome_state'] == 'FAILED'
    assert result['cell_dispositions'][0]['check_state'] == 'NUMERICAL_CHECKS_PASSED'
    assert result['cell_dispositions'][0]['failed_attempt_ids']
    assert result['complete_lineage_hash'] == digest(data[3])


@pytest.mark.parametrize('state', ['REGISTERED', 'RUNNING'])
def test_unfinished_original_attempt_withholds_release_and_is_not_a_failed_outcome(full_metadata, state):
    data = copy.deepcopy(full_metadata)
    data[3][0]['attempts'].append({'attempt_id': digest('pending')[:32], 'state': state,
                                 'artifact_id': None, 'cost': None})
    result = derive(data)
    assert result['qualified'] is False and result['state'] == 'SOURCE_ATTEMPTS_NOT_TERMINAL'
    assert result['unresolved_original_attempts'] == 1
    assert not result['cell_dispositions'][0]['failed_attempt_ids']
    assert result['cell_dispositions'][0]['unresolved_attempt_ids']


def test_prepared_plan_inventory_is_not_admission_or_scientific_authority():
    summary = describe_plan(load_plan(REFERENCE_PLAN))
    assert summary['approximation_executor_implemented'] is summary['aggregate_release_implemented'] is True
    assert summary['approximation_resource_admitted'] is summary['native_release_evidence_bound'] is False
    assert summary['scientific_groups_executed'] == 0
    assert summary['qualified_identifiability'] is summary['real_data_authorized'] is False


@pytest.mark.parametrize('damage', ['missing-k1', 'missing-k2', 'duplicate', 'foreign', 'artifact',
                                  'native', 'input', 'boundary', 'k1-sampling', 'k2-sampling'])
def test_aggregate_refuses_forged_or_missing_native_check_metadata(full_metadata, damage):
    data = copy.deepcopy(full_metadata)
    proofs = data[4]
    if damage == 'missing-k1': proofs[0]['k1_boundary_verified'] = False
    elif damage == 'missing-k2': proofs[1]['reference_execution_hash'] = None
    elif damage == 'duplicate': proofs.append(copy.deepcopy(proofs[0]))
    elif damage == 'foreign': proofs[0]['attempt_id'] = digest('foreign')[:32]
    elif damage == 'artifact': proofs[0]['artifact_id'] = digest('other')
    elif damage == 'native': proofs[0]['authorized_native_source_and_cost_bound'] = 1
    elif damage == 'input': proofs[0]['prefix_inputs_replayed'] = 0
    elif damage == 'boundary': proofs[0]['k1_boundary_execution_hash'] = None
    elif damage == 'k1-sampling': proofs[0]['reference_sampling_verified'] = True
    else: proofs[1]['reference_sampling_verified'] = False
    with pytest.raises(ResearchError): derive(data)


def test_fixture_and_formal_sources_cannot_borrow_pilot_release(full_metadata):
    assert derive(full_metadata, fixture=True)['qualified'] is False
    data = copy.deepcopy(full_metadata)
    for study in data[1]: study['admission']['mode'] = 'formal'
    assert derive(data)['state'] == 'UNSUPPORTED_SOURCE_STAGE'


def test_registered_plan_edit_cannot_silently_reapprove(full_metadata, monkeypatch):
    changed = load_plan(REFERENCE_PLAN)
    changed['sampling']['test_paths'] += 1
    monkeypatch.setattr(release, 'load_plan', lambda *a: changed)
    with pytest.raises(ResearchError, match='reapproval'):
        release.approved_master_binding(MetadataStore(full_metadata[0]), full_metadata[1], digest(full_metadata[0]))


def test_v2_failure_projection_preserves_source_and_diagnostics_not_large_arrays():
    report = {'state': 'INFERENCE_INAPPLICABLE', 'paths': [{'path_index': i,
        'state': 'INFERENCE_INAPPLICABLE', 'reasons': ['REFERENCE_MC_PRECISION_FAILED'],
        'reference_half_widths': {'mode_total_variation': None},
        'coarse_summary': {'means': [[0.0]*4]*1000},
        'reference_batch_summaries': [{'means': [[0.0]*4]*1000}]*8} for i in range(8)]}
    body = {'stage': 'approximation', 'error_code': 'INFERENCE_INAPPLICABLE',
        'recipe_hash': digest('recipe'), 'generated_inputs': [], 'initializations': [],
        'approximation_checks': report, 'cause': None}
    result = {'failure': body, 'source_schema': 'pirc28-synthetic-failure-v3'}
    before = copy.deepcopy(result)
    summary = disclose_failure(result, {'switching_recipe': {'schema_version': 'pirc28-synthetic-recipe-v2'}}, digest(result))
    assert len(encode(summary)) < 8192 and len(encode(result)) > 100000
    assert result == before and summary['complete_failure_hash'] == digest(body)
    assert summary['source_artifact_id'] == digest(result)
    assert summary['reports']['approximation_checks']['paths'][0]['reference_half_widths']['mode_total_variation'] is None
    assert 'reference_batch_summaries' not in summary['reports']['approximation_checks']['paths'][0]
    assert summary['qualified_identifiability'] is summary['real_data_authorized'] is False
    with pytest.raises(ResearchError, match='original artifact'):
        disclose_failure(result, {'switching_recipe': {'schema_version': 'pirc28-synthetic-recipe-v2'}}, digest('borrowed'))
