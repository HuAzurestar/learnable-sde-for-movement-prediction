"""Pre-fit scientific source checks, not execution or qualification authority.

Only the registered exploratory synthetic pilot is supported here. Passing
does not enable a V2 executor, waive ordinary admission or lift scientific
withholding. All work is metadata-only within the original funded attempt.
"""

from application.research_admission import command_binding, data_binding, plugin_binding
from application.research_budget import BudgetLedger
from application.research_evidence import authorize_study, require_unexpired_disclosure
from application.research_execution import execution_plan
from application.research_preregistration import hash_reference
from experiments.pirc25.affine import code_hash
from infrastructure.research_store import ResearchError, digest, encode
from .approximation import RECIPE_SCHEMA
from .evidence import _preregistration, _run_id, study_definition_binding
from .qualification import IdentifiabilitySpec
from .runtime import command, resume_command, switching_plugin
from .source_metadata import SourceMetadata
from .study import MAXIMUM_DEFINITION_BYTES, SCHEMA_V2, SyntheticStudyDefinition


def _require(condition, code, message):
    if not condition:
        raise ResearchError(code, message)


def _current_authority(store, study, cell, attempt, admissions, reserve, prefix):
    """Fresh mutable authority only; no whole-matrix reconstruction here."""
    events = store.events()
    _require(len(events) >= len(prefix) and (not prefix
        or events[len(prefix)-1]['hash'] == prefix[-1]['hash']),
        'CORRUPT_ARTIFACT', 'pre-fit authoritative prefix changed between phases')
    _require(store.attempts().get(attempt['attempt_id']) == attempt,
        'IDENTITY_CONFLICT', 'pre-fit original attempt is no longer running')
    _require([event for event in events if event['event_kind'] == 'ADMISSION'
        and event['payload'].get('attempt_id') == attempt['attempt_id']] == admissions
        and [event for event in events if event['event_kind'] == 'RESERVE'
        and event['payload'].get('attempt_id') == attempt['attempt_id']] == [reserve]
        and not any(event['event_kind'] == 'SETTLE'
        and event['payload'].get('attempt_id') == attempt['attempt_id'] for event in events),
        'UNQUALIFIED', 'pre-fit original admission/funding changed between phases')
    _require(not (store.path/'recovery-hold.json').exists(),
        'RECOVERY_REQUIRED', 'authoritative recovery holds scientific work')
    payload = reserve['payload']
    resource = payload.get('budget_mode') == 'resource-guarded-no-time-cap'
    balance = BudgetLedger(store).balance(cell['arm_id'],
        execution_policy=payload.get('execution_policy') if resource else None,
        source_code_hash=payload.get('source_code_hash') if resource else None)
    if balance['closed'] and resource and payload.get('restart_adjudication'):
        from application.research_restart_adjudication import resolve_retry_adjudication
        adjudication = resolve_retry_adjudication(store, payload['restart_adjudication'],
            execution_policy=payload['execution_policy'], source_code_hash=payload['source_code_hash'])
        _require(adjudication['retry_attempt_id'] == attempt['attempt_id'],
            'IDENTITY_CONFLICT', 'restart adjudication belongs to another attempt')
    else:
        _require(not balance['closed'],
            'BUDGET_EXHAUSTED', 'original scientific arm is closed')


def validate_scientific_source(store, study, cell, attempt, *, check_budget):
    """Revalidate canonical original metadata before any generated/fitted data."""
    _require(callable(check_budget), 'CONTRACT_MISMATCH', 'explicit pre-fit deadline poll required')
    _require(store._read_snapshot() is None, 'CONTRACT_MISMATCH',
        'pre-fit validation cannot inherit an outer writer lock')
    check_budget()
    with SourceMetadata(store, check_budget, 4*MAXIMUM_DEFINITION_BYTES) as metadata:
        _require(study.get('admission', {}).get('mode') == 'pilot'
            and study.get('admission', {}).get('purpose') == 'fit'
            and study.get('switching_recipe', {}).get('schema_version') == RECIPE_SCHEMA,
            'CONTRACT_MISMATCH', 'registered V2 exploratory synthetic pilot required')
        _require(study.get('code_hash') == code_hash() and study.get('runtime_binding') ==
            {'root': str(store.path.parent), 'store_id': store.store_id},
            'IDENTITY_CONFLICT', 'pre-fit source code/original shared store differs')
        canonical = metadata.manifest('study-'+study['study_id'])
        _require(digest(canonical['spec']) == digest(study) and canonical['spec_hash'] == digest(study)
            and sum(digest(value) == digest(cell) for value in study['cells']) == 1,
            'IDENTITY_CONFLICT', 'pre-fit source/cell differs from canonical registration')
        _require(attempt.get('state') == 'RUNNING'
            and attempt.get('run_id') == _run_id(study, cell)
            and store.attempts().get(attempt.get('attempt_id')) == attempt,
            'IDENTITY_CONFLICT', 'pre-fit source has no original running attempt')
        events = store.events()
        admissions = [event for event in events if event['event_kind'] == 'ADMISSION'
            and event['payload'].get('attempt_id') == attempt['attempt_id']]
        _require(len(admissions) == 1 and admissions[0]['payload'].get('run_id') == attempt['run_id'],
            'UNQUALIFIED', 'one original admission required before scientific work')
        admission_hash = admissions[0]['payload'].get('admission_hash')
        _require(hash_reference(admission_hash), 'UNQUALIFIED', 'canonical admission identity missing')
        receipt = store.manifest('admission-'+admission_hash)
        _require(receipt.get('admission_hash') == admission_hash
            and digest({name: value for name, value in receipt.items() if name != 'admission_hash'}) == admission_hash
            and all(receipt.get(name) == value for name, value in {
                'attempt_id': attempt['attempt_id'], 'run_id': attempt['run_id'],
                'spec_hash': digest(study), 'cell_hash': digest(cell), 'mode': 'pilot'}.items())
            and digest(receipt.get('spec')) == digest(study) and digest(receipt.get('cell')) == digest(cell),
            'IDENTITY_CONFLICT', 'pre-fit admission source/cell/stage differs')
        plugin = switching_plugin()
        _require(receipt.get('schema_version') == 'pirc25-admission-v1'
            and receipt.get('execution_kind') in ('run', 'resume')
            and receipt.get('plugin_hash') == plugin_binding(plugin)
            and receipt.get('registry_entry') == plugin.registry_entry.manifest()
            and receipt.get('resource_plan') == execution_plan(study, cell, plugin)
            and receipt.get('command_hash') == command_binding(
                resume_command if receipt.get('execution_kind') == 'resume' else command),
            'IDENTITY_CONFLICT', 'pre-fit admitted command/registry/resources differ')
        reserves = [event for event in events if event['event_kind'] == 'RESERVE'
            and event['payload'].get('attempt_id') == attempt['attempt_id']]
        _require(len(reserves) == 1 and not any(event['event_kind'] == 'SETTLE'
            and event['payload'].get('attempt_id') == attempt['attempt_id'] for event in events),
            'UNQUALIFIED', 'pre-fit work needs its original outstanding reservation')
        reserve = reserves[0]
        # SharedSupervisor funds command preparation first; AdmissionGate then
        # publishes its receipt, before the actual worker is launched.
        funding = reserve['payload']
        finite_funding = (type(funding.get('reserved_ms')) is int
            and 0 < funding['reserved_ms'] <= 1_800_000)
        resource_funding = (funding.get('reserved_ms') is None
            and funding.get('budget_mode') == 'resource-guarded-no-time-cap'
            and funding.get('source_code_hash') == study['code_hash']
            and type(funding.get('execution_policy')) is dict)
        _require(reserve['sequence'] < admissions[0]['sequence']
            and (finite_funding or resource_funding)
            and all(reserve['payload'].get(name) == value for name, value in {
                'attempt_id': attempt['attempt_id'], 'run_id': attempt['run_id'],
                'study_id': study['study_id'], 'arm_id': cell['arm_id'],
                'reservation_id': digest([store.store_id, attempt['attempt_id']])}.items()),
            'UNQUALIFIED', 'pre-fit funding belongs to a different source/arm/stage')
        # Refuse a borrowed grant before pinning an otherwise self-consistent
        # receipt. Only immutable matrix metadata may use the detached view.
        settings = study['admission']
        grant = store.authorization(settings['authorization_id'], version=settings.get('authorization_version'))
        documents = receipt.get('documents', {})
        _require(documents.get('authorization') == grant
            and digest(documents.get('package')) == settings['package_hash']
            and grant.get('protocol_hash') == study['protocol_hash']
            and cell['block_id'] in grant['block_ids'] and 'synthetic' in grant['visibilities'],
            'UNAUTHORIZED_DATA', 'pre-fit admission grant/package scope differs')
        _require(metadata.manifest('admission-'+admission_hash) == receipt,
            'IDENTITY_CONFLICT', 'pre-fit original receipt changed between phases')
        with store._read_transaction():
            _current_authority(store, study, cell, attempt, admissions, reserve, events)
        reference = study.get('pirc28_study_definition_hash')
        _require(hash_reference(reference), 'NEEDS_PREREGISTRATION', 'complete pre-fit master required')
        master = SyntheticStudyDefinition.from_dict(metadata.manifest('pirc28-study-definition-'+reference))
        value = master.to_dict()
        _require(master.definition_hash == reference and value['schema_version'] == SCHEMA_V2,
            'IDENTITY_CONFLICT', 'pre-fit master identity/approximation schema differs')
        definition = IdentifiabilitySpec.from_dict(value['identifiability_spec'])
        studies, protocols, consumed = [], {}, len(master.content)+len(encode(receipt))
        for study_id in master.study_ids:
            check_budget()
            source = metadata.manifest('study-'+study_id)['spec']
            protocol = metadata.manifest('protocol-'+source['admission']['protocol_id'])
            consumed += len(encode(source))+len(encode(protocol))
            _require(consumed <= 4*MAXIMUM_DEFINITION_BYTES, 'RESOURCE_LIMIT',
                'pre-fit complete source/protocol metadata exceeds quota')
            _require(source['admission']['mode'] == 'pilot'
                and digest(protocol) == source['protocol_hash']
                and data_binding(protocol) == source['data_hash'],
                'IDENTITY_CONFLICT', 'pre-fit source stage/protocol identities differ')
            studies.append(source)
            protocols[study_id] = protocol
        _require(study_definition_binding(metadata, studies, definition, events=events) == reference,
            'NEEDS_PREREGISTRATION', 'complete master must precede every original admission')
        expected = master.preregistration_fields(protocols)
        for source in studies:
            check_budget()
            preregistration = _preregistration(metadata, source, definition, source['cells'][0], events)
            prereg = metadata.manifest('preregistration-'+preregistration)
            consumed += len(encode(prereg))
            _require(consumed <= 4*MAXIMUM_DEFINITION_BYTES, 'RESOURCE_LIMIT',
                'pre-fit preregistration/admission metadata exceeds quota')
            _require(all(prereg.get(name) == field for name, field in expected.items()),
                'NEEDS_PREREGISTRATION', 'pre-fit preregistration does not freeze the complete matrix')
            _require(source['admission'].get('purpose') == 'fit' and source.get('comparison_plan') == {
                'reference_arm_id': value['arms']['1']['arm_id'],
                'candidate_arm_ids': [value['arms']['2']['arm_id']],
                'preregistration_hash': preregistration, 'failure_policy': 'retain-and-exclude-incomplete-blocks'},
                'NEEDS_PREREGISTRATION', 'original paired comparison/failure policy differs')
        _require(documents.get('protocol') == protocols[study['study_id']],
            'UNAUTHORIZED_DATA', 'pre-fit admission protocol differs')
        check_budget()
        # Pure full-matrix reconstruction above never owns the global OS lock.
        # This last short phase rechecks real authority, journals disclosure,
        # then completes original pinned files AFTER the fresh physical chain
        # and all grant I/O. No immutable view can answer permission questions.
        with store._read_transaction():
            _current_authority(store, study, cell, attempt, admissions, reserve, events)
            authorize_study(store, study['study_id'], grant, 'execute')
            authorize_study(store, study['study_id'], grant, 'fit')
            store._read_completion(
                lambda: _current_authority(store, study, cell, attempt, admissions, reserve, events),
                lambda: None)
            store._read_completion(metadata.finish, lambda: None)
    # No metadata/journal I/O after the final immutable-grant clock guard.
    check_budget()
    require_unexpired_disclosure(store, study['study_id'], grant, 'execute')
    require_unexpired_disclosure(store, study['study_id'], grant, 'fit')
    return {'study_definition_hash': reference, 'attempt_id': attempt['attempt_id'],
        'run_id': attempt['run_id'], 'admission_hash': admission_hash,
        'qualified_identifiability': False, 'real_data_authorized': False}
