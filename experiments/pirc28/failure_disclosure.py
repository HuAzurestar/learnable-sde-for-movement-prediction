"""Bounded V2 lineage summaries; full failures stay in canonical source artifacts."""

from infrastructure.research_store import ResearchError, digest, encode
from .approximation import RECIPE_SCHEMA


def disclose_failure(result, study, artifact_id):
    body = result['failure']
    if study['switching_recipe']['schema_version'] != RECIPE_SCHEMA:
        return body  # Historical engineering-only legacy disclosure contract.
    if artifact_id != digest(result):
        raise ResearchError('CONTRACT_MISMATCH', 'failure disclosure original artifact differs')
    reports = {}
    for name in ('approximation_checks', 'k1_boundary_checks'):
        report = body.get(name)
        reports[name] = None if report is None else {
            'report_hash': digest(report), 'state': report['state'],
            'paths': [{**{key: item for key, item in path.items() if key not in
                ('coarse_summary', 'fine_summary', 'reference_summary', 'reference_batch_summaries')},
                'complete_source_path_hash': digest(path)} for path in report['paths']]}
    value = {'schema_version': 'pirc28-bounded-failure-disclosure-v1',
        'source_artifact_id': artifact_id, 'source_schema': result['source_schema'],
        'complete_failure_hash': digest(body), 'stage': body['stage'], 'error_code': body['error_code'],
        'cause': body.get('cause'), 'recipe_hash': body['recipe_hash'],
        'generated_inputs': body['generated_inputs'], 'selection_hash': digest(body.get('selection')),
        'initializations': [{'initialization_id': item['initialization_id'], 'state': item['state'],
            'error_code': item.get('error_code'), 'model_hash': item.get('model_hash'),
            'last_accepted_state_hash': None if item.get('last_accepted_state') is None
                else digest(item['last_accepted_state'])} for item in body['initializations']],
        'reports': reports,
        'native_execution_hashes': {name: None if body.get(name) is None else digest(body[name])
                                  for name in ('reference_execution', 'k1_boundary_execution')},
        'successful_metrics_available': False, 'qualified_identifiability': False,
        'real_data_authorized': False}
    if len(encode(value)) > 64*1024:
        raise ResearchError('RESOURCE_LIMIT', 'bounded failure disclosure exceeds metadata quota')
    return value
