"""Internal supervisor-only qualification worker, never an unbudgeted CLI."""

import json
from pathlib import Path
import sys
import time

from experiments.pirc25.affine import code_hash
from infrastructure.research_control import WorkerControl
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, ResearchStore, atomic_write, digest, encode
from .computation import MAXIMUM_REQUEST_BYTES, _sources, computation_plan, validate_computation_result
from .evidence import collect_qualification
from .qualification import IdentifiabilitySpec


def main(argv=None):
    root, store_id, request_path, output = sys.argv[1:] if argv is None else argv
    control = WorkerControl.from_environment()
    if control is None:
        raise ResearchError('MISSING_INPUT', 'qualification requires the shared worker control channel')
    if (Path(output) != control.directory/'result.json'
            or Path(request_path) != control.directory/'qualification-request.json'):
        raise ResearchError('IDENTITY_CONFLICT', 'qualification paths are outside managed attempt')
    with opened_regular_file(control.directory, Path(request_path), maximum_bytes=MAXIMUM_REQUEST_BYTES) as (stream, _, _):
        content = stream.read(MAXIMUM_REQUEST_BYTES+1)
    request = json.loads(content)
    store = ResearchStore(Path(root), store_id)
    frozen = request['frozen']
    attempt = store.attempts().get(control.descriptor['attempt_id'])
    derived = store.manifest('study-'+request['study_id'])['spec']
    run = None if attempt is None else store.manifest('run-'+attempt['run_id'])
    if (content != encode(request) or len(content) > MAXIMUM_REQUEST_BYTES
            or request['schema_version'] != 'pirc28-qualification-computation-request-v1'
            or frozen['runtime_binding'] != {'root': root, 'store_id': store_id}
            or frozen['runtime_code_hash'] != code_hash() or request['spec_hash'] != digest(derived)
            or derived.get('pirc28_qualification_computation') != frozen
            or attempt is None or attempt['state'] != 'RUNNING'
            or request['attempt_id'] != attempt['attempt_id'] or run['study_id'] != request['study_id']
            or store.manifest('pirc28-qualification-request-'+attempt['attempt_id']) != request):
        raise ResearchError('IDENTITY_CONFLICT', 'qualification computation admission/source identity differs')
    definition = IdentifiabilitySpec.from_dict(frozen['definition'])
    if computation_plan(definition, frozen['resource_plan']['maximum_total_source_bytes']) != frozen['resource_plan']:
        raise ResearchError('RESOURCE_PLAN_REJECTED', 'qualification allocation differs from frozen source')

    def poll():
        if time.monotonic() >= control.descriptor['deadline']:
            raise ResearchError('TIMEOUT', 'qualification shared deadline reached')
        if control.poll() is not None:
            control.save({'request_hash': digest(request), 'restart': frozen['resource_plan']['recovery']},
                {'completed_steps': 0, 'total_steps': 1, 'throughput_per_second': 0, 'eta_seconds': 0})
            raise SystemExit(85)

    poll()
    _sources(store, frozen)
    report = collect_qualification(store, definition, sorted(frozen['study_hashes']), frozen['authorizations'],
        check_budget=poll, allow_fixture=frozen['allow_fixture'],
        maximum_total_bytes=frozen['resource_plan']['maximum_total_source_bytes'],
        expected_attempts_hash=frozen['source_attempts_hash'])
    result = {'schema_version': 'pirc28-qualification-computation-result-v1',
        'request_hash': digest(request), 'report_hash': digest(report), 'report': report}
    poll()
    validate_computation_result(store, result, request)
    atomic_write(Path(output), encode(result))
    poll()


if __name__ == '__main__':
    main()
