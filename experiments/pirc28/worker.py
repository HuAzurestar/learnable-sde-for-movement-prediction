"""Internal admitted worker. No standalone unbudgeted numerical CLI route."""

import json
import os
from pathlib import Path
import sys
import time

from application.research_execution import execution_plan
from experiments.pirc25.affine import code_hash
from infrastructure.research_control import WorkerControl
from infrastructure.research_files import opened_regular_file
from infrastructure.research_store import ResearchError, ResearchStore, atomic_write, digest, encode
from .pipeline import execute_recipe, recipe_resources
from .runtime import switching_plugin


def main():
    root, store_id, study_id, cell_hash, output, *restored = sys.argv[1:]
    store = ResearchStore(Path(root), store_id)
    spec = store.manifest('study-' + study_id)['spec']
    if spec['code_hash'] != code_hash() or spec.get('runtime_binding') != {'root': root, 'store_id': store_id}:
        raise ResearchError('IDENTITY_CONFLICT', 'worker source or shared store identity differs')
    cells = [cell for cell in spec['cells'] if digest(cell) == cell_hash]
    if len(cells) != 1 or cells[0].get('visibility') != 'synthetic':
        raise ResearchError('UNAUTHORIZED_DATA', 'this worker accepts only registered synthetic recipes')
    cell, plugin = cells[0], switching_plugin()
    plan = execution_plan(spec, cell, plugin)
    config, inputs = recipe_resources(spec['switching_recipe'], cell['fit_K'])
    if cell['execution']['config'] != config or cell['execution']['inputs'] != inputs:
        raise ResearchError('IDENTITY_CONFLICT', 'declared resources differ from the frozen synthetic recipe')
    control = WorkerControl.from_environment()
    if control is None:
        raise ResearchError('MISSING_INPUT', 'managed supervisor recovery/control channel is required')
    attempt = store.attempts().get(control.descriptor['attempt_id'])
    if attempt is None or attempt['state'] != 'RUNNING':
        raise ResearchError('IDENTITY_CONFLICT', 'control channel has no admitted running attempt')
    if Path(output) != control.directory / 'result.json':
        raise ResearchError('IDENTITY_CONFLICT', 'worker output is outside the managed attempt')
    # Fixture admission is explicitly engineering-only. Scientific admission
    # must bind the actual canonical recipe in the source data protocol.
    if spec['admission']['mode'] != 'fixture':
        import hashlib
        protocol = store.manifest('protocol-' + spec['admission']['protocol_id'])
        expected = hashlib.sha256(encode(spec['switching_recipe'])).hexdigest()
        if not protocol['blocks'] or any(block['dataset_id'] != 'synthetic-pirc28' or block['sha256'] != expected for block in protocol['blocks']):
            raise ResearchError('UNAUTHORIZED_DATA', 'scientific recipe source identity is not frozen synthetic input')
    saved = None
    if restored:
        # Owner created this small staging file after checkpoint authorization.
        if len(restored) != 2:
            raise ResearchError('CONTRACT_MISMATCH', 'restored state content binding is required')
        path = Path(restored[0])
        if path != control.directory / 'pirc28-resume-state.json':
            raise ResearchError('CONTRACT_MISMATCH', 'restored state path or byte quota differs')
        with opened_regular_file(control.directory, path, maximum_bytes=2*1024*1024) as (stream, _, _):
            value = json.loads(stream.read(2*1024*1024 + 1))
        if digest(value) != restored[1]:
            raise ResearchError('IDENTITY_CONFLICT', 'owner checkpoint staging content changed')
        saved = value['method_state']
        if value['rng_state'] != saved['rng_state']:
            raise ResearchError('IDENTITY_CONFLICT', 'restored RNG state differs')
    latest, started = {}, time.monotonic()
    total = len(spec['switching_recipe']['initializations'][str(cell['fit_K'])]) * (spec['switching_recipe']['inference'].get('max_iterations', 30)+1) + 3

    def progress(state):
        latest.clear()
        latest.update(state)

    def poll():
        now = time.monotonic()
        if now >= control.descriptor['deadline']:
            raise ResearchError('TIMEOUT', 'shared supervisor deadline reached')
        if control.poll() is not None and latest:
            completed = sum(state['completed_iterations'] + 1 for state in latest['fit_states'] if state is not None)
            elapsed = max(now-started, 1e-9)
            state = {'step': completed, 'data_position': {'accepted_EM_steps': completed},
                     'method_state': dict(latest), 'rng_state': latest['rng_state']}
            control.save(state, {'completed_steps': completed, 'total_steps': total,
                'throughput_per_second': completed/elapsed, 'eta_seconds': (total-completed)*elapsed/max(1, completed)})
            os._exit(85)

    payload = execute_recipe(spec['switching_recipe'], cell['fit_K'], cell['seed'],
                             check_budget=poll, progress_callback=progress, resume_state=saved)
    units = payload.pop('metric_units')
    result = {'schema_version': 'pirc25-result-v1', 'status': 'SUCCEEDED',
        'spec_hash': digest(spec), 'cell_hash': digest(cell), 'protocol_hash': spec['protocol_hash'],
        'input_hash': spec['data_hash'], 'output_hash': digest(payload), 'metric_units': units,
        'state_order': list(plugin.state_order), 'units': list(plugin.units), 'time_unit': 's',
        'resume_level': plugin.resume_level, 'qualification': 'qualified' if spec['admission']['mode'] == 'formal' else 'fixture',
        **payload}
    content = encode(result)
    if len(content) > plan['maximum_result_bytes']:
        raise ResearchError('RESOURCE_LIMIT', 'result exceeds its frozen byte quota')
    poll()
    atomic_write(Path(output), content)


if __name__ == '__main__':
    main()
