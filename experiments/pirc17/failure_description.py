"""Describe five retained interruptions from metadata, never retry forecasts."""
import argparse
import ast
from pathlib import Path

from .checkpoint_resume import load
from .protocol_core import canonical, digest, file_hash, read_json, unpack


def literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('frozen literal assignment missing')


def failure_entries(workloads, failures, previous):
    work = {w['work_id']: w for w in workloads}
    if len(work) != len(workloads) or len(failures) != 5:
        raise ValueError('exact five retained failures and unique workloads required')
    old, pending = previous['failures'], previous['parallel_pending']
    pending_ids = {p['work_id'] for p in pending.values()}
    if len(old) != 3 or len(pending) != 2 or set(failures) != set(old) | pending_ids or set(old) & pending_ids:
        raise ValueError('three prior failures plus two distinct in-flight interruptions required')
    rows = []
    for ordinal, wid in enumerate(sorted(failures), 1):
        w = work[wid]
        if w['kind'] != 'scientific_forecast' or w['generated_forecasts'] != 1:
            raise ValueError('original scientific forecast required')
        if wid in old:
            if failures[wid] != old[wid] or failures[wid] != 'interrupted/deadline/error; output retained':
                raise ValueError('original generic interruption must be preserved')
            status = 'prior_generic_controller_interruption'
        else:
            if failures[wid] != 'interrupted; full reserved cost retained':
                raise ValueError('interrupted reservation must not be converted into success')
            p = next(p for p in pending.values() if p['work_id'] == wid)
            if p['phase'] != w['phase'] or p['reserved_ns'] != int(w['max_active_seconds'] * 1_000_000_000):
                raise ValueError('original pending work/cost binding differs')
            status = 'predecessor_inflight_without_completion_reply'
        rows.append(dict(failure_label=f'F{ordinal:02d}', configuration=w['subject'],
            matrix=w['matrix'], origin_mode=w['origin_mode'], seed=w['seed'],
            terminal_reason=failures[wid], evidence_class=status,
            model_divergence_established=False, item_exception_available=False))
    return rows


def affected_families(entries, families):
    result = []
    for mode in ('causal_prefix', 'known_velocity', 'point_only'):
        for name, contrasts in families.items():
            matrix = 'terrain' if name.startswith('weighted-es-') else 'NEX326-methods'
            configurations = {c for pair in contrasts.values() for c in pair}
            failed = [r['failure_label'] for r in entries if r['matrix'] == matrix
                      and r['origin_mode'] == mode and r['configuration'] in configurations]
            if failed:
                result.append(dict(family=name, origin_mode=mode, failed_labels=failed,
                    disposition='unavailable_entire_registered_family',
                    affected_contrasts=list(contrasts), successful_subset_inference_allowed=False))
    return result


def project(checkpoint, predecessor):
    checkpoint, predecessor = Path(checkpoint), Path(predecessor)
    settings, _ = load(checkpoint)
    current, previous = read_json(checkpoint/'progress.json'), read_json(predecessor/'progress.json')
    if current['settings_id'] != settings['settings_id'] or previous['settings_id'] != settings['settings_id']:
        raise ValueError('same original checkpoint settings required')
    entries = failure_entries(settings['workloads'], current['failures'], previous)
    if set(current['failures']) & set(current['completed']):
        raise ValueError('failed work also marked completed')
    bundle = unpack(read_json(settings['bundle']))
    protocol, execution = unpack(bundle['protocol']), unpack(bundle['execution'])
    root = Path(__file__).resolve().parents[2]
    source_names = ('inference.py', 'method_comparisons.py', 'comparison_registry.py', 'decision_policy.py', 'formal_paired.py')
    sources = {}
    for name in source_names:
        key = 'PSDE-SDE/experiments/pirc17/'+name
        expected = execution['source_sha256'][key]
        if file_hash(root/'experiments/pirc17'/name) != expected or (name != 'formal_paired.py' and protocol['source_sha256'][key] != expected):
            raise ValueError('frozen family/disposition source differs')
        sources[key] = expected
    primary = literal_assignment((root/'experiments/pirc17/inference.py').read_text(encoding='utf-8'), 'PRIMARY_FAMILY')
    methods = literal_assignment((root/'experiments/pirc17/method_comparisons.py').read_text(encoding='utf-8'), 'FAMILY_DEFINITIONS')
    groups = literal_assignment((root/'experiments/pirc17/comparison_registry.py').read_text(encoding='utf-8'), 'GROUPS')
    families = {'weighted-es-primary': primary, 'weighted-es-lio': {g: ('lio-'+g, 'base') for g in groups}}
    families.update({k: {c: (c, a) for c, a in v.items()} for k, v in methods.items()})
    receipts = {}
    for label, relative in (
        ('last_serial_request', 'sessions/c951b692fe0d4c50a0de0ecced2bbe2a/requests/001483.json'),
        ('last_serial_closed', 'sessions/c951b692fe0d4c50a0de0ecced2bbe2a/closed.json'),
        ('predecessor_parallel_assignment', 'sessions/fc0b224119f247438e8fe290e7377ec9/assignment.json'),
        ('predecessor_parallel_run', 'sessions/fc0b224119f247438e8fe290e7377ec9/run.json')):
        path = predecessor/relative
        receipts[label] = dict(file_sha256=file_hash(path))
    request = read_json(predecessor/'sessions/c951b692fe0d4c50a0de0ecced2bbe2a/requests/001483.json')
    if request['work_id'] not in previous['failures'] or next(w for w in settings['workloads'] if w['work_id'] == request['work_id'])['subject'] != 'lio-river':
        raise ValueError('original river interruption request differs')
    closed = read_json(predecessor/'sessions/c951b692fe0d4c50a0de0ecced2bbe2a/closed.json')
    if closed['process_tree_closed'] is not True or closed['accounting']['active_processes'] != 0:
        raise ValueError('original serial process closure missing')
    assignment = read_json(predecessor/'sessions/fc0b224119f247438e8fe290e7377ec9/assignment.json')
    run = read_json(predecessor/'sessions/fc0b224119f247438e8fe290e7377ec9/run.json')
    if assignment['assignment_id'] != run['assignment_id'] or assignment['settings_id'] != settings['settings_id']:
        raise ValueError('predecessor disjoint assignment binding differs')
    for lane, pending in previous['parallel_pending'].items():
        if pending['work_id'] not in assignment['lanes'][lane] or Path(pending['reply']).is_file():
            raise ValueError('original in-flight no-reply evidence differs')
    for row, wid in zip(entries, sorted(current['failures'])):
        row['reply_file_present_in_checked_roots'] = any((p/'outputs'/wid/'reply.json').is_file() for p in (checkpoint, predecessor))
        if row['reply_file_present_in_checked_roots']:
            raise ValueError('new reply requires explicit evidence reconciliation')
    history = root.parent/'MPA/project/PIRC-17/gists/RUN-01-dev26.md'
    text = history.read_text(encoding='utf-8')
    annotations = []
    for marker, evidence in (
        ('Latest superseding observation:51108 CLOSED1/4ad6c8', 'shared_request_replace_WinError5'),
        ('Old80430 actually CLOSED1/2b5a58', 'supervisor_available_RAM_guard_not_allocation_failure')):
        paragraphs = [p for p in text.split('\n\n') if p.startswith(marker)]
        if len(paragraphs) != 1:
            raise ValueError('unique contemporaneous history paragraph required')
        annotations.append(dict(evidence=evidence, source_file_sha256=file_hash(history),
            paragraph_sha256=digest(paragraphs[0]), evidential_scope='contemporaneous_run_summary_not_item_exception_log'))
    return dict(schema_version='pirc17-failure-description-v1', settings_id=settings['settings_id'],
        protocol_sha256=bundle['protocol']['sha256'], execution_sha256=settings['execution_sha256'],
        matrix_sha256=settings['matrix_sha256'], current_failure_map_sha256=digest(current['failures']),
        predecessor_progress_file_sha256=file_hash(predecessor/'progress.json'), source_sha256=sources,
        metadata_receipts=receipts, contemporaneous_history_annotations=annotations, failures=entries,
        unavailable_families=affected_families(entries, families),
        scope=dict(new_fits=0, new_forecasts=0, new_particle_scores=0, retries=0,
            forecast_arrays_opened=False, independent_saved_output_audit=False,
            physical_power_loss_cause_proven_by_item_log=False, individual_model_failure_rate_estimable=False),
        interpretation='Five retained execution interruptions, not established model divergence; no successful-subset inference; no post-hoc family redefinition.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = project(args.checkpoint, args.predecessor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(canonical(result)+b'\n')


if __name__ == '__main__':
    main()
