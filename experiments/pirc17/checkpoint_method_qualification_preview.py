"""Private fit-only qualification evidence joined to existing method statistics.

Never runs fitting, prediction, scoring, Gaussian diagnostic draws, a raw
forecast-array replay, or the registered final mechanism/analysis handler.
Fit diagnostics are recomputed from unchanged saved models; the other seven
mechanisms remain explicitly pending. No categorical paper verdict is issued.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

from .cached_method_mechanisms import restore_cached_fit
from .checkpoint_method_preview import VERSION as STATISTICS_VERSION
from .checkpoint_resume import load
from .checkpoint_saved import source_leaf
from .inference import InferenceConfig
from .method_comparisons import DELTA_M, FAMILY_DEFINITIONS
from .method_mechanisms import mechanism_registry, model_gate, validate_gate
from .protocol_core import digest, file_hash, publish, read_json, unpack

VERSION = 'pirc17-private-fit-only-method-qualification-preview-v1'


def join_qualification(statistics, policy, ledger, mechanisms):
    required = {row['slot_id'] for row in ledger if row['disposition'] == 'REQUIRED'}
    if (statistics['schema_version'] != STATISTICS_VERSION or statistics['primary_rows'] != 6440
            or statistics['independent_blocks'] != 46 or len(required) != 28
            or set(mechanisms) != required or set(statistics['families']) != set(FAMILY_DEFINITIONS)
            or policy['inference_config'] != vars(InferenceConfig(delta_m=DELTA_M))):
        raise ValueError('whole frozen primary statistics and all mechanism dispositions required')
    families = {}
    for name, definition in FAMILY_DEFINITIONS.items():
        rules = policy['family_rules'][name]
        contrasts = {candidate: [candidate, control] for candidate, control in definition.items()}
        if (rules['contrasts'] != contrasts or set(statistics['families'][name]['results']) != set(contrasts)
                or statistics['families'][name]['config'] != policy['inference_config']):
            raise ValueError('prespecified contrasts/statistical parameters changed')
        comparisons = {}
        for candidate, control in definition.items():
            raw = statistics['families'][name]['results'][candidate]
            if raw['candidate'] != candidate or raw['control'] != control:
                raise ValueError('statistics changed registered participants')
            gates = {slot: mechanisms[slot] for slot in (candidate, control)}
            available = all(g['status'] == 'computed' for g in gates.values())
            comparisons[candidate] = dict(statistical_numbers=raw, participants=gates,
                mechanism_status='computed' if available else 'pending_original_saved_forecast_diagnostics',
                mechanism_passed=all(g['passed'] for g in gates.values()) if available else None,
                planning=rules['planning_at46'][candidate],
                numerical_applicability=rules['numerical_applicability'][candidate],
                final_raw_output_audit_completed=False, verdict_authorized=False)
        families[name] = comparisons
    return dict(schema_version=VERSION, families=families, mechanism_dispositions=mechanisms,
                mechanism_counts=dict(Counter(g['status'] for g in mechanisms.values())),
                scientific_claim_authorized=False, final_analysis_completed=False,
                registered_full_mechanism_work_completed=False,
                independent_raw_output_replay_completed=False,
                scope='Fit diagnostics only plus sealed pre-evaluation planning/numerical metadata; no whole-method or terrain qualification.')


def preview(directory, statistics_path, statistics_sha256):
    started = time.perf_counter()
    root, statistics_path = Path(directory).resolve(), Path(statistics_path).resolve()
    if statistics_path.parent != root/'offline-method-preview':
        raise ValueError('this checkpoint private method-statistics derivative required')
    statistics = unpack(read_json(statistics_path), expected_sha256=statistics_sha256)
    settings, imported = load(root)
    bundle = unpack(read_json(settings['bundle']), expected_sha256=settings['bundle_sha256'])
    matrix, protocol = unpack(bundle['matrix']), unpack(bundle['protocol'])
    registry = mechanism_registry()
    policy = protocol['components']['decision_policy']
    if (statistics['settings_id'] != settings['settings_id']
            or statistics['matrix_sha256'] != settings['matrix_sha256']
            or statistics['context_sha256'] != settings['context_sha256']
            or registry['sha256'] != policy['method_mechanism_registry_sha256']
            or bundle['matrix']['sha256'] != settings['matrix_sha256']):
        raise ValueError('statistics/registry/checkpoint source identity changed')
    ledger = matrix['method_ledger']
    if any(row['mechanism_definition'] != registry['slots'][row['slot_id']] for row in ledger):
        raise ValueError('sealed method mechanism ledger changed')
    fits, sources = {}, {}
    for work in matrix['workloads']:
        if work['kind'] != 'method_fit':
            continue
        binding = imported[work['work_id']]['manifest']
        record = read_json(binding['artifact_path'])
        value = unpack(record, expected_sha256=binding['artifact_sha256'])
        leaf, _ = source_leaf(record, binding['artifact_path'])
        original = unpack(leaf)
        groups = [g for g in matrix['training_groups']
                  if 'method-fit:'+g['training_components_sha256'] == work['fit_identity']]
        if (len(groups) != 1 or value['work_id'] != work['work_id']
                or value['fit_identity'] != work['fit_identity']
                or value['matrix_sha256'] != settings['matrix_sha256']
                or value['artifact'] != original['artifact']
                or value['parameter_identity'] != original['parameter_identity']
                or value['artifact']['slot_bindings'] != groups[0]['slots']
                or value['artifact']['training']['training_components'] != groups[0]['training_components']):
            raise ValueError('fit provenance/parameters/training ownership changed')
        dynamics = restore_cached_fit(original['artifact'])
        if dynamics.fit_identity != value['parameter_identity']:
            raise ValueError('fit parameter identity differs')
        fits[work['fit_identity']] = dynamics
        sources[work['fit_identity']] = dict(imported_fit_sha256=record['sha256'],
            original_fit_sha256=leaf['sha256'], parameter_identity=dynamics.fit_identity)
    mechanisms = {}
    for row in ledger:
        slot = row['slot_id']
        if row['disposition'] != 'REQUIRED':
            continue
        definition = registry['slots'][slot]
        if definition['source'] == 'fitted-model':
            receipt = model_gate(slot, fits[row['fit_identity']])
            validate_gate(receipt)
            mechanisms[slot] = dict(status='computed', passed=receipt['passed'], receipt=receipt,
                                    fit_source=sources[row['fit_identity']])
        else:
            mechanisms[slot] = dict(status='pending_original_saved_forecast_diagnostics', passed=None,
                required_source=definition['source'], reason='Deferred to original measured final mechanism handler; no placeholder pass or failure.')
    result = join_qualification(statistics, policy, ledger, mechanisms)
    result.update(settings_id=settings['settings_id'], matrix_sha256=settings['matrix_sha256'],
        statistics_sha256=statistics_sha256, mechanism_registry_sha256=registry['sha256'],
        saved_fit_sources=sources, restored_fit_count=len(fits), elapsed_seconds=time.perf_counter()-started,
        new_forecasts=0, new_fits=0, new_scoring_draws=0, forecast_arrays_read=0,
        source_file_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
            ('checkpoint_method_qualification_preview.py', 'method_mechanisms.py', 'cached_method_mechanisms.py')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--statistics-sha256', required=True)
    args = parser.parse_args()
    result = preview(args.directory, args.statistics, args.statistics_sha256)
    path, record = publish(args.directory/'offline-method-qualification-preview', result)
    print(dict(path=str(path), sha256=record['sha256'], mechanism_counts=result['mechanism_counts'],
        restored_fit_count=result['restored_fit_count'], elapsed_seconds=result['elapsed_seconds'],
        new_forecasts=0, new_fits=0, new_scoring_draws=0, forecast_arrays_read=0), flush=True)


if __name__ == '__main__':
    main()
