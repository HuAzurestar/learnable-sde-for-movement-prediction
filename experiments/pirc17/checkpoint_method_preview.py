"""Read-only method statistics while the full terrain matrix is still running.

This is NOT the registered final analysis, an independent raw-output audit, a
mechanism verdict, or a successful-subset substitute for any missing row.
Only the complete frozen primary method population is accepted. Forecast,
score and controller state are never written; output is a private derivative.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from .checkpoint_resume import load
from .inference import InferenceConfig, SEEDS
from .inference_guard import infer_guarded
from .method_comparisons import DELTA_M, FAMILY_DEFINITIONS
from .protocol_core import digest, file_hash, publish, read_json, unpack

VERSION = 'pirc17-private-primary-method-statistics-preview-v1'


def statistics(rows, population, required):
    """Validate the WHOLE prespecified grid before any family arithmetic."""
    expected = {(rank, seed, slot) for rank in range(len(population))
                for seed in SEEDS for slot in required}
    seen, origins, blocks = set(), {}, set()
    for row in rows:
        key = row['origin_rank'], row['seed'], row['configuration']
        if key not in expected or key in seen:
            raise ValueError('duplicate or unregistered method score')
        rank = row['origin_rank']
        source = population[rank]
        if (row['matrix'] != 'NEX326-methods' or row['origin_mode'] != 'causal_prefix'
                or row['partition'] != 'final_eval' or not row['scientific']
                or row['sample_id'] != source['sample_id']
                or row['independent_block_id'] != source['independent_block_id']
                or row['status'] != 'success' or not np.isfinite(row['score_m'])
                or not row['origin_id']):
            raise ValueError('unfinished/failed/wrong-population row; no subset inference')
        if rank in origins and origins[rank] != row['origin_id']:
            raise ValueError('one frozen rank changes origin identity')
        origins[rank] = row['origin_id']
        blocks.add(row['independent_block_id'])
        seen.add(key)
    if seen != expected or len(blocks) != len(population) or len(set(origins.values())) != len(population):
        raise ValueError('complete one-origin-per-independent-block grid required')
    families = {}
    for name, definition in FAMILY_DEFINITIONS.items():
        contrasts = {candidate: (candidate, control) for candidate, control in definition.items()}
        slots = {slot for pair in contrasts.values() for slot in pair}
        selected = [row for row in rows if row['configuration'] in slots]
        # False here means NOT EVALUATED, not empirical mechanism failure.
        # Retain only numerical outputs; do not export fictitious verdicts.
        result = infer_guarded(selected, config=InferenceConfig(delta_m=DELTA_M),
                               mechanism_passed={c: False for c in contrasts}, family=contrasts)
        for comparison in result['results'].values():
            del comparison['verdict']
            del comparison['reason']
        result.update(mechanism_status='not_evaluated', verdict_authorized=False,
                      registered_seed_role='Five forecast RNG seeds, not five fitted models.')
        result['uncertainty_scope'] = 'Independent primary blocks, conditional on five fixed forecast RNG seeds and saved fitted configurations.'
        families[name] = result
    return dict(primary_rows=len(rows), independent_blocks=len(blocks), families=families,
                counts_by_configuration=dict(Counter(row['configuration'] for row in rows)),
                scientific_claim_authorized=False, final_analysis_completed=False,
                independent_raw_output_replay_completed=False,
                limitations=['No mechanism, power, numerical applicability, terrain or final audit qualification.',
                             'Original final pipeline and original full GOAL remain unchanged.'])


def preview(directory, cache_directory):
    started = time.perf_counter()
    root, cache = Path(directory).resolve(), Path(cache_directory).resolve()
    if cache.parent != root/'offline-scores':
        raise ValueError('existing checkpoint offline-score cache required')
    settings, imported = load(root)
    bundle = unpack(read_json(settings['bundle']), expected_sha256=settings['bundle_sha256'])
    matrix = unpack(bundle['matrix'], expected_sha256=settings['matrix_sha256'])
    context = unpack(read_json(settings['context']), expected_sha256=settings['context_sha256'])
    population = unpack(context['population'])['selection']['selected']
    required = {slot['slot_id'] for slot in matrix['method_ledger'] if slot['disposition'] == 'REQUIRED'}
    works = [w for w in settings['workloads'] if w['kind'] == 'scientific_forecast'
             and w['matrix'] == 'NEX326-methods' and w['origin_mode'] == 'causal_prefix']
    if (len(population) != 46 or len(required) != 28 or len(works) != 6440
            or settings['workloads'] != matrix['workloads']
            or context['matrix_sha256'] != settings['matrix_sha256']):
        raise ValueError('original sealed primary method scope changed')
    progress, scores = read_json(root/'progress.json'), read_json(cache/'progress.json')
    if progress['settings_id'] != settings['settings_id'] or scores['cache_sha256'] != cache.name:
        raise ValueError('checkpoint/cache identity changed')
    rows, sources = [], {}
    for work in works:
        wid = work['work_id']
        completion = imported.get(wid, {}).get('manifest') or progress['completed'].get(wid)
        binding = None if completion is None else dict(path=completion['artifact_path'],
                                                       content_sha256=completion['artifact_sha256'])
        failure = progress['failures'].get(wid)
        indexed = scores['rows'].get(wid)
        source = digest(dict(binding=binding, failure=failure))
        if failure or binding is None or indexed != dict(status='success', source_sha256=source):
            raise ValueError('primary forecast/score missing or changed; no implicit scoring')
        paths = list((cache/'rows'/wid).glob('*.json'))
        matches = []
        for path in paths:
            record = read_json(path)
            value = unpack(record)
            scope = value['scope']
            if scope['source_sha256'] != source:
                continue  # Historical stale binding is not current evidence.
            if (path.stem != digest(scope) or scope['cache_sha256'] != cache.name
                    or scope['work_sha256'] != digest(work)
                    or value['schema_version'] != 'pirc17-checkpoint-scoring-v1-row'):
                raise ValueError('score cache scope changed')
            matches.append((record, value['row']))
        if len(matches) != 1:
            raise ValueError('one exact source-bound cached row required')
        record, row = matches[0]
        if (row['forecast_work_id'] != wid or row['configuration'] != work['subject']
                or row['origin_rank'] != work['origin_rank'] or row['seed'] != work['seed']):
            raise ValueError('score row changed work axes')
        rows.append(row)
        sources[wid] = record['sha256']
    result = statistics(rows, population, required)
    result.update(schema_version=VERSION, settings_id=settings['settings_id'],
                  matrix_sha256=settings['matrix_sha256'], context_sha256=settings['context_sha256'],
                  cache_sha256=cache.name, cached_row_sha256=sources,
                  elapsed_seconds=time.perf_counter()-started, new_forecasts=0, new_fits=0,
                  source_file_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
                      ('checkpoint_method_preview.py', 'inference.py', 'inference_guard.py', 'method_comparisons.py')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cache-directory', type=Path, required=True)
    args = parser.parse_args()
    result = preview(args.directory, args.cache_directory)
    path, record = publish(args.directory/'offline-method-preview', result)
    print(dict(path=str(path), sha256=record['sha256'], primary_rows=result['primary_rows'],
               independent_blocks=result['independent_blocks'], families=len(result['families']),
               comparisons=sum(len(f['results']) for f in result['families'].values()),
               elapsed_seconds=result['elapsed_seconds'], new_forecasts=0, new_fits=0), flush=True)


if __name__ == '__main__':
    main()
