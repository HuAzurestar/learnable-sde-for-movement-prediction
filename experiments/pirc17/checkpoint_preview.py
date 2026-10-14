"""Preliminary arithmetic over a captured score-cache index, never a forecast gate.

Does not generate paths, refit, rescore arrays, alter live progress or authorize
claims. Missing/error rows fail the whole registered comparison family; no
successful-only intersections. Raw uncertainty is conditional on saved scores,
pending the separate independent output/mechanism audit.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from .checkpoint_resume import load
from .formal_paired import FAMILIES, paired_family
from .decision_policy import registered_contrasts
from .protocol_core import digest, read_json, unpack


def summarize(rows, expected):
    counts = Counter(row['status'] for row in rows)
    complete = len(rows) == expected and counts['success'] == expected
    result = dict(expected=expected, counts=dict(counts), missing=expected-len(rows),
                  status='computed' if complete else 'statistical_failure',
                  weighted_es_m=None, es_by_time_m=None, ade_m=None, fde_m=None,
                  coverage_90_by_time=None)
    if complete:
        result.update(weighted_es_m=statistics.mean(r['score_m'] for r in rows),
            es_by_time_m=[statistics.mean(r['scores']['by_time'][i]['energy_score_m'] for r in rows) for i in range(4)],
            ade_m=statistics.mean(r['scores']['ade_grid_mean_m'] for r in rows),
            fde_m=statistics.mean(r['scores']['fde_m'] for r in rows),
            coverage_90_by_time=[statistics.mean(next(l['covered'] for l in r['scores']['by_time'][i]['region']['levels'] if l['level'] == .9) for r in rows) for i in range(4)])
    return result


def read_cached_row(cache, work, entry, source):
    """Select exactly the captured source version, not an arbitrary old file."""
    matches = []
    for path in (cache/'rows'/work['work_id']).glob('*.json'):
        value = unpack(read_json(path))
        scope = value.get('scope', {})
        if scope.get('source_sha256') != entry['source_sha256']:
            continue
        if (value.get('schema_version') != 'pirc17-checkpoint-scoring-v1-row'
                or scope.get('cache_sha256') != cache.name
                or scope.get('work_sha256') != digest(work)
                or scope['source_sha256'] != source
                or path.stem != digest(scope)):
            raise ValueError('cached score source/work/scope mismatch')
        row = value['row']
        if any(row.get(k) != v for k, v in dict(forecast_work_id=work['work_id'],
                configuration=work['subject'], matrix=work['matrix'], seed=work['seed'],
                origin_mode=work['origin_mode'], origin_rank=work['origin_rank'],
                partition='final_eval', scientific=True).items()):
            raise ValueError('cached row routing mismatch')
        if row['status'] != entry['status']:
            raise ValueError('cache index/row status mismatch')
        if row['status'] == 'success':
            if (type(row.get('score_m')) not in (int, float) or not math.isfinite(row['score_m'])
                    or not isinstance(row.get('scores'), dict)
                    or len(row['scores'].get('by_time', [])) != 4
                    or row['scores']['time_weighted_energy_score_m'] != row['score_m']):
                raise ValueError('empty or invalid successful score')
        elif row['status'] not in {'failed', 'unavailable'} or not row.get('reason'):
            raise ValueError('invalid failed score disposition')
        matches.append((row, value))
    if len(matches) != 1:
        raise ValueError('missing or ambiguous captured score row')
    return matches[0]


def preview(root, cache):
    settings, imported = load(root)
    # Capture the atomic scoring index before live predictor progress. Existing
    # score versions remain immutable; new rows are deliberately not included.
    index = read_json(cache/'progress.json')
    progress = read_json(root/'progress.json')
    context = unpack(read_json(settings['context']), expected_sha256=settings['context_sha256'])
    selected = unpack(context['population'])['selection']['selected']
    populations, origins = {}, {}
    for mode, count in [('causal_prefix', 46), ('known_velocity', 6), ('point_only', 6)]:
        group = selected[:count]
        if len(group) != count:
            raise ValueError('frozen population count changed')
        populations[mode] = {p['independent_block_id']: [digest(['pirc17-formal-origin-stream-v1', p['sample_id'], mode])] for p in group}
        origins[mode] = [(p['independent_block_id'], digest(['pirc17-formal-origin-stream-v1', p['sample_id'], mode])) for p in group]
    works = [w for w in settings['workloads'] if w['kind'] == 'scientific_forecast']
    if len(works) != 11020 or index['cache_sha256'] != cache.name:
        raise ValueError('registered population/cache changed')
    rows, errors, identities = [], [], []
    for work in works:
        wid = work['work_id']
        entry = index['rows'].get(wid)
        if entry is None:
            continue
        binding = None
        if wid in imported:
            m = imported[wid]['manifest']
            binding = dict(path=m['artifact_path'], content_sha256=m['artifact_sha256'])
        for state in (progress['failures'], progress['completed']):
            item = state.get(wid)
            if isinstance(item, dict) and item.get('artifact_path'):
                binding = dict(path=item['artifact_path'], content_sha256=item['artifact_sha256'])
        source = digest(dict(binding=binding, failure=progress['failures'].get(wid)))
        try:
            row, value = read_cached_row(cache, work, entry, source)
            block, origin = origins[work['origin_mode']][work['origin_rank']]
            if (row['independent_block_id'], row['origin_id']) != (block, origin):
                raise ValueError('cached origin differs from frozen population')
            identities.append((wid, digest(value)))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(dict(matrix=work['matrix'], configuration=work['subject'],
                               origin_mode=work['origin_mode'], reason=str(exc)))
            block, origin = origins[work['origin_mode']][work['origin_rank']]
            row = dict(matrix=work['matrix'], configuration=work['subject'], seed=work['seed'],
                origin_mode=work['origin_mode'], origin_rank=work['origin_rank'], partition='final_eval',
                independent_block_id=block, origin_id=origin, status='unavailable', reason=str(exc), score_m=None)
        rows.append(row)
    expected, grouped = Counter(), defaultdict(list)
    for w in works:
        expected[(w['matrix'], w['origin_mode'], w['subject'])] += 1
    for r in rows:
        grouped[(r['matrix'], r['origin_mode'], r['configuration'])].append(r)
    configurations = [dict(matrix=k[0], origin_mode=k[1], configuration=k[2], **summarize(grouped[k], n)) for k, n in sorted(expected.items())]
    families = []
    for mode in populations:
        for family_id in FAMILIES:
            contrasts = registered_contrasts(family_id)
            configs = {c for pair in contrasts.values() for c in pair}
            matrix = 'terrain' if family_id.startswith('weighted-es-') else 'NEX326-methods'
            subset = [r for r in rows if r['origin_mode'] == mode and r['matrix'] == matrix and r['configuration'] in configs]
            mechanisms = {k:dict(status='unavailable', passed=None, reason='independent saved-output mechanism audit pending; preview does not evaluate it') for k in contrasts}
            report = paired_family(subset, family_id=family_id, expected_origins_by_block=populations[mode], origin_mode=mode, mechanisms=mechanisms)
            # Sanitized report: no private origin/block/work identifiers.
            description = report['descriptive']
            effects = {} if description is None else {k:{a:b for a,b in v.items() if a != 'block_seed_delta_m'} for k,v in description['results'].items()}
            raw = {} if report['inference'] is None else report['inference']['results']
            families.append(dict(family_id=family_id, origin_mode=mode,
                status='computed_arithmetic_only' if description is not None else 'statistical_failure',
                independent_blocks=len(populations[mode]), expected_rows=report['expected_rows'],
                missing_rows=len(report['missing_rows']), failed_rows=len(report['failed_rows']),
                effects=effects, raw_inference=raw, qualified_results=report['results']))
    return dict(schema_version='pirc17-preliminary-cache-statistics-v1',
        captured_utc=datetime.now(timezone.utc).isoformat(), cache_sha256=cache.name,
        settings_id=settings['settings_id'], score_index_sha256=digest(index),
        predictor_index_sha256=digest(progress), captured_rows_sha256=digest(sorted(identities)),
        scientific_expected=11020, captured_scientific_rows=len(rows),
        row_status_counts=dict(Counter(r['status'] for r in rows)), cache_validation_errors=errors,
        new_forecasts=0, new_fits=0, new_particle_scores=0,
        independent_raw_output_audit_completed=False, scientific_claim_authorized=False,
        interpretation='Preliminary saved-score arithmetic. Missing/error rows fail the full registered family, never zero-filled or dropped. Complete families retain fixed 2000-resample original inference, but mechanism/owner audit and final acceptance are pending. No adaptive sample, step, horizon, seed, family or threshold tuning.',
        configurations=configurations, families=families)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('preview output must be separate from the live runtime')
    result = preview(args.root, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('captured_utc','captured_scientific_rows','row_status_counts','new_forecasts','new_fits')}))
    print(json.dumps([dict(family=f['family_id'], mode=f['origin_mode'], status=f['status'], missing=f['missing_rows'], failed=f['failed_rows']) for f in result['families']]))


if __name__ == '__main__':
    main()
