"""Deterministic PRIVATE paper-preparation CSVs from pinned derivatives only.

Not the public final export, a manuscript, scientific qualification or human
acceptance. No scores, intervals, fitted gates or forecasts are recomputed.
Unknowns remain empty cells, not zero or pass. Existing files are never replaced.
"""
import argparse
import csv
import hashlib
import io
import math
from pathlib import Path

from .method_comparisons import FAMILY_DEFINITIONS
from .protocol_core import canonical, envelope, file_hash, read_json, unpack

VERSION = 'pirc17-private-paper-method-tables-v1'
SOURCE_VERSION = 'pirc17-private-fit-only-method-qualification-preview-v1'
SEEDS = (20260814, 20260815, 20260816, 20260817, 20260818)


def tables(value):
    if (value['schema_version'] != SOURCE_VERSION or value['scientific_claim_authorized'] is not False
            or value['independent_raw_output_replay_completed'] is not False
            or set(value['families']) != set(FAMILY_DEFINITIONS)
            or len(value['mechanism_dispositions']) != 28):
        raise ValueError('whole pinned preliminary derivative required, not final cards')
    comparisons = []
    for family in sorted(FAMILY_DEFINITIONS):
        definition = FAMILY_DEFINITIONS[family]
        if set(value['families'][family]) != set(definition):
            raise ValueError('missing or extra registered comparison')
        for candidate in sorted(definition):
            item = value['families'][family][candidate]
            raw = item['statistical_numbers']
            bounds, seeds = raw['simultaneous_interval_m'], raw['seed_delta_m']
            if (raw['candidate'] != candidate or raw['control'] != definition[candidate]
                    or len(bounds) != 2 or len(seeds) != 5
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                           for x in [raw['delta_estimate_m'], raw['holm_adjusted_p_zero'], *seeds,
                                     *[b for b in bounds if b is not None]])
                    or not 0 <= raw['holm_adjusted_p_zero'] <= 1
                    or (None not in bounds and bounds[0] > bounds[1])
                    or item['verdict_authorized'] is not False
                    or item['final_raw_output_audit_completed'] is not False):
                raise ValueError('invalid source interval/seed/participant/qualification')
            # Do NOT force estimate inside interval, clip, round or recenter.
            row = dict(family_id=family, candidate=candidate, control=definition[candidate],
                origin_mode='causal_prefix', independent_blocks=46, forecast_seeds=5,
                particles=512, maximum_step_seconds=5, nominal_horizon_seconds=1800,
                delta_candidate_minus_control_m=raw['delta_estimate_m'],
                simultaneous_lower_m=bounds[0], simultaneous_upper_m=bounds[1],
                holm_adjusted_p_zero=raw['holm_adjusted_p_zero'],
                tail_check_passed=raw['tail_check_passed'],
                **{f'seed_{seed}_delta_m': effect for seed, effect in zip(SEEDS, seeds)},
                mechanism_status=item['mechanism_status'], mechanism_passed=item['mechanism_passed'],
                planning_qualified=item['planning']['qualified'], planning_reason=item['planning']['reason'],
                global_numerically_qualified=item['numerical_applicability']['global_numerically_qualified'],
                precision_invariant_effect_qualified=item['numerical_applicability']['precision_invariant_effect_qualified'],
                final_raw_output_audit_completed=False, scientific_claim_authorized=False,
                review_stage='PROVISIONAL-private-pre-audit')
            comparisons.append(row)
    mechanisms = []
    for slot, item in sorted(value['mechanism_dispositions'].items()):
        receipt = item.get('receipt')
        if item['status'] == 'computed':
            if receipt is None or item['passed'] != receipt['passed']:
                raise ValueError('computed mechanism needs its actual receipt')
        elif item['status'] != 'pending_original_saved_forecast_diagnostics' or item['passed'] is not None:
            raise ValueError('unknown mechanism must not be fabricated as failure/pass')
        mechanisms.append(dict(slot_id=slot, status=item['status'], passed=item['passed'],
            statistic=None if receipt is None else receipt['statistic'],
            value=None if receipt is None else receipt['value'],
            operator=None if receipt is None else receipt['operator'],
            threshold=None if receipt is None else receipt['threshold'],
            units=None if receipt is None else receipt['units'],
            required_source='fitted-model' if receipt is not None else item['required_source'],
            final_raw_output_audit_completed=False, scientific_claim_authorized=False,
            review_stage='PROVISIONAL-private-pre-audit'))
    return {'method-comparisons.csv': comparisons, 'method-mechanisms.csv': mechanisms}


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: '' if v is None else str(v).lower() if isinstance(v, bool) else v for k, v in row.items()})
    return stream.getvalue().encode('utf-8')


def export(value, source_sha256, output_root):
    rows = tables(value)
    outputs = {name: csv_bytes(data) for name, data in rows.items()}
    manifest = dict(schema_version=VERSION, source_qualification_sha256=source_sha256,
        source_statistics_sha256=value['statistics_sha256'], settings_id=value['settings_id'],
        matrix_sha256=value['matrix_sha256'],
        files={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=len(rows[name]))
               for name, data in outputs.items()},
        source_code_sha256=file_hash(Path(__file__)),
        scientific_claim_authorized=False, independent_raw_output_replay_completed=False,
        manuscript_written=False, human_accepted=False, new_forecasts=0, new_fits=0,
        new_scores=0, new_hypothesis_tests=0, new_mechanism_checks=0,
        limitations=['PRIVATE PROVISIONAL numerical paper-preparation tables, not accepted public evidence cards.',
            'Only the complete46-block primary method population; no terrain or secondary-mode table.',
            'Five forecast RNG seeds are not five fitted models or independent blocks.',
            'Candidate minus control ES(m); negative favours candidate; original within-family simultaneous intervals and Holm p-values.',
            'Empty cells mean pending/unavailable, not zero, failure or pass. No categorical verdict inferred.',
            'Mechanism consistency is not predictive benefit, causal attribution or precision invariance.',
            'Original full final analysis/audit/export/manuscript/review/human acceptance remain required.'])
    record = envelope(manifest)
    target = Path(output_root)/record['sha256']
    target.mkdir(parents=True, exist_ok=True)
    outputs['manifest.json'] = canonical(record)+b'\n'
    for name, data in outputs.items():
        path = target/name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('existing table bundle is corrupt; no overwrite/cleanup')
            continue
        with path.open('xb') as stream:
            stream.write(data)
    return target, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    args = parser.parse_args()
    root, source = args.directory.resolve(), args.qualification.resolve()
    if source.parent != root/'offline-method-qualification-preview':
        raise ValueError('this checkpoint pinned private qualification derivative required')
    value = unpack(read_json(source), expected_sha256=args.qualification_sha256)
    path, record = export(value, args.qualification_sha256, root/'offline-method-tables')
    print(dict(path=str(path), sha256=record['sha256'], files=record['payload']['files'],
        scientific_claim_authorized=False, manuscript_written=False, new_forecasts=0, new_scores=0), flush=True)


if __name__ == '__main__':
    main()
