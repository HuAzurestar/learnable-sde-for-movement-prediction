"""Raw, saved-source method mechanisms; no caller-supplied pass flags.

This is a component of the registered mechanisms/inference work, not a claim
that the whole work item is complete. Invoke inside its guarded, measured
handler. Independent verification repeats diagnostics from saved arrays and
fits, never reruns a forecast or substitutes legacy empirical gate results.
"""
from collections import Counter

from .comparison_registry import ORIGIN_MODES
from .formal_forecast_records import origin_stream_id
from .formal_scoring import ScoringConsumers
from .inference import SEEDS
from .method_mechanisms import (EXACT_REFERENCE, NUMERICAL_SLOTS, SCORE_SLOTS, VARIANCE_SLOTS, SCORE_DRAWS,
    integration_diagnostic, mechanism_registry, model_gate, score_diagnostic, validate_gate, variance_diagnostic)
from .protocol_core import digest, envelope, unpack

VERSION = 'pirc17-formal-raw-mechanisms-v1'


def aggregate_gate(definition, rows, *, expected_count):
    """All required instances must exist; no passing/successful intersection."""
    if len(rows) != expected_count:
        raise ValueError('mechanism row denominator differs')
    for row in rows:
        if row['gate'] is not None:
            validate_gate(row['gate'])
            if row['gate']['slot_id'] != definition['slot_id']:
                raise ValueError('mechanism row belongs to a different slot')
    complete = bool(rows) and all(row['gate'] is not None and row['gate']['status'] == 'computed' for row in rows)
    value = ((max if definition['operator'] == 'le' else min)(row['gate']['value'] for row in rows)) if complete else None
    passed = None if value is None else (value <= definition['threshold'] if definition['operator'] == 'le' else value >= definition['threshold'])
    return dict(slot_id=definition['slot_id'], definition_sha256=digest(definition), source=definition['source'],
        statistic=definition['statistic'], operator=definition['operator'], threshold=definition['threshold'], units=definition['units'],
        status='computed' if complete else 'unavailable', value=value, passed=passed, expected_count=expected_count,
        available_count=sum(row['gate'] is not None and row['gate']['status'] == 'computed' for row in rows),
        reason=None if complete else 'no admitted population or incomplete/undefined registered mechanism evidence',
        aggregation='all required instances; max for le, min for ge; no successful subset',
        rows_sha256=digest(rows), scope_limit=definition['scope_limit'], scientific_claim_authorized=False)


class MechanismConsumers:
    def __init__(self, scorer):
        if not isinstance(scorer, ScoringConsumers):
            raise ValueError('bound common scorer and saved sources required')
        self.scorer, self.saved = scorer, scorer.saved
        self.registry = mechanism_registry()
        m = unpack(self.saved.matrix)
        self.slots = {row['slot_id']: row for row in m['method_ledger']}
        if (set(self.slots) != set(self.registry['slots']) or len(m['method_ledger']) != 36
                or any(row['mechanism_definition'] != self.registry['slots'][slot] for slot, row in self.slots.items())):
            raise ValueError('full method ledger/mechanism definitions changed')
        self.max_draws = m['scoring_only_Gaussian_draws']

    def compute(self):
        saved, scorer = self.saved, self.scorer
        definitions = self.registry['slots']
        models, exclusions = {}, {}
        for slot, row in self.slots.items():
            definition = definitions[slot]
            if definition['disposition'] == 'EXCLUDED':
                exclusions[slot] = dict(disposition='EXCLUDED', reason=definition['reason'], scientific_rejection_implied=False)
            elif definition['source'] == 'fitted-model':
                model = saved.models.get(row['fit_identity'])
                receipt = saved.receipts.get(row['fit_identity'])
                gate = None if model is None else model_gate(slot, model.dynamics)
                models[slot] = dict(fit_identity=row['fit_identity'], fit_receipt_sha256=None if receipt is None else receipt['sha256'],
                    gate=gate, reason='fitted dependency unavailable' if gate is None else None)
        modes, draws, dependencies = {}, 0, set()
        for mode in ORIGIN_MODES:
            cases = saved.cases[mode]
            diagnostic_rows = {slot: [] for slot in sorted(NUMERICAL_SLOTS | SCORE_SLOTS)}
            variance_rows, variance_sources = [], []
            for rank, case in enumerate(cases):
                for seed in SEEDS:
                    cache = {}
                    def read(subject):
                        if subject not in cache:
                            reference = subject == EXACT_REFERENCE
                            work = saved.find(subject, mode, rank, seed,
                                kind='same_grid_reference' if reference else 'scientific_forecast',
                                matrix='NEX326-diagnostic' if reference else 'NEX326-methods')
                            dependencies.add(work['work_id'])
                            cache[subject] = work, saved.read(work)
                        return cache[subject]
                    for slot in sorted(diagnostic_rows):
                        work, prediction = read(slot)
                        witness = dict(origin_id=origin_stream_id(case), independent_block_id=case.independent_block_id,
                            seed=seed, work_id=work['work_id'], context_sha256=scorer.inputs.context_sha256(case),
                            forecast_sha256=None if prediction.record is None else prediction.record['sha256'],
                            gate=None, reason=prediction.reason)
                        if slot in NUMERICAL_SLOTS:
                            ref_work, reference = read(EXACT_REFERENCE)
                            witness.update(reference_work_id=ref_work['work_id'],
                                reference_sha256=None if reference.record is None else reference.record['sha256'])
                            if prediction.native is not None and reference.native is not None:
                                witness['gate'] = integration_diagnostic(slot, prediction.native, reference.native,
                                    approximate_context_sha256=witness['context_sha256'], reference_context_sha256=witness['context_sha256'])
                            elif witness['reason'] is None:
                                witness['reason'] = 'required same-grid reference unavailable: '+str(reference.reason)
                        elif prediction.forecast is not None:
                            witness['gate'] = score_diagnostic(slot, prediction.forecast.positions_m,
                                scorer.inputs.targets[case.sample_id].positions_m, seed=seed, origin_id=origin_stream_id(case),
                                input_identity_sha256=witness['context_sha256'])
                            draws += 4*SCORE_DRAWS
                        diagnostic_rows[slot].append(witness)
                    for slot in ('arm-20/full', 'arm-21/mc', 'arm-21/crn'):
                        work, prediction = read(slot)
                        row = scorer.row(work, prediction)
                        variance_sources.append(dict(work_id=work['work_id'], forecast_sha256=row['saved_forecast_sha256'],
                            status=row['status'], reason=row['reason'], common_score_m=row['score_m']))
                        if prediction.native is not None:
                            variance_rows.append(dict(block_id=case.independent_block_id, origin_id=origin_stream_id(case),
                                seed=seed, slot_id=slot, status=row['status'], score_m=row['score_m'],
                                context_sha256=row['context_sha256'], split='final_eval',
                                elapsed_seconds=prediction.forecast.elapsed_seconds.tolist(), diagnostics=prediction.native.diagnostics))
            expected_origins = {c.independent_block_id: [origin_stream_id(c)] for c in cases}
            expected_variance = len(cases)*len(SEEDS)*3
            if cases and len(variance_rows) == expected_variance:
                variance = variance_diagnostic(variance_rows, expected_origins_by_block=expected_origins)
            else:
                variance = {slot: None for slot in VARIANCE_SLOTS}
            all_rows = {slot: [row] for slot, row in models.items()}
            all_rows.update(diagnostic_rows)
            all_rows.update({slot: [dict(gate=gate, reason='incomplete frozen variance triplets' if gate is None else gate['unavailable_reason'])]
                            for slot, gate in variance.items()})
            aggregates = {slot: aggregate_gate(definitions[slot], rows,
                expected_count=len(cases)*len(SEEDS) if slot in diagnostic_rows else 1) for slot, rows in all_rows.items()}
            if len(aggregates) != 28:
                raise ValueError('all 28 required mechanisms must retain a disposition')
            modes[mode] = dict(expected_origins_by_block=expected_origins, raw_diagnostics=diagnostic_rows,
                variance=dict(expected_rows=expected_variance, available_rows=len(variance_rows), sources=variance_sources, gates=variance),
                gates=aggregates, counts_by_status=dict(Counter(g['status'] for g in aggregates.values())),
                hypotheses_authorized=False, role='primary' if mode == 'causal_prefix' else 'secondary-descriptive')
        if draws > self.max_draws:
            raise ValueError('registered scoring-only draw ceiling exceeded')
        return envelope(dict(schema_version=VERSION, protocol_sha256=saved.protocol['sha256'], execution_sha256=saved.execution['sha256'],
            matrix_sha256=saved.matrix['sha256'], saved_forecast_index_sha256=saved.index_identity(dependencies)['sha256'],
            scoring_inputs_sha256=scorer.inputs.identity['sha256'], mechanism_registry_sha256=self.registry['sha256'],
            fitted_model_gates=models, excluded_slots=exclusions, modes=modes,
            scoring_only_gaussian_draws_evaluated=draws, registered_scoring_only_draw_ceiling=self.max_draws,
            new_forecasts=0, new_fits=0, new_seeds=0, scientific_claim_authorized=False, numerically_qualified=False,
            reanalysis_scope='Deterministic replay of the same scoring-only draws is not independent empirical replication.'))

    def verify(self, record):
        unpack(record)
        if record != self.compute():
            raise ValueError('mechanism evidence differs from raw saved-source recomputation')
        return dict(raw_mechanisms_recomputed=True, new_forecasts=0, new_fits=0)
