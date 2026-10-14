"""Guarded saved-output mechanisms, paired inference and terrain conclusions.

The real controller still owns one-attempt accounting, closure and total IO/
validation costs. This consumer never fits, predicts, refines precision or
uses final data for planning power. Its verification repeats saved-source
arithmetic under the same fixed rules, not independent empirical replication.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from . import final_eval_guard as guard
from .comparison_registry import GROUPS, ORIGIN_MODES
from .decision_policy import decision_policy, registered_contrasts
from .formal_forecast_records import origin_stream_id
from .formal_matrix import terrain_ownership
from .formal_mechanisms import MechanismConsumers
from .formal_paired import FAMILIES, SEED_ROLE, describe_paired, paired_family, terrain_conclusions
from .formal_scoring import ScoringConsumers
from .protocol_core import digest, envelope, publish, read_json, sha256, under, unpack

VERSION = 'pirc17-formal-mechanisms-and-inference-v1'


class SavedScores:
    """Closed score artifacts, not a bypass for unfinished common-score work."""
    def __init__(self, *, scorer, root, index, access_journal):
        if not isinstance(scorer, ScoringConsumers):
            raise ValueError('bound real scorer required')
        self.scorer, self.root, self.index, self.access_journal = scorer, Path(root).resolve(), deepcopy(index), Path(access_journal).resolve()
        paths = []
        for wid, binding in self.index.items():
            if wid not in scorer.work or set(binding) != {'path', 'file_sha256', 'content_sha256', 'metrics_access_started_sha256'}:
                raise ValueError('exact closed common-score index required')
            paths.append(str(under(self.root, binding['path'])).casefold())
            for key in ('file_sha256', 'content_sha256', 'metrics_access_started_sha256'): sha256(binding[key])
        if len(set(paths)) != len(paths):
            raise ValueError('one score artifact cannot replace several registered works')
        self.identity = envelope(dict(schema_version=VERSION+'-score-index', context_sha256=scorer.saved.identity['sha256'],
            scoring_inputs_sha256=scorer.inputs.identity['sha256'], index={key: self.index.get(key) for key in sorted(scorer.work)}))

    def _access(self, key):
        """Check the actual earlier access-start file; closure remains controller-owned."""
        event = unpack(read_json(under(self.access_journal, key+'.json')), expected_sha256=key)
        scope = unpack(self.scorer.saved.input_identity)
        expected = dict(schema_version='pirc17-final-access-event-v1', event='started', access_kind='final_eval_metrics',
            **{k: scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256')})
        if (set(event) != set(expected) | {'attempt_id', 'at_utc'} or any(event[k] != v for k, v in expected.items())
                or not isinstance(event['attempt_id'], str) or len(event['attempt_id']) != 32
                or any(c not in '0123456789abcdef' for c in event['attempt_id'])
                or not isinstance(event['at_utc'], str)
                or datetime.fromisoformat(event['at_utc']).utcoffset() != timedelta(0)):
            raise ValueError('score artifact lacks its matching metric access-start evidence')

    def load(self):
        by_mode = {mode: [] for mode in ORIGIN_MODES}
        dispositions, checked_access = [], set()
        for wid, work in self.scorer.work.items():
            case = self.scorer.saved.case(work)
            binding = self.index.get(wid)
            payload = None
            if binding is not None:
                key = binding['metrics_access_started_sha256']
                if key not in checked_access:
                    self._access(key); checked_access.add(key)
                record = read_json(under(self.root, binding['path']), expected_file_sha256=binding['file_sha256'])
                payload = unpack(record, expected_sha256=binding['content_sha256'])
                # Domain replay, not a schema check or reliance on a PASS flag.
                self.scorer.verify(record, work=work, access_started_sha256=key)
            dispositions.append(dict(work_id=wid, origin_mode=work['origin_mode'], origin_rank=work['origin_rank'],
                admitted=case is not None, score_artifact_status='verified' if payload is not None else 'unavailable',
                score_artifact_sha256=None if binding is None else binding['content_sha256'],
                counts_by_status=None if payload is None else payload['counts_by_status']))
            if case is None:
                continue  # Capacity shortfall is recorded above, not an observed failure.
            if payload is None:
                for forecast in self.scorer.dependencies[wid]:
                    if forecast['scientific']:
                        by_mode[case.mode].append(dict(origin_id=origin_stream_id(case), independent_block_id=case.independent_block_id,
                            seed=forecast['seed'], configuration=forecast['subject'], matrix=forecast['matrix'], origin_mode=case.mode,
                            partition='final_eval', status='unavailable', score_m=None, energy_by_time_m=None, elapsed_seconds=None,
                            reason='required closed common-score artifact unavailable; no implicit replacement scoring',
                            source_score_work_id=wid, source_score_sha256=None))
            else:
                for row in payload['rows']:
                    if not row['scientific']: continue
                    value = {k: deepcopy(row[k]) for k in ('origin_id', 'independent_block_id', 'seed', 'configuration',
                        'matrix', 'origin_mode', 'partition', 'status', 'score_m', 'reason')}
                    metrics = row['scores']
                    value.update(source_score_work_id=wid, source_score_sha256=binding['content_sha256'],
                        energy_by_time_m=None if metrics is None else [r['energy_score_m'] for r in metrics['by_time']],
                        elapsed_seconds=None if metrics is None else [r['elapsed_seconds'] for r in metrics['by_time']])
                    by_mode[case.mode].append(value)
        return by_mode, dispositions


def owner_evidence(saved):
    """Recompute owner closure and inspect each independently restored terrain fit."""
    p, m, scope = unpack(saved.protocol), unpack(saved.matrix), unpack(saved.input_identity)
    expected = terrain_ownership(p['components']['terrain_configurations'])
    if m['terrain_ledger'] != expected:
        raise ValueError('terrain matrix owner closure differs from frozen configurations')
    configurations = {}
    for name, definition in expected.items():
        model = saved.models.get(definition['fit_identity'])
        receipt = saved.receipts.get(definition['fit_identity'])
        if (len(scope['configuration_columns'][name])*2 != definition['conditioner_input_dimension']
                or (model is None) != (receipt is None)):
            raise ValueError('actual encoder/fit availability differs from terrain ownership')
        if model is not None:
            identity = model.identity
            # A restart changes input transport/approval IDs, not the already
            # fitted model's training provenance. Check against its actual
            # original saved fit owner when the checkpoint reader supplies it.
            training_inputs = getattr(saved, 'training_input_sha256_by_fit', {})
            training_input_sha = training_inputs.get(definition['fit_identity'], saved.input_identity['sha256'])
            if (identity['configuration'] != name or identity['configuration_identity'] != definition['configuration_sha256']
                    or identity['conditioner_checkpoint']['input_dim'] != definition['conditioner_input_dimension']
                    or identity['training_identity'] != digest(dict(inputs=training_input_sha, configuration=name))
                    or unpack(receipt)['parameter_identity'] != identity['sha256']):
                raise ValueError('independently fitted terrain owner/dimension/parameter identity differs')
        configurations[name] = dict(definition=definition, status='computed' if model is not None else 'unavailable',
            owner_closure_verified=True, independent_fit_verified=model is not None,
            fit_receipt_sha256=None if receipt is None else receipt['sha256'],
            parameter_identity=None if model is None else model.identity['sha256'])
    return envelope(dict(schema_version=VERSION+'-terrain-ownership', matrix_sha256=saved.matrix['sha256'],
        configurations=configurations, composition_owners=expected['all-terrain']['composition_owners'],
        group_rule='Exact owner closure and independent refits; shared interaction owners stay joint. No unique/additive/synergy claim.',
        empirical_correlation_threshold_added=False, scientific_claim_authorized=False))


def _comparison_gates(family, *, method_gates, terrain_configs, mechanism_sha256, ownership_sha256):
    result = {}
    for name, pair in registered_contrasts(family).items():
        if family.startswith('weighted-es-'):
            evidence = {config: terrain_configs[config] for config in pair}
            available = all(row['status'] == 'computed' for row in evidence.values())
            passed = available and all(row['owner_closure_verified'] and row['independent_fit_verified'] for row in evidence.values())
            source = ownership_sha256
        else:
            evidence = {slot: method_gates[slot] for slot in pair}
            available = all(row['status'] == 'computed' for row in evidence.values())
            passed = available and all(row['passed'] for row in evidence.values())
            source = mechanism_sha256
        result[name] = dict(status='computed' if available else 'unavailable', passed=bool(passed) if available else None,
            source_sha256=source, required_participants=list(pair), evidence=deepcopy(evidence),
            scope='Both registered participants; terrain checks owner closure/refits, not a global numerical certificate.')
    return result


class AnalysisConsumers:
    def __init__(self, *, scores):
        if not isinstance(scores, SavedScores):
            raise ValueError('bound closed common scores required')
        self.scores, self.scorer, self.saved = scores, scores.scorer, scores.scorer.saved
        self.policy = decision_policy()
        if unpack(self.saved.protocol)['components']['decision_policy'] != self.policy:
            raise ValueError('registered decision policy differs')
        self.mechanisms = MechanismConsumers(self.scorer)
        self.work = {wid: w for wid, w in self.saved.work.items() if w['kind'] == 'mechanisms_and_inference'}
        if len(self.work) != 1:
            raise ValueError('one registered mechanisms-and-inference work required')
        self.attempted = set()

    def compute(self, work):
        if not isinstance(work, dict) or self.work.get(work.get('work_id')) != work:
            raise ValueError('exact registered mechanisms-and-inference work required')
        rows, dispositions = self.scores.load()
        mechanisms, ownership = self.mechanisms.compute(), owner_evidence(self.saved)
        mechanism_data, owner_data = unpack(mechanisms), unpack(ownership)
        modes = {}
        for mode in ORIGIN_MODES:
            population = {case.independent_block_id: [origin_stream_id(case)] for case in self.saved.cases[mode]}
            families = {}
            for family in FAMILIES:
                configurations = {c for pair in registered_contrasts(family).values() for c in pair}
                matrix = 'terrain' if family.startswith('weighted-es-') else 'NEX326-methods'
                selected = [row for row in rows[mode] if row['matrix'] == matrix and row['configuration'] in configurations]
                families[family] = paired_family(selected, family_id=family, expected_origins_by_block=population,
                    origin_mode=mode, mechanisms=_comparison_gates(family, method_gates=mechanism_data['modes'][mode]['gates'],
                        terrain_configs=owner_data['configurations'], mechanism_sha256=mechanisms['sha256'], ownership_sha256=ownership['sha256']))
            # The four registered terrain scoring-slot families are descriptive
            # views of the SAME forecast, not four new tests or horizons.
            primary = families['weighted-es-primary']
            slots = {}
            selected = [row for row in rows[mode] if row['matrix'] == 'terrain'
                and row['configuration'] in {c for pair in registered_contrasts('weighted-es-primary').values() for c in pair}]
            for i, nominal in enumerate((60, 300, 900, 1800)):
                complete = primary['descriptive'] is not None
                description = describe_paired([dict(r, score_m=r['energy_by_time_m'][i]) for r in selected], 'weighted-es-primary') if complete else None
                slots[str(nominal)] = dict(nominal_seconds=nominal, status='computed' if complete else 'unavailable',
                    actual_elapsed_seconds_range=None if not complete else [min(r['elapsed_seconds'][i] for r in selected), max(r['elapsed_seconds'][i] for r in selected)],
                    description=description, hypothesis_tests_performed=False, factor_verdict_authorized=False)
            modes[mode] = dict(expected_origins_by_block=population, independent_block_count=len(population),
                scientific_score_rows=len(rows[mode]), counts_by_status=dict(Counter(r['status'] for r in rows[mode])),
                families=families, terrain_scoring_slots=slots, role='primary' if mode == 'causal_prefix' else 'secondary-descriptive')
        primary = modes['causal_prefix']['families']
        factors = terrain_conclusions(primary['weighted-es-primary'], primary['weighted-es-lio'], ownership=ownership)
        ledger = []
        for slot in unpack(self.saved.matrix)['method_ledger']:
            refs = [dict(family_id=family, comparison=name, role='candidate' if pair[0] == slot['slot_id'] else 'control')
                for family in FAMILIES if not family.startswith('weighted-es-')
                for name, pair in registered_contrasts(family).items() if slot['slot_id'] in pair]
            ledger.append(dict(slot_id=slot['slot_id'], disposition=slot['disposition'], exclusion_reason=slot['exclusion_reason'],
                comparison_references=refs, full_anchor=slot['full_anchor'], scientific_rejection_implied_by_exclusion=False,
                role='registered comparisons above' if refs else 'excluded or descriptive/full alias; no additional hypothesis'))
        return dict(schema_version=VERSION, work_id=work['work_id'], protocol_sha256=self.saved.protocol['sha256'],
            execution_sha256=self.saved.execution['sha256'], matrix_sha256=self.saved.matrix['sha256'],
            population_sha256=self.saved.population['sha256'], scoring_inputs_sha256=self.scorer.inputs.identity['sha256'],
            decision_policy_sha256=self.policy['sha256'], saved_score_index_sha256=self.scores.identity['sha256'],
            score_work_dispositions=dispositions, raw_mechanisms=mechanisms, terrain_ownership=ownership,
            modes=modes, factor_conclusions=factors, method_ledger=ledger,
            history=dict(role='baseline-only', terrain_verdict=None), registered_seed_role=SEED_ROLE,
            new_forecasts=0, new_fits=0, scientific_claim_authorized=False, numerically_qualified=False,
            scope='Finite-budget predictor performance in the registered exposed cohort and short horizon; not untouched holdout, 72h or resolution-invariant evidence.',
            remaining_delivery='Runtime/replay, independent whole-output audit, all evidence cards, full manuscript/PDF and human acceptance remain separate work.')

    def _execute(self, access, work, output_directory):
        scope = unpack(self.saved.input_identity)
        if (not isinstance(access, guard.VerifiedAccess) or access.access_kind != 'final_eval_metrics'
                or any(getattr(access, k) != scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256'))):
            raise ValueError('matching guarded analysis access required')
        if work['work_id'] in self.attempted:
            raise ValueError('mechanisms-and-inference work already attempted')
        self.attempted.add(work['work_id'])
        payload = self.compute(work)
        payload['metrics_access_started_sha256'] = sha256(access.access_started_sha256)
        path, record = publish(output_directory, payload)
        return dict(artifact_path=str(path), artifact_sha256=record['sha256'], status='computed', generated_forecasts_attempted=0)

    def verify(self, record, *, work, access_started_sha256):
        expected = self.compute(work)
        expected['metrics_access_started_sha256'] = sha256(access_started_sha256)
        if unpack(record) != expected:
            raise ValueError('saved analysis differs from raw scores/mechanisms/paired-statistic recomputation')
        return dict(paired_statistics_recomputed=True, raw_mechanisms_recomputed=True, new_forecasts=0, new_fits=0)


def analyze_formal_work(*, analysis, work, output_directory, protocol, execution, approval_path, approval_sha256,
                        test_path, review_path, journal_directory, population_path, population_sha256, eligibility_path):
    return guard.guarded_call(access_kind='final_eval_metrics', protocol=protocol, execution=execution,
        approval_path=approval_path, approval_sha256=approval_sha256, test_path=test_path, review_path=review_path,
        journal_directory=journal_directory, population_path=population_path, population_sha256=population_sha256,
        eligibility_path=eligibility_path, operation=lambda access: analysis._execute(access, work, output_directory))
