"""Bound, read-only formal forecast collection. No generator or map query.

Construct inside the measured worker from the controller's closed artifact
index. An absent entry is unavailable; a changed indexed file is an integrity
error, never an excuse to select a successful subset. This is not an access
authorization or a substitute for the controller's complete-directory audit.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .comparison_registry import ORIGIN_MODES
from .formal_fit_records import restore_registered_fit
from .formal_forecast_records import KINDS, restore_forecast
from .formal_inputs import FinalPositionInputs, ScoringTargets
from .formal_origins import OriginCase
from .origins import frozen_array
from .formal_scope_cache import matrix_payload, own_matrix, owned_work_index
from .protocol_core import digest, envelope, read_json, sha256, under, unpack


@dataclass(frozen=True)
class SavedPrediction:
    status: str
    reason: str | None
    record: dict | None
    forecast: object | None
    native: object | None


class SavedForecasts:
    def __init__(self, *, protocol, execution, matrix, input_identity, population,
                 cases, fit_receipts, map_catalog, root, index, import_bridge=None):
        self.protocol, self.execution, self.matrix, self.input_identity, self.population, self.map_catalog = (
            deepcopy(x) for x in (protocol, execution, matrix, input_identity, population, map_catalog))
        self.matrix = own_matrix(self.matrix)
        p, e, scope, pop, catalog = (unpack(x) for x in (self.protocol, self.execution,
            self.input_identity, self.population, self.map_catalog))
        m = matrix_payload(self.matrix)
        if (e['protocol_sha256'] != protocol['sha256'] or e['matrix_sha256'] != matrix['sha256']
                or m['protocol_sha256'] != protocol['sha256'] or scope['protocol_sha256'] != protocol['sha256']
                or scope['execution_sha256'] != execution['sha256'] or scope['population_sha256'] != population['sha256']
                or any(catalog[k] != scope[k] for k in ('protocol_sha256', 'execution_sha256', 'approval_sha256', 'population_sha256'))):
            raise ValueError('saved collection protocol/execution/population bindings differ')
        if set(cases) != set(ORIGIN_MODES) or pop['selection']['secondary_selected'] != pop['selection']['selected'][:6]:
            raise ValueError('all three original population modes required')
        self.cases = {mode: tuple(deepcopy(group)) for mode, group in cases.items()}
        for mode, group in self.cases.items():
            selected = pop['selection']['selected' if mode == 'causal_prefix' else 'secondary_selected']
            if (len(group) > m['rank_counts'][mode] or any(not isinstance(c, OriginCase) for c in group)
                    or [dict(sample_id=c.sample_id, independent_block_id=c.independent_block_id, split='final_eval') for c in group] != selected
                    or any(c.mode != mode or c.population_sha256 != population['sha256'] for c in group)
                    or len({c.independent_block_id for c in group}) != len(group)):
                raise ValueError('saved cases differ from frozen one-origin-per-block population')
        all_work = m['workloads']
        # own_matrix already checked EVERY unique descriptor and complete ID.
        # Reuse that private immutable index, not an unverified producer index
        # or another full matrix scan on each independently restored context.
        self.work = dict(owned_work_index(self.matrix))
        self.forecasts = {k: w for k, w in self.work.items() if w['kind'] in KINDS}
        self.by_axes = {(w['kind'], w['matrix'], w['subject'], w['origin_mode'], w['origin_rank'], w['seed'], w['repetition']): w
                        for w in self.forecasts.values()}
        if len(self.by_axes) != len(self.forecasts):
            raise ValueError('duplicate forecast axes')
        self.receipts, self.models = deepcopy(fit_receipts), {}
        self.import_bridge = import_bridge
        fit_work = {w['fit_identity']: w for w in all_work if w['kind'] in {'method_fit', 'terrain_fit'}}
        if set(self.receipts)-set(fit_work):
            raise ValueError('unregistered saved fit')
        for key, receipt in self.receipts.items():
            self.models[key] = restore_registered_fit(receipt, work=fit_work[key], protocol=self.protocol,
                execution=self.execution, matrix=self.matrix, input_identity=self.input_identity, import_bridge=self.import_bridge)
        self.root, self.index, self._indexed_paths = Path(root).resolve(), {}, set()
        for key, binding in index.items():
            if key not in self.forecasts:
                raise ValueError('exact forecast index entries required')
            self.admit(self.forecasts[key], binding)
        # Context is stable as later inertial/replay/runtime work is closed.
        # Every consumer binds an exact dependency projection, not a moving
        # whole-run index or a reconstructed model for every scored origin.
        self.identity = envelope(dict(schema_version='pirc17-saved-forecast-context-v1', protocol_sha256=protocol['sha256'],
            execution_sha256=execution['sha256'], matrix_sha256=matrix['sha256'], input_sha256=input_identity['sha256'],
            population_sha256=population['sha256'], map_catalog_sha256=map_catalog['sha256'],
            fit_receipt_sha256={k: v['sha256'] for k, v in self.receipts.items()},
            case_sha256={k: [digest(c.identity()) for c in v] for k, v in self.cases.items()}))

    def admit(self, work, binding):
        """Register a newly closed artifact, never replace/retry an old one.

        Only the owning controller can attest closure. This method validates
        registration identity; read() still checks the actual bytes/domain.
        No fit restoration, forecast or raw-data access occurs here.
        """
        if not isinstance(work, dict) or self.forecasts.get(work.get('work_id')) != work:
            raise ValueError('exact registered saved forecast required')
        if work['work_id'] in self.index:
            raise ValueError('cannot replace an already registered saved forecast')
        if not isinstance(binding, dict) or set(binding) != {'path', 'file_sha256', 'content_sha256'}:
            raise ValueError('exact forecast index entries required')
        path = str(under(self.root, binding['path'])).casefold()
        sha256(binding['file_sha256']); sha256(binding['content_sha256'])
        if path in self._indexed_paths:
            raise ValueError('one artifact cannot stand in for several physical forecasts')
        self.index[work['work_id']] = deepcopy(binding)
        self._indexed_paths.add(path)

    def index_identity(self, work_ids):
        ids = tuple(work_ids)
        if len(ids) != len(set(ids)) or set(ids)-self.forecasts.keys():
            raise ValueError('unique registered forecast dependencies required')
        return envelope(dict(schema_version='pirc17-saved-forecast-dependencies-v1', context_sha256=self.identity['sha256'],
            index={key: deepcopy(self.index.get(key)) for key in sorted(ids)}))

    def case(self, work):
        group = self.cases[work['origin_mode']]
        return group[work['origin_rank']] if work['origin_rank'] < len(group) else None

    def find(self, subject, mode, rank, seed, *, kind='scientific_forecast', matrix='NEX326-methods'):
        return self.by_axes[kind, matrix, subject, mode, rank, seed, None]

    def read(self, work):
        if not isinstance(work, dict) or self.forecasts.get(work.get('work_id')) != work:
            raise ValueError('exact registered saved forecast required')
        case = self.case(work)
        binding = self.index.get(work['work_id'])
        if binding is None:
            return SavedPrediction('NOT_ADMITTED' if case is None else 'unavailable',
                'rank absent from frozen eligible population' if case is None else 'no closed saved forecast artifact; no regeneration',
                None, None, None)
        path = under(self.root, binding['path'])
        record = read_json(path, expected_file_sha256=binding['file_sha256'])
        payload = unpack(record, expected_sha256=binding['content_sha256'])
        scoring, native = restore_forecast(record, directory=path.parent, work=work, protocol=self.protocol,
            execution=self.execution, matrix=self.matrix, input_identity=self.input_identity, case=case,
            fit_receipt=self.receipts.get(work['fit_identity']), model=self.models.get(work['fit_identity']), map_catalog=self.map_catalog,
            import_bridge=self.import_bridge)
        return SavedPrediction(payload['status'], payload['reason'], record, scoring, native)


class ScoringInputs:
    """Frozen original targets, separate from every predictor and RNG key.

The positions loader's access receipt and original window hashes must also be
audited by the owning handler. Hashing a caller's array is not proof of raw-data
provenance; the constructor checks against the supplied guarded loader bundle.
"""
    def __init__(self, positions, saved):
        if not isinstance(positions, FinalPositionInputs) or not isinstance(saved, SavedForecasts):
            raise ValueError('guarded final position bundle and saved collection required')
        cases = saved.cases['causal_prefix']
        if (positions.population_sha256 != saved.population['sha256']
                or len(positions.prefixes) != len(cases) or len(positions.targets) != len(cases)):
            raise ValueError('scoring targets must preserve the complete frozen population')
        sha256(positions.access_started_sha256)
        rows, self.targets = [], {}
        for case, prefix, target in zip(cases, positions.prefixes, positions.targets):
            if (not isinstance(target, ScoringTargets) or prefix.sample_id != case.sample_id
                    or prefix.independent_block_id != case.independent_block_id or prefix.split != 'final_eval'
                    or prefix.population_sha256 != positions.population_sha256
                    or target.sample_id != case.sample_id or target.window_sha256 != case.window_sha256
                    or prefix.window_sha256 != case.window_sha256 or prefix.scoring_frame != case.scoring_frame
                    or not np.array_equal(prefix.score_seconds, case.score_seconds)
                    or target.elapsed_seconds.shape != (4,) or target.positions_m.shape != (4, 2)
                    or not np.array_equal(target.elapsed_seconds, case.score_seconds)
                    or not np.isfinite(target.positions_m).all()):
                raise ValueError('original target sample/window/frame/clock differs')
            self.targets[case.sample_id] = ScoringTargets(case.sample_id, case.window_sha256,
                frozen_array(target.elapsed_seconds), frozen_array(target.positions_m))
            rows.append(dict(sample_id=case.sample_id, independent_block_id=case.independent_block_id,
                window_sha256=case.window_sha256, scoring_frame=[case.scoring_frame.longitude, case.scoring_frame.latitude],
                elapsed_seconds=target.elapsed_seconds.tolist(), positions_m=target.positions_m.tolist()))
        self.identity = envelope(dict(schema_version='pirc17-formal-scoring-inputs-v1',
            protocol_sha256=saved.protocol['sha256'], execution_sha256=saved.execution['sha256'],
            population_sha256=positions.population_sha256, positions_access_started_sha256=positions.access_started_sha256,
            partition='final_eval', targets=rows, truth_passed_to_predictors=False))
        self.target_sha256 = {r['sample_id']: digest(r) for r in rows}
        for group in saved.cases.values():
            for case in group:
                target = self.targets[case.sample_id]
                primary = next(c for c in cases if c.sample_id == case.sample_id)
                if (case.window_sha256 != target.window_sha256 or case.scoring_frame != primary.scoring_frame
                        or not np.array_equal(case.score_seconds, target.elapsed_seconds)):
                    raise ValueError('secondary mode changed original target window/frame/clock')

    def context_sha256(self, case):
        return digest(dict(case_sha256=digest(case.identity()), scoring_inputs_sha256=self.identity['sha256'],
                           target_sha256=self.target_sha256[case.sample_id]))
