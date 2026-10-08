"""Lazy scientific reader for old imports plus new checkpoint forecasts.

No map provider, raw-data qualification, history replay, prediction or refit.
Only the requested artifact's source chain/arrays are read. Original model
training identities remain original: changing restart transport is not refit.
"""
from copy import deepcopy
from pathlib import Path

from .cached_method_mechanisms import restore_cached_fit
from .checkpoint_resume import load
from .direct_linear import restore_direct_dynamics
from .formal_forecast_records import KINDS, VERSION, forecast_scope, restore_forecast
from .formal_input_work import restore_input_context
from .formal_origins import build_origin_cases
from .formal_saved import SavedForecasts, SavedPrediction
from .method_training import FittedMethod
from .protocol_core import digest, envelope, read_json, unpack

# Transport fields may change across resumes. These scientific values cannot.
SCIENTIFIC_FIELDS = ('work_id', 'status', 'reason', 'arrays', 'diagnostics', 'parameter_identity',
    'fit_identity', 'prediction_seconds', 'generated_forecasts_attempted', 'forecast_version',
    'invalid_feature_rows', 'feature_query_rows', 'scientific_work')


def source_leaf(record, path, *, scientific=False):
    """Follow at most this ONE item's pointers, never its global ledger/bridge."""
    original = unpack(record)
    seen = set()
    for _ in range(16):
        payload = unpack(record)
        if scientific and any(payload.get(k) != original.get(k) for k in SCIENTIFIC_FIELDS):
            raise ValueError('import changed scientific forecast values')
        ref = payload.get('source_artifact')
        if ref is None:
            return record, Path(path)
        identity = (ref['path'], ref['content_sha256'])
        if identity in seen:
            raise ValueError('cyclic checkpoint source pointer')
        seen.add(identity)
        path = Path(ref['path'])
        record = read_json(path, expected_file_sha256=ref['file_sha256'])
        unpack(record, expected_sha256=ref['content_sha256'])
    raise ValueError('unbounded checkpoint source pointer chain')


class CheckpointSavedForecasts(SavedForecasts):
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        settings, imported = load(self.root)
        self.checkpoint_settings = settings
        self._imported_ids = frozenset(imported)
        bundle = unpack(read_json(settings['bundle']), expected_sha256=settings['bundle_sha256'])
        self.protocol, self.execution, self.matrix = (bundle[k] for k in ('protocol','execution','matrix'))
        context_record = read_json(settings['context'])
        unpack(context_record, expected_sha256=settings['context_sha256'])
        context = restore_input_context(context_record, protocol=self.protocol, execution=self.execution,
            matrix=self.matrix, approval_sha256=settings['approval_sha256'])
        self.input_identity, self.population, self.map_catalog = (context[k] for k in
            ('input_identity','population','map_catalog'))
        self.positions = context['positions']  # Scoring owner only; no predictor.
        self.cases = build_origin_cases(context['positions'].prefixes, context['population'], prior=context['prior'])
        m = unpack(self.matrix)
        self.work = {w['work_id']:w for w in m['workloads']}
        self.forecasts = {wid:w for wid,w in self.work.items() if w['kind'] in KINDS}
        self.by_axes = {(w['kind'],w['matrix'],w['subject'],w['origin_mode'],w['origin_rank'],w['seed'],w['repetition']):w
                        for w in self.forecasts.values()}
        self.receipts, self.models, self.training_input_sha256_by_fit = {}, {}, {}
        for w in self.work.values():
            if w['kind'] not in {'method_fit','terrain_fit'}:
                continue
            binding = imported[w['work_id']]['manifest']
            record = read_json(binding['artifact_path'])
            value = unpack(record, expected_sha256=binding['artifact_sha256'])
            leaf, _ = source_leaf(record, binding['artifact_path'])
            original = unpack(leaf)
            artifact = value['artifact']
            if (value['work_id'] != w['work_id'] or value['artifact'] != original['artifact']
                    or value['parameter_identity'] != original['parameter_identity']):
                raise ValueError('import changed fitted parameters/provenance')
            if w['kind'] == 'terrain_fit':
                model = restore_direct_dynamics(artifact['model'])
                identity = model.identity['sha256']
            else:
                dynamics = restore_cached_fit(artifact)
                model = FittedMethod(dynamics, artifact['training'], w['subject'], float(artifact['fit_seconds']))
                identity = dynamics.fit_identity
            if value['parameter_identity'] != identity:
                raise ValueError('saved model parameters changed')
            key = w['fit_identity']
            self.models[key], self.receipts[key] = model, record
            self.training_input_sha256_by_fit[key] = original['input_sha256']
        self.index = {wid:dict(path=entry['manifest']['artifact_path'],
            content_sha256=entry['manifest']['artifact_sha256']) for wid,entry in imported.items() if wid in self.forecasts}
        self.refresh()
        self.identity = envelope(dict(schema_version='pirc17-saved-forecast-context-v1',
            protocol_sha256=self.protocol['sha256'], execution_sha256=self.execution['sha256'],
            matrix_sha256=self.matrix['sha256'], input_sha256=self.input_identity['sha256'],
            population_sha256=self.population['sha256'], map_catalog_sha256=self.map_catalog['sha256'],
            fit_receipt_sha256={k:v['sha256'] for k,v in self.receipts.items()},
            case_sha256={k:[digest(c.identity()) for c in group] for k,group in self.cases.items()}))

    def refresh(self):
        """Refresh only the atomic progress index, never contexts/maps/arrays."""
        progress = read_json(self.root/'progress.json')
        if progress['settings_id'] != self.checkpoint_settings['settings_id']:
            raise ValueError('checkpoint settings changed')
        self.failures = deepcopy(progress['failures'])
        for wid, result in progress['failures'].items():
            if wid in self.forecasts and isinstance(result, dict) and result.get('artifact_path'):
                self.index[wid] = dict(path=result['artifact_path'], content_sha256=result['artifact_sha256'])
        for wid, result in progress['completed'].items():
            if wid in self.forecasts:
                binding = dict(path=result['artifact_path'], content_sha256=result['artifact_sha256'])
                if wid in self._imported_ids or (wid in self.index and self.index[wid] != binding):
                    raise ValueError('checkpoint replaced/duplicated a successful prediction')
                self.index[wid] = binding

    def terminal(self, work):
        return work['work_id'] in self.index or work['work_id'] in self.failures or self.case(work) is None

    def read(self, work):
        if self.forecasts.get(work.get('work_id')) != work:
            raise ValueError('exact registered saved forecast required')
        case, binding = self.case(work), self.index.get(work['work_id'])
        if binding is None:
            failure = self.failures.get(work['work_id'])
            return SavedPrediction('failed' if failure else 'NOT_ADMITTED' if case is None else 'unavailable',
                str(failure) if failure else 'no saved completion; no implicit prediction', None, None, None)
        path = Path(binding['path'])
        record = read_json(path)
        payload = unpack(record, expected_sha256=binding['content_sha256'])
        leaf, leaf_path = source_leaf(record, path, scientific=True)
        value = deepcopy(unpack(leaf))
        if value['schema_version'] != VERSION:
            raise ValueError('unsupported original saved forecast schema')
        fit = self.receipts.get(work['fit_identity'])
        # Rebind transport ONLY. Arrays, fitted parameters, clocks, random
        # streams, diagnostics and original measured kernel costs are unchanged.
        value.update(forecast_scope(work, protocol=self.protocol, execution=self.execution, matrix=self.matrix,
            input_identity=self.input_identity, case=case, fit_receipt=fit))
        if value['maps'] is not None:
            value['maps']['observation']['catalog_sha256'] = self.map_catalog['sha256']
        scoring, native = restore_forecast(envelope(value), directory=leaf_path.parent, work=work,
            protocol=self.protocol, execution=self.execution, matrix=self.matrix, input_identity=self.input_identity,
            case=case, fit_receipt=fit, model=self.models.get(work['fit_identity']), map_catalog=self.map_catalog)
        return SavedPrediction(payload['status'], payload['reason'], record, scoring, native)
