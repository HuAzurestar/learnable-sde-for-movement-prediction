"""Load saved causal inputs/26 models, not thousands of old predictions.

Online terrain encoding needs only the frozen feature specification. It does
not use any stored snapshot trajectory row, so rehashing all 7618 Parquet
files on each restart is unrelated to this operation. Raw online map assets
remain lazy and checked by the unchanged map provider when actually queried.
"""
import uuid
from pathlib import Path

from experiments.nex326.pirc21_adapter import FeatureSelection, FeatureSnapshotAdapter
from experiments.pirc22.consumer import load_benchmark_selection_binding
from .cached_method_mechanisms import restore_cached_fit
from .configurations import configuration_encoder, terrain_configurations
from .direct_linear import restore_direct_dynamics
from .features import CanonicalEncoder
from .formal_forecast_records import KINDS
from .formal_forecasts import ForecastConsumers
from .formal_input_work import restore_input_context
from .formal_maps import RegisteredMaps, _open_query
from .formal_origins import build_origin_cases
from .method_training import FittedMethod
from .protocol_core import digest, read_json, unpack


class OnlineTransforms(FeatureSnapshotAdapter):
    """Specification-only adapter; deliberately cannot load training rows."""
    def __init__(self, root, selection, *, manifest_sha256, spec_sha256):
        self.root = Path(root)
        self.manifest = read_json(self.root/'manifest.json', expected_file_sha256=manifest_sha256)
        self.spec = read_json(self.root/'feature_spec.json', expected_file_sha256=spec_sha256)
        self.selection = selection
        selection.validate()
        if (self.manifest['status'] != 'valid'
                or self.manifest['feature_spec_id'] != self.spec['feature_spec_id']
                or self.manifest['feature_spec_sha256'] != digest(self.spec)):
            raise ValueError('frozen feature specification changed')
        self._factor_by_id = {str(x['factor_id']): x for x in self.spec['factors']}
        self._variant_by_id = {str(x['variant_id']): x for x in self.spec['variants']}
        self._composition_by_id = {str(x['composition_id']): x for x in self.spec['compositions']}
        self._column_factor = {str(c['name']): fid for fid, factor in self._factor_by_id.items()
                               for c in factor['value_columns']}
        self._fit_state = self._fit_scope = None
        self._validate_selection()

    def _load(self, *args, **kwargs):
        raise RuntimeError('online checkpoint transforms cannot load snapshot rows')


def load_imports(settings):
    """ONE small metadata index read; no result/array traversal or copies."""
    manifest = unpack(read_json(settings['import_manifest']), expected_sha256=settings['import_manifest_sha256'])
    if (manifest['execution_sha256'] != settings['execution_sha256']
            or manifest['matrix_sha256'] != settings['matrix_sha256']):
        raise ValueError('saved completion index belongs to another execution')
    return manifest['imported_entries']


class CheckpointForecasts(ForecastConsumers):
    """Same execute/_method/_terrain kernels, simple saved-state constructor."""
    def __init__(self, settings, imported):
        bundle = unpack(read_json(settings['bundle']))
        if digest(bundle) != settings['bundle_sha256']:
            raise ValueError('saved scientific bundle changed')
        self.protocol, self.execution, self.matrix = (bundle[k] for k in ('protocol', 'execution', 'matrix'))
        if self.execution['sha256'] != settings['execution_sha256'] or self.matrix['sha256'] != settings['matrix_sha256']:
            raise ValueError('checkpoint execution/matrix changed')
        p, m = unpack(self.protocol), unpack(self.matrix)
        from .formal_environment import ConfiguredRuntime
        self.runtime = ConfiguredRuntime(self.protocol)  # Thread counts once, not per prediction.
        context_record = read_json(settings['context'])
        unpack(context_record, expected_sha256=settings['context_sha256'])
        context = restore_input_context(context_record, protocol=self.protocol,
            execution=self.execution, matrix=self.matrix, approval_sha256=settings['approval_sha256'])
        self.input_identity = context['input_identity']
        self.settings = p['forecast_contract']['forecast']
        self.cases = build_origin_cases(context['positions'].prefixes, context['population'], prior=context['prior'])
        # Target objects stay in the saved context/scoring owner, not in this
        # forecast consumer. All predictors receive only the causal cases.
        self.work = {w['work_id']: w for w in m['workloads'] if w['kind'] in KINDS}
        selected = load_benchmark_selection_binding()['selected_configuration']
        frozen = p['dataset_inputs']['snapshot']
        adapter = OnlineTransforms(settings['input_paths']['snapshot'], FeatureSelection(
            variant_ids=tuple(selected['variant_ids']), composition_ids=tuple(selected['composition_ids'])),
            manifest_sha256=frozen['manifest_sha256'], spec_sha256=frozen['feature_spec_file_sha256'])
        full = CanonicalEncoder(adapter)
        self.encoders = {name: configuration_encoder(full, name) for name in terrain_configurations()}
        if unpack(self.input_identity)['configuration_columns'] != {k:list(e.columns) for k,e in self.encoders.items()}:
            raise ValueError('saved model feature columns changed')
        self.models, self.receipts = {}, {}
        for work in m['workloads']:
            if work['kind'] not in {'method_fit', 'terrain_fit'}:
                continue
            record = read_json(imported[work['work_id']]['manifest']['artifact_path'])
            value = unpack(record, expected_sha256=imported[work['work_id']]['manifest']['artifact_sha256'])
            if (value['work_id'] != work['work_id'] or value['fit_identity'] != work['fit_identity']
                    or value['input_sha256'] != self.input_identity['sha256']):
                raise ValueError('saved model owner/input changed')
            artifact = value['artifact']
            if work['kind'] == 'terrain_fit':
                model = restore_direct_dynamics(artifact['model'])
                parameter_id = model.identity['sha256']
            else:
                dynamics = restore_cached_fit(artifact)
                model = FittedMethod(dynamics, artifact['training'], work['subject'], float(artifact['fit_seconds']))
                parameter_id = dynamics.fit_identity
            if value['parameter_identity'] != parameter_id:
                raise ValueError('saved model parameters changed')
            self.models[work['fit_identity']], self.receipts[work['fit_identity']] = model, record
        if len(self.models) != 26:
            raise ValueError('all 26 previously trained models are required; no refit')
        catalog = context['map_catalog']
        old = unpack(catalog)
        root = Path(settings['input_paths']['data_root'])
        policy = p['dataset_inputs']['online_maps']
        parents = sorted(old['registered_parent_witnesses'])
        query, sources = _open_query(root, policy, parents, self.execution)
        if sources != old['source_sha256']:
            query.close()
            raise ValueError('online map source changed')
        self.maps = RegisteredMaps(query, catalog, root=root, policy=policy, parents=parents, execution=self.execution)
        try:
            self.maps.observation()
        except BaseException:
            self.maps.close()
            raise
        self.attempted = set(imported) & self.work.keys()
        self.import_bridge = None
        self.driver_origin, self.drivers = None, {}
        self.runtime_hardware, self.runtime_maps, self.runtime_state = None, None, None
        self.runtime_session = uuid.uuid4().hex

    def close_all(self):
        super().close()
        self.maps.close()
