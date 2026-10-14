"""Explicit saved-science transport bridge; never a launch or generation API.

Old artifacts stay byte-identical and are restored under their ORIGINAL input,
approval and execution. A successor projection carries that provenance; only
named transport fields change. Both complete typed input contexts must agree
scientifically. Admission/closure and current human authority remain the real
entrypoint/controller's responsibility, not a property granted by this reader.
"""
from copy import deepcopy
from pathlib import Path

from . import formal_partial_predecessor as predecessor
from .formal_input_interruption import context_semantic_identity
from .formal_input_work import restore_input_context
from .formal_origins import build_origin_cases
from .formal_scope_cache import matrix_payload, own_matrix, same_matrix
from .formal_pinned_metadata import OwnedRecord, PinnedMetadata, mutable_record, record_bytes, record_payload
from .formal_metadata_batch import read_metadata, share_batch
from .protocol_core import canonical, digest, envelope, under, unpack

VERSION = 'pirc17-partial-science-scope-bridge-v1'
FIT_VERSION = 'pirc17-imported-fit-result-v1'
FORECAST_VERSION = 'pirc17-imported-forecast-result-v1'


def _contexts(*, predecessor_reference, source_context_reference,
              successor_context_reference, protocol, execution, matrix):
    # Both independently restored typed contexts may share this immutable
    # complete matrix meaning, NOT models, target arrays or mutable indexes.
    matrix = own_matrix(matrix)
    binding, _ = predecessor._load(predecessor_reference)
    b, costs, science, original, _ = predecessor._history(binding)
    source, _ = predecessor._load(source_context_reference)
    target, value = predecessor._load(successor_context_reference)
    # The old context is pinned by the independently restored complete source,
    # not merely a self-consistent record supplied alongside an old model.
    path, root = Path(source_context_reference['path']), Path(b['ledger_directory'])
    if (not path.is_relative_to(root) or str(under(root, path.relative_to(root).as_posix())) != str(path)
            or source['sha256'] != science['context_sha256']
            or protocol != original['protocol'] or matrix != original['matrix']
            or execution['sha256'] == original['execution']['sha256']
            or unpack(execution)['protocol_sha256'] != protocol['sha256']
            or unpack(execution)['matrix_sha256'] != matrix['sha256']):
        raise ValueError('partial science bridge must preserve original protocol/matrix/context')
    new_approval = unpack(value['input_identity'])['approval_sha256']
    if new_approval == costs['approval_sha256']:
        raise ValueError('partial science bridge requires distinct successor authority scope')
    old = restore_input_context(source, protocol=protocol, execution=original['execution'],
        matrix=matrix, approval_sha256=costs['approval_sha256'])
    new = restore_input_context(target, protocol=protocol, execution=execution,
        matrix=matrix, approval_sha256=new_approval)
    semantic = context_semantic_identity(source)
    if context_semantic_identity(target) != semantic:
        raise ValueError('partial science bridge changed original scientific input semantics')
    return binding, b, science, old, new, semantic


def build_scope(*, predecessor_reference, source_context_reference,
                successor_context_reference, protocol, execution, matrix):
    """Build a read-only scope proof, not new input qualification/access events."""
    contexts = _contexts(
        predecessor_reference=predecessor_reference, source_context_reference=source_context_reference,
        successor_context_reference=successor_context_reference, protocol=protocol, execution=execution, matrix=matrix)
    return _scope_record(predecessor_reference=predecessor_reference, source_context_reference=source_context_reference,
        successor_context_reference=successor_context_reference, protocol=protocol, execution=execution, matrix=matrix,
        contexts=contexts)


def _scope_record(*, predecessor_reference, source_context_reference, successor_context_reference,
                  protocol, execution, matrix, contexts):
    binding, _, _, old, new, semantic = contexts
    return envelope(dict(schema_version=VERSION, predecessor=deepcopy(predecessor_reference),
        source_context=deepcopy(source_context_reference), successor_context=deepcopy(successor_context_reference),
        protocol_sha256=protocol['sha256'], matrix_sha256=matrix['sha256'],
        predecessor_sha256=binding['sha256'], source_execution_sha256=old['execution']['sha256'],
        successor_execution_sha256=execution['sha256'], successor_input_sha256=new['input_identity']['sha256'],
        scientific_context_sha256=semantic, read_only=True, authorizes_execution=False,
        new_fits=0, new_forecasts=0, raw_sources_independently_reloaded=False))


class ScopeBridge:
    """One scoped, immutable source collection; no global cache or writer.

    Keep this object with the saved collection to avoid parsing the complete
    partial-source inventory per forecast. Ordinary reads recheck every byte;
    explicitly bracketed restoration batches may reuse private immutable fit
    metadata, with full entry/exit byte checks before handoff. Forecast arrays
    and domains are still checked per item. Owners must recheck source-reference
    stability at their measured admission/audit boundary.
    """
    def __init__(self, reference, *, protocol, execution, matrix, input_identity):
        self.reference = deepcopy(reference)
        record, value = predecessor._load(reference)
        contexts = _contexts(predecessor_reference=value['predecessor'],
            source_context_reference=value['source_context'], successor_context_reference=value['successor_context'],
            protocol=protocol, execution=execution, matrix=matrix)
        expected = _scope_record(predecessor_reference=value['predecessor'],
            source_context_reference=value['source_context'], successor_context_reference=value['successor_context'],
            protocol=protocol, execution=execution, matrix=matrix, contexts=contexts)
        if canonical(record) != canonical(expected) or input_identity['sha256'] != value['successor_input_sha256']:
            raise ValueError('partial science bridge scope/proof/input differs')
        _, self.source, self.science, self.old, self.new, _ = contexts
        if canonical(input_identity) != canonical(self.new['input_identity']):
            raise ValueError('partial science bridge successor input differs')
        self.record = record
        # Both typed contexts have already been independently checked. Keep
        # owned immutable matrix copies; NEVER cache a mutable caller's hash.
        self.old['matrix'] = own_matrix(self.old['matrix'])
        self.new['matrix'] = own_matrix(self.new['matrix'])
        # Complete, independently restored meanings, not trusted SHA headers.
        # Domain consumers can reuse these private immutable copies instead of
        # hashing the same full source catalog several times per saved array.
        # Artifact/source/reference byte checks remain unchanged below.
        for context in (self.old, self.new):
            for key in ('protocol', 'execution', 'input_identity', 'map_catalog'):
                context[key] = OwnedRecord(context[key])
        self.work = {w['work_id']: w for w in matrix_payload(self.old['matrix'])['workloads']}
        self.old_cases = build_origin_cases(self.old['positions'].prefixes, self.old['population'], prior=self.old['prior'])
        self.new_cases = build_origin_cases(self.new['positions'].prefixes, self.new['population'], prior=self.new['prior'])
        self._fit_models = {}
        self._fit_projections = {}
        self._fit_sources = {}
        self._nested_bridges = {}

    def check_references(self):
        """Bounded closure check, not repeated domain replay or native approval."""
        record, value = predecessor._load(self.reference)
        if record != self.record:
            raise ValueError('partial science bridge changed during consumption')
        binding, _ = predecessor._load(value['predecessor'])
        predecessor._history(binding)
        for key in ('source_context', 'successor_context'):
            predecessor._load(value[key])
        for bridge in self._nested_bridges.values():
            bridge.check_references()

    def _nested_bridge(self, record):
        """Retain each verified historical scope once, not once per forecast.

        The predictor-only projection may reuse already verified fit scopes,
        but must NEVER open a full context (and its scoring targets) itself.
        """
        value = record_payload(record)
        if value.get('schema_version') not in {FIT_VERSION, FORECAST_VERSION}:
            return None
        key = digest(value['import_scope'])
        if key not in self._nested_bridges:
            if isinstance(self, FitBridge):
                raise ValueError('target-free fit scope cannot open an unverified nested context')
            self._nested_bridges[key] = bridge_for(record, **{k: self.old[k] for k in
                ('protocol', 'execution', 'matrix', 'input_identity')})
        nested = self._nested_bridges[key]
        share_batch(self, nested)
        # These contexts are private copies already fully verified above;
        # external consumers still use bridge_for's complete scope check.
        # Do not reserialize the full11659-work matrix for every inner read.
        if (nested.reference != value['import_scope']
                or any(nested.new[k]['sha256'] != self.old[k]['sha256']
                       for k in ('protocol', 'execution', 'matrix', 'input_identity'))):
            raise ValueError('nested source belongs to a different verified consumer scope')
        return nested

    def _source(self, work):
        if self.work.get(work.get('work_id')) != work or work['work_id'] not in self.science['completed_sources']:
            raise ValueError('only exact successful original work may be imported')
        completed = self.science['completed_sources'][work['work_id']]
        binding = completed['artifact']
        if binding is None:
            raise ValueError('partial artifact owner required')
        path = under(self.source['ledger_directory'], binding['path'])
        ref = dict(binding, path=str(path))
        if work['kind'] in {'method_fit', 'terrain_fit'}:
            key = work['work_id']
            if key not in self._fit_sources:
                self._fit_sources[key] = PinnedMetadata(ref, root=Path(self.source['ledger_directory']))
            record, value = read_metadata(self, self._fit_sources[key], ref)
        else:
            record, value = predecessor._load(ref)
        nested = self._nested_bridge(record)
        if nested is not None:
            # Even a cached outer projection must notice changed inner bytes.
            # Provenance remains layered; never relabel a wrapper as raw science.
            nested._source(work)
        return ref, record, completed

    def _provenance(self, work):
        ref, record, completed = self._source(work)
        return record, dict(import_scope=deepcopy(self.reference), source_artifact=ref,
                           source_completion=deepcopy(completed))

    def fit_record(self, work):
        from .formal_fit_records import restore_registered_fit
        source, provenance = self._provenance(work)
        if work['kind'] not in {'method_fit', 'terrain_fit'}:
            raise ValueError('imported fit needs its original fit owner')
        # Source bytes/provenance are rechecked above on EVERY consumption.
        # Their full typed fit domain need only be validated once per bridge,
        # not again for every one of hundreds of immutable saved predictions.
        cached = self._fit_projections.get(work['work_id'])
        if cached is not None:
            payload = record_payload(cached)
            if any(payload[key] != value for key, value in provenance.items()):
                raise ValueError('cached imported fit source provenance changed')
            return deepcopy(cached)
        restore_registered_fit(mutable_record(source), work=work, import_bridge=self._nested_bridge(source), **{k: self.old[k] for k in
            ('protocol', 'execution', 'matrix', 'input_identity')})
        # Nested source metadata are immutable; only these top-level transport
        # fields change. Do not deepcopy/reparse all training rows per forecast.
        value = dict(record_payload(source))
        value.update(schema_version=FIT_VERSION, execution_sha256=self.new['execution']['sha256'],
            input_sha256=self.new['input_identity']['sha256'],
            approval_sha256=unpack(self.new['input_identity'])['approval_sha256'], **provenance)
        # In particular, terrain model.training_identity and all parameter
        # identities remain OLD: they are training provenance, not transport.
        projected = OwnedRecord(envelope(value))
        self._fit_projections[work['work_id']] = deepcopy(projected)
        return projected

    def restore_fit(self, record, *, work):
        if record_bytes(record) != record_bytes(self.fit_record(work)):
            raise ValueError('imported fit projection/provenance differs')
        from .formal_fit_records import restore_registered_fit
        _, source, _ = self._source(work)
        return restore_registered_fit(mutable_record(source), work=work, import_bridge=self._nested_bridge(source), **{k: self.old[k] for k in
            ('protocol', 'execution', 'matrix', 'input_identity')})

    def _case(self, work, case):
        mode, rank = work['origin_mode'], work['origin_rank']
        group = self.new_cases[mode]
        expected = group[rank] if rank < len(group) else None
        if (case is None) != (expected is None) or case is not None and canonical(case.identity()) != canonical(expected.identity()):
            raise ValueError('imported forecast changed its successor causal case')
        old = self.old_cases[mode]
        return old[rank] if rank < len(old) else None

    def _fit(self, work):
        fit_work = next(w for w in self.work.values() if w['kind'] in {'method_fit', 'terrain_fit'}
                        and w['fit_identity'] == work['fit_identity'])
        _, old_fit, _ = self._source(fit_work)
        if work['fit_identity'] not in self._fit_models:
            from .formal_fit_records import restore_registered_fit
            self._fit_models[work['fit_identity']] = restore_registered_fit(mutable_record(old_fit), work=fit_work,
                import_bridge=self._nested_bridge(old_fit),
                **{k: self.old[k] for k in ('protocol', 'execution', 'matrix', 'input_identity')})
        return fit_work, old_fit, self._fit_models[work['fit_identity']]

    def forecast_record(self, work, *, case, fit_receipt):
        record, source, ref, old_case, old_fit, model = self._forecast_projection(work, case=case, fit_receipt=fit_receipt)
        from .formal_forecast_records import restore_forecast
        restore_forecast(source, directory=Path(ref['path']).parent, work=work, case=old_case, fit_receipt=old_fit, model=model,
            import_bridge=self._nested_bridge(source),
            **{k: self.old[k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'map_catalog')})
        return record

    def _forecast_projection(self, work, *, case, fit_receipt):
        from .formal_forecast_records import forecast_scope
        source, provenance = self._provenance(work)
        if work['kind'] not in {'scientific_forecast', 'same_grid_reference'}:
            raise ValueError('only original successful prediction kinds may be imported')
        fit_work, old_fit, model = self._fit(work)
        if record_bytes(fit_receipt) != record_bytes(self.fit_record(fit_work)):
            raise ValueError('imported forecast must use the same imported fitted owner')
        old_case = self._case(work, case)
        value = deepcopy(record_payload(source))
        value.update(forecast_scope(work, case=case, fit_receipt=fit_receipt,
            **{k: self.new[k] for k in ('protocol', 'execution', 'matrix', 'input_identity')}))
        value.update(schema_version=FORECAST_VERSION, **provenance)
        if value['maps'] is not None:
            value['maps']['observation']['catalog_sha256'] = self.new['map_catalog']['sha256']
        return envelope(value), source, provenance['source_artifact'], old_case, old_fit, model

    def restore_forecast(self, record, *, work, case, fit_receipt, model, map_catalog):
        if record_bytes(map_catalog) != record_bytes(self.new['map_catalog']):
            raise ValueError('imported forecast changed its map admission catalog')
        expected, source, ref, old_case, old_fit, restored_model = self._forecast_projection(work, case=case, fit_receipt=fit_receipt)
        if record_bytes(record) != record_bytes(expected):
            raise ValueError('imported forecast projection/provenance differs')
        # The consumer's fitted dependency must match the old model that was
        # independently reconstructed, not merely an outer parameter label.
        identity = lambda x: x.dynamics.identity() if hasattr(x, 'dynamics') else x.identity
        if model is None or canonical(identity(model)) != canonical(identity(restored_model)):
            raise ValueError('imported forecast fitted parameter meaning differs')
        from .formal_forecast_records import restore_forecast
        return restore_forecast(source, directory=Path(ref['path']).parent,
            work=work, case=old_case, fit_receipt=old_fit, model=restored_model,
            import_bridge=self._nested_bridge(source),
            **{k: self.old[k] for k in ('protocol', 'execution', 'matrix', 'input_identity', 'map_catalog')})

    def for_fits(self):
        """Give the predictor only fit scope/parameters, NEVER scoring truth."""
        return FitBridge(self)


class FitBridge(ScopeBridge):
    """Target-free view for recovered predictors, not a scoring context.

    The complete typed contexts were independently checked by ScopeBridge.
    Keep only model scopes and fit-source metadata here. References to proof
    files are provenance, not target arrays handed to a prediction kernel.
    Independent fit consumers still reconstruct their OWN model objects.
    """
    def __init__(self, bridge):
        if not isinstance(bridge, ScopeBridge) or isinstance(bridge, FitBridge):
            raise ValueError('full verified scope required before target-free projection')
        self.reference, self.record = deepcopy(bridge.reference), deepcopy(bridge.record)
        self.old, self.new = ({k: deepcopy(context[k]) for k in ('protocol', 'execution', 'matrix', 'input_identity')}
                             for context in (bridge.old, bridge.new))
        self.source = dict(ledger_directory=bridge.source['ledger_directory'])
        self.work = {k: deepcopy(w) for k, w in bridge.work.items() if w['kind'] in {'method_fit', 'terrain_fit'}}
        self.science = dict(completed_sources={k: deepcopy(bridge.science['completed_sources'][k]) for k in self.work})
        # Construct nested target-free views at the independently verified
        # admission boundary, before giving this object to any predictor.
        self._nested_bridges = {}
        for work in self.work.values():
            _, record, _ = bridge._source(work)
            nested = bridge._nested_bridge(record)
            if nested is not None:
                key = digest(record_payload(record)['import_scope'])
                if key not in self._nested_bridges:
                    self._nested_bridges[key] = nested.for_fits()
        self._fit_models, self._fit_projections = {}, deepcopy(bridge._fit_projections)
        self._fit_sources = {}
        share_batch(bridge, self)

    def for_fits(self):
        raise ValueError('fit scope is already target-free')

    def forecast_record(self, *args, **kwargs):
        raise ValueError('target-free fit scope cannot consume saved forecasts')

    def restore_forecast(self, *args, **kwargs):
        raise ValueError('target-free fit scope cannot consume saved forecasts')


def bridge_for(record, *, protocol, execution, matrix, input_identity, bridge=None):
    ref = record_payload(record)['import_scope']
    if bridge is None:
        return ScopeBridge(ref, protocol=protocol, execution=execution, matrix=matrix, input_identity=input_identity)
    if (not isinstance(bridge, ScopeBridge) or bridge.reference != ref
            or not same_matrix(bridge.new['matrix'], matrix)
            or any(record_bytes(bridge.new[k]) != record_bytes(v) for k, v in
                   (('protocol', protocol), ('execution', execution), ('input_identity', input_identity)))):
        raise ValueError('imported artifact belongs to a different bridge/consumer')
    return bridge
