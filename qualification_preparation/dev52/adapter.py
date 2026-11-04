"""DEV52 source-only injected boundary. Importing this module imports no product.

The CLI never executes a graph. Programmatic execution requires explicit APIs;
those APIs retain every authority, native-stop, publication and accounting guard.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType

BASE = '556676102f0c2b1797291b2a8e66d0a4facb1402'
GRAPH_SHA = 'aab171c0138c28c99dc2121cab7cbaec50f2ecd12ab473483e1e5437c28de27b'
MANIFEST_SHA = '2f1aeae05c769c6c00bb090999200857ab828946e730324252b0541abacf8bc6'
STORE = 'infrastructure.research_store:ResearchStore'
BUDGET = 'application.research_budget:BudgetSpec'
LEDGER = 'application.research_budget:BudgetLedger'
SUPERVISOR = 'application.research_supervisor:ResearchSupervisor'
OWNER = 'application.walking_pilot_budget:owner_session'
GRID = 'application.walking_pilot_budget:GridClock'
METER = 'application.walking_pilot_budget:GridOwnerMeter'
PREPARE = 'application.walking_stage:prepare_package'
DERIVED = frozenset({'meter', 'prepare_callback', 'live_owner_capability',
                     'retained_prefix', 'execution_binding'})
PENDING = object()


class BoundaryError(ValueError):
    """A binding/preflight failure; never an original runtime verdict."""


def require(condition, message):
    if not condition:
        raise BoundaryError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def checked_bytes(path, sha):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'pinned bytes differ: ' + str(path))
    return raw


def verify_baseline(root):
    """Mechanical bytes + Git blob digest checks; no product import/execution."""
    root = Path(root)
    manifest = json.loads(checked_bytes(Path(__file__).with_name('baseline-manifest.json'), MANIFEST_SHA))
    require(len(manifest) == 686 and len({r['path'] for r in manifest}) == 686,
            'baseline manifest cardinality differs')
    for row in manifest:
        path = Path(row['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'manifest path escapes root')
        raw = checked_bytes(root / path, row['sha256'])
        require(len(raw) == row['bytes'], 'baseline size differs')
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        require(blob == row['oid'], 'baseline Git blob differs: ' + row['path'])
    return 686


def source_signature(root, api):
    """Extract parameter contracts from AST only. Never eval product code."""
    module, symbol = api.split(':')
    tree = ast.parse((Path(root) / (module.replace('.', '/') + '.py')).read_text(encoding='utf-8'))
    body = tree.body
    node = None
    for part in symbol.split('.'):
        node = next((n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part), None)
        require(node is not None, 'source symbol missing: ' + api)
        body = node.body
    params = []
    empty = inspect.Parameter.empty
    def default(n):
        try:
            return ast.literal_eval(n)
        except (ValueError, TypeError):
            require(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == 'time' and n.attr == 'monotonic', 'unsupported source default')
            return time.monotonic
    if isinstance(node, ast.ClassDef):
        # BudgetSpec's generated dataclass constructor is its declared fields.
        require(api == BUDGET, 'constructor source must name __init__')
        for field in body:
            if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                params.append(inspect.Parameter(field.target.id, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=empty if field.value is None else default(field.value)))
    else:
        args = node.args
        positional = args.posonlyargs + args.args
        defaults = [empty] * (len(positional) - len(args.defaults)) + [default(d) for d in args.defaults]
        for n, d in zip(positional, defaults):
            if n.arg != 'self':
                kind = inspect.Parameter.POSITIONAL_ONLY if n in args.posonlyargs else inspect.Parameter.POSITIONAL_OR_KEYWORD
                params.append(inspect.Parameter(n.arg, kind, default=d))
        require(args.vararg is None and args.kwarg is None, 'variadic source unsupported')
        for n, d in zip(args.kwonlyargs, args.kw_defaults):
            params.append(inspect.Parameter(n.arg, inspect.Parameter.KEYWORD_ONLY,
                          default=empty if d is None else default(d)))
    return inspect.Signature(params)


def plain_signature(fn, *, method=False):
    require(inspect.isfunction(fn) or inspect.isclass(fn), 'explicit Python function/class required')
    sig = inspect.signature(fn)
    ps = [p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()]
    if method:
        require(ps and ps[0].name == 'self', 'unbound method required')
        ps = ps[1:]
    return sig.replace(parameters=ps, return_annotation=inspect.Signature.empty)


@dataclass(frozen=True)
class Bindings:
    apis: dict
    owner_local: threading.local
    monotonic: object


@dataclass(frozen=True)
class DataValue:
    api: str
    kwargs: dict


class Resolver:
    """Typed references: field/item only index containers, never getattr/eval."""
    def __init__(self, externals, results, fixtures, *, static=False):
        self.externals, self.results, self.fixtures = externals, results, fixtures
        self.static = static

    def resolve(self, value):
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if not isinstance(value, dict):
            return value
        tags = [k for k in value if k.startswith('$')]
        if not tags:
            return {k: self.resolve(v) for k, v in value.items()}
        require(len(tags) == 1, 'ambiguous reference')
        tag = tags[0]
        name = value[tag]
        if tag == '$external':
            require(set(value) == {tag} and name in self.externals, 'missing/unknown external: ' + str(name))
            return self.externals[name]
        if tag == '$result':
            require(set(value) == {tag, 'path'} and name in self.results, 'unknown/forward result: ' + str(name))
            path = value['path']
            require(type(path) is list, 'result path must be a list')
            result = self.results[name]
            for key in path:
                require(type(key) in (int, str), 'invalid result path component')
                result = self.index(result, key)
            return result
        if tag == '$fixture_bytes':
            require(set(value) == {tag} and name in self.fixtures, 'unknown fixture')
            return self.fixtures[name]
        if tag == '$callback_arg':
            require(set(value) == {tag} and name in {'output', 'deadline'}, 'unknown callback argument')
            key = '@' + name
            require(key in self.externals, 'callback argument outside callback')
            return self.externals[key]
        if tag == '$expression':
            require(set(value) == {tag, 'args'} and type(value['args']) is list, 'invalid expression')
            args = [self.resolve(a) for a in value['args']]
            if name in {'field', 'item'}:
                require(len(args) == 2 and type(args[1]) is (str if name == 'field' else int), 'typed index differs')
                return self.index(*args)
            require(name == 'digest' and len(args) == 1, 'unknown expression')
            return PENDING if self.static and pending(args[0]) else digest(args[0])
        if tag == '$dataclass':
            require(set(value) == {tag, 'kwargs'} and name == BUDGET and type(value['kwargs']) is dict,
                    'unsupported dataclass')
            return DataValue(name, self.resolve(value['kwargs']))
        raise BoundaryError('unknown reference tag: ' + tag)

    def index(self, value, key):
        if value is PENDING and self.static:
            return PENDING
        require((type(value) is dict and type(key) is str and key in value)
                or (type(value) in (list, tuple) and type(key) is int and 0 <= key < len(value)), 'reference path/type differs')
        return value[key]


def pending(value):
    if value is PENDING:
        return True
    if isinstance(value, (list, tuple)):
        return any(pending(v) for v in value)
    if isinstance(value, dict):
        return any(pending(v) for v in value.values())
    return False


def reject_placeholders(value):
    if isinstance(value, str):
        require(value.strip().upper() not in {'UNKNOWN', 'UNPROVEN', 'TODO', 'PLACEHOLDER', 'MISSING'}
                and not value.startswith(('$external', '${', '<')), 'placeholder external rejected')
    elif type(value) in (dict, list, tuple):
        if type(value) is dict:
            require(not any(str(k).startswith('$') for k in value), 'deferred object is not an external value')
            value = value.values()
        for item in value:
            reject_placeholders(item)


class Adapter:
    """One pinned DEV50 graph; constructing/preflighting invokes no injected API."""
    def __init__(self, root, bindings, externals):
        self.root = Path(root)
        base = self.root / 'qualification_preparation/dev50'
        self.graph = json.loads(checked_bytes(base / 'invocation-spec.json', GRAPH_SHA))
        self.fixtures = {r['file']: checked_bytes(base / 'fixtures' / r['file'], r['sha256'])
                         for r in self.graph['fixture_manifest']}
        self._graph_digest = digest(self.graph)
        self.bindings = bindings
        self.external = dict(externals)
        self._used = False
        self.preflight()

    def preflight(self):
        verify_baseline(self.root)
        g, b = self.graph, self.bindings
        require(digest(g) == self._graph_digest, 'pinned graph mutated')
        require(type(b) is Bindings and isinstance(b.owner_local, threading.local), 'explicit owner thread-local required')
        expected = set(g['external_inputs']) - DERIVED
        require(set(self.external) == expected, 'external set differs: missing=' + str(sorted(expected-set(self.external)))
                + ' unknown=' + str(sorted(set(self.external)-expected)))
        require(all(v is not None for v in self.external.values()), 'missing external value')
        reject_placeholders(self.external)
        required = {c['callable'] for c in g['calls']} | {BUDGET, GRID, METER, PREPARE}
        require(set(b.apis) == required, 'API provider set differs')
        for api, fn in b.apis.items():
            constructor = inspect.isclass(fn)
            require(constructor == (api in {STORE, LEDGER, SUPERVISOR, GRID, METER, BUDGET}), 'API kind differs: '+api)
            source = api + '.__init__' if constructor and api != BUDGET else api
            method = not constructor and '.' in api.split(':')[1]
            actual = plain_signature(fn.__init__, method=True) if constructor and api != BUDGET else plain_signature(fn, method=method)
            expected_signature = source_signature(self.root, source)
            require(actual == expected_signature and all(type(p.default) is type(expected_signature.parameters[n].default)
                    for n,p in actual.parameters.items()), 'source signature differs: '+api)
            if method:
                cls, attr = api.rsplit('.', 1)
                require(cls in b.apis and getattr(b.apis[cls], attr) is fn, 'unbound receiver API differs: '+api)
        for cls, methods in ((STORE, ('_read_transaction','_events')), (GRID, ('check','state'))):
            for attr in methods:
                require(plain_signature(getattr(b.apis[cls], attr), method=True)
                        == source_signature(self.root, cls+'.'+attr), 'helper signature differs')
        require(inspect.unwrap(b.apis[OWNER]).__globals__.get('_owner') is b.owner_local
                and b.apis[GRID].check.__globals__.get('_owner') is b.owner_local, 'owner-local provenance differs')
        require(not hasattr(b.owner_local, 'capability') and not hasattr(b.owner_local, 'grid'), 'owner thread already occupied')
        require(callable(b.monotonic), 'explicit monotonic callable required')
        if b.monotonic is not time.monotonic:
            require(inspect.isfunction(b.monotonic) and str(inspect.signature(b.monotonic)) == '()', 'monotonic shape differs')
        e = self.external
        require(isinstance(e['live_grid'], b.apis[GRID]), 'live grid is not the injected GridClock')
        grid = e['live_grid']
        require(grid.pid == os.getpid() and grid.thread == threading.get_ident()
                and grid.scope is None and grid.invocation_id == e['invocation_id'], 'grid process/thread/invocation differs')
        require(type(grid.origin_ns) is int and 0 <= grid.origin_ns <= 2**63-1, 'grid origin differs')
        def hash_value(value):
            return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        for name in ('qualification','decision','billing_decision'):
            ref=e[name]
            require(type(ref) is dict and set(ref)=={'manifest_id','sha256'}
                    and type(ref['manifest_id']) is str and bool(ref['manifest_id']) and hash_value(ref['sha256']),
                    'immutable reference shape differs: '+name)
        previous=e['previous_execution_ref']
        require(type(previous) is dict and (set(previous)=={'manifest_id','sha256'}
                or set(previous)=={'policy_hash','config_hash','study_hash'}), 'predecessor reference shape differs')
        require(all(hash_value(v) for k,v in previous.items() if k!='manifest_id'), 'predecessor digest differs')
        require(type(e['original_config']) is dict and type(e['original_study']) is dict
                and type(e['original_policy']) is dict and type(e['metadata_files']) is dict, 'original input container differs')
        for field in ('inventory_path','grants_path','geometry_path'):
            path=e['metadata_files'].get(field)
            require(isinstance(path,(str,Path)) and Path(path).is_absolute(), 'metadata path shape differs: '+field)
        require(type(e['v1_source_grants']) is list and len(e['v1_source_grants'])==2
                and all(type(v) is dict for v in e['v1_source_grants'])
                and type(e['v1_internal_grant']) is dict, 'original grant containers differ')
        require(Path(e['runtime_root']).is_absolute() and type(e['store_id']) is str and bool(e['store_id']), 'store identity absent')
        require(type(e['allocation_ms']) is int and 0 < e['allocation_ms'] <= 120000, 'allocation milliseconds differ')
        require(type(e['allocation_seconds']) in (int, float) and math.isfinite(e['allocation_seconds'])
                and e['allocation_seconds'] == e['allocation_ms']/1000, 'allocation seconds differ')
        require(type(e['deadline']) in (int, float) and math.isfinite(e['deadline'])
                and type(e['deadline_ns']) is int and e['deadline_ns'] == int(e['deadline']*1000000000), 'deadline clock domain differs')
        for name, n in (('command_builder',1), ('save_report_bytes',0)):
            require(inspect.isfunction(e[name]), 'explicit callback function required: '+name)
            inspect.signature(e[name]).bind(*([PENDING]*n))
        require(type(e['report']) is dict and {'status','report_path','rows'} <= set(e['report']), 'report contract missing')
        # Resolve every static branch now, before even ResearchStore construction.
        ext = {**e, **dict.fromkeys(DERIVED, PENDING)}
        results = {}
        resolver = Resolver(ext, results, self.fixtures, static=True)
        for c in g['calls']:
            receiver = resolver.resolve(c['receiver'])
            args = resolver.resolve(c['args']); kwargs = resolver.resolve(c['kwargs'])
            fn = b.apis[c['callable']]
            inspect.signature(fn).bind(*(([receiver] if c['receiver'] is not None else []) + args), **kwargs)
            for value in args:
                if isinstance(value, DataValue):
                    inspect.signature(b.apis[value.api]).bind(**value.kwargs)
            require(c['id'] not in results, 'duplicate call id')
            results[c['id']] = PENDING
        prepare = g['prepare_callback']
        ext.update({'@output':PENDING,'@deadline':PENDING})
        inspect.signature(b.apis[PREPARE]).bind(*resolver.resolve(prepare['args']))
        return tuple(c['id'] for c in g['calls'])

    def run(self):
        """Explicit callable boundary. Never called by the CLI or automatically."""
        require(not self._used, 'adapter is single use; do not replay uncertain effects')
        self.preflight()
        self._used = True
        e = dict(self.external)
        results = {}
        resolver = Resolver(e, results, self.fixtures)
        b, g = self.bindings, self.graph

        def call(c):
            receiver = resolver.resolve(c['receiver'])
            args = resolver.resolve(c['args']); kwargs = resolver.resolve(c['kwargs'])
            args = [b.apis[v.api](**v.kwargs) if isinstance(v, DataValue) else v for v in args]
            if receiver is not None:
                cls = c['callable'].rsplit('.',1)[0]
                require(isinstance(receiver, b.apis[cls]), 'actual receiver type differs')
                args.insert(0, receiver)
            if c['id'] == 'activation':
                e['execution_binding'] = args[1]
            value = b.apis[c['callable']](*args, **kwargs)
            if c['operation'] == 'construct':
                require(isinstance(value, b.apis[c['callable']]), 'constructor result type differs')
            if c['id'].startswith('artifact'):
                index = int(c['id'][8:])
                require(value == g['artifact_predictions'][index]['predicted_metadata'], 'actual artifact metadata differs')
            results[c['id']] = value
            return value

        owner_index = next(i for i,c in enumerate(g['calls']) if c['id']=='owner')
        for c in g['calls'][:owner_index]:
            call(c)
        owner_call = g['calls'][owner_index]
        cfg = resolver.resolve(owner_call['kwargs']['config'])
        ledger, grid = results['ledger'], e['live_grid']
        e['meter'] = b.apis[METER](ledger,cfg,grid.origin_ns/1000000000,b.monotonic,e['invocation_id'],grid=grid)
        primary = None
        try:
            context = b.apis[OWNER](*resolver.resolve(owner_call['args']), **resolver.resolve(owner_call['kwargs']))
            with context as yielded:
                require(yielded is None, 'original owner_session must yield None')
                results['owner'] = yielded
                try:
                    grid.check(ledger,cfg['pilot_id'])
                    capability = getattr(b.owner_local,'capability',None)
                    require(capability == (os.getpid(),results['store'],cfg['pilot_id'],e['invocation_id']), 'current owner capability differs')
                    require(getattr(b.owner_local,'grid',None) is grid, 'owner grid differs')
                    with results['store']._read_transaction():
                        state = grid.state(ledger)
                        opened = state['open']
                        events = results['store']._events()
                        positions = [i for i,v in enumerate(events) if v['event_id']==opened['event_id'] and v['hash']==opened['hash']]
                        require(len(positions)==1 and positions[0]>0, 'non-genesis owner predecessor absent')
                        before = events[positions[0]-1]
                        prefix = {'event_id':before['event_id'],'hash':before['hash']}
                        require(opened['payload']['billing']['retained_prefix']==prefix and prefix['hash']!='0'*64,
                                'retained prefix is not actual owner predecessor')
                    # Scope-only values are never accepted as serialized externals.
                    e['retained_prefix'], e['live_owner_capability'] = prefix, capability
                    for c in g['calls'][owner_index+1:-1]:
                        if c['id']=='supervisor_run':
                            def prepare(output, deadline):
                                local = Resolver({**e,'@output':output,'@deadline':deadline},results,self.fixtures)
                                # Return the original tuple unchanged; supervisor owns all post-stop verification.
                                return b.apis[PREPARE](*local.resolve(g['prepare_callback']['args']))
                            e['prepare_callback'] = prepare
                        call(c)
                except BaseException as exc:
                    primary = exc
                    e['report'].update(status=getattr(exc,'code','CONFIG_OR_IO_ERROR'),error_type=type(exc).__name__)
                try:
                    call(g['calls'][-1])
                except BaseException as exc:
                    e['report']['settlement_error'] = getattr(exc,'code',type(exc).__name__)
                    if primary is None:
                        primary = exc
                    else:
                        raise primary from exc
                if primary is not None:
                    raise primary
        except BaseException as exc:
            if primary is not None and exc is not primary:
                raise primary from exc
            raise
        finally:
            e.pop('live_owner_capability',None)
        return MappingProxyType(results)


def main():
    print('DEV52 LIVE_EXECUTION_DISABLED: original qualification/resource/authority/OS bindings are not frozen')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())