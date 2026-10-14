"""Concrete CPU runtime binding, without loading research inputs.

Both the controller and retained worker configure this environment. Import and
verification cost belongs to their measured startup, not to per-forecast cold
latency. No fit, forecast, map query, CUDA operation or qualification is run.
The fingerprint binds observed versions, module entry files and loaded native
thread-pool libraries; it is not a hash of every installed package byte.
"""
from copy import deepcopy
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys

from .protocol_core import canonical, decode, digest, envelope, file_hash, unpack

VERSION = 'pirc17-effective-cpu-runtime-v1'
PACKAGES = ('numpy', 'torch', 'numba', 'llvmlite', 'duckdb', 'pyarrow',
            'rasterio', 'pandas', 'scipy', 'psutil', 'threadpoolctl')
HOST_FIELDS = ('platform', 'processor', 'physical_cpus', 'logical_cpus',
               'ram_total_bytes', 'python', 'numpy', 'torch', 'torch_threads',
               'torch_interop_threads')
PRELOAD = (*PACKAGES, 'pyarrow.parquet',
           'experiments.pirc17.formal_worker', 'experiments.pirc17.formal_results')


def _file(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, file_sha256=file_hash(path))


def _hardware():
    from .runtime import hardware_manifest
    value = decode(canonical(hardware_manifest()))  # TorchVersion is a str subclass.
    pools = value['loaded_threadpools']
    if not isinstance(pools, list) or not pools:
        raise ValueError('actual native thread-pool inventory required')
    value['loaded_threadpools'] = sorted(pools, key=lambda x: x['filepath'])
    return value


def _check_policy(observed, policy):
    for name in HOST_FIELDS:
        if type(observed.get(name)) is not type(policy[name]) or observed[name] != policy[name]:
            raise ValueError('sealed runtime mismatch: '+name)
    if (observed['device'] != 'cpu'
            or observed['numba_threads'] != policy['numba_threads_when_loaded']
            or observed['optional_map_packages'] != policy['map_packages']):
        raise ValueError('sealed CPU/Numba/map-package runtime mismatch')
    pools = [p for p in observed['loaded_threadpools']
             if p['internal_api'] == 'openblas' and 'numpy.libs' in Path(p['filepath']).parts]
    if (len(pools) != 1 or pools[0]['num_threads'] != policy['numpy_openblas_threads']
            or pools[0]['version'] != policy['numpy_openblas_version']):
        raise ValueError('actual NumPy OpenBLAS version/thread count differs')


class ConfiguredRuntime:
    """Retain native limits for the entire process; fail on runtime drift.

    Construct before research inputs, once per process. The two DuckDB probes
    inspect effective settings on empty, in-memory connections. They are NOT
    evidence of a raw-input read; the bound readers separately SET 2/4 threads.
    """
    def __init__(self, protocol):
        self.protocol_sha256 = protocol['sha256']
        self.policy = deepcopy(unpack(protocol)['resource_contract']['runtime_binding'])
        self.environment = {'OMP_NUM_THREADS': str(self.policy['torch_threads']),
            'MKL_NUM_THREADS': str(self.policy['torch_threads']),
            'OPENBLAS_NUM_THREADS': str(self.policy['numpy_openblas_threads']),
            'NUMBA_NUM_THREADS': str(self.policy['numba_threads_when_loaded'])}
        os.environ.update(self.environment)  # Process-local, not system settings.
        for name in PRELOAD:
            importlib.import_module(name)
        from .development_rollout import resolve_map_backend
        resolve_map_backend('multicell')  # Imports only; no provider/assets/query.
        import torch
        import numba
        from threadpoolctl import ThreadpoolController
        torch.set_num_threads(self.policy['torch_threads'])
        if torch.get_num_interop_threads() != self.policy['torch_interop_threads']:
            torch.set_num_interop_threads(self.policy['torch_interop_threads'])
        numba.set_num_threads(self.policy['numba_threads_when_loaded'])
        # Environment variables do not retune an already-loaded native pool.
        # The registered Windows host has a separate non-Torch Intel OpenMP
        # library. Bind only that pool to the registered native/Numba12 budget:
        # a preloaded parent and its fresh OMP=6 child otherwise differ12/6.
        # Do not touch Torch's libraries, its stub, or the MSVC OpenMP pool.
        # Inventory after the APIs above have loaded their native backends.
        torch_root = Path(torch.__file__).resolve().parent
        native_paths = [p['filepath'] for p in _hardware()['loaded_threadpools']
            if p['internal_api'] == 'openmp'
            and Path(p['filepath']).name.lower() == 'libiomp5md.dll'
            and not Path(p['filepath']).resolve().is_relative_to(torch_root)]
        if len(native_paths) != 1:
            raise ValueError('exactly one registered non-Torch Intel OpenMP pool required')
        self._openmp_limit = ThreadpoolController().select(filepath=native_paths).limit(
            limits=self.policy['numba_threads_when_loaded'])
        # Retain the handle; check()/verify() observe and reject later drift.
        # BLAS normalization remains limited to NumPy's registered pool. Other
        # libraries and all effective counts remain in the full fingerprint.
        paths = [p['filepath'] for p in _hardware()['loaded_threadpools']
                 if p['internal_api'] == 'openblas' and 'numpy.libs' in Path(p['filepath']).parts]
        if len(paths) != 1:
            raise ValueError('exactly one loaded NumPy OpenBLAS pool required')
        self._blas_limit = ThreadpoolController().select(filepath=paths).limit(
            limits=self.policy['numpy_openblas_threads'])
        self.duckdb_settings = self._probe_duckdb()
        self.baseline = self.check()

    def _probe_duckdb(self):
        import duckdb
        result = {}
        for role in ('input', 'eligibility'):
            requested = self.policy['duckdb_'+role+'_threads']
            if type(requested) is not int or requested < 1:
                raise ValueError('positive fixed DuckDB thread count required')
            with duckdb.connect(':memory:') as connection:
                connection.execute(f'SET threads={requested}')
                actual = connection.execute("SELECT current_setting('threads')").fetchone()[0]
                if actual != requested:
                    raise ValueError('effective DuckDB connection setting differs')
                result[role] = actual
        return result

    def check(self):
        """Read effective counts/pools, without reconfiguring or hiding drift."""
        value = _hardware()
        _check_policy(value, self.policy)
        if value['thread_environment'] != self.environment:
            raise ValueError('process thread environment changed')
        if hasattr(self, 'baseline') and canonical(value) != canonical(self.baseline):
            raise ValueError('loaded native pools or effective runtime changed')
        return value

    def identity(self):
        hardware = self.check()
        modules = {name: _file(importlib.import_module(name).__file__) for name in PACKAGES}
        return envelope(dict(schema_version=VERSION, protocol_sha256=self.protocol_sha256,
            runtime_policy_sha256=digest(self.policy), hardware=hardware,
            python_executable=_file(sys.executable), python_base_executable=_file(sys._base_executable),
            packages={name: importlib.metadata.version(name) for name in PACKAGES}, module_entry_files=modules,
            native_pool_files={p['filepath']: _file(p['filepath']) for p in hardware['loaded_threadpools']},
            duckdb_empty_connection_probes=self.duckdb_settings,
            fingerprint_scope='versions,module-entry-bytes,interpreter-bytes,loaded-native-pool-bytes; not whole installed trees',
            empirical_operations=0, final_eval_reads=0, final_eval_authorized=False))

    def verify(self, record):
        unpack(record)
        if canonical(record) != canonical(self.identity()):
            raise ValueError('actual runtime differs from the exact pre-evaluation seal')
        return self.check()
