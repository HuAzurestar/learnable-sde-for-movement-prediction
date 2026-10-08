"""Metadata/software tests: no research inputs, fits or predictions."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.pirc17 import formal_environment as module
from experiments.pirc17.formal_matrix import load_protocol
from experiments.pirc17.protocol_core import canonical, envelope, file_hash, read_json, unpack

ROOT = Path(__file__).resolve().parents[1]


def child(*args):
    environment = dict(os.environ, PYTHONPATH=str(ROOT.parent/'DSDE-SDE'), PYTHONIOENCODING='utf-8')
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=environment,
        capture_output=True, text=True, encoding='utf-8', timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


@pytest.fixture(scope='module')
def actual_environment(tmp_path_factory):
    directory = tmp_path_factory.mktemp('metadata-runtime')
    result = child('-m', 'experiments.pirc17.formal_entrypoint', 'environment',
                   '--output-directory', str(directory))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    record = read_json(summary['path'])
    assert record['sha256'] == summary['environment_sha256']
    return record


def test_fresh_cli_pins_actual_effective_counts_and_file_bytes(actual_environment):
    e = unpack(actual_environment)
    policy = unpack(load_protocol())['resource_contract']['runtime_binding']
    module._check_policy(e['hardware'], policy)
    assert e['duckdb_empty_connection_probes'] == {'input': 2, 'eligibility': 4}
    assert e['empirical_operations'] == e['final_eval_reads'] == 0
    assert e['final_eval_authorized'] is False
    for f in [e['python_executable'], e['python_base_executable'],
              *e['module_entry_files'].values(), *e['native_pool_files'].values()]:
        assert Path(f['path']).stat().st_size == f['bytes']
        assert file_hash(f['path']) == f['file_sha256']
    assert e['hardware']['torch_threads'] == 6
    assert e['hardware']['torch_interop_threads'] == 12
    assert e['hardware']['numba_threads'] == 12


def test_separate_fresh_process_reproduces_environment_identity(actual_environment, tmp_path):
    result = child('-m', 'experiments.pirc17.formal_entrypoint', 'environment',
                   '--output-directory', str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['environment_sha256'] == actual_environment['sha256']


@pytest.mark.parametrize('preload', [False, True], ids=['fresh', 'preloaded'])
def test_actual_parent_and_inherited_child_have_identical_native_binding(tmp_path, preload):
    # A controller can load numerics before configuring its runtime. Its fresh
    # worker then inherits OMP/MKL/etc BEFORE those libraries are imported.
    # Independent fresh CLI invocations do not exercise this asymmetric order.
    code = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
if sys.argv[2] == 'True':
    import numpy
    import torch
from experiments.pirc17.formal_matrix import load_protocol
from experiments.pirc17.formal_environment import ConfiguredRuntime
from experiments.pirc17.protocol_core import canonical, unpack
runtime = ConfiguredRuntime(load_protocol())
record = runtime.identity()
seal = Path(sys.argv[1]); seal.write_bytes(canonical(record))
worker_code = """
import json
import os
import sys
inherited = {name: os.environ.get(name) for name in
    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')}
from experiments.pirc17.formal_matrix import load_protocol
from experiments.pirc17.formal_environment import ConfiguredRuntime
from experiments.pirc17.protocol_core import read_json, unpack
runtime = ConfiguredRuntime(load_protocol())
record = read_json(sys.argv[1])
runtime.verify(record)
assert inherited == unpack(record)['hardware']['thread_environment']
print(json.dumps(dict(pid=os.getpid(), environment_sha256=runtime.identity()['sha256'],
    inherited=inherited, hardware=runtime.check())))
"""
worker = subprocess.run([sys.executable, '-u', '-c', worker_code, str(seal)],
    capture_output=True, text=True, encoding='utf-8', timeout=45,
    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
assert worker.returncode == 0, worker.stderr
value = json.loads(worker.stdout)
assert value['pid'] != os.getpid()
assert value['environment_sha256'] == record['sha256']
assert value['hardware'] == unpack(record)['hardware']
runtime.verify(record)  # Child construction must not change parent state.
print(json.dumps(dict(parent_pid=os.getpid(), child=value, preload=sys.argv[2])))
'''
    result = child('-u', '-c', code, str(tmp_path/'parent-seal.json'), str(preload))
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['parent_pid'] != value['child']['pid']
    assert value['child']['hardware']['torch_threads'] == 6
    assert value['child']['hardware']['numba_threads'] == 12
    assert value['child']['inherited']['OMP_NUM_THREADS'] == '6'
    assert value['child']['hardware']['loaded_threadpools']
    for pool in value['child']['hardware']['loaded_threadpools']:
        if pool['internal_api'] == 'openmp':
            path = Path(pool['filepath'])
            if path.name.lower() == 'libiomp5md.dll':
                assert pool['num_threads'] == (6 if 'torch' in path.parts else 12)
            elif path.name.lower() == 'libiompstubs5md.dll':
                assert pool['num_threads'] == 1
            elif path.name.lower() == 'vcomp140.dll':
                assert pool['num_threads'] == 6


def test_actual_non_torch_native_thread_drift_is_rejected_not_normalized(tmp_path):
    code = r'''
from pathlib import Path
from threadpoolctl import ThreadpoolController
from experiments.pirc17.formal_matrix import load_protocol
from experiments.pirc17.formal_environment import ConfiguredRuntime
runtime = ConfiguredRuntime(load_protocol())
pools = [p for p in runtime.check()['loaded_threadpools']
    if p['internal_api'] == 'openmp' and Path(p['filepath']).name.lower() == 'libiomp5md.dll'
    and 'torch' not in Path(p['filepath']).parts]
assert len(pools) == 1, 'Exact registered non-Torch Intel OpenMP pool required'
for pool in pools:
    assert pool['num_threads'] == 12
    with ThreadpoolController().select(filepath=pool['filepath']).limit(limits=6):
        try: runtime.check()
        except ValueError: pass
        else: raise AssertionError('native thread drift was silently accepted/repaired')
    runtime.check()
print('actual non-Torch native drift rejected')
'''
    result = child('-u', '-c', code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'actual non-Torch native drift rejected'


@pytest.mark.parametrize('field,value', [('torch_threads', 7), ('torch_interop_threads', 6),
    ('numba_threads', 1), ('device', 'cuda'), ('numpy', '0.0'), ('logical_cpus', 24)])
def test_effective_mismatch_is_rejected_not_repaired(actual_environment, field, value):
    hardware = deepcopy(unpack(actual_environment)['hardware'])
    hardware[field] = value
    with pytest.raises(ValueError):
        module._check_policy(hardware, unpack(load_protocol())['resource_contract']['runtime_binding'])


def test_real_thread_drift_and_rehashed_seal_are_rejected_in_process(actual_environment, tmp_path):
    seal = tmp_path/'seal.json'; seal.write_bytes(canonical(actual_environment))
    code = """
import sys
import torch
from experiments.pirc17.formal_matrix import load_protocol
from experiments.pirc17.formal_environment import ConfiguredRuntime
from experiments.pirc17.protocol_core import read_json, envelope, unpack
r = ConfiguredRuntime(load_protocol())
record = read_json(sys.argv[1]); r.verify(record)
p = unpack(record); p['packages']['numpy'] = 'tampered-but-rehashed'
try: r.verify(envelope(p))
except ValueError: pass
else: raise AssertionError('rehashed runtime seal accepted')
torch.set_num_threads(7)
try: r.check()
except ValueError: pass
else: raise AssertionError('actual effective thread drift accepted')
print('metadata-only drift checks passed')
"""
    result = child('-c', code, str(seal))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'metadata-only drift checks passed'


def test_environment_not_just_envvars_and_new_native_pool_rejected(actual_environment, monkeypatch):
    record = unpack(actual_environment)
    runtime = object.__new__(module.ConfiguredRuntime)
    runtime.policy = unpack(load_protocol())['resource_contract']['runtime_binding']
    runtime.environment = record['hardware']['thread_environment']
    runtime.baseline = deepcopy(record['hardware'])
    hardware = deepcopy(runtime.baseline)
    hardware['loaded_threadpools'].append(dict(filepath='unexpected-library', internal_api='other', num_threads=8))
    monkeypatch.setattr(module, '_hardware', lambda: hardware)
    with pytest.raises(ValueError, match='loaded native pools'):
        runtime.check()
    hardware = deepcopy(runtime.baseline)
    hardware['thread_environment']['OMP_NUM_THREADS'] = '100'
    with pytest.raises(ValueError, match='environment changed'):
        runtime.check()


def test_actual_blas_count_is_required_even_when_environment_claims_correct(actual_environment):
    hardware = deepcopy(unpack(actual_environment)['hardware'])
    for pool in hardware['loaded_threadpools']:
        if pool['internal_api'] == 'openblas': pool['num_threads'] = 1
    with pytest.raises(ValueError, match='OpenBLAS'):
        module._check_policy(hardware, unpack(load_protocol())['resource_contract']['runtime_binding'])
