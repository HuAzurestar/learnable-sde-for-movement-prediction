"""Shared bookkeeping/render workers must not eagerly load numerical engines."""

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [
    "infrastructure.research_store", "application.research_supervisor",
    "application.research_comparison", "experiments.pirc25.compare_worker",
    "experiments.pirc25.case_worker", "experiments.pirc25.__main__",
    "application.pirc26_runtime", "infrastructure.pirc26_worker",
    "application.pirc26_metrics",
    "infrastructure.pirc26_worker_control",
    "infrastructure.pirc26_checkpoint_contract",
    "infrastructure.pirc26_process_resources",
])
def test_actual_shared_entry_import_does_not_require_tensor_engines(module):
    # An independent interpreter proves that pytest's already-imported torch
    # cannot hide an eager dependency. Block the real import path, not a dummy
    # engine or an alternative implementation of the worker.
    script = """
import importlib
import importlib.abc
import sys
class NoNumericalEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy'}:
            raise AssertionError('unneeded numerical engine: ' + fullname)
sys.meta_path.insert(0, NoNumericalEngine())
importlib.import_module(sys.argv[1])
assert not {'torch', 'numpy', 'scipy'}.intersection(sys.modules)
print('shared-import-ok')
"""
    process = subprocess.run([sys.executable, "-B", "-c", script, module], cwd=ROOT,
                             capture_output=True, text=True, timeout=40)
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == "shared-import-ok"


@pytest.mark.parametrize("package, exports", [
    ("application", {"EvaluationPipeline": "pipelines", "EvidenceConditioner": "pipelines",
                     "RandomStreams": "runtime", "RunContext": "runtime"}),
    ("infrastructure", {"ArtifactWriter": "runs", "AtomicRunStore": "runs",
                        "JsonArtifactStore": "artifacts", "TorchModelStore": "checkpoint"}),
])
def test_existing_package_exports_keep_real_identity_and_star_import(package, exports):
    script = """
import importlib
import json
import sys
name, exports = sys.argv[1], json.loads(sys.argv[2])
package = importlib.import_module(name)
assert set(package.__all__) == set(exports)
namespace = {}
exec('from ' + name + ' import *', namespace)
for symbol, module in exports.items():
    original = getattr(importlib.import_module(name + '.' + module), symbol)
    assert getattr(package, symbol) is original
    assert namespace[symbol] is original
    assert symbol in dir(package)
try:
    package.not_a_public_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown package attribute accepted')
print('legacy-exports-ok')
"""
    import json
    process = subprocess.run([sys.executable, "-B", "-c", script, package, json.dumps(exports)],
                             cwd=ROOT, capture_output=True, text=True, timeout=40)
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == "legacy-exports-ok"
