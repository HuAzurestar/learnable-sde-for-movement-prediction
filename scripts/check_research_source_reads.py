"""Native actual raw/provider APIs; stdlib-only loading, not alternate policies.

Run on Windows and in the existing read-only, network-disabled Linux containers.
The shared observer follows both Path.open and os.open/fdopen wiring, so changing
the production opener cannot make the race or real-byte assertions vacuous.
"""

from contextlib import ExitStack
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for name in ("application", "infrastructure"):
    package = ModuleType(name)
    package.__path__ = [str(ROOT / name)]
    sys.modules[name] = package
sys.path.insert(0, str(ROOT))
STORE = importlib.import_module("infrastructure.research_store")
DATA = importlib.import_module("application.research_data")
from tests.research_file_observation import observe_file


class NativeSourceReads(unittest.TestCase):
    def exercise(self, entry, scenario):
        runtime = tempfile.TemporaryDirectory(prefix="pirc38-source-read-")
        self.addCleanup(runtime.cleanup)
        base = Path(runtime.name).resolve()
        root = base / "authorized"
        root.mkdir()
        content = b"synthetic authorized bytes\n"
        store = STORE.ResearchStore(root, "native-source", initialize=True)
        grant = {"authorization_id": "native", "study_id": "synthetic",
            "expires_at": "2099-01-01T00:00:00+00:00", "evidence_hash": STORE.digest("owner fixture"),
            "purposes": ["preview", "fit"], "visibilities": ["synthetic"], "block_ids": ["b"]}
        if entry == "raw":
            artifact = store.artifact(content, role="aggregate", visibility="synthetic", block_ids=["b"], study_id="synthetic")
            target = store.path / "artifacts" / artifact["artifact_id"]
            def read():
                return store.read_artifact(artifact["artifact_id"], purpose="preview", authorization=grant)
        else:
            target = root / "block.bin"
            target.write_bytes(content)
            protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "native", "study_id": "synthetic",
                "blocks": [{"block_id": "b", "dataset_id": "fixture", "release_id": "v1", "split_role": "train",
                    "fit_scope": True, "path": "block.bin", "sha256": hashlib.sha256(content).hexdigest()}]}
            ledger = DATA.EvaluationExposureLedger(store)
            ledger.register_protocol(protocol, STORE.digest(protocol))
            grant["protocol_hash"] = STORE.digest(protocol)
            def read():
                return ledger.read("native", "b", purpose="fit", authorization_id="native", data_root=root)
        store.authorize(grant)
        other = base / "outside.bin"
        other.write_bytes(b"x" * len(content) if scenario == "symlink" else content)
        touched, held = [], []
        original_os_open = os.open

        def replace():
            self.assertFalse(touched, "race must occur exactly at the actual selected opener")
            touched.append(True)
            if scenario == "replace":
                os.replace(other, target)
            elif scenario == "symlink":
                target.unlink()
                target.symlink_to(other)
            elif scenario == "fifo":
                target.unlink()
                os.mkfifo(target)
                # Keep the old blocking opener safe, then close the writer
                # immediately before its real read to expose EOF, not hang.
                held.append(original_os_open(target, os.O_RDWR | os.O_NONBLOCK))

        def before_read(*_):
            while held:
                os.close(held.pop())

        with ExitStack() as patches:
            observer = SimpleNamespace(setattr=lambda obj, name, value: patches.enter_context(patch.object(obj, name, value)))
            reads, handles = observe_file(observer, target,
                before_open=None if scenario == "normal" else replace, before_read=before_read)
            try:
                if scenario == "normal":
                    self.assertEqual(read(), content)
                    self.assertTrue(reads)
                    self.assertEqual(handles, [True])
                    self.assertEqual(store.events()[-1]["event_kind"], "READ_COMPLETED")
                else:
                    with self.assertRaisesRegex(STORE.ResearchError, "UNAUTHORIZED_DATA|CORRUPT_ARTIFACT"):
                        read()
                    self.assertEqual(touched, [True])
                    self.assertEqual(reads, [], "denial happened only after unauthorized/replaced bytes were consumed")
                    self.assertEqual(store.events()[-1]["event_kind"], "READ_FAILED")
            finally:
                before_read()


for entry in ("raw", "provider"):
    for scenario in ("normal", "replace", "symlink", "fifo"):
        def check(self, entry=entry, scenario=scenario):
            self.exercise(entry, scenario)
        if os.name == "nt" and scenario in {"symlink", "fifo"}:
            check = unittest.skip("POSIX-native link/FIFO; regular-file replacement remains tested on Windows")(check)
        setattr(NativeSourceReads, "test_" + entry + "_" + scenario, check)


if __name__ == "__main__":
    for module in (STORE, DATA):
        print(module.__name__ + "_sha256=" + hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), flush=True)
    unittest.main(verbosity=2)
