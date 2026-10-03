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
            data_root = root / "data"
            data_root.mkdir()
            target = data_root / "block.bin"
            target.write_bytes(content)
            protocol = {"schema_version": "pirc25-data-protocol-v1", "protocol_id": "native", "study_id": "synthetic",
                "blocks": [{"block_id": "b", "dataset_id": "fixture", "release_id": "v1", "split_role": "train",
                    "fit_scope": True, "path": "block.bin", "sha256": hashlib.sha256(content).hexdigest()}]}
            ledger = DATA.EvaluationExposureLedger(store)
            ledger.register_protocol(protocol, STORE.digest(protocol))
            grant["protocol_hash"] = STORE.digest(protocol)
            def read():
                return ledger.read("native", "b", purpose="fit", authorization_id="native", data_root=data_root)
        store.authorize(grant)
        other = base / "outside.bin"
        other.write_bytes(b"x" * len(content) if scenario == "symlink" else content)
        touched, held = [], []
        original_os_open = os.open
        source_root = target.parent
        outside_root = base / "outside"
        outside_root.mkdir()
        (outside_root / target.name).write_bytes(content)

        def redirect_root():
            self.assertFalse(touched, "root replacement must reach the real read boundary once")
            touched.append(True)
            source_root.rename(source_root.with_name(source_root.name + "-original"))
            if scenario == "root_replace":
                outside_root.rename(source_root)
            else:
                source_root.symlink_to(outside_root, target_is_directory=True)

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
            if scenario in {"root_journal", "root_authority"}:
                original_append = store._append
                def append(kind, *args, **kwargs):
                    result = original_append(kind, *args, **kwargs)
                    if kind == ("EXPOSURE_ALLOWED" if scenario == "root_authority" else "READ_STARTED"):
                        redirect_root()
                    return result
                observer.setattr(store, "_append", append)
            reads, handles = observe_file(observer, target,
                before_open=(None if scenario in {"normal", "root_journal", "root_authority"}
                    else redirect_root if scenario in {"root_open", "root_replace"} else replace),
                before_read=before_read)
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
                    # Provider path admission precedes READ_STARTED. A root
                    # redirected at EXPOSURE_ALLOWED is denied there already.
                    self.assertEqual(store.events()[-1]["event_kind"],
                        "EXPOSURE_ALLOWED" if entry == "provider" and scenario == "root_authority" else "READ_FAILED")
            finally:
                before_read()
                if source_root.is_symlink():
                    source_root.unlink()


for entry in ("raw", "provider"):
    for scenario in ("normal", "replace", "symlink", "fifo", "root_journal", "root_authority", "root_open", "root_replace"):
        def check(self, entry=entry, scenario=scenario):
            self.exercise(entry, scenario)
        if os.name == "nt" and scenario in {"symlink", "fifo", "root_journal", "root_authority", "root_open"}:
            check = unittest.skip("POSIX-native link/FIFO; regular-file replacement remains tested on Windows")(check)
        setattr(NativeSourceReads, "test_" + entry + "_" + scenario, check)


class NativeRootParents(unittest.TestCase):
    def exercise(self, scenario):
        with tempfile.TemporaryDirectory(prefix="pirc38-source-parent-") as directory:
            base = Path(directory).resolve()
            parent = base / "parent"
            root = parent / "data"
            root.mkdir(parents=True)
            target = root / "block.bin"
            target.write_bytes(b"synthetic parent fixture")
            if scenario == "normal_dotdot":
                root = root / ".." / "data"
            outside = base / "outside"
            (outside / "data").mkdir(parents=True)
            (outside / "data" / target.name).write_bytes(target.read_bytes())
            def redirect():
                parent.rename(base / "original")
                parent.symlink_to(outside, target_is_directory=True)
            try:
                with ExitStack() as patches:
                    observer = SimpleNamespace(setattr=lambda obj, name, value: patches.enter_context(patch.object(obj, name, value)))
                    reads, _ = observe_file(observer, target,
                        before_open=redirect if scenario == "open" else None)
                    if scenario == "initial":
                        redirect()
                    files = importlib.import_module("infrastructure.research_files")
                    if scenario in {"normal", "normal_dotdot"}:
                        with files.opened_regular_file(root, target) as (stream, _, _):
                            self.assertEqual(stream.read(24), b"synthetic parent fixture")
                        self.assertTrue(reads)
                    else:
                        with self.assertRaisesRegex(STORE.ResearchError, "UNAUTHORIZED_DATA|CORRUPT_ARTIFACT"):
                            with files.opened_regular_file(root, target) as (stream, _, _):
                                stream.read(24)
                        self.assertEqual(reads, [], "redirected parent bytes were consumed")
            finally:
                if parent.is_symlink():
                    parent.unlink()


for scenario in ("normal", "normal_dotdot", "initial", "open"):
    def check(self, scenario=scenario):
        self.exercise(scenario)
    if os.name == "nt" and scenario not in {"normal", "normal_dotdot"}:
        check = unittest.skip("POSIX-native parent directory symlink")(check)
    setattr(NativeRootParents, "test_parent_" + scenario, check)


if __name__ == "__main__":
    for module in (STORE, DATA):
        print(module.__name__ + "_sha256=" + hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), flush=True)
    unittest.main(verbosity=2)
