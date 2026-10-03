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
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for name in ("application", "infrastructure", "experiments", "experiments.pirc25"):
    package = ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
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
                # Moving the original directory preserves both file and root
                # inode. Namespace verification must still deny its redirect.
                parent.symlink_to(base / "original" if scenario == "open_original" else outside,
                    target_is_directory=True)
            try:
                with ExitStack() as patches:
                    observer = SimpleNamespace(setattr=lambda obj, name, value: patches.enter_context(patch.object(obj, name, value)))
                    reads, _ = observe_file(observer, target,
                        before_open=redirect if scenario in {"open", "open_original"} else None)
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


for scenario in ("normal", "normal_dotdot", "initial", "open", "open_original"):
    def check(self, scenario=scenario):
        self.exercise(scenario)
    if os.name == "nt" and scenario not in {"normal", "normal_dotdot"}:
        check = unittest.skip("POSIX-native parent directory symlink")(check)
    setattr(NativeRootParents, "test_parent_" + scenario, check)


class NativeMetadataReads(unittest.TestCase):
    def exercise(self, entry, scenario):
        with tempfile.TemporaryDirectory(prefix="pirc38-metadata-read-") as directory:
            base = Path(directory).resolve()
            store = STORE.ResearchStore(base, "metadata-read", initialize=True)
            expected = {"scope": "synthetic owner metadata"}
            store.publish("synthetic-metadata", expected)
            target = (store.path / "events" / "0000000000000001.json" if entry == "event"
                else store.path / "manifests" / "synthetic-metadata.json")
            outside = base / "outside"
            outside.mkdir()
            external = outside / target.name
            external.write_bytes(target.read_bytes())
            touched = []
            def change():
                self.assertFalse(touched)
                touched.append(True)
                if scenario == "replace":
                    os.replace(external, target)
                elif scenario == "symlink":
                    target.unlink()
                    target.symlink_to(external)
                else:
                    target.parent.rename(base / "original-metadata")
                    target.parent.symlink_to(outside, target_is_directory=True)
            try:
                if scenario == "root_initial":
                    change()
                with ExitStack() as patches:
                    observer = SimpleNamespace(setattr=lambda obj, name, value: patches.enter_context(patch.object(obj, name, value)))
                    reads, _ = observe_file(observer, target,
                        before_open=change if scenario in {"replace", "symlink", "root_open"} else None)
                    if scenario == "normal":
                        self.assertEqual(store.manifest("synthetic-metadata"), expected)
                        self.assertTrue(reads, "metadata check did not reach a real read")
                    else:
                        with self.assertRaisesRegex(STORE.ResearchError, "UNAUTHORIZED_DATA|CORRUPT_ARTIFACT"):
                            store.manifest("synthetic-metadata")
                        self.assertEqual(touched, [True])
                        self.assertEqual(reads, [], "redirected or replaced metadata bytes were consumed")
            finally:
                if target.is_symlink():
                    target.unlink()
                if target.parent.is_symlink():
                    target.parent.unlink()


for entry in ("manifest", "event"):
    for scenario in ("normal", "replace", "symlink", "root_initial", "root_open"):
        def check(self, entry=entry, scenario=scenario):
            self.exercise(entry, scenario)
        if os.name == "nt" and scenario in {"symlink", "root_initial", "root_open"}:
            check = unittest.skip("POSIX-native metadata file/directory symlink")(check)
        setattr(NativeMetadataReads, "test_" + entry + "_" + scenario, check)


class NativeCodeSourceReads(unittest.TestCase):
    def exercise(self, entry, scenario):
        with tempfile.TemporaryDirectory(prefix="pirc38-code-read-") as directory:
            base = Path(directory).resolve()
            root = base / "source"
            root.mkdir()
            if entry == "runtime":
                module = importlib.import_module("experiments.pirc25.affine")
                for name in ("config.py", "numerics.py", "registry.py"):
                    (root / name).write_bytes(b"# synthetic source\n")
                target = root / "config.py"
                reader = module.code_hash
            else:
                module = importlib.import_module("application.research_computation")
                target = root / "scripts/pirc25/__init__.py"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"# synthetic source\n")
                for args in (("init", "--quiet"), ("add", "scripts"),
                        ("commit", "--quiet", "-m", "synthetic source")):
                    subprocess.run(["git", "-C", str(root), "-c", "user.name=PIRC-fixture",
                        "-c", "user.email=fixture@example.invalid", *args], check=True,
                        capture_output=True, timeout=10)
                reader = lambda: module.paper_identity(root)
            outside = base / "outside"
            outside.mkdir()
            external = outside / target.relative_to(root)
            external.parent.mkdir(parents=True, exist_ok=True)
            external.write_bytes(target.read_bytes())
            if scenario.startswith("root_"):
                shutil.copytree(root, outside, dirs_exist_ok=True)
            touched = []
            def change():
                self.assertFalse(touched)
                touched.append(True)
                if scenario == "replace":
                    os.replace(external, target)
                elif scenario == "symlink":
                    target.unlink()
                    target.symlink_to(external)
                else:
                    root.rename(base / "original")
                    root.symlink_to(outside, target_is_directory=True)
            try:
                if scenario == "root_initial":
                    change()
                with ExitStack() as patches:
                    observer = SimpleNamespace(setattr=lambda obj, name, value:
                        patches.enter_context(patch.object(obj, name, value)))
                    if entry == "runtime":
                        observer.setattr(module, "ROOT", root)
                    observed = external if entry == "paper" and scenario == "root_initial" else target
                    reads, _ = observe_file(observer, observed,
                        before_open=change if scenario not in {"normal", "root_initial"} else None)
                    if scenario == "normal":
                        self.assertTrue(reader())
                        self.assertTrue(reads, "source check did not reach the actual reader")
                    else:
                        with self.assertRaisesRegex(STORE.ResearchError, "UNAUTHORIZED_DATA|CORRUPT_ARTIFACT"):
                            reader()
                        self.assertEqual(touched, [True])
                        self.assertEqual(reads, [], "identity consumed replaced/outside code bytes")
            finally:
                if root.is_symlink():
                    root.unlink()


for entry in ("runtime", "paper"):
    for scenario in ("normal", "replace", "symlink", "root_initial", "root_open"):
        def check(self, entry=entry, scenario=scenario):
            self.exercise(entry, scenario)
        if entry == "paper" and not shutil.which("git"):
            check = unittest.skip("actual paper identity needs native Git; use the Git validation image")(check)
        elif os.name == "nt" and scenario not in {"normal", "replace"}:
            check = unittest.skip("POSIX-native code source/root link")(check)
        setattr(NativeCodeSourceReads, "test_" + entry + "_" + scenario, check)


class NativeSourceEnumeration(unittest.TestCase):
    def exercise(self, link):
        with tempfile.TemporaryDirectory(prefix="pirc38-source-enumeration-") as directory:
            base = Path(directory).resolve()
            root, outside = base / "runtime", base / "outside"
            root.mkdir()
            outside.mkdir()
            for name in ("config.py", "numerics.py", "registry.py"):
                (root / name).write_bytes(b"# synthetic source\n")
            (root / "application").mkdir()
            (root / "application/__init__.py").write_bytes(b"")
            alias = root / "application/alias"
            if link:
                alias.symlink_to(outside, target_is_directory=True)
                source = outside / "code.py"
            else:
                alias.mkdir()
                source = alias / "code.py"
            source.write_bytes(b"VALUE = 111\n")
            affine = importlib.import_module("experiments.pirc25.affine")
            with patch.object(affine, "ROOT", root):
                if link:
                    # Python actually follows this source directory even though
                    # rglob('*.py') silently skips it. No alternate policy here.
                    source.write_bytes(b"VALUE = 222\n")
                    actual = subprocess.run([sys.executable, "-B", "-c",
                        "import sys; sys.path.insert(0,sys.argv[1]); from application.alias.code import VALUE; print(VALUE)",
                        str(root)], check=True, capture_output=True, text=True, timeout=10)
                    self.assertEqual(actual.stdout.strip(), "222")
                    with self.assertRaisesRegex(STORE.ResearchError, "UNAUTHORIZED_DATA"):
                        affine.code_hash()
                else:
                    before = affine.code_hash()
                    source.write_bytes(b"VALUE = 222\n")
                    self.assertNotEqual(affine.code_hash(), before)
                    self.assertEqual(affine.code_hash(), STORE.digest({
                        path.relative_to(root).as_posix(): hashlib.sha256(
                            path.read_text(encoding="utf-8").encode()).hexdigest()
                        for path in root.rglob("*.py")}))

    def test_normal_nested_sources_retain_legacy_hashes(self):
        self.exercise(False)

    @unittest.skipIf(os.name == "nt", "POSIX-native directory source alias")
    def test_importable_directory_alias_cannot_be_omitted_from_code_identity(self):
        self.exercise(True)


if __name__ == "__main__":
    sources = (STORE, DATA, importlib.import_module("infrastructure.research_files"),
        importlib.import_module("application.research_computation"),
        importlib.import_module("experiments.pirc25.affine"))
    for module in sources:
        print(module.__name__ + "_sha256=" + hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), flush=True)
    unittest.main(verbosity=2)
