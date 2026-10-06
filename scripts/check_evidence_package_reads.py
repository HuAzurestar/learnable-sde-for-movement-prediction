"""Native checks of the actual incoming-package file primitive, not the CLI.

Namespace loading avoids unrelated ML composition-root imports in slim images.
The real production evidence/store modules and dependencies are imported intact;
no file-read or validation implementation is substituted. Full API tests remain
separate and run in the normal application environment.
"""

import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for name in ("application", "infrastructure"):
    package = ModuleType(name)
    package.__path__ = [str(ROOT / name)]
    sys.modules[name] = package
sys.path.insert(0, str(ROOT))
EVIDENCE = importlib.import_module("application.research_evidence")


class NativePackageReads(unittest.TestCase):
    def setUp(self):
        runtime = tempfile.TemporaryDirectory(prefix="pirc38-package-read-")
        self.addCleanup(runtime.cleanup)
        self.root = Path(runtime.name).resolve()
        self.target = self.root / "incoming.json"
        self.content = b'{"frozen":true}'
        self.target.write_bytes(self.content)

    def read(self, limit=None):
        return EVIDENCE._package_file_bytes(self.root, self.target.name,
            len(self.content) if limit is None else limit)

    def test_exact_limit_and_empty_regular_file(self):
        self.assertEqual(self.read(), self.content)
        self.target.write_bytes(b"")
        self.assertEqual(self.read(0), b"")

    def test_oversize_rejected_before_open(self):
        with patch.object(EVIDENCE.os, "open", side_effect=AssertionError("oversize opened")):
            with self.assertRaisesRegex(EVIDENCE.ResearchError, "RESOURCE_PLAN_REJECTED"):
                self.read(len(self.content) - 1)

    def test_growth_after_opened_handle_stat(self):
        original = EVIDENCE.os.fstat
        touched = []

        def growing(fd):
            information = original(fd)
            if not touched:
                with self.target.open("ab") as writer:
                    writer.write(b"growth")
                touched.append(True)
            return information

        with patch.object(EVIDENCE.os, "fstat", growing):
            with self.assertRaisesRegex(EVIDENCE.ResearchError, "RESOURCE_PLAN_REJECTED"):
                self.read()
        self.assertEqual(touched, [True])

    def test_same_size_replacement_at_actual_open(self):
        other = self.root / "replacement.json"
        other.write_bytes(self.content)
        original = EVIDENCE.os.open

        def replacing(path, flags):
            os.replace(other, self.target)
            return original(path, flags)

        with patch.object(EVIDENCE.os, "open", replacing):
            with self.assertRaisesRegex(EVIDENCE.ResearchError, "CORRUPT_ARTIFACT"):
                self.read()
        self.assertFalse(other.exists())

    def test_native_open_handle_replacement_boundary(self):
        other = self.root / "replacement.json"
        other.write_bytes(self.content)
        original = EVIDENCE.os.fdopen

        class Stream:
            def __init__(self, actual):
                self.actual = actual

            def __enter__(self):
                self.actual.__enter__()
                return self

            def __exit__(self, *args):
                return self.actual.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.actual, name)

            def read(self, size):
                content = self.actual.read(size)
                os.replace(other, self_target)
                return content

        self_target = self.target
        with patch.object(EVIDENCE.os, "fdopen", lambda *args, **kwargs: Stream(original(*args, **kwargs))):
            if os.name == "nt":
                with self.assertRaises(PermissionError):
                    self.read()
                self.assertTrue(other.exists())
            else:
                with self.assertRaisesRegex(EVIDENCE.ResearchError, "CORRUPT_ARTIFACT"):
                    self.read()
                self.assertFalse(other.exists())

    def test_nonregular_directory_is_rejected_before_open(self):
        self.target.unlink()
        self.target.mkdir()
        with patch.object(EVIDENCE.os, "open", side_effect=AssertionError("directory opened")):
            with self.assertRaisesRegex(EVIDENCE.ResearchError, "UNAUTHORIZED_DATA"):
                self.read()

    @unittest.skipIf(os.name == "nt", "POSIX-native symlink; Windows refuses open-handle replacement separately")
    def test_native_symlink_is_rejected_before_open(self):
        other = self.root / "other.json"
        other.write_bytes(self.content)
        self.target.unlink()
        self.target.symlink_to(other)
        with patch.object(EVIDENCE.os, "open", side_effect=AssertionError("symlink opened")):
            with self.assertRaisesRegex(EVIDENCE.ResearchError, "UNAUTHORIZED_DATA"):
                self.read()

    @unittest.skipIf(os.name == "nt", "POSIX-native FIFO; Windows regular-file cases remain applicable")
    def test_native_fifo_is_rejected_without_blocking_open(self):
        self.target.unlink()
        os.mkfifo(self.target)
        with patch.object(EVIDENCE.os, "open", side_effect=AssertionError("FIFO opened")):
            with self.assertRaisesRegex(EVIDENCE.ResearchError, "UNAUTHORIZED_DATA"):
                self.read()


if __name__ == "__main__":
    print("production_evidence_sha256=" + hashlib.sha256(Path(EVIDENCE.__file__).read_bytes()).hexdigest(), flush=True)
    unittest.main(verbosity=2)
