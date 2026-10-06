"""Native stdlib-only checks of the actual production publication primitives.

Run on Windows or a Linux Python image with this checkout mounted read-only.
The actual production package keeps these imports stdlib-only. Its shared
opened-handle guard must retain the package context and the same error class;
this proves file operations, not the complete CLI or scientific wiring.
"""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'infrastructure' / 'research_store.py'
sys.path.insert(0, str(SOURCE.parents[1]))
from infrastructure import research_store as STORE


class NativePublication(unittest.TestCase):
    def setUp(self):
        self.runtime = tempfile.TemporaryDirectory(prefix='pirc38-publication-')
        self.addCleanup(self.runtime.cleanup)
        self.target = Path(self.runtime.name) / 'export.json'
        self.content = b'complete frozen evidence'

    def assert_clean(self):
        self.assertEqual(list(self.target.parent.glob('.*.staging')), [])

    def test_no_clobber_and_default_store_replace(self):
        STORE.atomic_write(self.target, self.content, immutable=True)
        with self.assertRaisesRegex(STORE.ResearchError, 'IDENTITY_CONFLICT'):
            STORE.atomic_write(self.target, b'different immutable version', immutable=True)
        self.assertEqual(self.target.read_bytes(), self.content)
        self.assert_clean()
        STORE.atomic_write(self.target, b'ordinary locked store replacement')
        self.assertEqual(self.target.read_bytes(), b'ordinary locked store replacement')
        self.assert_clean()

    def test_equal_retry_keeps_inode_and_guards_after_read(self):
        STORE.atomic_write(self.target, self.content, immutable=True)
        original = self.target.stat()
        guards = []

        def guard():
            guards.append(True)
            self.assertEqual(self.target.read_bytes(), self.content)

        STORE.atomic_write(self.target, self.content, immutable=True, before_replace=guard)
        final = self.target.stat()
        self.assertEqual((original.st_dev, original.st_ino, original.st_mtime_ns),
                         (final.st_dev, final.st_ino, final.st_mtime_ns))
        self.assertEqual(guards, [True, True])
        self.assert_clean()

    def test_growth_after_opened_handle_stat_is_rejected(self):
        self.target.write_bytes(self.content)
        original = STORE.os.fstat
        touched = []

        def growing(fd):
            info = original(fd)
            if not touched:
                with self.target.open('ab') as stream:
                    stream.write(b'grew after actual opened-handle stat')
                touched.append(True)
            return info

        with patch.object(STORE.os, 'fstat', growing):
            self.assertFalse(STORE._matches_file_content(self.target, self.content))
        self.assertTrue(touched)
        self.assertTrue(self.target.read_bytes().startswith(self.content))

    def test_unsupported_publication_fails_without_replace_fallback(self):
        with patch.object(STORE, '_publish_no_replace', side_effect=OSError('unsupported operation')):
            with self.assertRaises(OSError):
                STORE.atomic_write(self.target, self.content, immutable=True)
        self.assertFalse(self.target.exists())
        self.assert_clean()

    def test_exclusive_staging_collision_keeps_foreign_staging(self):
        self.target.write_bytes(b'original target')
        foreign = self.target.with_name('.export.json.fixed.staging')
        foreign.write_bytes(b'foreign staging')
        for immutable in (False, True):
            with self.subTest(immutable=immutable), patch.object(
                    STORE.uuid, 'uuid4', return_value=SimpleNamespace(hex='fixed')):
                with self.assertRaises(FileExistsError):
                    STORE.atomic_write(self.target, self.content, immutable=immutable)
            self.assertEqual(self.target.read_bytes(), b'original target')
            self.assertEqual(foreign.read_bytes(), b'foreign staging')

    def test_moved_staging_name_can_be_reused_without_first_writer_cleanup(self):
        # Linux link does not free the staging name. Default replace moves on
        # both platforms; Windows immutable rename also moves.
        original_sync = STORE._sync_directory
        modes = (False, True) if os.name == 'nt' else (False,)
        for immutable in modes:
            suffix = str(immutable)
            target = self.target.with_name('export-' + suffix + '.json')
            staging = target.with_name('.' + target.name + '.' + suffix + '.staging')

            def sync(path):
                original_sync(path)
                self.assertFalse(staging.exists())
                with staging.open('xb') as stream:
                    stream.write(b'new owner of the freed staging name')

            with self.subTest(immutable=immutable), patch.object(
                    STORE.uuid, 'uuid4', return_value=SimpleNamespace(hex=suffix)), patch.object(
                    STORE, '_sync_directory', sync):
                STORE.atomic_write(target, self.content, immutable=immutable)
            self.assertEqual(staging.read_bytes(), b'new owner of the freed staging name')
            self.assertEqual(target.read_bytes(), self.content)

    @unittest.skipIf(os.name == 'nt', 'native POSIX FIFO race')
    def test_regular_file_swapped_for_fifo_cannot_block_comparison(self):
        self.target.write_bytes(self.content)
        original = STORE.os.open
        touched = []

        def replaced(path, flags, *args, **kwargs):
            if path == self.target and not touched:
                self.target.unlink()  # Only this owned synthetic test file.
                os.mkfifo(self.target)
                touched.append(True)
            return original(path, flags, *args, **kwargs)

        with patch.object(STORE.os, 'open', replaced):
            self.assertFalse(STORE._matches_file_content(self.target, self.content))
        self.assertTrue(touched)

    @unittest.skipIf(os.name == 'nt', 'native POSIX symlink collision')
    def test_symlink_collision_does_not_read_or_replace_destination(self):
        other = self.target.with_name('other.json')
        other.write_bytes(self.content)
        self.target.symlink_to(other)
        with self.assertRaisesRegex(STORE.ResearchError, 'IDENTITY_CONFLICT'):
            STORE.atomic_write(self.target, self.content, immutable=True)
        self.assertTrue(self.target.is_symlink())
        self.assertEqual(other.read_bytes(), self.content)
        self.assert_clean()


if __name__ == '__main__':
    print('Production SHA256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest(), flush=True)
    print('Native platform:', os.name, flush=True)
    assert not any(name.split('.')[0] in {'torch', 'numpy', 'pandas', 'pyarrow', 'duckdb'}
                   for name in sys.modules), 'publication import unexpectedly loaded scientific dependencies'
    unittest.main(verbosity=2)
