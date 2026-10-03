"""Opened regular-file boundaries shared by authorized and owner reads.

This is not a grant cache: callers keep their own current authority and journals.
File identity is checked before any bytes and after the complete operation.
"""

from contextlib import contextmanager
import codecs
import errno
import hashlib
import os
from pathlib import Path
import stat

from .research_store import ResearchError


def _identity(information):
    return (information.st_dev, information.st_ino, information.st_size,
            information.st_mtime_ns, information.st_ctime_ns)


@contextmanager
def opened_regular_file(root, path, *, expected_size=None):
    # Normalize ordinary relative/.. spellings without following symlinks.
    root, path = Path(os.path.abspath(root)), Path(os.path.abspath(path))
    root_information = root.lstat()
    if not stat.S_ISDIR(root_information.st_mode) or root.resolve() != root:
        raise ResearchError("UNAUTHORIZED_DATA", "authorized root or its parent was redirected")
    before = path.lstat()
    if (not stat.S_ISDIR(root_information.st_mode) or not stat.S_ISREG(before.st_mode)
            or not path.resolve().is_relative_to(root)):
        raise ResearchError("UNAUTHORIZED_DATA", "source is not a regular file inside its authorized root")
    if expected_size is not None and before.st_size != expected_size:
        raise ResearchError("CORRUPT_ARTIFACT", "source frozen size differs before read")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ResearchError("UNAUTHORIZED_DATA", "source became a symbolic link") from exc
        raise
    try:
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with stream:
        opened = os.fstat(stream.fileno())

        def verify_identity():
            current, current_root = path.lstat(), root.lstat()
            if (not stat.S_ISREG(current.st_mode)
                    or not stat.S_ISREG(opened.st_mode)
                    or not stat.S_ISDIR(current_root.st_mode)
                    or root.resolve() != root
                    or not path.resolve().is_relative_to(root)
                    or (current_root.st_dev, current_root.st_ino) != (root_information.st_dev, root_information.st_ino)):
                raise ResearchError("UNAUTHORIZED_DATA", "source or authorized root changed at opened handle")
            if (_identity(before) != _identity(opened) or _identity(current) != _identity(opened)
                    or _identity(os.fstat(stream.fileno())) != _identity(opened)):
                raise ResearchError("CORRUPT_ARTIFACT", "source identity changed during read")
            if expected_size is not None and opened.st_size != expected_size:
                raise ResearchError("CORRUPT_ARTIFACT", "source frozen size differs at opened handle")

        verify_identity()
        yield stream, opened.st_size, verify_identity
        verify_identity()


def source_file_hash(root, path, *, maximum_bytes=None):
    """Stream the historical UTF-8/universal-newline source identity.

    A caller-specific admission quota is optional; runtime sources have no
    blanket paper-worker cap. Neither decoding nor hashing buffers a file.
    """
    with opened_regular_file(root, path) as (stream, size, _):
        if maximum_bytes is not None and size > maximum_bytes:
            raise ResearchError("UNAUTHORIZED_DATA", "source exceeds its declared quota")
        decoder = codecs.getincrementaldecoder("utf-8")()
        hasher, pending, consumed = hashlib.sha256(), "", 0
        while True:
            chunk = stream.read(min(64 * 1024, size - consumed + 1))
            consumed += len(chunk)
            if consumed > size:
                raise ResearchError("CORRUPT_ARTIFACT", "source grew during hash read")
            text = pending + decoder.decode(chunk, final=not chunk)
            pending = ""
            if chunk and text.endswith("\r"):
                text, pending = text[:-1], "\r"
            hasher.update(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
            if not chunk:
                break
        if consumed != size:
            raise ResearchError("CORRUPT_ARTIFACT", "source was truncated during hash read")
    return hasher.hexdigest()
