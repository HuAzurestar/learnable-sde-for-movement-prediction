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
import sys

from .research_store import ResearchError


def _identity(information):
    return (information.st_dev, information.st_ino, information.st_size,
            information.st_mtime_ns, information.st_ctime_ns)


def _source_in_root(root, path):
    if os.name == "nt" and len(root.drive) == len(path.drive) == 2:
        import nt
        # Compare the native final name to the SAME lexical root spelling in
        # that namespace. No stripping, reopening or resolving the root: the
        # stripped-name roundtrip in ntpath.realpath proves a representation
        # we never use to open/read anything. Query the complete source anew
        # at every boundary, including directory/ancestor redirects.
        final = Path(nt._getfinalpathname(str(path)))
        return final.is_relative_to(Path("\\\\?\\" + str(root)))
    if sys.platform == "linux" and getattr(os, "O_PATH", 0):
        # Query the CURRENT lexical name at every original boundary, not the
        # older payload descriptor: a parent can redirect to an external
        # hardlink while that older descriptor still has a name inside root.
        flags = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            fd = os.open(path, flags)
        except OSError:
            # Query-only opens are an optimization, not admission authority.
            # Keep the original fresh canonical check on unsupported systems.
            return path.resolve().is_relative_to(root)
        try:
            information = os.fstat(fd)
            if not (stat.S_ISREG(information.st_mode) or stat.S_ISDIR(information.st_mode)):
                return False
            try:
                final = Path(os.readlink("/proc/self/fd/" + str(fd)))
            except OSError:
                # Restricted/missing procfs must not turn into cached approval.
                return path.resolve().is_relative_to(root)
            return final.is_absolute() and final.is_relative_to(root)
        finally:
            os.close(fd)
    # Keep existing behavior for other POSIX and explicit device/UNC spellings.
    return path.resolve().is_relative_to(root)


@contextmanager
def opened_regular_file(root, path, *, expected_size=None, maximum_bytes=None, root_identity=None):
    # Normalize ordinary relative/.. spellings without following symlinks.
    root, path = Path(os.path.abspath(root)), Path(os.path.abspath(path))
    if maximum_bytes is not None and (type(maximum_bytes) is not int or maximum_bytes < 0):
        raise ResearchError("RESOURCE_PLAN_REJECTED", "invalid source byte quota")
    root_information = root.lstat()
    if not stat.S_ISDIR(root_information.st_mode):
        raise ResearchError("UNAUTHORIZED_DATA", "authorized root or its parent was redirected")
    if root_identity is not None and (root_information.st_dev, root_information.st_ino) != root_identity:
        raise ResearchError("UNAUTHORIZED_DATA", "source root differs from the original opened directory")
    before = path.lstat()
    # The complete resolved source must retain the lexical root prefix. This
    # also detects redirects of the root or ANY ancestor; separately resolving
    # the root at the same boundary adds no protection. Do not cache this check.
    if (not stat.S_ISREG(before.st_mode)
            or not _source_in_root(root, path)):
        raise ResearchError("UNAUTHORIZED_DATA", "source is not a regular file inside its authorized root")
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "source exceeds its declared byte quota")
    if expected_size is not None and before.st_size != expected_size:
        raise ResearchError("CORRUPT_ARTIFACT", "source frozen size differs before read")
    if os.name == "nt":
        from .research_windows_files import path_change_time, descriptor_change_time
        # st_ctime is creation time on supported Windows Python versions.
        # Capture native change identity BEFORE os.open, then query the held
        # descriptor at each complete boundary. Restoring mtime is not enough.
        before_change = path_change_time(path)
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
            held = os.fstat(stream.fileno())
            if (not stat.S_ISREG(current.st_mode)
                    or not stat.S_ISREG(opened.st_mode)
                    or not stat.S_ISDIR(current_root.st_mode)
                    or not _source_in_root(root, path)
                    or (current_root.st_dev, current_root.st_ino) != (root_information.st_dev, root_information.st_ino)):
                raise ResearchError("UNAUTHORIZED_DATA", "source or authorized root changed at opened handle")
            # Preserve package quota classification at EVERY actual metadata
            # boundary. Late growth above an admitted quota is a quota error,
            # before identity/parse checks and before any new payload bytes.
            if maximum_bytes is not None and any(info.st_size > maximum_bytes for info in (opened, current, held)):
                raise ResearchError("RESOURCE_PLAN_REJECTED", "source exceeds its declared byte quota")
            if (_identity(before) != _identity(opened) or _identity(current) != _identity(opened)
                    or _identity(held) != _identity(opened)):
                raise ResearchError("CORRUPT_ARTIFACT", "source identity changed during read")
            if os.name == "nt" and descriptor_change_time(stream.fileno()) != before_change:
                raise ResearchError("CORRUPT_ARTIFACT", "native source change identity differs during read")
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
