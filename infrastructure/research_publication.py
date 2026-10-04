"""Lexical publication admission and original directory/staging ownership.

POSIX operations are directory-relative; Linux immutable publication links
the original descriptor rather than whatever now occupies its staging name.
Windows publication and cleanup operate on verified original source handles.
Authority remains the caller's fresh callback, not a cached permission here.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import stat

from .research_store import ResearchError


def _object(info):
    return info.st_dev, info.st_ino


def lexical_directory(path):
    from .research_files import _source_in_root
    path = Path(os.path.abspath(path))
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or not _source_in_root(path, path):
        raise ResearchError('UNAUTHORIZED_DATA', 'publication directory or ancestor was redirected')
    return path, info


@contextmanager
def opened_directory(path):
    path, original = lexical_directory(path)
    if os.name == 'nt':
        from .research_windows_publication import directory_descriptor
        fd = directory_descriptor(path)
    else:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        def verify():
            try:
                _, current = lexical_directory(path)
            except FileNotFoundError as error:
                raise ResearchError('UNAUTHORIZED_DATA', 'publication directory disappeared') from error
            held = os.fstat(fd)
            if not stat.S_ISDIR(held.st_mode) or _object(current) != _object(original) or _object(held) != _object(original):
                raise ResearchError('UNAUTHORIZED_DATA', 'publication directory changed at held handle')
        verify()
        yield path, fd, verify
        verify()
    finally:
        os.close(fd)


def prepare_directory(path):
    """Create missing components relative to admitted existing parents."""
    path = Path(os.path.abspath(path))
    missing, ancestor = [], path
    while not ancestor.exists():
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    with opened_directory(ancestor):
        pass
    for name in reversed(missing):
        with opened_directory(ancestor) as (_, fd, verify):
            verify()
            if os.name == 'nt':
                from .research_windows_publication import create_directory
                create_directory(fd, name)
            else:
                try:
                    os.mkdir(name, dir_fd=fd)
                except FileExistsError:
                    pass  # The next admission still rejects redirected entries.
            verify()
        ancestor = ancestor / name
    with opened_directory(path):
        pass
    return path


class Publication:
    def __init__(self, path, temporary, parent_fd, verify_directory, content):
        self.path, self.temporary = path, temporary
        self.parent_fd, self.verify_directory = parent_fd, verify_directory
        self.stream = None
        self.stamp = None
        self.native_stamp = None
        self.created = False
        self.content = content

    def create_stage(self):
        # This is also the fault-test boundary: tests forward the actual
        # relative OS open rather than inventing identities or file results.
        self.verify_directory()
        if os.name == 'nt':
            from .research_windows_publication import create_stage
            fd = create_stage(self.parent_fd, self.temporary.name)
        else:
            fd = os.open(self.temporary.name, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                         0o666, dir_fd=self.parent_fd)
        self.created = True
        try:
            self.stream = os.fdopen(fd, 'w+b')
        except BaseException:
            os.close(fd)
            raise
        self.verify_directory()
        return self.stream

    def freeze(self):
        from .research_files import _identity
        self.stamp = _identity(os.fstat(self.stream.fileno()))
        if os.name == 'nt':
            from .research_windows_files import descriptor_change_time
            self.native_stamp = descriptor_change_time(self.stream.fileno())

    def verify(self):
        from .research_files import _identity
        self.verify_directory()
        held = os.fstat(self.stream.fileno())
        if not stat.S_ISREG(held.st_mode) or _identity(held) != self.stamp:
            raise ResearchError('CORRUPT_ARTIFACT', 'owned staging changed after flush')
        if os.name == 'nt':
            from .research_windows_files import descriptor_change_time
            if descriptor_change_time(self.stream.fileno()) != self.native_stamp:
                raise ResearchError('CORRUPT_ARTIFACT', 'native owned staging identity changed after flush')
        else:
            current = os.stat(self.temporary.name, dir_fd=self.parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or _object(current) != _object(held):
                raise ResearchError('CORRUPT_ARTIFACT', 'staging name no longer identifies the owned file')
        # A real fsync wrapper or another writer can mutate the SAME inode
        # before freeze; identity alone must not declare those bytes ours.
        self.stream.seek(0)
        offset = 0
        while offset < len(self.content):
            chunk = self.stream.read(min(64 * 1024, len(self.content) - offset))
            if not chunk or chunk != self.content[offset:offset + len(chunk)]:
                raise ResearchError('CORRUPT_ARTIFACT', 'owned staging content differs from declared payload')
            offset += len(chunk)
        if self.stream.read(1) or _identity(os.fstat(self.stream.fileno())) != self.stamp:
            raise ResearchError('CORRUPT_ARTIFACT', 'owned staging changed during publication verification')

    def publish(self, *, replace):
        self.verify()
        if os.name == 'nt':
            from .research_windows_publication import publish
            publish(self.parent_fd, self.stream.fileno(), self.path.name, replace=replace)
            self.created = False
        elif replace:
            # Ordinary authority-store replacement is serialized by its OS
            # store lock. Anchor both names to the original directory.
            os.replace(self.temporary.name, self.path.name,
                       src_dir_fd=self.parent_fd, dst_dir_fd=self.parent_fd)
            self.created = False
        else:
            # Linux linkat AT_SYMLINK_FOLLOW on /proc/self/fd does not require
            # CAP_DAC_READ_SEARCH (unlike AT_EMPTY_PATH). Unsupported systems
            # fail closed; never fall back to an overwriting rename.
            os.link('/proc/self/fd/' + str(self.stream.fileno()), self.path.name,
                    dst_dir_fd=self.parent_fd, follow_symlinks=True)
        return self.created

    def close(self):
        try:
            if self.created and self.stream is not None:
                if os.name == 'nt':
                    from .research_windows_publication import remove_owned
                    remove_owned(self.parent_fd, self.stream.fileno())
                else:
                    # Locate the original held object even after its parent
                    # or entry moved. Delete only a matching original inode;
                    # the staging UUID may now belong to another writer.
                    source = Path(os.readlink('/proc/self/fd/' + str(self.stream.fileno())))
                    parent = Path(os.readlink('/proc/self/fd/' + str(self.parent_fd)))
                    if source.parent == parent:
                        try:
                            current = os.stat(source.name, dir_fd=self.parent_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            return
                        if _object(current) == _object(os.fstat(self.stream.fileno())):
                            os.unlink(source.name, dir_fd=self.parent_fd)
        finally:
            if self.stream is not None:
                self.stream.close()
