"""Scope-bound original directory members, never source or authority caches."""
from contextlib import contextmanager
import errno
import fnmatch
import os
from pathlib import Path
import stat
import threading

from .research_store import ResearchError


class DirectoryMembers:
    def __init__(self, path, fd):
        self.path, self.fd = Path(path), fd
        self.owner = os.getpid(), threading.get_ident()
        information = os.fstat(fd)
        self.identity = information.st_dev, information.st_ino
        self.active = True

    def finish(self):
        self.active = False

    def check(self):
        if not self.active or self.owner != (os.getpid(), threading.get_ident()):
            raise ResearchError('CORRUPT_ARTIFACT', 'metadata directory scope is not owned and active')
        information = os.fstat(self.fd)
        if (not stat.S_ISDIR(information.st_mode)
                or (information.st_dev, information.st_ino) != self.identity):
            raise ResearchError('CORRUPT_ARTIFACT', 'original metadata directory handle changed')

    def names(self):
        self.check()
        if os.name == 'nt':
            from .research_windows_metadata import directory_names
            names = directory_names(self.fd)
        else:
            names = os.listdir(self.fd)
        self.check()
        return sorted(name for name in names if fnmatch.fnmatch(name, '*.json'))

    @contextmanager
    def opened_member(self, path):
        from .research_files import _identity
        self.check()
        path = Path(os.path.abspath(path))
        if path.parent != self.path or path.name in ('', '.', '..') or (os.name == 'nt' and ':' in path.name):
            raise ResearchError('UNAUTHORIZED_DATA', 'metadata member left its original directory')
        attribute_fd = None
        try:
            if os.name == 'nt':
                from .research_windows_metadata import member_descriptor, member_stamp
                attribute_fd = member_descriptor(self.fd, path.name)
                before = member_stamp(attribute_fd)
                self.check()
                fd = member_descriptor(self.fd, path.name, payload=True)
            else:
                information = os.stat(path.name, dir_fd=self.fd, follow_symlinks=False)
                if not stat.S_ISREG(information.st_mode):
                    raise ResearchError('UNAUTHORIZED_DATA', 'metadata member is not a regular file')
                before = _identity(information)
                self.check()
                try:
                    fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
                except OSError as error:
                    if error.errno == errno.ELOOP:
                        raise ResearchError('UNAUTHORIZED_DATA', 'metadata member became a symbolic link') from error
                    raise
            try:
                stream = os.fdopen(fd, 'rb')
            except BaseException:
                os.close(fd)
                raise
            with stream:
                def stamp(descriptor):
                    if os.name == 'nt':
                        # This fresh native stamp already queries fstat and
                        # checks regular/reparse attributes plus ChangeTime.
                        # Do not query the same descriptor twice per stamp.
                        return member_stamp(descriptor)
                    information = os.fstat(descriptor)
                    if not stat.S_ISREG(information.st_mode):
                        raise ResearchError('UNAUTHORIZED_DATA', 'metadata handle is not a regular file')
                    return _identity(information)

                opened = stamp(stream.fileno())
                opened_size = opened[0][2] if os.name == 'nt' else opened[2]

                def verify():
                    self.check()
                    held = stamp(stream.fileno())
                    if os.name == 'nt':
                        current_fd = member_descriptor(self.fd, path.name)
                        try:
                            current = stamp(current_fd)
                        finally:
                            os.close(current_fd)
                    else:
                        information = os.stat(path.name, dir_fd=self.fd, follow_symlinks=False)
                        if not stat.S_ISREG(information.st_mode):
                            raise ResearchError('UNAUTHORIZED_DATA', 'metadata entry changed from regular file')
                        current = _identity(information)
                    self.check()
                    if before != opened or current != opened or held != opened:
                        raise ResearchError('CORRUPT_ARTIFACT', 'metadata identity changed during read')

                verify()
                # verify() above requires the current/held sizes to equal the
                # actual opened stamp; later changes still fail final verify.
                yield stream, opened_size, verify
                verify()
        finally:
            if attribute_fd is not None:
                os.close(attribute_fd)
