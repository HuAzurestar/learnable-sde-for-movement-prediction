"""Fresh enumeration and file attributes through original Windows handles."""
from functools import lru_cache
import os
import stat

from .research_store import ResearchError


@lru_cache(maxsize=1)
def _directory_record():
    import ctypes as c
    from ctypes import wintypes as w

    class Record(c.Structure):
        _fields_ = [('NextEntryOffset', w.DWORD), ('FileIndex', w.DWORD)]
        _fields_ += [(name, c.c_longlong) for name in
                     ('CreationTime', 'LastAccessTime', 'LastWriteTime', 'ChangeTime',
                      'EndOfFile', 'AllocationSize')]
        _fields_ += [('FileAttributes', w.DWORD), ('FileNameLength', w.DWORD),
                     ('EaSize', w.DWORD), ('ShortNameLength', c.c_byte),
                     ('ShortName', w.WCHAR * 12), ('FileId', c.c_longlong),
                     ('FileName', w.WCHAR * 1)]
    return Record


def directory_names(fd):
    from .research_windows_files import _api
    from .research_windows_publication import _handle
    c, _, query, _, _ = _api()
    record = _directory_record()
    # Greater than the maximum native UNICODE_STRING plus its record header;
    # this bounds the query buffer, not the directory or source payload size.
    buffer = c.create_string_buffer(128 * 1024)
    kind = 11  # FileIdBothDirectoryRestartInfo, then resume the same handle.
    names = []
    while True:
        if not query(_handle(fd), kind, buffer, len(buffer)):
            code = c.get_last_error()
            if code == 18:  # ERROR_NO_MORE_FILES only; no path fallback.
                return names
            raise c.WinError(code)
        kind, offset = 10, 0
        while True:
            if offset + c.sizeof(record) > len(buffer):
                raise ResearchError('CORRUPT_ARTIFACT', 'invalid native directory record')
            entry = record.from_buffer(buffer, offset)
            length, following = entry.FileNameLength, entry.NextEntryOffset
            end = offset + record.FileName.offset + length
            boundary = offset + following if following else len(buffer)
            if (length % 2 or end > boundary or boundary > len(buffer)
                    or (following and (following % 8 or following < c.sizeof(record)))):
                raise ResearchError('CORRUPT_ARTIFACT', 'invalid native directory name')
            name = c.string_at(c.addressof(buffer) + offset + record.FileName.offset,
                               length).decode('utf-16-le', errors='surrogatepass')
            if name not in ('.', '..'):
                names.append(name)
            if not following:
                break
            offset += following


def member_descriptor(parent_fd, name, *, payload=False):
    from .research_windows_publication import _relative, _descriptor
    access = 0x80100000 if payload else 0x100080
    return _descriptor(_relative(parent_fd, name, access, 1), os.O_RDONLY | os.O_BINARY)


def member_stamp(fd):
    from .research_files import _identity
    from .research_windows_files import _api
    from .research_windows_publication import _handle
    c, BasicInfo, query, _, _ = _api()
    value = BasicInfo()
    if not query(_handle(fd), 0, c.byref(value), c.sizeof(value)):
        raise c.WinError(c.get_last_error())
    information = os.fstat(fd)
    if value.FileAttributes & (0x400 | 0x10) or not stat.S_ISREG(information.st_mode):
        raise ResearchError('UNAUTHORIZED_DATA', 'metadata member is not an original regular file')
    if value.ChangeTime <= 0:
        raise ResearchError('CORRUPT_ARTIFACT', 'native metadata change identity unavailable')
    return _identity(information), value.ChangeTime
