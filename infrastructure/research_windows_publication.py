"""Original directory-relative Windows I/O and owned-source handle publication.

Definitions only are cached; no path, authority or object identity is cached.
There is no path-only rename/delete fallback when a native operation fails.
"""
from functools import lru_cache
import os


@lru_cache(maxsize=1)
def _api():
    import ctypes as c
    from ctypes import wintypes as w
    from types import SimpleNamespace

    class Unicode(c.Structure):
        _fields_ = [('Length', w.USHORT), ('MaximumLength', w.USHORT), ('Buffer', w.LPWSTR)]

    class Attributes(c.Structure):
        _fields_ = [('Length', w.ULONG), ('RootDirectory', w.HANDLE),
                    ('ObjectName', c.POINTER(Unicode)), ('Attributes', w.ULONG),
                    ('SecurityDescriptor', c.c_void_p), ('SecurityQualityOfService', c.c_void_p)]

    class Status(c.Structure):
        _fields_ = [('Status', c.c_void_p), ('Information', c.c_size_t)]

    class Rename(c.Structure):
        _fields_ = [('ReplaceIfExists', w.BOOLEAN), ('RootDirectory', w.HANDLE),
                    ('FileNameLength', w.DWORD), ('FileName', w.WCHAR * 1)]

    kernel, native = c.WinDLL('kernel32', use_last_error=True), c.WinDLL('ntdll')
    create = kernel.CreateFileW
    create.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
    create.restype = w.HANDLE
    close = kernel.CloseHandle
    close.argtypes, close.restype = [w.HANDLE], w.BOOL
    ntcreate = native.NtCreateFile
    ntcreate.argtypes = [c.POINTER(w.HANDLE), w.DWORD, c.POINTER(Attributes), c.POINTER(Status),
                        c.c_void_p, w.ULONG, w.ULONG, w.ULONG, w.ULONG, c.c_void_p, w.ULONG]
    ntcreate.restype = c.c_long
    ntset = native.NtSetInformationFile
    ntset.argtypes = [w.HANDLE, c.POINTER(Status), c.c_void_p, w.ULONG, c.c_int]
    ntset.restype = c.c_long
    translate = native.RtlNtStatusToDosError
    translate.argtypes, translate.restype = [c.c_long], w.ULONG
    final = kernel.GetFinalPathNameByHandleW
    final.argtypes, final.restype = [w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD], w.DWORD
    setinfo = kernel.SetFileInformationByHandle
    setinfo.argtypes, setinfo.restype = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL
    return SimpleNamespace(c=c, w=w, Unicode=Unicode, Attributes=Attributes, Status=Status,
                          Rename=Rename, create=create, close=close, ntcreate=ntcreate,
                          ntset=ntset, translate=translate, final=final, setinfo=setinfo)


def _handle(fd):
    import msvcrt
    return msvcrt.get_osfhandle(fd)


def _descriptor(handle, flags):
    import msvcrt
    try:
        return msvcrt.open_osfhandle(handle, flags)
    except BaseException:
        _api().close(handle)
        raise


def directory_descriptor(path):
    api = _api()
    # READ_ATTRIBUTES | TRAVERSE | SYNCHRONIZE; all share modes, no reparse follow.
    handle = api.create(str(path), 0x1000a0, 7, None, 3, 0x02200000, None)
    if handle in (None, api.c.c_void_p(-1).value):
        raise api.c.WinError(api.c.get_last_error())
    return _descriptor(handle, os.O_RDONLY)


def _relative(parent_fd, name, access, disposition, *, directory=False):
    api = _api()
    text = api.c.create_unicode_buffer(name)
    length = len(name.encode('utf-16-le'))
    if length > 65532:
        raise ValueError('publication member name is too long')
    unicode = api.Unicode(length, length + 2, api.c.cast(text, api.w.LPWSTR))
    attributes = api.Attributes(api.c.sizeof(api.Attributes), _handle(parent_fd),
                               api.c.pointer(unicode), 0x40, None, None)
    result, status = api.w.HANDLE(), api.Status()
    options = 0x00200020 | (1 if directory else 0x40)
    code = api.ntcreate(api.c.byref(result), access, api.c.byref(attributes), api.c.byref(status),
                        None, 0x80, 7, disposition, options, None, 0)
    if code < 0:
        raise api.c.WinError(api.translate(code))
    return result.value


def create_stage(parent_fd, name):
    # Keep RW/share-delete open through publication. Requesting DELETE now
    # would unnecessarily prohibit ordinary Path.read_bytes in grant hooks.
    return _descriptor(_relative(parent_fd, name, 0xc0100000, 2), os.O_RDWR | os.O_BINARY)


def create_directory(parent_fd, name):
    fd = _descriptor(_relative(parent_fd, name, 0x1000a0, 3, directory=True), os.O_RDONLY)
    os.close(fd)


def descriptor_path(fd):
    api = _api()
    text = api.c.create_unicode_buffer(32768)
    length = api.final(_handle(fd), text, len(text), 0)
    if not length or length >= len(text):
        raise api.c.WinError(api.c.get_last_error())
    return text.value


def owned_delete_descriptor(parent_fd, source_fd):
    """Open DELETE on the current original object, never a reused UUID entry."""
    from pathlib import Path
    from .research_store import ResearchError
    source, parent = Path(descriptor_path(source_fd)), Path(descriptor_path(parent_fd))
    if source.parent != parent:
        raise ResearchError('UNAUTHORIZED_DATA', 'owned staging left its original directory')
    fd = _descriptor(_relative(parent_fd, source.name, 0x110080, 1), os.O_RDONLY)
    try:
        before, opened = os.fstat(source_fd), os.fstat(fd)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ResearchError('CORRUPT_ARTIFACT', 'staging entry no longer identifies the owned file')
        return fd
    except BaseException:
        os.close(fd)
        raise


def publish(parent_fd, source_fd, name, *, replace):
    api = _api()
    fd = owned_delete_descriptor(parent_fd, source_fd)
    try:
        encoded = name.encode('utf-16-le')
        buffer = api.c.create_string_buffer(max(api.c.sizeof(api.Rename),
                                               api.Rename.FileName.offset + len(encoded)))
        value = api.Rename.from_buffer(buffer)
        value.ReplaceIfExists, value.RootDirectory, value.FileNameLength = replace, _handle(parent_fd), len(encoded)
        api.c.memmove(api.c.addressof(buffer) + api.Rename.FileName.offset, encoded, len(encoded))
        status = api.Status()
        code = api.ntset(_handle(fd), api.c.byref(status), buffer, len(buffer), 10)
        if code < 0:
            raise api.c.WinError(api.translate(code))
    finally:
        os.close(fd)


def remove_owned(parent_fd, source_fd):
    api = _api()
    fd = owned_delete_descriptor(parent_fd, source_fd)
    try:
        value = api.w.BOOLEAN(True)
        if not api.setinfo(_handle(fd), 4, api.c.byref(value), api.c.sizeof(value)):
            raise api.c.WinError(api.c.get_last_error())
    finally:
        os.close(fd)
