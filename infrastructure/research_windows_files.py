"""Fresh Windows source change identity, not creation time or an authority cache.

Only API definitions are cached. Initial metadata opens request attributes,
not payload bytes; verification queries the original held data handle. No
path resolution, content buffering, timestamp rewriting or fallback stamps.
"""
from functools import lru_cache

from .research_store import ResearchError


@lru_cache(maxsize=1)
def _api():
    import ctypes
    from ctypes import wintypes

    class BasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in
                    ('CreationTime', 'LastAccessTime', 'LastWriteTime', 'ChangeTime')]
        _fields_.append(('FileAttributes', wintypes.DWORD))

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    return ctypes, BasicInfo, query, create, close


def _change_time(handle):
    ctypes, BasicInfo, query, _, _ = _api()
    value = BasicInfo()
    if not query(handle, 0, ctypes.byref(value), ctypes.sizeof(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    if value.ChangeTime <= 0:
        # Unsupported/unknown change identity cannot prove an unchanged file.
        raise ResearchError('CORRUPT_ARTIFACT', 'native source change identity unavailable')
    return value.ChangeTime


def path_change_time(path):
    ctypes, _, _, create, close = _api()
    # FILE_READ_ATTRIBUTES; share read/write/delete; OPEN_EXISTING;
    # FILE_FLAG_OPEN_REPARSE_POINT. Never create or follow a leaf reparse link.
    handle = create(str(path), 0x80, 0x7, None, 3, 0x00200000, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return _change_time(handle)
    finally:
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def descriptor_change_time(fd):
    import msvcrt
    return _change_time(msvcrt.get_osfhandle(fd))
