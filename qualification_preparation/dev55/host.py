"""Fail-closed future host. No native imports and no approving CLI.

Trust is installed by the embedding control plane, never parsed from a request.
The authority context is an independently reviewed live decision/enforcement
service, not a boolean receipt. This module creates no authority or balances.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import ctypes
import hashlib
import importlib.util
from importlib import metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading

BASE = '7f426dfa4fe0f0666ff9cec69301a81f6acb183c'
DRIVER_SHA = '9c1798786dc77e50c235fbe1fed8092f151a7a8c237af9ff261e4d59e152e7c3'
EXPIRY = '2026-11-03T23:59:00+08:00'
PRODUCT = {'application','infrastructure','models','estimation','evaluation'}
LIMITS = (1800, 120, 120, 2147483648, 4194304)

class Rejected(ValueError):
    def __init__(self, check, detail):
        self.check, self.detail = check, detail
        super().__init__(check + ': ' + detail)

def need(ok, check, detail):
    if not ok:
        raise Rejected(check, detail)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def identity(path):
    p = Path(path)
    need(p.is_absolute() and p == p.resolve(), 'path', str(p))
    for ancestor in (p, *p.parents):
        s = ancestor.lstat()
        need(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
             'redirect', str(ancestor))
    s = p.stat()
    need(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'file', str(p))
    return (str(p), s.st_size, hashlib.sha256(p.read_bytes()).hexdigest())

@dataclass(frozen=True)
class Frozen:
    repository: str
    runtime_root: str
    interpreter: tuple
    files: tuple  # (absolute path, size, sha256), complete source + dependency closure
    modules: tuple  # exact currently loaded module-name/file identities (no product)
    acl_reader: tuple  # pinned Windows PowerShell executable
    acl: tuple  # complete ordered ancestor (path, owner SID, SDDL) observations
    distributions: tuple  # metadata names/versions, no package imports
    resource_min_free_bytes: int
    resource_min_disk_bytes: int
    authority_reference: str  # locator ONLY, not a grant
    authority_code: tuple  # reviewed installed provider implementation file

    def request(self):
        repo = Path(self.repository)
        driver = repo / 'qualification_preparation/dev54/driver.py'
        return dict(scope='original205-artificial-only', baseline=BASE, driver=str(driver),
                    driver_sha256=DRIVER_SHA, argv=[self.interpreter[0], str(driver), '--mode',
                    'execute', '--runtime-root', self.runtime_root], runtime_root=self.runtime_root,
                    expiry=EXPIRY, cwd=self.repository, native=True, science=False)

def loaded_modules():
    result = []
    for name, module in tuple(sys.modules.items()):
        need(name.split('.')[0] not in PRODUCT, 'clean-process', name)
        file = getattr(module, '__file__', None)
        if file:
            result.append((name, str(Path(file).resolve())))
    return tuple(sorted(result))

def parent_paths(root):
    p = Path(root)
    need(p.is_absolute() and p == p.resolve(), 'root', 'absolute canonical root required')
    need(not p.exists() and not p.is_symlink(), 'root', 'fresh root required')
    need(p.parent.is_dir(), 'root', 'existing owned parent required')
    for a in p.parents:
        s = a.lstat()
        need(stat.S_ISDIR(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024
             and not stat.S_ISLNK(s.st_mode), 'root', 'redirected ancestor')
        need(not (a / '.git').exists(), 'root', 'Git ancestor')
    return tuple(str(a) for a in p.parents)

def read_acl(paths, executable):
    """Readonly supported Get-Acl/.NET security descriptor API, no ACL mutation."""
    need(os.name == 'nt', 'platform', 'Windows containment required')
    # Paths encoded as data on stdin, never interpolated into shell source.
    script = """$ErrorActionPreference='Stop'; $paths=[Console]::In.ReadToEnd()|ConvertFrom-Json;
$rows=@(foreach($p in $paths){$a=Get-Acl -LiteralPath $p; [pscustomobject]@{path=$p;owner=$a.GetOwner([System.Security.Principal.SecurityIdentifier]).Value;sddl=$a.Sddl}});
ConvertTo-Json -InputObject $rows -Compress"""
    r = subprocess.run([executable, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
                       input=json.dumps(list(paths)), text=True, capture_output=True, timeout=30)
    need(r.returncode == 0, 'acl-api', r.stderr[:1000])
    rows = json.loads(r.stdout.lstrip('\ufeff'))
    need(type(rows) is list and [r['path'] for r in rows] == list(paths), 'acl-api', 'incomplete result')
    return tuple((r['path'], r['owner'], r['sddl']) for r in rows)

def restrictive_acl(rows, trusted_sid, *, descendants=False):
    """Conservative allowlist: unknown/object/callback ACEs fail closed."""
    allowed = {trusted_sid, 'SY', 'BA', 'S-1-5-18', 'S-1-5-32-544'}
    for path, owner, sddl in rows:
        need(owner in allowed, 'acl-owner', path)
        need('D:' in sddl, 'acl-dacl', path)
        dacl = sddl.split('D:', 1)[1].split('S:', 1)[0]
        aces = re.findall(r'\(([^()]*)\)', dacl)
        need(bool(aces), 'acl-dacl', 'empty/unknown DACL')
        for ace in aces:
            parts = ace.split(';')
            need(len(parts) == 6 and parts[0] in {'A','D'} and not parts[3] and not parts[4],
                 'acl-ace', path)
            kind, flags, rights, _, _, sid = parts
            tokens = re.findall(r'..', flags)
            need(''.join(tokens) == flags and set(tokens) <= {'OI','CI','NP','IO','ID'},
                 'acl-ace', path + ' unknown inheritance flags')
            # IO is ineffective here, but OI/CI can grant on future descendants.
            # Conservatively include OI-only/NP cases; do not emulate token access.
            inheritable = descendants and bool({'OI','CI'} & set(tokens))
            if kind == 'D' or ('IO' in tokens and not inheritable) or sid in allowed:
                continue
            # Generic/file read and execute only. Everything uncertain is refused.
            need(rights in {'GR', 'GX', 'GRGX', 'FR', 'FX', 'FRFX', '0x1200a9', '0x120089'},
                 'acl-write', path + ' untrusted SID ' + sid)

def resources():
    need(os.name == 'nt', 'platform', 'Windows required')
    class Memory(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
            (n,ctypes.c_ulonglong) for n in ('total','available','page_total','page_free','virtual_total','virtual_free','extended')]
    m = Memory(); m.length = ctypes.sizeof(m)
    need(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))), 'memory-api', 'failed')
    return m.available

def native_images():
    """Readonly actual loaded PE/DLL inventory, including transitive dependencies."""
    need(os.name == 'nt', 'platform', 'Windows required')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    api.GetModuleFileNameExW.argtypes = [wintypes.HANDLE,wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    modules = (wintypes.HMODULE * 4096)(); size = wintypes.DWORD()
    process = kernel.GetCurrentProcess()
    need(bool(api.EnumProcessModules(process,modules,ctypes.sizeof(modules),ctypes.byref(size))) and size.value <= ctypes.sizeof(modules), 'images-api', 'incomplete inventory')
    paths = []
    for module in modules[:size.value // ctypes.sizeof(wintypes.HMODULE)]:
        buffer = ctypes.create_unicode_buffer(32768)
        count = api.GetModuleFileNameExW(process,module,buffer,len(buffer))
        need(0 < count < len(buffer)-1, 'images-api', 'path unavailable')
        paths.append(str(Path(buffer.value).resolve()))
    return tuple(sorted(set(paths)))

def environment(frozen, trusted_sid):
    """Observed metadata/bytes only; never import native dependencies."""
    need(os.name == 'nt', 'platform', 'Windows required')
    need(Path.cwd() == Path(frozen.repository), 'cwd', 'actual cwd differs')
    paths = parent_paths(frozen.runtime_root)
    need(identity(sys.executable) == frozen.interpreter, 'interpreter', 'identity differs')
    need(identity(frozen.acl_reader[0]) == frozen.acl_reader, 'acl-reader', 'identity differs')
    expected = {r[0]: r for r in frozen.files}
    need(len(expected) == len(frozen.files) and bool(expected), 'closure', 'duplicate/empty file closure')
    for row in frozen.files:
        need(identity(row[0]) == row, 'closure', row[0])
    need(identity(frozen.authority_code[0]) == frozen.authority_code, 'authority-code', 'identity differs')
    images = native_images()
    need(all(path in expected for path in images), 'native-closure', 'actual loaded image absent from frozen pins')
    current = loaded_modules()
    need(current == frozen.modules, 'modules', 'loaded module inventory differs')
    need(all(path in expected for _,path in current), 'closure', 'loaded file missing from pins')
    # Complete repository source closure: no extra executable/source/fixture path.
    repository = Path(frozen.repository)
    actual = {str(p) for p in repository.rglob('*') if p.is_file() and '.git' not in p.relative_to(repository).parts
             }
    need(actual == {p for p in expected if Path(p).is_relative_to(repository)}, 'source-set', 'extra/missing source path')
    acl = read_acl(paths, frozen.acl_reader[0])
    need(acl == frozen.acl, 'acl-change', 'ancestor descriptors differ')
    restrictive_acl(acl, trusted_sid)
    # Unchanged driver mkdir inherits this immediate parent's descriptor.
    # Inspect propagation even when the ACE is ineffective on the parent itself.
    restrictive_acl(acl[:1], trusted_sid, descendants=True)
    # A parent's DACL does not control access to an existing child file.
    pinned_paths = tuple(sorted(set(expected) | {frozen.interpreter[0],
                         frozen.acl_reader[0], frozen.authority_code[0]}))
    restrictive_acl(read_acl(pinned_paths, frozen.acl_reader[0]), trusted_sid)
    # Protect replacement routes for every pin, including interpreter/ACL reader.
    dep_parents = tuple(sorted({str(a) for p in pinned_paths for a in Path(p).parents}))
    restrictive_acl(read_acl(dep_parents, frozen.acl_reader[0]), trusted_sid)
    observed = tuple(sorted((d.metadata['Name'], d.version) for d in metadata.distributions()))
    need(observed == frozen.distributions and bool(observed), 'dependencies', 'metadata differs/missing')
    need(type(frozen.resource_min_free_bytes) is int and frozen.resource_min_free_bytes >= LIMITS[3], 'resource', 'memory floor')
    need(type(frozen.resource_min_disk_bytes) is int and frozen.resource_min_disk_bytes >= LIMITS[4], 'resource', 'disk floor')
    free = resources(); disk = shutil.disk_usage(Path(frozen.runtime_root).parent).free
    need(free >= frozen.resource_min_free_bytes and disk >= frozen.resource_min_disk_bytes, 'resource', 'observed availability insufficient')
    return dict(memory_available=free, disk_available=disk, runtime_margin='UNKNOWN')

def _admission(frozen, authority, trusted_sid, evidence, probe):
    """Internal test seam. Public factory always installs actual OS probe."""
    used = False
    expected = canonical(frozen.request())
    owner = (os.getpid(), threading.get_ident())
    @contextmanager
    def admission(request):
        nonlocal used
        primary = None
        try:
            need(not used, 'single-use', 'no replay of uncertain effects')
            used = True
            need((os.getpid(), threading.get_ident()) == owner, 'owner', 'process/thread differs')
            need(canonical(request) == expected, 'request', 'exact request differs')
            now = datetime.now(timezone.utc)
            need(now < datetime.fromisoformat(EXPIRY), 'expiry', 'fixed ceiling expired')
            # authority is a trusted embedding service, never taken from request.
            # Its context must retain real authorization/resource/stop enforcement
            # through the complete driver call, including original physical exit.
            with authority.hold(request, reference=frozen.authority_reference, limits=LIMITS) as lease:
                lease.checkpoint('before-preflight')  # raises unless current actual scope/resources valid
                observed = probe(frozen, trusted_sid)
                need(canonical(request) == expected, 'request', 'mutated during authority/probe')
                need(datetime.now(timezone.utc) < datetime.fromisoformat(EXPIRY), 'expiry', 'expired in preflight')
                lease.checkpoint('before-native')
                evidence.append(dict(event='preflight', observed=observed, native_qualification='UNKNOWN'))
                try:
                    yield
                except BaseException as body_error:
                    primary = body_error
                    raise
                finally:
                    # Never replace body failure by a terminal status or manufacture PASS.
                    evidence.append(dict(event='driver-exit', runtime_qualification='UNKNOWN'))
            evidence.append(dict(event='authority-context-exit'))
        except BaseException as exc:
            evidence.append(dict(event='rejected-or-body-error', check=getattr(exc,'check','external'), error=type(exc).__name__, detail=str(exc)))
            if primary is not None and exc is not primary:
                raise primary from exc
            raise
    return admission

def make_admission(frozen, *, authority, trusted_sid, evidence):
    """Trusted caller only. No CLI/JSON construction of authority is supported.

    The embedding service must be independently approved for this exact action;
    this API neither certifies an arbitrary object nor turns its shape into trust.
    """
    need(type(frozen) is Frozen, 'freeze', 'explicit immutable binding required')
    need(type(trusted_sid) is str and trusted_sid.startswith('S-1-'), 'sid', 'actual trusted owner SID required')
    need(bool(frozen.authority_reference), 'authority', 'missing actual decision locator')
    need(identity(frozen.authority_code[0]) == frozen.authority_code, 'authority-code', 'changed')
    need(frozen.authority_code[0] in {r[0] for r in frozen.files}, 'authority-code', 'not in closure')
    need(inspect.getsourcefile(type(authority)) is not None and str(Path(inspect.getsourcefile(type(authority))).resolve()) == frozen.authority_code[0], 'authority-origin', 'installed provider differs from reviewed implementation')
    return _admission(frozen, authority, trusted_sid, evidence, environment)

def future_entry(frozen, *, authority, trusted_sid, evidence):
    """FUTURE separately admitted only. Never exercised by DEV55."""
    admission = make_admission(frozen, authority=authority, trusted_sid=trusted_sid, evidence=evidence)
    path = Path(frozen.repository) / 'qualification_preparation/dev54/driver.py'
    need(identity(path)[2] == DRIVER_SHA, 'driver', 'reviewed bytes differ')
    # Only reviewed stdlib driver definitions load here. All original/native work
    # remains inside its admission context. No caller-supplied driver callback.
    spec = importlib.util.spec_from_file_location('_dev55_driver54', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.execute_after_admission(Path(frozen.repository), Path(frozen.runtime_root), admission, frozen.request()['argv'])

if __name__ == '__main__':
    print('DEV55_SOURCE_ONLY: no CLI authority; future_entry requires separately admitted trusted embedding')
    raise SystemExit(2)
