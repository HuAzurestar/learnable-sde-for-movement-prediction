"""Recovery observes executing workers, not unreaped Linux process records."""

import json
import io
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import infrastructure.process_tree as process_tree
from infrastructure.process_tree import ProcessTree, _linux_group_active, process_may_be_alive


LINUX = sys.platform.startswith("linux") and Path("/proc").is_dir()


def wait_for_zombie(pid):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return
        time.sleep(0.005)
    raise AssertionError("owned child did not reach unreaped zombie state")


@pytest.mark.skipif(not LINUX, reason="actual Linux /proc zombie observation")
def test_unreaped_group_leader_has_stopped():
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        wait_for_zombie(child.pid)  # Do not poll/wait: retain the actual zombie.
        os.kill(child.pid, 0)  # The former probe succeeds despite no execution.
        assert process_may_be_alive(child.pid) is False
    finally:
        child.wait(timeout=5)


@pytest.mark.skipif(not LINUX, reason="actual Linux process-group observation")
def test_live_group_leader_keeps_recovery_hold():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             start_new_session=True)
    tree = ProcessTree(child)
    try:
        assert process_may_be_alive(child.pid) is True
    finally:
        tree.terminate()
        assert tree.wait_stopped(timeout=5)
        child.wait(timeout=5)
        tree.close()


DESCENDANT_PROBE = r'''
import ctypes, json, os, signal, subprocess, sys, time
from pathlib import Path
from infrastructure.process_tree import process_may_be_alive
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'cannot isolate owned subreaper fixture')
leader_code = """
import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', sys.argv[1]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid, flush=True)
"""
leader = subprocess.Popen([sys.executable, '-c', leader_code, sys.argv[1]],
                          start_new_session=True, stdout=subprocess.PIPE, text=True)
descendant = None
try:
    descendant = int(leader.stdout.readline())
    leader.wait(timeout=5)
    assert not Path(f'/proc/{leader.pid}').exists()
    until = time.monotonic() + 5
    while True:
        fields = Path(f'/proc/{descendant}/stat').read_text().rsplit(')', 1)[1].split()
        if int(fields[1]) == os.getpid() and (sys.argv[2] != 'Z' or fields[0] == 'Z'):
            break
        if time.monotonic() >= until:
            raise AssertionError('owned descendant was not adopted in expected state')
        time.sleep(0.005)
    assert int(fields[2]) == leader.pid
    os.killpg(leader.pid, 0)
    print(json.dumps({'leader_absent': True, 'descendant_state': fields[0],
                      'may_be_alive': process_may_be_alive(leader.pid)}), flush=True)
finally:
    if leader.poll() is None:
        leader.kill()
        leader.wait(timeout=5)
    if descendant is not None:
        try:
            os.kill(descendant, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(descendant, 0)
    leader.stdout.close()
'''


@pytest.mark.skipif(not LINUX, reason="actual Linux isolated subreaper/process group")
@pytest.mark.parametrize("code,state,expected", [("pass", "Z", False),
                            ("import time; time.sleep(30)", "live", True)])
def test_descendants_outlive_reaped_leader_without_losing_live_hold(code, state, expected):
    result = subprocess.run([sys.executable, "-c", DESCENDANT_PROBE, code, state],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=15, check=True)
    observation = json.loads(result.stdout)
    assert observation["leader_absent"] is True
    assert (observation["descendant_state"] == "Z") == (state == "Z")
    assert observation["may_be_alive"] is expected


@pytest.mark.skipif(not LINUX, reason="Linux conservative /proc failure observation")
@pytest.mark.parametrize("failure", ["permission", "io", "malformed", "quota"])
def test_unknown_group_observation_never_proves_exit(monkeypatch, failure):
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        wait_for_zombie(child.pid)
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if path == Path(f"/proc/{child.pid}/stat"):
                if failure == "permission":
                    raise PermissionError("observation denied")
                if failure == "io":
                    raise OSError("observation unavailable")
                if failure == "malformed":
                    return "invalid process stat"
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
        if failure == "quota":
            original_iter = Path.iterdir
            monkeypatch.setattr(Path, "iterdir", lambda path:
                (Path("/proc/nonprocess") for _ in range(100_002))
                if path == Path("/proc") else original_iter(path))
        assert process_may_be_alive(child.pid) is True
    finally:
        child.wait(timeout=5)


@pytest.mark.parametrize("pid", [None, 1, True, -1, "123"])
def test_invalid_worker_identity_never_proves_exit(pid):
    assert process_may_be_alive(pid) is True


@pytest.mark.skipif(not LINUX, reason="actual Linux live PID outside recorded group")
def test_live_pid_outside_expected_group_keeps_recovery_hold():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert os.getpgid(child.pid) != child.pid
        assert process_may_be_alive(child.pid) is True
    finally:
        child.kill()
        child.wait(timeout=5)


def test_group_scan_allows_unrelated_kernel_group_and_parentheses(tmp_path, monkeypatch):
    monkeypatch.setattr(process_tree, "_linux_proc_visible", lambda: None)
    kernel, zombie = tmp_path / "2", tmp_path / "42"
    kernel.mkdir()
    zombie.mkdir()
    (kernel / "stat").write_text("2 (kernel) S 0 0 0", encoding="utf-8")
    (zombie / "stat").write_text("42 (comm) with) parentheses) Z 1 42 42", encoding="utf-8")
    original_iter = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path:
                        iter((kernel, zombie)) if path == Path("/proc") else original_iter(path))
    assert _linux_group_active(42, include_pid=True) is False


@pytest.mark.skipif(not LINUX, reason="actual Linux kernel probe with omitted visible record")
@pytest.mark.parametrize("new_group", [True, False])
def test_kernel_alive_without_enumerated_record_keeps_hold(monkeypatch, new_group):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             start_new_session=new_group)
    try:
        original_iter = Path.iterdir
        monkeypatch.setattr(Path, "iterdir", lambda path:
            (entry for entry in original_iter(path) if entry.name != str(child.pid))
            if path == Path("/proc") else original_iter(path))
        os.kill(child.pid, 0)
        assert process_may_be_alive(child.pid) is True
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.mark.skipif(not LINUX, reason="Linux observation mount restrictions")
@pytest.mark.parametrize("mount", ["proc /proc proc rw,hidepid=1 0 0\n",
    "proc /proc proc rw,hidepid=2 0 0\n", "proc /proc proc rw,hidepid=ptraceable 0 0\n",
    "proc /proc proc rw,pidns=other 0 0\n", "invalid mount observation",
    "x" * 1_048_577], ids=["hidepid1", "hidepid2", "ptraceable", "pidns", "malformed", "quota"])
def test_restricted_or_unknown_proc_mount_keeps_hold(monkeypatch, mount):
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        wait_for_zombie(child.pid)
        original_open = Path.open
        monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs:
            io.StringIO(mount) if path == Path("/proc/self/mounts")
            else original_open(path, *args, **kwargs))
        assert process_may_be_alive(child.pid) is True
        with pytest.raises(OSError):
            ProcessTree(child).active()
    finally:
        child.wait(timeout=5)
