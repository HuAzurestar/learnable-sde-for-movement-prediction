"""Real child completion must wake the wrapper, not await heartbeat sleep."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from infrastructure.process_tree import ProcessTree


@pytest.mark.parametrize("exit_code", [0, 85])
def test_actual_wrapper_waits_responsively_for_child_exit(tmp_path, exit_code):
    # Run main in an independent interpreter: its Windows job contains itself.
    # Keep GO's pipe open until exit so parent-loss containment is not triggered.
    driver = r'''
import sys, time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import research_worker as worker
heartbeat, exit_code = sys.argv[2], int(sys.argv[3])
real_popen = worker.subprocess.Popen
children = []
def popen(*args, **kwargs):
    child = real_popen(*args, **kwargs)
    children.append(child)
    return child
def delayed_poll_sleep(seconds):
    # Confirm actual exit after the actual first heartbeat, not a fake poll or
    # clock. Sleeping for a heartbeat interval cannot respond to that exit.
    code = children[0].wait(timeout=5)
    assert code == exit_code
    raise AssertionError('wrapper uses unconditional sleep despite actual child completion')
worker.subprocess.Popen = popen
worker.time = SimpleNamespace(monotonic=time.monotonic, sleep=delayed_poll_sleep)
child = "from pathlib import Path; import sys,time; path=Path(sys.argv[1]);\nwhile not path.exists(): time.sleep(0.001)\nraise SystemExit(int(sys.argv[2]))"
sys.argv = ['research_worker', heartbeat, str(time.monotonic()+20),
            sys.executable, '-c', child, heartbeat, str(exit_code)]
assert worker.main() == exit_code
print('ACTUAL_CHILD_EXIT', exit_code, flush=True)
'''
    infrastructure = Path(__file__).resolve().parents[1] / "infrastructure"
    heartbeat = tmp_path / "heartbeat.json"
    process = subprocess.Popen([sys.executable, "-c", driver, str(infrastructure),
                                str(heartbeat), str(exit_code)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=os.name != "nt",
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    tree = ProcessTree(process)
    try:
        process.stdin.write(b"GO\n")
        process.stdin.flush()
        process.wait(timeout=15)
        stdout, stderr = process.stdout.read(), process.stderr.read()
        assert process.returncode == 0, stderr.decode(errors="replace")
        assert stdout.strip() == f"ACTUAL_CHILD_EXIT {exit_code}".encode()
        assert heartbeat.is_file(), "actual heartbeat and child release were not reached"
    finally:
        tree.terminate()
        assert tree.wait_stopped(), "actual diagnostic process tree stop was not confirmed"
        process.wait(timeout=5)
        tree.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
