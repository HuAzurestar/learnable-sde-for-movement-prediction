"""Short native transport fixtures. No dataset reads, model fits, or forecasts."""
import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

# Executed as a file so it cannot be confused with a production runner module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pirc17 import formal_session as session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="counter")
    parser.add_argument("--fixture-directory")
    parser.add_argument("--artifact-count", type=int, default=0)
    parser.add_argument("--session-directory", required=True)
    parser.add_argument("--session-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "bad_ready":
        session._write(Path(args.session_directory)/"ready.json", {
            "schema_version": session.VERSION+"-ready", "session_sha256": "0"*64, "worker_pid": os.getpid()})
        time.sleep(20)
        return 0
    if args.mode == "oversized_ready":
        (Path(args.session_directory)/"ready.json").write_bytes(b"x"*(session.MAX_MESSAGE_BYTES+1))
        time.sleep(20)
        return 0
    if args.mode.startswith("bad_barrier_"):
        field = args.mode.removeprefix("bad_barrier_")
        original = session._write
        def corrupt(path, payload):
            if payload.get("schema_version") == session.VERSION+"-barrier":
                payload = dict(payload)
                payload[field] = {"sequence": True, "work_id": "0"*64, "result_sha256": "0"*64,
                    "request_sha256": "0"*64, "previous_barrier_sha256": "0"*64, "worker_pid": 1}[field]
            return original(path, payload)
        session._write = corrupt
    if args.mode == 'bootstrap_badbarrier':
        original = session._write
        def corrupt_bootstrap(path, payload):
            if payload.get('schema_version') == session.VERSION+'-bootstrap-barrier':
                payload = dict(payload, request_sha256='0'*64)
            return original(path, payload)
        session._write = corrupt_bootstrap
    def factory(config, contract):
        if args.mode == 'formal_input_denial':
            # Actual retained handler, deliberately missing human authority.
            # This mode must fail before any scientific input/fit/forecast.
            from experiments.pirc17.formal_worker import FormalWorker
            return FormalWorker(config, contract, input_options=session.read_json(args.fixture_directory))
        runtime = None
        after_ready_subprocesses = []
        if args.mode in {"platform_identity", "configured_runtime"}:
            # serve has already published ready, so this exercises the actual
            # handler-construction boundary missed by ordinary metadata CLIs.
            def audit(event, values):
                if event == "subprocess.Popen":
                    after_ready_subprocesses.append(str(values[1]))
            sys.addaudithook(audit)
            if args.mode == "configured_runtime":
                from experiments.pirc17.formal_environment import ConfiguredRuntime
                from experiments.pirc17.formal_matrix import load_protocol
                runtime = ConfiguredRuntime(load_protocol())
                runtime.verify(session.read_json(args.fixture_directory))
        calls = 0
        def handler(work, output):
            nonlocal calls
            calls += 1
            if args.mode == "deliver_saved_fixture":
                # Test-only transport of already computed synthetic artifacts.
                # This is never the formal data/training/prediction entrypoint.
                value = session.read_json(Path(args.fixture_directory)/(work['work_id']+'.json'))
                source = Path(value['source_directory'])
                for path in source.rglob('*'):
                    if path.is_file() and path.name != 'result.json':
                        target = output/path.relative_to(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                result = dict(value['manifest'])
                for key in ('artifact_path', 'context_path', 'qualification_path'):
                    if key in result:
                        result[key] = str(output/Path(result[key]).relative_to(source))
                return result
            if args.mode == "exception":
                raise ValueError("fixture exception, not private research data")
            if args.mode == "exit":
                os._exit(23)
            if args.mode == "artifact":
                (output/"fixture.bin").write_bytes(b"bounded software-only artifact")
            if args.mode == 'many_artifacts':
                if not 0 < args.artifact_count <= 20000:
                    raise ValueError('bounded software-only file inventory required')
                (output/'windows').mkdir()
                for index in range(args.artifact_count):
                    (output/'windows'/f'{index:064x}.json').write_bytes(b'{}')
            if args.mode == "sleep_second" and calls > 1:
                time.sleep(20)
            if args.mode == "child_second" and calls > 1:
                child = subprocess.Popen([sys.executable, "-I", "-S", "-c", "import time;time.sleep(20)"],
                    creationflags=subprocess.CREATE_NO_WINDOW)
                session._write(output/"child.json", {"pid": child.pid})
            if args.mode == "joined_child":
                subprocess.run([sys.executable, "-I", "-S", "-c", "pass"],
                    check=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            if args.mode == "platform_cache_reset":
                import platform
                platform._platform_cache.clear()
                platform._uname_cache = None
                platform.platform()  # Completed metadata helpers must fail too.
            if args.mode in {"platform_identity", "configured_runtime"}:
                import platform
                value = {"fixture": True, "count": calls, "pid": os.getpid(),
                    "platform": platform.platform(), "processor": platform.processor(),
                    "after_ready_subprocesses": list(after_ready_subprocesses)}
                if runtime is not None:
                    runtime.verify(session.read_json(args.fixture_directory))
                    value["environment_sha256"] = runtime.identity()["sha256"]
                return value
            return {"fixture": True, "count": calls, "pid": os.getpid()}
        if args.mode.startswith('bootstrap_'):
            bootstraps = 0
            def restore(output):
                nonlocal bootstraps
                bootstraps += 1
                if args.mode == 'bootstrap_sleep':
                    time.sleep(20)
                if args.mode == 'bootstrap_exception':
                    raise ValueError('synthetic restoration error')
                if args.mode == 'bootstrap_child':
                    subprocess.run([sys.executable, '-I', '-S', '-c', 'pass'],
                        check=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                (output/'software-only.bin').write_bytes(b'not a model or forecast')
                return dict(software_fixture_only=True, bootstraps=bootstraps, prediction_calls=calls)
            handler.bootstrap = restore
        return handler
    return session.serve(args.session_directory, args.session_sha256, factory)


if __name__ == "__main__":
    raise SystemExit(main())
