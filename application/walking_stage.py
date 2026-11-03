"""Private bounded data transfer, never an authority or a second ledger."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from datetime import datetime, timezone

from infrastructure.research_store import ResearchError, atomic_write, digest, encode

MAX_ITEM = 4 * 1024**2
MAX_TOTAL = 64 * 1024**2


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ResearchError("CONTRACT_MISMATCH", "duplicate stage field")
        value[key] = item
    return value


def regular(path, maximum):
    path = Path(path)
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
            raise ResearchError("UNAUTHORIZED_DATA", "stage link or redirection rejected")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ResearchError("UNAUTHORIZED_DATA", "stage file link or type differs")
    if info.st_size > maximum:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "stage file exceeds admitted quota")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        content = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns)
    if len(content) > maximum or identity(before) != identity(after) or identity(after) != identity(current):
        raise ResearchError("UNAUTHORIZED_DATA", "stage input changed during read")
    return content


def read_package(path):
    path = Path(path)
    raw = regular(path, 1024**2)
    package = json.loads(raw, object_pairs_hook=unique)
    if (package.get("schema") != "walking-stage-input-v2" or not isinstance(package.get("inputs"), list)
            or len(package["inputs"]) > 16 or type(package.get("deadline_monotonic")) not in (float, int)
            or not math.isfinite(package["deadline_monotonic"])):
        raise ResearchError("CONTRACT_MISMATCH", "invalid private stage package")
    values = {}; filenames = set(); total = len(raw)
    for item in package["inputs"]:
        name = item["relative_file"]
        if (not isinstance(name, str) or Path(name).name != name or name in {".", ".."}
                or any(c in name for c in "/\\:") or item["name"] in values or name in filenames):
            raise ResearchError("UNAUTHORIZED_DATA", "stage path or logical name differs")
        filenames.add(name)
        content = regular(path.parent / name, MAX_ITEM)
        total += len(content)
        if (len(content) != item["size_bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]
                or total > MAX_TOTAL):
            raise ResearchError("UNAUTHORIZED_DATA", "stage input hash or size differs")
        values[item["name"]] = content
    return package, values, hashlib.sha256(raw).hexdigest()


def prepare_package(store, config, attempt_id, inputs, output, deadline):
    """Only parent calls this, after RUNNING and inside the charged stage."""
    from .walking_pilot import preflight, cells, original_account_preflight, _read_internal_bytes
    from .research_budget import BudgetLedger
    items = []; protected = []
    with store._read_transaction():
        preflight(store, config)
        attempt = store._attempts()[attempt_id]
        run = store._manifest("run-" + attempt["run_id"]); cell = run["cell"]
        consumer = {"attempt_id": attempt_id, "run_id": run["run_id"], "entrypoint": "walking-pilot-v1"}
        def current(terminal=False):
            if ((not terminal and store._attempts()[attempt_id]["state"] != "RUNNING") or cell not in cells(config)
                    or (store.path / "recovery-hold.json").exists()):
                raise ResearchError("RECOVERY_REQUIRED", "stage input admission changed")
            if BudgetLedger(store)._balance(cell["arm_id"])["closed"]:
                raise ResearchError("BUDGET_EXHAUSTED", "stage arm closed")
        current(); original_account_preflight(store, config, cell["arm_id"])
        stage = cell["pilot_stage"]
        from .walking_pilot_budget import policy
        expiries=[policy(BudgetLedger(store),config["pilot_id"])["expires_at"],
                  store.authorization(config["artifact_grant"]["authorization_id"],version=config["artifact_grant"].get("version"))["expires_at"]]
        def add(name, artifact, content):
            if len(content) > MAX_ITEM or len(items) >= 16 or sum(len(v[1]) for v in items) + len(content) > MAX_TOTAL:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "stage input quota exceeded")
            metadata = store._manifest("artifact-" + artifact)
            protected.append(artifact)
            items.append(({"name": name, "artifact_id": artifact, "sha256": hashlib.sha256(content).hexdigest(),
                           "size_bytes": len(content), "role": metadata["role"], "block_ids": metadata["block_ids"],
                           "relative_file": "input-" + str(len(items)) + ".json"}, content))
        lineage = None
        if stage == "source":
            if config.get("source_mode") != "registered-artifacts":
                raise ResearchError("CONTRACT_MISMATCH", "B1 requires registered original artifacts")
            from .walking_pilot_sources import _read_population_bytes
            source = next(s for s in config["sources"] if s["source_id"] == cell["source_id"])
            disclosure = _read_population_bytes(store, config, source, consumer)
            if disclosure is not None:
                raw, grant, artifact = disclosure
                expiries.append(grant["expires_at"])
                # The original final guard is required before transfer as well
                # as after native stop; no grant text enters the package.
                store.verify_artifact_read(artifact, purpose=source["purpose"], authorization=grant, consumer=consumer)
                add("population", artifact, raw)
                lineage = {"artifact_id": artifact, "original_study_id": grant["study_id"], "authorization_hash": digest(grant)}
        elif stage == "prepare":
            for name, artifact in inputs["sources"].items():
                add(name, artifact, _read_internal_bytes(store, config, artifact, consumer))
        elif stage in {"train", "predict"}:
            artifact = inputs["prepared" if stage == "train" else "trained"]
            content = _read_internal_bytes(store, config, artifact, consumer)
            add("prepared" if stage == "train" else "trained", artifact, content)
            if stage == "predict":
                artifact = json.loads(content, object_pairs_hook=unique)["prepared"]
                add("prepared", artifact, _read_internal_bytes(store, config, artifact, consumer))
        else:
            raise ResearchError("CONTRACT_MISMATCH", "unknown B1 stage")
        store._read_completion(current, lambda: None)
    package = {"schema": "walking-stage-input-v2", "execution_id": config["execution_id"],
               "pilot_id": config["pilot_id"], "study_id": config["study_id"], "attempt_id": attempt_id,
               "run_id": run["run_id"], "cell_hash": run["cell_hash"], "code_hash": run["spec"]["code_hash"] if "spec" in run else store.manifest("study-" + run["study_id"])["spec"]["code_hash"],
               "config_hash": digest(config), "stage": stage, "arm_id": cell["arm_id"], "cell": cell,
               "deadline_monotonic": deadline, "authorization_expires_at": min(expiries,key=datetime.fromisoformat),
               "inputs": [i for i, _ in items], "lineage": lineage,
               "numeric_config": {"sources": [{k: s[k] for k in ("source_id", "role", "segment_ids") if k in s} for s in config["sources"]],
                                  "speed_limit": config["speed_limit"], "train_steps": config.get("train_steps", 100)},
               "input_refs": inputs}
    deadline=min(deadline,time.monotonic()+max(0,(datetime.fromisoformat(package["authorization_expires_at"])-datetime.now(timezone.utc)).total_seconds()))
    package["deadline_monotonic"]=deadline
    if len(encode(package))+sum(len(content) for _,content in items)>MAX_TOTAL:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "stage package total quota exceeded")
    root = Path(output).parent
    for item, content in items:
        atomic_write(root / item["relative_file"], content)
    package_path = root / "stage-input.json"
    atomic_write(package_path, encode(package))
    package_hash = hashlib.sha256(regular(package_path, 1024**2)).hexdigest()
    def verify(result=None, terminal=False):
        # Only called once native stop is confirmed. Revalidate original
        # authority and the actual original bytes, never the copy as authority.
        with store._read_transaction():
            preflight(store, config); current(terminal)
            for artifact in protected:
                if stage == "source":
                    store.verify_artifact_read(artifact, purpose=source["purpose"], authorization=grant, consumer=consumer)
                else:
                    reader = {"controller":True,"pilot_id":config["pilot_id"],"entrypoint":"walking-pilot-v1"} if terminal else consumer
                    _read_internal_bytes(store, config, artifact, reader)
            if read_package(package_path)[2] != package_hash:
                raise ResearchError("CHILD_ERROR_INVALID", "stage package changed")
            if result is not None and result.get("stage_binding") != result_binding(package, package_hash):
                raise ResearchError("CHILD_ERROR_INVALID", "worker result package identity differs")
            store._read_completion(lambda: current(terminal), lambda: None)
    return package_path, verify, deadline


def result_binding(package, package_hash):
    return {k: package[k] for k in ("execution_id", "attempt_id", "cell_hash")} | {"input_hash": package_hash}
