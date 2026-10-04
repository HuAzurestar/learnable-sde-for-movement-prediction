"""Single-machine authoritative research manifests and hash-chained events.

SQLite is deliberately absent here. All mutations hold an OS lock, and every
publication is flushed before its atomic rename. Runtime roots are explicit,
outside Git, and reopening requires the original store identity.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid


_AUDIT_EVENT_KINDS = frozenset({
    "EXPOSURE_ALLOWED", "EXPOSURE_DENIED", "READ_STARTED", "READ_COMPLETED", "READ_FAILED",
    "DISCLOSURE_ALLOWED", "DISCLOSURE_DENIED",
})


class ResearchError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.safe_details = detail
        super().__init__(f"{code}: {detail}")

    def envelope(self, *, run_id=None, attempt_id=None):
        return {"code": self.code, "retryable": self.code == "BUDGET_BUSY", "trace_id": uuid.uuid4().hex,
                "run_id": run_id, "attempt_id": attempt_id, "safe_details": self.safe_details}


def encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ResearchError("CONTRACT_MISMATCH", "invalid object identity")
    return value


def authorization_key(authorization_id, version=None):
    """A missing selector refers only to the original unversioned grant."""
    authorization_id = identifier(authorization_id)
    if version is None:
        return "authorization-" + authorization_id
    version = identifier(version)
    # A separate namespace avoids ambiguity with arbitrary legacy IDs,
    # including IDs that already contain separators or version-looking text.
    return "grant-" + digest([authorization_id, version])


def authorization_key_for_grant(grant):
    if "version" in grant:
        identifier(grant["version"])
    return authorization_key(grant["authorization_id"], grant.get("version"))


def _sync_directory(path: Path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _matches_file_content(path: Path, content: bytes) -> bool:
    """Compare a regular target using one bounded, non-following file handle.

    Size checks are against the opened file, not a pre-open pathname stat. A
    growing file cannot turn this into an unbounded read. Nonblocking/no-follow
    flags also prevent a raced POSIX FIFO or symlink from becoming a read.
    """
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return False
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != len(content):
                return False
            equal = stream.read(len(content) + 1) == content
            after = os.fstat(stream.fileno())
            current = path.lstat()
            identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            return equal and identity(before) == identity(after) == identity(current)
    except FileNotFoundError:
        return False


def _publish_no_replace(temporary: Path, path: Path):
    # Windows rename refuses an existing destination; POSIX rename overwrites.
    # Hard-link publication is atomic/no-clobber on POSIX. Staging is beside
    # the target, on the same filesystem; unsupported operations fail closed.
    if os.name == "nt":
        os.rename(temporary, path)
        return False  # The staging pathname is no longer owned by this writer.
    else:
        os.link(temporary, path)
        return True  # The fully flushed staging link still needs cleanup.


def atomic_write(path: Path, content: bytes, *, before_replace=None, immutable=False):
    """Flush staging, optionally recheck disclosure, then atomically publish.

    Authority-store callers hold their lock and use the ordinary two-argument
    form. External evidence publication may supply a guard after fsync; a
    failed guard preserves any existing target and removes only this staging.
    Immutable external exports never replace another publisher's target;
    identical content is accepted only after a bounded read and a fresh guard.
    """
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if before_replace is not None:
            before_replace()
        if immutable:
            try:
                created = _publish_no_replace(temporary, path)
            except FileExistsError:
                if not _matches_file_content(path, content):
                    raise ResearchError("IDENTITY_CONFLICT", "export exists with different content") from None
                if before_replace is not None:
                    before_replace()
        else:
            os.replace(temporary, path)
            created = False
        _sync_directory(path.parent)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path, timeout: float = 15):
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        until = time.monotonic() + timeout
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= until:
                    raise ResearchError("WRITER_BUSY", "store writer lock unavailable")
                time.sleep(0.01)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)


class ResearchStore:
    SCHEMA = "pirc25-contract-v1"
    TERMINAL = {"SUCCEEDED", "FAILED", "INTERRUPTED", "TIMEOUT", "BUDGET_EXHAUSTED", "PREFLIGHT_FAILED", "CANCELLED"}

    def __init__(self, runtime_root: Path, store_id: str, *, initialize=False, allow_corrupt=False):
        root = Path(runtime_root)
        if not root.is_absolute():
            raise ResearchError("CONTRACT_MISMATCH", "runtime root must be absolute")
        root = root.resolve()
        if any((parent / ".git").exists() for parent in (root, *root.parents)):
            raise ResearchError("UNAUTHORIZED_DATA", "runtime root must be outside Git")
        self.path = root / "pirc25"
        self.store_id = identifier(store_id)
        self.writer_epoch = uuid.uuid4().hex
        self._read_scope = threading.local()
        if initialize:
            self.path.mkdir(parents=True, exist_ok=True)
        if not self.path.is_dir():
            raise ResearchError("STORE_MISSING", "explicit initialization required")
        with self.lock():
            identity_path = self.path / "store.json"
            expected = {"schema_version": self.SCHEMA, "store_id": self.store_id,
                        "runtime_root": str(root)}
            if identity_path.exists():
                if self._json(identity_path) != expected:
                    raise ResearchError("IDENTITY_CONFLICT", "store identity or root changed")
            elif initialize:
                if any(p.name != ".writer.lock" for p in self.path.iterdir()):
                    raise ResearchError("CORRUPT_ARTIFACT", "missing store identity in nonempty root")
                for name in ("events", "manifests", "artifacts"):
                    (self.path / name).mkdir()
                atomic_write(identity_path, encode(expected))
                atomic_write(self.path / "head.json", encode({"sequence": 0, "hash": "0" * 64}))
            else:
                raise ResearchError("STORE_MISSING", "store identity missing")
            if not allow_corrupt:
                self._events()

    def lock(self):
        if self._read_snapshot() is not None:
            return nullcontext()  # This thread still owns the outer OS lock.
        return file_lock(self.path / ".writer.lock")

    def _read_snapshot(self):
        snapshot = getattr(self._read_scope, 'snapshot', None)
        # A forked process cannot inherit permission to bypass its parent's lock.
        return snapshot if snapshot is not None and snapshot[0] == os.getpid() else None

    def _event_lookup(self):
        from .research_event_lookup import VerifiedEventLookup
        snapshot = self._read_snapshot()
        if snapshot is None or snapshot[1] is None:
            # No permission to reuse a prefix outside the owned OS lock scope.
            return VerifiedEventLookup(self._events())
        cached = getattr(self._read_scope, "event_lookup", None)
        if cached is None or cached[0] is not snapshot[1]:
            cached = (snapshot[1], VerifiedEventLookup(snapshot[1]))
            self._read_scope.event_lookup = cached
        return cached[1]

    def _discard_event_lookup(self):
        if hasattr(self._read_scope, "event_lookup"):
            del self._read_scope.event_lookup

    def _manifest_event(self, object_id, *, last=False):
        event = self._event_lookup().get(("manifest", object_id), last=last)
        return json.loads(encode(event)) if event is not None else None

    def _manifest_events(self, prefix):
        return json.loads(encode(self._event_lookup().manifests(prefix)))

    def _prior_read_events(self, identity, before):
        return json.loads(encode(self._event_lookup().prior_reads(identity, before)))

    @contextmanager
    def _read_transaction(self):
        if self._read_snapshot() is not None:
            yield
            return
        with self.lock():
            events = self._events()
            self._read_scope.snapshot = (os.getpid(), events)
            self._read_scope.completions = []
            completed = False
            try:
                yield
                completed = True
            finally:
                del self._read_scope.snapshot
                self._discard_event_lookup()
                # Also detects physical corruption during I/O. No content leaves
                # the outer request until this fresh, uncached validation passes.
                try:
                    events = self._events()
                    if completed and self._read_scope.completions:
                        # Share only this freshly verified physical prefix, not
                        # permission decisions. Every guard rehashes its actual
                        # manifests under the same outer OS lock.
                        self._read_scope.snapshot = (os.getpid(), events)
                        try:
                            checks = tuple(self._read_scope.completions)
                            for authority, _ in checks:
                                authority()
                            # An earlier grant may expire during a later guard's
                            # I/O. All pure expiry checks therefore follow ALL I/O.
                            for _, expiry in checks:
                                expiry()
                        finally:
                            del self._read_scope.snapshot
                            self._discard_event_lookup()
                finally:
                    del self._read_scope.completions

    def _read_completion(self, authority, expiry):
        """Register disclosure guards for the actual outer scope completion."""
        if self._read_snapshot() is None:
            raise ResearchError("CONTRACT_MISMATCH", "read completion needs an owned verified scope")
        self._read_scope.completions.append((authority, expiry))

    @staticmethod
    def _json(path):
        try:
            from .research_files import opened_regular_file
            from .research_json import read_json
            path = Path(path)
            # Authoritative metadata has the same lexical root and opened-file
            # boundary as artifacts. Do not follow replaced event/manifests
            # directories, even when their copied contents retain valid hashes.
            with opened_regular_file(path.parent, path) as (stream, size, verify):
                return read_json(stream, size, verify_identity=verify)
        except (OSError, ValueError) as exc:
            raise ResearchError("CORRUPT_ARTIFACT", "invalid authoritative object") from exc

    def _events(self) -> list[dict]:
        snapshot = self._read_snapshot()
        if snapshot is not None and snapshot[1] is not None:
            # Do not let a caller mutate the verified prefix or appended payloads.
            return json.loads(encode(snapshot[1]))
        events = []
        previous = "0" * 64
        for sequence, path in enumerate(sorted((self.path / "events").glob("*.json")), 1):
            if path.name != f"{sequence:016d}.json":
                raise ResearchError("CORRUPT_ARTIFACT", "event sequence gap")
            event = self._json(path)
            if not isinstance(event, dict):
                raise ResearchError("CORRUPT_ARTIFACT", "event is not an object")
            claimed = event.get("hash")
            body = {key: value for key, value in event.items() if key != "hash"}
            if (body.get("sequence") != sequence or body.get("previous_hash") != previous
                    or claimed != digest(body)):
                raise ResearchError("CORRUPT_ARTIFACT", "event hash chain mismatch")
            previous = claimed
            events.append(event)
        head = self._json(self.path / "head.json")
        n = head.get("sequence", -1) if isinstance(head, dict) else -1
        if (not isinstance(n, int) or n < 0 or n > len(events)
                or head.get("hash") != (events[n - 1]["hash"] if n else "0" * 64)):
            raise ResearchError("CORRUPT_ARTIFACT", "authoritative tail lost or changed")
        # A lagging head is safe: the fully flushed chain is the source of truth.
        return events

    def events(self) -> list[dict]:
        with self.lock():
            return self._events()

    def quarantine_tail(self, reason: str):
        """Preserve invalid bytes and put every budget under a permanent hold.

        This is an explicit forensic recovery operation, not permission to
        recalculate or replenish a budget from incomplete history.
        """
        if not reason.strip():
            raise ResearchError("CONTRACT_MISMATCH", "recovery requires a reason")
        with self.lock():
            if self._read_snapshot() is not None:
                # Forensic recovery may replace the verified prefix. Keep the
                # outer OS lock, but never append against the old read snapshot.
                self._read_scope.snapshot = (os.getpid(), None)
                self._discard_event_lookup()
            paths = sorted((self.path / "events").glob("*.json"))
            prefix = []
            previous = "0" * 64
            for sequence, path in enumerate(paths, 1):
                try:
                    value = self._json(path)
                    body = {key: item for key, item in value.items() if key != "hash"}
                    if (path.name != f"{sequence:016d}.json" or body.get("sequence") != sequence
                            or body.get("previous_hash") != previous or value.get("hash") != digest(body)):
                        break
                except (ResearchError, ValueError, AttributeError):
                    break
                prefix.append(value)
                previous = value["hash"]
            quarantine = self.path / ("quarantine-" + uuid.uuid4().hex)
            quarantine.mkdir()
            hold = {"reason": reason, "valid_prefix_sequence": len(prefix), "valid_prefix_hash": previous,
                    "quarantine": quarantine.name, "budget_action": "all-arms-held-no-replenishment"}
            # Written first: interruption during recovery can never reopen budget.
            atomic_write(self.path / "recovery-hold.json", encode(hold))
            head_path = self.path / "head.json"
            if head_path.exists():
                atomic_write(quarantine / "original-head.json", head_path.read_bytes())
            for path in paths[len(prefix):]:
                path.rename(quarantine / path.name)
            atomic_write(head_path, encode({"sequence": len(prefix), "hash": previous}))
            self._append("RECOVERY_HOLD", hold)
            return hold

    def _append(self, kind: str, payload: dict, event_id: str | None = None) -> dict:
        lookup = self._event_lookup()
        event_id = identifier(event_id or uuid.uuid4().hex)
        existing = lookup.get(("event", event_id))
        if existing is not None:
            if existing["event_kind"] != kind or existing["payload"] != payload:
                raise ResearchError("IDENTITY_CONFLICT", "event replay changed content")
            return json.loads(encode(existing))
        body = {"event_id": event_id, "sequence": lookup.size + 1,
                "previous_hash": lookup.last_hash,
                "event_kind": kind, "payload": payload, "created_at": utc_now(),
                "writer_epoch": self.writer_epoch}
        event = {**body, "hash": digest(body)}
        try:
            self._write_journal(kind, self.path / "events" / f"{event['sequence']:016d}.json", encode(event))
        except BaseException:
            if self._read_snapshot() is not None:
                # Rename may have succeeded before a directory-sync failure.
                # Keep the lock, but physically reread before any later append.
                self._read_scope.snapshot = (os.getpid(), None)
                self._discard_event_lookup()
            raise
        snapshot = self._read_snapshot()
        if snapshot is not None and snapshot[1] is not None:
            # The event is durable even if publishing head.json fails afterwards.
            try:
                detached = json.loads(encode(event))
                snapshot[1].append(detached)
                lookup.append(detached)
            except BaseException:
                self._read_scope.snapshot = (os.getpid(), None)
                self._discard_event_lookup()
                raise
        self._write_journal(kind, self.path / "head.json", encode({"sequence": event["sequence"], "hash": event["hash"]}))
        return event

    @staticmethod
    def _write_journal(kind: str, path: Path, content: bytes):
        try:
            atomic_write(path, content)
        except OSError as exc:
            if kind not in _AUDIT_EVENT_KINDS:
                raise
            # Never return data after uncertain publication. The existing
            # physical recovery retains any event published before this failure;
            # neither its disappearance nor a successful read can be assumed.
            raise ResearchError("EXPOSURE_AUDIT_UNAVAILABLE",
                                "mandatory exposure audit could not be persisted; no data returned") from exc

    def append(self, kind: str, payload: dict, event_id=None) -> dict:
        with self.lock():
            return self._append(kind, payload, event_id)

    def _publish(self, object_id: str, manifest: dict) -> str:
        identifier(object_id)
        path = self.path / "manifests" / f"{object_id}.json"
        if path.exists():
            if self._json(path) != manifest:
                raise ResearchError("IDENTITY_CONFLICT", "immutable object already exists")
        else:
            atomic_write(path, encode(manifest))
        return digest(manifest)

    def publish(self, object_id: str, manifest: dict) -> str:
        with self.lock():
            sha = self._publish(object_id, manifest)
            self._append("MANIFEST", {"object_id": object_id, "sha256": sha}, "manifest-" + digest([object_id, sha]))
            return sha

    def _manifest(self, object_id: str) -> dict:
        identifier(object_id)
        event = self._manifest_event(object_id, last=True)
        if event is None:
            raise ResearchError("MISSING_INPUT", "object not published")
        value = self._json(self.path / "manifests" / f"{object_id}.json")
        if digest(value) != event["payload"]["sha256"]:
            raise ResearchError("CORRUPT_ARTIFACT", "manifest hash mismatch")
        return value

    def manifest(self, object_id: str) -> dict:
        with self.lock():
            return self._manifest(object_id)

    def register(self, spec: dict, expected_hash: str) -> dict:
        if digest(spec) != expected_hash or spec.get("schema_version") != self.SCHEMA:
            raise ResearchError("CONTRACT_MISMATCH", "spec hash or schema mismatch")
        for key in ("study_id", "experiment_id", "comparison_family"):
            identifier(spec.get(key))
        for key in ("protocol_hash", "code_hash", "data_hash", "feature_hash", "selection_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get(key, ""))):
                raise ResearchError("CONTRACT_MISMATCH", f"invalid {key}")
        arms = spec.get("arms", [])
        if not arms or len({arm.get("arm_id") for arm in arms}) != len(arms):
            raise ResearchError("CONTRACT_MISMATCH", "missing or duplicate arms")
        for arm in arms:
            for key in ("arm_id", "model_family_id", "method_family_id", "objective_id"):
                identifier(arm.get(key))
            if arm.get("budget_seconds") != 86400:
                raise ResearchError("CONTRACT_MISMATCH", "arm budget must preserve 24-hour cap")
        families = [(arm["model_family_id"], arm["method_family_id"], arm["objective_id"]) for arm in arms]
        if len(set(families)) != len(families):
            raise ResearchError("IDENTITY_CONFLICT", "one arm family cannot receive multiple budget identities")
        if not spec.get("cells") or len({digest(c) for c in spec["cells"]}) != len(spec["cells"]):
            raise ResearchError("CONTRACT_MISMATCH", "missing or duplicate cells")
        if any(c.get("arm_id") not in {a["arm_id"] for a in arms} for c in spec["cells"]):
            raise ResearchError("CONTRACT_MISMATCH", "cell arm is not registered")
        object_id = "study-" + spec["study_id"]
        with self.lock():
            for event in self._events():
                if event["event_kind"] != "MANIFEST" or not event["payload"]["object_id"].startswith("study-"):
                    continue
                old = self._manifest(event["payload"]["object_id"])["spec"]
                for registered in old["arms"]:
                    for proposed in arms:
                        keys = ("model_family_id", "method_family_id", "objective_id")
                        same_family = all(registered[k] == proposed[k] for k in keys)
                        if (registered["arm_id"] == proposed["arm_id"]) != same_family:
                            raise ResearchError("IDENTITY_CONFLICT", "arm family cannot be relabeled to reset its budget")
            sha = self._publish(object_id, {"spec": spec, "spec_hash": expected_hash})
            self._append("MANIFEST", {"object_id": object_id, "sha256": sha}, "manifest-" + digest([object_id, sha]))
        return {"study_id": spec["study_id"], "experiment_id": spec["experiment_id"], "spec_hash": expected_hash}

    def register_run(self, study_id: str, cell: dict) -> str:
        with self.lock():
            study = self._manifest("study-" + identifier(study_id))
            if cell not in study["spec"]["cells"]:
                raise ResearchError("CONTRACT_MISMATCH", "cell was not preregistered")
            run = {"study_id": study_id, "experiment_id": study["spec"]["experiment_id"],
                   "spec_hash": study["spec_hash"], "arm_id": cell["arm_id"],
                   "cell_hash": digest(cell), "seed": cell.get("seed"), "cell": cell}
            run_id = digest(run)
            object_id = "run-" + run_id
            sha = self._publish(object_id, {**run, "run_id": run_id})
            self._append("MANIFEST", {"object_id": object_id, "sha256": sha}, "manifest-" + digest([object_id, sha]))
            return run_id

    def new_attempt(self, run_id: str, *, parent_attempt_id=None, reason=None) -> str:
        with self.lock():
            self._manifest("run-" + identifier(run_id))
            states = self._attempts()
            related = [attempt for attempt in states.values() if attempt["run_id"] == run_id]
            if any(a["state"] == "SUCCEEDED" for a in related):
                raise ResearchError("IDENTITY_CONFLICT", "successful cell is immutable and reusable")
            if any(a["state"] not in self.TERMINAL for a in related):
                raise ResearchError("IDENTITY_CONFLICT", "cell already has a live attempt")
            if related:
                if len(related) >= 3:
                    raise ResearchError("RETRY_EXHAUSTED", "at most two explicit retries are supported")
                parent = states.get(parent_attempt_id)
                if not reason or not parent or parent["run_id"] != run_id:
                    raise ResearchError("CONTRACT_MISMATCH", "retry requires failed parent and reason")
            elif parent_attempt_id is not None:
                raise ResearchError("CONTRACT_MISMATCH", "retry parent not found")
            attempt_id = uuid.uuid4().hex
            self._append("ATTEMPT", {"attempt_id": attempt_id, "run_id": run_id,
                         "parent_attempt_id": parent_attempt_id, "reason": reason, "state": "REGISTERED",
                         "started_at": None, "ended_at": None, "error_code": None,
                         "artifact_manifest_hash": None})
            return attempt_id

    def _attempts(self) -> dict:
        attempts = {}
        for event in self._events():
            if event["event_kind"] == "ATTEMPT":
                payload = event["payload"]
                attempts[payload["attempt_id"]] = payload
        return attempts

    def attempts(self) -> dict:
        with self.lock():
            return self._attempts()

    def transition(self, attempt_id: str, state: str, *, error_code=None, artifact_id=None):
        with self.lock():
            return self._transition(attempt_id, state, error_code=error_code, artifact_id=artifact_id)

    def _transition(self, attempt_id: str, state: str, *, error_code=None, artifact_id=None):
        attempts = self._attempts()
        if attempt_id not in attempts:
            raise ResearchError("MISSING_INPUT", "attempt not registered")
        current = attempts[attempt_id]
        if current["state"] in self.TERMINAL:
            if current["state"] == state and current.get("artifact_id") == artifact_id and current["error_code"] == error_code:
                return current
            raise ResearchError("IDENTITY_CONFLICT", "attempt terminal state cannot change")
        if state not in self.TERMINAL | {"RUNNING"} or (state == "RUNNING" and current["state"] != "REGISTERED"):
            raise ResearchError("CONTRACT_MISMATCH", "invalid attempt transition")
        artifact_hash = None
        if state == "SUCCEEDED":
            if current["state"] != "RUNNING" or artifact_id is None or error_code is not None:
                raise ResearchError("CONTRACT_MISMATCH", "success needs running attempt and published artifact")
            metadata = self._manifest("artifact-" + identifier(artifact_id))
            run = self._manifest("run-" + current["run_id"])
            if metadata["study_id"] != run["study_id"]:
                raise ResearchError("CONTRACT_MISMATCH", "artifact belongs to another study")
            self._verified_artifact_content(metadata)
            artifact_hash = digest(metadata)
        updated = {**current, "state": state, "error_code": error_code,
                   "artifact_id": artifact_id, "artifact_manifest_hash": artifact_hash,
                   "started_at": utc_now() if state == "RUNNING" else current["started_at"],
                   "ended_at": utc_now() if state in self.TERMINAL else None}
        self._append("ATTEMPT", updated)
        return updated

    def authorize(self, authorization: dict):
        """Explicit local-owner operation, never exposed by the read-only API."""
        identifier(authorization.get("authorization_id"))
        identifier(authorization.get("study_id"))
        if (not authorization.get("purposes") or not authorization.get("visibilities")
                or not isinstance(authorization.get("block_ids"), list)
                or not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get("evidence_hash", "")))):
            raise ResearchError("CONTRACT_MISMATCH", "authorization scope/evidence missing")
        try:
            expiry = datetime.fromisoformat(authorization["expires_at"])
            if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
                raise ValueError("expired")
        except (KeyError, ValueError, TypeError) as exc:
            raise ResearchError("UNAUTHORIZED_DATA", "authorization expired or missing expiry") from exc
        self.publish(authorization_key_for_grant(authorization), authorization)

    def authorization(self, authorization_id, *, version=None):
        key = authorization_key(authorization_id, version)
        grant = self.manifest(key)
        if authorization_key_for_grant(grant) != key:
            raise ResearchError("CONTRACT_MISMATCH", "authorization version binding differs")
        return grant

    def artifact(self, content: bytes, *, role: str, visibility: str, block_ids: list[str], study_id: str,
                 media_type="application/json") -> dict:
        if visibility not in {"synthetic", "restricted", "public"}:
            raise ResearchError("CONTRACT_MISMATCH", "unknown visibility")
        sha = hashlib.sha256(content).hexdigest()
        metadata = {"artifact_id": sha, "sha256": sha, "size_bytes": len(content), "study_id": identifier(study_id),
                    "media_type": media_type, "role": role,
                    "visibility": visibility, "block_ids": sorted(set(block_ids))}
        with self.lock():
            path = self.path / "artifacts" / sha
            if path.exists():
                self._verified_artifact_content(metadata)
            else:
                atomic_write(path, content)
            object_id = "artifact-" + sha
            manifest_hash = self._publish(object_id, metadata)
            self._append("MANIFEST", {"object_id": object_id, "sha256": manifest_hash}, "manifest-" + digest([object_id, manifest_hash]))
        return metadata

    def _verified_artifact_content(self, metadata):
        """Integrity primitive; caller holds authority lock and controls access."""
        artifact_id = metadata["artifact_id"]
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise ResearchError("CORRUPT_ARTIFACT", "artifact identity is invalid")
        size = metadata.get("size_bytes")
        if type(size) is not int or size < 0 or metadata.get("sha256") != artifact_id:
            raise ResearchError("CORRUPT_ARTIFACT", "artifact frozen size or hash is invalid")
        from .research_files import opened_regular_file
        # Keep the established pathname: resolving it here would bless a
        # directory symlink installed after the disclosure journal as a root.
        root = self.path / "artifacts"
        path = root / artifact_id
        try:
            with opened_regular_file(root, path, expected_size=size) as (stream, _, _):
                # A changed file cannot turn a previously admitted small
                # artifact into an unbounded read. The extra byte detects
                # growth even after the open-handle size check.
                content = stream.read(size + 1)
        except FileNotFoundError as exc:
            raise ResearchError("MISSING_INPUT", "published artifact bytes are missing") from exc
        if (len(content) != size
                or hashlib.sha256(content).hexdigest() != artifact_id):
            raise ResearchError("CORRUPT_ARTIFACT", "artifact hash mismatch")
        return content

    def _artifact_read_allowed(self, metadata, purpose, authorization):
        """Fresh authority, not an authorization cached by the read scope."""
        if not isinstance(authorization, dict):
            return False
        try:
            grant_id = authorization_key_for_grant(authorization)
            grant = self._manifest(grant_id)
            allowed = (self._manifest("artifact-" + metadata["artifact_id"]) == metadata
                       and grant == authorization and metadata["study_id"] == grant["study_id"]
                       and purpose in {"preview", "export", "evaluate", "resume"}
                       and purpose in grant["purposes"]
                       and metadata["visibility"] in grant["visibilities"]
                       and set(metadata["block_ids"]) <= set(grant["block_ids"]))
        except (ResearchError, KeyError, TypeError, ValueError):
            return False
        if allowed and metadata["role"] != "qualification":
            # Old derived artifacts may claim synthetic visibility. Retain the
            # authoritative source-lineage check and the standalone owner case.
            try:
                study = self._manifest("study-" + metadata["study_id"])
            except ResearchError as exc:
                if exc.code != "MISSING_INPUT":
                    raise
                study = None
            if study is not None:
                from infrastructure.research_visibility import study_visibility
                allowed = study_visibility(self._manifest, study["spec"]) in grant["visibilities"]
        try:
            # Lineage reads can themselves outlive or invalidate this grant.
            return (allowed and self._manifest(grant_id) == grant
                    and datetime.fromisoformat(grant["expires_at"]) > datetime.now(timezone.utc))
        except (ResearchError, KeyError, TypeError, ValueError):
            return False

    def _require_artifact_read(self, metadata, purpose, authorization, request):
        if not self._artifact_read_allowed(metadata, purpose, authorization):
            self._append("EXPOSURE_DENIED", {**request, "allowed": False})
            raise ResearchError("UNAUTHORIZED_DATA", "artifact disclosure not authorized")

    def _require_artifact_expiry(self, authorization, request):
        # No further successful I/O follows this check. The outer scope has
        # already performed its uncached physical event-chain verification.
        try:
            unexpired = datetime.fromisoformat(authorization["expires_at"]) > datetime.now(timezone.utc)
        except (KeyError, TypeError, ValueError):
            unexpired = False
        if not unexpired:
            self.append("EXPOSURE_DENIED", {**request, "allowed": False})
            raise ResearchError("UNAUTHORIZED_DATA", "artifact disclosure not authorized")

    def verify_artifact_read(self, artifact_id: str, *, purpose: str, authorization: dict):
        """Recheck a consumer's final disclosure after its extra source I/O."""
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise ResearchError("UNAUTHORIZED_DATA", "invalid artifact ID")
        request = {"artifact_id": artifact_id, "purpose": purpose,
                   "authorization_hash": digest(authorization), "allowed": True}
        with self._read_transaction():
            metadata = self._manifest("artifact-" + artifact_id)
            self._require_artifact_read(metadata, purpose, authorization, request)
            self._read_completion(
                lambda: self._require_artifact_read(metadata, purpose, authorization, request),
                lambda: self._require_artifact_expiry(authorization, request))
        self._require_artifact_expiry(authorization, request)

    def read_artifact(self, artifact_id: str, *, purpose: str, authorization: dict) -> bytes:
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise ResearchError("UNAUTHORIZED_DATA", "invalid artifact ID")
        with self._read_transaction():
            metadata = self._manifest("artifact-" + artifact_id)
            allowed = self._artifact_read_allowed(metadata, purpose, authorization)
            request = {"artifact_id": artifact_id, "purpose": purpose,
                       "authorization_hash": digest(authorization), "allowed": allowed}
            self._append("EXPOSURE_ALLOWED" if allowed else "EXPOSURE_DENIED", request)
            if not allowed:
                raise ResearchError("UNAUTHORIZED_DATA", "artifact disclosure not authorized")
            self._require_artifact_read(metadata, purpose, authorization, request)
            self._append("READ_STARTED", request)
            self._require_artifact_read(metadata, purpose, authorization, request)
            try:
                content = self._verified_artifact_content(metadata)
            except (OSError, ResearchError):
                self._append("READ_FAILED", request)
                raise
            self._append("READ_COMPLETED", request)
            self._require_artifact_read(metadata, purpose, authorization, request)
            self._read_completion(
                lambda: self._require_artifact_read(metadata, purpose, authorization, request),
                lambda: self._require_artifact_expiry(authorization, request))
        self._require_artifact_expiry(authorization, request)
        return content
