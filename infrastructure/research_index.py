"""Disposable SQLite projection of the authoritative research event chain."""

from __future__ import annotations

import json
import sqlite3

from .research_store import ResearchError, ResearchStore, encode, digest, identifier


class ResearchIndex:
    def __init__(self, store: ResearchStore):
        self.store = store
        self.path = store.path / "research.sqlite3"

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ResearchError("CONTRACT_MISMATCH", "unsupported research index version")
        except Exception:
            connection.close()
            raise
        return connection

    def rebuild(self):
        # No live connections are retained; store lock also serializes queries.
        with self.store.lock():
            events = self.store._events()
            try:
                connection = self._connect()
            except sqlite3.DatabaseError:
                # Preserve corrupt bytes for diagnosis, including any WAL.
                import uuid
                suffix = ".corrupt-" + uuid.uuid4().hex
                for extra in ("", "-wal", "-shm"):
                    candidate = self.path.with_name(self.path.name + extra)
                    if candidate.exists():
                        candidate.rename(candidate.with_name(candidate.name + suffix))
                connection = self._connect()
            try:
                with connection:
                    connection.executescript("""
                        BEGIN IMMEDIATE;
                        DROP TABLE IF EXISTS attempt_index;
                        DROP TABLE IF EXISTS run_index;
                        DROP TABLE IF EXISTS study_index;
                        DROP TABLE IF EXISTS objects;
                        DROP TABLE IF EXISTS watermark;
                        CREATE TABLE objects (object_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                                              manifest TEXT NOT NULL, sha256 TEXT NOT NULL);
                        CREATE TABLE watermark (singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                                                sequence INTEGER NOT NULL, event_hash TEXT NOT NULL);
                        CREATE TABLE study_index (study_id TEXT PRIMARY KEY, protocol_hash TEXT NOT NULL,
                                                  spec_hash TEXT NOT NULL UNIQUE);
                        CREATE TABLE run_index (run_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES study_index(study_id),
                                                arm_id TEXT NOT NULL, cell_hash TEXT NOT NULL, seed TEXT NOT NULL,
                                                UNIQUE(study_id, arm_id, cell_hash, seed));
                        CREATE TABLE attempt_index (attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES run_index(run_id),
                            parent_attempt_id TEXT REFERENCES attempt_index(attempt_id),
                            state TEXT NOT NULL CHECK(state IN ('REGISTERED','RUNNING','SUCCEEDED','FAILED','INTERRUPTED',
                                                               'TIMEOUT','BUDGET_EXHAUSTED','PREFLIGHT_FAILED','CANCELLED')),
                            started_at TEXT, ended_at TEXT, error_code TEXT, manifest_hash TEXT);
                        CREATE INDEX attempt_lookup ON attempt_index(run_id, state, attempt_id);
                        CREATE INDEX arm_lookup ON run_index(study_id, arm_id, run_id);
                        PRAGMA user_version=1;
                    """)
                    for event in events:
                        if event["event_kind"] == "MANIFEST":
                            payload = event["payload"]
                            # The complete chain was verified once under this
                            # same writer lock. Validate each manifest against
                            # that authoritative event, not a SQLite value or
                            # another whole-chain scan for every object.
                            object_id = identifier(payload["object_id"])
                            value = self.store._json(self.store.path / "manifests" / (object_id + ".json"))
                            if digest(value) != payload["sha256"]:
                                raise ResearchError("CORRUPT_ARTIFACT", "manifest hash mismatch")
                            connection.execute("INSERT INTO objects VALUES (?, ?, ?, ?)", (
                                payload["object_id"], payload["object_id"].split("-", 1)[0],
                                encode(value).decode(), payload["sha256"],
                            ))
                            if payload["object_id"].startswith("study-"):
                                connection.execute("INSERT INTO study_index VALUES (?, ?, ?)", (
                                    value["spec"]["study_id"], value["spec"]["protocol_hash"], value["spec_hash"]))
                            elif payload["object_id"].startswith("run-"):
                                connection.execute("INSERT INTO run_index VALUES (?, ?, ?, ?, ?)", (
                                    value["run_id"], value["study_id"], value["arm_id"], value["cell_hash"], str(value["seed"])))
                        elif event["event_kind"] == "ATTEMPT":
                            value = event["payload"]
                            connection.execute("""INSERT INTO attempt_index VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(attempt_id) DO UPDATE SET state=excluded.state, started_at=excluded.started_at,
                                ended_at=excluded.ended_at, error_code=excluded.error_code, manifest_hash=excluded.manifest_hash""", (
                                value["attempt_id"], value["run_id"], value["parent_attempt_id"], value["state"],
                                value["started_at"], value["ended_at"], value["error_code"], value["artifact_manifest_hash"]))
                    connection.execute("INSERT INTO watermark VALUES (1, ?, ?)", (
                        len(events), events[-1]["hash"] if events else "0" * 64,
                    ))
            finally:
                connection.close()
        return len(events)

    def list(self, *, kind="study", limit=50, after="", watermark=None):
        if not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ResearchError("CONTRACT_MISMATCH", "page limit must be 1..200")
        with self.store.lock():
            events = self.store._events()
            if not self.path.exists():
                raise ResearchError("INDEX_STALE", "index missing; rebuild required")
            try:
                connection = self._connect()
                try:
                    indexed = connection.execute("SELECT sequence,event_hash FROM watermark").fetchone()
                    expected = (len(events), events[-1]["hash"] if events else "0" * 64)
                    if indexed != expected:
                        raise ResearchError("INDEX_STALE", "index watermark behind authoritative events")
                    if watermark is not None and watermark != indexed[0]:
                        raise ResearchError("CURSOR_STALE", "event watermark changed")
                    rows = connection.execute(
                        "SELECT object_id,manifest FROM objects WHERE kind=? AND object_id>? ORDER BY object_id LIMIT ?",
                        (kind, after, limit + 1),
                    ).fetchall()
                finally:
                    connection.close()
            except sqlite3.DatabaseError as exc:
                raise ResearchError("INDEX_STALE", "index corrupt; rebuild required") from exc
            return {"items": [{"object_id": row[0], "manifest": json.loads(row[1])} for row in rows[:limit]],
                    "next_cursor": rows[limit - 1][0] if len(rows) > limit else None,
                    "watermark": indexed[0]}
