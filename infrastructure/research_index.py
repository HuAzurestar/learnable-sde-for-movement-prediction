"""Disposable SQLite projection of the authoritative research event chain."""

from __future__ import annotations

import json
import sqlite3

from .research_store import ResearchError, ResearchStore, encode, digest, identifier


_READ_KINDS = ('READ_STARTED', 'READ_COMPLETED', 'READ_FAILED')
_EXPOSURE_KINDS = _READ_KINDS + ('EXPOSURE_ALLOWED', 'EXPOSURE_DENIED')


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
            if version not in (0, 1, 2):
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
                        DROP TABLE IF EXISTS exposure_blocks;
                        DROP TABLE IF EXISTS exposure_events;
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
                        CREATE TABLE exposure_events (
                            sequence INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE,
                            event_kind TEXT NOT NULL, event_json TEXT NOT NULL);
                        CREATE TABLE exposure_blocks (
                            block_id TEXT NOT NULL, sequence INTEGER NOT NULL REFERENCES exposure_events(sequence),
                            PRIMARY KEY(block_id, sequence));
                        PRAGMA user_version=2;
                    """)
                    artifact_scopes = {}
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
                            elif payload['object_id'].startswith('artifact-'):
                                # Old disclosure events have only an artifact
                                # ID. Use its actual hash-verified manifest,
                                # never its grant, to reconstruct block scope.
                                blocks = value.get('block_ids')
                                artifact_id = payload['object_id'][len('artifact-'):]
                                if (value.get('artifact_id') != artifact_id or not isinstance(blocks, list)
                                        or any(not isinstance(block, str) or not block for block in blocks)):
                                    raise ResearchError('CORRUPT_ARTIFACT', 'artifact exposure scope malformed')
                                artifact_scopes[artifact_id] = blocks
                        elif event["event_kind"] == "ATTEMPT":
                            value = event["payload"]
                            connection.execute("""INSERT INTO attempt_index VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(attempt_id) DO UPDATE SET state=excluded.state, started_at=excluded.started_at,
                                ended_at=excluded.ended_at, error_code=excluded.error_code, manifest_hash=excluded.manifest_hash""", (
                                value["attempt_id"], value["run_id"], value["parent_attempt_id"], value["state"],
                                    value["started_at"], value["ended_at"], value["error_code"], value["artifact_manifest_hash"]))
                    # A single verified prefix and one manifest pass suffice:
                    # no per-event rescans or SQLite-as-authority lookups.
                    for event in events:
                        if event['event_kind'] not in _EXPOSURE_KINDS:
                            continue
                        payload = event['payload']
                        if not isinstance(payload, dict):
                            raise ResearchError('CORRUPT_ARTIFACT', 'exposure event scope malformed')
                        blocks = {payload[key] for key in ('block_id', 'source_block_id')
                                  if isinstance(payload.get(key), str) and payload[key]}
                        if payload.get('artifact_id'):
                            try:
                                blocks.update(artifact_scopes[payload['artifact_id']])
                            except (KeyError, TypeError) as exc:
                                raise ResearchError('CORRUPT_ARTIFACT', 'artifact exposure scope unresolved') from exc
                        connection.execute('INSERT INTO exposure_events VALUES (?, ?, ?, ?)', (
                            event['sequence'], event['event_id'], event['event_kind'], encode(event).decode()))
                        connection.executemany('INSERT INTO exposure_blocks VALUES (?, ?)', (
                            (block, event['sequence']) for block in sorted(blocks)))
                    connection.execute("INSERT INTO watermark VALUES (1, ?, ?)", (
                        len(events), events[-1]["hash"] if events else "0" * 64,
                    ))
            finally:
                connection.close()
        return len(events)

    def exposures(self, *, block_id, limit=50, after=0, before=None, watermark=None, include_checks=False):
        """Page original read events by declared block/source-block scope.

        Optionally include allow/deny checks. This is audit metadata only: an
        empty disposable projection is NOT authorization or proof of blindness.
        Source/content aliases and imported external history remain the data
        gate's responsibility. Schema1 list queries stay compatible, but its
        missing exposure projection requires an explicit rebuild.
        """
        identifier(block_id)
        if (type(limit) is not int or not 1 <= limit <= 200 or type(after) is not int or after < 0
                or (before is not None and (type(before) is not int or before < 1))
                or type(include_checks) is not bool):
            raise ResearchError('CONTRACT_MISMATCH', 'invalid exposure range or page limit')
        with self.store.lock():
            events = self.store._events()
            if not self.path.exists():
                raise ResearchError('INDEX_STALE', 'index missing; rebuild required')
            try:
                connection = self._connect()
                try:
                    if connection.execute('PRAGMA user_version').fetchone()[0] != 2:
                        raise ResearchError('INDEX_STALE', 'exposure projection missing; rebuild required')
                    indexed = connection.execute('SELECT sequence,event_hash FROM watermark').fetchone()
                    expected = (len(events), events[-1]['hash'] if events else '0' * 64)
                    if indexed != expected:
                        raise ResearchError('INDEX_STALE', 'index watermark behind authoritative events')
                    if watermark is not None and watermark != indexed[0]:
                        raise ResearchError('CURSOR_STALE', 'event watermark changed')
                    kinds = _EXPOSURE_KINDS if include_checks else _READ_KINDS
                    rows = connection.execute(
                        'SELECT e.sequence,e.event_json FROM exposure_blocks b '
                        'JOIN exposure_events e ON e.sequence=b.sequence '
                        'WHERE b.block_id=? AND b.sequence>? AND b.sequence<? '
                        'AND e.event_kind IN (' + ','.join('?' for _ in kinds) + ') '
                        'ORDER BY b.sequence LIMIT ?',
                        (block_id, after, before if before is not None else len(events) + 1, *kinds, limit + 1),
                    ).fetchall()
                    selected = []
                    manifest_lookup = None
                    for sequence, event_json in rows:
                        event = json.loads(event_json)
                        if (not 1 <= sequence <= len(events)
                                or event_json.encode('utf-8') != encode(events[sequence - 1])):
                            raise ResearchError('INDEX_STALE', 'indexed exposure differs from original event')
                        payload = event['payload']
                        blocks = {payload[key] for key in ('block_id', 'source_block_id')
                                  if isinstance(payload.get(key), str) and payload[key]}
                        if payload.get('artifact_id'):
                            # A forged edge can point to an authentic event.
                            # Verify scope too, using current physical metadata
                            # bound to this already verified authoritative log.
                            # Do not rescan that log per selected output row.
                            if manifest_lookup is None:
                                from .research_event_lookup import VerifiedEventLookup
                                manifest_lookup = VerifiedEventLookup(events)
                            object_id = 'artifact-' + identifier(payload['artifact_id'])
                            source = manifest_lookup.get(('manifest', object_id), last=True)
                            if source is None:
                                raise ResearchError('CORRUPT_ARTIFACT', 'artifact exposure scope unresolved')
                            value = self.store._json(self.store.path / 'manifests' / (object_id + '.json'))
                            if digest(value) != source['payload']['sha256']:
                                raise ResearchError('CORRUPT_ARTIFACT', 'artifact exposure scope hash differs')
                            scoped = value.get('block_ids')
                            if (not isinstance(scoped, list)
                                    or any(not isinstance(block, str) or not block for block in scoped)):
                                raise ResearchError('CORRUPT_ARTIFACT', 'artifact exposure scope malformed')
                            blocks.update(scoped)
                        if block_id not in blocks:
                            raise ResearchError('INDEX_STALE', 'indexed block differs from original event scope')
                        selected.append(event)
                finally:
                    connection.close()
            except ResearchError:
                # ResearchError subclasses ValueError. Preserve authoritative
                # CORRUPT_ARTIFACT/CURSOR_STALE rather than relabeling the source
                # failure as damage to a disposable projection.
                raise
            except (sqlite3.DatabaseError, ValueError, TypeError) as exc:
                raise ResearchError('INDEX_STALE', 'index corrupt; rebuild required') from exc
            return {'items': selected[:limit],
                    'next_cursor': rows[limit - 1][0] if len(rows) > limit else None,
                    'watermark': indexed[0],
                    'scope_contract': 'declared-block-source-block-and-artifact-manifest',
                    'permission_decision': 'not-provided'}

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
