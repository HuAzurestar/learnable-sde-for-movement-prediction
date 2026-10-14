"""Small strict identity/immutable-publication primitives for the formal seal."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit lowercase SHA256 identity required")
    return value


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "\x00" in value or any(part in {"", ".", ".."} for part in value.split("/"))
            or PurePosixPath(value).is_absolute()):
        raise ValueError("canonical relative path required")
    return value


def under(root, relative):
    root = Path(root).resolve()
    candidate = root.joinpath(*relative_path(relative).split("/"))
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
        raise ValueError("bound path escapes its logical root")
    return candidate


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in sealed evidence")
        result[key] = value
    return result


def decode(raw):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant is forbidden")
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number is forbidden")
        return result
    return json.loads(raw, object_pairs_hook=_object, parse_constant=reject_constant, parse_float=finite_float)


def read_json(path, *, expected_file_sha256=None, max_bytes=32*1024*1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("bounded regular JSON file required")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError("JSON file grew beyond its bound")
    if expected_file_sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256(expected_file_sha256):
        raise ValueError("bound JSON file hash mismatch")
    return decode(raw)


def envelope(payload):
    return {"payload": payload, "sha256": digest(payload)}


def unpack(value, *, expected_sha256=None):
    if (not isinstance(value, dict) or set(value) != {"payload", "sha256"}
            or not isinstance(value["payload"], dict) or digest(value["payload"]) != sha256(value["sha256"])
            or (expected_sha256 is not None and value["sha256"] != sha256(expected_sha256))):
        raise ValueError("sealed content identity mismatch")
    return value["payload"]


def publish(directory, payload):
    """Content-addressed, exclusive write; never edit/overwrite an old seal.

An interrupted partial publication is invalid and stays visible for recovery.
No verifier treats mere filename/existence or an incomplete JSON as a seal.
"""
    record = envelope(payload)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (record["sha256"] + ".json")
    with path.open("xb") as stream:
        stream.write(canonical(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path, record
