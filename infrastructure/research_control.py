"""Bounded file frames between the budget owner and its managed worker.

Stdlib-only so a numerical worker can poll without importing model packages.
This is a trusted-worker protocol, not an OS sandbox or a data authorization.
"""

import json
import math
import os
from pathlib import Path
import secrets
import time
import uuid


ENVIRONMENT = "PIRC25_WORKER_CONTROL"
SCHEMA = "pirc25-worker-control-v1"


class ControlError(ValueError):
    pass


def canonical(value, limit):
    remaining = 65536
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0 or depth > 32:
            raise ControlError("checkpoint frame exceeds structural quota")
        if type(item) is dict:
            if len(item) > remaining or any(type(key) is not str or len(key) > 128 for key in item):
                raise ControlError("checkpoint frame keys exceed quota")
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) in {list, tuple}:
            if len(item) > remaining:
                raise ControlError("checkpoint frame array exceeds quota")
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > min(limit, 65536):
                raise ControlError("checkpoint frame string exceeds quota")
        elif type(item) is int:
            if item.bit_length() > 256:
                raise ControlError("checkpoint frame integer exceeds quota")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ControlError("checkpoint frame must be finite")
        elif item is not None and type(item) is not bool:
            raise ControlError("checkpoint frame must be JSON")
    try:
        content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ControlError("checkpoint frame must be finite UTF-8 JSON") from exc
    if len(content) > limit:
        raise ControlError("checkpoint frame exceeds byte quota")
    return content


def read_frame(path, limit):
    if path.is_symlink():
        raise ControlError("checkpoint frame cannot be a symlink")
    try:
        with path.open("rb") as stream:
            content = stream.read(limit + 1)
    except FileNotFoundError:
        return None
    if len(content) > limit:
        raise ControlError("checkpoint frame exceeds byte quota")
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ControlError("checkpoint frame is invalid JSON") from exc
    if content != canonical(value, limit):
        raise ControlError("checkpoint frame must be canonical JSON")
    return value


def write_frame(path, value, limit):
    content = canonical(value, limit)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class CheckpointExchange:
    def __init__(self, directory, attempt_id, deadline, byte_limit):
        self.directory, self.attempt_id = Path(directory), attempt_id
        self.deadline, self.byte_limit = deadline, byte_limit
        self.token = secrets.token_hex(32)
        self.request_id = None
        self.accepted = False

    def environment(self):
        descriptor = {"schema_version": SCHEMA, "directory": str(self.directory), "attempt_id": self.attempt_id,
            "deadline": self.deadline, "byte_limit": self.byte_limit, "token": self.token}
        return {ENVIRONMENT: canonical(descriptor, 16384).decode()}

    def request(self):
        self.request_id = uuid.uuid4().hex
        value = {"schema_version": SCHEMA, "attempt_id": self.attempt_id, "token": self.token,
            "request_id": self.request_id, "deadline": self.deadline}
        write_frame(self.directory / "checkpoint-request.json", value, 16384)
        return self.request_id

    def response(self):
        if self.accepted:
            return None
        value = read_frame(self.directory / "checkpoint-response.json", self.byte_limit)
        if value is None:
            return None
        if (type(value) is not dict or set(value) != {"schema_version", "attempt_id", "token", "request_id", "state", "progress"}
                or value["schema_version"] != SCHEMA or value["attempt_id"] != self.attempt_id
                or value["token"] != self.token or self.request_id is None or value["request_id"] != self.request_id):
            raise ControlError("checkpoint response identity differs")
        return value

    def acknowledge(self, artifact_id):
        self.accepted = True
        write_frame(self.directory / "checkpoint-ack.json", {"schema_version": SCHEMA,
            "request_id": self.request_id, "artifact_id": artifact_id}, 16384)


class WorkerControl:
    def __init__(self, descriptor):
        if (type(descriptor) is not dict or set(descriptor) != {"schema_version", "directory", "attempt_id", "deadline", "byte_limit", "token"}
                or descriptor["schema_version"] != SCHEMA or type(descriptor["directory"]) is not str
                or type(descriptor["deadline"]) not in {float, int} or not math.isfinite(descriptor["deadline"])
                or type(descriptor["byte_limit"]) is not int or not 0 < descriptor["byte_limit"] <= 64 * 1024 * 1024):
            raise ControlError("checkpoint control descriptor is invalid")
        self.descriptor = descriptor
        self.directory = Path(descriptor["directory"])
        self.request = None

    @classmethod
    def from_environment(cls):
        content = os.environ.get(ENVIRONMENT)
        if content is None:
            return None
        if len(content) > 16384:
            raise ControlError("checkpoint descriptor exceeds quota")
        return cls(json.loads(content))

    def poll(self):
        value = read_frame(self.directory / "checkpoint-request.json", 16384)
        if value is None:
            return None
        if (type(value) is not dict or set(value) != {"schema_version", "attempt_id", "token", "request_id", "deadline"}
                or any(value[key] != self.descriptor[key] for key in ("schema_version", "attempt_id", "token", "deadline"))
                or type(value["request_id"]) is not str or len(value["request_id"]) != 32):
            raise ControlError("checkpoint request identity differs")
        self.request = value
        return value

    def save(self, state, progress):
        if self.request is None or time.monotonic() >= self.descriptor["deadline"]:
            raise ControlError("checkpoint save has no live request")
        frame = {key: self.request[key] for key in ("schema_version", "attempt_id", "token", "request_id")}
        write_frame(self.directory / "checkpoint-response.json", {**frame, "state": state, "progress": progress},
            self.descriptor["byte_limit"])
        while time.monotonic() < self.descriptor["deadline"]:
            ack = read_frame(self.directory / "checkpoint-ack.json", 16384)
            if ack is not None:
                if (type(ack) is not dict or set(ack) != {"schema_version", "request_id", "artifact_id"}
                        or ack["schema_version"] != SCHEMA or ack["request_id"] != self.request["request_id"]):
                    raise ControlError("checkpoint acknowledgement identity differs")
                return ack["artifact_id"]
            time.sleep(0.01)
        raise ControlError("checkpoint save was not acknowledged before deadline")
