"""Explicit owner-authorized observed-state block reads and causal decoding.

Raw positions/contexts never enter logs. The shared exposure ledger is the
only read authority; a decoded DTO is not permission or model qualification.
No test data is read by this module merely being imported.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from application.research_data import EvaluationExposureLedger
from application.research_preregistration import source_identity
from infrastructure.research_store import ResearchError, digest


MAX_BYTES = 8 * 1024 * 1024
PURPOSES = {"train": "fit", "selection": "select", "validation": "validate", "test": "evaluate", "final-eval": "evaluate"}


def read_block(store, protocol_id, block_id, *, authorization_id, purpose, consumer=None, authorization_version=None):
    """Owner-only read; verifies frozen size before materializing bounded bytes.

    Caller still needs complete AdmissionGate/registry/budget preflight before
    a job is dispatched. This read does not grant execute or disclosure rights.
    """
    protocol = store.manifest("protocol-" + protocol_id)
    matches = [b for b in protocol["blocks"] if b["block_id"] == block_id]
    if len(matches) != 1:
        raise ResearchError("UNAUTHORIZED_DATA", "requested block is absent from frozen protocol")
    block = matches[0]
    if type(block.get("size_bytes")) is not int or not 0 < block["size_bytes"] <= MAX_BYTES:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "phase-space JSON block needs a bounded frozen byte size")
    grant = store.authorization(authorization_id, version=authorization_version)
    root = grant.get("data_root")
    if type(root) is not str or not Path(root).is_absolute():
        raise ResearchError("UNAUTHORIZED_DATA", "explicit authorized data root required")
    ledger = EvaluationExposureLedger(store)
    arguments = {"purpose": purpose, "authorization_id": authorization_id, "authorization_version": authorization_version,
                 "data_root": Path(root), "consumer": consumer}
    metadata = ledger.verify(protocol_id, block_id, **arguments)
    if PURPOSES.get(block["split_role"]) != purpose:
        raise ResearchError("UNAUTHORIZED_DATA", "phase-space data purpose differs from source role")
    if metadata["size_bytes"] != block["size_bytes"]:
        raise ResearchError("CORRUPT_ARTIFACT", "actual phase-space block size differs from frozen input")
    content = ledger.read(protocol_id, block_id, **arguments)
    if len(content) != block["size_bytes"] or hashlib.sha256(content).hexdigest() != block["sha256"]:
        raise ResearchError("CORRUPT_ARTIFACT", "phase-space input changed across authorized reads")
    try:
        decoded = content.decode("utf-8")
    except UnicodeError as exc:
        raise ResearchError("CONTRACT_MISMATCH", "phase-space input must be UTF-8 JSON") from exc
    return {"schema_version": "pirc26-admitted-block-v1", "protocol_hash": digest(protocol),
            "block_id": block_id, "source_identity": source_identity(block), "split_role": block["split_role"],
            "purpose": purpose, "content_sha256": block["sha256"], "content_utf8": decoded}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResearchError("CONTRACT_MISMATCH", "duplicate field in observed-state input")
        result[key] = value
    return result


def _bounded_document(content):
    if type(content) is not str or len(content.encode("utf-8")) > MAX_BYTES:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "observed-state JSON byte quota")
    try:
        document = json.loads(content, object_pairs_hook=_unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        remaining, stack = 1048576, [(document, 0)]
        while stack:
            item, depth = stack.pop()
            remaining -= 1
            if remaining < 0 or depth > 12:
                raise ResearchError("RESOURCE_PLAN_REJECTED", "observed-state JSON structural quota")
            if type(item) is dict:
                stack.extend((child, depth + 1) for child in item.values())
            elif type(item) is list:
                stack.extend((child, depth + 1) for child in item)
        return document
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("CONTRACT_MISMATCH", "invalid finite observed-state JSON") from exc


def _matrix(value, rows, columns):
    return (type(value) is list and len(value) == rows
            and all(type(row) is list and len(row) == columns
                    and all(type(v) in (int, float) and math.isfinite(v) for v in row) for row in value))


class ObservedBlock:
    """Numerical worker DTO with past-only velocity/context construction."""

    def __init__(self, admission, document, model):
        import torch
        from domain import ModelContext
        self.torch, self.ModelContext = torch, ModelContext
        self.model, self.role, self.purpose = model, admission["split_role"], admission["purpose"]
        self.source_identity, self.block_id = admission["source_identity"], admission["block_id"]
        self.content_hash = admission["content_sha256"]
        self.velocity_source = document["velocity_source"]
        self.segments = []
        dtype, device = model.velocity_factor.dtype, model.velocity_factor.device
        for segment in document["segments"]:
            count = len(segment["time"])
            time = torch.tensor(segment["time"], dtype=dtype, device=device)
            position = torch.tensor(segment["position"], dtype=dtype, device=device)
            # Also reject collapsing timestamps after the declared dtype cast.
            if not torch.isfinite(time).all() or not torch.isfinite(position).all() or not (time.diff() > 0).all():
                raise ResearchError("CONTRACT_MISMATCH", "physical observations invalid after dtype conversion")
            if self.velocity_source == "backward-difference-v1":
                velocity = (position[1:] - position[:-1]) / time.diff()[:, None]
                first = 1
                state = torch.cat((position[1:], velocity), -1)
            else:
                velocity = torch.tensor(segment["velocity"], dtype=dtype, device=device)
                first = 0
                state = torch.cat((position, velocity), -1)
            condition = torch.tensor(segment["condition"], dtype=dtype, device=device).reshape(count, model.spec.context_dim)
            if not torch.isfinite(state).all() or not torch.isfinite(condition).all():
                raise ResearchError("NONFINITE", "physical state/context construction")
            self.segments.append({"segment_id": segment["segment_id"], "original_time": tuple(segment["time"]),
                "time": time[first:], "state": state, "condition": condition[first:], "first_original_index": first})

    def transitions(self, *, batch_size):
        from estimation.phase_space import TransitionBatch
        if self.role != "train" or self.purpose != "fit":
            raise ResearchError("UNAUTHORIZED_DATA", "only admitted train blocks construct fitting transitions")
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "bounded O1 batch size required")
        batches = []
        for segment in self.segments:
            t, z, c = segment["time"], segment["state"], segment["condition"]
            for first in range(0, len(t) - 1, batch_size):
                last = min(len(t) - 1, first + batch_size)
                context = self.ModelContext(c[first:last])
                batches.append(TransitionBatch(t[first:last], z[first:last], z[first+1:last+1],
                    t[first+1:last+1] - t[first:last], context, self.model.spec.train_binding_hash))
                if len(batches) > 256:
                    raise ResearchError("RESOURCE_PLAN_REJECTED", "O1 decoded batch quota")
        return batches

    def forecast_request(self, segment_id, original_origin_index, time_grid, *, sample_count, brownian_root_id, chunk_size=256):
        from inference.phase_space import ForecastRequest
        selected = [s for s in self.segments if s["segment_id"] == segment_id]
        if len(selected) != 1 or type(original_origin_index) is not int:
            raise ResearchError("CONTRACT_MISMATCH", "explicit forecast origin is absent")
        segment = selected[0]
        index = original_origin_index - segment["first_original_index"]
        if not 0 <= index < len(segment["state"]):
            raise ResearchError("CONTRACT_MISMATCH", "origin lacks a causal observed velocity")
        origin = segment["original_time"][original_origin_index]
        if not time_grid or time_grid[0] != origin:
            raise ResearchError("CONTRACT_MISMATCH", "forecast grid must start at the known observed origin")
        req = ForecastRequest(tuple(segment["state"][index].tolist()), tuple(time_grid), origin, sample_count,
                              brownian_root_id, tuple(segment["condition"][index].tolist()), chunk_size)
        req.validate(self.model)
        return req

    def truth(self, segment_id, request):
        # This DTO is constructed only after the owner's role-appropriate read.
        # Truth is returned separately, never passed into forecast or a context.
        segment = next((s for s in self.segments if s["segment_id"] == segment_id), None)
        if segment is None:
            raise ResearchError("CONTRACT_MISMATCH", "truth segment is absent")
        indices = []
        for time in request.time_grid:
            try:
                index = segment["original_time"].index(time) - segment["first_original_index"]
            except ValueError as exc:
                raise ResearchError("CONTRACT_MISMATCH", "truth is not observed on the registered grid") from exc
            if index < 0:
                raise ResearchError("CONTRACT_MISMATCH", "truth velocity unavailable at initial observation")
            indices.append(index)
        return segment["state"][indices].clone()


def decode_block(admission, model, *, max_observations=1048576):
    """Validate the authorized transport before allocating numerical tensors."""
    from models.phase_space import PhaseSpaceSDE
    if type(max_observations) is not int or not 1 <= max_observations <= 1048576:
        raise ResearchError("RESOURCE_PLAN_REJECTED", "explicit bounded observation capacity required")
    fields = {"schema_version", "protocol_hash", "block_id", "source_identity", "split_role", "purpose", "content_sha256", "content_utf8"}
    if (type(admission) is not dict or set(admission) != fields or admission["schema_version"] != "pirc26-admitted-block-v1"
            or not isinstance(model, PhaseSpaceSDE) or PURPOSES.get(admission["split_role"]) != admission["purpose"]
            or re.fullmatch(r"[0-9a-f]{64}", str(admission["protocol_hash"])) is None
            or type(admission["source_identity"]) is not dict
            or set(admission["source_identity"]) != {"dataset_id", "release_id", "source_block_id", "sha256"}
            or any(type(v) is not str or not 1 <= len(v) <= 128 for v in admission["source_identity"].values())
            or admission["source_identity"]["sha256"] != admission["content_sha256"]
            or type(admission["content_utf8"]) is not str
            or hashlib.sha256(admission["content_utf8"].encode()).hexdigest() != admission["content_sha256"]):
        raise ResearchError("CONTRACT_MISMATCH", "observed-state transport identity differs")
    document = _bounded_document(admission["content_utf8"])
    required = {"schema_version", "block_id", "coordinate_frame", "state_units", "time_unit", "velocity_source",
                "train_binding_hash", "normalizer_hash", "context_hash", "segments"}
    if (type(document) is not dict or set(document) != required or document["schema_version"] != "pirc26-observed-block-v1"
            or document["block_id"] != admission["block_id"] or document["coordinate_frame"] != model.spec.coordinate_frame
            or document["state_units"] != ["m", "m", "m/s", "m/s"] or document["time_unit"] != "s"
            or document["velocity_source"] not in ("measured", "backward-difference-v1")
            or any(document[name] != getattr(model.spec, name) for name in ("train_binding_hash", "normalizer_hash", "context_hash"))
            or type(document["segments"]) is not list or not 1 <= len(document["segments"]) <= 256):
        raise ResearchError("CONTRACT_MISMATCH", "observed-state units/provenance/schema differ")
    seen, observations, elements = set(), 0, 0
    for segment in document["segments"]:
        fields = {"segment_id", "time", "position", "condition", "condition_available_at"}
        if document["velocity_source"] == "measured":
            fields.add("velocity")
        if (type(segment) is not dict or set(segment) != fields or type(segment["segment_id"]) is not str
                or not 1 <= len(segment["segment_id"]) <= 128 or segment["segment_id"] in seen
                or type(segment["time"]) is not list or not 3 <= len(segment["time"]) <= 4098):
            raise ResearchError("CONTRACT_MISMATCH", "bounded unique observed segments required")
        t = segment["time"]
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in t)
                or any(a >= b for a, b in zip(t, t[1:])) or not _matrix(segment["position"], len(t), 2)
                or not _matrix(segment["condition"], len(t), model.spec.context_dim)
                or type(segment["condition_available_at"]) is not list or len(segment["condition_available_at"]) != len(t)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v > at
                       for v, at in zip(segment["condition_available_at"], t))
                or document["velocity_source"] == "measured" and not _matrix(segment["velocity"], len(t), 2)):
            raise ResearchError("CONTRACT_MISMATCH", "finite chronological observations and past-only context required")
        seen.add(segment["segment_id"])
        observations += len(t)
        elements += len(t) * (6 + model.spec.context_dim)
        if observations > max_observations or elements > 1048576:
            raise ResearchError("RESOURCE_PLAN_REJECTED", "decoded observed-state tensor quota")
    return ObservedBlock(admission, document, model)
