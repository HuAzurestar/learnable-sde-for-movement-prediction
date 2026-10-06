"""Bounded completed-chunk statistics and stateless next-stream identities.

No process, store or budget policy lives here. A runtime-owned callback may
publish the state only after a whole sample chunk has been incorporated.
"""

import json
import math

import numpy as np

from domain.errors import DataValidationError
from domain.frozen_dynamics import content_hash


class ChunkState:
    def __init__(self, request, method, counts, costs, *, phase=0, proposal=(), restored=None):
        request.validate()
        self.request = request
        self.counts, self.costs = tuple(counts), tuple(costs)
        self.weighted = method == "importance"
        self.identity = {"schema_version": "endpoint-chunk-state-v1", "request_hash": request.request_hash,
            "method_hash": content_hash({"method": method, "counts": counts, "costs": costs,
                                         "phase": phase, "proposal": proposal})}
        self.rng = {"scheme": "per-sample-seedsequence-v1", "seed": request.seed,
            "coupling_id": request.coupling_id, "phase": phase, "bit_generator": "PCG64",
            "numpy_version": np.__version__}
        empty = ({"n": 0, "hits": 0, "log_w": None, "log_w2": None, "log_event": None,
                  "log_event2": None, "max_log_w": None} if self.weighted else
                 {"n": 0, "mean": 0.0, "m2": 0.0, "hits": 0})
        self.statistics = [dict(empty) for _ in counts]
        self.level, self.position = 0, 0
        if restored is not None:
            self._restore(restored)

    def _restore(self, state):
        def reject():
            raise DataValidationError("checkpoint identity, chunk boundary or statistics differ")
        try:
            # Detach an explicitly bounded JSON state, never a live caller dict.
            if type(state) is not dict or set(state) != {"step", "data_position", "method_state", "rng_state", "chunk_complete"}:
                reject()
            content = json.dumps(state, allow_nan=False)
            if len(content.encode()) > 16384:
                reject()
            state = json.loads(content)
            position, method = state["data_position"], state["method_state"]
            if (state["chunk_complete"] is not True or state["rng_state"] != self.rng
                    or type(position) is not dict or set(position) != {"level", "next_sample"}
                    or type(method) is not dict or set(method) != set(self.identity) | {"statistics"}
                    or any(method[key] != value for key, value in self.identity.items())):
                reject()
            level, offset = position["level"], position["next_sample"]
            if (type(level) is not int or not 0 <= level <= len(self.counts) or type(offset) is not int
                    or offset < 0 or (level == len(self.counts) and offset != 0)
                    or (level < len(self.counts) and (offset >= self.counts[level] or offset % self.request.chunk_size))):
                reject()
            statistics = method["statistics"]
            if type(statistics) is not list or len(statistics) != len(self.counts):
                reject()
            for index, item in enumerate(statistics):
                count = self.counts[index] if index < level else offset if index == level else 0
                if (type(item) is not dict or set(item) != set(self.statistics[index])
                        or type(item["n"]) is not int or item["n"] != count
                        or type(item["hits"]) is not int or not 0 <= item["hits"] <= count):
                    reject()
                for key, value in item.items():
                    if key in {"n", "hits"}:
                        continue
                    if self.weighted:
                        expected_none = count == 0 or (key in {"log_event", "log_event2"} and item["hits"] == 0)
                        if (value is None) != expected_none:
                            reject()
                    if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                        reject()
                if not self.weighted and (item["m2"] < 0 or count == 0 and (item["mean"] != 0 or item["m2"] != 0)):
                    reject()
            self.statistics, self.level, self.position = statistics, level, offset
            if type(state["step"]) is not int or state["step"] != self.completed_work:
                reject()
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise DataValidationError("invalid finite chunk checkpoint") from exc

    @property
    def completed_work(self):
        return sum(item["n"] * cost for item, cost in zip(self.statistics, self.costs))

    @property
    def total_work(self):
        return sum(n * cost for n, cost in zip(self.counts, self.costs))

    def completed(self, level, stop, callback):
        self.level, self.position = (level + 1, 0) if stop == self.counts[level] else (level, stop)
        if callback is not None:
            state = {"step": self.completed_work, "data_position": {"level": self.level, "next_sample": self.position},
                "method_state": {**self.identity, "statistics": self.statistics}, "rng_state": self.rng, "chunk_complete": True}
            # Prevent a callback from mutating live estimator state.
            callback(json.loads(json.dumps(state, allow_nan=False)), self.total_work)
