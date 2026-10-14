"""Bounded lossless numeric/JSON frames; no engine, pickle or authority."""

import base64
import json
import math
import struct
import zlib

from infrastructure.research_control import canonical
from infrastructure.research_store import ResearchError

LIMIT = 4 * 1024 * 1024
TEXT_LIMIT = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", message)


def chunks(raw):
    text = base64.b64encode(raw).decode("ascii")
    return [text[i:i+16384] for i in range(0, len(text), 16384)]


def unchunks(parts, maximum):
    require(type(parts) is list and len(parts) <= (maximum * 4 // 3 + 16384) // 16384
            and all(type(s) is str and 0 < len(s) <= 16384 for s in parts), "bounded binary chunks required")
    try:
        raw = base64.b64decode("".join(parts), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "invalid binary encoding") from exc
    require(len(raw) <= maximum and chunks(raw) == parts, "noncanonical/oversized binary chunks")
    return raw


def inspect_array(value, shape, dtype):
    require(type(value) is dict and set(value) == {"shape", "dtype", "data"}
            and type(value["shape"]) is list and all(type(n) is int for n in value["shape"])
            and value["shape"] == shape and value["dtype"] == dtype, "forecast tensor shape/dtype differs")
    count = math.prod(shape)
    require(all(type(n) is int and 0 <= n <= 10000 for n in shape) and count <= 2_000_000,
            "forecast tensor quota")
    width, fmt = (4, "<f") if dtype == "float32" else (8, "<d")
    require(dtype in {"float32", "float64"}, "CPU floating dtype required")
    raw = unchunks(value["data"], count * width)
    require(len(raw) == count * width and all(math.isfinite(x[0]) for x in struct.iter_unpack(fmt, raw)),
            "finite exact-width forecast values required")
    return raw


def pack_json(value):
    # This is scientific state compression, not a shortened/quantized history.
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()
    require(len(content) <= TEXT_LIMIT, "lossless JSON state quota")
    result = {"schema_version": "pirc26-lossless-json-v1", "size_bytes": len(content), "data": chunks(zlib.compress(content))}
    canonical(result, LIMIT)
    return result


def unpack_json(value):
    require(type(value) is dict and set(value) == {"schema_version", "size_bytes", "data"}
            and value["schema_version"] == "pirc26-lossless-json-v1"
            and type(value["size_bytes"]) is int and 0 < value["size_bytes"] <= TEXT_LIMIT, "closed lossless JSON frame")
    compressed = unchunks(value["data"], LIMIT)
    stream = zlib.decompressobj()
    try:
        raw = stream.decompress(compressed, value["size_bytes"] + 1)
        require(stream.eof and not stream.unused_data and not stream.unconsumed_tail
                and len(raw) == value["size_bytes"], "compressed length/trailing data differs")
        result = json.loads(raw)
        require(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False,
                           ensure_ascii=False).encode() == raw, "noncanonical lossless JSON")
        return result
    except (ValueError, zlib.error, RecursionError, UnicodeError) as exc:
        raise ResearchError("CHECKPOINT_INCOMPATIBLE", "malformed compressed JSON") from exc
