"""Actual metadata JSON paths reject damage without whole-file allocation."""

import codecs
import json
from pathlib import Path
import tracemalloc

import pytest

from experiments.pirc25.upstream import AdmissionError, Binding, canonical_hash, validate_binding
from infrastructure.research_store import ResearchError, ResearchStore
from tests.research_file_observation import observe_file


PAYLOAD = {"schema_version": "synthetic-v1", "status": "frozen", "note": "合成-π-😀",
           "numbers": [0, -1, 2.5, 1e-20], "flags": [True, False, None]}
BLOCK = 64 * 1024


def actual_entry(tmp_path, entry):
    if entry == "metadata":
        store = ResearchStore(tmp_path, "json-bounds", initialize=True)
        store.publish("synthetic-json", PAYLOAD)
        target = store.path / "manifests/synthetic-json.json"
        return target, lambda: store.manifest("synthetic-json"), ResearchError
    target = tmp_path / "metadata.json"
    binding = Binding("synthetic", target.name, PAYLOAD["schema_version"],
                      canonical_hash(PAYLOAD), PAYLOAD["status"])
    return target, lambda: validate_binding(tmp_path, binding), AdmissionError


@pytest.mark.parametrize("entry", ["metadata", "upstream"])
@pytest.mark.parametrize("damage", ["initial", "unterminated-string", "trailing"])
def test_actual_pregrown_damaged_json_has_bounded_reads_and_memory(tmp_path, monkeypatch, entry, damage):
    target, read, error = actual_entry(tmp_path, entry)
    padding = b"x" * (8 * 1024 * 1024)
    if damage == "initial":
        content = b"!" + padding
    elif damage == "unterminated-string":
        content = b'{"note":"' + padding
    else:
        content = json.dumps(PAYLOAD).encode("utf-8") + b" " * len(padding) + b"!"
    target.write_bytes(content)
    reads, handles = observe_file(monkeypatch, target)
    tracemalloc.start()
    try:
        with pytest.raises(error, match="CORRUPT_ARTIFACT|MISSING_INPUT"):
            read()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, (
        f"damaged JSON allocated {peak} bytes; requests={[row['size'] for row in reads]}")
    assert reads and all(0 <= row["size"] <= BLOCK for row in reads)
    assert handles == [True], "validation and consumption must use one actual pinned descriptor"


@pytest.mark.parametrize("entry", ["metadata", "upstream"])
def test_valid_large_formatted_json_is_not_rejected_or_materialized(tmp_path, monkeypatch, entry):
    target, read, _ = actual_entry(tmp_path, entry)
    content = (" \r\n\t" * (5 * 1024 * 1024) + json.dumps(PAYLOAD, ensure_ascii=False, indent=2)
               + "\r\n").encode("utf-8")
    target.write_bytes(content)
    assert len(content) > 16 * 1024 * 1024
    reads, handles = observe_file(monkeypatch, target)
    tracemalloc.start()
    try:
        assert read() == PAYLOAD
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, f"legal formatting allocated the whole file: {peak} bytes"
    assert reads and all(0 <= row["size"] <= BLOCK for row in reads)
    assert handles == [True]


@pytest.mark.parametrize("encoding,bom", [
    ("utf-8", b""), ("utf-8", codecs.BOM_UTF8),
    ("utf-16-le", b""), ("utf-16-be", b""),
    ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE),
    ("utf-32-le", b""), ("utf-32-be", b""),
    ("utf-32-le", codecs.BOM_UTF32_LE), ("utf-32-be", codecs.BOM_UTF32_BE),
])
def test_actual_metadata_preserves_large_json_byte_encoding_compatibility(tmp_path, encoding, bom):
    target, read, _ = actual_entry(tmp_path, "metadata")
    text = " " * (BLOCK + 1) + json.dumps(PAYLOAD, ensure_ascii=False, indent=2)
    content = bom + text.encode(encoding)
    target.write_bytes(content)
    assert json.loads(content) == PAYLOAD
    assert read() == PAYLOAD


@pytest.mark.parametrize("body", [
    r'{"same":1,"same":2,"nested":[{},[],true,false,null,-0,1.0,-1e+3]}',
    r'{"escapes":"\"\\\/\b\f\n\r\t\u0000\uD800\uDC00\ud800","unicode":"π😀"}',
    r'[NaN,Infinity,-Infinity,1e999,-1e999]',
    "[" * 800 + "0" + "]" * 800,
])
def test_large_metadata_parser_matches_original_stdlib_value_semantics(tmp_path, body):
    target = tmp_path / "metadata.json"
    content = (" " * (BLOCK + 1) + body).encode("utf-8")
    target.write_bytes(content)
    # Standalone actual reader control: journal publication has its own strict
    # finite canonical-value contract, which this test does not bypass.
    expected = json.loads(content)
    actual = ResearchStore._json(target)
    assert json.dumps(actual, ensure_ascii=True) == json.dumps(expected, ensure_ascii=True)
