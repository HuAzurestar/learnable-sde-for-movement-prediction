"""Real streamed metadata agrees with the original stdlib JSON semantics."""

import codecs
import json
import os
import random
import sys

import pytest

from infrastructure.research_json import JSON_BLOCK_BYTES
from infrastructure.research_store import ResearchError, ResearchStore
from tests.research_file_observation import observe_file


def assert_original_value_or_rejection(target, content):
    target.write_bytes(content)
    try:
        original = json.loads(content)
    except (ValueError, RecursionError):
        with pytest.raises((ResearchError, RecursionError)):
            ResearchStore._json(target)
    else:
        current = ResearchStore._json(target)
        assert json.dumps(current, ensure_ascii=True) == json.dumps(original, ensure_ascii=True)


@pytest.mark.parametrize("body", [
    '""', '"π😀"', '"\\uD800\\uDC00"', '"\\ud800"',
    '"\\\\\\\"\\/\\b\\f\\n\\r\\t\\u0000"',
    "null", "true", "false", "NaN", "Infinity", "-Infinity", "0", "-0",
    "0.0", "-0.0", "123456789012345678901234567890", "1e-123", "-12.345e+234",
    '{"x":1,"x":2,"nested":[{},[],true,false,null]}',
    "", " ", '"unterminated', '"\\x00"', '"\\u123"', '"\\u00xx"', '"raw\ncontrol"',
    "-", "+1", "01", "-01", ".1", "1.", "1e", "1e+", "1e-", "1١",
    "[1,]", "[,1]", "[true false]", "[}", '{"x":}', '{"x":1,}', "{1:2}",
    '{"x" 1}', '{"x":1 "y":2}', 'null null', 'True', 'undefined',
])
def test_actual_token_split_at_normal_read_boundary_matches_stdlib(tmp_path, body):
    encoded = body.encode("utf-8", errors="surrogatepass")
    # Split each actual token/document near its middle at the unmodified64KiB
    # production boundary, including number, literal, UTF-8 and escape states.
    content = b" " * (JSON_BLOCK_BYTES - len(encoded) // 2) + encoded
    assert_original_value_or_rejection(tmp_path / "metadata.json", content)


@pytest.mark.parametrize("encoding,bom", [
    ("utf-8", b""), ("utf-8", codecs.BOM_UTF8),
    ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE),
    ("utf-32-le", codecs.BOM_UTF32_LE), ("utf-32-be", codecs.BOM_UTF32_BE),
])
@pytest.mark.parametrize("offset", range(-3, 4))
def test_actual_multibyte_and_escape_boundary_positions(tmp_path, encoding, bom, offset):
    note = "π😀\\\"\n" * 4
    body = json.dumps({"note": note}, ensure_ascii=False).encode(encoding)
    unit = len(" ".encode(encoding))
    prefix_size = max(0, (JSON_BLOCK_BYTES - len(bom) - len(body) // 2) // unit + offset)
    content = bom + " ".encode(encoding) * prefix_size + body
    assert_original_value_or_rejection(tmp_path / "metadata.json", content)


def test_seeded_real_metadata_corpus_and_mutations_match_stdlib(tmp_path):
    rng = random.Random(38610)
    strings = ["", "ascii", 'quote"\\/{}[]', "π😀", "\n\r\t", "\ud800", "\udc00"]

    def value(depth=0):
        kind = rng.randrange(8 if depth < 4 else 6)
        if kind == 0:
            return None
        if kind == 1:
            return bool(rng.randrange(2))
        if kind == 2:
            return rng.randrange(-10**12, 10**12)
        if kind == 3:
            return rng.uniform(-10**12, 10**12)
        if kind in (4, 5):
            return rng.choice(strings)
        if kind == 6:
            return [value(depth + 1) for _ in range(rng.randrange(5))]
        return {str(index) + rng.choice(strings): value(depth + 1) for index in range(rng.randrange(5))}

    target = tmp_path / "metadata.json"
    for index in range(180):
        encoded = json.dumps(value(), ensure_ascii=bool(index % 2)).encode("utf-8", errors="surrogatepass")
        samples = [encoded]
        if encoded:
            position = rng.randrange(len(encoded))
            samples.extend([encoded[:position] + encoded[position + 1:],
                            encoded[:position] + b"!" + encoded[position + 1:]])
        for sample in samples:
            content = b" " * (JSON_BLOCK_BYTES - len(sample) // 2) + sample
            assert_original_value_or_rejection(target, content)


def test_large_collection_consumes_all_actual_bytes_from_one_descriptor(tmp_path, monkeypatch):
    value = {"rows": [{"value": index, "note": "π😀"} for index in range(5000)]}
    content = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    target = tmp_path / "metadata.json"
    target.write_bytes(content)
    assert len(content) > JSON_BLOCK_BYTES
    reads, handles = observe_file(monkeypatch, target)
    assert ResearchStore._json(target) == value
    assert sum(row["bytes"] for row in reads) == 2 * len(content)
    assert all(0 <= row["size"] <= JSON_BLOCK_BYTES for row in reads)
    assert handles == [True]


def test_validation_finishes_before_large_values_are_constructed(tmp_path, monkeypatch):
    value = {"note": "x" * (2 * JSON_BLOCK_BYTES)}
    target = tmp_path / "metadata.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    size = target.stat().st_size
    original, parsed = json.loads, []

    def load(content, *args, **kwargs):
        if not parsed:
            assert sum(row["bytes"] for row in reads) >= size
            assert any(row["bytes"] == 0 for row in reads), "first value constructed before actual validation EOF"
        parsed.append(content if len(content) < 32 else len(content))
        return original(content, *args, **kwargs)

    reads, _ = observe_file(monkeypatch, target)
    monkeypatch.setattr(json, "loads", load)
    # Actual stdlib scalar decoding remains in the construction pass only.
    assert ResearchStore._json(target) == value
    assert parsed and max(value for value in parsed if isinstance(value, int)) > JSON_BLOCK_BYTES
    assert sum(row["bytes"] for row in reads) == 2 * target.stat().st_size


def test_actual_mutation_after_validation_denied_before_construction_pass(tmp_path, monkeypatch):
    target = tmp_path / "metadata.json"
    content = b" " * (JSON_BLOCK_BYTES + 1) + b'{"note":"original"}'
    target.write_bytes(content)
    before, changed = target.stat(), []

    def mutate(stream, value):
        if not value and not changed:
            target.write_bytes(content.replace(b"original", b"changed!"))
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            changed.append(True)

    reads, handles = observe_file(monkeypatch, target, after_read=mutate)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        ResearchStore._json(target)
    assert changed == [True]
    assert sum(row["bytes"] for row in reads) == len(content)
    assert handles == [True], "no reopen or second-pass bytes after identity changed"


def test_actual_growth_in_construction_pass_stays_inside_original_size(tmp_path, monkeypatch):
    target = tmp_path / "metadata.json"
    content = b" " * (JSON_BLOCK_BYTES + 1) + b'{"note":"frozen"}'
    target.write_bytes(content)
    validated, changed = [], []

    def grow(stream, value):
        if not value:
            validated.append(True)
        elif validated and not changed:
            with target.open("ab") as writer:
                writer.write(b" " * (1024 * 1024))
            changed.append(True)

    reads, handles = observe_file(monkeypatch, target, after_read=grow)
    with pytest.raises(ResearchError, match="CORRUPT_ARTIFACT"):
        ResearchStore._json(target)
    assert changed == [True]
    assert sum(row["bytes"] for row in reads) <= 2 * len(content) + 1
    assert all(0 <= row["size"] <= JSON_BLOCK_BYTES for row in reads)
    assert handles == [True]


def test_current_interpreter_integer_conversion_limit_is_preserved(tmp_path):
    getter = getattr(sys, "get_int_max_str_digits", None)
    if getter is None or getter() == 0:
        pytest.skip("interpreter has no enabled decimal integer conversion limit")
    target = tmp_path / "metadata.json"
    for count in (getter() - 1, getter(), getter() + 1):
        body = b"1" * count
        content = b" " * (JSON_BLOCK_BYTES + 1) + body
        assert_original_value_or_rejection(target, content)
