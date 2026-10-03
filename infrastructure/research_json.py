"""Same-handle JSON reads without retaining an entire large encoded file.

Small metadata keeps the original stdlib decoder. Large metadata is first
syntax-checked without constructing values, then consumed from the same pinned
descriptor. Returned decoded values still have their natural memory cost.
"""

import codecs
import json
import re
import sys

from .research_store import ResearchError


JSON_BLOCK_BYTES = 64 * 1024
_SPACE = re.compile(r"[ \t\r\n]*")
_STRING_END = re.compile(r'["\\\x00-\x1f]')
_DIGITS = re.compile(r"[0-9]*")
_DISCARDED_SCALAR = object()


def _text_chunks(stream, size, first, encoding):
    decoder = codecs.getincrementaldecoder(encoding or json.detect_encoding(first))(
        "strict" if encoding else "surrogatepass")
    consumed, chunk = 0, first
    while True:
        consumed += len(chunk)
        if consumed > size:
            raise ResearchError("CORRUPT_ARTIFACT", "JSON source grew during read")
        text = decoder.decode(chunk, final=not chunk)
        if text:
            yield text
        if not chunk:
            if consumed != size:
                raise ResearchError("CORRUPT_ARTIFACT", "JSON source was truncated during read")
            return
        del chunk
        chunk = stream.read(min(JSON_BLOCK_BYTES, size - consumed + 1))


class _Cursor:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.text, self.index = "", 0

    def peek(self):
        while self.index == len(self.text):
            self.text = next(self.chunks, "")
            self.index = 0
            if not self.text:
                return None
        return self.text[self.index]

    def take(self):
        value = self.peek()
        if value is None:
            self.error()
        self.index += 1
        return value

    def error(self):
        # Never retain the complete damaged source in JSONDecodeError.doc.
        raise json.JSONDecodeError("invalid large JSON", self.text, self.index)

    def whitespace(self):
        while self.peek() is not None:
            self.index = _SPACE.match(self.text, self.index).end()
            if self.index < len(self.text):
                return

    def string(self, build, *, max_chars=None):
        assert self.take() == '"'
        pieces = ['"'] if build else None
        captured, discarded = 1, False
        def collect(value):
            nonlocal captured, discarded
            if build and not discarded:
                captured += len(value)
                if max_chars is not None and captured > max_chars:
                    pieces.clear()
                    discarded = True
                else:
                    pieces.append(value)
        while self.peek() is not None:
            found = _STRING_END.search(self.text, self.index)
            end = found.start() if found else len(self.text)
            if build:
                collect(self.text[self.index:end])
            self.index = end
            if not found:
                continue
            character = self.take()
            if build:
                collect(character)
            if character == '"':
                if discarded:
                    return _DISCARDED_SCALAR
                return json.loads("".join(pieces)) if build else None
            if character != "\\":
                self.error()
            escape = self.take()
            if build:
                collect(escape)
            if escape == "u":
                for _ in range(4):
                    digit = self.take()
                    if digit not in "0123456789abcdefABCDEF":
                        self.error()
                    if build:
                        collect(digit)
            elif escape not in '"\\/bfnrt':
                self.error()
        self.error()

    def literal(self, spelling, build):
        for character in spelling:
            if self.take() != character:
                self.error()
        return json.loads(spelling) if build else None

    def digits(self, pieces, *, collect=None):
        count = 0
        while self.peek() is not None:
            end = _DIGITS.match(self.text, self.index).end()
            count += end - self.index
            if collect is not None:
                collect(self.text[self.index:end])
            elif pieces is not None:
                pieces.append(self.text[self.index:end])
            self.index = end
            if end < len(self.text):
                return count
        return count

    def number(self, build, *, max_chars=None):
        pieces = [] if build else None
        captured, discarded = 0, False
        def collect(value):
            nonlocal captured, discarded
            if build and not discarded:
                captured += len(value)
                if max_chars is not None and captured > max_chars:
                    pieces.clear()
                    discarded = True
                else:
                    pieces.append(value)
        def digits():
            return self.digits(None, collect=collect if build else None)

        def take():
            character = self.take()
            if build:
                collect(character)

        if self.peek() == "-":
            take()
            if self.peek() == "I":
                self.literal("Infinity", False)
                return json.loads("-Infinity") if build else None
        if self.peek() == "0":
            take()
            digit_count = 1
        elif self.peek() is not None and "1" <= self.peek() <= "9":
            digit_count = digits()
        else:
            self.error()
        floating = False
        if self.peek() == ".":
            take()
            floating = True
            if not digits():
                self.error()
        if self.peek() in ("e", "E"):
            take()
            floating = True
            if self.peek() in ("+", "-"):
                take()
            if not digits():
                self.error()
        limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
        if not floating and limit and digit_count > limit:
            raise ValueError("JSON integer exceeds current interpreter conversion limit")
        if discarded:
            return _DISCARDED_SCALAR
        return json.loads("".join(pieces)) if build else None


def _parse(chunks, *, build):
    return _parse_cursor(_Cursor(chunks), build=build)


def _parse_cursor(cursor, *, build, consume_all=True):
    keys = {}  # Match stdlib's document-local object-key sharing, not a cache.
    # Frame = [grammar state, decoded container/value, pending object key].
    # The validation pass retains states only, not keys, strings or containers.
    frames = [["root-value", None, None]]

    def finish(value):
        frame = frames[-1]
        if frame[0] == "root-value":
            frame[0], frame[1] = "root-done", value
        elif frame[0] in ("array-first", "array-value"):
            if build:
                frame[1].append(value)
            frame[0] = "array-end"
        elif frame[0] == "object-value":
            if build:
                frame[1][frame[2]] = value
            frame[0], frame[2] = "object-end", None
        else:
            cursor.error()

    while True:
        cursor.whitespace()
        character, frame = cursor.peek(), frames[-1]
        state = frame[0]
        if state == "root-done":
            if consume_all and character is not None:
                cursor.error()
            return frame[1]
        if (state in ("object-first", "object-end") and character == "}"
                or state in ("array-first", "array-end") and character == "]"):
            cursor.take()
            finish(frames.pop()[1])
        elif state in ("object-first", "object-key"):
            if character != '"':
                cursor.error()
            key = cursor.string(build)
            frame[2] = keys.setdefault(key, key) if build else None
            frame[0] = "object-colon"
        elif state == "object-colon":
            if character != ":":
                cursor.error()
            cursor.take()
            frame[0] = "object-value"
        elif state in ("object-end", "array-end"):
            if character != ",":
                cursor.error()
            cursor.take()
            frame[0] = "object-key" if state == "object-end" else "array-value"
        elif character in ("{", "["):
            cursor.take()
            frames.append(["object-first" if character == "{" else "array-first",
                           ({} if character == "{" else []) if build else None, None])
            if len(frames) > sys.getrecursionlimit():
                raise RecursionError("JSON exceeds current interpreter nesting limit")
        elif character == '"':
            finish(cursor.string(build))
        elif character in ("t", "f", "n", "N", "I"):
            finish(cursor.literal({"t": "true", "f": "false", "n": "null",
                                   "N": "NaN", "I": "Infinity"}[character], build))
        elif character is not None and character in "-0123456789":
            finish(cursor.number(build))
        else:
            cursor.error()


def read_json(stream, size, *, encoding=None, verify_identity=None):
    """Preserve default JSON values/encodings without a large source buffer."""
    first = stream.read(min(JSON_BLOCK_BYTES, size + 1))
    if len(first) > size:
        raise ResearchError("CORRUPT_ARTIFACT", "JSON source grew during read")
    if size <= JSON_BLOCK_BYTES:
        if len(first) != size:
            raise ResearchError("CORRUPT_ARTIFACT", "JSON source was truncated during read")
        return json.loads(first.decode(encoding) if encoding else first)
    _parse(_text_chunks(stream, size, first, encoding), build=False)
    if verify_identity is not None:
        verify_identity()  # Do not construct values from a changed validation pass.
    stream.seek(0)
    first = stream.read(min(JSON_BLOCK_BYTES, size + 1))
    return _parse(_text_chunks(stream, size, first, encoding), build=True)


def read_metadata_header(stream, size, expected_fields):
    """Syntax-check JSON while selecting only bounded top-level scalar fields.

    Expected fields are frozen schema/status/identity metadata, not a request
    for a file payload. Oversized or non-scalar selected values are represented
    by a private sentinel and must fail the caller's exact typed comparison.
    Unselected keys/values are parsed without constructing decoded objects.
    """
    first = stream.read(min(JSON_BLOCK_BYTES, size + 1))
    cursor = _Cursor(_text_chunks(stream, size, first, 'utf-8'))
    cursor.whitespace()
    if cursor.take() != '{':
        cursor.error()
    result = {}
    key_limit = 12 * max((len(key) for key in expected_fields), default=0) + 2
    cursor.whitespace()
    if cursor.peek() != '}':
        while True:
            if cursor.peek() != '"':
                cursor.error()
            key = cursor.string(True, max_chars=key_limit)
            cursor.whitespace()
            if cursor.take() != ':':
                cursor.error()
            cursor.whitespace()
            if key not in expected_fields:
                _parse_cursor(cursor, build=False, consume_all=False)
            else:
                character, expected = cursor.peek(), expected_fields[key]
                if character == '"':
                    value = cursor.string(True, max_chars=12 * max(64, len(str(expected))) + 2)
                elif character in ('t', 'f', 'n'):
                    value = cursor.literal({'t': 'true', 'f': 'false', 'n': 'null'}[character], True)
                elif character is not None and character in '-0123456789':
                    value = cursor.number(True, max_chars=max(128, len(str(expected)) + 8))
                else:
                    _parse_cursor(cursor, build=False, consume_all=False)
                    value = _DISCARDED_SCALAR
                result[key] = value
            cursor.whitespace()
            character = cursor.take()
            if character == '}':
                break
            if character != ',':
                cursor.error()
            cursor.whitespace()
    else:
        cursor.take()
    cursor.whitespace()
    if cursor.peek() is not None:
        cursor.error()
    return result
