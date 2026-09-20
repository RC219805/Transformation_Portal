"""Bounded semantic comparison of CPython source caches without loading code.

Marshal reference flags depend on live object reference counts. Hashing their
wire bytes therefore cannot establish source equivalence. This reader accepts
only the immutable value forms emitted by the supported source compiler. It
never calls marshal.loads or constructs a code object from cache bytes.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Iterable

MAX_BYTECODE_BYTES = 16 * 1024 * 1024
MAX_BYTECODE_NODES = 262144
MAX_BYTECODE_ITEMS = 65536
MAX_BYTECODE_DEPTH = 64
_SUPPORTED_LAYOUTS = frozenset({(3, 11), (3, 12), (3, 13)})


def _digest(kind: bytes, parts: Iterable[bytes] = ()) -> bytes:
    result = hashlib.sha256(kind)
    for part in parts:
        result.update(len(part).to_bytes(8, "little"))
        result.update(part)
    return result.digest()


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0
        self.nodes = 0
        self.references: list[bytes | None] = []

    def take(self, count: int) -> bytes:
        if count < 0 or count > len(self.payload) - self.position:
            raise ValueError("Truncated or oversized bytecode field")
        start = self.position
        self.position += count
        return self.payload[start : self.position]

    def integer(self) -> int:
        return int.from_bytes(self.take(4), "little", signed=True)

    def count(self, *, short: bool = False) -> int:
        count = self.take(1)[0] if short else self.integer()
        if not 0 <= count <= MAX_BYTECODE_ITEMS or count > len(self.payload) - self.position:
            raise ValueError("Bytecode container exceeds its item bound")
        return count

    def node(self, depth: int = 0) -> bytes:
        if depth > MAX_BYTECODE_DEPTH:
            raise ValueError("Bytecode exceeds its nesting bound")
        self.nodes += 1
        if self.nodes > MAX_BYTECODE_NODES:
            raise ValueError("Bytecode exceeds its node bound")
        marker = self.take(1)[0]
        referenced = bool(marker & 0x80)
        tag = marker & 0x7F
        if referenced and tag in {ord("N"), ord("F"), ord("T"), ord(".")}:
            # CPython ignores FLAG_REF on these singleton cases instead of
            # reserving a slot. Reject it so our indices cannot diverge from
            # the interpreter's reference table on malicious cache bytes.
            raise ValueError("Bytecode singleton cannot declare a reference")
        if tag == ord("r"):
            index = self.integer()
            if referenced or not 0 <= index < len(self.references) or self.references[index] is None:
                raise ValueError("Invalid or cyclic bytecode reference")
            value = self.references[index]
            assert value is not None
            return value
        reference_index = len(self.references) if referenced else None
        if referenced:
            self.references.append(None)
        if tag in {ord("N"), ord("F"), ord("T"), ord(".")}:
            result = _digest(bytes([tag]))
        elif tag == ord("i"):
            result = _digest(b"integer", [self.take(4)])
        elif tag == ord("l"):
            digits = self.integer()
            if abs(digits) > MAX_BYTECODE_BYTES // 2:
                raise ValueError("Bytecode integer exceeds its byte bound")
            result = _digest(b"long", [digits.to_bytes(4, "little", signed=True), self.take(abs(digits) * 2)])
        elif tag in {ord("g"), ord("y")}:
            result = _digest(bytes([tag]), [self.take(8 if tag == ord("g") else 16)])
        elif tag in {ord("s"), ord("u"), ord("t"), ord("a"), ord("A"), ord("z"), ord("Z")}:
            length = self.take(1)[0] if tag in {ord("z"), ord("Z")} else self.integer()
            content = self.take(length)
            if tag == ord("s"):
                result = _digest(b"bytes", [content])
            else:
                if tag in {ord("a"), ord("A"), ord("z"), ord("Z")}:
                    content.decode("ascii")
                else:
                    content.decode("utf-8", errors="surrogatepass")
                result = _digest(b"text", [content])
        elif tag in {ord("("), ord(")"), ord(">")}:
            count = self.count(short=tag == ord(")"))
            children = [self.node(depth + 1) for _ in range(count)]
            if tag == ord(">"):
                children.sort()
            result = _digest(b"frozenset" if tag == ord(">") else b"tuple", children)
        elif tag == ord("c"):
            # CPython 3.11-3.13: argcount, posonlyargcount, kwonlyargcount,
            # stacksize, flags; code, consts, names, localsplusnames,
            # localspluskinds, filename, name, qualname; firstlineno,
            # linetable and exceptiontable. Bind every field.
            fields = [self.take(20)]
            fields.extend(self.node(depth + 1) for _ in range(8))
            fields.append(self.take(4))
            fields.extend(self.node(depth + 1) for _ in range(2))
            result = _digest(b"code", fields)
        else:
            raise ValueError("Unsupported bytecode value type")
        if reference_index is not None:
            self.references[reference_index] = result
        return result


def source_cache_semantic_digest(payload: bytes, *, python_version: tuple[int, int] | None = None) -> bytes:
    """Return a fixed-size digest of bounded immutable compiler values.

    References reuse already computed digests instead of expanding their target;
    nested shared values cannot amplify memory or comparison time. Code layouts
    outside the explicitly supported interpreter family fail closed.
    """
    version = sys.version_info[:2] if python_version is None else python_version
    if version not in _SUPPORTED_LAYOUTS:
        raise ValueError("Unsupported Python bytecode layout")
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_BYTECODE_BYTES:
        raise ValueError("Bytecode exceeds its payload bound")
    if payload[0] & 0x7F != ord("c"):
        raise ValueError("Source bytecode must contain a top-level code object")
    reader = _Reader(payload)
    result = reader.node()
    if reader.position != len(payload):
        raise ValueError("Bytecode contains trailing data")
    return result
