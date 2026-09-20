"""Executable source-cache authority tolerates representation-only marshal changes."""

from __future__ import annotations

import importlib.util
import marshal
import py_compile
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v4 import _bytecode, runtime

pytestmark = pytest.mark.unit


def _integer(value):
    return value.to_bytes(4, "little", signed=True)


def _node(payload):
    reader = _bytecode._Reader(payload)
    digest = reader.node()
    assert reader.position == len(payload)
    return digest


def test_frozenset_reference_count_changes_wire_bytes_without_changing_code(tmp_path):
    source = tmp_path / "membership.py"
    source.write_text("def match(value):\n    return value in {'portal', 'frontdoor'}\n")
    code = compile(source.read_bytes(), str(source), "exec", dont_inherit=True)
    before = marshal.dumps(code)
    held_frozenset = code.co_consts[0].co_consts[-1]
    assert isinstance(held_frozenset, frozenset)
    after = marshal.dumps(code)
    assert before != after
    assert _bytecode.source_cache_semantic_digest(before) == _bytecode.source_cache_semantic_digest(after)
    cache = Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(str(source), cfile=str(cache), doraise=True)
    header = cache.read_bytes()[:16]
    for payload in (before, after):
        cache.write_bytes(header + payload)
        runtime._directory_inventory(tmp_path)


def test_unicode_surrogate_constants_are_valid_compiler_values():
    code = compile('value = "\\ud800"\n', "/source.py", "exec", dont_inherit=True)
    assert len(_bytecode.source_cache_semantic_digest(marshal.dumps(code))) == 32


def test_reference_and_string_encodings_have_equivalent_semantic_digests():
    first = b"(" + _integer(2) + bytes([ord("z") | 0x80, 3]) + b"abc" + b"r" + _integer(0)
    second = b")\x02u" + _integer(3) + b"abcz\x03abc"
    assert _node(first) == _node(second)
    assert _node(b"s" + _integer(3) + b"abc") != _node(b"z\x03abc")


def test_frozenset_order_is_normalized_and_cardinality_is_bound():
    first = b">" + _integer(2) + b"i" + _integer(1) + b"i" + _integer(2)
    second = b">" + _integer(2) + b"i" + _integer(2) + b"i" + _integer(1)
    duplicate = b">" + _integer(2) + b"i" + _integer(1) + b"i" + _integer(1)
    assert _node(first) == _node(second)
    assert _node(first) != _node(duplicate)
    assert len({_node(tag) for tag in (b"N", b"T", b"F", b".")}) == 4


@pytest.mark.parametrize(
    "payload, reason",
    [
        (b"r" + _integer(0), "reference"),
        (bytes([ord("(") | 0x80]) + _integer(1) + b"r" + _integer(0), "cyclic"),
        (b"(" + _integer(-1), "item bound"),
        (b"(" + _integer(_bytecode.MAX_BYTECODE_ITEMS + 1), "item bound"),
        (b"s" + _integer(2**31 - 1), "Truncated or oversized"),
        (b"l" + _integer(-(2**31)), "integer exceeds"),
        (b"?", "Unsupported"),
    ],
)
def test_malformed_cache_objects_fail_before_allocation(payload, reason):
    with pytest.raises(ValueError, match=reason):
        _node(payload)


@pytest.mark.parametrize("tag", ["N", "F", "T", "."])
def test_singleton_reference_flags_cannot_shift_interpreter_reference_indices(tag):
    with pytest.raises(ValueError, match="singleton cannot declare"):
        _node(bytes([ord(tag) | 0x80]))


def test_nested_and_many_nodes_have_independent_bounds(monkeypatch):
    monkeypatch.setattr(_bytecode, "MAX_BYTECODE_DEPTH", 8)
    with pytest.raises(ValueError, match="nesting"):
        _node(b")\x01" * 10 + b"N")
    monkeypatch.setattr(_bytecode, "MAX_BYTECODE_NODES", 2)
    with pytest.raises(ValueError, match="node bound"):
        _node(b")\x03NNN")


def test_shared_reference_graph_never_expands_its_children():
    payload = b"(" + _integer(50) + bytes([ord(")") | 0x80, 0])
    for index in range(49):
        payload += bytes([ord(")") | 0x80, 2]) + (b"r" + _integer(index)) * 2
    reader = _bytecode._Reader(payload)
    assert len(reader.node()) == 32
    assert len(reader.references) == 50
    assert reader.nodes == 149


def test_top_level_payload_layout_and_trailing_bytes_fail_closed():
    code = compile("value = 1\n", "/source.py", "exec", dont_inherit=True)
    payload = marshal.dumps(code)
    with pytest.raises(ValueError, match="trailing"):
        _bytecode.source_cache_semantic_digest(payload + b"N")
    with pytest.raises(ValueError, match="top-level"):
        _bytecode.source_cache_semantic_digest(b"N")
    with pytest.raises(ValueError, match="layout"):
        _bytecode.source_cache_semantic_digest(payload, python_version=(3, 14))
    with pytest.raises(ValueError, match="payload bound"):
        _bytecode.source_cache_semantic_digest(b"x" * (_bytecode.MAX_BYTECODE_BYTES + 1))


@pytest.mark.parametrize("field", ["co_filename", "co_firstlineno", "co_linetable", "co_exceptiontable", "co_consts"])
def test_code_metadata_and_constant_changes_remain_authorizing(field):
    code = compile("value = 1\n", "/source.py", "exec", dont_inherit=True)
    replacement = {
        "co_filename": "/different.py",
        "co_firstlineno": code.co_firstlineno + 1,
        "co_linetable": b"changed",
        "co_exceptiontable": b"changed",
        "co_consts": (2, None),
    }[field]
    changed = code.replace(**{field: replacement})
    assert _bytecode.source_cache_semantic_digest(marshal.dumps(code)) != _bytecode.source_cache_semantic_digest(
        marshal.dumps(changed)
    )
