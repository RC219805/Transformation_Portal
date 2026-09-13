"""Bounded shard-map parsing rejects traversal before filesystem access."""

from __future__ import annotations

import json

import pytest

from transformation_portal.core.security.checkpoint_index import parse_checkpoint_weight_map

pytestmark = [pytest.mark.unit, pytest.mark.security]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.bin",
        "/outside.bin",
        "weights/../../outside.bin",
        "C:/outside.bin",
        "C:outside.bin",
        "\\\\host\\share",
        "a\\b",
        "./weights.bin",
        "a//b",
        "",
        "a\x00b",
        None,
    ],
)
def test_unsafe_shard_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="unsafe weight-map path"):
        parse_checkpoint_weight_map(json.dumps({"weight_map": {"layer": path}}).encode())


@pytest.mark.parametrize(
    "raw",
    [
        b'{"weight_map":{"x":"a.bin","x":"b.bin"}}',
        b'{"weight_map":{},"weight_map":{"x":"a.bin"}}',
        b'{"metadata":NaN,"weight_map":{"x":"a.bin"}}',
        b"[",
        b"[]",
        b'{"weight_map":{}}',
        b"[" * 2000,
    ],
)
def test_malformed_or_ambiguous_indexes_fail(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_checkpoint_weight_map(raw)


def test_index_bytes_and_entries_are_bounded() -> None:
    raw = b'{"weight_map":{"a":"one.bin","b":"two.bin"}}'
    with pytest.raises(ValueError, match="oversized"):
        parse_checkpoint_weight_map(raw, maximum_bytes=len(raw) - 1)
    with pytest.raises(ValueError, match="too many weight-map entries"):
        parse_checkpoint_weight_map(raw, maximum_entries=1)


def test_nested_shards_and_shared_tensor_files_remain_valid() -> None:
    weight_map = {"layer.a": "weights/model-00001.safetensors", "layer.b": "weights/model-00001.safetensors"}
    assert parse_checkpoint_weight_map(json.dumps({"metadata": {}, "weight_map": weight_map}).encode()) == weight_map
