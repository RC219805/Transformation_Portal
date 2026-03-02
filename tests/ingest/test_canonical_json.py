"""Tests for JSON canonicalization helpers used across final emitters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from transformation_portal.ingest.canonical_json import canonicalize_json, to_jsonable

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - dependency may be optional in some local envs
    pytest.skip("Hypothesis not installed", allow_module_level=True)


def test_to_jsonable_normalizes_numpy_scalars_and_arrays() -> None:
    payload = {
        "a": np.int64(5),
        "b": np.float32(0.25),
        "c": np.bool_(True),
        "d": np.array([1, 2, 3], dtype=np.int64),
        "e": {np.int64(3), np.int64(1), np.int64(2)},
    }

    normalized = to_jsonable(payload)

    assert normalized["a"] == 5
    assert normalized["b"] == 0.25
    assert normalized["c"] is True
    assert normalized["d"] == [1, 2, 3]
    assert normalized["e"] == [1, 2, 3]


def test_canonicalize_json_accepts_numpy_payload() -> None:
    payload = {"x": np.int64(9), "y": np.array([np.int64(4), np.int64(5)])}
    blob = canonicalize_json(payload)
    parsed = json.loads(blob.decode("utf-8"))
    assert parsed == {"x": 9, "y": [4, 5]}


def test_to_jsonable_raises_on_unsupported_type() -> None:
    class _Unsupported:
        __slots__ = ()

    with pytest.raises(TypeError, match="Unsupported type for canonical JSON serialization"):
        to_jsonable(_Unsupported())


@settings(max_examples=120, deadline=None)
@given(
    payload=st.recursive(
        base=st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-10_000, max_value=10_000),
            st.floats(allow_nan=False, allow_infinity=False, width=64),
            st.text(min_size=0, max_size=16),
            st.integers(min_value=-10_000, max_value=10_000).map(np.int64),
            st.floats(allow_nan=False, allow_infinity=False, width=32).map(np.float32),
            st.booleans().map(np.bool_),
            st.lists(st.integers(-50, 50), max_size=8).map(lambda xs: np.array(xs, dtype=np.int64)),
            st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), max_size=8).map(
                lambda xs: np.array(xs, dtype=np.float32)
            ),
            st.lists(st.booleans(), max_size=8).map(lambda xs: np.array(xs, dtype=np.bool_)),
        ),
        extend=lambda children: st.one_of(
            st.lists(children, max_size=5),
            st.dictionaries(st.text(min_size=1, max_size=12), children, max_size=5),
            st.sets(
                st.one_of(
                    st.integers(min_value=-100, max_value=100),
                    st.text(min_size=0, max_size=8),
                    st.booleans(),
                    st.integers(min_value=-100, max_value=100).map(np.int64),
                ),
                max_size=5,
            ),
        ),
        max_leaves=30,
    )
)
def test_canonicalize_json_property_numpy_payload_idempotent(payload) -> None:
    """Property: canonical serialization should be stable and idempotent."""
    normalized = to_jsonable(payload)

    # Normalization is idempotent.
    assert normalized == to_jsonable(normalized)

    blob_first = canonicalize_json(payload)
    blob_second = canonicalize_json(payload)
    assert blob_first == blob_second

    # Canonical bytes should not change once normalized.
    assert blob_first == canonicalize_json(normalized)

    parsed = json.loads(blob_first.decode("utf-8"))
    assert blob_first == canonicalize_json(parsed)
