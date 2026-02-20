from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.spatial_ai.ingest.fuzz_corpus_loader import load_corpus_cases
from transformation_portal.spatial_ai.ingest import ColorSpaceError, LinearDecoder

# ---------------------------------------------------------------------------
# Allowed exception contracts
# ---------------------------------------------------------------------------

_ALLOWED_METADATA_EXC = (ValueError,)
_ALLOWED_COLORSPACE_EXC = (ColorSpaceError,)


# ---------------------------------------------------------------------------
# RAW stub for metadata validation
# ---------------------------------------------------------------------------


def _make_raw_stub(wb: Any, bl: Any, raw_image_shape: Any) -> Any:
    """Build a SimpleNamespace raw stub — hasattr returns False for absent attrs."""
    import types as _types

    kwargs: dict[str, Any] = {}
    if wb is not None:
        kwargs["camera_whitebalance"] = wb
    if bl is not None:
        kwargs["black_level_per_channel"] = bl
    if raw_image_shape is not None:
        kwargs["raw_image"] = np.zeros(raw_image_shape, dtype=np.uint16)
    return _types.SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# Hypothesis strategies (bounded + shrink-friendly)
# ---------------------------------------------------------------------------

_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-10, 10),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.text(min_size=0, max_size=8),
)

_numeric_seq = st.lists(
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    min_size=0,
    max_size=12,
)

_ragged_seq = st.lists(
    st.one_of(_scalar, _numeric_seq),
    min_size=0,
    max_size=6,
)

_wb_strategy = st.one_of(_scalar, _numeric_seq, _ragged_seq)
_bl_strategy = st.one_of(_scalar, _numeric_seq, _ragged_seq)

_raw_shape_strategy = st.one_of(
    st.none(),
    st.tuples(st.integers(1, 32), st.integers(1, 32)),
    st.tuples(st.integers(1, 32), st.integers(1, 32), st.integers(1, 4)),
)

_matrix_flat9 = st.lists(
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    min_size=9,
    max_size=9,
)

_matrix_3x3 = st.lists(
    st.lists(
        st.floats(allow_nan=True, allow_infinity=True, width=64),
        min_size=3,
        max_size=3,
    ),
    min_size=3,
    max_size=3,
)

_matrix_wrong_len = st.lists(_scalar, min_size=0, max_size=20).filter(lambda x: len(x) != 9)

_matrix_payload = st.one_of(
    st.none(),
    _matrix_flat9,
    _matrix_3x3,
    _matrix_wrong_len,
    _scalar,
    _ragged_seq,
)


# ---------------------------------------------------------------------------
# Invariant 1: metadata validation only raises ValueError
# ---------------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(wb=_wb_strategy, bl=_bl_strategy, raw_shape=_raw_shape_strategy)
def test_validate_raw_metadata_strict_exception_typing(wb, bl, raw_shape):
    """_validate_raw_metadata must only raise ValueError (never TypeError etc.)."""
    raw = _make_raw_stub(wb, bl, raw_shape)
    decoder = LinearDecoder()
    try:
        decoder._validate_raw_metadata(raw)
    except _ALLOWED_METADATA_EXC:
        return
    except Exception as exc:
        pytest.fail(f"_validate_raw_metadata leaked invalid exception type " f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Invariant 2: matrix selection never raises; valid result is (9,)
# ---------------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(cm=_matrix_payload, xyz=_matrix_payload)
def test_select_valid_color_matrix_never_raises(cm, xyz):
    """_select_valid_color_matrix must never raise under any input."""
    decoder = LinearDecoder()
    try:
        result = decoder._select_valid_color_matrix(cm, xyz)
    except Exception as exc:
        pytest.fail(f"_select_valid_color_matrix must never raise, leaked " f"{type(exc).__name__}: {exc}")
    if result is not None:
        arr = np.asarray(result, dtype=np.float64)
        assert arr.size == 9, f"Valid result must have 9 elements, got shape {arr.shape}"


# ---------------------------------------------------------------------------
# Invariant 3: RAW color space detection only raises ColorSpaceError
# ---------------------------------------------------------------------------


@settings(max_examples=80, deadline=None)
@given(cm=_matrix_payload, xyz=_matrix_payload)
def test_detect_raw_color_space_strict_typing(cm, xyz):
    """_detect_raw_color_space must only raise ColorSpaceError."""
    import tempfile

    class _FakeRaw:
        def __init__(self, color_matrix: Any, rgb_xyz_matrix: Any) -> None:
            self.color_matrix = color_matrix
            self.rgb_xyz_matrix = rgb_xyz_matrix

        def __enter__(self) -> "_FakeRaw":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

    # Store original rawpy module if it exists
    original_rawpy = sys.modules.get("rawpy")

    try:
        # Create fake rawpy module
        fake_rawpy = types.ModuleType("rawpy")
        fake_rawpy.imread = lambda _: _FakeRaw(cm, xyz)
        sys.modules["rawpy"] = fake_rawpy

        decoder = LinearDecoder()

        # Create temp file without using pytest fixture
        with tempfile.NamedTemporaryFile(suffix=".dng", delete=False) as f:
            tmp_path = Path(f.name)
            tmp_path.write_bytes(b"x")

        try:
            decoder._detect_raw_color_space(tmp_path)
        except _ALLOWED_COLORSPACE_EXC:
            pass
        except Exception as exc:
            pytest.fail(f"_detect_raw_color_space leaked invalid exception type " f"{type(exc).__name__}: {exc}")
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
    finally:
        # Restore original rawpy module
        if original_rawpy is not None:
            sys.modules["rawpy"] = original_rawpy
        else:
            sys.modules.pop("rawpy", None)


# ---------------------------------------------------------------------------
# Frozen regression corpus (deterministic replay)
# ---------------------------------------------------------------------------


def _coerce_wb_bl(value: Any) -> Any:
    """Coerce JSON-loaded WB/BL values: list of mixed numerics/strings pass through."""
    return value


@pytest.mark.parametrize(
    "case",
    load_corpus_cases(),
    ids=lambda c: c.get("name", "case"),
)
def test_ingest_fuzz_corpus(case: dict[str, Any]) -> None:
    """Replay frozen corpus cases deterministically."""
    kind = case.get("kind")
    decoder = LinearDecoder()

    if kind == "metadata":
        raw = _make_raw_stub(
            wb=case.get("camera_whitebalance"),
            bl=case.get("black_level_per_channel"),
            raw_image_shape=(tuple(case["raw_image_shape"]) if case.get("raw_image_shape") else None),
        )
        expect = case["expect"]
        if expect.get("raises") is None:
            decoder._validate_raw_metadata(raw)
        else:
            with pytest.raises(ValueError) as ei:
                decoder._validate_raw_metadata(raw)
            if expect.get("msg_contains"):
                assert expect["msg_contains"] in str(ei.value)

    elif kind == "matrix":
        result = decoder._select_valid_color_matrix(
            case.get("color_matrix"),
            case.get("rgb_xyz_matrix"),
        )
        if case["expect"]["returns"] == "none":
            assert result is None
        else:
            assert result is not None
            assert np.asarray(result).size == 9

    else:
        raise AssertionError(f"Unknown corpus case kind: {kind!r}")
