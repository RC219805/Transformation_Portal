"""Tests for validation guardrails.

Tests cover:
- Bit depth validation (strict/non-strict modes)
- Dtype validation (float32 enforcement)
- Gamma validation (linear light requirement)
- Range validation (NaN/Inf/negative detection)
- Schema version validation
- Comprehensive linear output validation

Architecture: ADR-023, Issue #890 Phase I
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transformation_portal.spatial_ai.ingest import (
    CURRENT_SCHEMA_VERSION,
    BitDepthViolationError,
    LinearityViolationError,
    RangeViolationError,
    SchemaVersionError,
    validate_bit_depth,
    validate_dtype,
    validate_gamma,
    validate_linear_output,
    validate_range,
    validate_schema_version,
)


class TestBitDepthValidation:
    """Test bit depth validation."""

    def test_8bit_rejected_when_strict(self, tmp_path: Path):
        """Test that 8-bit input is rejected with strict=True."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        path = tmp_path / "test.png"

        with pytest.raises(BitDepthViolationError):
            validate_bit_depth(path, arr, min_bits=16, strict=True)

    def test_8bit_allowed_when_not_strict(self, tmp_path: Path):
        """Test that 8-bit input is allowed with strict=False."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        path = tmp_path / "test.png"

        # Should not raise
        validate_bit_depth(path, arr, min_bits=16, strict=False)

    def test_16bit_always_allowed(self, tmp_path: Path):
        """Test that 16-bit input is always allowed."""
        arr = np.zeros((10, 10, 3), dtype=np.uint16)
        path = tmp_path / "test.png"

        # Should not raise
        validate_bit_depth(path, arr, min_bits=16, strict=True)
        validate_bit_depth(path, arr, min_bits=16, strict=False)

    def test_float32_always_allowed(self, tmp_path: Path):
        """Test that float32 input is always allowed."""
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        path = tmp_path / "test.exr"

        # Should not raise
        validate_bit_depth(path, arr, min_bits=16, strict=True)


class TestDtypeValidation:
    """Test dtype validation."""

    def test_float32_passes(self):
        """Test that float32 dtype passes validation."""
        arr = np.zeros((10, 10, 3), dtype=np.float32)

        # Should not raise
        validate_dtype(arr, expected_dtype=np.dtype(np.float32))

    def test_float16_rejected(self):
        """Test that float16 dtype is rejected when expecting float32."""
        arr = np.zeros((10, 10, 3), dtype=np.float16)

        with pytest.raises(LinearityViolationError, match="dtype"):
            validate_dtype(arr, expected_dtype=np.dtype(np.float32))

    def test_uint16_rejected(self):
        """Test that uint16 dtype is rejected when expecting float32."""
        arr = np.zeros((10, 10, 3), dtype=np.uint16)

        with pytest.raises(LinearityViolationError, match="dtype"):
            validate_dtype(arr, expected_dtype=np.dtype(np.float32))


class TestGammaValidation:
    """Test gamma validation."""

    def test_linear_gamma_passes(self):
        """Test that gamma=1.0 passes validation."""
        # Should not raise
        validate_gamma(1.0, expected=1.0)

    def test_gamma_2_2_rejected(self):
        """Test that gamma=2.2 is rejected."""
        with pytest.raises(LinearityViolationError, match="gamma"):
            validate_gamma(2.2, expected=1.0)

    def test_gamma_tolerance(self):
        """Test that gamma within tolerance passes."""
        # Should not raise (within 1e-6 tolerance)
        validate_gamma(1.0000001, expected=1.0, tolerance=1e-6)

        # Should raise (outside tolerance)
        with pytest.raises(LinearityViolationError):
            validate_gamma(1.001, expected=1.0, tolerance=1e-6)


class TestRangeValidation:
    """Test range validation."""

    def test_valid_range_passes(self):
        """Test that valid range [0, 1] passes."""
        arr = np.random.rand(10, 10, 3).astype(np.float32)

        # Should not raise
        validate_range(arr)

    def test_hdr_values_allowed(self):
        """Test that HDR values >1.0 are allowed."""
        arr = np.random.rand(10, 10, 3).astype(np.float32) * 5.0  # Range [0, 5]

        # Should not raise (allow_above_one=True by default)
        validate_range(arr, allow_above_one=True)

    def test_nan_detected(self):
        """Test that NaN values are detected and rejected."""
        arr = np.array([[[0.5, 1.0, np.nan]]], dtype=np.float32)

        with pytest.raises(RangeViolationError, match="NaN"):
            validate_range(arr, check_nan=True)

    def test_inf_detected(self):
        """Test that infinite values are detected and rejected."""
        arr = np.array([[[0.5, np.inf, 1.0]]], dtype=np.float32)

        with pytest.raises(RangeViolationError, match="Infinite"):
            validate_range(arr, check_inf=True)

    def test_negative_rejected(self):
        """Test that negative values are rejected."""
        arr = np.array([[[-0.1, 0.5, 1.0]]], dtype=np.float32)

        with pytest.raises(RangeViolationError, match="negative"):
            validate_range(arr, allow_negative=False)

    def test_negative_allowed_when_permitted(self):
        """Test that negative values can be allowed."""
        arr = np.array([[[-0.1, 0.5, 1.0]]], dtype=np.float32)

        # Should not raise
        validate_range(arr, allow_negative=True, check_nan=False, check_inf=False)


class TestSchemaVersionValidation:
    """Test schema version validation."""

    def test_current_version_passes(self, tmp_path: Path):
        """Test that current schema version passes."""
        manifest = {"schema_version": CURRENT_SCHEMA_VERSION, "data": []}
        path = tmp_path / "manifest.json"

        # Should not raise
        validate_schema_version(manifest, path)

    def test_unsupported_version_rejected(self, tmp_path: Path):
        """Test that unsupported version is rejected."""
        manifest = {"schema_version": "99.0.0", "data": []}
        path = tmp_path / "manifest.json"

        with pytest.raises(SchemaVersionError):
            validate_schema_version(manifest, path)

    def test_missing_version_field_rejected(self, tmp_path: Path):
        """Test that missing schema_version field is rejected."""
        manifest = {"data": []}
        path = tmp_path / "manifest.json"

        with pytest.raises(KeyError, match="schema_version"):
            validate_schema_version(manifest, path)


class TestLinearOutputValidation:
    """Test comprehensive linear output validation."""

    def test_valid_linear_output_passes(self, tmp_path: Path):
        """Test that valid linear output passes all checks."""
        arr = np.random.rand(10, 10, 3).astype(np.float32)
        gamma = 1.0
        path = tmp_path / "test.tiff"

        # Should not raise
        validate_linear_output(arr, gamma, path)

    def test_invalid_gamma_rejected(self, tmp_path: Path):
        """Test that non-linear gamma is rejected."""
        arr = np.random.rand(10, 10, 3).astype(np.float32)
        gamma = 2.2
        path = tmp_path / "test.tiff"

        with pytest.raises(LinearityViolationError, match="gamma"):
            validate_linear_output(arr, gamma, path)

    def test_invalid_dtype_rejected(self, tmp_path: Path):
        """Test that non-float32 dtype is rejected."""
        arr = np.random.rand(10, 10, 3).astype(np.float16)
        gamma = 1.0
        path = tmp_path / "test.tiff"

        with pytest.raises(LinearityViolationError, match="dtype"):
            validate_linear_output(arr, gamma, path)

    def test_nan_values_rejected(self, tmp_path: Path):
        """Test that NaN values are rejected."""
        arr = np.array([[[0.5, 1.0, np.nan]]], dtype=np.float32)
        gamma = 1.0
        path = tmp_path / "test.tiff"

        with pytest.raises(RangeViolationError):
            validate_linear_output(arr, gamma, path)

    def test_negative_values_rejected(self, tmp_path: Path):
        """Test that negative values are rejected."""
        arr = np.array([[[-0.1, 0.5, 1.0]]], dtype=np.float32)
        gamma = 1.0
        path = tmp_path / "test.tiff"

        with pytest.raises(RangeViolationError):
            validate_linear_output(arr, gamma, path)

    def test_hdr_values_accepted(self, tmp_path: Path):
        """Test that HDR values >1.0 are accepted."""
        arr = np.random.rand(10, 10, 3).astype(np.float32) * 5.0
        gamma = 1.0
        path = tmp_path / "test.exr"

        # Should not raise
        validate_linear_output(arr, gamma, path)


# Pytest markers
pytestmark = [
    pytest.mark.unit,  # Fast unit tests
]
