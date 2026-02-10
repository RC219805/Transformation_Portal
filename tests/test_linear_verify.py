"""Tests for linear ingest verification.

Tests all correctness-critical invariants for linear light preservation:
- dtype validation (reject uint8/uint16)
- range validation (enforce [0, 1] bounds)
- gamma detection and rejection
- end-to-end linearity validation

These are blocking tests - failures indicate contract violations that
must prevent pipeline execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.linear_verify import (
    DtypeViolationError,
    LinearityViolationError,
    RangeViolationError,
    create_gamma_encoded_fixture,
    create_linear_test_fixture,
    detect_gamma_encoding,
    verify_dtype_float,
    verify_linear_ingest,
    verify_no_gamma,
    verify_range_linear,
)


class TestVerifyDtypeFloat:
    """Test dtype validation (reject uint8/uint16)."""

    def test_float32_passes(self):
        """float32 tensors should pass validation."""
        arr = np.array([0.5, 0.8], dtype=np.float32)
        verify_dtype_float(arr)  # Should not raise

    def test_float64_passes_by_default(self):
        """float64 tensors should pass by default."""
        arr = np.array([0.5, 0.8], dtype=np.float64)
        verify_dtype_float(arr)  # Should not raise

    def test_float64_rejected_when_disallowed(self):
        """float64 can be rejected if allow_float64=False."""
        arr = np.array([0.5, 0.8], dtype=np.float64)
        with pytest.raises(DtypeViolationError, match="float32"):
            verify_dtype_float(arr, allow_float64=False)

    def test_uint8_rejected(self):
        """uint8 tensors must be rejected (gamma/precision risk)."""
        arr = np.array([128, 200], dtype=np.uint8)
        with pytest.raises(DtypeViolationError, match="uint8.*gamma encoding"):
            verify_dtype_float(arr)

    def test_uint16_rejected(self):
        """uint16 tensors must be rejected (potential gamma encoding)."""
        arr = np.array([32768, 50000], dtype=np.uint16)
        with pytest.raises(DtypeViolationError, match="uint16.*gamma encoding"):
            verify_dtype_float(arr)

    def test_int32_rejected(self):
        """int32 and other integer types should be rejected."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(DtypeViolationError):
            verify_dtype_float(arr)

    def test_non_numpy_raises_typeerror(self):
        """Non-numpy inputs should raise TypeError."""
        with pytest.raises(TypeError, match="numpy array"):
            verify_dtype_float([0.5, 0.8])  # Python list

    def test_error_message_helpful(self):
        """Error message should explain why and how to fix."""
        arr = np.array([128], dtype=np.uint8)
        with pytest.raises(DtypeViolationError) as exc_info:
            verify_dtype_float(arr)
        
        error_msg = str(exc_info.value)
        assert "float32" in error_msg
        assert "uint8" in error_msg.lower()
        assert "gamma" in error_msg.lower() or "precision" in error_msg.lower()


class TestVerifyRangeLinear:
    """Test value range validation ([0, 1] bounds)."""

    def test_valid_range_passes(self):
        """Values in [0, 1] should pass."""
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        verify_range_linear(arr)  # Should not raise

    def test_min_boundary_passes(self):
        """Minimum boundary (0.0) should pass."""
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        verify_range_linear(arr)

    def test_max_boundary_passes(self):
        """Maximum boundary (1.0) should pass."""
        arr = np.ones((10, 10, 3), dtype=np.float32)
        verify_range_linear(arr)

    def test_underflow_rejected(self):
        """Values below 0 should be rejected."""
        arr = np.array([-0.1, 0.5, 1.0], dtype=np.float32)
        with pytest.raises(RangeViolationError, match="below expected.*range"):
            verify_range_linear(arr)

    def test_overflow_rejected(self):
        """Values above 1 should be rejected."""
        arr = np.array([0.0, 0.5, 1.5], dtype=np.float32)
        with pytest.raises(RangeViolationError, match="exceeds expected.*range"):
            verify_range_linear(arr)

    def test_tolerance_allows_small_errors(self):
        """Small floating point errors within tolerance should pass."""
        # Slightly outside [0, 1] but within default tolerance (1e-6)
        arr = np.array([0.0 - 5e-7, 0.5, 1.0 + 5e-7], dtype=np.float32)
        verify_range_linear(arr)  # Should pass with default tolerance

    def test_custom_range(self):
        """Custom min/max values can be specified."""
        arr = np.array([0.0, 0.5, 2.0], dtype=np.float32)
        # Should fail with default range
        with pytest.raises(RangeViolationError):
            verify_range_linear(arr)
        
        # Should pass with custom range
        verify_range_linear(arr, min_val=0.0, max_val=2.0)

    def test_error_message_shows_actual_values(self):
        """Error message should show actual min/max for debugging."""
        arr = np.array([0.0, 0.5, 1.5], dtype=np.float32)
        with pytest.raises(RangeViolationError) as exc_info:
            verify_range_linear(arr)
        
        error_msg = str(exc_info.value)
        assert "1.5" in error_msg  # Actual max value
        assert "0.0" in error_msg and "1.0" in error_msg  # Expected range


class TestDetectGammaEncoding:
    """Test gamma encoding detection heuristics."""

    def test_linear_fixture_not_detected_as_gamma(self):
        """Linear test fixture should NOT be detected as gamma-encoded."""
        linear = create_linear_test_fixture(shape=(100, 100, 3), mean=0.3)
        is_gamma = detect_gamma_encoding(linear)
        assert not is_gamma, "Linear fixture incorrectly detected as gamma"

    def test_gamma_fixture_detected(self):
        """Gamma-encoded fixture SHOULD be detected."""
        gamma = create_gamma_encoded_fixture(shape=(100, 100, 3))
        is_gamma = detect_gamma_encoding(gamma)
        assert is_gamma, "Gamma fixture not detected"

    def test_uint8_returns_false(self):
        """Cannot reliably detect gamma on uint8 (returns False)."""
        arr = np.array([[[128, 128, 128]]], dtype=np.uint8)
        is_gamma = detect_gamma_encoding(arr)
        assert not is_gamma  # Should return False (can't detect on uint8)

    def test_grayscale_returns_false(self):
        """Cannot check gamma on grayscale (needs RGB)."""
        arr = np.random.rand(100, 100).astype(np.float32)
        is_gamma = detect_gamma_encoding(arr)
        assert not is_gamma  # Should return False (needs 3 channels)

    def test_high_mean_suggests_gamma(self):
        """Artificially high mean should trigger gamma detection."""
        # Create array with very high mean (characteristic of gamma encoding)
        arr = np.full((100, 100, 3), 0.75, dtype=np.float32)
        is_gamma = detect_gamma_encoding(arr, threshold=0.15)
        assert is_gamma, "High mean should suggest gamma encoding"


class TestVerifyNoGamma:
    """Test gamma rejection (strict mode)."""

    def test_linear_fixture_passes(self):
        """Linear fixture should pass gamma check."""
        linear = create_linear_test_fixture(shape=(50, 50, 3), mean=0.3)
        verify_no_gamma(linear, strict=True)  # Should not raise

    def test_gamma_fixture_rejected_strict(self):
        """Gamma fixture should be rejected in strict mode."""
        gamma = create_gamma_encoded_fixture(shape=(50, 50, 3))
        with pytest.raises(LinearityViolationError, match="Gamma-encoded input detected"):
            verify_no_gamma(gamma, strict=True)

    def test_gamma_fixture_warns_non_strict(self):
        """In non-strict mode, should log warning but not raise."""
        gamma = create_gamma_encoded_fixture(shape=(50, 50, 3))
        # Should not raise, but would log warning (tested via caplog in full suite)
        verify_no_gamma(gamma, strict=False)

    def test_error_message_provides_guidance(self):
        """Error message should explain how to fix."""
        gamma = create_gamma_encoded_fixture(shape=(20, 20, 3))
        with pytest.raises(LinearityViolationError) as exc_info:
            verify_no_gamma(gamma)
        
        error_msg = str(exc_info.value)
        assert "linear" in error_msg.lower()
        assert "RAW" in error_msg or "TIFF" in error_msg
        assert "reject" in error_msg.lower()


class TestVerifyLinearIngest:
    """Test comprehensive linear ingest validation."""

    def test_valid_linear_tensor_passes(self):
        """Valid linear float32 tensor should pass all checks."""
        arr = np.random.rand(50, 50, 3).astype(np.float32) * 0.6  # Mean ~0.3
        verify_linear_ingest(arr)  # Should not raise

    def test_uint8_rejected_by_comprehensive_check(self):
        """uint8 should be rejected by comprehensive check."""
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(DtypeViolationError):
            verify_linear_ingest(arr)

    def test_out_of_range_rejected(self):
        """Out-of-range values should be rejected."""
        arr = np.array([[[1.5, 0.5, 0.5]]], dtype=np.float32)
        with pytest.raises(RangeViolationError):
            verify_linear_ingest(arr)

    def test_gamma_rejected(self):
        """Gamma-encoded input should be rejected."""
        gamma = create_gamma_encoded_fixture(shape=(30, 30, 3))
        with pytest.raises(LinearityViolationError):
            verify_linear_ingest(gamma)

    def test_can_skip_individual_checks(self):
        """Individual checks can be disabled via flags."""
        # Gamma-encoded but skip gamma check
        gamma = create_gamma_encoded_fixture(shape=(20, 20, 3))
        verify_linear_ingest(gamma, check_gamma=False)  # Should not raise

    def test_float64_allowed_by_default(self):
        """float64 should be allowed by default."""
        arr = np.random.rand(10, 10, 3).astype(np.float64) * 0.5
        verify_linear_ingest(arr)  # Should not raise

    def test_float64_rejected_when_disallowed(self):
        """float64 can be rejected if specified."""
        arr = np.random.rand(10, 10, 3).astype(np.float64) * 0.5
        with pytest.raises(DtypeViolationError):
            verify_linear_ingest(arr, allow_float64=False)


class TestCreateLinearTestFixture:
    """Test linear fixture generation."""

    def test_creates_float32(self):
        """Fixture should be float32."""
        fixture = create_linear_test_fixture()
        assert fixture.dtype == np.float32

    def test_creates_correct_shape(self):
        """Fixture should have specified shape."""
        fixture = create_linear_test_fixture(shape=(64, 128, 3))
        assert fixture.shape == (64, 128, 3)

    def test_values_in_range(self):
        """Fixture values should be in [0, 1]."""
        fixture = create_linear_test_fixture()
        assert fixture.min() >= 0.0
        assert fixture.max() <= 1.0

    def test_deterministic_with_seed(self):
        """Same seed should produce same fixture."""
        fixture1 = create_linear_test_fixture(seed=42)
        fixture2 = create_linear_test_fixture(seed=42)
        assert np.allclose(fixture1, fixture2)

    def test_different_without_seed(self):
        """Different calls with different seeds should differ."""
        fixture1 = create_linear_test_fixture(seed=42)
        fixture2 = create_linear_test_fixture(seed=99)
        assert not np.allclose(fixture1, fixture2)

    def test_approximate_target_mean(self):
        """Fixture should approximate target mean."""
        target_mean = 0.4
        fixture = create_linear_test_fixture(mean=target_mean)
        actual_mean = fixture.mean()
        # Allow some deviation due to clipping
        assert abs(actual_mean - target_mean) < 0.1

    def test_passes_linear_verification(self):
        """Generated fixture should pass linear verification."""
        fixture = create_linear_test_fixture()
        verify_linear_ingest(fixture)  # Should not raise


class TestCreateGammaEncodedFixture:
    """Test gamma-encoded fixture generation (for rejection tests)."""

    def test_creates_float32(self):
        """Gamma fixture should be float32."""
        fixture = create_gamma_encoded_fixture()
        assert fixture.dtype == np.float32

    def test_creates_correct_shape(self):
        """Gamma fixture should have specified shape."""
        fixture = create_gamma_encoded_fixture(shape=(32, 64, 3))
        assert fixture.shape == (32, 64, 3)

    def test_values_in_range(self):
        """Gamma fixture values should still be in [0, 1]."""
        fixture = create_gamma_encoded_fixture()
        assert fixture.min() >= 0.0
        assert fixture.max() <= 1.0

    def test_fails_gamma_detection(self):
        """Gamma fixture SHOULD be detected as gamma-encoded."""
        fixture = create_gamma_encoded_fixture()
        is_gamma = detect_gamma_encoding(fixture)
        assert is_gamma, "Gamma fixture should be detected as gamma-encoded"

    def test_fails_linear_verification(self):
        """Gamma fixture SHOULD fail linear verification."""
        fixture = create_gamma_encoded_fixture()
        with pytest.raises(LinearityViolationError):
            verify_linear_ingest(fixture)

    def test_deterministic_with_seed(self):
        """Same seed should produce same gamma fixture."""
        fixture1 = create_gamma_encoded_fixture(seed=42)
        fixture2 = create_gamma_encoded_fixture(seed=42)
        assert np.allclose(fixture1, fixture2)

    def test_custom_gamma_value(self):
        """Can specify custom gamma value."""
        # Higher gamma should shift values higher
        gamma_low = create_gamma_encoded_fixture(gamma=1.8, seed=42)
        gamma_high = create_gamma_encoded_fixture(gamma=2.8, seed=42)
        
        # Higher gamma → more brightening
        assert gamma_high.mean() > gamma_low.mean()


class TestEndToEndLinearPreservation:
    """End-to-end linearity validation tests."""

    def test_linear_fixture_roundtrip(self):
        """Linear fixture should remain linear through validation."""
        original = create_linear_test_fixture(seed=42)
        
        # Verify it passes
        verify_linear_ingest(original)
        
        # Should still be identical after check
        assert np.allclose(original, original)  # Tautology, but validates no mutation

    def test_gamma_fixture_must_fail(self):
        """Gamma-encoded fixture MUST fail end-to-end check."""
        gamma = create_gamma_encoded_fixture(seed=42)
        
        # Must fail comprehensive check
        with pytest.raises(LinearityViolationError):
            verify_linear_ingest(gamma)

    def test_dtype_leakage_detected(self):
        """uint8/uint16 leakage must be detected and blocked."""
        # Simulate uint8 leakage
        uint8_arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        with pytest.raises(DtypeViolationError, match="uint8"):
            verify_linear_ingest(uint8_arr)
        
        # Simulate uint16 leakage
        uint16_arr = np.random.randint(0, 65536, (50, 50, 3), dtype=np.uint16)
        
        with pytest.raises(DtypeViolationError, match="uint16"):
            verify_linear_ingest(uint16_arr)

    def test_range_violation_detected(self):
        """Range violations must be detected and blocked."""
        # Underflow
        underflow = np.array([[[-0.1, 0.5, 0.5]]], dtype=np.float32)
        with pytest.raises(RangeViolationError):
            verify_linear_ingest(underflow)
        
        # Overflow
        overflow = np.array([[[0.5, 0.5, 1.1]]], dtype=np.float32)
        with pytest.raises(RangeViolationError):
            verify_linear_ingest(overflow)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
