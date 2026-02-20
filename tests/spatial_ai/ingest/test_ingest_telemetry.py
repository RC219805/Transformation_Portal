"""Tests for ingest boundary telemetry instrumentation.

Validates that LinearDecoder emits structured events at key decision points:
- Validation failures (field, reason)
- Matrix fallbacks (from_, to)
- Postprocess guard violations (dtype/shape)

Zero overhead when telemetry is not provided (default NullTelemetry).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.ingest import LinearDecoder
from transformation_portal.spatial_ai.ingest.telemetry import IngestTelemetry, NullTelemetry


class ListTelemetry:
    """Test helper: accumulates events in a list."""

    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


class TestNullTelemetryDefault:
    """Test that NullTelemetry is the default and is a no-op."""

    def test_null_telemetry_is_default(self):
        """LinearDecoder without telemetry arg uses NullTelemetry."""
        decoder = LinearDecoder()
        assert isinstance(decoder._telemetry, NullTelemetry)

    def test_null_telemetry_emit_does_not_raise(self):
        """NullTelemetry.emit() is a no-op and never raises."""
        telemetry = NullTelemetry()
        telemetry.emit("test.event", field="value", reason="test")
        # No assertion needed — just verify no exception


class TestValidationFailureEvents:
    """Test telemetry emission on metadata validation failures."""

    def _make_raw(self, *, wb=None, bl=None, raw_image_shape=None):
        """Build a minimal mock rawpy object with configurable attributes."""
        raw = types.SimpleNamespace()
        if wb is not None:
            raw.camera_whitebalance = wb
        if bl is not None:
            raw.black_level_per_channel = bl
        if raw_image_shape is not None:
            raw.raw_image = np.zeros(raw_image_shape, dtype=np.uint16)
        return raw

    def test_wb_non_numeric_emits_validation_failed(self):
        """Non-numeric WB emits ingest.validation_failed with reason=non_numeric."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(wb="bad_value")

        with pytest.raises(ValueError, match="unparseable to float64"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "camera_whitebalance"
        assert fields["reason"] == "non_numeric"

    def test_wb_empty_emits_validation_failed(self):
        """Empty WB emits ingest.validation_failed with reason=empty."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(wb=[])

        with pytest.raises(ValueError, match="is empty"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "camera_whitebalance"
        assert fields["reason"] == "empty"

    def test_wb_invalid_channel_count_emits_validation_failed(self):
        """WB with wrong channel count emits validation_failed with reason=invalid_channel_count."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(wb=[1.0, 1.0])  # Expected 4, got 2

        with pytest.raises(ValueError, match="unexpected channel count"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "camera_whitebalance"
        assert fields["reason"] == "invalid_channel_count"

    def test_wb_nan_emits_validation_failed(self):
        """WB with NaN emits validation_failed with reason=nan."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(wb=[1.0, np.nan, 1.0, 1.0])

        with pytest.raises(ValueError, match="NaN values"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "camera_whitebalance"
        assert fields["reason"] == "nan"

    def test_wb_inf_emits_validation_failed(self):
        """WB with infinity emits validation_failed with reason=inf."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(wb=[1.0, 1.0, np.inf, 1.0])

        with pytest.raises(ValueError, match="infinity values"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "camera_whitebalance"
        assert fields["reason"] == "inf"

    def test_wb_non_positive_emits_validation_failed(self):
        """WB with zero/negative gain emits validation_failed with reason=non_positive."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(wb=[1.0, 0.0, 1.0, 1.0])  # Zero is non-positive

        with pytest.raises(ValueError, match="zero or negative gain"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "camera_whitebalance"
        assert fields["reason"] == "non_positive"

    def test_bl_non_numeric_emits_validation_failed(self):
        """Non-numeric BL emits ingest.validation_failed with reason=non_numeric."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(bl="bad_value")

        with pytest.raises(ValueError, match="unparseable to float64"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "black_level_per_channel"
        assert fields["reason"] == "non_numeric"

    def test_bl_empty_emits_validation_failed(self):
        """Empty BL emits validation_failed with reason=empty."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(bl=[])

        with pytest.raises(ValueError, match="is empty"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "black_level_per_channel"
        assert fields["reason"] == "empty"

    def test_bl_nan_emits_validation_failed(self):
        """BL with NaN emits validation_failed with reason=nan."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(bl=[512.0, np.nan, 512.0])

        with pytest.raises(ValueError, match="NaN"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "black_level_per_channel"
        assert fields["reason"] == "nan"

    def test_bl_inf_emits_validation_failed(self):
        """BL with infinity emits validation_failed with reason=inf."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(bl=[512.0, np.inf, 512.0])

        with pytest.raises(ValueError, match="infinity values"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "black_level_per_channel"
        assert fields["reason"] == "inf"

    def test_bl_negative_emits_validation_failed(self):
        """BL with negative value emits validation_failed with reason=negative."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(bl=[512.0, -10.0, 512.0])

        with pytest.raises(ValueError, match="negative values"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "black_level_per_channel"
        assert fields["reason"] == "negative"

    def test_bl_invalid_channel_count_emits_validation_failed(self):
        """BL with invalid channel count emits validation_failed."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(bl=[512.0, 512.0])  # Expected 1, 3, or 4; got 2

        with pytest.raises(ValueError, match="unexpected channel count"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "black_level_per_channel"
        assert fields["reason"] == "invalid_channel_count"

    def test_raw_image_wrong_ndim_emits_validation_failed(self):
        """raw_image with wrong ndim emits validation_failed with reason=wrong_ndim."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(raw_image_shape=(100, 100, 3))  # Expected 2D, got 3D

        with pytest.raises(ValueError, match="expected 2D"):
            decoder._validate_raw_metadata(raw)

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.validation_failed"
        assert fields["field"] == "raw_image"
        assert fields["reason"] == "wrong_ndim"

    def test_no_event_on_valid_metadata(self):
        """Valid metadata does not emit any events."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)
        raw = self._make_raw(
            wb=[2.0, 1.0, 1.5, 1.0],
            bl=[512, 512, 512, 512],
            raw_image_shape=(100, 100),
        )

        decoder._validate_raw_metadata(raw)  # Must not raise

        assert telemetry.events == []


class TestMatrixFallbackEvent:
    """Test telemetry emission when color_matrix is rejected and rgb_xyz_matrix is used."""

    def test_matrix_fallback_emits_event(self):
        """color_matrix invalid → rgb_xyz_matrix fallback emits ingest.matrix_fallback_used."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)

        zero_color_matrix = np.zeros(9)  # Invalid (norm < 1e-6)
        valid_rgb_xyz_matrix = np.eye(3)  # Valid

        result = decoder._select_valid_color_matrix(zero_color_matrix, valid_rgb_xyz_matrix)

        assert result is not None
        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.matrix_fallback_used"
        assert fields["from_"] == "color_matrix"
        assert fields["to"] == "rgb_xyz_matrix"

    def test_no_event_when_color_matrix_valid(self):
        """No event when color_matrix is valid (no fallback)."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)

        valid_color_matrix = np.eye(3)
        fallback_rgb_xyz = np.eye(3) * 2.0

        result = decoder._select_valid_color_matrix(valid_color_matrix, fallback_rgb_xyz)

        assert result is not None
        assert telemetry.events == []

    def test_no_event_when_color_matrix_none(self):
        """No event when color_matrix is None (not present-but-invalid)."""
        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)

        result = decoder._select_valid_color_matrix(None, np.eye(3))

        assert result is not None
        assert telemetry.events == []


class TestPostprocessGuardEvents:
    """Test telemetry emission on postprocess guard failures."""

    def _install_fake_rawpy(self, monkeypatch, postprocess_output):
        """Install a fake rawpy module that returns postprocess_output."""
        fake_rawpy_module = types.ModuleType("rawpy")

        class _FakeRaw:
            def __init__(self, path: str):
                self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
                self.black_level_per_channel = [512, 512, 512, 512]
                self.raw_image = np.zeros((100, 100), dtype=np.uint16)
                self.color_matrix = None
                self.rgb_xyz_matrix = np.eye(3)

            def postprocess(self, **kwargs):
                return postprocess_output

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _fake_imread(path: str):
            return _FakeRaw(path)

        fake_rawpy_module.imread = _fake_imread
        fake_rawpy_module.ColorSpace = types.SimpleNamespace(sRGB=1)
        fake_rawpy_module.DemosaicAlgorithm = types.SimpleNamespace(AHD=3)
        fake_rawpy_module.HighlightMode = types.SimpleNamespace(Clip=0)

        monkeypatch.setitem(vars(), "rawpy", fake_rawpy_module)
        # monkeypatch.setitem(vars(LinearDecoder), "rawpy", fake_rawpy_module)

        # Also inject into sys.modules
        import sys

        monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy_module)

    def test_postprocess_dtype_guard_emits_event(self, tmp_path, monkeypatch):
        """postprocess returning wrong dtype emits ingest.postprocess_guard_failed."""
        # Fake postprocess returns float32 instead of uint16
        fake_output = np.random.rand(100, 100, 3).astype(np.float32)
        self._install_fake_rawpy(monkeypatch, fake_output)

        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)

        raw_path = tmp_path / "test.dng"
        raw_path.write_bytes(b"fake")

        with pytest.raises(RuntimeError, match="expected uint16"):
            decoder._decode_raw(raw_path, "RAW")

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.postprocess_guard_failed"
        assert fields["reason"] == "dtype_mismatch"
        assert "float32" in fields["dtype"]

    def test_postprocess_shape_guard_emits_event(self, tmp_path, monkeypatch):
        """postprocess returning wrong shape emits ingest.postprocess_guard_failed."""
        # Fake postprocess returns wrong shape
        fake_output = np.zeros((100, 100), dtype=np.uint16)  # 2D instead of (H, W, 3)
        self._install_fake_rawpy(monkeypatch, fake_output)

        telemetry = ListTelemetry()
        decoder = LinearDecoder(telemetry=telemetry)

        raw_path = tmp_path / "test.dng"
        raw_path.write_bytes(b"fake")

        with pytest.raises(RuntimeError, match="expected .* from postprocess"):
            decoder._decode_raw(raw_path, "RAW")

        assert len(telemetry.events) == 1
        event, fields = telemetry.events[0]
        assert event == "ingest.postprocess_guard_failed"
        assert fields["reason"] == "shape_mismatch"
        assert fields["shape"] == (100, 100)
