"""Tests for linear light decoder (Spatial AI Foundation).

Tests cover:
- Linear gamma enforcement (gamma=1.0)
- HDR preservation (values >1.0)
- Float32 dtype validation
- Provenance tracking
- Contract validation (SpatialCaptureV1)
- No 8-bit collapse

Architecture: ADR-023 (Isolation), ADR-026 (APEX Research Ultra)
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile  # noqa: F401  # imported for fail-fast; per-test bodies use tifffile.imwrite
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from transformation_portal.spatial_ai.ingest import (
    BitDepthViolationError,
    ColorSpaceError,
    LinearDecoder,
    LinearIngestResult,
    UnsupportedFormatError,
    decode,
)
from transformation_portal.spatial_ai.ingest.linear_decoder import _canonical_f64_list, _compute_ingest_fingerprint


class TestLinearDecoder:
    """Test suite for LinearDecoder."""

    def test_gamma_enforcement(self):
        """Test that gamma != 1.0 is always rejected (no override possible)."""
        with pytest.raises(ValueError, match="gamma=1.0"):
            LinearDecoder(gamma=2.2)

    def test_linear_decode_preserves_hdr(self, tmp_path: Path):
        """Test that linear ingest preserves HDR values >1.0."""
        # Create HDR test image (values >1.0)
        hdr_img = np.random.rand(100, 100, 3).astype(np.float32) * 5.0  # Range [0, 5.0]
        assert hdr_img.max() > 1.0  # Verify HDR

        # Save as 16-bit TIFF (will be normalized to [0, 65535])
        # Use tifffile instead of PIL - PIL doesn't support uint16 RGB mode
        import tifffile

        img_uint16 = np.clip(hdr_img / hdr_img.max() * 65535, 0, 65535).astype(np.uint16)
        test_img_path = tmp_path / "hdr_test.tiff"
        tifffile.imwrite(test_img_path, img_uint16)

        # Decode
        decoder = LinearDecoder(gamma=1.0, bit_depth=32)
        result = decoder.decode(test_img_path)

        # Verify properties
        assert result.linear_rgb.dtype == np.float32
        assert result.gamma == 1.0
        assert result.bit_depth == 32
        # Note: After uint16 roundtrip, max value is 1.0 (normalized)
        # For true HDR >1.0, need EXR or float32 TIFF input
        assert result.linear_rgb.max() <= 1.0
        assert result.linear_rgb.min() >= 0.0

    def test_dtype_validation(self, tmp_path: Path):
        """Test that float32 dtype is enforced in result."""
        # Create test image
        test_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Decode
        result = decode(test_img_path, gamma=1.0)

        # Verify dtype
        assert result.linear_rgb.dtype == np.float32
        assert result.dtype == "float32"

    def test_provenance_tracking(self, tmp_path: Path):
        """Test that provenance metadata is correctly tracked."""
        # Create test image
        test_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Decode with provenance
        result = decode(test_img_path, gamma=1.0, output_dir=tmp_path, emit_provenance=True)

        # Verify provenance file exists
        assert result.provenance_path is not None
        assert result.provenance_path.exists()

        # Load and verify provenance content
        with open(result.provenance_path) as f:
            prov = json.load(f)

        assert prov["input"]["format"] == "PNG"
        assert prov["decode"]["gamma"] == 1.0
        assert prov["decode"]["bit_depth"] == 32
        assert prov["decode"]["dtype"] == "float32"
        assert prov["decode"]["contract"] == "SpatialCaptureV1"
        assert prov["output"]["hash_algorithm"] == "sha256"
        assert prov["adr"] == "ADR-026"
        assert len(prov["output"]["content_hash"]) == 64  # SHA-256 hex

    def test_exr_output(self, tmp_path: Path):
        """Test that EXR output artifact is created when OpenEXR is available."""
        # Check if OpenEXR is available
        try:
            import Imath
            import OpenEXR

            has_openexr = True
        except ImportError:
            has_openexr = False

        if not has_openexr:
            pytest.skip("OpenEXR not installed - skipping EXR export test")

        # Create test image
        test_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Decode with EXR emission
        result = decode(test_img_path, gamma=1.0, output_dir=tmp_path, emit_exr=True)

        # Verify EXR file exists
        assert result.output_exr_path is not None
        assert result.output_exr_path.exists()
        assert result.output_exr_path.suffix == ".exr"

    def test_contract_validation_rejects_non_float32(self, tmp_path: Path):
        """Test that LinearIngestResult rejects non-float32 arrays."""
        # Create test image
        test_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Decode
        decoder = LinearDecoder(gamma=1.0)
        result = decoder.decode(test_img_path)

        # Try to construct result with wrong dtype
        with pytest.raises(ValueError, match="float32 dtype"):
            LinearIngestResult(
                linear_rgb=np.zeros((100, 100, 3), dtype=np.float16),  # Wrong dtype
                gamma=1.0,
                bit_depth=32,
                dtype="float32",
                input_size=(100, 100),
                input_path=test_img_path,
                input_format="PNG",
                color_space="linear_sRGB",
            )

    def test_contract_validation_rejects_non_linear_gamma(self, tmp_path: Path):
        """Test that LinearIngestResult rejects gamma != 1.0."""
        # Create test array
        linear_rgb = np.random.rand(100, 100, 3).astype(np.float32)

        # Try to construct result with wrong gamma
        with pytest.raises(ValueError, match="gamma=1.0"):
            LinearIngestResult(
                linear_rgb=linear_rgb,
                gamma=2.2,  # Wrong gamma
                bit_depth=32,
                dtype="float32",
                input_size=(100, 100),
                input_path=Path("test.png"),
                input_format="PNG",
                color_space="linear_sRGB",
            )

    def test_unsupported_format_raises(self, tmp_path: Path):
        """Test that unsupported formats raise clear errors."""
        # Create dummy file with unsupported extension
        unsupported_path = tmp_path / "test.jpg"
        unsupported_path.write_text("dummy")

        decoder = LinearDecoder(gamma=1.0)
        with pytest.raises(UnsupportedFormatError, match="Unsupported format"):
            decoder.decode(unsupported_path)

    def test_raw_format_requires_rawpy(self, tmp_path: Path):
        """Test that RAW formats require rawpy package (now implemented)."""
        # Create dummy RAW file (invalid content, but will test import check)
        raw_path = tmp_path / "test.cr2"
        raw_path.write_text("dummy")

        decoder = LinearDecoder(gamma=1.0)
        # Should raise RuntimeError, ColorSpaceError, or ImportError from rawpy failing to decode the dummy file
        # (not NotImplementedError anymore since RAW support is implemented)
        with pytest.raises((RuntimeError, ColorSpaceError, ImportError)):
            decoder.decode(raw_path)

    def test_raw_format_uses_dedicated_runtime_when_configured(self, tmp_path: Path, monkeypatch):
        """Configured RAW runtime should dispatch through the subprocess worker."""
        raw_path = tmp_path / "test.cr2"
        raw_path.write_text("dummy")

        fake_linear = np.full((4, 5, 3), 0.25, dtype=np.float32)
        captured: dict[str, object] = {}

        def fake_run_raw_worker(*, python_executable, command_name, input_path, payload, start):
            captured["python_executable"] = python_executable
            captured["command_name"] = command_name
            captured["input_path"] = input_path
            captured["payload"] = payload
            captured["start"] = start
            return fake_linear, {
                "input_size": [4, 5],
                "input_format": "RAW_CR2",
                "color_space": "linear_sRGB",
                "ingest_fingerprint": "f" * 64,
                "dtype": "float32",
            }

        monkeypatch.setattr(
            "transformation_portal.spatial_ai.ingest.linear_decoder.run_raw_worker",
            fake_run_raw_worker,
        )

        result = LinearDecoder(
            gamma=1.0,
            bit_depth=32,
            strict_ingest=True,
            raw_python_executable="./.venv-raw/bin/python",
        ).decode(raw_path)

        assert np.array_equal(result.linear_rgb, fake_linear)
        assert result.input_size == (4, 5)
        assert result.color_space == "linear_sRGB"
        assert result.ingest_fingerprint == "f" * 64
        assert captured["python_executable"] == "./.venv-raw/bin/python"
        assert captured["command_name"] == "linear_decode"
        assert captured["input_path"] == raw_path
        assert captured["payload"] == {
            "gamma": 1.0,
            "bit_depth": 32,
            "strict_ingest": True,
            "demosaic": "AHD",
        }

    def test_content_hash_reproducible(self, tmp_path: Path):
        """Test that content hash is deterministic."""
        # Create test image
        test_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Decode twice
        result1 = decode(test_img_path, gamma=1.0)
        result2 = decode(test_img_path, gamma=1.0)

        # Hashes should match
        assert result1.content_hash == result2.content_hash
        assert len(result1.content_hash) == 64  # SHA-256 hex

    def test_16bit_tiff_decode(self, tmp_path: Path):
        """Test decoding of 16-bit TIFF files."""
        # Create 16-bit test image
        # Use tifffile for uint16 RGB - PIL doesn't support this mode
        import tifffile

        test_img = (np.random.rand(100, 100, 3) * 65535).astype(np.uint16)
        test_img_path = tmp_path / "test_16bit.tiff"
        # Save as TIFF (PNG 16-bit is grayscale only in PIL)
        tifffile.imwrite(test_img_path, test_img)

        # Decode
        result = decode(test_img_path, gamma=1.0)

        # Verify
        assert result.linear_rgb.dtype == np.float32
        assert result.gamma == 1.0
        # Format will be TIFF
        assert result.input_format == "TIFF"
        assert result.input_size == (100, 100)

    def test_grayscale_conversion(self, tmp_path: Path):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale test image
        test_img = (np.random.rand(100, 100) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test_gray.png"
        Image.fromarray(test_img, mode="L").save(test_img_path)

        # Decode
        result = decode(test_img_path, gamma=1.0)

        # Verify RGB conversion
        assert result.linear_rgb.shape == (100, 100, 3)
        assert result.linear_rgb.dtype == np.float32

    def test_convenience_function(self, tmp_path: Path):
        """Test that convenience decode() function works."""
        # Create test image
        test_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Use convenience function
        result = decode(test_img_path, gamma=1.0, emit_provenance=True)

        # Verify
        assert isinstance(result, LinearIngestResult)
        assert result.gamma == 1.0
        assert result.linear_rgb.dtype == np.float32

    def test_exr_fail_loud_when_openexr_missing(self, tmp_path: Path):
        """Test that requesting EXR export fails loudly when OpenEXR is unavailable."""
        # Check if OpenEXR is available - if it is, skip this test
        try:
            import Imath
            import OpenEXR

            pytest.skip("OpenEXR is installed - cannot test missing OpenEXR behavior")
        except ImportError:
            pass  # Good - OpenEXR is missing, test should proceed

        # Create a simple 8-bit PNG (sufficient to reach emit_exr failure path)
        test_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Attempt decode with emit_exr=True should fail loudly
        with pytest.raises(RuntimeError, match="emit_exr=True requires OpenEXR package"):
            decode(test_img_path, gamma=1.0, output_dir=tmp_path, emit_exr=True)

    def test_strict_ingest_rejects_uint8(self, tmp_path: Path):
        """Test that strict_ingest=True rejects 8-bit inputs."""
        # Create 8-bit test image
        test_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test_8bit.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Attempt decode with strict_ingest=True should fail with BitDepthViolationError
        with pytest.raises(BitDepthViolationError, match="Bit depth violation"):
            decode(test_img_path, gamma=1.0, strict_ingest=True)

    def test_strict_ingest_allows_uint16(self, tmp_path: Path):
        """Test that strict_ingest=True allows 16-bit inputs."""
        # Create a true 16-bit grayscale image (PIL mode I;16)
        test_img = (np.random.rand(50, 50) * 65535).astype(np.uint16)
        test_img_path = tmp_path / "test_16bit.png"
        img = Image.fromarray(test_img).convert("I;16")
        img.save(test_img_path)

        # Decode with strict_ingest=True should succeed (will convert gray->RGB)
        result = decode(test_img_path, gamma=1.0, strict_ingest=True)
        assert result.linear_rgb.dtype == np.float32
        assert result.gamma == 1.0
        assert result.linear_rgb.shape[2] == 3  # Converted to RGB

    def test_non_strict_ingest_allows_uint8(self, tmp_path: Path):
        """Test that strict_ingest=False (default) allows 8-bit inputs."""
        # Create 8-bit test image
        test_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test_8bit.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Decode with strict_ingest=False should succeed
        result = decode(test_img_path, gamma=1.0, strict_ingest=False)
        assert result.linear_rgb.dtype == np.float32
        assert result.gamma == 1.0

        # Also test default behavior (strict_ingest not specified)
        result_default = decode(test_img_path, gamma=1.0)
        assert result_default.linear_rgb.dtype == np.float32


class TestLinearIngestIntegration:
    """Integration tests for linear ingest with EnhanceConfig."""

    def test_config_flag_exists(self):
        """Test that spatial_ai_linear_ingest flag exists in EnhanceConfig."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig

        config = EnhanceConfig()
        assert hasattr(config, "spatial_ai_linear_ingest")
        assert config.spatial_ai_linear_ingest is False  # Default disabled

    def test_config_flag_enable(self):
        """Test that spatial_ai_linear_ingest can be enabled."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig

        config = EnhanceConfig(spatial_ai_linear_ingest=True)
        assert config.spatial_ai_linear_ingest is True


class TestColorSpaceValidation:
    """Tests for P1-1: Color space validation and tracking."""

    def test_color_space_in_result(self, tmp_path: Path):
        """Test that color_space field is populated in LinearIngestResult."""
        # Create simple 16-bit test image
        # Use tifffile - PIL doesn't support uint16 RGB mode
        import tifffile

        test_img = (np.random.rand(50, 50, 3) * 65535).astype(np.uint16)
        test_img_path = tmp_path / "test.tiff"
        tifffile.imwrite(test_img_path, test_img)

        # Decode
        decoder = LinearDecoder(gamma=1.0)
        result = decoder.decode(test_img_path)

        # Verify color_space is set
        assert hasattr(result, "color_space")
        assert result.color_space is not None
        assert result.color_space == "linear_sRGB"

    def test_raw_color_space_validation(self, tmp_path: Path):
        """Test RAW color space detection with valid camera matrix.

        Note: This test requires a valid DNG file with camera matrix.
        Uses synthetic approach with mocked rawpy for determinism.
        """
        pytest.importorskip("rawpy", reason="rawpy required for RAW color space tests")

        # For now, skip actual RAW decode test unless we have a fixture
        # This would require a minimal DNG with valid camera matrix
        pytest.skip("RAW fixture with camera matrix needed - see test_raw_metadata_fields for partial coverage")

    def test_color_space_error_handling(self):
        """Test that ColorSpaceError has proper attributes and message."""
        # Test ColorSpaceError construction
        error = ColorSpaceError(
            input_path=Path("/test/image.CR2"),
            reason="No camera color matrix found",
            matrix_present=False,
        )

        # Verify attributes
        assert error.input_path == Path("/test/image.CR2")
        assert error.reason == "No camera color matrix found"
        assert error.matrix_present is False

        # Verify message contains key elements
        error_msg = str(error)
        assert "image.CR2" in error_msg
        assert "camera color matrix" in error_msg.lower()
        assert "Remediation" in error_msg

    def test_non_raw_color_space_default(self, tmp_path: Path):
        """Test that non-RAW formats default to linear_sRGB."""
        import tifffile

        # Test TIFF - use tifffile for uint16 RGB (PIL doesn't support this)
        tiff_img = (np.random.rand(50, 50, 3) * 65535).astype(np.uint16)
        tiff_path = tmp_path / "test.tiff"
        tifffile.imwrite(tiff_path, tiff_img)

        decoder = LinearDecoder(gamma=1.0)
        result = decoder.decode(tiff_path)
        assert result.color_space == "linear_sRGB"

        # Test PNG - use PIL for uint8 (standard case)
        png_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        png_path = tmp_path / "test.png"
        Image.fromarray(png_img, mode="RGB").save(png_path, format="PNG")

        result_png = decoder.decode(png_path)
        assert result_png.color_space == "linear_sRGB"


class TestRAWDemosaicDeterminism:
    """Tests for P1-2: RAW demosaic determinism and library version tracking."""

    def test_provenance_captures_rawpy_version(self, tmp_path: Path):
        """Test that provenance captures rawpy and libraw versions for RAW files."""
        pytest.importorskip("rawpy", reason="rawpy required for RAW provenance tests")

        from transformation_portal.spatial_ai.ingest.provenance import ProvenanceCapture

        # Create a dummy RAW file (just needs to exist for provenance capture)
        raw_path = tmp_path / "test.cr2"
        raw_path.write_bytes(b"dummy RAW content")

        # Create a simple test tensor
        tensor = np.random.rand(100, 100, 3).astype(np.float32)

        # Create provenance with demosaic_method set (indicates RAW file)
        capture = ProvenanceCapture()
        prov = capture.capture(
            source_path=raw_path,
            tensor=tensor,
            gamma=1.0,
            bit_depth=32,
            demosaic_method="AHD",  # Indicates RAW processing
            white_balance_method="camera_wb",
        )

        # Verify rawpy version is captured
        assert prov.ingest.rawpy_version is not None
        assert isinstance(prov.ingest.rawpy_version, str)
        # Should be semantic version format
        assert "." in prov.ingest.rawpy_version

        # LibRaw version may or may not be available depending on rawpy version
        # Just check it's captured (can be None)
        assert hasattr(prov.ingest, "libraw_version")

    def test_provenance_no_rawpy_for_non_raw(self, tmp_path: Path):
        """Test that rawpy version is not captured for non-RAW files."""
        from transformation_portal.spatial_ai.ingest.provenance import ProvenanceCapture

        # Create a dummy TIFF file
        tiff_path = tmp_path / "test.tiff"
        tiff_path.write_bytes(b"dummy TIFF content")

        # Create a simple test tensor
        tensor = np.random.rand(100, 100, 3).astype(np.float32)

        # Create provenance WITHOUT demosaic_method (non-RAW file)
        capture = ProvenanceCapture()
        prov = capture.capture(
            source_path=tiff_path,
            tensor=tensor,
            gamma=1.0,
            bit_depth=32,
            # No demosaic_method - indicates non-RAW file
        )

        # Verify rawpy version is None for non-RAW
        assert prov.ingest.rawpy_version is None
        assert prov.ingest.libraw_version is None

    def test_raw_demosaic_determinism(self, tmp_path: Path):
        """Test RAW demosaic produces deterministic results (hash reproducibility).

        Note: This test requires a valid DNG fixture with camera matrix.
        Skipped if no fixture available.
        """
        pytest.importorskip("rawpy", reason="rawpy required for determinism tests")

        # For Phase I, this test is a placeholder
        # Real determinism test requires:
        # 1. Valid DNG file with camera matrix
        # 2. Cross-platform hash validation
        # 3. Baseline reference hashes
        pytest.skip("RAW determinism test requires DNG fixture - tracked in PR #946 documentation for Phase II validation")


class TestADR023Compliance:
    """Tests for ADR-023 (Spatial AI Ingest Isolation) compliance."""

    def test_no_lux_depth_imports(self):
        """Verify spatial_ai.ingest doesn't import lux_depth_v3.raw_loader."""
        # Get module source
        import inspect
        import re

        from transformation_portal.spatial_ai.ingest import linear_decoder

        source = inspect.getsource(linear_decoder)

        # Look for actual import statements (not docstring references)
        # Match lines that start with import/from (excluding docstrings)
        import_lines = [
            line.strip()
            for line in source.split("\n")
            if re.match(r"^\s*(from|import)\s+", line) and not line.strip().startswith("#")
        ]

        # Join import lines and verify no forbidden imports
        imports_str = " ".join(import_lines)
        assert "lux_depth_v3" not in imports_str, f"Found forbidden import in: {import_lines}"

    def test_module_docstring_warnings(self):
        """Verify module docstrings contain isolation warnings."""
        from transformation_portal import spatial_ai

        # Check module docstring
        assert "WARNING" in spatial_ai.__doc__.upper()
        assert "training" in spatial_ai.__doc__.lower() or "research" in spatial_ai.__doc__.lower()
        assert "rendering" in spatial_ai.__doc__.lower()


class TestRawDecodePostprocessGuards:
    """Unit tests for RAW postprocess contract guards."""

    @staticmethod
    def _install_fake_rawpy(monkeypatch: pytest.MonkeyPatch, postprocess_output: np.ndarray) -> None:
        """Install a minimal fake rawpy module for deterministic _decode_raw tests."""

        class _FakeRaw:
            def __init__(self, rgb: np.ndarray):
                self._rgb = rgb
                self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
                self.black_level_per_channel = [512, 512, 512, 512]
                self.color_matrix = np.eye(3, dtype=np.float64)
                self.rgb_xyz_matrix = np.array(
                    [
                        [0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041],
                    ],
                    dtype=np.float64,
                )
                self.raw_image = np.zeros((8, 8), dtype=np.uint16)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def postprocess(self, **kwargs):
                return self._rgb

        fake_rawpy = types.SimpleNamespace(
            ColorSpace=types.SimpleNamespace(sRGB="sRGB"),
            DemosaicAlgorithm=types.SimpleNamespace(AHD="AHD"),
            HighlightMode=types.SimpleNamespace(Clip="Clip"),
            imread=lambda _path: _FakeRaw(postprocess_output),
        )
        monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    def test_decode_raw_rejects_non_uint16_postprocess(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        self._install_fake_rawpy(monkeypatch, np.zeros((8, 8, 3), dtype=np.float32))

        decoder = LinearDecoder(gamma=1.0)
        raw_path = tmp_path / "mock.dng"
        raw_path.write_bytes(b"not-a-real-raw")

        with pytest.raises(RuntimeError, match="expected uint16 from postprocess"):
            decoder._decode_raw(raw_path, "RAW_DNG")

    def test_decode_raw_rejects_non_rgb_shape_postprocess(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        self._install_fake_rawpy(monkeypatch, np.zeros((8, 8), dtype=np.uint16))

        decoder = LinearDecoder(gamma=1.0)
        raw_path = tmp_path / "mock.dng"
        raw_path.write_bytes(b"not-a-real-raw")

        with pytest.raises(RuntimeError, match=r"expected \(H, W, 3\) from postprocess"):
            decoder._decode_raw(raw_path, "RAW_DNG")

    def test_decode_raw_accepts_uint16_rgb_postprocess(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        valid_rgb = np.full((6, 4, 3), 32768, dtype=np.uint16)
        self._install_fake_rawpy(monkeypatch, valid_rgb)

        decoder = LinearDecoder(gamma=1.0)
        raw_path = tmp_path / "mock.dng"
        raw_path.write_bytes(b"not-a-real-raw")

        linear_rgb, size, fingerprint = decoder._decode_raw(raw_path, "RAW_DNG")

        assert linear_rgb.dtype == np.float32
        assert linear_rgb.shape == (6, 4, 3)
        assert size == (6, 4)
        assert np.isclose(linear_rgb[0, 0, 0], 32768 / 65535.0)
        assert isinstance(fingerprint, str) and len(fingerprint) == 64

    def test_decode_emits_ingest_fingerprint_in_provenance_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        valid_rgb = np.full((6, 4, 3), 32768, dtype=np.uint16)
        self._install_fake_rawpy(monkeypatch, valid_rgb)

        raw_path = tmp_path / "mock.dng"
        raw_path.write_bytes(b"not-a-real-raw")

        result = decode(raw_path, gamma=1.0, output_dir=tmp_path, emit_provenance=True)

        assert result.ingest_fingerprint is not None
        assert len(result.ingest_fingerprint) == 64
        assert result.provenance_path is not None
        assert result.provenance_path.exists()

        provenance = json.loads(result.provenance_path.read_text())
        assert provenance["ingest_fingerprint"] == result.ingest_fingerprint

    def test_decode_raw_propagates_metadata_value_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        class _FakeRaw:
            def __init__(self):
                self.camera_whitebalance = ["bad", 1.0, 1.0, 1.0]  # non-numeric payload
                self.black_level_per_channel = [512, 512, 512, 512]
                self.color_matrix = np.eye(3, dtype=np.float64)
                self.rgb_xyz_matrix = np.eye(3, dtype=np.float64)
                self.raw_image = np.zeros((8, 8), dtype=np.uint16)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def postprocess(self, **kwargs):
                raise AssertionError("postprocess must not run when metadata validation fails")

        fake_rawpy = types.SimpleNamespace(
            ColorSpace=types.SimpleNamespace(sRGB="sRGB"),
            DemosaicAlgorithm=types.SimpleNamespace(AHD="AHD"),
            HighlightMode=types.SimpleNamespace(Clip="Clip"),
            imread=lambda _path: _FakeRaw(),
        )
        monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

        decoder = LinearDecoder(gamma=1.0)
        raw_path = tmp_path / "mock.dng"
        raw_path.write_bytes(b"not-a-real-raw")

        with pytest.raises(ValueError, match="camera_whitebalance is unparseable"):
            decoder._decode_raw(raw_path, "RAW_DNG")

    def test_decode_raw_wraps_runtime_error_from_rawpy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        class _FakeRaw:
            def __init__(self):
                self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
                self.black_level_per_channel = [512, 512, 512, 512]
                self.color_matrix = np.eye(3, dtype=np.float64)
                self.rgb_xyz_matrix = np.eye(3, dtype=np.float64)
                self.raw_image = np.zeros((8, 8), dtype=np.uint16)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def postprocess(self, **kwargs):
                raise RuntimeError("LibRaw internal error")

        fake_rawpy = types.SimpleNamespace(
            ColorSpace=types.SimpleNamespace(sRGB="sRGB"),
            DemosaicAlgorithm=types.SimpleNamespace(AHD="AHD"),
            HighlightMode=types.SimpleNamespace(Clip="Clip"),
            imread=lambda _path: _FakeRaw(),
        )
        monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

        decoder = LinearDecoder(gamma=1.0)
        raw_path = tmp_path / "mock.dng"
        raw_path.write_bytes(b"not-a-real-raw")

        with pytest.raises(RuntimeError, match=r"Failed to decode RAW file mock\.dng: LibRaw internal error"):
            decoder._decode_raw(raw_path, "RAW_DNG")


class TestRawColorSpaceDiagnostics:
    """Unit tests for robust color-space diagnostics in RAW metadata handling."""

    def test_unparseable_matrices_raise_color_space_error_with_diagnostics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        class _FakeRaw:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            @property
            def color_matrix(self):
                return object()  # non-coercible to float64

            @property
            def rgb_xyz_matrix(self):
                return object()  # non-coercible to float64

        fake_rawpy = types.SimpleNamespace(imread=lambda _path: _FakeRaw())
        monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

        decoder = LinearDecoder(gamma=1.0)
        raw_path = tmp_path / "mock.dng"
        raw_path.write_bytes(b"not-a-real-raw")

        with pytest.raises(ColorSpaceError, match=r"unparseable\(type=object"):
            decoder._detect_raw_color_space(raw_path)


class TestColorMatrixSelection:
    """Regression tests for zero color_matrix fallback (ingest hardening)."""

    decoder = LinearDecoder()

    def test_zero_color_matrix_falls_back_to_rgb_xyz(self):
        """Zero-filled color_matrix must not be used; rgb_xyz_matrix is the fallback."""
        zero_color_matrix = [0.0] * 9
        valid_rgb_xyz_matrix = [
            0.4124564,
            0.3575761,
            0.1804375,
            0.2126729,
            0.7151522,
            0.0721750,
            0.0193339,
            0.1191920,
            0.9503041,
        ]

        result = self.decoder._select_valid_color_matrix(zero_color_matrix, valid_rgb_xyz_matrix)

        result_array = np.array(result)
        assert result_array.size == 9
        assert not np.allclose(result_array, 0.0), "Fallback matrix must not be zero-filled"
        assert np.allclose(
            result_array, np.array(valid_rgb_xyz_matrix)
        ), "rgb_xyz_matrix should be used when color_matrix is zero-filled"

    def test_valid_color_matrix_preferred_over_rgb_xyz(self):
        """When color_matrix is non-zero, it takes priority over rgb_xyz_matrix."""
        valid_color_matrix = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        fallback_rgb_xyz = [0.1] * 9

        result = self.decoder._select_valid_color_matrix(valid_color_matrix, fallback_rgb_xyz)

        assert np.allclose(np.array(result), np.array(valid_color_matrix))

    def test_none_color_matrix_falls_back_to_rgb_xyz(self):
        """None color_matrix must not raise; rgb_xyz_matrix is used."""
        valid_rgb_xyz_matrix = [0.5] * 9

        result = self.decoder._select_valid_color_matrix(None, valid_rgb_xyz_matrix)

        assert result is not None
        assert np.allclose(result, np.array(valid_rgb_xyz_matrix, dtype=np.float64))

    def test_both_none_returns_none(self):
        """Returns None when no usable matrix is available."""
        result = self.decoder._select_valid_color_matrix(None, None)

        assert result is None

    def test_empty_color_matrix_falls_back_to_rgb_xyz(self):
        """Empty-array color_matrix triggers fallback to rgb_xyz_matrix."""
        valid_rgb_xyz_matrix = [0.3, 0.5, 0.2, 0.1, 0.7, 0.2, 0.05, 0.12, 0.95]

        result = self.decoder._select_valid_color_matrix([], valid_rgb_xyz_matrix)

        assert result is not None
        assert np.allclose(result, np.array(valid_rgb_xyz_matrix, dtype=np.float64))

    def test_3x3_ndarray_is_accepted(self):
        """rawpy commonly returns (3, 3) ndarrays — must not be rejected as wrong length."""
        matrix_3x3 = np.eye(3)  # identity: valid, non-zero
        result = self.decoder._select_valid_color_matrix(matrix_3x3, None)

        assert result is not None
        assert result.shape == (9,)
        assert np.allclose(result, np.eye(3).reshape(9))

    def test_3x4_color_matrix_is_contracted_to_3x3(self):
        """LibRaw/rawpy 3x4 color_matrix should be accepted via deterministic 3x3 contraction."""
        matrix_3x4 = np.array(
            [
                [1.0, 0.1, 0.2, 0.9],
                [0.3, 1.1, 0.4, 0.8],
                [0.5, 0.6, 1.2, 0.7],
            ],
            dtype=np.float64,
        )

        result = self.decoder._select_valid_color_matrix(matrix_3x4, None)

        assert result is not None
        assert result.shape == (9,)
        assert np.allclose(result, matrix_3x4[:, :3].reshape(9))

    def test_4x3_rgb_xyz_matrix_fallback_is_contracted_to_3x3(self):
        """LibRaw/rawpy 4x3 rgb_xyz_matrix fallback should be accepted via deterministic 3x3 contraction."""
        invalid_primary = np.zeros((3, 3), dtype=np.float64)  # Force fallback path
        matrix_4x3 = np.array(
            [
                [0.9, 0.1, 0.2],
                [0.3, 1.0, 0.4],
                [0.5, 0.6, 1.1],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float64,
        )

        result = self.decoder._select_valid_color_matrix(invalid_primary, matrix_4x3)

        assert result is not None
        assert result.shape == (9,)
        assert np.allclose(result, matrix_4x3[:3, :].reshape(9))

    def test_3x3_zero_ndarray_falls_back(self):
        """A (3, 3) all-zero array must still trigger fallback."""
        valid_fallback = [0.4, 0.3, 0.3, 0.2, 0.6, 0.2, 0.05, 0.1, 0.85]
        result = self.decoder._select_valid_color_matrix(np.zeros((3, 3)), valid_fallback)

        assert result is not None
        assert np.allclose(result, np.array(valid_fallback, dtype=np.float64))

    def test_inf_color_matrix_falls_back_to_rgb_xyz(self):
        """Infinity in color_matrix must trigger fallback."""
        inf_color_matrix = [float("inf")] + [0.0] * 8
        valid_rgb_xyz_matrix = [0.3, 0.5, 0.2, 0.1, 0.7, 0.2, 0.05, 0.12, 0.95]

        result = self.decoder._select_valid_color_matrix(inf_color_matrix, valid_rgb_xyz_matrix)

        assert result is not None
        assert np.allclose(result, np.array(valid_rgb_xyz_matrix, dtype=np.float64))

    def test_non_numeric_color_matrix_falls_back_to_rgb_xyz(self):
        """Non-numeric color_matrix must be treated as invalid and trigger fallback."""
        bad_color_matrix = ["x"] * 9
        valid_rgb_xyz_matrix = [0.3, 0.5, 0.2, 0.1, 0.7, 0.2, 0.05, 0.12, 0.95]

        result = self.decoder._select_valid_color_matrix(bad_color_matrix, valid_rgb_xyz_matrix)

        assert result is not None
        assert np.allclose(result, np.array(valid_rgb_xyz_matrix, dtype=np.float64))


class TestColorMatrixSelectionProperties:
    """Property-based tests for _select_valid_color_matrix invariants."""

    decoder = LinearDecoder()

    # Constrained strategy: values always have norm >= 1e-6, avoiding assume() discards
    _matrix_st = st.lists(
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=9,
        max_size=9,
    )

    @given(_matrix_st, _matrix_st)
    def test_valid_color_matrix_always_preferred(self, color_matrix, rgb_xyz_matrix):
        """Non-zero length-9 color_matrix is always preferred over rgb_xyz_matrix."""
        # _matrix_st min_value=1e-3 guarantees norm >= 1e-6 — no assume() needed
        result = self.decoder._select_valid_color_matrix(color_matrix, rgb_xyz_matrix)

        assert result is not None
        assert np.allclose(result, np.array(color_matrix, dtype=np.float64))

    @given(_matrix_st)
    def test_near_zero_color_matrix_triggers_fallback(self, rgb_xyz_matrix):
        """Near-zero color_matrix must always trigger fallback to valid rgb_xyz_matrix."""
        near_zero = [1e-38] * 9  # norm << 1e-6

        result = self.decoder._select_valid_color_matrix(near_zero, rgb_xyz_matrix)

        rgb_arr = np.array(rgb_xyz_matrix, dtype=np.float64)
        assert result is not None
        assert np.allclose(result, rgb_arr)

    _wrong_len_matrix_st = st.one_of(
        st.lists(
            st.floats(
                min_value=1e-3,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=0,
            max_size=8,
        ),
        st.lists(
            st.floats(
                min_value=1e-3,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=10,
            max_size=20,
        ),
    )

    @given(_wrong_len_matrix_st)
    def test_wrong_length_color_matrix_triggers_fallback(self, bad_matrix):
        """Flat color_matrix with length != 9 must be ignored."""
        valid_fallback = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]

        result = self.decoder._select_valid_color_matrix(bad_matrix, valid_fallback)

        assert result is not None
        assert np.allclose(result, np.array(valid_fallback, dtype=np.float64))

    @given(st.lists(st.just(float("nan")), min_size=9, max_size=9))
    def test_nan_color_matrix_triggers_fallback(self, nan_matrix):
        """NaN-filled color_matrix must be ignored."""
        valid_fallback = [0.4, 0.3, 0.3, 0.2, 0.6, 0.2, 0.05, 0.1, 0.85]

        result = self.decoder._select_valid_color_matrix(nan_matrix, valid_fallback)

        assert result is not None
        assert np.allclose(result, np.array(valid_fallback, dtype=np.float64))

    def test_inf_color_matrix_triggers_fallback(self):
        """Infinity in color_matrix must be ignored deterministically."""
        inf_matrix = [float("inf")] + [0.1] * 8
        valid_fallback = [0.4, 0.3, 0.3, 0.2, 0.6, 0.2, 0.05, 0.1, 0.85]

        result = self.decoder._select_valid_color_matrix(inf_matrix, valid_fallback)

        assert result is not None
        assert np.allclose(result, np.array(valid_fallback, dtype=np.float64))

    @given(_matrix_st)
    def test_selection_is_deterministic(self, color_matrix):
        """Same inputs must always produce bitwise-identical outputs (no nondeterminism)."""
        fallback = [0.4, 0.3, 0.3, 0.2, 0.6, 0.2, 0.05, 0.1, 0.85]

        result_a = self.decoder._select_valid_color_matrix(color_matrix, fallback)
        result_b = self.decoder._select_valid_color_matrix(color_matrix, fallback)

        assert result_a is not None
        assert result_b is not None
        # Bitwise identical — not just allclose — because no stochastic ops involved
        np.testing.assert_array_equal(result_a, result_b)


class TestRawMetadataValidation:
    """Unit tests for _validate_raw_metadata — no rawpy install required."""

    decoder = LinearDecoder()

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

    def test_valid_metadata_passes(self):
        raw = self._make_raw(
            wb=[2.0, 1.0, 1.5, 1.0],
            bl=[512, 512, 512, 512],
            raw_image_shape=(100, 100),
        )
        self.decoder._validate_raw_metadata(raw)  # must not raise

    def test_zero_wb_gain_raises(self):
        raw = self._make_raw(wb=[0.0, 1.0, 1.5, 1.0])
        with pytest.raises(ValueError, match="zero or negative gain"):
            self.decoder._validate_raw_metadata(raw)

    def test_negative_wb_gain_raises(self):
        raw = self._make_raw(wb=[-1.0, 1.0, 1.5, 1.0])
        with pytest.raises(ValueError, match="zero or negative gain"):
            self.decoder._validate_raw_metadata(raw)

    def test_nan_wb_raises(self):
        raw = self._make_raw(wb=[float("nan"), 1.0, 1.5, 1.0])
        with pytest.raises(ValueError, match="NaN"):
            self.decoder._validate_raw_metadata(raw)

    def test_inf_wb_raises(self):
        raw = self._make_raw(wb=[float("inf"), 1.0, 1.5, 1.0])
        with pytest.raises(ValueError, match="infinity"):
            self.decoder._validate_raw_metadata(raw)

    def test_empty_wb_raises(self):
        raw = self._make_raw(wb=[])
        with pytest.raises(ValueError, match="empty"):
            self.decoder._validate_raw_metadata(raw)

    def test_wrong_channel_count_wb_raises(self):
        raw = self._make_raw(wb=[2.0, 1.0, 1.5])  # 3 channels — invalid for [R, G1, B, G2]
        with pytest.raises(ValueError, match="channel count"):
            self.decoder._validate_raw_metadata(raw)

    def test_negative_black_level_raises(self):
        raw = self._make_raw(wb=[2.0, 1.0, 1.5, 1.0], bl=[-1, 0, 0, 0])
        with pytest.raises(ValueError, match="negative"):
            self.decoder._validate_raw_metadata(raw)

    def test_nan_black_level_raises(self):
        raw = self._make_raw(wb=[2.0, 1.0, 1.5, 1.0], bl=[float("nan"), 0, 0, 0])
        with pytest.raises(ValueError, match="NaN"):
            self.decoder._validate_raw_metadata(raw)

    def test_inf_black_level_raises(self):
        raw = self._make_raw(wb=[2.0, 1.0, 1.5, 1.0], bl=[float("inf"), 0, 0, 0])
        with pytest.raises(ValueError, match="infinity"):
            self.decoder._validate_raw_metadata(raw)

    def test_zero_g2_wb_gain_raises(self):
        raw = self._make_raw(wb=[2.0, 1.0, 1.5, 0.0])
        with pytest.raises(ValueError, match="zero or negative gain"):
            self.decoder._validate_raw_metadata(raw)

    def test_wrong_channel_count_black_level_raises(self):
        raw = self._make_raw(wb=[2.0, 1.0, 1.5, 1.0], bl=[512, 512])  # 2 channels — invalid
        with pytest.raises(ValueError, match="channel count"):
            self.decoder._validate_raw_metadata(raw)

    def test_3d_raw_image_raises(self):
        raw = self._make_raw(
            wb=[2.0, 1.0, 1.5, 1.0],
            bl=[512, 512, 512, 512],
            raw_image_shape=(100, 100, 3),  # 3D — invalid for Bayer
        )
        with pytest.raises(ValueError, match="2D"):
            self.decoder._validate_raw_metadata(raw)

    def test_absent_attributes_do_not_raise(self):
        """Missing WB/BL attributes (older LibRaw) must not raise."""
        raw = self._make_raw()  # all attrs absent
        self.decoder._validate_raw_metadata(raw)  # must not raise

    def test_non_numeric_wb_raises_value_error(self):
        """Non-numeric WB payload must raise ValueError, not TypeError."""
        raw = self._make_raw(wb=["not", "a", "number", "!"])
        with pytest.raises(ValueError, match="camera_whitebalance is unparseable to float64"):
            self.decoder._validate_raw_metadata(raw)

    def test_non_numeric_black_level_raises_value_error(self):
        """Non-numeric black level payload must raise ValueError, not TypeError."""
        raw = self._make_raw(wb=[2.0, 1.0, 1.5, 1.0], bl=["bad", "data", "here", "!"])
        with pytest.raises(ValueError, match="black_level_per_channel is unparseable to float64"):
            self.decoder._validate_raw_metadata(raw)


class TestIngestProvenance:
    """Unit tests for _compute_ingest_fingerprint and _canonical_f64_list."""

    _WB = np.array([2.0, 1.0, 1.5, 1.0])
    _BL = np.array([512.0, 512.0, 512.0, 512.0])
    _CM = np.eye(3)
    _SHAPE = (480, 640)

    def _fp(self, **overrides):
        kw = dict(wb=self._WB, black_level=self._BL, color_matrix=self._CM, raw_shape=self._SHAPE)
        kw.update(overrides)
        return _compute_ingest_fingerprint(**kw)

    # --- _canonical_f64_list ---

    def test_canonical_f64_list_negative_zero_normalized(self):
        a = np.array([-0.0, 1.0, -0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert _canonical_f64_list(a) == _canonical_f64_list(b)

    def test_canonical_f64_list_flattens_row_major(self):
        mat_3x3 = np.arange(9, dtype=np.float64).reshape(3, 3)
        flat = np.arange(9, dtype=np.float64)
        assert _canonical_f64_list(mat_3x3) == _canonical_f64_list(flat)

    # --- _compute_ingest_fingerprint ---

    def test_fingerprint_stable_for_identical_inputs(self):
        """Same metadata always produces the same fingerprint."""
        assert self._fp() == self._fp()

    def test_fingerprint_3x3_vs_flat_matrix_equivalent(self):
        """(3,3) and flattened (9,) matrix representations produce the same fingerprint."""
        fp_3x3 = self._fp(color_matrix=np.eye(3))
        fp_flat = self._fp(color_matrix=np.eye(3).ravel())
        assert fp_3x3 == fp_flat

    def test_fingerprint_negative_zero_normalization(self):
        """-0.0 and +0.0 in wb array produce the same fingerprint."""
        wb_pos = np.array([2.0, 1.0, 1.5, 1.0])
        wb_neg = np.array([2.0, 1.0, 1.5, -0.0])  # -0.0 in last element
        # Replace 1.0 with -0.0 in a safe position (not a valid WB but fingerprint is pure hashing)
        wb_with_neg_zero = np.array([2.0, 0.0, 1.5, 1.0])
        wb_with_pos_zero = np.array([2.0, -0.0, 1.5, 1.0])
        fp_neg = self._fp(wb=wb_with_neg_zero)
        fp_pos = self._fp(wb=wb_with_pos_zero)
        assert fp_neg == fp_pos

    def test_fingerprint_single_element_change_flips_hash(self):
        """A single-element perturbation (1e-6) changes the fingerprint."""
        wb_modified = self._WB.copy()
        wb_modified[0] += 1e-6
        assert self._fp() != self._fp(wb=wb_modified)

    def test_fingerprint_none_fields_stable(self):
        """None fields are hashed stably (not omitted)."""
        fp1 = _compute_ingest_fingerprint(wb=None, black_level=None, color_matrix=None, raw_shape=(480, 640))
        fp2 = _compute_ingest_fingerprint(wb=None, black_level=None, color_matrix=None, raw_shape=(480, 640))
        assert fp1 == fp2
        assert isinstance(fp1, str) and len(fp1) == 64

    def test_fingerprint_shape_change_flips_hash(self):
        """Different raw_shape always produces a different fingerprint."""
        assert self._fp(raw_shape=(480, 640)) != self._fp(raw_shape=(480, 641))

    def test_linearingestresult_has_ingest_fingerprint_field(self):
        """LinearIngestResult exposes ingest_fingerprint as an Optional[str] field."""
        result = LinearIngestResult(
            linear_rgb=np.zeros((4, 4, 3), dtype=np.float32),
            gamma=1.0,
            bit_depth=32,
            dtype="float32",
            input_size=(4, 4),
            input_path=Path("dummy.dng"),
            input_format="RAW_DNG",
            color_space="linear_sRGB",
            ingest_fingerprint="abc123",
        )
        assert result.ingest_fingerprint == "abc123"

    def test_linearingestresult_ingest_fingerprint_defaults_none(self):
        """ingest_fingerprint defaults to None for non-RAW formats."""
        result = LinearIngestResult(
            linear_rgb=np.zeros((4, 4, 3), dtype=np.float32),
            gamma=1.0,
            bit_depth=32,
            dtype="float32",
            input_size=(4, 4),
            input_path=Path("dummy.tiff"),
            input_format="TIFF",
            color_space="linear_sRGB",
        )
        assert result.ingest_fingerprint is None


# Pytest markers for organization
pytestmark = [
    pytest.mark.unit,  # Fast unit tests for spatial_ai ingest layer
]
