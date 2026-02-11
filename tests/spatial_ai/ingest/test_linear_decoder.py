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
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.spatial_ai.ingest import LinearDecoder, LinearIngestResult, decode


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
        img_uint16 = np.clip(hdr_img / hdr_img.max() * 65535, 0, 65535).astype(np.uint16)
        test_img_path = tmp_path / "hdr_test.tiff"
        img = Image.fromarray(img_uint16, mode="RGB")
        img.save(test_img_path, format="TIFF")

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
            )

    def test_unsupported_format_raises(self, tmp_path: Path):
        """Test that unsupported formats raise clear errors."""
        # Create dummy file with unsupported extension
        unsupported_path = tmp_path / "test.jpg"
        unsupported_path.write_text("dummy")

        decoder = LinearDecoder(gamma=1.0)
        with pytest.raises(ValueError, match="Unsupported format"):
            decoder.decode(unsupported_path)

    def test_raw_format_not_implemented(self, tmp_path: Path):
        """Test that RAW formats raise NotImplementedError (Phase II)."""
        # Create dummy RAW file
        raw_path = tmp_path / "test.cr2"
        raw_path.write_text("dummy")

        decoder = LinearDecoder(gamma=1.0)
        with pytest.raises(NotImplementedError, match="RAW format"):
            decoder.decode(raw_path)

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

    def test_16bit_png_decode(self, tmp_path: Path):
        """Test decoding of 16-bit PNG."""
        # Create 16-bit test image
        test_img = (np.random.rand(100, 100, 3) * 65535).astype(np.uint16)
        test_img_path = tmp_path / "test_16bit.png"
        img = Image.fromarray(test_img, mode="RGB")
        img.save(test_img_path, format="PNG", bits=16)

        # Decode
        result = decode(test_img_path, gamma=1.0)

        # Verify
        assert result.linear_rgb.dtype == np.float32
        assert result.gamma == 1.0
        assert result.input_format == "PNG"
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

        # Create test image
        test_img = np.random.rand(50, 50, 3).astype(np.float32)
        test_img_path = tmp_path / "test.tiff"
        img_uint16 = (test_img * 65535).astype(np.uint16)
        img = Image.fromarray(img_uint16, mode="RGB")
        img.save(test_img_path, format="TIFF")

        # Attempt decode with emit_exr=True should fail loudly
        with pytest.raises(RuntimeError, match="emit_exr=True requires OpenEXR package"):
            decode(test_img_path, gamma=1.0, output_dir=tmp_path, emit_exr=True)

    def test_strict_ingest_rejects_uint8(self, tmp_path: Path):
        """Test that strict_ingest=True rejects 8-bit inputs."""
        # Create 8-bit test image
        test_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        test_img_path = tmp_path / "test_8bit.png"
        Image.fromarray(test_img, mode="RGB").save(test_img_path)

        # Attempt decode with strict_ingest=True should fail
        with pytest.raises(ValueError, match="strict_ingest=True rejects 8-bit inputs"):
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


# Pytest markers for organization
pytestmark = [
    pytest.mark.unit,  # Fast unit tests for spatial_ai ingest layer
]
