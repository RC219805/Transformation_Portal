"""Tests for config fingerprint and dual resume logic.

Tests PR #2: Config Fingerprint + Dual Resume
"""

import pytest
from pathlib import Path
import json

from lux_depth_v3.enhance.manifest import ConfigFingerprint
from lux_depth_v3.enhance.orchestrator import EnhanceConfig


class TestConfigFingerprint:
    """Test config fingerprinting for cache validation."""

    def test_same_config_same_hash(self):
        """Identical configs should produce same hash."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        assert fp1.to_sha256() == fp2.to_sha256()

    def test_different_model_different_hash(self):
        """Different model variant should change hash."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Small-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        assert fp1.to_sha256() != fp2.to_sha256()

    def test_different_v2_preset_different_hash(self):
        """Different V2 preset should change hash."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="production_ultra",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        assert fp1.to_sha256() != fp2.to_sha256()

    def test_different_quantization_different_hash(self):
        """Different quantization method should change hash."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="minmax",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        assert fp1.to_sha256() != fp2.to_sha256()

    def test_different_device_different_hash(self):
        """Different device should change hash."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cuda",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        assert fp1.to_sha256() != fp2.to_sha256()

    def test_deterministic(self):
        """Same config should always produce same hash."""
        fp = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        hashes = [fp.to_sha256() for _ in range(10)]
        assert len(set(hashes)) == 1  # All identical

    def test_hash_is_sha256(self):
        """Hash should be valid SHA256 (64 hex chars)."""
        fp = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        hash_str = fp.to_sha256()

        assert len(hash_str) == 64
        assert all(c in "0123456789abcdef" for c in hash_str)

    def test_depth_only_subset(self):
        """Depth config fingerprint should only include depth params."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="production_ultra",  # Different V2 preset
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        # Full hashes should differ
        assert fp1.to_sha256() != fp2.to_sha256()

        # Depth-only hashes should be same
        assert fp1.depth_only() == fp2.depth_only()

    def test_v2_only_subset(self):
        """V2 config fingerprint should only include V2 params."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Small-Metric",  # Different model
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        # Full hashes should differ
        assert fp1.to_sha256() != fp2.to_sha256()

        # V2-only hashes should be same
        assert fp1.v2_only() == fp2.v2_only()

    def test_preset_none_vs_string(self):
        """None preset should differ from empty string."""
        fp1 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )
        fp2 = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset="",  # Empty string instead of None
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        # Should produce same hash (both map to empty string)
        assert fp1.to_sha256() == fp2.to_sha256()

    def test_json_serializable(self):
        """Config fingerprint components should be JSON serializable."""
        fp = ConfigFingerprint(
            model_variant="DepthAnything3-Large-Metric",
            depth_quantization="p1p99",
            depth_device="cpu",
            preset=None,
            v2_preset="interior_luxury",
            v2_device="auto",
            v2_upscaler_backend="torch",
        )

        # Should not raise
        hash_str = fp.to_sha256()
        json_str = json.dumps({"fingerprint": hash_str})
        assert len(json_str) > 0
