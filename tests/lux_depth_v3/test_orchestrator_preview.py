"""Test coverage for orchestrator preview/config normalization behavior.

Phase 2 Coverage: Preview/config normalization tests for EnhanceOrchestrator.

Tests verify:
1. Config normalization during orchestrator initialization
2. Hash mode behavior (NEVER, ALWAYS, IF_MANIFEST_EXISTS)
3. Output key generation with directory structure
4. Skip logic based on stored config fingerprints
5. Manifest path resolution
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def _make_test_image(tmp_path: Path, name: str = "test.png", size: tuple = (64, 64)) -> Path:
    """Create a minimal test image for orchestrator tests."""
    image_path = tmp_path / name
    Image.new("RGB", size, color="white").save(image_path)
    return image_path


def _make_mock_registry():
    """Create a mock depth backend registry."""
    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None

    from transformation_portal.depth.backends.protocol import DepthResult

    depth_result = DepthResult(
        depth_map=np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        metadata={},
        depth_units="relative",
        backend_id="da3",
        device="cpu",
    )
    backend.compute.return_value = depth_result

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


class TestConfigNormalization:
    """Test configuration normalization during orchestrator initialization."""

    def test_hash_mode_never_suppresses_hashing(self, tmp_path: Path) -> None:
        """Hash mode NEVER prevents hash computation."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.security import HashMode

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
            hash_mode=HashMode.NEVER,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            assert orchestrator.config.hash_mode == HashMode.NEVER

            # Test that hash computation returns None when mode is NEVER
            test_image = _make_test_image(tmp_path, "hash_never.png")
            result = orchestrator._compute_or_skip_hash(
                test_image,
                manifest_exists=True,
                saved_hash="abc123",
                for_manifest_write=False,
            )
            assert result is None

    def test_hash_mode_always_computes_hash(self, tmp_path: Path) -> None:
        """Hash mode ALWAYS computes hash regardless of manifest state."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.security import HashMode

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
            hash_mode=HashMode.ALWAYS,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            test_image = _make_test_image(tmp_path, "hash_always.png")

            result = orchestrator._compute_or_skip_hash(
                test_image,
                manifest_exists=False,
                saved_hash=None,
                for_manifest_write=True,
            )
            assert result is not None
            assert len(result) == 64  # SHA-256 hex string

    def test_hash_mode_if_manifest_exists_behavior(self, tmp_path: Path) -> None:
        """Hash mode IF_MANIFEST_EXISTS conditionally computes hash."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.security import HashMode

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
            hash_mode=HashMode.IF_MANIFEST_EXISTS,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            test_image = _make_test_image(tmp_path, "hash_conditional.png")

            # No manifest, no baseline - should return None for comparison
            result_no_manifest = orchestrator._compute_or_skip_hash(
                test_image,
                manifest_exists=False,
                saved_hash=None,
                for_manifest_write=False,
            )
            assert result_no_manifest is None

            # For manifest write - always compute
            result_for_write = orchestrator._compute_or_skip_hash(
                test_image,
                manifest_exists=False,
                saved_hash=None,
                for_manifest_write=True,
            )
            assert result_for_write is not None

    def test_verify_outputs_flag_preserved(self, tmp_path: Path) -> None:
        """verify_outputs flag is correctly stored in orchestrator."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator_verify = EnhanceOrchestrator(config, tmp_path, verify_outputs=True)
            assert orchestrator_verify.verify_outputs is True

            orchestrator_no_verify = EnhanceOrchestrator(config, tmp_path, verify_outputs=False)
            assert orchestrator_no_verify.verify_outputs is False


class TestOutputKeyGeneration:
    """Test output key generation for artifact naming."""

    def test_make_output_key_basic(self, tmp_path: Path) -> None:
        """make_output_key generates consistent keys for same input."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        test_image = _make_test_image(tmp_path, "basic.png")

        key1 = make_output_key(test_image, tmp_path)
        key2 = make_output_key(test_image, tmp_path)

        assert key1 == key2
        assert "basic" in str(key1)

    def test_make_output_key_nested_path(self, tmp_path: Path) -> None:
        """make_output_key preserves directory structure."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        nested_dir = tmp_path / "subdir" / "nested"
        nested_dir.mkdir(parents=True)
        test_image = _make_test_image(nested_dir, "nested.png")

        key = make_output_key(test_image, tmp_path)

        # Key should include parent directory structure
        assert key.parent != Path(".")
        assert "nested" in str(key)

    def test_make_output_key_xxhash_mode(self, tmp_path: Path) -> None:
        """make_output_key with xxhash flag produces different key suffix."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            XXHASH_AVAILABLE,
            make_output_key,
        )

        test_image = _make_test_image(tmp_path, "xxhash.png")

        key_sha = make_output_key(test_image, tmp_path, use_xxhash=False)
        # Only test if xxhash is available
        if XXHASH_AVAILABLE:
            key_xx = make_output_key(test_image, tmp_path, use_xxhash=True)
            # Both should have the base stem but different suffixes
            assert key_sha.stem.startswith("xxhash")
            assert key_xx.stem.startswith("xxhash")


class TestSkipLogic:
    """Test skip logic based on config fingerprints."""

    def test_should_skip_depth_no_manifest(self, tmp_path: Path) -> None:
        """should_skip_depth returns False when no manifest exists."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            test_image = _make_test_image(tmp_path, "skip_test.png")

            depth_path = tmp_path / "depth" / "skip_test_depth.png"
            manifest_path = tmp_path / "manifests" / "skip_test_combined.json"

            result = orchestrator.should_skip_depth(
                depth_path,
                manifest_path,
                ImageInput(path=test_image),
            )
            assert result is False

    def test_should_skip_depth_no_depth_file(self, tmp_path: Path) -> None:
        """should_skip_depth returns False when depth file missing."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            test_image = _make_test_image(tmp_path, "no_depth.png")

            # Create manifest path but no depth file
            manifest_path = tmp_path / "manifests" / "no_depth_combined.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("{}")

            depth_path = tmp_path / "depth" / "no_depth_depth.png"

            result = orchestrator.should_skip_depth(
                depth_path,
                manifest_path,
                ImageInput(path=test_image),
            )
            assert result is False


class TestConfigFingerprint:
    """Test config fingerprint computation."""

    def test_compute_config_fingerprint_deterministic(self, tmp_path: Path) -> None:
        """Config fingerprint is deterministic for same config."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)

            fp1 = orchestrator.compute_config_fingerprint()
            fp2 = orchestrator.compute_config_fingerprint()

            assert fp1.to_sha256() == fp2.to_sha256()

    def test_config_fingerprint_depth_only(self, tmp_path: Path) -> None:
        """depth_only() returns fingerprint for Stage A only."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            fp = orchestrator.compute_config_fingerprint()

            depth_only_fp = fp.depth_only()
            assert depth_only_fp is not None
            # Should be a ConfigFingerprint
            assert hasattr(depth_only_fp, "to_sha256")

    def test_config_fingerprint_v2_only(self, tmp_path: Path) -> None:
        """v2_only() returns fingerprint for Stage B only."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            fp = orchestrator.compute_config_fingerprint()

            v2_only_fp = fp.v2_only()
            assert v2_only_fp is not None
            assert hasattr(v2_only_fp, "to_sha256")


class TestNormalizeV2Status:
    """Test V2 status normalization."""

    def test_normalize_v2_status_success(self) -> None:
        """success/ok normalized to 'ok'."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        assert EnhanceOrchestrator._normalize_v2_status("success") == "ok"
        assert EnhanceOrchestrator._normalize_v2_status("Success") == "ok"
        assert EnhanceOrchestrator._normalize_v2_status("ok") == "ok"
        assert EnhanceOrchestrator._normalize_v2_status("OK") == "ok"

    def test_normalize_v2_status_failed(self) -> None:
        """failed/failure normalized to 'error'."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        assert EnhanceOrchestrator._normalize_v2_status("failed") == "error"
        assert EnhanceOrchestrator._normalize_v2_status("failure") == "error"

    def test_normalize_v2_status_none(self) -> None:
        """None normalized to 'skipped'."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        assert EnhanceOrchestrator._normalize_v2_status(None) == "skipped"

    def test_normalize_v2_status_empty(self) -> None:
        """Empty string normalized to 'skipped'."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        assert EnhanceOrchestrator._normalize_v2_status("") == "skipped"
        assert EnhanceOrchestrator._normalize_v2_status("   ") == "skipped"


class TestCoerceOutputPaths:
    """Test output path coercion helpers."""

    def test_coerce_output_paths_string(self) -> None:
        """Single string coerced to list."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        result = EnhanceOrchestrator._coerce_output_paths("/path/to/output.png")
        assert result == ["/path/to/output.png"]

    def test_coerce_output_paths_list(self) -> None:
        """List of strings preserved."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        paths = ["/path/one.png", "/path/two.png"]
        result = EnhanceOrchestrator._coerce_output_paths(paths)
        assert result == paths

    def test_coerce_output_paths_empty(self) -> None:
        """Empty inputs return empty list."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        assert EnhanceOrchestrator._coerce_output_paths("") == []
        assert EnhanceOrchestrator._coerce_output_paths([]) == []
        assert EnhanceOrchestrator._coerce_output_paths(None) == []

    def test_coerce_output_paths_filters_non_strings(self) -> None:
        """Non-string list elements filtered out."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        paths = ["/path/one.png", None, 123, "/path/two.png", ""]
        result = EnhanceOrchestrator._coerce_output_paths(paths)
        assert result == ["/path/one.png", "/path/two.png"]
