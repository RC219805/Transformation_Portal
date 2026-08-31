"""Test coverage for orchestrator manifest creation.

Phase 2 Coverage: Manifest creation tests for EnhanceOrchestrator.

Tests verify:
1. Manifest file structure and schema
2. Config fingerprint embedding
3. Backend selection metadata in manifests
4. Timing metadata accuracy
5. Input/output path recording
6. Manifest loading and validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
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


def _make_mock_depth_result():
    """Create a deterministic synthetic depth result."""
    from transformation_portal.depth.backends.protocol import DepthResult

    return DepthResult(
        depth_map=np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        metadata={},
        depth_units="relative",
        backend_id="da3",
        device="cpu",
    )


def _make_mock_registry():
    """Create a mock depth backend registry."""
    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None
    backend.compute.return_value = _make_mock_depth_result()

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


def _create_orchestrator(tmp_path: Path, **config_kwargs):
    """Create an orchestrator instance with mocked backend registry."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    defaults = {
        "depth_backend": "da3",
        "depth_device": "cpu",
        "enable_v2": False,
        "enable_materials_v3": False,
    }
    defaults.update(config_kwargs)
    config = EnhanceConfig(**defaults)

    with patch(
        "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
        return_value=_make_mock_registry(),
    ):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        return orchestrator


def _get_manifest_data(tmp_path: Path, image_name: str) -> Dict[str, Any]:
    """Process an image and return the manifest data."""
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    orchestrator = _create_orchestrator(tmp_path)
    test_image = _make_test_image(tmp_path, image_name)

    result = orchestrator.enhance_image(
        ImageInput(path=test_image),
        input_root=tmp_path,
    )

    manifest_path = Path(result["manifest"])
    with open(manifest_path) as f:
        return json.load(f)


class TestManifestStructure:
    """Test manifest file structure and schema."""

    def test_manifest_contains_input_section(self, tmp_path: Path) -> None:
        """Manifest contains input metadata section."""
        manifest = _get_manifest_data(tmp_path, "input_section.png")
        assert "input" in manifest

    def test_manifest_contains_depth_section(self, tmp_path: Path) -> None:
        """Manifest contains depth metadata section."""
        manifest = _get_manifest_data(tmp_path, "depth_section.png")
        assert "depth" in manifest

    def test_manifest_contains_timing_section(self, tmp_path: Path) -> None:
        """Manifest contains timing metadata section."""
        manifest = _get_manifest_data(tmp_path, "timing_section.png")
        assert "timing" in manifest

    def test_manifest_contains_config_fingerprint(self, tmp_path: Path) -> None:
        """Manifest contains config fingerprint section."""
        manifest = _get_manifest_data(tmp_path, "fingerprint_section.png")
        assert "config_fingerprint" in manifest

    def test_manifest_contains_backend_selection(self, tmp_path: Path) -> None:
        """Manifest contains backend selection metadata."""
        manifest = _get_manifest_data(tmp_path, "backend_section.png")
        assert "backend_selection" in manifest

    def test_manifest_is_valid_json(self, tmp_path: Path) -> None:
        """Manifest is valid JSON."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "valid_json.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        manifest_path = Path(result["manifest"])
        with open(manifest_path) as f:
            # Should not raise
            data = json.load(f)
        assert isinstance(data, dict)


class TestDeprecatedOutputFlags:
    @pytest.mark.parametrize("emit_report", [True, False])
    def test_emit_report_value_never_suppresses_combined_manifest(
        self,
        tmp_path: Path,
        emit_report: bool,
    ) -> None:
        from transformation_portal.lux_depth_v3.config import DeprecatedOutputFlagWarning
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        with pytest.warns(DeprecatedOutputFlagWarning, match="combined report is always produced"):
            orchestrator = _create_orchestrator(tmp_path, emit_report=emit_report)
        test_image = _make_test_image(tmp_path, f"report_{emit_report}.png")

        result = orchestrator.enhance_image(ImageInput(path=test_image), input_root=tmp_path)

        manifest_path = Path(result["manifest"])
        assert manifest_path.is_file()
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["input"] is not None

    def test_emit_marketing_warns_without_creating_fictional_artifact(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.config import DeprecatedOutputFlagWarning
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        with pytest.warns(DeprecatedOutputFlagWarning, match="no marketing artifact is produced"):
            orchestrator = _create_orchestrator(tmp_path, emit_marketing=True)
        test_image = _make_test_image(tmp_path, "source.png")

        result = orchestrator.enhance_image(ImageInput(path=test_image), input_root=tmp_path)

        assert Path(result["manifest"]).is_file()
        assert not [path for path in tmp_path.rglob("*") if "marketing" in path.name.lower()]


class TestInputMetadata:
    """Test input metadata in manifests."""

    def test_input_contains_image_path(self, tmp_path: Path) -> None:
        """Input section contains image path."""
        manifest = _get_manifest_data(tmp_path, "input_path.png")
        assert "image_path" in manifest["input"]
        assert "input_path" in manifest["input"]["image_path"]

    def test_input_sha256_present_when_hash_mode_always(self, tmp_path: Path) -> None:
        """Input SHA256 is present when hash mode is ALWAYS."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
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
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "sha256_check.png")
            result = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

            with open(result["manifest"]) as f:
                manifest = json.load(f)

            assert manifest["input"]["image_sha256"] is not None
            assert len(manifest["input"]["image_sha256"]) == 64

    def test_input_sha256_null_when_hash_mode_never(self, tmp_path: Path) -> None:
        """Input SHA256 is null when hash mode is NEVER."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
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
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "no_sha256.png")
            result = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

            with open(result["manifest"]) as f:
                manifest = json.load(f)

            assert manifest["input"]["image_sha256"] is None


class TestDepthMetadata:
    """Test depth metadata in manifests."""

    def test_depth_contains_model(self, tmp_path: Path) -> None:
        """Depth section contains model identifier."""
        manifest = _get_manifest_data(tmp_path, "depth_model.png")
        assert "model" in manifest["depth"]

    def test_depth_contains_runtime_seconds(self, tmp_path: Path) -> None:
        """Depth section contains runtime in seconds."""
        manifest = _get_manifest_data(tmp_path, "depth_runtime.png")
        assert "runtime_seconds" in manifest["depth"]
        assert isinstance(manifest["depth"]["runtime_seconds"], (int, float))

    def test_depth_contains_path(self, tmp_path: Path) -> None:
        """Depth section contains depth file path."""
        manifest = _get_manifest_data(tmp_path, "depth_path.png")
        assert "depth_path" in manifest["depth"]

    def test_depth_contains_stats(self, tmp_path: Path) -> None:
        """Depth section contains processing stats."""
        manifest = _get_manifest_data(tmp_path, "depth_stats.png")
        assert "stats" in manifest["depth"]


class TestConfigFingerprint:
    """Test config fingerprint in manifests."""

    def test_fingerprint_contains_depth_backend(self, tmp_path: Path) -> None:
        """Config fingerprint contains depth backend."""
        manifest = _get_manifest_data(tmp_path, "fp_backend.png")
        assert "depth_backend" in manifest["config_fingerprint"]

    def test_fingerprint_contains_model_variant(self, tmp_path: Path) -> None:
        """Config fingerprint contains model variant."""
        manifest = _get_manifest_data(tmp_path, "fp_variant.png")
        assert "model_variant" in manifest["config_fingerprint"]

    def test_fingerprint_deterministic_across_runs(self, tmp_path: Path) -> None:
        """Config fingerprint is deterministic for same config."""
        manifest1 = _get_manifest_data(tmp_path, "fp_det1.png")
        manifest2 = _get_manifest_data(tmp_path, "fp_det2.png")

        # Same config should produce same fingerprint fields
        assert manifest1["config_fingerprint"]["depth_backend"] == manifest2["config_fingerprint"]["depth_backend"]
        assert manifest1["config_fingerprint"]["model_variant"] == manifest2["config_fingerprint"]["model_variant"]

    def test_depth_pro_manifest_config_fingerprint_uses_depth_pro_model_id(self, tmp_path: Path) -> None:
        """Per-image manifests serialize the resolved Depth Pro model identity."""
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        orchestrator = _create_orchestrator(tmp_path, depth_backend="depth_pro")
        fingerprint = orchestrator.compute_config_fingerprint()
        manifest_path = tmp_path / "depth_pro_manifest.json"

        CombinedManifest(config_fingerprint=fingerprint).save(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["config_fingerprint"]["depth_backend"] == "depth_pro"
        assert manifest["config_fingerprint"]["model_variant"] == "apple/ml-depth-pro"


class TestBackendSelectionMetadata:
    """Test backend selection metadata in manifests."""

    def test_backend_selection_contains_requested(self, tmp_path: Path) -> None:
        """Backend selection contains requested backend."""
        manifest = _get_manifest_data(tmp_path, "bs_requested.png")
        assert "requested_backend" in manifest["backend_selection"]

    def test_backend_selection_contains_resolved(self, tmp_path: Path) -> None:
        """Backend selection contains resolved backend."""
        manifest = _get_manifest_data(tmp_path, "bs_resolved.png")
        assert "resolved_backend" in manifest["backend_selection"]
        assert manifest["backend_selection"]["resolved_backend"] == "da3"

    def test_backend_selection_contains_status(self, tmp_path: Path) -> None:
        """Backend selection contains resolution status."""
        manifest = _get_manifest_data(tmp_path, "bs_status.png")
        assert "resolution_status" in manifest["backend_selection"]

    def test_backend_selection_contains_device(self, tmp_path: Path) -> None:
        """Backend selection contains device info."""
        manifest = _get_manifest_data(tmp_path, "bs_device.png")
        assert "device" in manifest["backend_selection"]

    def test_backend_selection_contains_attempts(self, tmp_path: Path) -> None:
        """Backend selection contains attempt history."""
        manifest = _get_manifest_data(tmp_path, "bs_attempts.png")
        assert "attempts" in manifest["backend_selection"]
        assert isinstance(manifest["backend_selection"]["attempts"], list)


class TestTimingMetadata:
    """Test timing metadata in manifests."""

    def test_timing_contains_depth_seconds(self, tmp_path: Path) -> None:
        """Timing section contains depth processing time."""
        manifest = _get_manifest_data(tmp_path, "timing_depth.png")
        assert "depth_seconds" in manifest["timing"]
        assert isinstance(manifest["timing"]["depth_seconds"], (int, float))

    def test_timing_contains_total_seconds(self, tmp_path: Path) -> None:
        """Timing section contains total processing time."""
        manifest = _get_manifest_data(tmp_path, "timing_total.png")
        assert "total_seconds" in manifest["timing"]
        assert isinstance(manifest["timing"]["total_seconds"], (int, float))

    def test_timing_contains_timestamp_utc(self, tmp_path: Path) -> None:
        """Timing section contains UTC timestamp."""
        manifest = _get_manifest_data(tmp_path, "timing_utc.png")
        assert "timestamp_utc" in manifest["timing"]
        # Should be ISO format
        assert "T" in manifest["timing"]["timestamp_utc"]

    def test_timing_total_greater_than_depth(self, tmp_path: Path) -> None:
        """Total time is at least as large as depth time."""
        manifest = _get_manifest_data(tmp_path, "timing_compare.png")
        assert manifest["timing"]["total_seconds"] >= manifest["timing"]["depth_seconds"]


class TestManifestLoading:
    """Test manifest loading and validation."""

    def test_combined_manifest_load_roundtrip(self, tmp_path: Path) -> None:
        """CombinedManifest can load a written manifest."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "load_test.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        manifest_path = Path(result["manifest"])
        loaded = CombinedManifest.load(manifest_path)

        assert loaded is not None
        assert loaded.input is not None
        assert loaded.depth is not None

    def test_manifest_load_preserves_config_fingerprint(self, tmp_path: Path) -> None:
        """Loading manifest preserves config fingerprint."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "fp_preserve.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        manifest_path = Path(result["manifest"])
        loaded = CombinedManifest.load(manifest_path)

        assert loaded.config_fingerprint is not None
        assert loaded.config_fingerprint.depth_backend is not None

    def test_manifest_load_preserves_backend_selection(self, tmp_path: Path) -> None:
        """Loading manifest preserves backend selection."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "bs_preserve.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        manifest_path = Path(result["manifest"])
        loaded = CombinedManifest.load(manifest_path)

        assert loaded.backend_selection is not None
        assert loaded.backend_selection.resolved_backend == "da3"

    def test_materials_v3_segmentation_metadata_passthrough_roundtrips(self, tmp_path: Path) -> None:
        """The APEX Materials V3 soft-passthrough warning lands inside
        ``materials_v3.segmentation_metadata`` as ``pixel_ops_passthrough`` and
        ``warnings``. Both must survive a manifest write/read roundtrip so the
        run-card cache and downstream consumers see the non-fatal signal."""
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest, MaterialsV3Metadata

        passthrough_payload = {
            "code": "APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE",
            "message": "Materials V3 masks present but every implemented op was below confidence threshold.",
            "details": {
                "material_count": 4,
                "implemented_materials": ["glass", "water", "foliage", "stone"],
                "applied_ops_count": 0,
                "blocked_reasons": {"below_confidence_threshold": 4},
            },
        }
        materials_v3 = MaterialsV3Metadata(
            enabled=True,
            segmentation_metadata={
                "warnings": ["APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE"],
                "errors": [],
                "pixel_ops_passthrough": passthrough_payload,
                "mask_artifact_path": None,
            },
        )

        manifest = CombinedManifest()
        manifest.materials_v3 = materials_v3

        manifest_path = tmp_path / "passthrough_manifest.json"
        manifest.save(manifest_path)
        loaded = CombinedManifest.load(manifest_path)

        assert loaded.materials_v3 is not None
        seg_meta = loaded.materials_v3.segmentation_metadata
        assert seg_meta is not None
        assert seg_meta["warnings"] == ["APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE"]
        assert seg_meta["pixel_ops_passthrough"] == passthrough_payload


class TestManifestEnvironment:
    """Test environment metadata in manifests."""

    def test_manifest_contains_environment(self, tmp_path: Path) -> None:
        """Manifest contains environment section."""
        manifest = _get_manifest_data(tmp_path, "env_check.png")
        # Environment may be in repro or top-level
        has_env = "environment" in manifest or (
            "repro" in manifest and manifest["repro"] is not None and "environment" in manifest["repro"]
        )
        assert has_env

    def test_manifest_contains_start_time(self, tmp_path: Path) -> None:
        """Manifest contains start time."""
        manifest = _get_manifest_data(tmp_path, "start_time.png")
        assert "start_time" in manifest
        assert "T" in manifest["start_time"]  # ISO format

    def test_manifest_contains_end_time(self, tmp_path: Path) -> None:
        """Manifest contains end time."""
        manifest = _get_manifest_data(tmp_path, "end_time.png")
        assert "end_time" in manifest
        assert "T" in manifest["end_time"]  # ISO format

    def test_end_time_after_start_time(self, tmp_path: Path) -> None:
        """End time is after or equal to start time."""
        manifest = _get_manifest_data(tmp_path, "time_order.png")
        # Lexicographic comparison works for ISO timestamps
        assert manifest["end_time"] >= manifest["start_time"]
