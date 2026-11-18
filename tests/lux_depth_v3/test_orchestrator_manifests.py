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


class TestDottedInputNamesKeepSeparateArtifacts:
    """Inputs with interior dots must not share PBR, mask or Materials hand-off artifacts.

    An output key is ``<stem>_<ext>_<hash8>`` with no extension. Naming an
    artifact from ``Path(key).stem`` cut the key at its last interior dot and
    dropped the discriminating hash, so ``2024.05.01_front.jpg`` and
    ``2024.05.01_back.jpg`` both wrote ``2024.05_*``.
    """

    FRONT = "2024.05.01_front.jpg"
    BACK = "2024.05.01_back.jpg"

    def test_legacy_batch_writes_one_pbr_set_per_dotted_input(self, tmp_path: Path) -> None:
        """Both images report ok, so each must own maps of its own dimensions."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        front = _make_test_image(inputs, self.FRONT, size=(64, 48))
        back = _make_test_image(inputs, self.BACK, size=(80, 40))
        orchestrator = _create_orchestrator(
            output,
            generate_pbr=True,
            emit_run_card=False,
            enable_parallel_processing=False,
        )

        results = orchestrator.enhance_batch(inputs, input_files=[back, front])

        assert sorted((Path(r["image"]).name, r["status"]) for r in results) == [(self.BACK, "ok"), (self.FRONT, "ok")]
        normal_paths = set()
        for result in results:
            key = make_output_key(Path(result["image"]), inputs)
            pbr_assets = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))["pbr_assets"]
            for map_name in ("normal", "roughness", "ao"):
                assert Path(pbr_assets[f"{map_name}_path"]) == output / "pbr" / f"{key.name}_{map_name}.png"
            with Image.open(result["image"]) as source, Image.open(pbr_assets["normal_path"]) as normal:
                assert normal.size == source.size
            normal_paths.add(pbr_assets["normal_path"])
        assert len(normal_paths) == 2
        assert len(list((output / "pbr").iterdir())) == 6

    def test_prepared_batch_writes_one_pbr_set_per_dotted_input(self, tmp_path: Path) -> None:
        """The production path delivered one input the other input's maps and still passed evidence."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        sizes = {"2024.05.01_front.png": (64, 48), "2024.05.01_back.png": (80, 40)}
        images = []
        for name, size in sizes.items():
            pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            pixels[..., 0] = np.linspace(0, 255, size[0]).astype(np.uint8)[None, :]
            Image.fromarray(pixels).save(inputs / name)
            images.append(inputs / name)
        prepared = prepare_lux_execution(
            EnhanceConfig(depth_backend="synthetic", allow_synthetic_fallback=True, enable_v2=False, generate_pbr=True),
            inputs,
            sorted(images),
        )
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, output)

        results = orchestrator.enhance_batch(inputs)

        assert sorted((Path(r["image"]).name, r["status"]) for r in results) == sorted((name, "ok") for name in sizes)
        normal_names = set()
        for result in results:
            key = make_output_key(Path(result["image"]), inputs)
            pbr_assets = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))["pbr_assets"]
            normal_path = Path(pbr_assets["normal_path"])
            assert normal_path.name == f"{key.name}_normal.png"
            with Image.open(normal_path) as normal:
                assert normal.size == sizes[Path(result["image"]).name]
            normal_names.add(normal_path.name)
        assert len(normal_names) == 2

    def test_materials_handoff_and_mask_artifacts_keep_dotted_keys_distinct(self, tmp_path: Path) -> None:
        """The Materials V3 temp hand-off and both mask artifacts are named from the full key."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        inputs = tmp_path / "inputs"
        orchestrator = _create_orchestrator(tmp_path / "output")
        front_key = make_output_key(inputs / self.FRONT, inputs)
        back_key = make_output_key(inputs / self.BACK, inputs)
        assert front_key.stem == back_key.stem, "fixture must exercise the truncating-stem collision"

        front_handoff = orchestrator._expected_materials_v3_enhanced_path(front_key)
        back_handoff = orchestrator._expected_materials_v3_enhanced_path(back_key)
        assert front_handoff.name == f"{front_key.name}_materials_v3_enhanced.png"
        assert back_handoff.name == f"{back_key.name}_materials_v3_enhanced.png"

        front_artifact = orchestrator._segmentation_mask_artifact_path(front_key)
        assert front_artifact.name == f"{front_key.name}_materials_v3_masks.npz"
        assert front_artifact != orchestrator._segmentation_mask_artifact_path(back_key)

        mask_dir = tmp_path / "masks"
        front_masks = orchestrator._serialize_material_masks({"sky": np.ones((4, 4), dtype=np.float32)}, front_key, mask_dir)
        back_masks = orchestrator._serialize_material_masks({"sky": np.zeros((4, 4), dtype=np.float32)}, back_key, mask_dir)
        assert front_masks == mask_dir / f"{front_key.name}_materials_v3_masks.npz"
        assert back_masks == mask_dir / f"{back_key.name}_materials_v3_masks.npz"
        # The second image's masks must not have replaced the first image's.
        with np.load(front_masks) as archive:
            assert float(archive["sky"].min()) == 1.0

    @pytest.mark.parametrize("relative_input", ["image.jpg", "scene1/IMG_0042.png", "a/b/render final.tiff", "noext"])
    def test_artifact_names_are_unchanged_without_interior_dots(self, tmp_path: Path, relative_input: str) -> None:
        """For a key without an interior dot ``stem == name``, so every name stays exactly as it was."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        inputs = tmp_path / "inputs"
        output = tmp_path / "output"
        orchestrator = _create_orchestrator(output, generate_pbr=True)
        key = make_output_key(inputs / relative_input, inputs)
        assert key.stem == key.name

        assert orchestrator._expected_materials_v3_enhanced_path(key) == (
            output / "temp" / f"{key.stem}_materials_v3_enhanced.png"
        )
        assert orchestrator._segmentation_mask_artifact_path(key) == (
            output / "segmentation" / key.parent / f"{key.stem}_materials_v3_masks.npz"
        )
        mask_dir = tmp_path / "masks"
        assert orchestrator._serialize_material_masks({"sky": np.ones((4, 4), dtype=np.float32)}, key, mask_dir) == (
            mask_dir / f"{key.stem}_materials_v3_masks.npz"
        )
        pbr_assets = orchestrator._generate_pbr_stage(np.linspace(0.0, 1.0, 32 * 24, dtype=np.float32).reshape(24, 32), key)
        assert pbr_assets is not None
        for map_name in ("normal", "roughness", "ao"):
            assert Path(pbr_assets[f"{map_name}_path"]) == output / "pbr" / f"{key.stem}_{map_name}.png"


class TestAuditProvenancePreflight:
    """A RAW/TIFF input needs exiftool; a host without it must fail before inference.

    The requirement used to surface only in ``_write_manifest``, the last
    per-image step, so every TIFF ran the full depth stage, failed with the
    same environment error and left depth outputs behind without a manifest.
    """

    MESSAGE_PREFIX = "Audit-grade provenance for RAW/TIFF requires exiftool."

    def test_tiff_without_exiftool_fails_before_any_stage_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each TIFF is an immediate error row; the PNG beside it is unaffected."""
        from transformation_portal.lux_depth_v3 import provenance

        monkeypatch.setattr(provenance, "get_exiftool_version", lambda: None)
        monkeypatch.setattr(provenance, "_check_exiftool_available", lambda: False)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        tiff_names = [f"render_{index}.tif" for index in range(3)]
        for name in tiff_names:
            _make_test_image(inputs, name)
        _make_test_image(inputs, "preview.png")
        orchestrator = _create_orchestrator(output, emit_run_card=False, enable_parallel_processing=False)

        results = orchestrator.enhance_batch(inputs)

        by_name = {Path(result["image"]).name: result for result in results}
        assert sorted(by_name) == sorted([*tiff_names, "preview.png"])
        assert by_name["preview.png"]["status"] == "ok"
        for name in tiff_names:
            assert by_name[name]["status"] == "error"
            assert by_name[name]["error"].startswith(self.MESSAGE_PREFIX)
            assert by_name[name]["attempts"] == []
        # Only the PNG reached depth inference ...
        assert orchestrator.depth_backend.compute.call_count == 1
        # ... and no failed input left a Stage A output behind.
        depth_outputs = sorted(path.name for path in (output / "depth").rglob("*") if path.is_file())
        assert len(depth_outputs) == 2
        assert all(name.startswith("preview_png_") for name in depth_outputs)

    def test_prepared_tiff_batch_without_exiftool_spends_no_inference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The production path still fails closed at evidence, but before any depth compute."""
        from transformation_portal.lux_depth_v3 import provenance
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_evidence import ExecutionEvidenceError
        from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        monkeypatch.setattr(provenance, "get_exiftool_version", lambda: None)
        monkeypatch.setattr(provenance, "_check_exiftool_available", lambda: False)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        images = []
        for index in range(2):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[..., 0] = np.linspace(0, 255, 64).astype(np.uint8)[None, :]
            pixels[..., 2] = 40 * index
            Image.fromarray(pixels).save(inputs / f"render_{index}.tif")
            images.append(inputs / f"render_{index}.tif")
        prepared = prepare_lux_execution(
            EnhanceConfig(depth_backend="synthetic", allow_synthetic_fallback=True, enable_v2=False),
            inputs,
            images,
        )
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")
        real_compute = orchestrator.depth_backend.compute
        compute_calls: list[Any] = []

        def counting_compute(image: Any) -> Any:
            compute_calls.append(image)
            return real_compute(image)

        monkeypatch.setattr(orchestrator.depth_backend, "compute", counting_compute)

        with pytest.raises(ExecutionEvidenceError):
            orchestrator.enhance_batch(inputs)

        assert compute_calls == []

    def test_tiff_with_exiftool_still_runs_and_records_the_tool_version(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The preflight asks the provenance toolchain probe, so an available tool passes through."""
        from transformation_portal.lux_depth_v3 import provenance
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        monkeypatch.setattr(provenance, "get_exiftool_version", lambda: "12.76")
        monkeypatch.setattr(provenance, "_check_exiftool_available", lambda: True)
        monkeypatch.setattr(provenance, "extract_exif_metadata", lambda path: {"SourceFile": str(path)})
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        image = _make_test_image(inputs, "render.tif")
        orchestrator = _create_orchestrator(tmp_path / "output")

        result = orchestrator.enhance_image(ImageInput(path=image), input_root=inputs)

        assert result["status"] == "ok"
        assert orchestrator.depth_backend.compute.call_count == 1
        manifest_path = Path(result["manifest"])
        sidecar = json.loads(manifest_path.with_name(f"{manifest_path.stem}_provenance.json").read_text(encoding="utf-8"))
        assert sidecar["toolchain"]["exiftool_version"] == "12.76"
        assert sidecar["exif"] == {"SourceFile": str(image)}


class TestFloatDepthFreshness:
    """F10: a float-depth ``.npy`` must always pair with the depth PNG written beside it."""

    @staticmethod
    def _run(output: Path, image: Path, **config_kwargs) -> Dict[str, Any]:
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(output, **config_kwargs)
        return orchestrator.enhance_image(ImageInput(path=image), input_root=image.parent)

    def test_depth_rewrite_without_float_depth_removes_stale_npy(self, tmp_path: Path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        image = inputs / "room.png"
        Image.new("RGB", (64, 64), color="white").save(image)
        first = self._run(output, image, save_float_depth=True)
        float_depth = Path(first["depth_float_path"])
        assert float_depth == Path(first["depth_path"]).with_suffix(".npy")
        assert float_depth.is_file()

        # New content forces Stage A to rewrite the depth PNG without float depth.
        Image.new("RGB", (64, 64), color="black").save(image)
        second = self._run(output, image, save_float_depth=False)

        assert second["status"] == "ok"
        assert Path(second["depth_path"]).with_suffix(".npy") == float_depth
        assert second["depth_float_path"] is None
        assert not float_depth.exists()

    def test_float_depth_write_failure_leaves_no_partial_npy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        image = _make_test_image(inputs, "room.png")

        def _partial_save(file, arr, *args, **kwargs):
            Path(file).write_bytes(b"\x93NUMPY partial")
            raise OSError("disk full")

        monkeypatch.setattr(np, "save", _partial_save)
        with pytest.raises(Exception):
            self._run(output, image, save_float_depth=True)

        depth_dir = output / "depth"
        assert not list(depth_dir.glob("*.npy"))
        assert not [p.name for p in depth_dir.iterdir() if p.name.startswith(".tmp_")]


class TestInterruptedStageARetiresManifest:
    """F14: an interrupted Stage A rewrite must not leave new artifacts beside the old manifest."""

    def test_interrupted_depth_rewrite_is_not_resumed_under_old_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        image = _make_test_image(inputs, "room.png")
        image_input = ImageInput(path=image)

        first = _create_orchestrator(output).enhance_image(image_input, input_root=inputs)
        depth_path = Path(first["depth_path"])
        manifest_path = Path(first["manifest"])
        # Control: an untouched run 1 is resumable under the same config.
        assert _create_orchestrator(output).should_skip_depth(depth_path, manifest_path, image_input) is True

        real_writer = orchestrator_module.atomic_write_depth_u16_png_with_stats

        def _write_then_interrupt(*args, **kwargs):
            real_writer(*args, **kwargs)
            raise KeyboardInterrupt("operator interrupted the run after the depth rewrite")

        monkeypatch.setattr(orchestrator_module, "atomic_write_depth_u16_png_with_stats", _write_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            _create_orchestrator(output, quality_tier="premium").enhance_image(image_input, input_root=inputs)
        monkeypatch.setattr(orchestrator_module, "atomic_write_depth_u16_png_with_stats", real_writer)

        assert depth_path.is_file()
        assert not manifest_path.exists()
        assert _create_orchestrator(output).should_skip_depth(depth_path, manifest_path, image_input) is False

    def test_completed_rewrite_still_writes_manifest(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        image_input = ImageInput(path=_make_test_image(inputs, "room.png"))

        _create_orchestrator(output).enhance_image(image_input, input_root=inputs)
        second = _create_orchestrator(output, quality_tier="premium").enhance_image(image_input, input_root=inputs)

        assert second["status"] == "ok"
        manifest = json.loads(Path(second["manifest"]).read_text())
        assert manifest["config_fingerprint"]["quality_tier"] == "premium"

    def test_rewrite_after_retirement_keeps_restricted_manifest_mode(self, tmp_path: Path) -> None:
        """Retiring the manifest must not widen an operator-restricted mode on the rewritten manifest."""
        import stat

        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        image_input = ImageInput(path=_make_test_image(inputs, "room.png"))

        first = _create_orchestrator(output).enhance_image(image_input, input_root=inputs)
        manifest_path = Path(first["manifest"])
        manifest_path.chmod(0o600)

        second = _create_orchestrator(output, quality_tier="premium").enhance_image(image_input, input_root=inputs)

        assert Path(second["manifest"]) == manifest_path
        assert json.loads(manifest_path.read_text())["config_fingerprint"]["quality_tier"] == "premium"
        assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    def test_manifest_retirement_is_durable_and_an_empty_manifest_is_never_resumed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retirement must reach the directory like the durable Stage A writes that follow it."""
        from transformation_portal.lux_depth_v3 import io_atomic
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        image_input = ImageInput(path=_make_test_image(inputs, "room.png"))
        first = _create_orchestrator(output).enhance_image(image_input, input_root=inputs)
        depth_path = Path(first["depth_path"])
        manifest_path = Path(first["manifest"])

        events: list[tuple[str, str]] = []
        real_fsync_directory = io_atomic._fsync_directory
        real_unlink = Path.unlink

        def _recording_fsync_directory(directory: Path) -> None:
            events.append(("fsync", Path(directory).name))
            real_fsync_directory(directory)

        def _recording_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
            if self == manifest_path:
                events.append(("unlink", self.parent.name))
            real_unlink(self, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(io_atomic, "_fsync_directory", _recording_fsync_directory)
            patched.setattr(Path, "unlink", _recording_unlink)
            _create_orchestrator(output, quality_tier="premium").enhance_image(image_input, input_root=inputs)

        retired_at = events.index(("unlink", "manifests"))
        # The directory entry removal is synced before any later depth write.
        assert events[retired_at + 1] == ("fsync", "manifests")
        assert ("fsync", "depth") in events[retired_at + 2 :]

        # State left if the process stops between mode restoration and publish.
        manifest_path.write_bytes(b"")
        resumed = _create_orchestrator(output, quality_tier="premium")
        assert resumed.should_skip_depth(depth_path, manifest_path, image_input) is False


class TestStageAReuseBindsDepthShapingSettings:
    """F11: legacy Stage A reuse must not survive a change of RAW decode or depth postprocessing."""

    @staticmethod
    def _first_run(tmp_path: Path):
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        output = tmp_path / "output"
        image_input = ImageInput(path=_make_test_image(inputs, "room.png"))
        first = _create_orchestrator(output).enhance_image(image_input, input_root=inputs)
        return output, image_input, Path(first["depth_path"]), Path(first["manifest"])

    def test_reuse_requires_matching_stage_a_inputs(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.config import PostprocessingConfig

        output, image_input, depth_path, manifest_path = self._first_run(tmp_path)

        assert _create_orchestrator(output).should_skip_depth(depth_path, manifest_path, image_input) is True
        for overrides in (
            {"raw_demosaic": "VNG"},
            {"raw_ingest_mode": "force_rawpy"},
            {"depth_postprocessing": PostprocessingConfig(apply_median_filter=True, median_kernel_size=7)},
        ):
            orchestrator = _create_orchestrator(output, **overrides)
            assert orchestrator.should_skip_depth(depth_path, manifest_path, image_input) is False, overrides

    def test_manifest_without_recorded_stage_a_inputs_is_regenerated(self, tmp_path: Path) -> None:
        output, image_input, depth_path, manifest_path = self._first_run(tmp_path)

        manifest = json.loads(manifest_path.read_text())
        assert "stage_a_inputs_sha256" in manifest["depth"]["stats"]
        del manifest["depth"]["stats"]["stage_a_inputs_sha256"]
        manifest_path.write_text(json.dumps(manifest))
        assert _create_orchestrator(output).should_skip_depth(depth_path, manifest_path, image_input) is False


class TestManifestEncoding:
    """Manifests are UTF-8 JSON whatever the process locale's default encoding is."""

    @staticmethod
    def _force_ascii_default_open(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
        """Make text-mode ``open`` calls without an explicit encoding behave as under ``LC_ALL=C``."""
        import builtins

        real_open = builtins.open

        def ascii_default_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if "b" not in mode and len(args) < 2 and "encoding" not in kwargs:
                kwargs["encoding"] = "ascii"
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(module, "open", ascii_default_open, raising=False)

    def test_combined_and_batch_manifests_reload_non_ascii_under_ascii_locale(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from transformation_portal.lux_depth_v3 import manifest as manifest_module
        from transformation_portal.lux_depth_v3.manifest import BatchManifest, CombinedManifest, InputMetadata

        self._force_ascii_default_open(monkeypatch, manifest_module)
        image_path = "/shoot/Küche/Wohnzimmer_é.png"
        combined_path = tmp_path / "combined.json"
        CombinedManifest(input=InputMetadata(image_path=image_path)).save(combined_path)
        batch_path = tmp_path / "batch.json"
        BatchManifest("id", "start", "end", {"note": "Küche"}, [], {}).write(batch_path)

        assert "Küche".encode("utf-8") in combined_path.read_bytes()
        assert CombinedManifest.load(combined_path).input.image_path == image_path
        assert BatchManifest.load(batch_path).config == {"note": "Küche"}
