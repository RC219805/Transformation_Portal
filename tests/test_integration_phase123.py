"""Integration tests for Phase 1+2+3 interoperability (Fix #6).

Validates:
- All optimizations can be disabled (backward compatibility)
- Phase optimizations work in isolation
- Phases work together correctly
- Manifest format backward compatibility

ADR-019 Note:
Uses backend protocol mocks (DA3Backend.compute) instead of legacy
orchestrator.DA3InferenceEngine pattern.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.manifest import CombinedManifest, InputMetadata
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

# Mark all tests in this module as ML tier (require depth processing)
pytestmark = pytest.mark.ml

# ============================================================================
# Shared Test Fixtures
# ============================================================================


@pytest.fixture
def mock_depth_result():
    """Create a realistic mock DepthResult with proper array shapes (backend protocol)."""

    def _create(height=256, width=256):
        from transformation_portal.depth.backends.protocol import DepthResult as BackendDepthResult

        depth_map = np.random.rand(height, width).astype(np.float32)
        original_image = (np.random.rand(height, width, 3) * 255).astype(np.uint8)

        return BackendDepthResult(
            depth_map=depth_map,
            original_image=original_image,
            metadata={"model": "mock", "backend": "test"},
            depth_units="relative",
            focal_length_px=None,
            field_of_view_deg=None,
            backend_id="mock",
            device="cpu",
            dtype="float32",
            input_size=(height, width),
            warnings=[],
        )

    return _create


@pytest.fixture(autouse=True)
def mock_backend_compute(mock_depth_result):
    """Auto-setup mock for DA3Backend.compute() (ADR-019 compatible).

    Mocks the backend's compute() method to return fake depth results,
    allowing integration tests to run without actual ML dependencies.

    Also mocks ensure_available() to prevent orchestrator from creating
    a fallback mock backend when transformers is not installed.
    """
    with patch("transformation_portal.depth.backends.da3.DA3Backend.ensure_available"):
        with patch("transformation_portal.depth.backends.da3.DA3Backend.compute") as mock_compute:
            # Configure backend compute mock to return realistic depth result
            def _mock_compute(image, **kwargs):
                # Extract dimensions from input image
                if hasattr(image, "size"):
                    width, height = image.size
                elif hasattr(image, "shape"):
                    height, width = image.shape[:2]
                else:
                    height, width = 256, 256

                return mock_depth_result(height=height, width=width)

            mock_compute.side_effect = _mock_compute
            yield mock_compute


class TestPhase123Integration:
    """Integration tests for Phase 1+2+3 interoperability."""

    def test_all_optimizations_disabled_works(self, tmp_path):
        """Ensure all optimizations can be disabled (sequential fallback)."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        img.save(img_path, quality=95)

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            enable_manifest_cache=False,
            enable_parallel_processing=False,
            enable_depth_cache=False,
            enable_v2=False,
            chunked_hashing=False,
            # Explicitly disable all Phase 1-3 features
        )

        orch = EnhanceOrchestrator(config, tmp_path / "output")
        result = orch.enhance_image(ImageInput(img_path))

        # Verify: works correctly with all optimizations disabled
        assert result["status"] == "ok"
        assert "depth_path" in result
        assert "manifest" in result

    def test_phase1_only_enabled(self, tmp_path):
        """Test Phase 1 optimizations in isolation."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        img.save(img_path, quality=95)

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            # Phase 1: Enabled
            enable_manifest_cache=True,
            chunked_hashing=True,
            # Phase 2+3: Disabled
            enable_parallel_processing=False,
            enable_depth_cache=False,
            enable_v2=False,
        )

        orch = EnhanceOrchestrator(config, tmp_path / "output")

        # Process same image twice to test manifest cache
        result1 = orch.enhance_image(ImageInput(img_path))
        result2 = orch.enhance_image(ImageInput(img_path))

        # Verify: Phase 1 features work
        assert result1["status"] == "ok"
        assert result2["status"] == "ok"

    def test_phase1_phase2_enabled(self, tmp_path):
        """Test Phase 1+2 work together."""
        # Create test images
        test_images = []
        for i in range(5):
            img_path = tmp_path / "input" / f"test_{i}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))
            img.save(img_path, quality=95)
            test_images.append(ImageInput(img_path))

        config = EnhanceConfig(
            model_key="da3-metric",
            # Phase 1: Enabled
            enable_manifest_cache=True,
            chunked_hashing=True,
            # Phase 2: Enabled
            enable_parallel_processing=True,
            enable_depth_cache=True,
            max_parallel_workers=2,
            # Phase 3: Disabled
            enable_v2=False,
        )

        prepared = prepare_lux_execution(
            config,
            tmp_path / "input",
            [image.path for image in test_images],
        )
        orch = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")

        # Process batch
        results = orch.enhance_batch(
            prepared.input_root,
            input_files=list(prepared.input_files),
        )

        # Verify: caching + parallelization work together
        assert len(results) == 5
        assert all(r.get("status") in ["ok", "skipped"] for r in results)

    def test_all_optimizations_enabled(self, tmp_path):
        """Test Phase 1+2+3 all enabled."""
        # Create test images
        test_images = []
        for i in range(5):
            img_path = tmp_path / "input" / f"test_{i}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))
            img.save(img_path, quality=95)
            test_images.append(ImageInput(img_path))

        config = EnhanceConfig(
            model_key="da3-metric",
            # Phase 1: Enabled
            enable_manifest_cache=True,
            chunked_hashing=True,
            # Phase 2: Enabled
            enable_parallel_processing=True,
            enable_depth_cache=True,
            max_parallel_workers=2,
            # Phase 3: Enabled (PBR, skip V2 and CoreML for speed)
            enable_v2=False,
            generate_pbr=True,
        )

        prepared = prepare_lux_execution(
            config,
            tmp_path / "input",
            [image.path for image in test_images],
        )
        orch = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")

        # Process batch
        results = orch.enhance_batch(
            prepared.input_root,
            input_files=list(prepared.input_files),
        )

        # Verify: all optimizations coexist correctly
        assert len(results) == 5
        assert all(r.get("status") in ["ok", "skipped"] for r in results)

        # Verify PBR maps were generated
        pbr_dir = tmp_path / "output" / "pbr"
        if pbr_dir.exists():
            pbr_files = list(pbr_dir.glob("*"))
            # Should have normal, roughness, AO for at least some images
            assert len(pbr_files) > 0, "No PBR maps generated"

    def test_manifest_format_backward_compatible(self, tmp_path):
        """Ensure new manifests can be read by old code."""
        # Create a manifest with Phase 1+2+3 features (including pbr_assets)
        manifest_path = tmp_path / "new_manifest.json"

        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg", image_sha256="abc123", image_size_bytes=1000, image_dimensions=[100, 100]
            ),
            pbr_assets={
                "normal_path": "/path/to/normal.png",
                "roughness_path": "/path/to/roughness.png",
                "ao_path": "/path/to/ao.png",
                "runtime_seconds": 0.5,
                "config": {"normal_strength": 1.0, "roughness_strength": 0.8, "ao_strength": 0.5},
            },
        )
        manifest.write(manifest_path)

        # Read with minimal config (simulate old code without new features)
        loaded = CombinedManifest.load(manifest_path)

        # Verify: old code can still read new manifests
        assert loaded.input.image_sha256 == "abc123"

        # Verify: new fields are preserved
        assert hasattr(loaded, "pbr_assets")
        if loaded.pbr_assets:
            assert "normal_path" in loaded.pbr_assets

    def test_config_default_values_backward_compatible(self):
        """Ensure EnhanceConfig without new flags uses safe defaults."""
        # Create config without explicitly setting new flags
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            # Omit all Phase 1-3 flags (use defaults)
        )

        # Verify: defaults are backward-compatible
        assert config.enable_manifest_cache is True  # Phase 1 default: enabled
        assert config.chunked_hashing is True  # Phase 1 default: enabled
        assert config.enable_parallel_processing is True  # Phase 2 default: enabled
        # Note: enable_depth_cache defaults to False (opt-in for storage reasons)
        assert config.enable_depth_cache is False  # Phase 2 default: disabled (opt-in)

        # Verify: optional features default to sensible values
        assert config.max_parallel_workers is None  # Auto-detect
        assert config.depth_cache_max_size_gb == 10.0  # Default cache size


class TestGracefulDegradation:
    """Test graceful degradation when optional features are unavailable."""

    def test_xxhash_unavailable_fallback(self, tmp_path):
        """Test graceful fallback when xxhash is unavailable."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        img.save(img_path, quality=95)

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            use_xxhash=True,  # Request xxhash
            enable_v2=False,
        )

        # Patch xxhash as unavailable (it's defined in orchestrator module)
        with patch("transformation_portal.lux_depth_v3.orchestrator.XXHASH_AVAILABLE", False):
            orch = EnhanceOrchestrator(config, tmp_path / "output")
            result = orch.enhance_image(ImageInput(img_path))

        # Verify: falls back to SHA-256, still works
        assert result["status"] == "ok"

    def test_msgpack_unavailable_fallback(self, tmp_path):
        """Test graceful fallback when msgpack is unavailable."""
        manifest_path = tmp_path / "test_manifest.json"

        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg", image_sha256="abc123", image_size_bytes=1000, image_dimensions=[100, 100]
            )
        )

        # Patch msgpack as unavailable
        with patch("transformation_portal.lux_depth_v3.manifest.MSGPACK_AVAILABLE", False):
            # Write manifest (should use JSON fallback)
            manifest.write(manifest_path)

            # Read manifest (should use JSON)
            loaded = CombinedManifest.load(manifest_path)

        # Verify: JSON fallback works
        assert loaded.input.image_sha256 == "abc123"
        assert manifest_path.suffix == ".json"


class TestRegressionPrevention:
    """Prevent regressions in critical workflows."""

    def test_single_image_workflow_unchanged(self, tmp_path):
        """Ensure single-image workflow behavior unchanged."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        img.save(img_path, quality=95)

        # Use default config (all optimizations enabled)
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            enable_v2=False,
        )

        orch = EnhanceOrchestrator(config, tmp_path / "output")
        result = orch.enhance_image(ImageInput(img_path))

        # Verify: single-image processing works as before
        assert result["status"] == "ok"
        assert "depth_path" in result
        assert "manifest" in result
        assert "runtime_s" in result

        # Verify outputs exist
        depth_path = Path(result["depth_path"])
        manifest_path = Path(result["manifest"])
        assert depth_path.exists()
        assert manifest_path.exists()

    def test_batch_workflow_correctness(self, tmp_path):
        """Ensure batch workflow produces correct outputs for all images."""
        # Create test batch
        test_images = []
        for i in range(10):
            img_path = tmp_path / "input" / f"test_{i}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (256, 256), color=(i * 25, i * 25, i * 25))
            img.save(img_path, quality=95)
            test_images.append(ImageInput(img_path))

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            enable_v2=False,
            enable_parallel_processing=True,
        )

        orch = EnhanceOrchestrator(config, tmp_path / "output")
        results = orch.enhance_batch_parallel(test_images, input_root=tmp_path / "input")

        # Verify: all images processed correctly
        assert len(results) == 10
        assert all(r.get("status") in ["ok", "skipped"] for r in results)

        # Verify: each image has depth and manifest
        for result in results:
            if result["status"] == "ok":
                assert "depth_path" in result or result.get("depth_path") is not None
                assert "manifest" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
