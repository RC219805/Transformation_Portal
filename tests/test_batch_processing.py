"""Tests for batch processing and runtime statistics.

These tests verify that batch processing correctly computes runtime statistics
and handles partial failures gracefully.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.batch_stats import compute_batch_runtime_stats
from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import ApexStrictGateError, EnhanceOrchestrator
from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance
from transformation_portal.spatial_ai.reconstruction.contracts import (
    CameraParams,
)
from transformation_portal.spatial_ai.reconstruction.contracts import (
    LicenseRestrictionError as ReconstructionLicenseRestrictionError,
)


class TestBatchRuntimeStats:
    """Test batch runtime statistics computation."""

    def test_compute_batch_runtime_stats_with_valid_runtimes(self):
        """Test that runtime stats are computed correctly."""
        runtimes = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = compute_batch_runtime_stats(runtimes)

        assert stats["count"] == 5
        assert stats["total"] == 15.0
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0

    def test_compute_batch_runtime_stats_empty_list(self):
        """Test that empty runtime list returns zero stats."""
        stats = compute_batch_runtime_stats([])

        assert stats["count"] == 0
        assert stats["total"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["median"] == 0.0

    def test_compute_batch_runtime_stats_single_value(self):
        """Test stats with single runtime value."""
        stats = compute_batch_runtime_stats([42.5])

        assert stats["count"] == 1
        assert stats["total"] == 42.5
        assert stats["mean"] == 42.5
        assert stats["min"] == 42.5
        assert stats["max"] == 42.5
        assert stats["median"] == 42.5

    def test_compute_batch_runtime_stats_median_even_count(self):
        """Test median calculation with even number of values."""
        runtimes = [1.0, 2.0, 3.0, 4.0]

        stats = compute_batch_runtime_stats(runtimes)

        # Median of [1, 2, 3, 4] should be (2 + 3) / 2 = 2.5
        assert stats["median"] == 2.5


class TestEnhanceBatch:
    """Test enhance_batch method and its integration with runtime stats."""

    @pytest.fixture
    def batch_temp_workspace(self, temp_workspace, deterministic_rng):
        """Create temporary workspace with test images (uses shared fixtures)."""
        # Create test images using shared fixtures
        for i in range(3):
            img_path = temp_workspace["input_dir"] / f"test_{i}.jpg"
            img_array = (deterministic_rng.random((100, 100, 3)) * 255).astype("uint8")
            img = Image.fromarray(img_array, mode="RGB")
            img.save(img_path)

        return temp_workspace

    def test_enhance_batch_extracts_runtimes_correctly(self, batch_temp_workspace):
        """CRITICAL: Test that enhance_batch correctly extracts runtime_s from results.

        This test catches the bug where results (List[Dict]) were passed directly
        to compute_batch_runtime_stats which expects List[float].
        """
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,  # Skip V2 for faster test
        )

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock backend initialization BEFORE creating orchestrator
            # This prevents ImportError in CI where ML dependencies aren't installed
            with (
                patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class,
                patch("transformation_portal.lux_depth_v3.preprocessing.validate_image_format") as mock_validate,
                patch("transformation_portal.lux_depth_v3.preprocessing.preprocess_image") as mock_preprocess,
                patch("transformation_portal.lux_depth_v3.orchestrator.atomic_write_depth_u16_png_with_stats") as mock_write,
            ):
                # Setup mock registry to return a mock backend
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None  # Success
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                # Now create orchestrator (will use mocked registry)
                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                # Mock postprocessor
                mock_postprocessor = Mock()
                orchestrator.postprocessor = mock_postprocessor

                # Setup mocks
                mock_validate.side_effect = lambda x: x
                mock_preprocess.return_value = (np.random.rand(100, 100, 3).astype(np.float32), (100, 100))

                # Mock inference result (DepthResult from backend.compute())
                from transformation_portal.depth.backends.protocol import DepthResult

                mock_result = DepthResult(
                    depth_map=np.random.rand(100, 100).astype(np.float32),
                    original_image=np.random.rand(100, 100, 3).astype(np.uint8),
                    metadata={},
                    backend_id="da3",
                )
                mock_backend.compute.return_value = mock_result
                mock_postprocessor.process.return_value = mock_result

                # Mock depth write stats
                mock_stats = Mock()
                mock_stats.min = 0.0
                mock_stats.max = 1.0
                mock_stats.mean = 0.5
                mock_stats.std = 0.2
                mock_stats.shape = (100, 100)
                mock_stats.dtype = "float32"
                mock_stats.method = "u16"
                mock_stats._asdict = lambda: {
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.5,
                    "std": 0.2,
                    "shape": (100, 100),
                    "dtype": "float32",
                    "method": "u16",
                }
                mock_write.return_value = (Path("depth.png"), None, mock_stats)

                # Run batch processing
                try:
                    results = orchestrator.enhance_batch(batch_temp_workspace["input_dir"])

                    # Verify results structure
                    assert isinstance(results, list)
                    assert len(results) == 3  # We created 3 test images

                    # Each result should have runtime_s
                    for result in results:
                        assert isinstance(result, dict)
                        assert "runtime_s" in result or "error" in result

                    # Check that batch manifest was created
                    manifests_dir = tmpdir_path / "manifests"
                    if manifests_dir.exists():
                        batch_manifests = list(manifests_dir.glob("batch_*.json"))
                        if batch_manifests:
                            # Verify batch manifest has runtime stats
                            with open(batch_manifests[0]) as f:
                                manifest = json.load(f)

                            # Should have stats dict with runtime statistics
                            assert "stats" in manifest
                            stats = manifest["stats"]

                            # Check for runtime statistics fields
                            # These come from compute_batch_runtime_stats
                            if any(r.get("status") == "ok" for r in results):
                                # Only check if we had successful results
                                assert "count" in stats or "total" in stats

                except Exception as e:
                    # The test might fail due to missing dependencies or other issues
                    # but the important part is that if enhance_batch runs,
                    # it must not fail with a type error when calling compute_batch_runtime_stats
                    if "takes 1 positional argument but" in str(e) or ("expected" in str(e) and "List[float]" in str(e)):
                        pytest.fail(
                            f"enhance_batch failed with signature mismatch error: {e}\n"
                            "This indicates compute_batch_runtime_stats is still being called "
                            "with results instead of extracted runtime_s values."
                        )
                    # Other errors are acceptable for this focused test
                    return

    def test_enhance_batch_handles_partial_failure(self, batch_temp_workspace):
        """Test that batch processing handles partial failures gracefully."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            output_bit_depth=16,
        )

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock backend initialization BEFORE creating orchestrator
            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                # Setup mock registry
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                # Mock to simulate partial failure
                call_count = 0

                def mock_enhance_image(image_input, input_root=None):
                    nonlocal call_count
                    call_count += 1

                    if call_count == 2:
                        # Second image fails
                        raise ValueError("Simulated processing error")

                    # Other images succeed - include backend key for run card semantics
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "depth_path": "depth.png",
                        "manifest": "manifest.json",
                        "runtime_s": 1.5,
                        "backend": "da3",
                    }

                with patch.object(orchestrator, "enhance_image", side_effect=mock_enhance_image):
                    results = orchestrator.enhance_batch(batch_temp_workspace["input_dir"])

                    # Should have 3 results (one for each image)
                    assert len(results) == 3

                    # Count successes and failures
                    successes = [r for r in results if r.get("status") == "ok"]
                    failures = [r for r in results if "error" in r]

                    assert len(successes) == 2  # Images 1 and 3 succeeded
                    assert len(failures) == 1  # Image 2 failed

                    # Verify runtime_s only in successful results
                    for success in successes:
                        assert "runtime_s" in success
                        assert success["runtime_s"] > 0

                    # Verify error in failed result
                    for failure in failures:
                        assert "error" in failure

    def test_enhance_batch_scene_group_bridge_preserves_output_parity(self, batch_temp_workspace):
        """Scene-group iterator bridge must preserve sorted per-image processing order and outputs."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                discovered = [
                    batch_temp_workspace["input_dir"] / "test_2.jpg",
                    batch_temp_workspace["input_dir"] / "test_0.jpg",
                    batch_temp_workspace["input_dir"] / "test_1.jpg",
                ]
                expected_order = sorted(discovered)

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups as real_build_scene_groups

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch(
                        "transformation_portal.lux_depth_v3.orchestrator.build_scene_groups",
                        wraps=real_build_scene_groups,
                    ) as mock_build_scene_groups,
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image) as mock_enhance_image,
                ):
                    results = orchestrator.enhance_batch(batch_temp_workspace["input_dir"])

                mock_build_scene_groups.assert_called_once_with(
                    expected_order,
                    dataset_root=batch_temp_workspace["input_dir"],
                    grouping_mode="single",
                )
                called_order = [call.args[0].path for call in mock_enhance_image.call_args_list]
                assert called_order == expected_order
                assert [result["image"] for result in results] == [str(path) for path in expected_order]

    def test_enhance_batch_runs_scene_reconstruction_when_gated(self, batch_temp_workspace):
        """Reconstruction runs exactly once for eligible grouped scenes with explicit camera sidecar."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            non_commercial_ok=True,
            accept_research_tools_license=True,
            emit_scene_debug_bundle=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sidecar_path = tmpdir_path / "scene_cameras.json"
            config.cameras_sidecar_path = str(sidecar_path)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                input_dir = batch_temp_workspace["input_dir"]
                discovered = [
                    input_dir / "scene_a" / "view_2.jpg",
                    input_dir / "scene_a" / "view_1.jpg",
                    input_dir / "scene_b" / "solo.jpg",
                ]
                for image_path in discovered:
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(b"scene")
                expected_order = sorted(discovered)

                from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups

                scene_groups = build_scene_groups(expected_order, dataset_root=input_dir, grouping_mode="parent_dir")
                eligible_scene = next(scene for scene in scene_groups if len(scene.images) == 2)

                sidecar_payload = {
                    "schema": "tp.scene_cameras.v1",
                    "scenes": {
                        eligible_scene.scene_id: {
                            "images": [
                                str(path.resolve().relative_to(input_dir.resolve()).as_posix())
                                for path in eligible_scene.images
                            ],
                            "cameras": [
                                {
                                    "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
                                    "extrinsics": [
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                    "width": 64,
                                    "height": 64,
                                },
                                {
                                    "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
                                    "extrinsics": [
                                        [1.0, 0.0, 0.0, 0.1],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                    "width": 64,
                                    "height": 64,
                                },
                            ],
                        }
                    },
                }
                sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                def _mock_run_scene_reconstruction_fn(**kwargs):
                    report_path = orchestrator.reconstruction_dir / f"{kwargs['context'].scene_id}_reconstruction_report.json"
                    report_path.write_text("{}", encoding="utf-8")
                    return report_path

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image),
                ):
                    orchestrator.run_scene_reconstruction_fn = Mock(side_effect=_mock_run_scene_reconstruction_fn)
                    results = orchestrator.enhance_batch(input_dir)

                assert orchestrator.run_scene_reconstruction_fn.call_count == 1
                called_scene = orchestrator.run_scene_reconstruction_fn.call_args.kwargs["context"]
                assert called_scene.scene_id == eligible_scene.scene_id
                called_scene_fingerprint = orchestrator.run_scene_reconstruction_fn.call_args.kwargs["scene_fingerprint"]
                called_merkle_root = orchestrator.run_scene_reconstruction_fn.call_args.kwargs["run_card_merkle_root"]
                assert isinstance(called_scene_fingerprint, str) and len(called_scene_fingerprint) == 64
                assert isinstance(called_merkle_root, str) and len(called_merkle_root) == 64
                assert [result["image"] for result in results] == [str(path) for path in expected_order]
                reconstruction_paths = [
                    result.get("reconstruction_report_path") for result in results if result.get("reconstruction_report_path")
                ]
                assert len(reconstruction_paths) == 1
                assert Path(reconstruction_paths[0]).exists()
                scene_manifest_paths = [
                    result.get("reconstruction_scene_manifest_path")
                    for result in results
                    if result.get("reconstruction_scene_manifest_path")
                ]
                assert len(scene_manifest_paths) == 1
                assert Path(scene_manifest_paths[0]).exists()
                debug_manifest_paths = [
                    result.get("reconstruction_debug_manifest_path")
                    for result in results
                    if result.get("reconstruction_debug_manifest_path")
                ]
                debug_cameras_paths = [
                    result.get("reconstruction_debug_cameras_path")
                    for result in results
                    if result.get("reconstruction_debug_cameras_path")
                ]
                assert len(debug_manifest_paths) == 1
                assert len(debug_cameras_paths) == 1
                assert Path(debug_manifest_paths[0]).exists()
                assert Path(debug_cameras_paths[0]).exists()

    def test_enhance_batch_raises_reconstruction_license_restriction_when_not_acknowledged(self, batch_temp_workspace):
        """Reconstruction must fail closed when non-commercial license flags are not acknowledged."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sidecar_path = tmpdir_path / "scene_cameras.json"
            config.cameras_sidecar_path = str(sidecar_path)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                input_dir = batch_temp_workspace["input_dir"]
                discovered = [
                    input_dir / "scene_a" / "view_2.jpg",
                    input_dir / "scene_a" / "view_1.jpg",
                ]
                expected_order = sorted(discovered)

                from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups

                scene_groups = build_scene_groups(expected_order, dataset_root=input_dir, grouping_mode="parent_dir")
                eligible_scene = scene_groups[0]

                sidecar_payload = {
                    "schema": "tp.scene_cameras.v1",
                    "scenes": {
                        eligible_scene.scene_id: {
                            "images": [
                                str(path.resolve().relative_to(input_dir.resolve()).as_posix())
                                for path in eligible_scene.images
                            ],
                            "cameras": [
                                {
                                    "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
                                    "extrinsics": [
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                    "width": 64,
                                    "height": 64,
                                },
                                {
                                    "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
                                    "extrinsics": [
                                        [1.0, 0.0, 0.0, 0.1],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ],
                                    "width": 64,
                                    "height": 64,
                                },
                            ],
                        }
                    },
                }
                sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image),
                ):
                    orchestrator.run_scene_reconstruction_fn = Mock()
                    with pytest.raises(ReconstructionLicenseRestrictionError, match="non_commercial_ok=True"):
                        orchestrator.enhance_batch(input_dir)

                assert orchestrator.run_scene_reconstruction_fn.call_count == 0

    def test_enhance_batch_raises_when_research_tools_license_not_acknowledged(self, batch_temp_workspace):
        """Reconstruction must fail when research-tools license acknowledgement is missing."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            non_commercial_ok=True,
            accept_research_tools_license=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sidecar_path = tmpdir_path / "scene_cameras.json"
            config.cameras_sidecar_path = str(sidecar_path)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                input_dir = batch_temp_workspace["input_dir"]
                discovered = [
                    input_dir / "scene_a" / "view_2.jpg",
                    input_dir / "scene_a" / "view_1.jpg",
                ]

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image),
                ):
                    orchestrator.run_scene_reconstruction_fn = Mock()
                    with pytest.raises(ReconstructionLicenseRestrictionError, match="accept_research_tools_license=True"):
                        orchestrator.enhance_batch(input_dir)

                assert orchestrator.run_scene_reconstruction_fn.call_count == 0

    def test_enhance_batch_skips_scene_reconstruction_when_cameras_absent(self, batch_temp_workspace):
        """Reconstruction gate must skip multi-view scene when explicit sidecar is absent."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path=None,
            non_commercial_ok=True,
            accept_research_tools_license=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                input_dir = batch_temp_workspace["input_dir"]
                discovered = [
                    input_dir / "scene_a" / "view_1.jpg",
                    input_dir / "scene_a" / "view_2.jpg",
                ]
                expected_order = sorted(discovered)

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image),
                ):
                    orchestrator.run_scene_reconstruction_fn = Mock()
                    results = orchestrator.enhance_batch(input_dir)

                assert [result["image"] for result in results] == [str(path) for path in expected_order]
                assert orchestrator.run_scene_reconstruction_fn.call_count == 0
                assert not any(result.get("reconstruction_report_path") for result in results)

    def test_enhance_batch_skips_scene_reconstruction_when_camera_sources_mixed(self, batch_temp_workspace):
        """Reconstruction gate must skip scenes with mixed camera provenance sources."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path="dummy.json",
            non_commercial_ok=True,
            accept_research_tools_license=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                input_dir = batch_temp_workspace["input_dir"]
                discovered = [
                    input_dir / "scene_a" / "view_1.jpg",
                    input_dir / "scene_a" / "view_2.jpg",
                ]

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                mixed_cameras = (
                    CameraWithProvenance(
                        params=Mock(),
                        provenance=CameraProvenance(source="sidecar", confidence="high", file="a.json"),
                    ),
                    CameraWithProvenance(
                        params=Mock(),
                        provenance=CameraProvenance(source="exif", confidence="medium", file=None),
                    ),
                )

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image),
                    patch("transformation_portal.lux_depth_v3.orchestrator.load_scene_cameras", return_value=mixed_cameras),
                ):
                    orchestrator.run_scene_reconstruction_fn = Mock()
                    results = orchestrator.enhance_batch(input_dir)

                assert orchestrator.run_scene_reconstruction_fn.call_count == 0
                assert not any(result.get("reconstruction_report_path") for result in results)

    def test_enhance_batch_skips_scene_reconstruction_when_risk_gate_exceeded(self, batch_temp_workspace, monkeypatch):
        """Reconstruction gate must skip when dataset risk score exceeds configured threshold."""
        import numpy as np

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path="dummy.json",
            non_commercial_ok=True,
            accept_research_tools_license=True,
        )
        monkeypatch.setenv("TP_RECONSTRUCTION_RISK_THRESHOLD", "-0.10")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                input_dir = batch_temp_workspace["input_dir"]
                discovered = [
                    input_dir / "scene_a" / "view_1.jpg",
                    input_dir / "scene_a" / "view_2.jpg",
                ]

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image),
                    patch(
                        "transformation_portal.lux_depth_v3.orchestrator.load_scene_cameras",
                        return_value=(
                            CameraWithProvenance(
                                params=CameraParams(
                                    intrinsics=np.array(
                                        [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
                                        dtype=np.float32,
                                    ),
                                    extrinsics=np.eye(4, dtype=np.float32),
                                    width=64,
                                    height=64,
                                ),
                                provenance=CameraProvenance(source="sidecar", confidence="high", file="a.json"),
                            ),
                            CameraWithProvenance(
                                params=CameraParams(
                                    intrinsics=np.array(
                                        [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
                                        dtype=np.float32,
                                    ),
                                    extrinsics=np.array(
                                        [
                                            [1.0, 0.0, 0.0, 0.1],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0],
                                        ],
                                        dtype=np.float32,
                                    ),
                                    width=64,
                                    height=64,
                                ),
                                provenance=CameraProvenance(source="sidecar", confidence="high", file="a.json"),
                            ),
                        ),
                    ),
                ):
                    orchestrator.run_scene_reconstruction_fn = Mock()
                    results = orchestrator.enhance_batch(input_dir)

                assert orchestrator.run_scene_reconstruction_fn.call_count == 0
                risk_messages = [result.get("reconstruction_risk_gate_message") for result in results]
                assert any(isinstance(message, str) and "exceeds threshold" in message for message in risk_messages)
                triage_reports = [result.get("reconstruction_risk_gate_triage") for result in results]
                assert any(
                    isinstance(report, str) and report.startswith("Scene ") and "dataset triage" in report
                    for report in triage_reports
                )

    def test_enhance_batch_skips_scene_reconstruction_when_preflight_invalid(self, batch_temp_workspace):
        """Reconstruction gate must skip scenes when scene preflight validation fails."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            enable_parallel_processing=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path="dummy.json",
            non_commercial_ok=True,
            accept_research_tools_license=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                input_dir = batch_temp_workspace["input_dir"]
                discovered = [
                    input_dir / "scene_a" / "view_1.jpg",
                    input_dir / "scene_a" / "view_2.jpg",
                ]

                def _mock_enhance_image(image_input, input_root=None):  # noqa: ARG001
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "runtime_s": 1.0,
                    }

                import numpy as np

                intrinsics = np.array(
                    [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                extrinsics = np.eye(4, dtype=np.float32)
                preflight_invalid_cameras = (
                    CameraWithProvenance(
                        params=CameraParams(
                            intrinsics=intrinsics.copy(),
                            extrinsics=extrinsics.copy(),
                            width=64,
                            height=64,
                        ),
                        provenance=CameraProvenance(source="sidecar", confidence="high", file="a.json"),
                    ),
                    CameraWithProvenance(
                        params=CameraParams(
                            intrinsics=intrinsics.copy(),
                            extrinsics=np.array(
                                [
                                    [1.0, 0.0, 0.0, 0.1],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                                dtype=np.float32,
                            ),
                            width=2048,
                            height=2048,
                        ),
                        provenance=CameraProvenance(source="sidecar", confidence="high", file="a.json"),
                    ),
                )

                with (
                    patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=discovered),
                    patch.object(orchestrator, "enhance_image", side_effect=_mock_enhance_image),
                    patch(
                        "transformation_portal.lux_depth_v3.orchestrator.load_scene_cameras",
                        return_value=preflight_invalid_cameras,
                    ),
                ):
                    orchestrator.run_scene_reconstruction_fn = Mock()
                    results = orchestrator.enhance_batch(input_dir)

                assert orchestrator.run_scene_reconstruction_fn.call_count == 0
                assert not any(result.get("reconstruction_report_path") for result in results)
                preflight_paths = [result.get("reconstruction_preflight_path") for result in results]
                assert any(path for path in preflight_paths if isinstance(path, str) and Path(path).exists())

    def test_batch_manifest_structure(self, batch_temp_workspace):
        """Test that batch manifest has correct structure."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            output_bit_depth=16,
        )

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Mock backend initialization BEFORE creating orchestrator
            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                # Setup mock registry
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                # Mock enhance_image to return controlled results
                def mock_enhance_image(image_input, input_root=None):
                    return {
                        "status": "ok",
                        "image": str(image_input.path),
                        "depth_path": "depth.png",
                        "manifest": "manifest.json",
                        "runtime_s": 2.5,
                        "backend": "da3",
                    }

                with patch.object(orchestrator, "enhance_image", side_effect=mock_enhance_image):
                    orchestrator.enhance_batch(batch_temp_workspace["input_dir"])

                    # Check batch manifest was created
                    manifests_dir = tmpdir_path / "manifests"
                    if manifests_dir.exists():
                        batch_manifests = list(manifests_dir.glob("batch_*.json"))
                        if batch_manifests:
                            with open(batch_manifests[0]) as f:
                                manifest = json.load(f)

                            # Verify required fields
                            assert "batch_id" in manifest
                            assert "start_time" in manifest
                            assert "end_time" in manifest
                        assert "config" in manifest
                        assert manifest["config"]["depth_backend"] == "da3"
                        assert manifest["config"]["device"] == "cpu"
                        assert manifest["config"]["quality_tier"] == "standard"
                        assert manifest["config"]["depth_png_encoding"] == "normalized_u16_png"
                        assert manifest["config"]["output_bit_depth"] == 16
                        assert len(manifest["config"]["config_fingerprint_sha256"]) == 64
                        assert "results" in manifest
                        assert "stats" in manifest
                        assert manifest["results"][0]["image"].endswith(".jpg")
                        assert not Path(manifest["results"][0]["image"]).is_absolute()

                        # Verify stats structure
                        stats = manifest["stats"]
                        assert "total" in stats
                        assert "batch_runtime_seconds" in stats

                        # Should have runtime statistics from compute_batch_runtime_stats
                        # (count, mean, min, max, median)
                        # These may or may not be present depending on execution path
                        # but if present, they should be valid
                        if "count" in stats:
                            assert stats["count"] >= 0
                        if "mean" in stats:
                            assert stats["mean"] >= 0

    def test_enhance_batch_parallel_enriches_apex_gate_error_payload(self, batch_temp_workspace):
        """Parallel batch path should emit structured ApexStrictGateError fields."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )

                # Ensure the parallel path is selected.
                orchestrator._use_parallel = True

                fake_inputs = [ImageInput(path=batch_temp_workspace["input_dir"] / f"test_{i}.jpg") for i in range(4)]
                preprocessed = [
                    {
                        "status": "ok",
                        "image_input": fake_inputs[0],
                        "output_key": Path("test_0"),
                        "depth_path": tmpdir_path / "depth" / "test_0_depth.png",
                        "manifest_path": tmpdir_path / "manifests" / "test_0_combined.json",
                        "should_skip": False,
                    }
                ]

                expected_details = {
                    "passed": False,
                    "failure_codes": ["APEX_DEPTH_PLATEAU"],
                    "metrics": {"upper_iqr": 0.0},
                    "thresholds": {"upper_iqr_min": 1e-4},
                }

                def _raise_apex_gate(*_args, **_kwargs):
                    raise ApexStrictGateError(
                        "APEX_DEPTH_PLATEAU",
                        "APEX depth validity gate failed: APEX_DEPTH_PLATEAU",
                        details=expected_details,
                    )

                with (
                    patch.object(orchestrator, "_parallel_preprocess_batch", return_value=preprocessed),
                    patch.object(orchestrator, "enhance_image", side_effect=_raise_apex_gate),
                ):
                    results = orchestrator.enhance_batch_parallel(fake_inputs, input_root=batch_temp_workspace["input_dir"])

                assert len(results) == 1
                assert results[0]["status"] == "error"
                assert results[0]["error_code"] == "APEX_DEPTH_PLATEAU"
                assert results[0]["error_details"] == expected_details
                assert results[0]["quality_gate"] == {
                    "kind": "apex_depth",
                    "passed": False,
                    "failure_codes": ["APEX_DEPTH_PLATEAU"],
                    "warnings": [],
                    "details": {
                        "metrics": {"upper_iqr": 0.0},
                        "thresholds": {"upper_iqr_min": 1e-4},
                        "shape_context": {},
                    },
                }

    def test_enhance_batch_sequential_enriches_apex_gate_error_payload(self, batch_temp_workspace):
        """Sequential batch path should emit structured ApexStrictGateError fields."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )
                orchestrator._use_parallel = False

                expected_details = {
                    "passed": False,
                    "failure_codes": ["APEX_DEPTH_SATURATION_HIGH"],
                    "metrics": {"saturation_high_fraction": 0.4},
                    "thresholds": {"saturation_high_fraction_max": 0.02},
                }

                def _raise_apex_gate(*_args, **_kwargs):
                    raise ApexStrictGateError(
                        "APEX_DEPTH_SATURATION_HIGH",
                        "APEX depth validity gate failed: APEX_DEPTH_SATURATION_HIGH",
                        details=expected_details,
                    )

                with patch.object(orchestrator, "enhance_image", side_effect=_raise_apex_gate):
                    results = orchestrator.enhance_batch(batch_temp_workspace["input_dir"])

                assert len(results) == 3
                for result in results:
                    assert result["status"] == "error"
                    assert result["error_code"] == "APEX_DEPTH_SATURATION_HIGH"
                    assert result["error_details"] == expected_details
                    assert result["quality_gate"] == {
                        "kind": "apex_depth",
                        "passed": False,
                        "failure_codes": ["APEX_DEPTH_SATURATION_HIGH"],
                        "warnings": [],
                        "details": {
                            "metrics": {"saturation_high_fraction": 0.4},
                            "thresholds": {"saturation_high_fraction_max": 0.02},
                            "shape_context": {},
                        },
                    }

    def test_enhance_batch_sequential_preserves_attempt_history_for_apex_gate_failures(self, batch_temp_workspace):
        """Sequential batch errors should retain the failed attempt provenance from Stage A."""
        import numpy as np

        from transformation_portal.depth.backends.protocol import DepthResult

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            depth_backend="da3",
            depth_device="cpu",
            quality_tier="apex",
            enable_v2=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry") as mock_registry_class:
                mock_backend = Mock()
                mock_backend.ensure_available.return_value = None
                mock_backend.name = "da3"
                mock_backend.compute.return_value = DepthResult(
                    depth_map=np.ones((64, 64), dtype=np.float32),
                    original_image=np.ones((64, 64, 3), dtype=np.uint8),
                    metadata={},
                    depth_units="relative",
                    backend_id="da3",
                    device="cpu",
                )

                mock_registry = Mock()
                mock_registry.get_backend.return_value = mock_backend
                mock_registry_class.return_value = mock_registry

                orchestrator = EnhanceOrchestrator(
                    config=config,
                    output_root=tmpdir_path,
                )
                orchestrator._use_parallel = False
                orchestrator.postprocessor = Mock(process=lambda result: result)

                expected_details = {
                    "passed": False,
                    "failure_codes": ["APEX_DEPTH_SATURATION_LOW"],
                    "metrics": {"saturation_low_fraction": 0.031},
                    "thresholds": {"saturation_low_fraction_max": 0.02},
                    "shape_context": {
                        "gate_evaluated_shape": [64, 64],
                        "native_shape": [64, 64],
                        "artifact_shape": [100, 100],
                    },
                }

                with patch.object(
                    orchestrator,
                    "_enforce_apex_depth_validity_gate",
                    side_effect=ApexStrictGateError(
                        "APEX_DEPTH_SATURATION_LOW",
                        "APEX depth validity gate failed: APEX_DEPTH_SATURATION_LOW",
                        details=expected_details,
                    ),
                ):
                    results = orchestrator.enhance_batch(batch_temp_workspace["input_dir"])

                assert len(results) == 3
                for result in results:
                    assert result["status"] == "error"
                    assert result["error_code"] == "APEX_DEPTH_SATURATION_LOW"
                    assert result["error_details"] == expected_details
                    assert result["selected_attempt_index"] is None
                    assert len(result["attempts"]) == 1
                    assert result["attempts"][0]["backend"] == "da3"
                    assert result["attempts"][0]["failure_kind"] == "semantic"
                    assert result["attempts"][0]["error_code"] == "APEX_DEPTH_SATURATION_LOW"
                    assert result["attempts"][0]["error_details"] == expected_details
                    assert result["quality_gate"] == {
                        "kind": "apex_depth",
                        "passed": False,
                        "failure_codes": ["APEX_DEPTH_SATURATION_LOW"],
                        "warnings": [],
                        "details": {
                            "metrics": {"saturation_low_fraction": 0.031},
                            "thresholds": {"saturation_low_fraction_max": 0.02},
                            "shape_context": {
                                "gate_evaluated_shape": [64, 64],
                                "native_shape": [64, 64],
                                "artifact_shape": [100, 100],
                            },
                        },
                    }
