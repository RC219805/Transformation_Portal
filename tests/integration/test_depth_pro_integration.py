"""Integration tests for Depth Pro with real checkpoint.

These tests require:
1. depth-pro package installed: pip install depth-pro
2. Checkpoint downloaded: checkpoints/depth_pro.pt (1.9 GB)

Run with: pytest tests/integration/test_depth_pro_integration.py -v -s
Mark: @pytest.mark.slow, @pytest.mark.ml
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Skip if checkpoint not available
CHECKPOINT_PATH = Path("checkpoints/depth_pro.pt")
CHECKPOINT_AVAILABLE = CHECKPOINT_PATH.exists()

skip_if_no_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_AVAILABLE,
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}. Download with: "
    "curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt",
)

# Try importing depth_pro
try:
    import depth_pro  # noqa: F401

    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False

skip_if_no_depth_pro = pytest.mark.skipif(
    not DEPTH_PRO_AVAILABLE, reason="depth-pro package not installed. Install with: pip install depth-pro"
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.ml
@skip_if_no_checkpoint
@skip_if_no_depth_pro
class TestDepthProIntegration:
    """Integration tests with real Depth Pro checkpoint."""

    def test_checkpoint_exists_and_size(self):
        """Verify checkpoint file exists and has expected size (~1.9 GB)."""
        assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"

        size_gb = CHECKPOINT_PATH.stat().st_size / (1024**3)
        assert 1.5 < size_gb < 2.5, f"Unexpected checkpoint size: {size_gb:.2f} GB (expected ~1.9 GB)"

        print(f"✓ Checkpoint found: {size_gb:.2f} GB")

    def test_checkpoint_sha256(self):
        """Verify checkpoint SHA-256 hash matches expected value."""
        import hashlib

        expected_hash = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"

        print(f"Computing SHA-256 for {CHECKPOINT_PATH} (this may take a minute)...")
        h = hashlib.sha256()
        with open(CHECKPOINT_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        actual_hash = h.hexdigest()

        print(f"Expected: {expected_hash}")
        print(f"Actual:   {actual_hash}")

        assert actual_hash == expected_hash, f"SHA-256 mismatch! Expected {expected_hash}, got {actual_hash}"
        print("✓ SHA-256 verified")

    def test_stage_inference_with_real_checkpoint(self):
        """Test DepthProStage inference with real checkpoint."""
        from transformation_portal.stage_graph.stage import StageContext, StageStatus
        from transformation_portal.stage_graph.stages.depth_pro import DepthProStage

        # Create test image
        test_image = Image.new("RGB", (640, 480), color=(100, 150, 200))

        # Initialize stage
        stage = DepthProStage(checkpoint_path=CHECKPOINT_PATH, device="cpu", strict_validation=True)

        # Create context
        context = StageContext(artifacts={"image": test_image})

        # Run inference
        print("Running inference with DepthProStage...")
        result = stage.compute(context)

        # Verify result
        assert result.status == StageStatus.COMPLETED, f"Inference failed: {result.error}"
        assert "depth_map" in result.artifacts
        assert "depth_provenance" in result.artifacts

        depth_map = result.artifacts["depth_map"]
        assert isinstance(depth_map, np.ndarray)
        assert depth_map.dtype == np.float32
        assert depth_map.shape == (480, 640)  # H, W

        # Verify depth is metric (reasonable values in meters)
        assert np.all(np.isfinite(depth_map)), "Depth map contains non-finite values"
        assert depth_map.min() >= 0, "Depth cannot be negative"
        assert depth_map.max() < 1000, "Unreasonably large depth values"

        # Verify provenance
        prov = result.artifacts["depth_provenance"]
        assert prov["status"] == "ok"
        assert prov["engine"] == "apple_depth_pro"
        assert "checkpoint" in prov
        assert prov["checkpoint"]["sha256"] == "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"

        print(f"✓ Inference successful")
        print(f"  Depth range: {depth_map.min():.2f} - {depth_map.max():.2f} meters")
        print(f"  Median depth: {np.median(depth_map):.2f} meters")
        print(f"  Inference time: {prov['timing']['inference_sec']:.3f}s")

    def test_backend_inference_with_real_checkpoint(self):
        """Test DepthProBackend inference with real checkpoint."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        # Mock config
        class MockConfig:
            non_commercial_ok = True
            accept_apple_depth_pro_research_license = True
            depth_device = "cpu"
            depth_pro_checkpoint_path = str(CHECKPOINT_PATH)

        # Create backend
        backend = DepthProBackend(MockConfig())

        # Ensure available
        backend.ensure_available()

        # Create test image
        test_image = Image.new("RGB", (640, 480), color=(100, 150, 200))

        # Run inference
        print("Running inference with DepthProBackend...")
        result = backend.compute(test_image, device="cpu")

        # Verify result
        assert result.depth_map is not None
        assert isinstance(result.depth_map, np.ndarray)
        assert result.depth_map.dtype == np.float32
        assert result.depth_units == "meters"
        assert result.is_metric is True
        assert result.backend_id == "depth_pro"

        print(f"✓ Backend inference successful")
        print(f"  Depth units: {result.depth_units}")
        print(f"  Shape: {result.depth_map.shape}")
        print(f"  Range: {result.depth_map.min():.2f} - {result.depth_map.max():.2f} meters")

    def test_registry_integration(self):
        """Test DepthBackendRegistry with Depth Pro."""
        from transformation_portal.depth.backends import DepthBackendRegistry

        # Mock config
        class MockConfig:
            non_commercial_ok = True
            accept_apple_depth_pro_research_license = True
            depth_device = "cpu"
            depth_pro_checkpoint_path = str(CHECKPOINT_PATH)

        # Get backend from registry
        registry = DepthBackendRegistry()
        backend = registry.get_backend("depth_pro", MockConfig())

        assert backend.name == "depth_pro"
        assert backend.requires_checkpoint is True

