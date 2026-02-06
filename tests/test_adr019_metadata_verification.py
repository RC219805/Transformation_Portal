"""Verification tests for ADR-019 backend metadata capture.

Ensures that backend selection metadata is correctly captured in manifests
and depth metadata files for both DA3 and Depth Pro backends.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

# Mark all tests as ML tier
pytestmark = pytest.mark.ml

# Check if depth_anything_3 is available
try:
    import depth_anything_3  # noqa: F401

    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False


@pytest.fixture
def test_input_dir(tmp_path):
    """Create a test input directory with an image."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    image_path = input_dir / "test_image.png"
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    img.save(image_path)

    return input_dir


@pytest.mark.skipif(
    not DA3_AVAILABLE,
    reason="DA3 requires model download - disabled in offline CI",
)
def test_da3_backend_metadata_in_depth_stats(tmp_path, test_input_dir):
    """Test that DA3 backend metadata is correctly captured in depth stats."""
    output_dir = tmp_path / "output"

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        generate_pbr=False,
    )

    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Process images
    results = orchestrator.enhance_batch(input_dir=test_input_dir)

    assert len(results) == 1
    assert results[0]["status"] == "ok"

    # Find depth metadata file
    depth_metadata_files = list(output_dir.glob("depth/**/*_metadata.json"))
    assert len(depth_metadata_files) > 0, "No depth metadata files found"

    # Read and verify metadata
    with open(depth_metadata_files[0]) as f:
        metadata = json.load(f)

    # Verify backend is captured correctly
    assert "stats" in metadata
    assert metadata["stats"]["backend"] == "da3"
    assert metadata["stats"]["license"] == "commercial"  # DA3 is commercial use
    assert metadata["stats"]["unit"] == "relative"  # DA3 produces relative depth


@pytest.mark.skipif(
    not Path("checkpoints/depth_pro.pt").exists(),
    reason="Depth Pro checkpoint not available",
)
def test_depth_pro_backend_metadata_in_depth_stats(tmp_path, test_input_dir):
    """Test that Depth Pro backend metadata is correctly captured in depth stats."""
    output_dir = tmp_path / "output"

    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="cpu",
        accept_apple_depth_pro_research_license=True,
        non_commercial_ok=True,
        enable_v2=False,
        generate_pbr=False,
    )

    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Process images
    results = orchestrator.enhance_batch(input_dir=test_input_dir)

    assert len(results) == 1
    assert results[0]["status"] == "ok"

    # Find depth metadata file
    depth_metadata_files = list(output_dir.glob("depth/**/*_metadata.json"))
    assert len(depth_metadata_files) > 0, "No depth metadata files found"

    # Read and verify metadata
    with open(depth_metadata_files[0]) as f:
        metadata = json.load(f)

    # Verify backend is captured correctly
    assert "stats" in metadata
    assert metadata["stats"]["backend"] == "depth_pro"
    assert metadata["stats"]["license"] == "research_only"  # Depth Pro is research-only
    assert metadata["stats"]["unit"] == "meters"  # Depth Pro produces metric depth


@pytest.mark.skipif(
    not DA3_AVAILABLE,
    reason="DA3 requires model download - disabled in offline CI",
)
def test_backend_metadata_in_manifest(tmp_path, test_input_dir):
    """Test that backend selection metadata is captured in manifest."""
    output_dir = tmp_path / "output"

    config = EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        generate_pbr=False,
    )

    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Process images
    results = orchestrator.enhance_batch(input_dir=test_input_dir)

    assert len(results) == 1

    # Find manifest file
    manifest_files = list(output_dir.glob("manifests/**/*.json"))
    assert len(manifest_files) > 0, "No manifest files found"

    # Read and verify manifest
    with open(manifest_files[0]) as f:
        manifest = json.load(f)

    # Verify backend_selection is present
    assert "backend_selection" in manifest
    assert manifest["backend_selection"]["requested_backend"] == "da3"
    assert manifest["backend_selection"]["resolved_backend"] == "da3"
    assert manifest["backend_selection"]["resolution_status"] == "success"


def test_fallback_backend_metadata(tmp_path):
    """Test that fallback backend metadata is correctly captured."""
    output_dir = tmp_path / "output"

    # Request a backend that will fail (invalid checkpoint path for depth_pro)
    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device="cpu",
        depth_pro_checkpoint_path="/nonexistent/checkpoint.pt",
        accept_apple_depth_pro_research_license=True,
        non_commercial_ok=True,
        enable_v2=False,
        generate_pbr=False,
    )

    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Should have fallen back to DA3
    assert orchestrator.depth_backend.name == "da3"
    assert orchestrator._backend_metadata.requested_backend == "depth_pro"
    assert orchestrator._backend_metadata.resolved_backend == "da3"
    assert orchestrator._backend_metadata.resolution_status == "fallback"
