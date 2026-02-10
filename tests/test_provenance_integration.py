"""Integration tests for provenance capture in the orchestrator.

These tests validate that provenance sidecars are correctly captured
and written during the full pipeline execution.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.lux_depth_v3.config import DA3Config, EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from transformation_portal.lux_depth_v3.provenance import PROVENANCE_SCHEMA_VERSION


@pytest.fixture
def minimal_enhance_config(tmp_path: Path) -> EnhanceConfig:
    """Minimal EnhanceConfig for testing."""
    return EnhanceConfig(
        model_variant=ModelVariant.METRIC_SMALL,  # Use smallest variant for testing
        depth_device="cpu",
        depth_quantization="none",
        depth_fallback="skip",
        enable_v2=False,  # Disable V2 for faster testing
        force_depth=True,  # Force depth computation (no caching)
        verify_depth_writes=False,  # Skip verification for speed
    )


@pytest.fixture
def orchestrator_with_provenance(minimal_enhance_config: EnhanceConfig, tmp_path: Path) -> EnhanceOrchestrator:
    """Create orchestrator configured for provenance testing."""
    output_dir = tmp_path / "output"
    
    orchestrator = EnhanceOrchestrator(
        config=minimal_enhance_config,
        output_dir=output_dir,
        enable_cache=False,  # Disable cache for clean testing
    )
    
    return orchestrator


@pytest.mark.integration
class TestProvenanceIntegration:
    """Integration tests for provenance capture during pipeline execution."""

    def test_provenance_sidecar_created_for_tiff(
        self,
        orchestrator_with_provenance: EnhanceOrchestrator,
        tmp_path: Path,
    ):
        """Test that provenance sidecar is created during TIFF processing."""
        # Check if exiftool is available
        try:
            result = subprocess.run(
                ["exiftool", "-ver"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                pytest.skip("exiftool not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("exiftool not available")

        # Use real TIFF fixture
        fixture_path = Path(
            "tests/fixtures/pipelines/750_picacho_lane/input/750Picacho_GreatRoom_UltraQuality.tif"
        )
        if not fixture_path.exists():
            pytest.skip("Real TIFF fixture not available")

        # Create ImageInput
        image_input = ImageInput(path=fixture_path)

        # Process image (this should trigger provenance capture)
        # Mock the depth backend to avoid ML dependencies
        with patch.object(
            orchestrator_with_provenance,
            "depth_backend",
            MagicMock(
                name="synthetic",
                compute=MagicMock(
                    return_value=MagicMock(
                        depth_map=MagicMock(shape=(100, 100)),
                        depth=MagicMock(shape=(100, 100)),
                        metadata={"cached": False},
                    )
                ),
            ),
        ):
            try:
                result = orchestrator_with_provenance.enhance_image(image_input)
            except Exception as e:
                # We expect this might fail due to synthetic backend,
                # but provenance should still be captured
                pytest.fail(f"Unexpected error during processing: {e}")

        # Verify provenance sidecar was created
        manifests_dir = orchestrator_with_provenance.manifests_dir
        provenance_files = list(manifests_dir.rglob("*_provenance.json"))

        assert len(provenance_files) > 0, "No provenance sidecar files found"

        # Verify sidecar content
        provenance_file = provenance_files[0]
        with open(provenance_file) as f:
            provenance_data = json.load(f)

        # Validate schema version
        assert provenance_data["schema_version"] == PROVENANCE_SCHEMA_VERSION

        # Validate required fields
        assert "input" in provenance_data
        assert provenance_data["input"]["file_path"] == str(fixture_path)
        assert provenance_data["input"]["file_sha256"]
        assert provenance_data["input"]["file_size_bytes"] > 0

        # Validate EXIF metadata
        assert "exif" in provenance_data
        assert isinstance(provenance_data["exif"], dict)

        # Validate toolchain
        assert "toolchain" in provenance_data
        assert provenance_data["toolchain"]["python_version"]
        assert provenance_data["toolchain"]["exiftool_version"]

        # Validate ingest context
        assert "ingest_context" in provenance_data
        assert provenance_data["ingest_context"]["config_fingerprint"]
        assert provenance_data["ingest_context"]["ingest_timestamp_utc"]
        assert provenance_data["ingest_context"]["host_os"]

    def test_provenance_hard_fails_without_exiftool(
        self,
        orchestrator_with_provenance: EnhanceOrchestrator,
        tmp_path: Path,
    ):
        """Test that processing hard-fails when exiftool is not available."""
        # Use real TIFF fixture
        fixture_path = Path(
            "tests/fixtures/pipelines/750_picacho_lane/input/750Picacho_Pool_UltraQuality.tif"
        )
        if not fixture_path.exists():
            pytest.skip("Real TIFF fixture not available")

        # Create ImageInput
        image_input = ImageInput(path=fixture_path)

        # Mock exiftool to not be available
        with patch(
            "transformation_portal.lux_depth_v3.provenance._check_exiftool_available",
            return_value=False,
        ):
            # Mock depth backend
            with patch.object(
                orchestrator_with_provenance,
                "depth_backend",
                MagicMock(
                    name="synthetic",
                    compute=MagicMock(
                        return_value=MagicMock(
                            depth_map=MagicMock(shape=(100, 100)),
                            depth=MagicMock(shape=(100, 100)),
                            metadata={"cached": False},
                        )
                    ),
                ),
            ):
                # Should raise RuntimeError about missing exiftool
                with pytest.raises(RuntimeError) as exc_info:
                    orchestrator_with_provenance.enhance_image(image_input)

                assert "exiftool" in str(exc_info.value).lower()
                assert "install" in str(exc_info.value).lower()

    def test_provenance_sidecar_deterministic(
        self,
        orchestrator_with_provenance: EnhanceOrchestrator,
        tmp_path: Path,
    ):
        """Test that repeated runs produce identical provenance (determinism)."""
        # Check if exiftool is available
        try:
            result = subprocess.run(
                ["exiftool", "-ver"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                pytest.skip("exiftool not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("exiftool not available")

        # Use real TIFF fixture
        fixture_path = Path(
            "tests/fixtures/pipelines/750_picacho_lane/input/750Picacho_GreatRoom_UltraQuality.tif"
        )
        if not fixture_path.exists():
            pytest.skip("Real TIFF fixture not available")

        # Create ImageInput
        image_input = ImageInput(path=fixture_path)

        # Process twice with same config
        provenance_files = []

        for run_num in range(2):
            # Create fresh orchestrator for each run
            output_dir = tmp_path / f"output_run{run_num}"
            orchestrator = EnhanceOrchestrator(
                config=orchestrator_with_provenance.config,
                output_dir=output_dir,
                enable_cache=False,
            )

            # Mock depth backend
            with patch.object(
                orchestrator,
                "depth_backend",
                MagicMock(
                    name="synthetic",
                    compute=MagicMock(
                        return_value=MagicMock(
                            depth_map=MagicMock(shape=(100, 100)),
                            depth=MagicMock(shape=(100, 100)),
                            metadata={"cached": False},
                        )
                    ),
                ),
            ):
                try:
                    orchestrator.enhance_image(image_input)
                except Exception:
                    pass  # Ignore processing errors

            # Find provenance file
            manifests_dir = orchestrator.manifests_dir
            prov_files = list(manifests_dir.rglob("*_provenance.json"))
            assert len(prov_files) > 0
            provenance_files.append(prov_files[0])

        # Load both provenance files
        with open(provenance_files[0]) as f:
            prov1 = json.load(f)

        with open(provenance_files[1]) as f:
            prov2 = json.load(f)

        # Deterministic fields should be identical
        assert prov1["input"]["file_sha256"] == prov2["input"]["file_sha256"]
        assert prov1["input"]["file_size_bytes"] == prov2["input"]["file_size_bytes"]
        assert prov1["input"]["file_path"] == prov2["input"]["file_path"]
        assert prov1["exif"] == prov2["exif"]  # EXIF should be identical
        assert prov1["toolchain"]["exiftool_version"] == prov2["toolchain"]["exiftool_version"]

        # Nondeterministic fields (timestamps) will differ - that's expected
        # But the structure should be identical
        assert set(prov1.keys()) == set(prov2.keys())
