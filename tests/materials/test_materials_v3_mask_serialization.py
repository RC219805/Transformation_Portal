"""Integration tests for Materials V3 mask serialization and V2 integration.

Tests the complete flow:
1. Orchestrator computes material masks via Materials V3
2. Masks serialized to NPZ file in the requested output directory
3. V2 runner passes explicit masks_file path to subprocess
4. enhance_image.py loads and uses masks
5. Temporary masks cleaned up after V2 completes
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory structure."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    return output_root


@pytest.fixture
def mock_depth_backend():
    """Mock depth backend to avoid ML dependencies."""
    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
        yield


@pytest.fixture
def mock_da3_available():
    """Mock DA3Backend.ensure_available() to succeed in offline CI."""
    with patch("transformation_portal.depth.backends.da3.DA3Backend.ensure_available"):
        yield


class TestMaskSerialization:
    """Test mask serialization helper function."""

    def test_serialize_empty_masks_returns_none(self, temp_output_dir, mock_depth_backend, mock_da3_available):
        """Empty masks dictionary should return None."""
        config = EnhanceConfig(enable_v2=False, depth_device="cpu")
        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        output_key = Path("test_image_abc123")
        temp_dir = temp_output_dir / "temp"

        result = orchestrator._serialize_material_masks({}, output_key, temp_dir)
        assert result is None

    def test_serialize_valid_masks(self, temp_output_dir, mock_depth_backend, mock_da3_available):
        """Valid masks should be serialized to NPZ file."""
        config = EnhanceConfig(enable_v2=False, depth_device="cpu")
        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        # Create test masks
        masks = {
            "glass": np.random.rand(64, 64).astype(np.float32),
            "water": np.random.rand(64, 64).astype(np.float32),
        }

        output_key = Path("test_image_abc123")
        temp_dir = temp_output_dir / "temp"

        # Serialize
        mask_path = orchestrator._serialize_material_masks(masks, output_key, temp_dir)

        # Verify file created
        assert mask_path is not None
        assert mask_path.exists()
        assert mask_path.suffix == ".npz"
        assert "test_image_abc123" in mask_path.stem

        # Verify content
        with np.load(mask_path) as data:
            loaded_masks = {key: data[key] for key in data.files}

        assert set(loaded_masks.keys()) == {"glass", "water"}
        assert np.array_equal(loaded_masks["glass"], masks["glass"])
        assert np.array_equal(loaded_masks["water"], masks["water"])

    def test_serialize_orders_npz_entries_deterministically(self, temp_output_dir, mock_depth_backend, mock_da3_available):
        """NPZ entry ordering should be stable regardless of input dict insertion order."""
        config = EnhanceConfig(enable_v2=False, depth_device="cpu")
        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        masks = {
            "water": np.ones((4, 4), dtype=np.float32),
            "glass": np.zeros((4, 4), dtype=np.float32),
        }

        output_key = Path("test_image_abc123")
        output_dir = temp_output_dir / "temp"
        mask_path = orchestrator._serialize_material_masks(masks, output_key, output_dir)

        assert mask_path is not None
        with np.load(mask_path) as data:
            assert data.files == ["glass", "water"]

    def test_serialize_invalid_dtype_returns_none(self, temp_output_dir, mock_depth_backend, mock_da3_available):
        """Invalid mask dtype should return None with warning."""
        config = EnhanceConfig(enable_v2=False, depth_device="cpu")
        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        # Create mask with invalid dtype
        masks = {"glass": np.ones((64, 64), dtype=np.int32)}  # Invalid: should be float

        output_key = Path("test_image_abc123")
        temp_dir = temp_output_dir / "temp"

        result = orchestrator._serialize_material_masks(masks, output_key, temp_dir)
        assert result is None

    def test_serialize_invalid_shape_returns_none(self, temp_output_dir, mock_depth_backend, mock_da3_available):
        """Invalid mask shape should return None with warning."""
        config = EnhanceConfig(enable_v2=False, depth_device="cpu")
        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        # Create mask with invalid shape (3D instead of 2D)
        masks = {"glass": np.ones((64, 64, 3), dtype=np.float32)}

        output_key = Path("test_image_abc123")
        temp_dir = temp_output_dir / "temp"

        result = orchestrator._serialize_material_masks(masks, output_key, temp_dir)
        assert result is None

    def test_serialize_oversized_file_returns_none(self, temp_output_dir, mock_depth_backend, mock_da3_available, monkeypatch):
        """Oversized mask file should be rejected and cleaned up."""
        config = EnhanceConfig(enable_v2=False, depth_device="cpu")
        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        # Mock stat() to report oversized file
        original_stat = Path.stat

        def mock_stat(self, *, follow_symlinks=True):
            if self.suffix == ".npz" and self.name.startswith(".tmp_"):
                # Report oversized temp file (150MB)
                result = type("obj", (object,), {"st_size": 150 * 1024 * 1024})()
                return result
            return original_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(Path, "stat", mock_stat)

        # Create small test masks (won't actually be oversized)
        masks = {"glass": np.ones((64, 64), dtype=np.float32)}

        output_key = Path("test_image_abc123")
        temp_dir = temp_output_dir / "temp"

        # Should return None and clean up
        result = orchestrator._serialize_material_masks(masks, output_key, temp_dir)
        assert result is None

        # Verify no .npz or temp files left behind
        if temp_dir.exists():
            remaining_files = list(temp_dir.glob("*.npz")) + list(temp_dir.glob(".tmp_*"))
            assert len(remaining_files) == 0, f"Cleanup failed: {remaining_files}"


class TestV2RunnerMaskIntegration:
    """Test V2 runner with masks_file parameter."""

    def test_runner_accepts_masks_file(self):
        """V2Runner.run should accept masks_file parameter."""
        from transformation_portal.lux_depth_v3.v2_runner import V2Runner

        runner = V2Runner()

        # Check that run method signature includes masks_file
        import inspect

        sig = inspect.signature(runner.run)
        assert "masks_file" in sig.parameters

    def test_runner_builds_command_with_masks_file(self, tmp_path, monkeypatch):
        """V2Runner should add --masks-file to command when masks_file provided."""
        from transformation_portal.lux_depth_v3.v2_runner import V2Runner

        # Create a mock script file
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        script_path = repo_root / "scripts" / "enhance_image.py"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/usr/bin/env python3\nprint('mock')")

        # Mock subprocess.run to capture command
        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.append(cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", mock_run)

        # Create runner and run with masks_file
        runner = V2Runner()
        runner.repo_root = repo_root
        runner.script_path = script_path

        input_path = tmp_path / "input.jpg"
        input_path.write_text("mock")
        output_dir = tmp_path / "output"
        masks_file = tmp_path / "temp" / "test_masks.npz"

        runner.run(input_path=input_path, depth_dir=None, output_dir=output_dir, preset="default", masks_file=masks_file)

        # Verify --masks-file in command
        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        assert "--masks-file" in cmd
        masks_file_idx = cmd.index("--masks-file")
        assert str(masks_file) == cmd[masks_file_idx + 1]

    def test_runner_omits_masks_file_when_none(self, tmp_path, monkeypatch):
        """V2Runner should omit --masks-file when masks_file is None."""
        from transformation_portal.lux_depth_v3.v2_runner import V2Runner

        # Create a mock script file
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        script_path = repo_root / "scripts" / "enhance_image.py"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/usr/bin/env python3\nprint('mock')")

        # Mock subprocess.run to capture command
        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.append(cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", mock_run)

        # Create runner and run without masks_file
        runner = V2Runner()
        runner.repo_root = repo_root
        runner.script_path = script_path

        input_path = tmp_path / "input.jpg"
        input_path.write_text("mock")
        output_dir = tmp_path / "output"

        runner.run(input_path=input_path, depth_dir=None, output_dir=output_dir, preset="default", masks_file=None)

        # Verify --masks-file NOT in command
        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        assert "--masks-file" not in cmd


class TestCleanupBehavior:
    """Test cleanup of temporary mask files."""

    def test_cleanup_on_success(self, temp_output_dir, mock_depth_backend, mock_da3_available, monkeypatch):
        """Temporary masks should be cleaned up after successful V2 run."""
        config = EnhanceConfig(
            enable_v2=True,
            v2_preset="default",
            enable_materials_v3=True,
            enable_material_segmentation=True,
            material_segmentation_backend="stub",
            depth_device="cpu",
        )

        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        # Create test masks
        masks = {"glass": np.random.rand(64, 64).astype(np.float32)}
        output_key = Path("test_image_abc123")
        temp_dir = temp_output_dir / "temp"

        # Serialize masks
        mask_path = orchestrator._serialize_material_masks(masks, output_key, temp_dir)
        assert mask_path.exists()

        # Mock V2 runner to succeed
        with patch.object(orchestrator.v2_runner, "run", return_value={"status": "success", "runtime_s": 1.0}):
            from transformation_portal.lux_depth_v3.input_manager import ImageInput

            # Create ImageInput
            test_input_path = temp_output_dir / "test.jpg"
            test_input_path.write_text("mock")
            image_input = ImageInput(path=test_input_path, metadata={"sha256": "abc123"})

            orchestrator._run_v2_stage(
                image_input=image_input,
                depth_path=None,
                output_key=output_key,
                v2_log_path=temp_output_dir / "logs" / "test.log",
                manifest_path=temp_output_dir / "manifests" / "test.json",
                skip_depth=False,
                materials_v3_result={"material_masks": masks},
            )

        # Verify cleanup happened after success
        assert not mask_path.exists(), "Temporary masks should be cleaned up after V2 success"

    def test_cleanup_on_failure(self, temp_output_dir, mock_depth_backend, mock_da3_available):
        """Temporary masks should be cleaned up even if V2 subprocess fails."""
        config = EnhanceConfig(
            enable_v2=True,
            v2_preset="default",
            enable_materials_v3=True,
            enable_material_segmentation=True,
            material_segmentation_backend="stub",
            depth_device="cpu",
        )

        orchestrator = EnhanceOrchestrator(config, temp_output_dir)

        # Create test masks
        masks = {"glass": np.random.rand(64, 64).astype(np.float32)}
        output_key = Path("test_image_abc123")
        temp_dir = temp_output_dir / "temp"

        # Serialize masks
        mask_path = orchestrator._serialize_material_masks(masks, output_key, temp_dir)
        assert mask_path.exists()

        # Simulate V2 failure by mocking V2 runner
        with patch.object(orchestrator.v2_runner, "run", side_effect=RuntimeError("V2 failed")):
            from transformation_portal.lux_depth_v3.input_manager import ImageInput

            # Create ImageInput with correct signature (path and optional metadata)
            test_input_path = temp_output_dir / "test.jpg"
            test_input_path.write_text("mock")
            image_input = ImageInput(path=test_input_path, metadata={"sha256": "abc123"})

            try:
                orchestrator._run_v2_stage(
                    image_input=image_input,
                    depth_path=None,
                    output_key=output_key,
                    v2_log_path=temp_output_dir / "logs" / "test.log",
                    manifest_path=temp_output_dir / "manifests" / "test.json",
                    skip_depth=False,
                    materials_v3_result={"material_masks": masks},
                )
            except RuntimeError:
                pass  # Expected

        # Verify cleanup happened despite failure
        assert not mask_path.exists(), "Temporary masks should be cleaned up even on V2 failure"


class TestBackwardCompatibility:
    """Test backward compatibility when masks are not provided."""

    def test_v2_runner_works_without_masks(self, tmp_path, monkeypatch):
        """V2 runner should work normally when masks_file is None (backward compatibility)."""
        from transformation_portal.lux_depth_v3.v2_runner import V2Runner

        # Create a mock script file
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        script_path = repo_root / "scripts" / "enhance_image.py"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/usr/bin/env python3\nprint('mock')")

        # Mock subprocess.run
        def mock_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", mock_run)

        # Create runner and run without any masks
        runner = V2Runner()
        runner.repo_root = repo_root
        runner.script_path = script_path

        input_path = tmp_path / "input.jpg"
        input_path.write_text("mock")
        output_dir = tmp_path / "output"

        # Should not raise
        result = runner.run(input_path=input_path, depth_dir=None, output_dir=output_dir, preset="default")

        assert result["status"] == "success"

    def test_orchestrator_works_without_materials_v3(self, temp_output_dir, mock_depth_backend, mock_da3_available):
        """Orchestrator should work normally when Materials V3 is disabled."""
        config = EnhanceConfig(
            enable_materials_v3=False,  # Materials V3 disabled
            enable_v2=False,
            depth_device="cpu",
        )

        # Should initialize without errors
        orchestrator = EnhanceOrchestrator(config, temp_output_dir)
        assert orchestrator.materials_v3_engine is None
