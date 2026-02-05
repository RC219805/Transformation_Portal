"""Tests for V2 script validation and fail-fast behavior (Issue #1).

This test suite validates:
1. Fail-fast validation when V2 script is missing
2. V2 stage can be disabled (enable_v2=False)
3. V2 stage can be skipped (v2_preset=None)
4. Clear error messages guide users to solutions
5. PBR-only workflows function without V2 script

Coverage target: Issue #1 from PBR Implementation Audit
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator


@pytest.fixture
def temp_output():
    """Create temporary output directory."""
    tmpdir = tempfile.mkdtemp(prefix="test_orchestrator_v2_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestV2ValidationFailFast:
    """Test V2 script validation and fail-fast behavior."""

    def test_v2_enabled_script_missing_raises_error(self, temp_output):
        """Test fail-fast when V2 enabled but script missing."""
        config = EnhanceConfig(enable_v2=True, v2_preset="default", depth_device="cpu")

        # Simulate missing script (since scripts/enhance_image.py now exists)
        original_exists = Path.exists

        def mock_exists(self):
            if "enhance_image.py" in str(self):
                return False
            return original_exists(self)

        with patch("pathlib.Path.exists", mock_exists):
            # Should raise FileNotFoundError during initialization
            with pytest.raises(FileNotFoundError) as exc_info:
                EnhanceOrchestrator(config=config, output_root=temp_output)

            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "scripts/enhance_image.py" in error_msg
            assert "enable_v2=False" in error_msg or "v2_preset=None" in error_msg
            assert "PBR-only" in error_msg

    def test_v2_disabled_no_error_when_script_missing(self, temp_output):
        """Test V2 disabled allows initialization without script."""
        config = EnhanceConfig(enable_v2=False, v2_preset="default", depth_device="cpu")  # Ignored when enable_v2=False

        # Should succeed even without V2 script
        orchestrator = EnhanceOrchestrator(config=config, output_root=temp_output)

        # Verify V2 runner is None
        assert orchestrator.v2_runner is None

    def test_v2_preset_none_no_error_when_script_missing(self, temp_output):
        """Test v2_preset=None allows initialization without script."""
        config = EnhanceConfig(enable_v2=True, v2_preset=None, depth_device="cpu")  # None = skip V2

        # Should succeed even without V2 script
        orchestrator = EnhanceOrchestrator(config=config, output_root=temp_output)

        # Verify V2 runner is None
        assert orchestrator.v2_runner is None

    def test_v2_disabled_overrides_preset(self, temp_output):
        """Test enable_v2=False overrides v2_preset."""
        config = EnhanceConfig(enable_v2=False, v2_preset="premium", depth_device="cpu")  # Should be ignored

        # Should succeed - enable_v2 takes precedence
        orchestrator = EnhanceOrchestrator(config=config, output_root=temp_output)
        assert orchestrator.v2_runner is None


class TestV2ConfigCombinations:
    """Test various V2 configuration combinations."""

    def test_default_config_requires_v2_script(self, temp_output):
        """Test default config expects V2 script to exist."""
        config = EnhanceConfig()  # Defaults: enable_v2=True, v2_preset="default"

        # Simulate missing script
        original_exists = Path.exists

        def mock_exists(self):
            if "enhance_image.py" in str(self):
                return False
            return original_exists(self)

        with patch("pathlib.Path.exists", mock_exists):
            # Should fail with helpful error
            with pytest.raises(FileNotFoundError) as exc_info:
                EnhanceOrchestrator(config=config, output_root=temp_output)

            assert "enhance_image.py" in str(exc_info.value)

    def test_pbr_only_config_no_v2_required(self, temp_output):
        """Test PBR-only config doesn't require V2 script."""
        config = EnhanceConfig(enable_v2=False, generate_pbr=True, pbr_normal_strength=1.5, depth_device="cpu")

        # Should succeed
        orchestrator = EnhanceOrchestrator(config=config, output_root=temp_output)
        assert orchestrator.v2_runner is None
        assert orchestrator.config.generate_pbr is True


class TestV2ErrorMessages:
    """Test error message quality and actionability."""

    def test_error_message_contains_solutions(self, temp_output):
        """Test error message provides actionable solutions."""
        config = EnhanceConfig(enable_v2=True, v2_preset="default")

        # Simulate missing script
        original_exists = Path.exists

        def mock_exists(self):
            if "enhance_image.py" in str(self):
                return False
            return original_exists(self)

        with patch("pathlib.Path.exists", mock_exists):
            with pytest.raises(FileNotFoundError) as exc_info:
                EnhanceOrchestrator(config=config, output_root=temp_output)

            error_msg = str(exc_info.value)

            # Should mention expected script location
            assert "scripts/enhance_image.py" in error_msg

            # Should provide at least 2 solutions
            solution_count = 0
            if "enable_v2=False" in error_msg or "Set enable_v2=False" in error_msg:
                solution_count += 1
            if "v2_preset=None" in error_msg or "Set v2_preset=None" in error_msg:
                solution_count += 1
        if "Create" in error_msg or "create" in error_msg:
            solution_count += 1

        assert solution_count >= 2, "Error message should provide multiple solutions"

    def test_error_message_mentions_pbr_only_workflow(self, temp_output):
        """Test error message mentions PBR-only workflow option."""
        config = EnhanceConfig(enable_v2=True, v2_preset="default")

        with pytest.raises(FileNotFoundError) as exc_info:
            EnhanceOrchestrator(config=config, output_root=temp_output)

        error_msg = str(exc_info.value).lower()
        assert "pbr" in error_msg or "workflow" in error_msg


class TestV2ConfigMigration:
    """Test backward compatibility and migration paths."""

    def test_old_style_v2_preset_string_still_works(self, temp_output):
        """Test backward compatibility with string v2_preset."""
        # Old style: v2_preset is always a string, enable_v2 controls behavior
        config = EnhanceConfig(
            enable_v2=False,
            v2_preset="default",  # Old code would set this
        )

        # Should work with enable_v2=False
        orchestrator = EnhanceOrchestrator(config=config, output_root=temp_output)
        assert orchestrator.v2_runner is None

    def test_new_style_v2_preset_optional(self, temp_output):
        """Test new style with Optional[str] v2_preset."""
        # New style: v2_preset=None explicitly skips V2
        config = EnhanceConfig(
            enable_v2=True,
            v2_preset=None,  # Explicitly skip V2
        )

        # Should work
        orchestrator = EnhanceOrchestrator(config=config, output_root=temp_output)
        assert orchestrator.v2_runner is None
