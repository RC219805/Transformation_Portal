"""Tests for deprecation warnings and backward compatibility.

Ensures that:
1. Deprecated modules issue FutureWarning on import
2. Backward compatibility shims work correctly
3. Migration guide information is provided
4. Warning messages are clear and actionable
"""

import importlib
import sys
import warnings
import pytest


def assert_deprecation_warning(module_name: str, required_substrings: list) -> None:
    """Assert that a module issues a proper deprecation warning.

    Uses importlib.reload() to re-run module initialization even if already imported,
    ensuring the warning is always captured regardless of test order.

    Args:
        module_name: Full module path (e.g., "transformation_portal.depth")
        required_substrings: List of strings that must appear in warning message
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FutureWarning)

        # Reload if already imported, import if not
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)

    # Find deprecation warnings (don't assume it's the only warning)
    deprecation_warnings = [
        wrn for wrn in w
        if issubclass(wrn.category, FutureWarning)
        and "deprecated" in str(wrn.message).lower()
    ]

    assert deprecation_warnings, (
        f"No deprecation FutureWarning captured for {module_name}. "
        f"Captured warnings: {[str(x.message) for x in w]}"
    )

    # Check the warning message contains required information
    message = str(deprecation_warnings[0].message).lower()
    for substring in required_substrings:
        assert substring.lower() in message, (
            f"Expected '{substring}' in warning for {module_name}. "
            f"Got: {deprecation_warnings[0].message}"
        )


class TestDeprecationWarnings:
    """Test that deprecated modules issue proper warnings."""

    def test_depth_module_issues_deprecation_warning(self):
        """Test that importing from depth/ issues FutureWarning."""
        assert_deprecation_warning(
            "transformation_portal.depth",
            ["deprecated", "v2.0.0", "depth_canonical", "migration"]
        )

    def test_lux_depth_v3_module_issues_deprecation_warning(self):
        """Test that importing from lux_depth_v3/ issues FutureWarning."""
        assert_deprecation_warning(
            "transformation_portal.lux_depth_v3",
            ["deprecated", "v2.0.0", "depth_canonical"]
        )

    def test_depth_intelligence_module_issues_deprecation_warning(self):
        """Test that importing from depth_intelligence/ issues FutureWarning."""
        assert_deprecation_warning(
            "transformation_portal.depth_intelligence",
            ["deprecated", "v2.0.0"]
        )


class TestBackwardCompatibility:
    """Test that backward compatibility shims work correctly."""

    def test_depth_architectural_pipeline_shim_works(self):
        """Test that ArchitecturalDepthPipeline shim points to DepthPipeline."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warning for this test

            from transformation_portal.depth import ArchitecturalDepthPipeline
            from transformation_portal.depth_canonical import DepthPipeline

            # Should be the same class
            assert ArchitecturalDepthPipeline is DepthPipeline, (
                "ArchitecturalDepthPipeline should be a shim to DepthPipeline"
            )

    def test_depth_config_shim_works(self):
        """Test that DepthConfig shim points to UnifiedDepthConfig."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warning

            from transformation_portal.depth import DepthConfig
            from transformation_portal.depth_canonical import UnifiedDepthConfig

            # Should be the same class
            assert DepthConfig is UnifiedDepthConfig, (
                "DepthConfig should be a shim to UnifiedDepthConfig"
            )

    def test_generate_pbr_maps_shim_works(self):
        """Test that generate_pbr_maps shim from lux_depth_v3 works."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warning

            from transformation_portal.lux_depth_v3 import generate_pbr_maps as lux_pbr
            from transformation_portal.depth_canonical import generate_pbr_maps as canonical_pbr

            # Should be the same function
            assert lux_pbr is canonical_pbr, (
                "generate_pbr_maps should point to canonical implementation"
            )

    def test_old_imports_still_accessible(self):
        """Test that old imports are still accessible (not removed)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warnings

            # These should all work without errors
            from transformation_portal.depth import DepthAnythingV2Model, DepthCache
            from transformation_portal.lux_depth_v3 import EnhanceOrchestrator, EnhanceConfig
            # Note: depth_intelligence was never fully implemented, so skip its imports

            # Just verify they import successfully
            assert DepthAnythingV2Model is not None
            assert DepthCache is not None
            assert EnhanceOrchestrator is not None
            assert EnhanceConfig is not None


class TestWarningStackLevel:
    """Test that warnings appear at the correct stack level (caller's location)."""

    def test_warning_stacklevel_is_correct(self):
        """Test that warning appears at caller's location, not in module.

        Note: When using importlib.reload(), the stack includes importlib/__init__.py,
        but we verify that with a direct import, the warning points to the caller.
        """
        from pathlib import Path
        import tempfile
        import subprocess

        # Create a temporary test script to verify stacklevel in clean import
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import warnings
import sys
from pathlib import Path

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", FutureWarning)
    import transformation_portal.depth as mod

# Find deprecation warning
deprecation_warnings = [
    warning for warning in w
    if issubclass(warning.category, FutureWarning)
    and "deprecated" in str(warning.message).lower()
]

if not deprecation_warnings:
    print("FAIL: No deprecation warning found")
    sys.exit(1)

# Verify warning points to THIS script, not the module
warned_file = Path(deprecation_warnings[0].filename).resolve()
script_file = Path(__file__).resolve()
module_file = Path(mod.__file__).resolve()

# CRITICAL: Warning should point to caller (this script)
if warned_file == script_file:
    print(f"PASS: Warning correctly points to caller")
    sys.exit(0)
elif warned_file == module_file:
    print(f"FAIL: Warning points to module, not caller. stacklevel is wrong!")
    sys.exit(1)
else:
    print(f"FAIL: Warning points to unexpected location: {warned_file}")
    print(f"Expected: {script_file}")
    sys.exit(1)
""")
            temp_script = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )

            assert result.returncode == 0, (
                f"Stacklevel test failed in subprocess:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            assert "PASS" in result.stdout, f"Expected PASS in output, got: {result.stdout}"
        finally:
            Path(temp_script).unlink()

        # Test 2: Reload scenario - verify warning is issued (location may vary with reload)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", FutureWarning)

            if 'transformation_portal.depth' in sys.modules:
                importlib.reload(sys.modules['transformation_portal.depth'])
            else:
                import transformation_portal.depth

        # At minimum, verify warning exists during reload
        deprecation_warnings = [
            warning for warning in w
            if issubclass(warning.category, FutureWarning)
            and "deprecated" in str(warning.message).lower()
        ]
        assert len(deprecation_warnings) >= 1, "Should have deprecation warning on reload"


class TestMigrationScriptFunctionality:
    """Test the migration script can find deprecated imports."""

    def test_migration_script_exists(self):
        """Test that migration script exists."""
        from pathlib import Path

        script_path = Path("scripts/migrate_to_depth_canonical.py")
        assert script_path.exists(), "Migration script should exist"

    def test_migration_script_is_executable(self, tmp_path):
        """Test that migration script can detect deprecated imports."""
        from pathlib import Path
        import subprocess
        import sys

        # Create a test file with deprecated import
        test_file = tmp_path / "test_code.py"
        test_file.write_text(
            "from transformation_portal.depth import ArchitecturalDepthPipeline\n"
            "pipeline = ArchitecturalDepthPipeline()\n"
        )

        # Run migration script in scan mode
        script_path = Path("scripts/migrate_to_depth_canonical.py")
        result = subprocess.run(
            [sys.executable, str(script_path), "--scan", str(tmp_path)],
            capture_output=True,
            text=True,
        )

        # Should find the deprecated import (exit code 1)
        assert result.returncode == 1, "Should find deprecated imports"
        assert "deprecated" in result.stderr.lower() or "deprecated" in result.stdout.lower()


class TestDeprecationDocumentation:
    """Test that deprecation is properly documented."""

    def test_deprecation_notice_in_module_docstrings(self):
        """Test that deprecated modules have deprecation notices in docstrings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings

            import transformation_portal.depth as depth_mod
            import transformation_portal.lux_depth_v3 as lux_mod
            import transformation_portal.depth_intelligence as intel_mod

            # Check docstrings mention deprecation
            assert "DEPRECATED" in depth_mod.__doc__, "depth module should have DEPRECATED in docstring"
            assert "DEPRECATED" in lux_mod.__doc__, "lux_depth_v3 module should have DEPRECATED in docstring"
            assert "DEPRECATED" in intel_mod.__doc__, "depth_intelligence module should have DEPRECATED in docstring"

            # Check migration guide is mentioned
            assert "migration" in depth_mod.__doc__.lower(), "depth should mention migration guide"
            assert "migration" in lux_mod.__doc__.lower(), "lux_depth_v3 should mention migration guide"
            assert "migration" in intel_mod.__doc__.lower(), "depth_intelligence should mention migration guide"

    def test_deprecation_timeline_documented(self):
        """Test that deprecation timeline is documented in module docstrings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings

            import transformation_portal.depth as depth_mod

            # Check timeline is mentioned
            docstring = depth_mod.__doc__
            assert "v1.8.0" in docstring or "Feb 2026" in docstring, "Should mention when deprecation started"
            assert "v2.0.0" in docstring or "Aug 2026" in docstring, "Should mention when module will be removed"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
