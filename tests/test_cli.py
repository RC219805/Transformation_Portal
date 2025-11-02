"""Tests for CLI module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import CLI module
sys.path.insert(0, str(ROOT / "src"))


class TestCLIImport:
    """Tests for CLI module import."""

    def test_cli_module_imports(self):
        """Test that CLI module can be imported."""
        from transformation_portal.cli import (
            render_cli,
            process_cli,
            analyze_cli,
            main,
        )
        
        # Verify functions are callable
        assert callable(render_cli)
        assert callable(process_cli)
        assert callable(analyze_cli)
        assert callable(main)

    def test_cli_apps_exist(self):
        """Test that CLI apps are defined."""
        from transformation_portal.cli import (
            app,
            render_app,
            process_app,
            analyze_app,
        )
        
        # Verify apps are typer instances
        assert app is not None
        assert render_app is not None
        assert process_app is not None
        assert analyze_app is not None

    def test_cli_exports(self):
        """Test that CLI module exports expected symbols."""
        from transformation_portal import cli
        
        # Check __all__ exports
        assert hasattr(cli, '__all__')
        expected_exports = [
            'app',
            'render_app',
            'process_app',
            'analyze_app',
            'render_cli',
            'process_cli',
            'analyze_cli',
            'main',
            'version',
            'info',
        ]
        
        for export in expected_exports:
            assert export in cli.__all__, f"Missing export: {export}"


class TestCLIFunctions:
    """Tests for CLI functions."""

    def test_render_cli_callable(self):
        """Test that render_cli is callable."""
        from transformation_portal.cli import render_cli
        assert callable(render_cli)

    def test_process_cli_callable(self):
        """Test that process_cli is callable."""
        from transformation_portal.cli import process_cli
        assert callable(process_cli)

    def test_analyze_cli_callable(self):
        """Test that analyze_cli is callable."""
        from transformation_portal.cli import analyze_cli
        assert callable(analyze_cli)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
