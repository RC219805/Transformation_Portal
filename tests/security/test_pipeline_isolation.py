"""Tests for ADR-023 pipeline isolation enforcement (AST-based).

Tests cover:
- Absolute spatial_ai import detection via AST
- Relative import detection (2-dot, 3-dot, 4-dot)
- No false positives on docstrings/comments
- Import statement line number reporting

Architecture: ADR-023 (Isolation), ADR-026 (APEX Research Ultra), Phase 1.1 (Item 4)
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "security"))

from verify_pipeline_isolation import check_imports_ast  # noqa: E402


class TestIsolationCheckAST:
    """Test that AST-based isolation check correctly detects real imports."""

    def test_detects_absolute_imports(self, tmp_path: Path):
        """Test that absolute spatial_ai imports are detected."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from transformation_portal.spatial_ai import LinearDecoder\n")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 1
        assert "spatial_ai" in violations[0]
        assert ":1:" in violations[0]  # Line number

    def test_detects_two_dot_relative_imports(self, tmp_path: Path):
        """Test that 2-dot relative imports are detected."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from ..spatial_ai import LinearDecoder\n")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 1
        assert "spatial_ai" in violations[0]

    def test_detects_three_dot_relative_imports(self, tmp_path: Path):
        """Test that 3-dot relative imports are detected (Bug P2-C context)."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from ...spatial_ai import LinearDecoder\n")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 1
        assert "spatial_ai" in violations[0]

    def test_detects_four_dot_relative_imports(self, tmp_path: Path):
        """Test that 4-dot relative imports are detected."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from ....spatial_ai import something\n")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 1
        assert "spatial_ai" in violations[0]

    def test_detects_direct_import_statements(self, tmp_path: Path):
        """Test that direct 'import X' statements are detected."""
        test_file = tmp_path / "test.py"
        test_file.write_text("import transformation_portal.spatial_ai\n")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 1
        assert "spatial_ai" in violations[0]

    def test_ignores_docstring_mentions(self, tmp_path: Path):
        """Test that docstring mentions of spatial_ai are ignored (AST precision)."""
        test_file = tmp_path / "test.py"
        test_file.write_text('''"""This module does NOT import spatial_ai.

WARNING: Do not import from ...spatial_ai here.
"""
import numpy as np
''')

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 0, "Docstring mentions should not be flagged"

    def test_ignores_comment_mentions(self, tmp_path: Path):
        """Test that comment mentions of spatial_ai are ignored."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""# TODO: Consider using spatial_ai for this
# from ...spatial_ai import LinearDecoder  # Commented out
import numpy as np
""")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 0, "Comment mentions should not be flagged"

    def test_safe_imports_not_flagged(self, tmp_path: Path):
        """Test that non-spatial_ai imports are not flagged."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""from transformation_portal.depth import DepthBackend
import numpy as np
from ..depth.backends import ensemble
from ...lux_depth_v3 import config
""")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 0, "Safe imports should not be flagged"

    def test_detects_submodule_imports(self, tmp_path: Path):
        """Test that submodule imports like lux_depth_v3.raw_loader are detected."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from transformation_portal.lux_depth_v3.raw_loader import load_raw\n")

        violations = check_imports_ast(test_file, ["lux_depth_v3.raw_loader"])
        assert len(violations) == 1
        assert "raw_loader" in violations[0]

    def test_reports_line_numbers(self, tmp_path: Path):
        """Test that violations include line numbers for easy debugging."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""import numpy as np

from transformation_portal.spatial_ai import LinearDecoder

import pandas as pd
""")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 1
        assert ":3:" in violations[0], "Should report line 3"

    def test_handles_multiple_violations(self, tmp_path: Path):
        """Test that multiple violations in same file are all reported."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""from ..spatial_ai import LinearDecoder
from ...spatial_ai.ingest import decode
""")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) == 2
        assert any(":1:" in v for v in violations)
        assert any(":2:" in v for v in violations)

    def test_handles_syntax_errors_gracefully(self, tmp_path: Path):
        """Test that syntax errors are reported as violations."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from ..spatial_ai import ( # unclosed paren\n")

        violations = check_imports_ast(test_file, ["spatial_ai"])
        assert len(violations) > 0
        assert any("SyntaxError" in v for v in violations)


# Pytest markers (using registered markers from pyproject.toml)
pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
]
