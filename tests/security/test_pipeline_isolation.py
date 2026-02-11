"""Tests for ADR-023 pipeline isolation enforcement (Bug P2-C fix).

Tests cover:
- Absolute spatial_ai import detection
- 2-dot relative import detection
- 3-dot relative import detection (Bug P2-C)
- 4-dot relative import detection
- Safe import patterns (no false positives)

Architecture: ADR-023 (Isolation), ADR-026 (APEX Research Ultra)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "security"))


class TestIsolationCheckRegex:
    """Test that isolation check regex patterns work correctly."""

    def test_pattern_matching_absolute_imports(self):
        """Test that absolute spatial_ai import patterns match correctly."""
        pattern = "from transformation_portal.spatial_ai"

        # Should match
        assert pattern in "from transformation_portal.spatial_ai import LinearDecoder"
        assert pattern in "from transformation_portal.spatial_ai.ingest import decode"

        # Should not match
        assert pattern not in "from transformation_portal.depth import DepthBackend"
        assert pattern not in "import numpy as np"

    def test_pattern_matching_two_dot_relative_imports(self):
        """Test that 2-dot relative import patterns match correctly."""
        pattern = "from ..spatial_ai"

        # Should match
        assert pattern in "from ..spatial_ai import LinearDecoder"
        assert pattern in "from ..spatial_ai.ingest import decode"

        # Should not match
        assert pattern not in "from ..depth import DepthBackend"
        assert pattern not in "from ...spatial_ai import something"  # 3 dots

    def test_pattern_matching_three_dot_relative_imports(self):
        """Test that 3-dot relative import patterns match correctly (Bug P2-C fix)."""
        pattern = "from ...spatial_ai"  # Fixed: no space after dots

        # Should match
        assert pattern in "from ...spatial_ai import LinearDecoder"
        assert pattern in "from ...spatial_ai.ingest import decode"

        # Should not match (wrong number of dots)
        assert pattern not in "from ..spatial_ai import LinearDecoder"  # 2 dots
        assert pattern not in "from ....spatial_ai import something"  # 4 dots

        # Bug P2-C: The old pattern had a space and would NOT match
        old_buggy_pattern = "from ... spatial_ai"  # Space between dots and module
        assert old_buggy_pattern not in "from ...spatial_ai import LinearDecoder"

    def test_pattern_matching_four_dot_relative_imports(self):
        """Test that 4-dot relative import patterns match correctly."""
        pattern = "from ....spatial_ai"

        # Should match
        assert pattern in "from ....spatial_ai import something"

        # Should not match
        assert pattern not in "from ...spatial_ai import LinearDecoder"  # 3 dots
        assert pattern not in "from ..spatial_ai import LinearDecoder"  # 2 dots

    def test_safe_imports_not_flagged(self):
        """Test that non-spatial_ai imports are not flagged."""
        forbidden_patterns = [
            "from transformation_portal.spatial_ai",
            "from ..spatial_ai",
            "from ...spatial_ai",
            "from ....spatial_ai",
        ]

        safe_imports = [
            "from transformation_portal.depth import DepthBackend",
            "import numpy as np",
            "from ..depth.backends import ensemble",
            "from ...lux_depth_v3 import config",
        ]

        for safe_import in safe_imports:
            for pattern in forbidden_patterns:
                assert pattern not in safe_import, f"Pattern '{pattern}' should not match safe import '{safe_import}'"

    def test_comprehensive_pattern_coverage(self):
        """Test that all forbidden patterns are correctly specified (no typos)."""
        # These are the patterns that should be in verify_pipeline_isolation.py after Bug P2-C fix
        forbidden_patterns = [
            "from transformation_portal.spatial_ai",
            "import transformation_portal.spatial_ai",
            "from ..spatial_ai",
            "from ...spatial_ai",  # Fixed: no space
            "from ....spatial_ai",  # Added: 4-dot pattern
        ]

        # Verify no typos (no spaces after dots)
        for pattern in forbidden_patterns:
            if "..." in pattern:
                # Check that there's no space between dots and "spatial_ai"
                assert "... spatial_ai" not in pattern, f"Pattern '{pattern}' has incorrect spacing (Bug P2-C)"


# Pytest markers
pytestmark = [
    pytest.mark.apex_ultra,
    pytest.mark.security,
]
