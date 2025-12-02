#!/usr/bin/env python3
"""Tests for the auto_resolver security module.

Tests cover:
- Command injection prevention via package name validation
- Pattern matching and confidence scoring
- Command generation with proper sanitization
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path for importing
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


class TestPackageNameValidation:
    """Tests for package name validation to prevent command injection."""

    def test_valid_simple_package_name(self):
        """Test that simple package names are accepted."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("numpy") is True
        assert AutoFixer._validate_package_name("Pillow") is True
        assert AutoFixer._validate_package_name("pytest") is True

    def test_valid_package_with_hyphen(self):
        """Test that packages with hyphens are accepted."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("scikit-learn") is True
        assert AutoFixer._validate_package_name("my-package") is True

    def test_valid_package_with_underscore(self):
        """Test that packages with underscores are accepted."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("my_package") is True
        assert AutoFixer._validate_package_name("typing_extensions") is True

    def test_valid_package_with_dots(self):
        """Test that packages with dots are accepted."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("zope.interface") is True

    def test_valid_package_with_numbers(self):
        """Test that packages with numbers are accepted."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("oauth2") is True
        assert AutoFixer._validate_package_name("py3dns") is True

    def test_reject_command_injection_semicolon(self):
        """Test that command injection via semicolon is rejected."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("numpy; rm -rf /") is False

    def test_reject_command_injection_pipe(self):
        """Test that command injection via pipe is rejected."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("numpy | cat /etc/passwd") is False

    def test_reject_command_injection_backticks(self):
        """Test that command injection via backticks is rejected."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("`whoami`") is False

    def test_reject_command_injection_subshell(self):
        """Test that command injection via $() is rejected."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("$(whoami)") is False

    def test_reject_empty_package_name(self):
        """Test that empty package names are rejected."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("") is False

    def test_reject_too_long_package_name(self):
        """Test that excessively long package names are rejected."""
        from security.auto_resolver import AutoFixer

        long_name = "a" * 101
        assert AutoFixer._validate_package_name(long_name) is False

    def test_reject_package_starting_with_hyphen(self):
        """Test that package names starting with hyphen are rejected."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("-malicious") is False

    def test_reject_package_ending_with_hyphen(self):
        """Test that package names ending with hyphen are rejected."""
        from security.auto_resolver import AutoFixer

        assert AutoFixer._validate_package_name("malicious-") is False


class TestResolutionStrategy:
    """Tests for resolution strategy enum."""

    def test_resolution_strategies_exist(self):
        """Test that all expected resolution strategies are defined."""
        from security.auto_resolver import ResolutionStrategy

        assert hasattr(ResolutionStrategy, 'CONSTRAINT_BLOCK')
        assert hasattr(ResolutionStrategy, 'VENDOR_REPLACE')
        assert hasattr(ResolutionStrategy, 'UPGRADE')
        assert hasattr(ResolutionStrategy, 'REMOVE')
        assert hasattr(ResolutionStrategy, 'WORKAROUND')


class TestPatternLearner:
    """Tests for pattern learning functionality."""

    def test_builtin_patterns_exist(self, tmp_path):
        """Test that built-in patterns are defined."""
        from security.auto_resolver import PatternLearner

        learner = PatternLearner(tmp_path)

        # Check that patterns are loaded
        assert len(learner.patterns) > 0

    def test_basicsr_pattern_included(self, tmp_path):
        """Test that the basicsr CVE pattern is included."""
        from security.auto_resolver import PatternLearner

        learner = PatternLearner(tmp_path)

        # Find the basicsr pattern
        basicsr_patterns = [
            p for p in learner.patterns
            if p.pattern_id == "basicsr_cve_2024_27763"
        ]

        assert len(basicsr_patterns) == 1
        assert basicsr_patterns[0].confidence_base == 0.95


class TestConfidenceScorer:
    """Tests for confidence scoring."""

    def test_confidence_level_enum(self):
        """Test that confidence levels are properly defined."""
        from security.auto_resolver import ConfidenceLevel

        assert hasattr(ConfidenceLevel, 'HIGH')
        assert hasattr(ConfidenceLevel, 'MEDIUM')
        assert hasattr(ConfidenceLevel, 'LOW')
        assert hasattr(ConfidenceLevel, 'UNCERTAIN')


class TestResolutionPattern:
    """Tests for resolution pattern dataclass."""

    def test_pattern_creation(self):
        """Test that resolution patterns can be created."""
        from security.auto_resolver import ResolutionPattern, ResolutionStrategy

        pattern = ResolutionPattern(
            pattern_id="test_pattern",
            vulnerability_type="test",
            package_pattern=r"^test$",
            strategy=ResolutionStrategy.UPGRADE,
            confidence_base=0.8,
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.confidence_base == 0.8


class TestAutoFixer:
    """Tests for auto fixer functionality."""

    def test_autofixer_creation(self, tmp_path):
        """Test that auto fixer can be created."""
        from security.auto_resolver import AutoFixer

        fixer = AutoFixer(tmp_path)
        assert fixer is not None

    def test_invalid_package_rejected_in_generate_commands(self, tmp_path):
        """Test that invalid packages are rejected in command generation."""
        from security.auto_resolver import (
            AutoFixer, ResolutionPattern, ResolutionStrategy
        )

        fixer = AutoFixer(tmp_path)

        pattern = ResolutionPattern(
            pattern_id="test",
            vulnerability_type="test",
            package_pattern=r".*",
            strategy=ResolutionStrategy.UPGRADE,
            confidence_base=0.8,
        )

        # Test with injection attempt
        commands = fixer._generate_commands(pattern, "numpy; rm -rf /")
        assert commands == []  # Should return empty list for invalid package
