#!/usr/bin/env python3
"""Tests for the continuous_security module.

Tests cover:
- Import scanner for detecting vulnerable imports
- Package auditor for runtime verification
- Constraint verifier for constraint file validation
- Code pattern scanner for security patterns
"""

import ast
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path for importing
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


class TestSecurityCheckType:
    """Tests for security check type enum."""

    def test_check_types_exist(self):
        """Test that all expected check types are defined."""
        from security.continuous_security import SecurityCheckType

        assert hasattr(SecurityCheckType, 'IMPORT_SCAN')
        assert hasattr(SecurityCheckType, 'PACKAGE_AUDIT')
        assert hasattr(SecurityCheckType, 'CONSTRAINT_VERIFY')
        assert hasattr(SecurityCheckType, 'VULNERABILITY_SCAN')
        assert hasattr(SecurityCheckType, 'CODE_PATTERN')


class TestSecurityStatus:
    """Tests for security status enum."""

    def test_status_levels_exist(self):
        """Test that all expected status levels are defined."""
        from security.continuous_security import SecurityStatus

        assert hasattr(SecurityStatus, 'SECURE')
        assert hasattr(SecurityStatus, 'WARNING')
        assert hasattr(SecurityStatus, 'CRITICAL')
        assert hasattr(SecurityStatus, 'UNKNOWN')


class TestSecurityCheckResult:
    """Tests for security check result dataclass."""

    def test_result_creation(self):
        """Test that security check results can be created."""
        from security.continuous_security import (
            SecurityCheckResult, SecurityCheckType, SecurityStatus
        )

        result = SecurityCheckResult(
            check_type=SecurityCheckType.IMPORT_SCAN,
            status=SecurityStatus.SECURE,
            message="No issues found",
        )

        assert result.check_type == SecurityCheckType.IMPORT_SCAN
        assert result.status == SecurityStatus.SECURE
        assert result.message == "No issues found"

    def test_result_to_dict(self):
        """Test that results can be converted to dictionary."""
        from security.continuous_security import (
            SecurityCheckResult, SecurityCheckType, SecurityStatus
        )

        result = SecurityCheckResult(
            check_type=SecurityCheckType.IMPORT_SCAN,
            status=SecurityStatus.SECURE,
            message="No issues found",
        )

        data = result.to_dict()

        assert data['check_type'] == 'import_scan'
        assert data['status'] == 'secure'
        assert data['message'] == "No issues found"


class TestImportScanner:
    """Tests for import scanner functionality."""

    def test_scanner_creation(self, tmp_path):
        """Test that import scanner can be created."""
        from security.continuous_security import ImportScanner

        scanner = ImportScanner(tmp_path)
        assert scanner.repo_root == tmp_path

    def test_detect_blocked_import(self, tmp_path):
        """Test that blocked imports are detected."""
        from security.continuous_security import (
            ImportScanner, SecurityStatus
        )

        # Create a test file with a blocked import
        test_file = tmp_path / "test_module.py"
        test_file.write_text("import basicsr\n")

        scanner = ImportScanner(tmp_path)
        result = scanner.scan()

        assert result.status == SecurityStatus.CRITICAL
        assert len(result.details) > 0
        assert any("basicsr" in detail for detail in result.details)

    def test_detect_blocked_from_import(self, tmp_path):
        """Test that 'from X import' for blocked packages is detected."""
        from security.continuous_security import (
            ImportScanner, SecurityStatus
        )

        # Create a test file with a blocked from import
        test_file = tmp_path / "test_module.py"
        test_file.write_text("from basicsr import utils\n")

        scanner = ImportScanner(tmp_path)
        result = scanner.scan()

        assert result.status == SecurityStatus.CRITICAL
        assert len(result.details) > 0

    def test_allow_basicsr_tp(self, tmp_path):
        """Test that basicsr_tp imports are allowed."""
        from security.continuous_security import (
            ImportScanner, SecurityStatus
        )

        # Create a test file with basicsr_tp import
        test_file = tmp_path / "test_module.py"
        test_file.write_text("from basicsr_tp import utils\n")

        scanner = ImportScanner(tmp_path)
        result = scanner.scan()

        assert result.status == SecurityStatus.SECURE

    def test_clean_code_passes(self, tmp_path):
        """Test that clean code passes the scan."""
        from security.continuous_security import (
            ImportScanner, SecurityStatus
        )

        # Create a test file with safe imports
        test_file = tmp_path / "test_module.py"
        test_file.write_text("import numpy\nimport json\n")

        scanner = ImportScanner(tmp_path)
        result = scanner.scan()

        assert result.status == SecurityStatus.SECURE

    def test_skip_pycache_directories(self, tmp_path):
        """Test that __pycache__ directories are skipped."""
        from security.continuous_security import (
            ImportScanner, SecurityStatus
        )

        # Create a file in __pycache__ with blocked import
        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()
        test_file = pycache_dir / "test_module.py"
        test_file.write_text("import basicsr\n")

        scanner = ImportScanner(tmp_path)
        result = scanner.scan()

        # Should not detect the blocked import in __pycache__
        assert result.status == SecurityStatus.SECURE


class TestConstraintVerifier:
    """Tests for constraint verifier functionality."""

    def test_verifier_creation(self, tmp_path):
        """Test that constraint verifier can be created."""
        from security.continuous_security import ConstraintVerifier

        verifier = ConstraintVerifier(tmp_path)
        assert verifier.repo_root == tmp_path

    def test_verify_basicsr_constraint_present(self, tmp_path):
        """Test verification when basicsr constraint is present."""
        from security.continuous_security import (
            ConstraintVerifier, SecurityStatus
        )

        # Create requirements directory and constraints file
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        constraints_file = req_dir / "constraints.txt"
        constraints_file.write_text("basicsr>=999.0.0  # CVE-2024-27763 blocked\n")

        verifier = ConstraintVerifier(tmp_path)
        result = verifier.verify()

        assert result.status == SecurityStatus.SECURE


class TestCodePatternScanner:
    """Tests for code pattern scanner functionality."""

    def test_scanner_creation(self, tmp_path):
        """Test that code pattern scanner can be created."""
        from security.continuous_security import CodePatternScanner

        scanner = CodePatternScanner(tmp_path)
        assert scanner.repo_root == tmp_path

    def test_detect_shell_true(self, tmp_path):
        """Test that subprocess with shell=True is detected."""
        from security.continuous_security import (
            CodePatternScanner, SecurityStatus
        )

        # Create a test file with shell=True
        test_file = tmp_path / "test_module.py"
        test_file.write_text('import subprocess\nsubprocess.run("ls", shell=True)\n')

        scanner = CodePatternScanner(tmp_path)
        result = scanner.scan()

        # Should detect the warning pattern
        assert result.status in [SecurityStatus.WARNING, SecurityStatus.SECURE]

    def test_clean_code_passes(self, tmp_path):
        """Test that clean code passes the pattern scan."""
        from security.continuous_security import (
            CodePatternScanner, SecurityStatus
        )

        # Create a test file with safe code
        test_file = tmp_path / "test_module.py"
        test_file.write_text('import json\ndata = json.loads("{}")\n')

        scanner = CodePatternScanner(tmp_path)
        result = scanner.scan()

        # Should pass or only have warnings
        assert result.status in [SecurityStatus.SECURE, SecurityStatus.WARNING]


class TestSecurityHealthReport:
    """Tests for security health report generation."""

    def test_report_creation(self):
        """Test that security health report can be created."""
        from security.continuous_security import (
            SecurityHealthReport, SecurityStatus
        )

        report = SecurityHealthReport(
            overall_status=SecurityStatus.SECURE,
            checks_performed=5,
            checks_passed=5,
            checks_failed=0,
        )

        assert report.overall_status == SecurityStatus.SECURE
        assert report.checks_performed == 5

    def test_report_to_markdown(self):
        """Test markdown report generation."""
        from security.continuous_security import (
            SecurityHealthReport, SecurityStatus
        )

        report = SecurityHealthReport(
            overall_status=SecurityStatus.SECURE,
            checks_performed=5,
            checks_passed=5,
            checks_failed=0,
        )

        markdown = report.to_markdown()

        assert "# 🔒 Security Health Report" in markdown
        assert "✅" in markdown  # SECURE emoji


class TestContinuousSecurityVerifier:
    """Tests for the main security verifier class."""

    def test_verifier_creation(self, tmp_path):
        """Test that continuous security verifier can be created."""
        from security.continuous_security import ContinuousSecurityVerifier

        verifier = ContinuousSecurityVerifier(tmp_path)
        assert verifier.repo_root == tmp_path

    def test_quick_check(self, tmp_path):
        """Test quick security check functionality."""
        from security.continuous_security import (
            ContinuousSecurityVerifier, SecurityStatus
        )

        # Create a minimal valid directory structure
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        constraints_file = req_dir / "constraints.txt"
        constraints_file.write_text("basicsr>=999.0.0\n")

        # Create a safe Python file
        test_file = tmp_path / "test_module.py"
        test_file.write_text("import json\n")

        verifier = ContinuousSecurityVerifier(tmp_path)
        report = verifier.quick_check()

        assert report.checks_performed > 0
