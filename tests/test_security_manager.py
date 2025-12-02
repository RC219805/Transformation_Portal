#!/usr/bin/env python3
"""Tests for the security_manager module.

Tests cover:
- Dependency scanner for package detection
- Mitigation engine for pattern matching and suggestions
- Security knowledge base operations
- Security manager scan and report functionality
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path for importing
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


class TestSeverityLevel:
    """Tests for severity level enum."""

    def test_severity_levels_exist(self):
        """Test that all expected severity levels are defined."""
        from security.security_manager import SeverityLevel

        assert hasattr(SeverityLevel, 'CRITICAL')
        assert hasattr(SeverityLevel, 'HIGH')
        assert hasattr(SeverityLevel, 'MEDIUM')
        assert hasattr(SeverityLevel, 'LOW')
        assert hasattr(SeverityLevel, 'INFORMATIONAL')


class TestMitigationType:
    """Tests for mitigation type enum."""

    def test_mitigation_types_exist(self):
        """Test that all expected mitigation types are defined."""
        from security.security_manager import MitigationType

        assert hasattr(MitigationType, 'CONSTRAINT')
        assert hasattr(MitigationType, 'VENDOR')
        assert hasattr(MitigationType, 'UPGRADE')
        assert hasattr(MitigationType, 'REMOVE')


class TestMitigationStatus:
    """Tests for mitigation status enum."""

    def test_mitigation_statuses_exist(self):
        """Test that all expected mitigation statuses are defined."""
        from security.security_manager import MitigationStatus

        assert hasattr(MitigationStatus, 'NOT_STARTED')
        assert hasattr(MitigationStatus, 'IN_PROGRESS')
        assert hasattr(MitigationStatus, 'MITIGATED')
        assert hasattr(MitigationStatus, 'ACCEPTED_RISK')
        assert hasattr(MitigationStatus, 'FALSE_POSITIVE')


class TestVulnerability:
    """Tests for vulnerability dataclass."""

    def test_vulnerability_creation(self):
        """Test that vulnerability objects can be created."""
        from security.security_manager import Vulnerability, SeverityLevel

        vuln = Vulnerability(
            cve_id="CVE-2024-27763",
            package_name="basicsr",
            affected_versions="<1.0.0",
            severity=SeverityLevel.CRITICAL,
            description="Command injection vulnerability",
        )

        assert vuln.cve_id == "CVE-2024-27763"
        assert vuln.package_name == "basicsr"
        assert vuln.severity == SeverityLevel.CRITICAL

    def test_vulnerability_to_dict(self):
        """Test that vulnerabilities can be converted to dictionary."""
        from security.security_manager import Vulnerability, SeverityLevel

        vuln = Vulnerability(
            cve_id="CVE-2024-27763",
            package_name="basicsr",
            affected_versions="<1.0.0",
            severity=SeverityLevel.CRITICAL,
            description="Command injection vulnerability",
        )

        data = vuln.to_dict()

        assert data['cve_id'] == "CVE-2024-27763"
        assert data['severity'] == 'critical'

    def test_vulnerability_from_dict(self):
        """Test that vulnerabilities can be created from dictionary."""
        from security.security_manager import Vulnerability, SeverityLevel

        data = {
            'cve_id': "CVE-2024-27763",
            'package_name': "basicsr",
            'affected_versions': "<1.0.0",
            'severity': 'critical',
            'description': "Command injection vulnerability",
        }

        vuln = Vulnerability.from_dict(data)

        assert vuln.cve_id == "CVE-2024-27763"
        assert vuln.severity == SeverityLevel.CRITICAL


class TestMitigation:
    """Tests for mitigation dataclass."""

    def test_mitigation_creation(self):
        """Test that mitigation objects can be created."""
        from security.security_manager import (
            Mitigation, MitigationType, MitigationStatus
        )

        mitigation = Mitigation(
            cve_id="CVE-2024-27763",
            mitigation_type=MitigationType.CONSTRAINT,
            status=MitigationStatus.NOT_STARTED,
            description="Block via constraint",
        )

        assert mitigation.cve_id == "CVE-2024-27763"
        assert mitigation.mitigation_type == MitigationType.CONSTRAINT

    def test_mitigation_to_dict(self):
        """Test that mitigations can be converted to dictionary."""
        from security.security_manager import (
            Mitigation, MitigationType, MitigationStatus
        )

        mitigation = Mitigation(
            cve_id="CVE-2024-27763",
            mitigation_type=MitigationType.CONSTRAINT,
            status=MitigationStatus.NOT_STARTED,
            description="Block via constraint",
        )

        data = mitigation.to_dict()

        assert data['cve_id'] == "CVE-2024-27763"
        assert data['mitigation_type'] == 'constraint'
        assert data['status'] == 'not_started'


class TestSecurityPolicy:
    """Tests for security policy dataclass."""

    def test_policy_creation(self):
        """Test that security policies can be created."""
        from security.security_manager import SecurityPolicy

        policy = SecurityPolicy(
            name="block-basicsr",
            description="Block basicsr due to CVE-2024-27763",
            blocked_packages={"basicsr": "CVE-2024-27763"},
        )

        assert policy.name == "block-basicsr"
        assert "basicsr" in policy.blocked_packages


class TestDependencyScanner:
    """Tests for dependency scanner functionality."""

    def test_scanner_creation(self, tmp_path):
        """Test that dependency scanner can be created."""
        from security.security_manager import DependencyScanner

        scanner = DependencyScanner(tmp_path)
        assert scanner.repo_root == tmp_path

    def test_find_requirements_files(self, tmp_path):
        """Test finding requirements files."""
        from security.security_manager import DependencyScanner

        # Create requirements file
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy==1.24.0\npillow>=10.0\n")

        scanner = DependencyScanner(tmp_path)
        files = scanner._find_requirements_files()

        assert len(files) >= 1

    def test_get_blocked_packages(self, tmp_path):
        """Test detection of blocked packages from constraints."""
        from security.security_manager import DependencyScanner

        # Create constraints file
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        constraints_file = req_dir / "constraints.txt"
        constraints_file.write_text("basicsr>=999.0.0  # CVE-2024-27763 blocked\n")

        scanner = DependencyScanner(tmp_path)
        blocked = scanner.get_blocked_packages()

        assert "basicsr" in blocked


class TestMitigationEngine:
    """Tests for mitigation engine functionality."""

    def test_engine_creation(self, tmp_path):
        """Test that mitigation engine can be created."""
        from security.security_manager import MitigationEngine

        engine = MitigationEngine(tmp_path)
        assert engine.repo_root == tmp_path

    def test_has_built_in_patterns(self, tmp_path):
        """Test that engine has built-in mitigation patterns."""
        from security.security_manager import MitigationEngine

        engine = MitigationEngine(tmp_path)

        # Should have at least the basicsr pattern
        assert len(engine.MITIGATION_PATTERNS) > 0
        assert "basicsr" in engine.MITIGATION_PATTERNS


class TestSecurityKnowledgeBase:
    """Tests for security knowledge base."""

    def test_kb_creation(self, tmp_path):
        """Test that knowledge base can be created."""
        from security.security_manager import SecurityKnowledgeBase

        kb = SecurityKnowledgeBase(tmp_path)
        assert kb.repo_root == tmp_path

    def test_index_vulnerability(self, tmp_path):
        """Test indexing a vulnerability to knowledge base."""
        from security.security_manager import (
            SecurityKnowledgeBase, Vulnerability, SeverityLevel
        )

        kb = SecurityKnowledgeBase(tmp_path)

        vuln = Vulnerability(
            cve_id="CVE-2024-12345",
            package_name="test-package",
            affected_versions="<1.0.0",
            severity=SeverityLevel.HIGH,
            description="Test vulnerability",
        )

        chunk_id = kb.index_vulnerability(vuln)

        # Should return a chunk ID
        assert chunk_id is not None
        assert len(chunk_id) > 0


class TestSecurityManager:
    """Tests for the main security manager class."""

    def test_manager_creation(self, tmp_path):
        """Test that security manager can be created."""
        from security.security_manager import SecurityManager

        manager = SecurityManager(tmp_path)
        assert manager.repo_root == tmp_path

    def test_scan_empty_repo(self, tmp_path):
        """Test scanning an empty repository."""
        from security.security_manager import SecurityManager

        manager = SecurityManager(tmp_path)
        report = manager.scan()

        assert report is not None
        assert hasattr(report, 'vulnerabilities')

    def test_get_blocked_packages(self, tmp_path):
        """Test getting blocked packages."""
        from security.security_manager import SecurityManager

        # Create constraints file with blocked package
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        constraints_file = req_dir / "constraints.txt"
        constraints_file.write_text("basicsr>=999.0.0  # CVE-2024-27763 blocked\n")

        manager = SecurityManager(tmp_path)
        blocked = manager.scanner.get_blocked_packages()

        assert "basicsr" in blocked


class TestSecurityReport:
    """Tests for security report generation."""

    def test_report_creation(self):
        """Test that security reports can be created."""
        from security.security_manager import SecurityReport

        report = SecurityReport()

        assert hasattr(report, 'vulnerabilities')
        assert hasattr(report, 'mitigations')

    def test_report_to_markdown(self):
        """Test markdown report generation."""
        from security.security_manager import SecurityReport

        report = SecurityReport()
        markdown = report.to_markdown()

        assert "Security" in markdown
        assert "Report" in markdown
