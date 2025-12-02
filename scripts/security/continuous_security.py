#!/usr/bin/env python3
"""
Continuous Security Verification System
========================================
Ensures security integrity is maintained as a routine, continuous practice.

This module provides:
- Pre-commit security checks
- Continuous import monitoring
- Runtime package verification
- Automated security health checks
- Integration with CI/CD pipelines

Security Verification Layers:
    1. PRE-COMMIT: Check before any code is committed
    2. CI/CD: Verify on every push and pull request
    3. RUNTIME: Check when code is imported/executed
    4. SCHEDULED: Daily/weekly comprehensive scans
    5. ON-DEMAND: Manual verification triggers

Usage:
    # Quick security check (for pre-commit)
    python continuous_security.py quick

    # Full security audit
    python continuous_security.py full

    # Verify specific aspect
    python continuous_security.py verify --imports
    python continuous_security.py verify --packages
    python continuous_security.py verify --constraints

    # Generate security health report
    python continuous_security.py health

Author: Transformation Portal
Version: 1.0.0
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("continuous_security")


# =============================================================================
# Constants
# =============================================================================


class SecurityCheckType(Enum):
    """Types of security checks."""
    IMPORT_SCAN = "import_scan"
    PACKAGE_AUDIT = "package_audit"
    CONSTRAINT_VERIFY = "constraint_verify"
    VULNERABILITY_SCAN = "vulnerability_scan"
    CODE_PATTERN = "code_pattern"


class SecurityStatus(Enum):
    """Security status levels."""
    SECURE = "secure"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# Known vulnerable packages and their mitigations
BLOCKED_PACKAGES: Dict[str, Dict[str, Any]] = {
    "basicsr": {
        "cve": "CVE-2024-27763",
        "severity": "medium",
        "reason": "Command injection vulnerability in SLURM utilities",
        "mitigation": "Use vendored basicsr_tp package instead",
        "constraint": "basicsr>=999.0.0",
    },
}

# Patterns that indicate potential security issues
DANGEROUS_PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "subprocess_shell",
        "pattern": r"subprocess\.(run|call|Popen)\([^)]*shell\s*=\s*True",
        "severity": "warning",
        "message": "subprocess with shell=True can be dangerous",
    },
    {
        "name": "eval_exec",
        "pattern": r"\b(eval|exec)\s*\(",
        "severity": "warning",
        "message": "eval/exec can execute arbitrary code",
    },
    {
        "name": "pickle_load",
        "pattern": r"pickle\.loads?\s*\(",
        "severity": "warning",
        "message": "pickle.load can execute arbitrary code during deserialization",
    },
]


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SecurityCheckResult:
    """Result of a security check."""
    check_type: SecurityCheckType
    status: SecurityStatus
    message: str
    details: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class SecurityHealthReport:
    """Comprehensive security health report."""
    overall_status: SecurityStatus = SecurityStatus.UNKNOWN
    checks_performed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    warnings: int = 0
    results: List[SecurityCheckResult] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    recommendations: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        status_emoji = {
            SecurityStatus.SECURE: "✅",
            SecurityStatus.WARNING: "⚠️",
            SecurityStatus.CRITICAL: "🔴",
            SecurityStatus.UNKNOWN: "❓",
        }

        lines = [
            "# 🔒 Security Health Report",
            "",
            f"**Generated:** {self.generated_at}",
            f"**Overall Status:** {status_emoji.get(self.overall_status, '❓')} {self.overall_status.value.upper()}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Checks Performed | {self.checks_performed} |",
            f"| Passed | {self.checks_passed} |",
            f"| Failed | {self.checks_failed} |",
            f"| Warnings | {self.warnings} |",
            "",
        ]

        if self.results:
            lines.extend([
                "## Check Results",
                "",
            ])
            for result in self.results:
                emoji = status_emoji.get(result.status, "❓")
                lines.append(f"### {emoji} {result.check_type.value}")
                lines.append(f"**Status:** {result.status.value}")
                lines.append(f"**Message:** {result.message}")
                if result.details:
                    lines.append("**Details:**")
                    for detail in result.details[:5]:  # Limit to 5 details
                        lines.append(f"- {detail}")
                lines.append("")

        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Security Checkers
# =============================================================================


class ImportScanner:
    """Scans for vulnerable imports."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def scan(self) -> SecurityCheckResult:
        """Scan all Python files for vulnerable imports."""
        violations = []

        for py_file in self.repo_root.rglob("*.py"):
            # Skip certain directories
            if any(skip in str(py_file) for skip in [
                ".git", "__pycache__", ".venv", "venv", "basicsr_tp"
            ]):
                continue

            file_violations = self._check_file(py_file)
            violations.extend(file_violations)

        if violations:
            return SecurityCheckResult(
                check_type=SecurityCheckType.IMPORT_SCAN,
                status=SecurityStatus.CRITICAL,
                message=f"Found {len(violations)} vulnerable import(s)",
                details=violations,
            )

        return SecurityCheckResult(
            check_type=SecurityCheckType.IMPORT_SCAN,
            status=SecurityStatus.SECURE,
            message="No vulnerable imports detected",
        )

    def _check_file(self, py_file: Path) -> List[str]:
        """Check a single file for vulnerable imports."""
        violations = []

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(py_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_parts = node.module.split('.')
                        if module_parts[0] in BLOCKED_PACKAGES and module_parts[0] != 'basicsr_tp':
                            rel_path = py_file.relative_to(self.repo_root)
                            violations.append(
                                f"{rel_path}:{node.lineno} - imports from blocked package '{module_parts[0]}'"
                            )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_parts = alias.name.split('.')
                        if module_parts[0] in BLOCKED_PACKAGES and module_parts[0] != 'basicsr_tp':
                            rel_path = py_file.relative_to(self.repo_root)
                            violations.append(
                                f"{rel_path}:{node.lineno} - imports blocked package '{module_parts[0]}'"
                            )

        except (SyntaxError, UnicodeDecodeError):
            pass

        return violations


class PackageAuditor:
    """Audits installed packages for vulnerabilities."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def audit(self) -> SecurityCheckResult:
        """Audit installed packages."""
        violations = []

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            packages = json.loads(result.stdout)
            installed = {pkg["name"].lower(): pkg["version"] for pkg in packages}

            for pkg_name, pkg_info in BLOCKED_PACKAGES.items():
                if pkg_name.lower() in installed:
                    violations.append(
                        f"CRITICAL: {pkg_name}=={installed[pkg_name.lower()]} is installed! "
                        f"({pkg_info['cve']})"
                    )

        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            return SecurityCheckResult(
                check_type=SecurityCheckType.PACKAGE_AUDIT,
                status=SecurityStatus.WARNING,
                message=f"Package audit incomplete: {e}",
            )

        if violations:
            return SecurityCheckResult(
                check_type=SecurityCheckType.PACKAGE_AUDIT,
                status=SecurityStatus.CRITICAL,
                message=f"Found {len(violations)} vulnerable package(s) installed",
                details=violations,
            )

        return SecurityCheckResult(
            check_type=SecurityCheckType.PACKAGE_AUDIT,
            status=SecurityStatus.SECURE,
            message="No vulnerable packages installed",
        )


class ConstraintVerifier:
    """Verifies security constraints are properly configured."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.constraints_file = repo_root / "requirements" / "constraints.txt"

    def verify(self) -> SecurityCheckResult:
        """Verify constraints file is properly configured."""
        issues = []

        # Check constraints file exists
        if not self.constraints_file.exists():
            return SecurityCheckResult(
                check_type=SecurityCheckType.CONSTRAINT_VERIFY,
                status=SecurityStatus.CRITICAL,
                message="constraints.txt file is missing!",
                details=["Create requirements/constraints.txt with blocked packages"],
            )

        # Read constraints
        with open(self.constraints_file, 'r') as f:
            content = f.read()

        # Verify all blocked packages have constraints
        for pkg_name, pkg_info in BLOCKED_PACKAGES.items():
            expected_constraint = pkg_info.get("constraint", f"{pkg_name}>=999.0.0")
            if pkg_name not in content.lower():
                issues.append(f"Missing constraint for blocked package: {pkg_name}")
            elif ">=999" not in content:
                issues.append(f"Constraint for {pkg_name} may not be blocking properly")

        if issues:
            return SecurityCheckResult(
                check_type=SecurityCheckType.CONSTRAINT_VERIFY,
                status=SecurityStatus.WARNING,
                message="Constraint configuration issues found",
                details=issues,
            )

        return SecurityCheckResult(
            check_type=SecurityCheckType.CONSTRAINT_VERIFY,
            status=SecurityStatus.SECURE,
            message="All security constraints properly configured",
        )


class CodePatternScanner:
    """Scans for potentially dangerous code patterns."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def scan(self) -> SecurityCheckResult:
        """Scan for dangerous code patterns."""
        findings = []

        for py_file in self.repo_root.rglob("*.py"):
            # Skip certain directories
            if any(skip in str(py_file) for skip in [
                ".git", "__pycache__", ".venv", "venv", "test_", "_test.py"
            ]):
                continue

            file_findings = self._check_file(py_file)
            findings.extend(file_findings)

        if findings:
            return SecurityCheckResult(
                check_type=SecurityCheckType.CODE_PATTERN,
                status=SecurityStatus.WARNING,
                message=f"Found {len(findings)} potentially dangerous pattern(s)",
                details=findings[:10],  # Limit details
            )

        return SecurityCheckResult(
            check_type=SecurityCheckType.CODE_PATTERN,
            status=SecurityStatus.SECURE,
            message="No dangerous patterns detected",
        )

    def _check_file(self, py_file: Path) -> List[str]:
        """Check a file for dangerous patterns."""
        findings = []

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()

            for pattern_info in DANGEROUS_PATTERNS:
                pattern = re.compile(pattern_info["pattern"])
                for i, line in enumerate(lines, 1):
                    if pattern.search(line):
                        rel_path = py_file.relative_to(self.repo_root)
                        findings.append(
                            f"{rel_path}:{i} - {pattern_info['message']}"
                        )

        except (UnicodeDecodeError, IOError):
            pass

        return findings


# =============================================================================
# Continuous Security Verifier
# =============================================================================


class ContinuousSecurityVerifier:
    """Main interface for continuous security verification."""

    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            repo_root = self._find_repo_root()
        self.repo_root = repo_root

        # Initialize checkers
        self.import_scanner = ImportScanner(repo_root)
        self.package_auditor = PackageAuditor(repo_root)
        self.constraint_verifier = ConstraintVerifier(repo_root)
        self.pattern_scanner = CodePatternScanner(repo_root)

        # State storage
        self.state_dir = repo_root / ".github" / "security"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.last_check_file = self.state_dir / "last_check.json"

    def _find_repo_root(self) -> Path:
        """Find repository root."""
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if (current / '.git').exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        return Path.cwd()

    def quick_check(self) -> SecurityHealthReport:
        """Run quick security checks (for pre-commit)."""
        report = SecurityHealthReport()

        # Only run fast checks
        checks = [
            self.import_scanner.scan,
            self.constraint_verifier.verify,
        ]

        for check in checks:
            result = check()
            report.results.append(result)
            report.checks_performed += 1

            if result.status == SecurityStatus.SECURE:
                report.checks_passed += 1
            elif result.status == SecurityStatus.WARNING:
                report.warnings += 1
            elif result.status == SecurityStatus.CRITICAL:
                report.checks_failed += 1

        report.overall_status = self._determine_overall_status(report)
        self._save_check_result(report)
        return report

    def full_audit(self) -> SecurityHealthReport:
        """Run comprehensive security audit."""
        report = SecurityHealthReport()

        checks = [
            self.import_scanner.scan,
            self.package_auditor.audit,
            self.constraint_verifier.verify,
            self.pattern_scanner.scan,
        ]

        for check in checks:
            result = check()
            report.results.append(result)
            report.checks_performed += 1

            if result.status == SecurityStatus.SECURE:
                report.checks_passed += 1
            elif result.status == SecurityStatus.WARNING:
                report.warnings += 1
            elif result.status == SecurityStatus.CRITICAL:
                report.checks_failed += 1

        report.overall_status = self._determine_overall_status(report)
        report.recommendations = self._generate_recommendations(report)
        self._save_check_result(report)
        return report

    def verify_specific(self, check_type: str) -> SecurityCheckResult:
        """Run a specific security check."""
        checkers = {
            "imports": self.import_scanner.scan,
            "packages": self.package_auditor.audit,
            "constraints": self.constraint_verifier.verify,
            "patterns": self.pattern_scanner.scan,
        }

        if check_type not in checkers:
            raise ValueError(f"Unknown check type: {check_type}")

        return checkers[check_type]()

    def _determine_overall_status(self, report: SecurityHealthReport) -> SecurityStatus:
        """Determine overall security status."""
        if report.checks_failed > 0:
            return SecurityStatus.CRITICAL
        elif report.warnings > 0:
            return SecurityStatus.WARNING
        elif report.checks_passed == report.checks_performed:
            return SecurityStatus.SECURE
        return SecurityStatus.UNKNOWN

    def _generate_recommendations(self, report: SecurityHealthReport) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        for result in report.results:
            if result.status == SecurityStatus.CRITICAL:
                if result.check_type == SecurityCheckType.IMPORT_SCAN:
                    recommendations.append(
                        "Replace vulnerable imports with secure alternatives (e.g., basicsr -> basicsr_tp)"
                    )
                elif result.check_type == SecurityCheckType.PACKAGE_AUDIT:
                    recommendations.append(
                        "Remove vulnerable packages and reinstall with constraints: "
                        "pip install -c requirements/constraints.txt -r requirements.txt"
                    )
                elif result.check_type == SecurityCheckType.CONSTRAINT_VERIFY:
                    recommendations.append(
                        "Update requirements/constraints.txt with proper blocking constraints"
                    )

        if not recommendations:
            recommendations.append("Continue maintaining current security practices")

        return recommendations

    def _save_check_result(self, report: SecurityHealthReport) -> None:
        """Save check result for tracking."""
        try:
            with open(self.last_check_file, 'w') as f:
                json.dump({
                    "status": report.overall_status.value,
                    "checks_performed": report.checks_performed,
                    "checks_passed": report.checks_passed,
                    "checks_failed": report.checks_failed,
                    "timestamp": report.generated_at,
                }, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save check result: {e}")

    def get_last_check(self) -> Optional[Dict[str, Any]]:
        """Get last check result."""
        if self.last_check_file.exists():
            try:
                with open(self.last_check_file, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                pass
        return None


# =============================================================================
# Runtime Security Guard
# =============================================================================


def security_guard() -> bool:
    """
    Runtime security guard - call this at module import time to verify security.

    Usage:
        # At the top of critical modules
        from scripts.security.continuous_security import security_guard
        if not security_guard():
            raise RuntimeError("Security verification failed")

    Returns:
        True if security checks pass, False otherwise.
    """
    try:
        # Quick check for vulnerable packages
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "basicsr"],
            capture_output=True,
            check=False
        )
        if result.returncode == 0:
            logger.critical("SECURITY VIOLATION: basicsr package is installed!")
            return False

        return True

    except Exception as e:
        logger.warning(f"Security guard check failed: {e}")
        return True  # Don't block on check failures


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Continuous Security Verification System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quick check
    subparsers.add_parser("quick", help="Run quick security checks (for pre-commit)")

    # Full audit
    subparsers.add_parser("full", help="Run comprehensive security audit")

    # Verify specific
    verify_parser = subparsers.add_parser("verify", help="Verify specific aspect")
    verify_parser.add_argument(
        "--imports", action="store_true", help="Check for vulnerable imports"
    )
    verify_parser.add_argument(
        "--packages", action="store_true", help="Audit installed packages"
    )
    verify_parser.add_argument(
        "--constraints", action="store_true", help="Verify constraint configuration"
    )
    verify_parser.add_argument(
        "--patterns", action="store_true", help="Scan for dangerous patterns"
    )

    # Health report
    subparsers.add_parser("health", help="Generate security health report")

    # Status
    subparsers.add_parser("status", help="Show last check status")

    args = parser.parse_args()

    verifier = ContinuousSecurityVerifier()

    if args.command == "quick":
        report = verifier.quick_check()
        if report.overall_status == SecurityStatus.SECURE:
            print("✅ Quick security check PASSED")
            sys.exit(0)
        else:
            print(report.to_markdown())
            sys.exit(1 if report.overall_status == SecurityStatus.CRITICAL else 0)

    elif args.command == "full":
        report = verifier.full_audit()
        print(report.to_markdown())
        sys.exit(1 if report.overall_status == SecurityStatus.CRITICAL else 0)

    elif args.command == "verify":
        if args.imports:
            result = verifier.verify_specific("imports")
        elif args.packages:
            result = verifier.verify_specific("packages")
        elif args.constraints:
            result = verifier.verify_specific("constraints")
        elif args.patterns:
            result = verifier.verify_specific("patterns")
        else:
            parser.print_help()
            sys.exit(1)

        emoji = "✅" if result.status == SecurityStatus.SECURE else "❌"
        print(f"{emoji} {result.check_type.value}: {result.message}")
        if result.details:
            for detail in result.details:
                print(f"  - {detail}")
        sys.exit(0 if result.status == SecurityStatus.SECURE else 1)

    elif args.command == "health":
        report = verifier.full_audit()
        print(report.to_markdown())

    elif args.command == "status":
        last_check = verifier.get_last_check()
        if last_check:
            status = last_check.get("status", "unknown")
            timestamp = last_check.get("timestamp", "unknown")
            emoji = "✅" if status == "secure" else "⚠️" if status == "warning" else "❌"
            print(f"{emoji} Last check: {status.upper()}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Passed: {last_check.get('checks_passed', 0)}/{last_check.get('checks_performed', 0)}")
        else:
            print("No previous security checks found")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
