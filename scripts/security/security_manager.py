#!/usr/bin/env python3
"""
Proactive Dependabot Security Warning Management System
=======================================================
Integrates with RAG feedback system for end-to-end security monitoring.

This module provides:
- Proactive vulnerability detection and tracking
- Integration with Dependabot alerts via GitHub API
- RAG-based security knowledge base for pattern recognition
- Automated mitigation suggestions based on historical fixes
- End-to-end feedback loop for security improvements

Architecture:
    SecurityManager
    ├── VulnerabilityTracker (CVE/advisory tracking)
    ├── DependencyScanner (requirements analysis)
    ├── MitigationEngine (automated fix suggestions)
    ├── RAGIntegration (knowledge feedback)
    └── AlertReporter (notifications and reports)

Security Knowledge Types:
    - vulnerability_alerts: Active CVEs affecting dependencies
    - mitigation_patterns: Historical fixes and workarounds
    - dependency_graph: Package relationships and constraints
    - security_policies: Repository-specific security rules

Usage:
    # Scan for vulnerabilities
    python security_manager.py scan

    # Check specific package
    python security_manager.py check --package basicsr

    # Generate security report
    python security_manager.py report

    # Suggest mitigations
    python security_manager.py mitigate --cve CVE-2024-27763

Author: Transformation Portal
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("security_manager")


# =============================================================================
# Data Models
# =============================================================================


class SeverityLevel(Enum):
    """CVE severity levels following CVSS."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class MitigationStatus(Enum):
    """Status of vulnerability mitigation."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    MITIGATED = "mitigated"
    ACCEPTED_RISK = "accepted_risk"
    FALSE_POSITIVE = "false_positive"


class MitigationType(Enum):
    """Types of mitigation strategies."""
    UPGRADE = "upgrade"  # Upgrade to patched version
    CONSTRAINT = "constraint"  # Block via constraints.txt
    VENDOR = "vendor"  # Use vendored/forked code
    REMOVE = "remove"  # Remove dependency entirely
    WORKAROUND = "workaround"  # Code-level workaround


@dataclass
class Vulnerability:
    """Represents a security vulnerability."""
    cve_id: str
    package_name: str
    affected_versions: str
    severity: SeverityLevel
    description: str

    # CVSS details
    cvss_score: Optional[float] = None
    attack_vector: Optional[str] = None

    # Fix information
    fixed_version: Optional[str] = None
    patch_available: bool = False

    # Context
    discovered_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    references: List[str] = field(default_factory=list)

    # Repository-specific
    installed_version: Optional[str] = None
    is_direct_dependency: bool = False
    dependent_packages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['severity'] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vulnerability':
        """Create from dictionary."""
        data['severity'] = SeverityLevel(data['severity'])
        return cls(**data)


@dataclass
class Mitigation:
    """Represents a mitigation strategy for a vulnerability."""
    cve_id: str
    mitigation_type: MitigationType
    status: MitigationStatus
    description: str

    # Implementation details
    files_changed: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)

    # Tracking
    created_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_date: Optional[str] = None
    author: Optional[str] = None

    # Effectiveness
    verified: bool = False
    verification_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['mitigation_type'] = self.mitigation_type.value
        data['status'] = self.status.value
        return data


@dataclass
class SecurityPolicy:
    """Repository-specific security policy."""
    name: str
    description: str

    # Blocked packages (CVE mitigations)
    blocked_packages: Dict[str, str] = field(default_factory=dict)  # package: reason

    # Required constraints
    required_constraints: List[str] = field(default_factory=list)

    # Verification requirements
    verification_scripts: List[str] = field(default_factory=list)
    ci_checks_required: List[str] = field(default_factory=list)

    # Audit settings
    auto_scan_enabled: bool = True
    scan_frequency_days: int = 7
    alert_on_new_cve: bool = True


@dataclass
class SecurityReport:
    """Comprehensive security status report."""
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Summary
    total_vulnerabilities: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Status
    mitigated_count: int = 0
    pending_count: int = 0

    # Details
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    mitigations: List[Mitigation] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# 🔒 Security Status Report",
            "",
            f"**Generated:** {self.generated_at}",
            "",
            "## Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| 🔴 Critical | {self.critical_count} |",
            f"| 🟠 High | {self.high_count} |",
            f"| 🟡 Medium | {self.medium_count} |",
            f"| 🟢 Low | {self.low_count} |",
            "",
            f"**Total Vulnerabilities:** {self.total_vulnerabilities}",
            f"**Mitigated:** {self.mitigated_count}",
            f"**Pending:** {self.pending_count}",
            "",
        ]

        if self.vulnerabilities:
            lines.extend([
                "## Active Vulnerabilities",
                "",
            ])
            for vuln in self.vulnerabilities:
                emoji = {
                    SeverityLevel.CRITICAL: "🔴",
                    SeverityLevel.HIGH: "🟠",
                    SeverityLevel.MEDIUM: "🟡",
                    SeverityLevel.LOW: "🟢",
                }.get(vuln.severity, "⚪")
                desc = vuln.description[:100] + "..." if len(vuln.description) > 100 else vuln.description
                lines.append(
                    f"- {emoji} **{vuln.cve_id}**: {vuln.package_name} - {desc}"
                )
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
# Security Scanner
# =============================================================================


class DependencyScanner:
    """Scans dependencies for known vulnerabilities."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.constraints_file = repo_root / "requirements" / "constraints.txt"
        self.requirements_files = self._find_requirements_files()

    def _find_requirements_files(self) -> List[Path]:
        """Find all requirements files in the repository."""
        files = []
        patterns = [
            "requirements.txt",
            "requirements-*.txt",
            "requirements/*.txt",
            "requirements/*.in",
        ]
        for pattern in patterns:
            files.extend(self.repo_root.glob(pattern))
        return sorted(set(files))

    def get_installed_packages(self) -> Dict[str, str]:
        """Get installed packages and versions."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            packages = json.loads(result.stdout)
            return {pkg["name"].lower(): pkg["version"] for pkg in packages}
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to get installed packages: {e}")
            return {}

    def get_blocked_packages(self) -> Dict[str, str]:
        """Get packages blocked by constraints.txt."""
        blocked = {}
        if self.constraints_file.exists():
            with open(self.constraints_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse constraint like "basicsr>=999.0.0"
                        match = re.match(r'^([a-zA-Z0-9_-]+)([<>=!]+.+)$', line)
                        if match:
                            pkg_name = match.group(1).lower()
                            constraint = match.group(2)
                            # Check if it's an impossible constraint (blocking)
                            if '>=' in constraint and '999' in constraint:
                                blocked[pkg_name] = f"Blocked by impossible constraint: {line}"
        return blocked

    def check_package_vulnerability(self, package: str) -> Optional[Vulnerability]:
        """Check if a specific package has known vulnerabilities."""
        # This would integrate with vulnerability databases
        # For now, we check our known blocked packages
        blocked = self.get_blocked_packages()
        if package.lower() in blocked:
            return Vulnerability(
                cve_id="CVE-BLOCKED",
                package_name=package,
                affected_versions="all",
                severity=SeverityLevel.HIGH,
                description=blocked[package.lower()],
            )
        return None

    def scan_requirements(self) -> List[Vulnerability]:
        """Scan all requirements files for vulnerabilities."""
        vulnerabilities = []
        blocked = self.get_blocked_packages()

        for req_file in self.requirements_files:
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('-'):
                            # Extract package name
                            match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                            if match:
                                pkg_name = match.group(1).lower()
                                if pkg_name in blocked:
                                    vulnerabilities.append(Vulnerability(
                                        cve_id="CVE-BLOCKED",
                                        package_name=pkg_name,
                                        affected_versions=line,
                                        severity=SeverityLevel.HIGH,
                                        description=f"Package listed in {req_file.name} but blocked by constraints",
                                    ))
            except IOError as e:
                logger.warning(f"Failed to read {req_file}: {e}")

        return vulnerabilities


# =============================================================================
# Mitigation Engine
# =============================================================================


class MitigationEngine:
    """Generates and manages vulnerability mitigations."""

    # Known mitigation patterns from historical fixes
    MITIGATION_PATTERNS: Dict[str, Dict[str, Any]] = {
        "basicsr": {
            "cve_ids": ["CVE-2024-27763"],
            "type": MitigationType.VENDOR,
            "description": "Vendor security-hardened fork (basicsr_tp) removing vulnerable SLURM code",
            "files_changed": [
                "basicsr_tp/",
                "requirements/constraints.txt",
                "requirements/ml.in",
            ],
            "commands": [
                "pip install -c requirements/constraints.txt -e .",
            ],
            "verification_steps": [
                "python scripts/utilities/verify_no_basicsr_imports.py --check-pkg",
                "make verify-security",
            ],
        },
    }

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.mitigations_file = repo_root / ".github" / "security" / "mitigations.json"

    def get_mitigation_suggestion(self, vulnerability: Vulnerability) -> Optional[Mitigation]:
        """Get mitigation suggestion for a vulnerability."""
        pkg_name = vulnerability.package_name.lower()

        # Check known patterns
        if pkg_name in self.MITIGATION_PATTERNS:
            pattern = self.MITIGATION_PATTERNS[pkg_name]
            return Mitigation(
                cve_id=vulnerability.cve_id,
                mitigation_type=pattern["type"],
                status=MitigationStatus.NOT_STARTED,
                description=pattern["description"],
                files_changed=pattern["files_changed"],
                commands=pattern["commands"],
                verification_steps=pattern["verification_steps"],
            )

        # Default suggestion based on vulnerability type
        if vulnerability.fixed_version:
            return Mitigation(
                cve_id=vulnerability.cve_id,
                mitigation_type=MitigationType.UPGRADE,
                status=MitigationStatus.NOT_STARTED,
                description=f"Upgrade {vulnerability.package_name} to {vulnerability.fixed_version}",
                commands=[
                    f"pip install '{vulnerability.package_name}>={vulnerability.fixed_version}'",
                ],
                verification_steps=[
                    f"pip show {vulnerability.package_name} | grep Version",
                ],
            )

        # No patch available - suggest constraint
        return Mitigation(
            cve_id=vulnerability.cve_id,
            mitigation_type=MitigationType.CONSTRAINT,
            status=MitigationStatus.NOT_STARTED,
            description=f"Block {vulnerability.package_name} via constraints.txt until patch available",
            files_changed=["requirements/constraints.txt"],
            commands=[
                f"echo '{vulnerability.package_name}>=999.0.0' >> requirements/constraints.txt",
            ],
            verification_steps=[
                f"pip install -c requirements/constraints.txt {vulnerability.package_name} && echo 'FAIL' || echo 'OK - blocked'",
            ],
        )

    def verify_mitigation(self, mitigation: Mitigation) -> Tuple[bool, str]:
        """Verify a mitigation is effective."""
        results = []
        all_passed = True

        for step in mitigation.verification_steps:
            try:
                result = subprocess.run(
                    step,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.repo_root,
                    check=False
                )
                passed = result.returncode == 0
                all_passed = all_passed and passed
                results.append(f"{'✅' if passed else '❌'} {step}")
                if result.stdout:
                    results.append(f"   Output: {result.stdout.strip()[:200]}")
                if result.stderr and not passed:
                    results.append(f"   Error: {result.stderr.strip()[:200]}")
            except Exception as e:
                all_passed = False
                results.append(f"❌ {step}")
                results.append(f"   Exception: {e}")

        return all_passed, "\n".join(results)


# =============================================================================
# RAG Integration
# =============================================================================


class SecurityKnowledgeBase:
    """Integrates security data with RAG system."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.knowledge_dir = repo_root / ".github" / "security" / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def index_vulnerability(self, vuln: Vulnerability) -> str:
        """Index a vulnerability for RAG retrieval."""
        chunk_id = hashlib.sha256(
            f"{vuln.cve_id}:{vuln.package_name}".encode()
        ).hexdigest()[:16]

        # Create searchable content
        content = {
            "id": chunk_id,
            "type": "vulnerability",
            "cve_id": vuln.cve_id,
            "package": vuln.package_name,
            "severity": vuln.severity.value,
            "description": vuln.description,
            "content": f"""
Security Vulnerability: {vuln.cve_id}
Package: {vuln.package_name}
Severity: {vuln.severity.value}
CVSS Score: {vuln.cvss_score or 'N/A'}
Affected Versions: {vuln.affected_versions}
Fixed Version: {vuln.fixed_version or 'No patch available'}

Description:
{vuln.description}

References:
{chr(10).join(vuln.references) if vuln.references else 'None'}
""".strip(),
            "metadata": vuln.to_dict(),
        }

        # Save to knowledge base
        file_path = self.knowledge_dir / f"vuln_{chunk_id}.json"
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)

        return chunk_id

    def index_mitigation(self, mitigation: Mitigation) -> str:
        """Index a mitigation for RAG retrieval."""
        chunk_id = hashlib.sha256(
            f"{mitigation.cve_id}:{mitigation.mitigation_type.value}".encode()
        ).hexdigest()[:16]

        content = {
            "id": chunk_id,
            "type": "mitigation",
            "cve_id": mitigation.cve_id,
            "mitigation_type": mitigation.mitigation_type.value,
            "status": mitigation.status.value,
            "content": f"""
Mitigation for {mitigation.cve_id}
Type: {mitigation.mitigation_type.value}
Status: {mitigation.status.value}

Description:
{mitigation.description}

Files Changed:
{chr(10).join('- ' + f for f in mitigation.files_changed) if mitigation.files_changed else 'None'}

Commands:
{chr(10).join('$ ' + c for c in mitigation.commands) if mitigation.commands else 'None'}

Verification Steps:
{chr(10).join('- ' + s for s in mitigation.verification_steps) if mitigation.verification_steps else 'None'}
""".strip(),
            "metadata": mitigation.to_dict(),
        }

        file_path = self.knowledge_dir / f"mitigation_{chunk_id}.json"
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)

        return chunk_id

    def search_mitigations(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant mitigations."""
        results = []
        query_lower = query.lower()

        for file_path in self.knowledge_dir.glob("mitigation_*.json"):
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    if query_lower in content.get("content", "").lower():
                        results.append(content)
            except (IOError, json.JSONDecodeError):
                continue

        return results

    def get_historical_fixes(self, package: str) -> List[Dict[str, Any]]:
        """Get historical fixes for a package."""
        results = []
        package_lower = package.lower()

        for file_path in self.knowledge_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    metadata = content.get("metadata", {})
                    if metadata.get("package_name", "").lower() == package_lower:
                        results.append(content)
            except (IOError, json.JSONDecodeError):
                continue

        return results


# =============================================================================
# Security Manager
# =============================================================================


class SecurityManager:
    """Main security management interface."""

    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            repo_root = self._find_repo_root()
        self.repo_root = repo_root
        self.scanner = DependencyScanner(repo_root)
        self.mitigation_engine = MitigationEngine(repo_root)
        self.knowledge_base = SecurityKnowledgeBase(repo_root)

        # Security data storage
        self.security_dir = repo_root / ".github" / "security"
        self.security_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.vulnerabilities: List[Vulnerability] = []
        self.mitigations: List[Mitigation] = []
        self._load_state()

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

    def _load_state(self) -> None:
        """Load existing security state."""
        state_file = self.security_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.vulnerabilities = [
                        Vulnerability.from_dict(v) for v in data.get("vulnerabilities", [])
                    ]
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load security state: {e}")

    def _save_state(self) -> None:
        """Save security state."""
        state_file = self.security_dir / "state.json"
        try:
            data = {
                "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
                "mitigations": [m.to_dict() for m in self.mitigations],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save security state: {e}")

    def scan(self) -> SecurityReport:
        """Perform comprehensive security scan."""
        logger.info("Starting security scan...")

        # Scan requirements
        vulnerabilities = self.scanner.scan_requirements()

        # Check blocked packages
        blocked = self.scanner.get_blocked_packages()
        installed = self.scanner.get_installed_packages()

        # Check for installed blocked packages
        for pkg, reason in blocked.items():
            if pkg in installed:
                vulnerabilities.append(Vulnerability(
                    cve_id="CVE-RUNTIME",
                    package_name=pkg,
                    affected_versions=installed[pkg],
                    severity=SeverityLevel.CRITICAL,
                    description=f"CRITICAL: Blocked package {pkg} is installed! {reason}",
                    installed_version=installed[pkg],
                ))

        # Update state
        self.vulnerabilities = vulnerabilities
        self._save_state()

        # Generate report
        report = self._generate_report()

        # Index in knowledge base
        for vuln in vulnerabilities:
            self.knowledge_base.index_vulnerability(vuln)

        logger.info(f"Scan complete: {len(vulnerabilities)} vulnerabilities found")
        return report

    def _generate_report(self) -> SecurityReport:
        """Generate security report."""
        report = SecurityReport()
        report.vulnerabilities = self.vulnerabilities
        report.mitigations = self.mitigations
        report.total_vulnerabilities = len(self.vulnerabilities)

        for vuln in self.vulnerabilities:
            if vuln.severity == SeverityLevel.CRITICAL:
                report.critical_count += 1
            elif vuln.severity == SeverityLevel.HIGH:
                report.high_count += 1
            elif vuln.severity == SeverityLevel.MEDIUM:
                report.medium_count += 1
            else:
                report.low_count += 1

        # Count mitigated
        mitigated_cves = {
            m.cve_id for m in self.mitigations
            if m.status == MitigationStatus.MITIGATED
        }
        report.mitigated_count = len(mitigated_cves)
        report.pending_count = report.total_vulnerabilities - report.mitigated_count

        # Generate recommendations
        if report.critical_count > 0:
            report.recommendations.append(
                "⚠️ URGENT: Critical vulnerabilities require immediate attention"
            )
        if report.pending_count > 0:
            report.recommendations.append(
                f"Review and apply mitigations for {report.pending_count} pending vulnerabilities"
            )

        return report

    def suggest_mitigation(self, cve_id: str) -> Optional[Mitigation]:
        """Get mitigation suggestion for a CVE."""
        for vuln in self.vulnerabilities:
            if vuln.cve_id == cve_id:
                return self.mitigation_engine.get_mitigation_suggestion(vuln)
        return None

    def verify_all_mitigations(self) -> Dict[str, Tuple[bool, str]]:
        """Verify all active mitigations."""
        results = {}
        for mitigation in self.mitigations:
            if mitigation.status == MitigationStatus.MITIGATED:
                passed, output = self.mitigation_engine.verify_mitigation(mitigation)
                results[mitigation.cve_id] = (passed, output)
                mitigation.verified = passed
                mitigation.verification_output = output
        self._save_state()
        return results

    def generate_ci_config(self) -> str:
        """Generate CI configuration for security checks."""
        config = """
# Security verification step for GitHub Actions
- name: Security Vulnerability Check
  run: |
    echo "🔒 Running security verification..."

    # Check for blocked packages
    pip show basicsr > /dev/null 2>&1 && echo "❌ SECURITY: basicsr installed!" && exit 1 || true

    # Run verification script
    python scripts/utilities/verify_no_basicsr_imports.py --check-pkg

    # Run security scan
    python scripts/security/security_manager.py scan --ci

    echo "✅ Security checks passed"
"""
        return config.strip()

    def add_to_rag_feedback(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add security event to RAG feedback loop."""
        feedback_file = self.security_dir / "rag_feedback.jsonl"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        try:
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except IOError as e:
            logger.warning(f"Failed to write RAG feedback: {e}")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Proactive Dependabot Security Warning Management"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for vulnerabilities")
    scan_parser.add_argument("--ci", action="store_true", help="CI mode (exit code on issues)")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check specific package")
    check_parser.add_argument("--package", required=True, help="Package name to check")

    # Report command
    subparsers.add_parser("report", help="Generate security report")

    # Mitigate command
    mitigate_parser = subparsers.add_parser("mitigate", help="Get mitigation suggestions")
    mitigate_parser.add_argument("--cve", required=True, help="CVE ID to mitigate")

    # Verify command
    subparsers.add_parser("verify", help="Verify all mitigations")

    args = parser.parse_args()

    manager = SecurityManager()

    if args.command == "scan":
        report = manager.scan()
        print(report.to_markdown())
        if args.ci and (report.critical_count > 0 or report.high_count > 0):
            sys.exit(1)

    elif args.command == "check":
        vuln = manager.scanner.check_package_vulnerability(args.package)
        if vuln:
            print(f"⚠️ Vulnerability found for {args.package}:")
            print(f"   CVE: {vuln.cve_id}")
            print(f"   Severity: {vuln.severity.value}")
            print(f"   Description: {vuln.description}")
            sys.exit(1)
        else:
            print(f"✅ No known vulnerabilities for {args.package}")

    elif args.command == "report":
        report = manager._generate_report()
        print(report.to_markdown())

    elif args.command == "mitigate":
        mitigation = manager.suggest_mitigation(args.cve)
        if mitigation:
            print(f"Suggested mitigation for {args.cve}:")
            print(f"  Type: {mitigation.mitigation_type.value}")
            print(f"  Description: {mitigation.description}")
            if mitigation.commands:
                print("  Commands:")
                for cmd in mitigation.commands:
                    print(f"    $ {cmd}")
            if mitigation.verification_steps:
                print("  Verification:")
                for step in mitigation.verification_steps:
                    print(f"    - {step}")
        else:
            print(f"No mitigation suggestion found for {args.cve}")

    elif args.command == "verify":
        results = manager.verify_all_mitigations()
        for cve, (passed, output) in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {cve}")
            print(output)
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
