#!/usr/bin/env python3
"""
Unified pre-flight validation script for local development environment.

Validates that all required tools and dependencies are available before
running validation targets. Designed to fail fast with actionable guidance.

Exit codes:
    0 - All checks passed
    1 - Soft failure: environment is usable but suboptimal
    2 - Hard failure: environment cannot run validation targets

Usage:
    make check-environment
    .venv/bin/python scripts/validation/check_local_environment.py
    .venv/bin/python scripts/validation/check_local_environment.py --strict
    .venv/bin/python scripts/validation/check_local_environment.py --check python
    .venv/bin/python scripts/validation/check_local_environment.py --check node
    .venv/bin/python scripts/validation/check_local_environment.py --check chrome
    .venv/bin/python scripts/validation/check_local_environment.py --check ports
    .venv/bin/python scripts/validation/check_local_environment.py --check dependency-health
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Optional


class ExitCode(IntEnum):
    """Exit codes for pre-flight validation."""

    PASS = 0
    SOFT_FAIL = 1
    HARD_FAIL = 2


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    passed: bool
    message: str
    is_hard_requirement: bool = True
    guidance: Optional[str] = None


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTDOOR_ROOT = REPO_ROOT / "web" / "secure-landing"
# Platform-aware venv interpreter path
if sys.platform == "win32":
    REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    _VENV_PYTHON_REL = r".venv\Scripts\python.exe"
    _VENV_ACTIVATE_CMD = r".venv\Scripts\activate"
else:
    REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
    _VENV_PYTHON_REL = ".venv/bin/python"
    _VENV_ACTIVATE_CMD = "source .venv/bin/activate"

# Required Python version
MIN_PYTHON_VERSION = (3, 11)

# Required Node version (major only for frontdoor)
REQUIRED_NODE_MAJOR = 22

# Ports used by validation targets
VALIDATION_PORTS = {
    3000: "managed frontdoor (Next.js)",
    8000: "FastAPI backend",
}

# Environment variables checked (not required, but reported)
CHECKED_ENV_VARS = [
    "TP_API_KEY",
    "TP_FRONTDOOR_USERS_FILE",
    "TP_FRONTDOOR_USERNAME",
    "TP_FRONTDOOR_PASSWORD",
]

DA3_CONTAMINATION_MARKERS = (
    "depth-anything-3",
    "xformers",
    "numpy<2",
)


def check_python_version() -> CheckResult:
    """Check if Python version meets minimum requirements."""
    current = sys.version_info[:2]
    passed = current >= MIN_PYTHON_VERSION
    min_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
    current_str = f"{current[0]}.{current[1]}"

    if passed:
        return CheckResult(
            name="Python version",
            passed=True,
            message=f"Python {current_str} meets requirement (>={min_str})",
        )
    else:
        return CheckResult(
            name="Python version",
            passed=False,
            message=f"Python {current_str} does not meet requirement (>={min_str})",
            guidance=f"Install Python {min_str}+ via pyenv, system package manager, or python.org",
        )


def check_node_version() -> CheckResult:
    """Check if Node.js version meets frontdoor requirements."""
    node_path = shutil.which("node")
    if not node_path:
        return CheckResult(
            name="Node.js version",
            passed=False,
            message="Node.js is not installed or not in PATH",
            guidance="Install Node.js 22.x via nvm, fnm, volta, or nodejs.org",
        )

    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version_str = result.stdout.strip()
        # Parse version like "v22.5.0"
        match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", version_str)
        if not match:
            return CheckResult(
                name="Node.js version",
                passed=False,
                message=f"Could not parse Node.js version: {version_str}",
                guidance="Ensure 'node --version' returns a valid version string",
            )

        major = int(match.group(1))
        if major == REQUIRED_NODE_MAJOR:
            return CheckResult(
                name="Node.js version",
                passed=True,
                message=f"Node.js {version_str} meets requirement ({REQUIRED_NODE_MAJOR}.x)",
            )
        else:
            return CheckResult(
                name="Node.js version",
                passed=False,
                message=f"Node.js {version_str} does not match required {REQUIRED_NODE_MAJOR}.x",
                guidance=f"Switch to Node {REQUIRED_NODE_MAJOR}.x: nvm use {REQUIRED_NODE_MAJOR} / fnm use "
                f"{REQUIRED_NODE_MAJOR}",
            )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Node.js version",
            passed=False,
            message="Node.js version check timed out",
            guidance="Ensure 'node --version' completes quickly",
        )
    except Exception as e:
        return CheckResult(
            name="Node.js version",
            passed=False,
            message=f"Error checking Node.js version: {e}",
        )


def check_chrome_available() -> CheckResult:
    """Check if Chrome/Chromium is available for browser smoke tests."""
    chrome_candidates = [
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        # Linux
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        # Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    # Also check TP_PORTAL_BROWSER_BINARY env var
    env_binary = os.environ.get("TP_PORTAL_BROWSER_BINARY")
    if env_binary:
        chrome_candidates.insert(0, env_binary)

    for candidate in chrome_candidates:
        if "/" in candidate or "\\" in candidate:
            # Absolute path
            if Path(candidate).exists():
                return CheckResult(
                    name="Chrome/Chromium",
                    passed=True,
                    message=f"Found browser at: {candidate}",
                    is_hard_requirement=False,
                )
        else:
            # Command name, check in PATH
            if shutil.which(candidate):
                return CheckResult(
                    name="Chrome/Chromium",
                    passed=True,
                    message=f"Found browser: {candidate}",
                    is_hard_requirement=False,
                )

    return CheckResult(
        name="Chrome/Chromium",
        passed=False,
        message="Chrome or Chromium not found (required for browser smoke tests)",
        is_hard_requirement=False,
        guidance="Install Chrome/Chromium or set TP_PORTAL_BROWSER_BINARY env var",
    )


def check_port_available(port: int, description: str) -> CheckResult:
    """Check if a specific port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                # Port is in use
                return CheckResult(
                    name=f"Port {port}",
                    passed=False,
                    message=f"Port {port} is in use ({description})",
                    is_hard_requirement=False,
                    guidance=f"Stop the service using port {port} or use a different port",
                )
            else:
                # Port is available
                return CheckResult(
                    name=f"Port {port}",
                    passed=True,
                    message=f"Port {port} is available ({description})",
                    is_hard_requirement=False,
                )
    except Exception as e:
        return CheckResult(
            name=f"Port {port}",
            passed=True,
            message=f"Could not check port {port} (assuming available): {e}",
            is_hard_requirement=False,
            guidance="Manually verify port availability if validation fails",
        )


def check_ports_available() -> list[CheckResult]:
    """Check if all validation ports are available."""
    return [check_port_available(port, desc) for port, desc in VALIDATION_PORTS.items()]


def check_frontdoor_dependencies() -> CheckResult:
    """Check if frontdoor npm dependencies are installed."""
    node_modules = FRONTDOOR_ROOT / "node_modules"
    if not node_modules.exists():
        return CheckResult(
            name="Frontdoor dependencies",
            passed=False,
            message="node_modules not found in web/secure-landing/",
            is_hard_requirement=False,
            guidance="Run: cd web/secure-landing && npm install",
        )

    # Check for key dependencies
    key_deps = ["next", "better-sqlite3", "argon2"]
    missing = [dep for dep in key_deps if not (node_modules / dep).exists()]
    if missing:
        return CheckResult(
            name="Frontdoor dependencies",
            passed=False,
            message=f"Missing npm packages: {', '.join(missing)}",
            is_hard_requirement=False,
            guidance="Run: cd web/secure-landing && npm install",
        )

    return CheckResult(
        name="Frontdoor dependencies",
        passed=True,
        message="Frontdoor npm dependencies installed",
        is_hard_requirement=False,
    )


def check_env_vars() -> list[CheckResult]:
    """Check for common environment variables (advisory only)."""
    results = []
    # Variables that should be masked when displaying their values
    sensitive_patterns = ("KEY", "PASSWORD", "SECRET", "TOKEN", "CREDENTIAL")

    for var in CHECKED_ENV_VARS:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values based on variable name patterns
            is_sensitive = any(pat in var.upper() for pat in sensitive_patterns)
            display_value = "****" if is_sensitive else value[:20]
            results.append(
                CheckResult(
                    name=f"Env: {var}",
                    passed=True,
                    message=f"Set to: {display_value}",
                    is_hard_requirement=False,
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"Env: {var}",
                    passed=True,  # Not having these is fine, they have defaults
                    message="Not set (defaults will be used)",
                    is_hard_requirement=False,
                )
            )
    return results


def _is_virtualenv_interpreter() -> bool:
    """Return whether the active interpreter is running from a virtualenv."""
    if os.environ.get("VIRTUAL_ENV"):
        return True
    if hasattr(sys, "real_prefix"):
        return True
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def _repo_venv_matches_current_interpreter() -> bool:
    """Return whether the active interpreter matches the repo-local .venv."""
    if not REPO_VENV_PYTHON.exists():
        return False

    current_executable = Path(sys.executable)
    if not current_executable.exists():
        return False

    try:
        return current_executable.samefile(REPO_VENV_PYTHON)
    except OSError:
        return False


def check_venv_active() -> CheckResult:
    """Check if a virtual environment is active."""
    if REPO_VENV_PYTHON.exists() and not _repo_venv_matches_current_interpreter():
        return CheckResult(
            name="Python venv",
            passed=False,
            message=(f"Repo venv exists at {REPO_VENV_PYTHON}, but the current interpreter is {sys.executable}"),
            guidance=(
                f"Use `{_VENV_PYTHON_REL} ...` or `{_VENV_ACTIVATE_CMD}`. "
                "If `.venv` is broken, run `make repair-core-venv`."
            ),
        )

    if _is_virtualenv_interpreter():
        return CheckResult(
            name="Python venv",
            passed=True,
            message=f"Active: {os.environ.get('VIRTUAL_ENV') or sys.prefix}",
            is_hard_requirement=False,
        )

    return CheckResult(
        name="Python venv",
        passed=True,
        message="No venv active (using system Python)",
        is_hard_requirement=False,
        guidance=f"Consider: make venv && {_VENV_ACTIVATE_CMD}",
    )


def _summarize_pip_check_output(output: str) -> str:
    """Collapse pip check output into a compact single-line summary."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "pip check reported dependency issues"
    return "; ".join(lines[:2])


def _dependency_health_guidance(output: str) -> str:
    """Return actionable dependency-health guidance."""
    combined = output.lower()
    if any(marker in combined for marker in DA3_CONTAMINATION_MARKERS):
        return (
            "Repair the core repo environment with `make repair-core-venv`, then reinstall "
            "the isolated DA3 runtime with `./scripts/setup/install_da3_runtime.sh`."
        )
    return "Repair the repo environment with `make repair-core-venv`, then rerun this pre-flight check."


def check_dependency_health() -> CheckResult:
    """Check whether the active interpreter has a consistent dependency graph."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Python dependency health",
            passed=False,
            message="`python -m pip check` timed out",
            guidance="Repair the repo environment with `make repair-core-venv`, then retry.",
        )
    except Exception as exc:
        return CheckResult(
            name="Python dependency health",
            passed=False,
            message=f"Could not run `{sys.executable} -m pip check`: {exc}",
            guidance="Repair the repo environment with `make repair-core-venv`, then retry.",
        )

    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0:
        return CheckResult(
            name="Python dependency health",
            passed=True,
            message=f"`{sys.executable} -m pip check` passed",
        )

    return CheckResult(
        name="Python dependency health",
        passed=False,
        message=_summarize_pip_check_output(combined_output),
        guidance=_dependency_health_guidance(combined_output),
    )


def run_all_checks(
    checks: Optional[list[str]] = None,
) -> tuple[list[CheckResult], ExitCode]:
    """Run all or specified pre-flight checks."""
    results: list[CheckResult] = []

    check_map: dict[str, Callable[[], CheckResult | list[CheckResult]]] = {
        "python": check_python_version,
        "node": check_node_version,
        "chrome": check_chrome_available,
        "ports": check_ports_available,
        "frontdoor": check_frontdoor_dependencies,
        "venv": check_venv_active,
        "dependency-health": check_dependency_health,
        "env": check_env_vars,
    }

    if checks:
        # Run only specified checks
        for check_name in checks:
            if check_name in check_map:
                result = check_map[check_name]()
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
    else:
        # Run all checks
        for check_func in check_map.values():
            result = check_func()
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

    # Determine exit code
    hard_failures = [r for r in results if not r.passed and r.is_hard_requirement]
    soft_failures = [r for r in results if not r.passed and not r.is_hard_requirement]

    if hard_failures:
        return results, ExitCode.HARD_FAIL
    elif soft_failures:
        return results, ExitCode.SOFT_FAIL
    else:
        return results, ExitCode.PASS


def print_results(results: list[CheckResult], exit_code: ExitCode) -> None:
    """Print check results in a human-readable format."""
    print("\n" + "=" * 60)
    print("  Transformation Portal — Local Environment Pre-flight Check")
    print("=" * 60 + "\n")

    # Group results
    passed = [r for r in results if r.passed]

    for result in results:
        status = "✓" if result.passed else "✗"
        req = "" if result.is_hard_requirement else " (optional)"
        print(f"  {status} {result.name}{req}")
        print(f"      {result.message}")
        if result.guidance and not result.passed:
            print(f"      → {result.guidance}")
        print()

    # Summary
    print("-" * 60)
    print(f"  Passed: {len(passed)}/{len(results)}")

    if exit_code == ExitCode.PASS:
        print("  Status: ✓ All checks passed — ready for validation")
    elif exit_code == ExitCode.SOFT_FAIL:
        print("  Status: ⚠ Some optional checks failed — validation may be limited")
    else:
        print("  Status: ✗ Hard requirements not met — cannot run validation")

    print("=" * 60 + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for local development environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                 Run all checks
    %(prog)s --strict        Fail on any check failure (including optional)
    %(prog)s --check python  Check only Python version
    %(prog)s --check node    Check only Node.js version
    %(prog)s --check chrome  Check only Chrome availability
    %(prog)s --check ports   Check only port availability
    %(prog)s --check dependency-health  Check pip dependency health
    %(prog)s --quiet         Only output failures
        """,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat soft failures as hard failures",
    )
    parser.add_argument(
        "--check",
        choices=["python", "node", "chrome", "ports", "frontdoor", "venv", "dependency-health", "env"],
        action="append",
        help="Run only specified check(s)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output failures",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    results, exit_code = run_all_checks(args.check)

    if args.strict and exit_code == ExitCode.SOFT_FAIL:
        exit_code = ExitCode.HARD_FAIL

    if args.json:
        import json

        output = {
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "is_hard_requirement": r.is_hard_requirement,
                    "guidance": r.guidance,
                }
                for r in results
            ],
            "exit_code": int(exit_code),
            "status": (
                "pass" if exit_code == ExitCode.PASS else ("soft_fail" if exit_code == ExitCode.SOFT_FAIL else "hard_fail")
            ),
        }
        print(json.dumps(output, indent=2))
    elif args.quiet:
        failed = [r for r in results if not r.passed]
        if failed:
            for result in failed:
                print(f"✗ {result.name}: {result.message}")
                if result.guidance:
                    print(f"  → {result.guidance}")
    else:
        print_results(results, exit_code)

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
