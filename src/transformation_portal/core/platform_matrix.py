"""Platform matrix definition and detection for deterministic ML reproducibility.

This module provides explicit platform identification for the ML dependency system,
enabling deterministic, reproducible builds across environments.

Platform Matrix (Axes):
    - OS: Darwin (macOS), Linux
    - ISA: arm64 (Apple Silicon/ARM), x86_64 (Intel/AMD)
    - Accel: cpu, mps (Apple Metal), cuda (NVIDIA)

Canonical Platform Targets:
    - darwin-x86_64-cpu   (macOS Intel)
    - darwin-arm64-cpu    (macOS Apple Silicon, CPU-only)
    - darwin-arm64-mps    (macOS Apple Silicon, Metal)
    - linux-x86_64-cpu    (Linux Intel/AMD, CPU baseline)
    - linux-x86_64-cuda   (Linux Intel/AMD, NVIDIA GPU)
    - linux-arm64-cpu     (Linux ARM, CPU-only)

Design Principles (ADR-032):
    - Accel is NEVER inferred from OS - always explicit via profile
    - OS and ISA are detected from platform module
    - Platform fingerprint is included in CAS key for reproducibility
    - Environment fingerprint (pip freeze hash) enables drift detection

Example:
    >>> matrix = PlatformMatrix.detect()
    >>> matrix.canonical_target
    'darwin-arm64-cpu'
    >>> matrix.to_dict()
    {'os': 'Darwin', 'isa': 'arm64', 'accel': 'cpu'}
    >>> fingerprint = get_env_fingerprint()
    'sha256:abc123...'
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class PlatformOS(str, Enum):
    """Supported operating systems."""

    DARWIN = "Darwin"
    LINUX = "Linux"

    @classmethod
    def detect(cls) -> "PlatformOS":
        """Detect current OS from platform module."""
        system = platform.system()
        if system == "Darwin":
            return cls.DARWIN
        elif system == "Linux":
            return cls.LINUX
        else:
            raise ValueError(f"Unsupported operating system: {system} (expected Darwin or Linux)")


class PlatformISA(str, Enum):
    """Supported instruction set architectures."""

    ARM64 = "arm64"
    X86_64 = "x86_64"

    @classmethod
    def detect(cls) -> "PlatformISA":
        """Detect current ISA from platform module."""
        machine = platform.machine().lower()
        # Normalize architecture names
        if machine in ("arm64", "aarch64"):
            return cls.ARM64
        elif machine in ("x86_64", "amd64"):
            return cls.X86_64
        else:
            raise ValueError(f"Unsupported processor architecture: {machine} (expected arm64/aarch64 or x86_64/amd64)")


class PlatformAccel(str, Enum):
    """Supported acceleration backends."""

    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"

    @classmethod
    def default_for_platform(cls, os: PlatformOS, isa: PlatformISA) -> "PlatformAccel":
        """Return conservative default acceleration (always CPU).

        IMPORTANT: Accel is explicit via profile, never auto-detected.
        This method returns CPU as the safe default when no profile is specified.
        """
        return cls.CPU


ML_CORE_LOCKFILE_BY_PLATFORM: dict[tuple[PlatformOS, PlatformISA], str] = {
    (PlatformOS.DARWIN, PlatformISA.ARM64): "ml-core-darwin-arm64.txt",
}


@dataclass(frozen=True)
class PlatformMatrix:
    """Immutable platform identification for reproducibility.

    Attributes:
        os: Operating system (Darwin, Linux)
        isa: Instruction set architecture (arm64, x86_64)
        accel: Acceleration backend (cpu, mps, cuda) - explicit, not inferred

    The canonical_target property produces strings like:
        - darwin-arm64-mps
        - linux-x86_64-cuda
        - darwin-x86_64-cpu
    """

    os: PlatformOS
    isa: PlatformISA
    accel: PlatformAccel = field(default=PlatformAccel.CPU)

    @classmethod
    def detect(cls, accel: Optional[str] = None) -> "PlatformMatrix":
        """Detect current platform matrix.

        Args:
            accel: Explicit acceleration profile ("cpu", "mps", "cuda").
                   If None, defaults to "cpu" (safe baseline).

        Returns:
            PlatformMatrix with detected OS/ISA and specified acceleration.

        Raises:
            ValueError: If OS or ISA is unsupported, or accel is invalid.
        """
        os = PlatformOS.detect()
        isa = PlatformISA.detect()

        if accel is not None:
            try:
                accel_enum = PlatformAccel(accel)
            except ValueError:
                valid = [a.value for a in PlatformAccel]
                raise ValueError(f"Invalid acceleration '{accel}', must be one of {valid}")
        else:
            accel_enum = PlatformAccel.default_for_platform(os, isa)

        return cls(os=os, isa=isa, accel=accel_enum)

    @property
    def canonical_target(self) -> str:
        """Canonical platform target string.

        Format: {os}-{isa}-{accel} (lowercased)

        Examples:
            darwin-arm64-mps
            linux-x86_64-cuda
            darwin-x86_64-cpu
        """
        return f"{self.os.value.lower()}-{self.isa.value}-{self.accel.value}"

    @property
    def is_macos(self) -> bool:
        """True if running on macOS."""
        return self.os == PlatformOS.DARWIN

    @property
    def is_linux(self) -> bool:
        """True if running on Linux."""
        return self.os == PlatformOS.LINUX

    @property
    def is_apple_silicon(self) -> bool:
        """True if running on Apple Silicon (macOS ARM64)."""
        return self.os == PlatformOS.DARWIN and self.isa == PlatformISA.ARM64

    @property
    def is_macos_intel(self) -> bool:
        """True if running on macOS Intel (x86_64)."""
        return self.os == PlatformOS.DARWIN and self.isa == PlatformISA.X86_64

    def check_ml_security_posture(self) -> dict[str, Any]:
        """Check ML security posture for this platform.

        Returns:
            Dictionary with security status:
            - platform: Platform canonical target
            - cve_2025_32434_note: Security note for CVE-2025-32434
            - mitigation: Required mitigation steps
            - secure: True if platform has secure torch wheels available
            - ml_supported: True if ML stack is supported on this platform

        Security Context:
            The supported Apple Silicon lane now pins torch==2.13.0.
            Linux and macOS Intel ML locks are retired unsupported manifests and
            are not installable checked-in requirement files.
            All platforms must use weights_only=True for torch.load() calls as
            defense in depth.
        """
        base_mitigation = (
            "All torch.load() calls must use weights_only=True as defense in depth. "
            "Use transformation_portal.core.security.torch_security.safe_load() "
            "or explicitly pass weights_only=True."
        )

        if self.is_macos_intel:
            return {
                "platform": self.canonical_target,
                "cve_2025_32434_note": (
                    "macOS Intel (x86_64) ML lockfiles are retired unsupported manifests. "
                    "No checked-in macOS Intel ML lock is supported or installable for ML workloads. "
                    "weights_only=True remains required hardening but is not a remediation for retired baselines."
                ),
                "mitigation": base_mitigation,
                "secure": False,  # macOS Intel has no supported torch>=2.13.0 baseline.
                "ml_supported": False,
            }
        if self.is_apple_silicon:
            return {
                "platform": self.canonical_target,
                "cve_2025_32434_note": (
                    "Supported Apple Silicon lane uses torch==2.13.0. "
                    "weights_only=True remains required for checkpoint defense in depth."
                ),
                "mitigation": base_mitigation,
                "secure": True,
                "ml_supported": True,
            }
        else:
            return {
                "platform": self.canonical_target,
                "cve_2025_32434_note": (
                    "Linux ML lockfiles are retired unsupported manifests. "
                    "Use a repo-managed secure subprocess runtime instead of a checked-in Linux ML core lock."
                ),
                "mitigation": base_mitigation,
                "secure": False,
                "ml_supported": False,
            }

    def assert_ml_supported(self) -> None:
        """Assert that ML stack is supported on this platform.

        Raises:
            RuntimeError: If ML stack is not supported (e.g., macOS Intel)

        Call this at ML stack initialization to enforce security policy.
        This is a HARD FAIL, not a warning.
        """
        status = self.check_ml_security_posture()
        if not status.get("ml_supported", True):
            raise RuntimeError(
                f"ML stack not supported on {status['platform']}. "
                f"{status['cve_2025_32434_note']} "
                "Migrate to macOS Apple Silicon or a repo-managed secure subprocess runtime."
            )

    def warn_if_insecure_ml_platform(self) -> None:
        """Emit warning if ML stack has known security considerations.

        Call this when initializing ML workloads to alert users about
        security hardening required for checkpoint loading.

        Note: This is a warning, not a hard fail. Use assert_ml_supported()
        for enforcement.
        """
        import warnings

        status = self.check_ml_security_posture()
        if not status["secure"]:
            warnings.warn(
                f"[{status['platform']}] {status['cve_2025_32434_note']} " f"Mitigation: {status['mitigation']}",
                UserWarning,
                stacklevel=2,
            )

    @property
    def supports_mps(self) -> bool:
        """True if platform can use MPS acceleration.

        MPS is only available on macOS ARM64 (Apple Silicon).
        """
        return self.is_apple_silicon

    @property
    def supports_cuda(self) -> bool:
        """True if platform can potentially use CUDA.

        CUDA is only available on Linux x86_64 (with NVIDIA GPU).
        This is a static check - actual GPU availability requires runtime detection.
        """
        return self.os == PlatformOS.LINUX and self.isa == PlatformISA.X86_64

    def to_dict(self) -> Dict[str, str]:
        """Export as dictionary for JSON serialization."""
        return {
            "os": self.os.value,
            "isa": self.isa.value,
            "accel": self.accel.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "PlatformMatrix":
        """Reconstruct from dictionary."""
        return cls(
            os=PlatformOS(data["os"]),
            isa=PlatformISA(data["isa"]),
            accel=PlatformAccel(data["accel"]),
        )

    def __str__(self) -> str:
        return self.canonical_target


def _normalize_pip_freeze(freeze_output: str) -> str:
    """Normalize pip freeze output for deterministic hashing.

    Normalization rules:
    1. Exclude editable installs (-e packages) - these contain local paths
    2. Sort packages alphabetically (pip freeze order is not deterministic)
    3. Strip whitespace and empty lines
    4. Handle both Unix (LF) and Windows (CRLF) line endings

    Args:
        freeze_output: Raw output from `pip freeze`

    Returns:
        Normalized, sorted package list as a single string
    """
    # Use splitlines() to handle both LF and CRLF
    lines = freeze_output.strip().splitlines()
    # Filter out editable installs and empty lines
    packages = [line.strip() for line in lines if line.strip() and not line.strip().startswith("-e")]
    # Sort for deterministic ordering
    packages.sort()
    return "\n".join(packages)


def get_env_fingerprint() -> str:
    """Compute SHA256 fingerprint of the current pip environment.

    Returns:
        Environment fingerprint in format "sha256:..."

    This fingerprint can be used in CAS keys to ensure identical
    environments produce identical results. Environment drift
    (package updates, version changes) will change the fingerprint,
    invalidating cached artifacts.

    Note:
        Uses normalized `pip freeze` output for cross-platform consistency.
        Normalization excludes editable installs and sorts packages
        to ensure identical environments produce identical fingerprints.
        If pip is unavailable, returns a placeholder fingerprint.
    """
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            # Normalize pip freeze output for deterministic hashing
            normalized = _normalize_pip_freeze(result.stdout)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            return f"sha256:{digest}"
        else:
            # pip freeze failed - return placeholder
            return "sha256:unknown-pip-freeze-failed"
    except subprocess.TimeoutExpired:
        # pip freeze timed out
        return "sha256:unknown-pip-timeout"
    except (FileNotFoundError, OSError):
        # pip not available
        return "sha256:unknown-pip-unavailable"


def get_pip_version() -> str:
    """Get the current pip version.

    Returns:
        Pip version string, or "unknown" if unavailable.
    """
    try:
        result = subprocess.run(
            ["pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            # Output format: "pip X.Y.Z from ..."
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return parts[1]
        return "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"


def compute_lockfile_hash(lockfile_path: str) -> str:
    """Compute SHA256 hash of a lockfile.

    Args:
        lockfile_path: Path to the lockfile (e.g., "ml-core-darwin-arm64.txt")

    Returns:
        Hash in format "sha256:..." or "sha256:unknown" if file not found.
    """
    from pathlib import Path

    try:
        path = Path(lockfile_path)
        if path.is_file():
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            return f"sha256:{digest}"
        return "sha256:unknown-file-not-found"
    except (OSError, IOError):
        return "sha256:unknown-read-error"


def determine_ml_core_lockfile_name(matrix: Optional[PlatformMatrix] = None) -> str:
    """Return the canonical ML core lockfile name for a platform matrix."""
    effective_matrix = matrix or CURRENT_PLATFORM
    if effective_matrix is None:
        raise ValueError("Cannot determine ML core lockfile for unsupported platform")

    try:
        return ML_CORE_LOCKFILE_BY_PLATFORM[(effective_matrix.os, effective_matrix.isa)]
    except KeyError as exc:
        raise ValueError(
            "No supported checked-in ML core lockfile contract for platform "
            f"{effective_matrix.os.value}/{effective_matrix.isa.value}"
        ) from exc


def get_platform_fingerprint(
    accel: Optional[str] = None,
    lockfile_path: Optional[str] = None,
    include_security_profile: bool = True,
) -> Dict[str, Any]:
    """Get comprehensive platform fingerprint for artifact provenance.

    This fingerprint includes all information needed for CAS identity:
    - Platform matrix (os, arch, accel)
    - Python version
    - Pip version
    - Environment fingerprint (pip freeze hash)
    - Lockfile hash (if provided)
    - Security profile hash (CVE mitigation status)

    Args:
        accel: Explicit acceleration profile ("cpu", "mps", "cuda").
               If None, defaults to "cpu".
        lockfile_path: Path to the lockfile used for installation.
                       If provided, its hash is included in the fingerprint.
        include_security_profile: If True, include security profile in fingerprint.
                                  Default True for CAS identity completeness.

    Returns:
        Dictionary with complete platform identity for CAS.
    """
    import sys

    matrix = PlatformMatrix.detect(accel)

    fingerprint: Dict[str, Any] = {
        "platform_id": matrix.canonical_target,
        "platform": matrix.to_dict(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "pip_version": get_pip_version(),
        "env_fingerprint": get_env_fingerprint(),
    }

    if lockfile_path:
        fingerprint["lockfile_hash"] = compute_lockfile_hash(lockfile_path)

    # Include security profile for CAS identity (ADR-032 security layer)
    if include_security_profile:
        fingerprint["security_profile"] = get_security_profile()

    return fingerprint


def get_security_profile() -> Dict[str, Any]:
    """Get current security profile for CAS identity.

    Returns:
        Dictionary with CANONICAL security profile information.
        Uses STATIC policy values only - no runtime-derived state.

    CRITICAL: This function returns canonical, deterministic values
    that are the same regardless of:
    - Import order
    - Runtime enforcement state
    - Process initialization timing

    This ensures artifacts from different security configurations
    (e.g., with/without torch.load enforcement) are not mixed,
    while maintaining deterministic CAS identity computation.
    """
    try:
        from transformation_portal.core.security.torch_security import (
            SECURITY_PROFILE_VERSION,
            get_canonical_security_profile,
            get_security_profile_hash,
        )

        # Use canonical profile (static policy, not runtime state)
        canonical = get_canonical_security_profile()
        return {
            "version": SECURITY_PROFILE_VERSION,
            "policy": canonical,
            "profile_hash": get_security_profile_hash(),
        }
    except ImportError:
        # torch_security module not available - use deterministic fallback
        return {
            "version": "unavailable",
            "policy": {"torch_load_enforced": False, "weights_only": False},
            "profile_hash": "sha256:module-unavailable",
        }


def compute_cas_identity(
    accel: Optional[str] = None,
    lockfile_path: Optional[str] = None,
    include_security_profile: bool = True,
) -> str:
    """Compute a single CAS identity string from platform fingerprint.

    This produces a deterministic hash that can be used as part of a CAS key
    to ensure reproducibility. If the platform, environment, lockfile, or
    security profile changes, the CAS identity will change, invalidating
    cached artifacts.

    Args:
        accel: Explicit acceleration profile ("cpu", "mps", "cuda").
        lockfile_path: Path to the lockfile used for installation.
        include_security_profile: If True, include security profile in identity.
                                  Default True for complete security posture.

    Returns:
        CAS identity in format "sha256:..."
    """
    import json

    fingerprint = get_platform_fingerprint(accel, lockfile_path, include_security_profile)
    # Add schema version for evolvability - old artifacts are invalidated on schema change
    fingerprint["cas_schema_version"] = CAS_IDENTITY_SCHEMA_VERSION
    # Sort keys for deterministic serialization
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# CAS Identity Schema Version (ADR-032)
# Increment this when CAS identity computation changes to invalidate old artifacts
# v1: Initial schema (platform_id, env_fingerprint, lockfile_hash)
# v2: Added security_profile for checkpoint-loading security tracking
CAS_IDENTITY_SCHEMA_VERSION = "adr-032-v2"


# Pre-compute current platform at module load for fast access
try:
    CURRENT_PLATFORM = PlatformMatrix.detect()
except ValueError:
    # Fallback for unsupported platforms (e.g., Windows)
    CURRENT_PLATFORM = None
