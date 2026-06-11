"""PyTorch security utilities for checkpoint loading hardening.

This module provides safe model loading functions that reduce checkpoint
loading risk and preserve a single repository-wide torch.load() policy.

MITIGATION STRATEGY:
    Supported repository lanes require torch >= 2.12.0. Runtime hardening still
    enforces weights_only=True for all torch.load() calls as defense in depth.
    Frozen historical lanes are not considered remediated by runtime hardening.

    Benefits:
    - Keeps supported lanes on a patched PyTorch baseline
    - Reduces checkpoint deserialization risk during model loading
    - Keeps managed/model-loading trust boundaries explicit
    - Maintains a single repository-wide torch.load() policy

CVE-2025-32434 Details:
    - Vulnerability: Remote Code Execution via torch.load()
    - CVSS Score: 9.8 (Critical)
    - Affected versions: torch < 2.6.0
    - Attack vector: Malicious .pt/.pth model files
    - Remediation: upgrade torch to a patched supported baseline

ENFORCEMENT:
    This module provides MANDATORY global enforcement of safe_load() via:
    - install_global_enforcement(): Patches torch.load to enforce weights_only=True
    - is_enforcement_installed(): Check if enforcement is active
    - SECURITY_PROFILE_VERSION: Version identifier for CAS identity

    Call install_global_enforcement() at ML stack initialization to ensure
    ALL torch.load() calls are protected, not just direct safe_load() usage.

Usage:
    >>> from transformation_portal.core.security.torch_security import (
    ...     safe_load, install_global_enforcement
    ... )
    >>> install_global_enforcement()  # Must be called at startup
    >>> state_dict = safe_load("model.pt", map_location="cpu")

Security Profile (CAS Identity):
    The SECURITY_PROFILE_VERSION is included in CAS identity to ensure
    artifacts from different security configurations are not mixed.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Optional, Union

# Security profile version for CAS identity
# Increment when security enforcement logic changes
SECURITY_PROFILE_VERSION = "torch_safe_load_v2"
MINIMUM_SUPPORTED_TORCH_VERSION = "2.12.0"
CVE_2025_32434_FIXED_TORCH_VERSION = "2.6.0"

# Track whether enforcement has been installed
_enforcement_installed = False
_original_torch_load = None


def safe_load(
    path: Union[str, Path],
    *,
    map_location: Optional[Any] = None,
) -> Any:
    """Safely load PyTorch model weights with checkpoint hardening.

    This function wraps torch.load() with weights_only=True to reduce
    deserialization risk from model files. Supported deployments must still
    use a patched torch baseline.

    Args:
        path: Path to the model file (.pt, .pth, etc.)
        map_location: Device mapping for loaded tensors (e.g., "cpu", "cuda:0")

    Returns:
        Loaded model state dict or weights

    Raises:
        FileNotFoundError: If model file does not exist
        RuntimeError: If loading fails
        pickle.UnpicklingError: If file contains non-tensor objects

    Example:
        >>> state_dict = safe_load("model.pt", map_location="cpu")
        >>> model.load_state_dict(state_dict)

    Security:
        This function enforces weights_only=True for defense in depth. Files
        containing custom classes or functions will fail to load. This is
        intentional for security.

        If you need to load a file with custom objects, you must:
        1. Verify the source is trusted
        2. Use torch.load() directly with explicit documentation
        3. Add security review comment explaining the trust boundary
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch is required for safe_load(). " "Install with: pip install torch") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return torch.load(path, map_location=map_location, weights_only=True)


def check_torch_security_compliance() -> dict[str, Any]:
    """Check if the torch installation meets security requirements.

    Returns:
        Dictionary with security compliance status:
        - torch_version: Installed torch version
        - cve_2025_32434_vulnerable: True if vulnerable to CVE-2025-32434
        - minimum_supported_torch_version: Repository supported baseline
        - supported_security_baseline_met: True if torch >= supported baseline
        - mitigation_available: True if weights_only mitigation is available
        - recommendation: Security recommendation string

    Example:
        >>> status = check_torch_security_compliance()
        >>> if not status["supported_security_baseline_met"]:
        ...     print("Upgrade PyTorch to the supported baseline")
    """
    try:
        import torch
        from packaging.version import Version
    except ImportError:
        return {
            "torch_version": None,
            "cve_2025_32434_vulnerable": None,
            "minimum_supported_torch_version": MINIMUM_SUPPORTED_TORCH_VERSION,
            "supported_security_baseline_met": False,
            "mitigation_available": False,
            "recommendation": "PyTorch not installed",
        }

    torch_version = torch.__version__.split("+")[0]  # Strip build suffix (+cpu, +cu118, etc.)

    try:
        version = Version(torch_version)
        vulnerable = version < Version(CVE_2025_32434_FIXED_TORCH_VERSION)
        supported_baseline_met = version >= Version(MINIMUM_SUPPORTED_TORCH_VERSION)
    except Exception:
        # Non-standard version string
        vulnerable = True
        supported_baseline_met = False

    return {
        "torch_version": torch_version,
        "cve_2025_32434_vulnerable": vulnerable,
        "minimum_supported_torch_version": MINIMUM_SUPPORTED_TORCH_VERSION,
        "supported_security_baseline_met": supported_baseline_met,
        "mitigation_available": True,
        "recommendation": (
            f"Upgrade PyTorch to >={MINIMUM_SUPPORTED_TORCH_VERSION}; "
            "weights_only=True or safe_load() remain mandatory defense in depth"
            if not supported_baseline_met
            else "Torch version meets the supported security baseline; keep weights_only=True or safe_load() for defense in depth"
        ),
    }


def warn_if_vulnerable() -> None:
    """Emit a warning if torch is below the supported security baseline.

    This can be called at module import time to alert users of vulnerable
    or unsupported torch installations.
    """
    status = check_torch_security_compliance()
    if not status.get("supported_security_baseline_met", False) and status.get("torch_version") is not None:
        warnings.warn(
            f"PyTorch {status['torch_version']} does not meet the supported "
            f"security baseline >= {MINIMUM_SUPPORTED_TORCH_VERSION}. "
            "Upgrade torch; weights_only=True and safe_load() remain mandatory "
            "checkpoint-loading hardening.",
            UserWarning,
            stacklevel=2,
        )


def _enforced_torch_load(
    f: Any,
    map_location: Any = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Enforced torch.load wrapper that mandates weights_only=True.

    This function is installed as a replacement for torch.load when
    global enforcement is active. It ensures all model loading uses
    weights_only=True for checkpoint-loading defense in depth.

    Args:
        f: File path or file-like object
        map_location: Device mapping for loaded tensors
        pickle_module: Deprecated - not supported when weights_only=True.
        weights_only: Must be True (enforced). Passing False raises SecurityError.
        **kwargs: Additional torch.load keyword arguments

    Returns:
        Loaded model data

    Raises:
        SecurityError: If weights_only=False is explicitly passed
        RuntimeError: If enforcement is not installed
    """
    if not _enforcement_installed or _original_torch_load is None:
        raise RuntimeError("Torch security enforcement not installed. " "Call install_global_enforcement() at startup.")

    # Warn if pickle_module is provided (not supported with weights_only=True)
    if pickle_module is not None:
        warnings.warn(
            "pickle_module parameter is not supported when weights_only=True. " "The parameter will be ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Block explicit weights_only=False
    if not weights_only:
        raise SecurityError(
            "CVE-2025-32434: weights_only=False is blocked for security. "
            "Use a supported torch baseline plus weights_only=True or safe_load() "
            "to load model files. "
            "If you must load untrusted custom objects, document the trust "
            "boundary and use _unsafe_torch_load_bypass() with security review."
        )

    # Call original torch.load with enforced weights_only=True
    return _original_torch_load(
        f,
        map_location=map_location,
        weights_only=True,
        **kwargs,
    )


class SecurityError(RuntimeError):
    """Raised when a security policy is violated.

    This exception indicates that code attempted to bypass security
    hardening, such as using torch.load() without weights_only=True.
    """

    pass


# Backward-compatible public name used by older callers and tests.
SecurityPolicyViolation = SecurityError


def install_global_enforcement() -> bool:
    """Install global enforcement of safe torch.load behavior.

    This function patches torch.load to enforce weights_only=True on ALL calls,
    not just direct safe_load() usage. Supported deployments must also run on
    a patched torch baseline.

    MUST be called at ML stack initialization before any model loading.

    Returns:
        True if enforcement was installed (or already active)
        False if torch is not available

    Example:
        >>> from transformation_portal.core.security.torch_security import (
        ...     install_global_enforcement
        ... )
        >>> install_global_enforcement()
        True
        >>> import torch
        >>> torch.load("model.pt")  # Now enforces weights_only=True

    Security:
        After calling this function:
        - torch.load() enforces weights_only=True automatically
        - torch.load(..., weights_only=False) raises SecurityError
        - CAS identity includes security_profile for auditability

    Note:
        This is idempotent - calling multiple times is safe.
    """
    global _enforcement_installed, _original_torch_load

    if _enforcement_installed:
        return True

    try:
        import torch
    except ImportError:
        return False

    # Save original torch.load before patching
    _original_torch_load = torch.load

    # Patch torch.load with enforced wrapper
    torch.load = _enforced_torch_load  # type: ignore[assignment]

    _enforcement_installed = True

    # Log enforcement installation (not warning - this is expected behavior)
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"Torch security enforcement installed (profile: {SECURITY_PROFILE_VERSION}). "
        "All torch.load() calls now enforce weights_only=True."
    )

    return True


def is_enforcement_installed() -> bool:
    """Check if global torch.load enforcement is active.

    Returns:
        True if install_global_enforcement() has been called successfully

    Example:
        >>> is_enforcement_installed()
        False
        >>> install_global_enforcement()
        True
        >>> is_enforcement_installed()
        True
    """
    return _enforcement_installed


def uninstall_global_enforcement() -> bool:
    """Remove global enforcement and restore original torch.load.

    WARNING: This should only be used in testing scenarios.
    Production code should NOT call this function.

    Returns:
        True if enforcement was removed
        False if enforcement was not installed

    Security:
        Calling this function removes checkpoint-loading hardening.
        Only use for testing or in controlled environments.
    """
    global _enforcement_installed, _original_torch_load

    if not _enforcement_installed:
        return False

    try:
        import torch
    except ImportError:
        return False

    if _original_torch_load is not None:
        torch.load = _original_torch_load  # type: ignore[assignment]
        _original_torch_load = None

    _enforcement_installed = False
    return True


def _unsafe_torch_load_bypass(
    f: Any,
    map_location: Any = None,
    *,
    _security_review_approved: bool = False,
    **kwargs: Any,
) -> Any:
    """Bypass security enforcement for trusted model files.

    DANGER: This function bypasses checkpoint-loading hardening.

    Only use this for:
    - Loading models with custom classes that CANNOT use weights_only=True
    - Files from FULLY TRUSTED sources (your own training pipeline)
    - Code that has undergone security review

    Args:
        f: File path or file-like object
        map_location: Device mapping for loaded tensors
        _security_review_approved: Must be True to confirm security review
        **kwargs: Additional torch.load keyword arguments

    Returns:
        Loaded model data

    Raises:
        SecurityError: If _security_review_approved is not True

    Security Review Checklist:
        Before using this function, verify:
        [ ] The model file source is fully trusted (your own pipeline)
        [ ] The file has not been modified by untrusted parties
        [ ] Custom classes in the file are safe and expected
        [ ] This usage is documented in security review notes
    """
    if not _security_review_approved:
        raise SecurityError(
            "Security review approval required. "
            "Set _security_review_approved=True after completing security review. "
            "See docstring for security review checklist."
        )

    # Use original torch.load if enforcement is installed, otherwise import fresh
    if _original_torch_load is not None:
        load_fn = _original_torch_load
    else:
        try:
            import torch

            load_fn = torch.load
        except ImportError as e:
            raise ImportError("PyTorch is required") from e

    return load_fn(f, map_location=map_location, **kwargs)


def get_security_profile_hash() -> str:
    """Get hash of current security profile for CAS identity.

    Returns:
        SHA256 hash of security profile configuration

    This hash should be included in CAS identity to ensure artifacts
    from different security configurations are not mixed.

    IMPORTANT: This function returns a CANONICAL hash based on static policy,
    NOT runtime state. This ensures deterministic CAS identity across:
    - Different import orders
    - Different enforcement timing
    - Multiple processes

    Example:
        >>> get_security_profile_hash()
        'sha256:abc123...'
    """
    # CRITICAL: Use ONLY static policy values, NOT runtime state
    # Runtime state (_enforcement_installed) would break determinism
    profile_data = {
        "policy_version": SECURITY_PROFILE_VERSION,
        "minimum_supported_torch_version": MINIMUM_SUPPORTED_TORCH_VERSION,
        "cve_2025_32434_posture": "fixed_by_supported_torch_baseline",
        "torch_load_policy": "weights_only_true",
    }
    # Use canonical JSON for deterministic hash (repo guardrail requires this)
    from transformation_portal.ingest.canonical_json import canonicalize_json

    profile_bytes = canonicalize_json(profile_data)
    digest = hashlib.sha256(profile_bytes).hexdigest()
    return f"sha256:{digest}"


def assert_enforcement_installed() -> None:
    """Assert that global torch.load enforcement is active.

    Raises:
        RuntimeError: If enforcement is not installed

    Use this at ML stack entry points to validate the security
    invariant that torch.load is patched before any model loading.

    Example:
        >>> install_global_enforcement()
        >>> assert_enforcement_installed()  # Passes
        >>> # If enforcement not installed:
        >>> assert_enforcement_installed()  # Raises RuntimeError
    """
    if not _enforcement_installed:
        raise RuntimeError(
            "Torch security enforcement not installed. "
            "Call install_global_enforcement() at ML stack initialization "
            "BEFORE importing any modules that use torch.load(). "
            "This is required for checkpoint-loading hardening."
        )


def get_canonical_security_profile() -> dict[str, Any]:
    """Get canonical security profile for CAS identity.

    Returns:
        Dictionary with STATIC security policy values only.
        NO runtime-derived values are included.

    CRITICAL: This profile is designed for CAS identity inclusion.
    It must be:
    - Deterministic (same output every time)
    - Static (not affected by import order or runtime state)
    - Canonical (consistent across processes and environments)

    Example:
        >>> get_canonical_security_profile()
        {
            'policy_version': 'torch_safe_load_v2',
            'minimum_supported_torch_version': '2.12.0',
            'torch_load_enforced': True,
            'weights_only': True,
            'cve_mitigation': 'fixed-by-supported-torch-baseline'
        }
    """
    return {
        "policy_version": SECURITY_PROFILE_VERSION,
        "minimum_supported_torch_version": MINIMUM_SUPPORTED_TORCH_VERSION,
        "torch_load_enforced": True,  # Policy requirement, not runtime state
        "weights_only": True,  # Policy requirement
        "cve_mitigation": "fixed-by-supported-torch-baseline",
    }
