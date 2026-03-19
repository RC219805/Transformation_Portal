"""PyTorch security utilities for CVE-2025-32434 mitigation.

This module provides safe model loading functions that mitigate CVE-2025-32434,
a critical RCE vulnerability (CVSS 9.8) in torch.load() when loading untrusted
model files.

MITIGATION STRATEGY (ADR-032 Determinism Preservation):
    Instead of upgrading to torch >= 2.6.0 (which would break CAS determinism),
    we mitigate at runtime by enforcing weights_only=True for all torch.load() calls.

    Benefits:
    - Preserves pinned torch==2.2.2 for deterministic CAS identity
    - Mitigates RCE vulnerability by disabling arbitrary code execution
    - Maintains cross-platform reproducibility

CVE-2025-32434 Details:
    - Vulnerability: Remote Code Execution via torch.load()
    - CVSS Score: 9.8 (Critical)
    - Affected versions: torch < 2.6.0
    - Attack vector: Malicious .pt/.pth model files
    - Mitigation: Use weights_only=True parameter

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
SECURITY_PROFILE_VERSION = "torch_safe_load_v1"

# Track whether enforcement has been installed
_enforcement_installed = False
_original_torch_load = None


def safe_load(
    path: Union[str, Path],
    *,
    map_location: Optional[Any] = None,
) -> Any:
    """Safely load PyTorch model weights with CVE-2025-32434 mitigation.

    This function wraps torch.load() with weights_only=True to prevent
    arbitrary code execution from malicious model files.

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
        This function enforces weights_only=True which disables arbitrary
        code execution during model loading. Files containing custom classes
        or functions will fail to load - this is intentional for security.

        If you need to load a file with custom objects, you must:
        1. Verify the source is trusted
        2. Use torch.load() directly with explicit documentation
        3. Add security review comment explaining the trust boundary
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for safe_load(). "
            "Install with: pip install torch"
        ) from e

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
        - mitigation_available: True if weights_only mitigation is available
        - recommendation: Security recommendation string

    Example:
        >>> status = check_torch_security_compliance()
        >>> if status["cve_2025_32434_vulnerable"]:
        ...     print("Use safe_load() for all model loading")
    """
    try:
        import torch
        from packaging.version import Version
    except ImportError:
        return {
            "torch_version": None,
            "cve_2025_32434_vulnerable": None,
            "mitigation_available": False,
            "recommendation": "PyTorch not installed",
        }

    torch_version = torch.__version__.split("+")[0]  # Strip build suffix (+cpu, +cu118, etc.)

    try:
        version = Version(torch_version)
        vulnerable = version < Version("2.6.0")
    except Exception:
        # Non-standard version string
        vulnerable = True

    return {
        "torch_version": torch_version,
        "cve_2025_32434_vulnerable": vulnerable,
        "mitigation_available": True,
        "recommendation": (
            "Use safe_load() or weights_only=True for all torch.load() calls"
            if vulnerable
            else "Torch version includes CVE-2025-32434 fix"
        ),
    }


def warn_if_vulnerable() -> None:
    """Emit a warning if torch is vulnerable to CVE-2025-32434.

    This can be called at module import time to alert users of vulnerable
    torch installations and the need for mitigation.
    """
    status = check_torch_security_compliance()
    if status.get("cve_2025_32434_vulnerable"):
        warnings.warn(
            f"PyTorch {status['torch_version']} is vulnerable to CVE-2025-32434. "
            "Always use weights_only=True with torch.load() or use "
            "transformation_portal.core.security.torch_security.safe_load().",
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
    weights_only=True to mitigate CVE-2025-32434.

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
    global _original_torch_load

    if not _enforcement_installed or _original_torch_load is None:
        raise RuntimeError(
            "Torch security enforcement not installed. "
            "Call install_global_enforcement() at startup."
        )

    # Warn if pickle_module is provided (not supported with weights_only=True)
    if pickle_module is not None:
        warnings.warn(
            "pickle_module parameter is not supported when weights_only=True. "
            "The parameter will be ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Block explicit weights_only=False
    if not weights_only:
        raise SecurityError(
            "CVE-2025-32434: weights_only=False is blocked for security. "
            "Use weights_only=True or safe_load() to load model files. "
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
    mitigations, such as using torch.load() without weights_only=True.
    """

    pass


def install_global_enforcement() -> bool:
    """Install global enforcement of safe torch.load behavior.

    This function patches torch.load to enforce weights_only=True on ALL calls,
    not just direct safe_load() usage. This ensures CVE-2025-32434 mitigation
    is applied system-wide.

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
        Calling this function removes CVE-2025-32434 protection.
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

    ⚠️ DANGER: This function bypasses CVE-2025-32434 protections.

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
    global _original_torch_load

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

    Example:
        >>> get_security_profile_hash()
        'sha256:abc123...'
    """
    profile_data = {
        "version": SECURITY_PROFILE_VERSION,
        "enforcement_installed": _enforcement_installed,
        "cve_2025_32434_mitigation": "weights_only_true",
    }
    # Use sorted keys for deterministic hash
    import json

    profile_str = json.dumps(profile_data, sort_keys=True)
    digest = hashlib.sha256(profile_str.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
