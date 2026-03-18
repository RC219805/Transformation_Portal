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

Usage:
    >>> from transformation_portal.core.security.torch_security import safe_load
    >>> state_dict = safe_load("model.pt", map_location="cpu")

Note:
    All torch.load() calls in the codebase should use safe_load() or
    explicitly pass weights_only=True.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Union


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

    torch_version = torch.__version__.split("+")[0]  # Strip +cpu/+cu* suffix

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
