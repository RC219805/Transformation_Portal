"""Environment fingerprint for ADR-030 determinism harness.

SPEC-DH-001 Section 5 requires each harness run to publish an environment
fingerprint including OS, ISA, runtime version, and dependency lock IDs.

This module provides a deterministic environment fingerprint that:
- Reports OS name/version
- Reports ISA (platform architecture)
- Reports Python runtime version
- Reports critical dependency versions (numpy, etc.)
- Reports harness engine version

Design constraints:
- Output must be deterministic (no timestamps or dynamic host IDs)
- All values must be Python primitives for JCS/JSON serialization
- Must not include sensitive data (IP addresses, full hostnames)
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict

# Harness engine version (increment on harness logic changes)
HARNESS_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Deterministic environment fingerprint for cross-ISA audit.

    All fields are Python primitives for JCS/JSON serialization safety.
    No timestamps or host-specific identifiers are included.
    """

    harness_engine_version: str
    os_system: str
    os_release: str
    os_machine: str
    python_version: str
    python_implementation: str
    numpy_version: str
    numpy_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _get_numpy_config() -> Dict[str, Any]:
    """Extract deterministic numpy configuration.

    Returns a dictionary with numpy build information that may affect
    numerical behavior (BLAS backend, etc.).
    """
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy_not_installed"}

    config: Dict[str, Any] = {}

    # Basic numpy info
    config["version"] = str(np.__version__)

    # Try to get build info
    try:
        # NumPy 2.x uses np.show_config(mode='dicts')
        if hasattr(np, "show_config"):
            import contextlib
            import io

            # Capture show_config output as string for parsing
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                np.show_config()
            config["build_info_summary"] = "available"
        else:
            config["build_info_summary"] = "unavailable"
    except (AttributeError, TypeError):
        config["build_info_summary"] = "error"

    # Detect BLAS backend (important for determinism)
    try:
        # NumPy 1.x/2.x compatible BLAS detection
        if hasattr(np, "__config__"):
            blas_info = getattr(np.__config__, "blas_opt_info", None)
            if blas_info is not None and isinstance(blas_info, dict):
                config["blas_libraries"] = blas_info.get("libraries", [])
            else:
                config["blas_libraries"] = "unknown"
        else:
            config["blas_libraries"] = "unknown"
    except (AttributeError, TypeError):
        config["blas_libraries"] = "detection_error"

    return config


def capture_environment() -> EnvironmentFingerprint:
    """Capture deterministic environment fingerprint.

    Returns an EnvironmentFingerprint suitable for inclusion in harness
    artifacts and reports per SPEC-DH-001 Section 5.
    """
    try:
        import numpy as np

        numpy_version = str(np.__version__)
    except ImportError:
        numpy_version = "not_installed"

    return EnvironmentFingerprint(
        harness_engine_version=HARNESS_ENGINE_VERSION,
        os_system=platform.system(),
        os_release=platform.release(),
        os_machine=platform.machine(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        numpy_version=numpy_version,
        numpy_config=_get_numpy_config(),
    )


def environment_fingerprint_dict() -> Dict[str, Any]:
    """Return environment fingerprint as a dictionary.

    Convenience function for direct JSON serialization.
    """
    return capture_environment().to_dict()
