"""
BasicSR-TP: Security-Hardened Vendored BasicSR Components
==========================================================

This package contains minimal, security-hardened components from BasicSR v1.4.2,
vendored to eliminate dependency on vulnerable SLURM distributed utilities.

Security Advisory: CVE-2024-27763
----------------------------------
- Vulnerability: Command injection in BasicSR ≤ 1.4.2 via SLURM_NODELIST
- CVSS Score: 5.3 (Medium)
- Impact: Local privilege escalation in SLURM environments
- Mitigation: This vendored version removes all SLURM-related code

Components Included:
--------------------
- RRDBNet: ESRGAN super-resolution architecture (required for Real-ESRGAN)

Components Excluded:
--------------------
- All distributed training utilities (including vulnerable dist_util.py)
- SLURM integration code
- Training infrastructure
- Data loaders and augmentation pipelines
- Metrics and loss functions
- Registry system (simplified for standalone use)

Usage:
------
This package is a drop-in replacement for BasicSR's RRDBNet architecture:

    # Old (vulnerable):
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # New (secure):
    from basicsr_tp.archs.rrdbnet_arch import RRDBNet

License: Apache-2.0 (inherited from BasicSR)
Original: https://github.com/XPixelGroup/BasicSR
Vendored: 2025-11-23 for Transformation Portal
"""

__version__ = "1.4.2-tp1"  # Based on BasicSR 1.4.2, TP security patch 1
__author__ = "Transformation Portal (vendored from XPixelGroup BasicSR)"
__license__ = "Apache-2.0"

__all__ = ["RRDBNet"]


# Lazy import of RRDBNet to avoid requiring torch at package import time
# This allows accessing package metadata (__version__, etc.) without torch installed
def __getattr__(name):
    """Lazy import RRDBNet only when accessed.
    
    This prevents ImportError when torch is not installed but user only needs
    package metadata or wants to check if basicsr_tp is available.
    """
    if name == "RRDBNet":
        try:
            from basicsr_tp.archs.rrdbnet_arch import RRDBNet
            return RRDBNet
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' from basicsr_tp: torch is required. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
