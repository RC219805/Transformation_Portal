"""
Architecture modules for BasicSR-TP.

This module contains neural network architectures extracted from BasicSR.
Only RRDBNet is included for Real-ESRGAN compatibility.
"""

__all__ = ["RRDBNet"]


# Lazy import to avoid requiring torch at import time
def __getattr__(name):
    """Lazy import RRDBNet only when accessed."""
    if name == "RRDBNet":
        from basicsr_tp.archs.rrdbnet_arch import RRDBNet
        return RRDBNet
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
