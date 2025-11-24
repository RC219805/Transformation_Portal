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
        try:
            from basicsr_tp.archs.rrdbnet_arch import RRDBNet
            return RRDBNet
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}': torch is required for basicsr_tp.archs. "
                f"Install with: pip install torch\nOriginal error: {e}"
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
