"""
Architecture modules for BasicSR-TP.

This module contains neural network architectures extracted from BasicSR.
Only RRDBNet is included for Real-ESRGAN compatibility.
"""

# Note: RRDBNet is available via lazy loading in __getattr__ below
# Omitted from __all__ to avoid static analysis warnings
__all__ = []


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
