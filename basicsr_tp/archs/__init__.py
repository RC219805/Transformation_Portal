"""
Architecture modules for BasicSR-TP.

This module contains neural network architectures extracted from BasicSR.
Only RRDBNet is included for Real-ESRGAN compatibility.
"""

from basicsr_tp.archs.rrdbnet_arch import RRDBNet

__all__ = ["RRDBNet"]
