"""Gold Standard Lux Depth Pipeline V2.

Modular, GPU-accelerated, production-oriented pipeline with optional
automatic material segmentation and service-mode operation.

This package is designed to be used either via the CLI module:

    python -m lux_depth_v2.cli ...

or embedded as a library:

    from lux_depth_v2.pipeline import LuxPipelineV2
"""

__all__ = ["config", "pipeline"]
