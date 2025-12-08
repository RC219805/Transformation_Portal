"""
Universal Hardening Framework for Transformation Portal.

This module provides generic hardening capabilities that can be applied
to any pipeline, not just lux_depth_v2. It enables security, reproducibility,
and observability for legacy pipelines with minimal refactoring.

Key Components:
    - UniversalHardenedWrapper: Generic hardening wrapper for any pipeline
    - wrap_function: Wrap standalone functions with hardening

Example:
    >>> from transformation_portal.hardening import UniversalHardenedWrapper
    >>> from lux_depth_v2.hardening.policy import HardeningPolicy
    >>> 
    >>> policy = HardeningPolicy.load()
    >>> wrapped = UniversalHardenedWrapper(my_pipeline, policy)
    >>> result = wrapped.process(input_path)
"""

from .universal import UniversalHardenedWrapper, Pipeline, wrap_function

__all__ = [
    "UniversalHardenedWrapper",
    "Pipeline",
    "wrap_function",
]

__version__ = "2.0.0"
