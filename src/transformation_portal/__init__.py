"""
Transformation Portal - Professional Image and Video Processing Toolkit

A comprehensive suite of AI-powered tools and pipelines for luxury real estate
rendering, architectural visualization, and editorial post-production.

Key Components:
- Pipelines: High-level processing workflows (Lux Render, Depth Processing)
- Processors: Core image/video processing engines (Material Response, TIFF, Video)
- Enhancers: Specialized enhancement tools (Aerial, Board Material)
- Analyzers: Code quality and workflow analysis tools
- Rendering: Rendering workflow utilities
- Utils: Shared utilities and helpers
"""

__version__ = "0.1.0"
__author__ = "RC219805"

# Lazy imports for commonly used components
# This reduces initial import time while maintaining convenience

def _lazy_import(module_path, attr_name):
    """Lazy import helper to defer loading until needed."""
    def _loader():
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    return _loader

# Pipelines (lazy loaded)
_lux_render = None
_depth_tools = None

# Processors (lazy loaded)
_material_response = None
_video_grader = None

def get_lux_render_pipeline():
    """Get the Lux Render Pipeline (lazy loaded)."""
    global _lux_render
    if _lux_render is None:
        from .pipelines import lux_render_pipeline as _lux_render
    return _lux_render

def get_material_response():
    """Get Material Response processor (lazy loaded)."""
    global _material_response
    if _material_response is None:
        from .processors.material_response import core as _material_response
    return _material_response

# Convenience exports for backward compatibility
__all__ = [
    '__version__',
    '__author__',
    'get_lux_render_pipeline',
    'get_material_response',
]

