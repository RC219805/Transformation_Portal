"""
Model wrappers for depth estimation.
"""

# Lazy imports to avoid ImportError when dependencies are not installed
# This allows pytest to import the root directory without requiring all dependencies

# Decision: lazy_imports - Variables in __all__ are loaded dynamically via __getattr__
# This is intentional to support optional dependencies (torch, coremltools)
__all__ = ["DepthAnythingV2Model", "ModelBackend", "ModelVariant", "CoreMLDepthModel"]  # pylint: disable=undefined-all-variable

def __getattr__(name):
    """Lazy import of depth estimation modules."""
    if name in ("DepthAnythingV2Model", "ModelBackend", "ModelVariant"):
        from .depth_anything_v2 import DepthAnythingV2Model, ModelBackend, ModelVariant
        globals()["DepthAnythingV2Model"] = DepthAnythingV2Model
        globals()["ModelBackend"] = ModelBackend
        globals()["ModelVariant"] = ModelVariant
        return globals()[name]
    if name == "CoreMLDepthModel":
        from .coreml_wrapper import CoreMLDepthModel
        globals()["CoreMLDepthModel"] = CoreMLDepthModel
        return CoreMLDepthModel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
