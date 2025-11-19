"""
Hyper-Reality Enhancement Module
Part of the Transformation_Portal luxury image processing pipeline
"""

# Lazy imports to avoid dependency errors in CI
__all__ = [
    'HyperRealityProcessor',
    'EnhancementConfig',
    'QualityMode',
    'enhance_image'
]

__version__ = '3.0.0'

def __getattr__(name):
    """Lazy import pattern to avoid loading torch when not needed"""
    if name in __all__:
        from .hyper_reality_enhancement import (
            HyperRealityProcessor,
            EnhancementConfig,
            QualityMode,
            enhance_image
        )
        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
