"""
Property-Specific Training Components for 750 Picacho Lane.

This package provides specialized training infrastructure for creating
property-specific enhancement models optimized for luxury real estate.

Components:
- PicachoAnalyzer: Property image analysis and material detection
- DepthSynthesis: Depth map generation using Depth Anything V2
- DatasetGenerator: Multi-scale augmented dataset creation
- PicachoTrainer: Multi-stage property-specific training
- PicachoInference: Production 4K 16-bit TIFF processing
"""

from .picacho_analyzer import (
    PicachoAnalyzer,
    PropertyReport,
    MaterialDetection,
    ColorPalette,
    ArchitecturalFeatures
)
from .depth_synthesis import (
    DepthSynthesis,
    DepthSynthesisConfig,
    SynthesizedDepth
)
from .dataset_generator import (
    DatasetGenerator,
    DatasetConfig,
    TrainingSample
)
from .picacho_trainer import (
    PicachoTrainer,
    TrainingConfig,
    TrainingStage
)
from .picacho_inference import (
    PicachoInference,
    InferenceConfig,
    EnhancedOutput
)

__all__ = [
    # Analyzer
    "PicachoAnalyzer",
    "PropertyReport",
    "MaterialDetection",
    "ColorPalette",
    "ArchitecturalFeatures",
    # Depth
    "DepthSynthesis",
    "DepthSynthesisConfig",
    "SynthesizedDepth",
    # Dataset
    "DatasetGenerator",
    "DatasetConfig",
    "TrainingSample",
    # Training
    "PicachoTrainer",
    "TrainingConfig",
    "TrainingStage",
    # Inference
    "PicachoInference",
    "InferenceConfig",
    "EnhancedOutput",
]

__version__ = "1.0.0"
