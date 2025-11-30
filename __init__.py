"""
Hyper-Reality Enhancement Module
Part of the Transformation_Portal luxury image processing pipeline

Version: 3.1.0

New in 3.1.0:
- PerceptualQualityAssessor for true perceptual measurement
- Enhanced training pipeline (v2) with LPIPS integration
- Automatic model weight loading
- Material-specific fidelity scoring
"""

# pylint: disable=possibly-unused-variable

# Core enhancement API
from .hyper_reality_enhancement import (
    HyperRealityProcessor,
    EnhancementConfig,
    QualityMode,
    enhance_image,
    # Neural network components
    CausticGenerator,
    AtmosphericSynthesizer,
    MaterialTranscendence,
    SpatialHarmonics,
)

# Model management
from .model_loader import (
    ModelLoader,
    load_pretrained_weights,
)

# Quality assessment
from .perceptual_quality_assessment import (
    PerceptualQualityAssessor,
    QualityReport,
    QualityTargets,
    QualityDomain,
    assess_quality,
)

__all__ = [
    # Core API
    'HyperRealityProcessor',
    'EnhancementConfig',
    'QualityMode',
    'enhance_image',
    
    # Neural networks
    'CausticGenerator',
    'AtmosphericSynthesizer',
    'MaterialTranscendence',
    'SpatialHarmonics',
    
    # Model management
    'ModelLoader',
    'load_pretrained_weights',
    
    # Quality assessment
    'PerceptualQualityAssessor',
    'QualityReport',
    'QualityTargets',
    'QualityDomain',
    'assess_quality',
]

__version__ = '3.1.0'
