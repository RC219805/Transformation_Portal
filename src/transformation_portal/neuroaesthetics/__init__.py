"""Computational neuroaesthetics for luxury real estate imagery.

Quantifies and optimizes emotional resonance through scientifically validated principles:

1. **Golden Ratio Composition** (φ ≈ 1.618)
   - Systematic proportional relationships create measurable coherence
   - Position key features at 1:1.618 divisions

2. **Color Harmony Analysis**
   - Harmonious combinations activate medial orbitofrontal cortex
   - Disharmonious palettes trigger automatic amygdala responses
   - CIELAB color space analysis

3. **Spatial Frequency Balance**
   - Human visual system most sensitive to 2-6 cycles per degree
   - Balanced LSF/HSF content reduces visual stress
   - Coarse-to-fine processing (magnocellular → parvocellular)

4. **Emotional Triggers**
   - Nostalgia: warm palettes, natural materials, heritage details
   - Aspiration: high spatial quality, abundant light, golden ratio
   - Desire: quality craftsmanship, exclusive features, believable luxury

Based on research:
- Aesthetic Triad model (sensory-motor, emotion-valuation, knowledge-meaning)
- fMRI-validated neural activation patterns
- 82%+ ML prediction accuracy for aesthetic preference
"""

from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer
from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer
from transformation_portal.neuroaesthetics.spatial_frequency import SpatialFrequencyAnalyzer
from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

__all__ = [
    'GoldenRatioAnalyzer',
    'ColorHarmonyAnalyzer',
    'SpatialFrequencyAnalyzer',
    'EmotionalOptimizer',
]
