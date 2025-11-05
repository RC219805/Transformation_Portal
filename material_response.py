"""Material Response - Backward compatibility wrapper.

This module provides backward compatibility by delegating to the actual
implementation in src/transformation_portal/processors/material_response/core.py.

For new code, please import directly from:
    from transformation_portal.processors.material_response import core

This wrapper exists to maintain compatibility with existing scripts that
import from the repository root.
"""

# Import all public API from the core implementation
from transformation_portal.processors.material_response.core import (
    MaterialResponseExample,
    MaterialResponsePrinciple,
    MaterialAestheticProfile,
    LightingProfile,
    ViewerProfile,
    EmotionalResonance,
    ContextualResonance,
    NeuroAestheticEngine,
    GlobalLuxurySemantics,
    FutureStatePredictor,
    CognitiveMaterialResponse,
    violates,
    MarketingClaimValidator,
    MaterialResponseValidator,
    compose_operations,
    apply_transformation_tensor,
    QuantumMaterialResponse,
)

__all__ = [
    "MaterialResponseExample",
    "MaterialResponsePrinciple",
    "MaterialAestheticProfile",
    "LightingProfile",
    "ViewerProfile",
    "EmotionalResonance",
    "ContextualResonance",
    "NeuroAestheticEngine",
    "GlobalLuxurySemantics",
    "FutureStatePredictor",
    "CognitiveMaterialResponse",
    "violates",
    "MarketingClaimValidator",
    "MaterialResponseValidator",
    "compose_operations",
    "apply_transformation_tensor",
    "violates",
]
