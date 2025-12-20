"""Depth Anything 3 (DA3) Integration Module for Transformation Portal.

This module provides production-ready integration of Depth Anything 3 models,
offering unified any-view monocular and multi-view depth inference with metric
depth output and camera pose estimation capabilities.

Key Features:
- Monocular and multi-view depth estimation
- Metric depth output (absolute scale)
- Camera pose estimation
- Enhanced geometric reconstruction
- Point cloud and TSDF fusion
- GPU/CPU/MPS acceleration
- Official DA3 CLI integration with backend service support

Integration Modes:
- Native Python API (default)
- Official DA3 CLI wrapper (--use-cli)
- Backend service acceleration (--use-backend, 10-20x speedup)

Models Available:
- DA3NESTED-GIANT-LARGE: unified any-view model with metric output
- DA3-GIANT/LARGE/BASE/SMALL: any-view/multi-view models
- DA3METRIC-LARGE: monocular metric depth
- DA3MONO-LARGE: high-quality relative monocular depth

Architecture:
Input Manager → Preprocessing → DA3 Inference (Native or CLI)
              → Postprocessing/Fusion → Validation/Quality Gates
              → Output/Export
"""

from __future__ import annotations

import os

# Fix OpenMP duplicate library issue (must be set before importing DA3)
if os.environ.get('KMP_DUPLICATE_LIB_OK') != 'TRUE':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__version__ = "0.1.0"
__author__ = "RC219805"

# Core components (safe imports - no heavy dependencies)
from lux_depth_v3.config import (
    DA3Config,
    DA3CLIConfig,
    DA3APIConfig,
    ModelVariant,
    InferenceMode,
    Preset,
)
from lux_depth_v3.input_manager import InputManager, ImageInput

# Lazy imports for optional DA3 components (may require GPU/torch)
# DO NOT import these eagerly - they pull in PyTorch and DA3 models
DA3InferenceEngine = None
DepthQualityMetrics = None
DA3Backend = None

try:
    from lux_depth_v3.validation import DepthQualityMetrics as _DQM
    DepthQualityMetrics = _DQM
except ImportError:
    pass

try:
    from lux_depth_v3.inference import DA3InferenceEngine as _Engine
    DA3InferenceEngine = _Engine
except ImportError:
    pass

try:
    from lux_depth_v3.da3_wrapper import (
        DA3Backend as _Backend,
        DA3CLI,
        check_da3_cli_available,
    )
    DA3Backend = _Backend
except ImportError:
    DA3CLI = None
    check_da3_cli_available = None

try:
    from lux_depth_v3.reference_view import (
        RefViewStrategy,
        RefViewSelectionResult,
        ReferenceViewSelector,
        select_reference_view,
    )
except ImportError:
    RefViewStrategy = None
    RefViewSelectionResult = None
    ReferenceViewSelector = None
    select_reference_view = None

try:
    from lux_depth_v3.da3_integration import (
        DA3DepthEstimator,
        DA3Result,
        estimate_depth,
        convert_to_metric_depth,
    )
except ImportError:
    DA3DepthEstimator = None
    DA3Result = None
    estimate_depth = None
    convert_to_metric_depth = None

_DA3_AVAILABLE = DA3InferenceEngine is not None

__all__ = [
    "__version__",
    "DA3Config",
    "DA3CLIConfig",
    "DA3APIConfig",
    "ModelVariant",
    "InferenceMode",
    "Preset",
    "InputManager",
    "ImageInput",
    "DA3InferenceEngine",
    "DepthQualityMetrics",
    "DA3Backend",
    "DA3CLI",
    "check_da3_cli_available",
    "RefViewStrategy",
    "RefViewSelectionResult",
    "ReferenceViewSelector",
    "select_reference_view",
    "DA3DepthEstimator",
    "DA3Result",
    "estimate_depth",
    "convert_to_metric_depth",
]
