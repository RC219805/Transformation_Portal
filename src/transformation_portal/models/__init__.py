"""Model loading and management utilities.

This package provides manifest-aware model loading with strict revision verification.
"""

from transformation_portal.models.hf_lock import HFModelLockRecord, resolve_all_required_files
from transformation_portal.models.hf_manifest_loader import (
    HFManifestLoaderError,
    HFResolvedLocalModel,
    resolve_manifest_model,
)

__all__ = [
    "HFModelLockRecord",
    "resolve_all_required_files",
    "HFManifestLoaderError",
    "HFResolvedLocalModel",
    "resolve_manifest_model",
]
