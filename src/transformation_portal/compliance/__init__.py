"""Compliance and licensing module for Transformation Portal.

This module provides tools for enforcing licensing constraints, including
non-commercial model restrictions plus research-only and attestation gates
for materials backends.

Key exports:
- require_non_commercial: Decorator to enforce non-commercial opt-in
- LicenseRestrictionError: Exception raised when licensing constraints violated
- validate_non_commercial_preset: Validator for preset licensing markers
- validate_materials_preset: Validator for materials tier/license/attestation gates
- load_and_validate_preset: Load and validate preset YAML files
"""

from .licensing import (
    LicenseRestrictionError,
    load_and_validate_preset,
    require_non_commercial,
    validate_materials_preset,
    validate_non_commercial_preset,
)

__all__ = [
    "LicenseRestrictionError",
    "require_non_commercial",
    "validate_non_commercial_preset",
    "validate_materials_preset",
    "load_and_validate_preset",
]
