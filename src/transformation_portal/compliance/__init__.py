"""Compliance and licensing module for Transformation Portal.

This module provides tools for enforcing licensing constraints, particularly
around non-commercial models like Depth Anything V3.1 (DA3 1.1, CC BY-NC 4.0).

Key exports:
- require_non_commercial: Decorator to enforce non-commercial opt-in
- LicenseRestrictionError: Exception raised when licensing constraints violated
- validate_non_commercial_preset: Validator for preset licensing markers
- load_and_validate_preset: Load and validate preset YAML files
"""

from .licensing import (
    LicenseRestrictionError,
    load_and_validate_preset,
    require_non_commercial,
    validate_non_commercial_preset,
)

__all__ = [
    "LicenseRestrictionError",
    "require_non_commercial",
    "validate_non_commercial_preset",
    "load_and_validate_preset",
]
