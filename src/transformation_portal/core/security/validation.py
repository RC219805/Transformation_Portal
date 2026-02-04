"""
Input Validation Framework.

Provides generic validation logic for ensuring data integrity before processing.
"""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class ValidationError(ValueError):
    """Raised when critical validation fails."""

    pass


class InputValidator:
    """Base validator class."""

    @staticmethod
    def check_not_empty(value: Any, field_name: str) -> Optional[str]:
        if value is None or (isinstance(value, (str, list, dict)) and not value):
            return f"{field_name} cannot be empty"
        return None

    @staticmethod
    def check_type(value: Any, expected_type: type, field_name: str) -> Optional[str]:
        if not isinstance(value, expected_type):
            return f"{field_name} must be of type {expected_type.__name__}"
        return None
