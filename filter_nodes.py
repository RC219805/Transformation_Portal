"""Utility helpers for constructing FFmpeg filter nodes safely."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping


_OPERATION_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_PARAM_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _stringify_value(value: Any) -> str:
    """Normalise a parameter value to a string suitable for FFmpeg syntax."""

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int,)):  # bool already handled above.
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Float parameters must be finite numbers")
        return ("%0.6f" % value).rstrip("0").rstrip(".") or "0"

    if isinstance(value, Path):
        value = str(value)

    if isinstance(value, str):
        if value == "":
            raise ValueError("String parameters cannot be empty")
        if "\n" in value or "\r" in value:
            raise ValueError("String parameters cannot contain newlines")
        return value

    raise TypeError(
        "Filter parameter values must be strings, numbers, booleans or Paths"
    )


@dataclass(frozen=True)
class FilterNode:
    """Represent a single FFmpeg filter node with validated syntax."""

    operation: str
    parameters: Dict[str, str]  # Sanitized; constructor accepts Mapping[str, Any]

    def __init__(self, operation: str, parameters: Mapping[str, Any] | None = None):
        if not isinstance(operation, str) or not operation:
            raise TypeError("operation must be a non-empty string")
        if not _OPERATION_PATTERN.match(operation):
            raise ValueError(
                "operation must contain only alphanumeric characters and underscores"
            )

        object.__setattr__(self, "operation", operation)
        sanitized = self._sanitize(parameters or {})
        object.__setattr__(self, "parameters", sanitized)

    @staticmethod
    def _sanitize(parameters: Mapping[str, Any]) -> Dict[str, str]:
        """Return a sanitized, type-checked copy of the provided parameters."""

        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")

        sanitized: Dict[str, str] = {}
        for key, value in parameters.items():
            if value is None:
                # Skip unset parameters. This mirrors FFmpeg behaviour where
                # omitted values fall back to filter defaults.
                continue

            if not isinstance(key, str) or not key:
                raise TypeError("Parameter names must be non-empty strings")
            if not _PARAM_PATTERN.match(key):
                raise ValueError(
                    "Parameter names must contain only alphanumeric characters and underscores"
                )

            sanitized[key] = _stringify_value(value)

        return dict(sorted(sanitized.items(), key=lambda x: str(x[0])))

    def compile(self) -> str:
        """Compile the node into FFmpeg filter syntax."""

        return self._validated_syntax()

    def _validated_syntax(self) -> str:
        if not self.parameters:
            return self.operation

        parts = [f"{name}={value}" for name, value in self.parameters.items()]
        return f"{self.operation}=" + ":".join(parts)


__all__ = ["FilterNode"]

