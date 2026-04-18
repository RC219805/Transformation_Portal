"""
Core Observability Integration

Re-exports from existing observability module with integration helpers.
Compatibility note: retained as an internal/shared helper surface with
direct smoke coverage, but it currently has no production imports.
"""

from .integration import create_logger, setup_logging, setup_metrics

__all__ = [
    "setup_logging",
    "setup_metrics",
    "create_logger",
]
