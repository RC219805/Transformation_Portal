"""
Core Observability Integration

Re-exports from existing observability module with integration helpers.
"""

from .integration import (
    create_logger,
    setup_logging,
    setup_metrics,
)

__all__ = [
    "setup_logging",
    "setup_metrics",
    "create_logger",
]
