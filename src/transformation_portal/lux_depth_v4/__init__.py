"""LuxDepthV4 photographic API; optional model runtimes stay subprocess-only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from transformation_portal.lux_depth_v4.lifecycle import LuxDepthV4Request, PreparedLuxExecutionV4, prepare

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v4.pipeline import LuxDepthV4Result
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher

__all__ = ["LuxDepthV4Request", "PreparedLuxExecutionV4", "prepare", "run"]


def run(
    prepared: PreparedLuxExecutionV4,
    *,
    cancellation: Callable[[], bool] | None = None,
    publisher: GenerationPublisher | None = None,
) -> LuxDepthV4Result:
    """Execute one prepared batch using the shared executor."""
    from transformation_portal.lux_depth_v4.pipeline import run as execute

    return execute(prepared, cancellation=cancellation, publisher=publisher)
