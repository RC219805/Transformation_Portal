"""FastAPI route factories for the portal origin."""

from .jobs import JobRouteHandlers, create_jobs_router

__all__ = ["JobRouteHandlers", "create_jobs_router"]
