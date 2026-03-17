"""Real-time evaluation dashboard package.

This package provides a FastAPI-based dashboard for:
- Real-time evaluation streaming via WebSocket
- Pipeline execution monitoring
- DAG visualization and editing
- Artifact and lineage browsing
- Experiment tracking
- GPU monitoring
"""

from transformation_portal.dashboard.server import (
    DashboardServer,
    broadcast_event,
    create_app,
)

__all__ = [
    # Core server
    "DashboardServer",
    "broadcast_event",
    "create_app",
]
