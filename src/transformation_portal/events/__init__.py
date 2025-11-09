"""Event sourcing for Transformation Portal operations.

Track all operations as events for debugging, replay, and audit trails.

Example:
    >>> from transformation_portal.events import EventStore, event
    >>>
    >>> store = EventStore()
    >>>
    >>> @event("image.enhanced")
    ... def enhance_image(image_path):
    ...     return process(image_path)
"""

from .decorators import (
    event,
    tracked,
)
from .replay import (
    EventReplayer,
    replay_events,
)
from .store import (
    Event,
    EventStore,
    get_global_store,
)

__all__ = [
    'Event',
    'EventStore',
    'get_global_store',
    'event',
    'tracked',
    'replay_events',
    'EventReplayer',
]

__version__ = '1.0.0'
