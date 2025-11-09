"""Event replay for debugging and testing."""

from typing import Any, Callable, List, Optional

from .store import Event, EventStore


class EventReplayer:
    """Replay events for debugging and testing.

    Example:
        >>> replayer = EventReplayer(event_store)
        >>>
        >>> # Replay specific events
        >>> events = store.get_events_by_type("image.enhanced")
        >>> replayer.replay(events, on_event=lambda e: print(e.type))
    """

    def __init__(self, event_store: EventStore):
        """Initialize replayer.

        Args:
            event_store: Event store to replay from
        """
        self.store = event_store

    def replay(
        self,
        events: List[Event],
        on_event: Optional[Callable[[Event], Any]] = None,
        dry_run: bool = True
    ) -> List[Any]:
        """Replay events.

        Args:
            events: Events to replay
            on_event: Callback for each event
            dry_run: If True, don't actually execute operations

        Returns:
            List of replay results
        """
        results = []

        for event in events:
            if on_event:
                result = on_event(event)
                results.append(result)

            if not dry_run:
                # TODO: Implement actual operation replay
                # This would require a registry of operation handlers
                pass

        return results

    def replay_correlation(
        self,
        correlation_id: str,
        on_event: Optional[Callable[[Event], Any]] = None
    ) -> List[Any]:
        """Replay all events with a specific correlation ID.

        Args:
            correlation_id: Correlation identifier
            on_event: Callback for each event

        Returns:
            List of replay results
        """
        events = self.store.get_events_by_correlation(correlation_id)
        return self.replay(events, on_event=on_event)


def replay_events(
    event_store: EventStore,
    event_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Event]:
    """Replay events from store.

    Args:
        event_store: Event store
        event_type: Filter by event type
        limit: Maximum events to replay

    Returns:
        List of replayed events
    """
    if event_type:
        events = event_store.get_events_by_type(event_type, limit=limit)
    else:
        events = event_store.get_events(limit=limit)

    replayer = EventReplayer(event_store)
    replayer.replay(events, on_event=lambda e: print(f"Replaying: {e.type}"))

    return events
