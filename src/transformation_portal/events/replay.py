"""Event replay for debugging and testing."""

from typing import Any, Callable, Dict, List, Optional

from .store import Event, EventStore


class OperationRegistry:
    """Registry for operation handlers used during event replay.

    Handlers are functions that take an Event and perform the actual
    operation replay logic.

    Example:
        >>> registry = OperationRegistry()
        >>>
        >>> def handle_image_enhanced(event: Event) -> Any:
        ...     # Extract data and re-run the operation
        ...     print(f"Re-enhancing image: {event.data}")
        ...     return {"replayed": True}
        >>>
        >>> registry.register("image.enhanced", handle_image_enhanced)
    """

    def __init__(self):
        """Initialize the operation registry."""
        self._handlers: Dict[str, Callable[[Event], Any]] = {}

    def register(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        """Register a handler for an event type.

        Args:
            event_type: The event type to handle (e.g., "image.enhanced")
            handler: Function that takes an Event and returns a result
        """
        self._handlers[event_type] = handler

    def get_handler(self, event_type: str) -> Optional[Callable[[Event], Any]]:
        """Get the handler for an event type.

        Args:
            event_type: The event type to look up

        Returns:
            Handler function if registered, None otherwise
        """
        return self._handlers.get(event_type)

    def has_handler(self, event_type: str) -> bool:
        """Check if a handler is registered for an event type.

        Args:
            event_type: The event type to check

        Returns:
            True if handler exists, False otherwise
        """
        return event_type in self._handlers

    def unregister(self, event_type: str) -> None:
        """Unregister a handler for an event type.

        Args:
            event_type: The event type to unregister
        """
        self._handlers.pop(event_type, None)

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()

    def get_registered_types(self) -> List[str]:
        """Get list of all registered event types.

        Returns:
            List of event types with registered handlers
        """
        return list(self._handlers.keys())


class EventReplayer:
    """Replay events for debugging and testing.

    Example:
        >>> replayer = EventReplayer(event_store)
        >>>
        >>> # Register an operation handler
        >>> def handle_image(event):
        ...     print(f"Processing: {event.data}")
        ...     return {"status": "replayed"}
        >>> replayer.registry.register("image.enhanced", handle_image)
        >>>
        >>> # Replay specific events
        >>> events = store.get_events_by_type("image.enhanced")
        >>> replayer.replay(events, on_event=lambda e: print(e.type))
    """

    def __init__(
        self,
        event_store: EventStore,
        operation_registry: Optional[OperationRegistry] = None
    ):
        """Initialize replayer.

        Args:
            event_store: Event store to replay from
            operation_registry: Optional registry of operation handlers.
                If not provided, a new empty registry is created.
        """
        self.store = event_store
        self.registry = operation_registry or OperationRegistry()

    def replay(
        self,
        events: List[Event],
        on_event: Optional[Callable[[Event], Any]] = None,
        dry_run: bool = True,
        skip_unregistered: bool = True
    ) -> List[Any]:
        """Replay events.

        Args:
            events: Events to replay.
            on_event: Optional callback called for each event. The return value of the callback is appended to the results list.
            dry_run: If True, don't actually execute operations; only the callback (if provided) is called.
            skip_unregistered: If True, skip events without registered handlers.
                If False, raise ValueError for unregistered events.

        Returns:
            List of replay results. The structure of the list depends on the arguments:

            - If ``dry_run=True`` and ``on_event`` is provided:
                Returns a list of results from the callback (i.e., ``[on_event(event) for event in events]``).

            - If ``dry_run=False`` and ``on_event`` is not provided:
                Returns a list of dicts, one per event, each with keys:
                    - ``event_id``: The event's ID.
                    - ``event_type``: The event's type.
                    - ``status``: "success" or "error".
                    - ``result``: The handler's return value (if successful).
                    - ``error``: The error message (if an exception occurred).

            - If both ``on_event`` is provided and ``dry_run=False``:
                Returns a list containing both callback results and handler result dicts, in the order they are appended.
                For each event, the callback result is appended first (if any), then the handler result dict (if any).

        Raises:
            ValueError: If skip_unregistered is False and an event has no handler.
        """
        results = []

        for event in events:
            if on_event:
                result = on_event(event)
                results.append(result)

            if not dry_run:
                # Execute actual operation replay using registered handlers
                handler = self.registry.get_handler(event.type)

                if handler is not None:
                    try:
                        replay_result = handler(event)
                        results.append({
                            'event_id': event.id,
                            'event_type': event.type,
                            'status': 'success',
                            'result': replay_result
                        })
                    except Exception as e:
                        results.append({
                            'event_id': event.id,
                            'event_type': event.type,
                            'status': 'error',
                            'error': str(e)
                        })
                elif not skip_unregistered:
                    raise ValueError(
                        f"No handler registered for event type: {event.type}"
                    )

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
