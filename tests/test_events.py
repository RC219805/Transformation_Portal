"""Tests for event replay and operation registry."""

import time
import uuid
from pathlib import Path

import pytest

from transformation_portal.events import (
    Event,
    EventReplayer,
    EventStore,
    OperationRegistry,
)


@pytest.fixture
def temp_event_store(tmp_path):
    """Create a temporary event store for testing."""
    return EventStore(storage_path=tmp_path / 'events')


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        Event(
            id=str(uuid.uuid4()),
            type="image.enhanced",
            timestamp=time.time(),
            data={
                'function': 'enhance_image',
                'args': ['image1.jpg'],
                'kwargs': {'preset': 'golden_hour'}
            }
        ),
        Event(
            id=str(uuid.uuid4()),
            type="depth.estimated",
            timestamp=time.time(),
            data={
                'function': 'estimate_depth',
                'args': ['image2.jpg'],
                'kwargs': {'model': 'midas'}
            }
        ),
        Event(
            id=str(uuid.uuid4()),
            type="image.enhanced",
            timestamp=time.time(),
            data={
                'function': 'enhance_image',
                'args': ['image3.jpg'],
                'kwargs': {'preset': 'sunset'}
            }
        ),
    ]


class TestOperationRegistry:
    """Test the OperationRegistry class."""

    def test_register_and_get_handler(self):
        """Test registering and retrieving handlers."""
        registry = OperationRegistry()

        def handler(event: Event):
            return f"Handled {event.type}"

        registry.register("test.event", handler)
        retrieved = registry.get_handler("test.event")

        assert retrieved is not None
        assert retrieved == handler

    def test_get_nonexistent_handler(self):
        """Test getting a handler that doesn't exist."""
        registry = OperationRegistry()
        handler = registry.get_handler("nonexistent.event")
        assert handler is None

    def test_has_handler(self):
        """Test checking for handler existence."""
        registry = OperationRegistry()

        def handler(event: Event):
            return "test"

        assert not registry.has_handler("test.event")
        registry.register("test.event", handler)
        assert registry.has_handler("test.event")

    def test_unregister(self):
        """Test unregistering a handler."""
        registry = OperationRegistry()

        def handler(event: Event):
            return "test"

        registry.register("test.event", handler)
        assert registry.has_handler("test.event")

        registry.unregister("test.event")
        assert not registry.has_handler("test.event")

    def test_clear(self):
        """Test clearing all handlers."""
        registry = OperationRegistry()

        def handler1(event: Event):
            return "test1"

        def handler2(event: Event):
            return "test2"

        registry.register("event1", handler1)
        registry.register("event2", handler2)

        assert len(registry.get_registered_types()) == 2

        registry.clear()
        assert len(registry.get_registered_types()) == 0

    def test_get_registered_types(self):
        """Test getting list of registered types."""
        registry = OperationRegistry()

        def handler(event: Event):
            return "test"

        registry.register("type1", handler)
        registry.register("type2", handler)
        registry.register("type3", handler)

        types = registry.get_registered_types()
        assert set(types) == {"type1", "type2", "type3"}


class TestEventReplayer:
    """Test the EventReplayer class."""

    def test_init_with_default_registry(self, temp_event_store):
        """Test initializing replayer with default registry."""
        replayer = EventReplayer(temp_event_store)
        assert replayer.store == temp_event_store
        assert isinstance(replayer.registry, OperationRegistry)

    def test_init_with_custom_registry(self, temp_event_store):
        """Test initializing replayer with custom registry."""
        custom_registry = OperationRegistry()
        replayer = EventReplayer(temp_event_store, custom_registry)
        assert replayer.registry == custom_registry

    def test_dry_run_replay(self, temp_event_store, sample_events):
        """Test replaying events in dry run mode."""
        replayer = EventReplayer(temp_event_store)

        # Add events to store
        for event in sample_events:
            temp_event_store.append(event)

        # Replay in dry run mode (default)
        results = replayer.replay(sample_events)

        # In dry run mode, handlers aren't called
        assert isinstance(results, list)

    def test_replay_with_on_event_callback(self, temp_event_store, sample_events):
        """Test replay with on_event callback."""
        replayer = EventReplayer(temp_event_store)

        called_events = []

        def callback(event):
            called_events.append(event.type)
            return event.type

        results = replayer.replay(sample_events, on_event=callback)

        assert len(called_events) == 3
        assert called_events == ["image.enhanced", "depth.estimated", "image.enhanced"]

    def test_replay_with_handler_execution(self, temp_event_store, sample_events):
        """Test replaying events with actual handler execution."""
        replayer = EventReplayer(temp_event_store)

        executed = []

        def image_handler(event: Event):
            executed.append(f"enhanced:{event.data.get('args', [])[0]}")
            return {"status": "processed"}

        def depth_handler(event: Event):
            executed.append(f"depth:{event.data.get('args', [])[0]}")
            return {"status": "estimated"}

        replayer.registry.register("image.enhanced", image_handler)
        replayer.registry.register("depth.estimated", depth_handler)

        results = replayer.replay(sample_events, dry_run=False)

        # Check that handlers were executed
        assert len(executed) == 3
        assert "enhanced:image1.jpg" in executed
        assert "enhanced:image3.jpg" in executed
        assert "depth:image2.jpg" in executed

        # Check results structure
        assert len(results) == 3
        for result in results:
            assert 'event_id' in result
            assert 'event_type' in result
            assert 'status' in result
            assert result['status'] == 'success'

    def test_replay_with_handler_error(self, temp_event_store):
        """Test replay when handler raises an error."""
        replayer = EventReplayer(temp_event_store)

        def failing_handler(event: Event):
            raise RuntimeError("Handler failed!")

        replayer.registry.register("test.event", failing_handler)

        event = Event(
            id=str(uuid.uuid4()),
            type="test.event",
            timestamp=time.time(),
            data={'test': 'data'}
        )

        results = replayer.replay([event], dry_run=False)

        assert len(results) == 1
        assert results[0]['status'] == 'error'
        assert 'Handler failed!' in results[0]['error']

    def test_replay_skip_unregistered(self, temp_event_store, sample_events):
        """Test skipping events without registered handlers."""
        replayer = EventReplayer(temp_event_store)

        # Register handler for only one event type
        def image_handler(event: Event):
            return {"status": "processed"}

        replayer.registry.register("image.enhanced", image_handler)

        # Replay with skip_unregistered=True (default)
        results = replayer.replay(sample_events, dry_run=False)

        # Should only process image.enhanced events
        assert len(results) == 2  # Only the two image.enhanced events

    def test_replay_error_on_unregistered(self, temp_event_store, sample_events):
        """Test raising error for unregistered event types."""
        replayer = EventReplayer(temp_event_store)

        # Don't register any handlers

        with pytest.raises(ValueError, match="No handler registered for event type"):
            replayer.replay(
                sample_events,
                dry_run=False,
                skip_unregistered=False
            )

    def test_replay_correlation(self, temp_event_store):
        """Test replaying events by correlation ID."""
        replayer = EventReplayer(temp_event_store)
        correlation_id = str(uuid.uuid4())

        # Create correlated events
        events = [
            Event(
                id=str(uuid.uuid4()),
                type="batch.started",
                timestamp=time.time(),
                data={'batch_id': '123'},
                correlation_id=correlation_id
            ),
            Event(
                id=str(uuid.uuid4()),
                type="batch.completed",
                timestamp=time.time(),
                data={'batch_id': '123'},
                correlation_id=correlation_id
            ),
        ]

        for event in events:
            temp_event_store.append(event)

        # Replay by correlation
        results = replayer.replay_correlation(correlation_id)

        assert isinstance(results, list)


def test_integration_with_event_store(temp_event_store):
    """Test integration between EventStore and EventReplayer."""
    store = temp_event_store
    replayer = EventReplayer(store)

    processed_images = []

    def handler(event: Event):
        image_path = event.data.get('args', [])[0] if event.data.get('args') else None
        if image_path:
            processed_images.append(image_path)
        return {"processed": image_path}

    replayer.registry.register("image.processed", handler)

    # Store some events
    for i in range(5):
        event = Event(
            id=str(uuid.uuid4()),
            type="image.processed",
            timestamp=time.time(),
            data={'function': 'process', 'args': [f'image{i}.jpg'], 'kwargs': {}}
        )
        store.append(event)

    # Get events from store
    events = store.get_events_by_type("image.processed")

    # Replay them
    results = replayer.replay(events, dry_run=False)

    assert len(processed_images) == 5
    assert all(f'image{i}.jpg' in processed_images for i in range(5))
    assert len(results) == 5
