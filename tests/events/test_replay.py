"""Tests for events.replay module.

Covers:
- OperationRegistry handler management
- EventReplayer replay logic
- Dry-run vs. actual execution
- Error handling during replay
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.events.replay import (
    EventReplayer,
    OperationRegistry,
    replay_events,
)
from transformation_portal.events.store import Event, EventStore

pytestmark = pytest.mark.unit


class TestOperationRegistry:
    """Tests for OperationRegistry."""

    def test_register_handler(self):
        """Test registering a handler."""
        registry = OperationRegistry()

        def my_handler(event: Event) -> Any:
            return {"handled": event.type}

        registry.register("test.event", my_handler)

        assert registry.has_handler("test.event")
        assert registry.get_handler("test.event") is my_handler

    def test_register_replaces_existing(self):
        """Test registering replaces existing handler."""
        registry = OperationRegistry()

        handler1 = lambda e: "first"
        handler2 = lambda e: "second"

        registry.register("test.event", handler1)
        registry.register("test.event", handler2)

        handler = registry.get_handler("test.event")
        assert handler is handler2

    def test_get_handler_nonexistent(self):
        """Test get_handler returns None for unregistered type."""
        registry = OperationRegistry()

        assert registry.get_handler("nonexistent") is None

    def test_has_handler(self):
        """Test has_handler checks registration."""
        registry = OperationRegistry()

        registry.register("exists", lambda e: None)

        assert registry.has_handler("exists") is True
        assert registry.has_handler("does.not.exist") is False

    def test_unregister(self):
        """Test unregistering a handler."""
        registry = OperationRegistry()

        registry.register("to.remove", lambda e: None)
        assert registry.has_handler("to.remove") is True

        registry.unregister("to.remove")
        assert registry.has_handler("to.remove") is False

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent handler doesn't raise."""
        registry = OperationRegistry()

        # Should not raise
        registry.unregister("never.registered")

    def test_clear(self):
        """Test clearing all handlers."""
        registry = OperationRegistry()

        registry.register("type.one", lambda e: None)
        registry.register("type.two", lambda e: None)
        registry.register("type.three", lambda e: None)

        registry.clear()

        assert registry.has_handler("type.one") is False
        assert registry.has_handler("type.two") is False
        assert registry.has_handler("type.three") is False

    def test_get_registered_types(self):
        """Test getting list of registered types."""
        registry = OperationRegistry()

        registry.register("alpha", lambda e: None)
        registry.register("beta", lambda e: None)
        registry.register("gamma", lambda e: None)

        types = registry.get_registered_types()

        assert isinstance(types, tuple)  # Immutable
        assert set(types) == {"alpha", "beta", "gamma"}

    def test_get_registered_types_empty(self):
        """Test get_registered_types returns empty tuple when no handlers."""
        registry = OperationRegistry()

        types = registry.get_registered_types()

        assert types == ()


class TestEventReplayer:
    """Tests for EventReplayer."""

    def test_replayer_creation(self, tmp_path):
        """Test EventReplayer creation."""
        store = EventStore(tmp_path / "events")

        replayer = EventReplayer(store)

        assert replayer.store is store
        assert isinstance(replayer.registry, OperationRegistry)

    def test_replayer_with_custom_registry(self, tmp_path):
        """Test EventReplayer with custom registry."""
        store = EventStore(tmp_path / "events")
        registry = OperationRegistry()
        registry.register("custom.type", lambda e: "custom")

        replayer = EventReplayer(store, operation_registry=registry)

        assert replayer.registry is registry
        assert replayer.registry.has_handler("custom.type")

    def test_replay_dry_run(self, tmp_path):
        """Test replay in dry-run mode."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="test", timestamp=1000.0, data={}))
        store.append(Event(id="evt-2", type="test", timestamp=2000.0, data={}))

        replayer = EventReplayer(store)

        callback_calls = []

        def on_event(event):
            callback_calls.append(event.id)
            return f"processed-{event.id}"

        events = store.get_events(reverse=False)
        results = replayer.replay(events, on_event=on_event, dry_run=True)

        # Callback should be called for each event
        assert callback_calls == ["evt-1", "evt-2"]
        # Results should be callback return values
        assert results == ["processed-evt-1", "processed-evt-2"]

    def test_replay_with_handlers(self, tmp_path):
        """Test replay with registered handlers."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="image.processed", timestamp=1000.0, data={"path": "img.jpg"}))

        replayer = EventReplayer(store)

        handler_calls = []

        def image_handler(event):
            handler_calls.append(event.id)
            return {"status": "replayed", "path": event.data["path"]}

        replayer.registry.register("image.processed", image_handler)

        events = store.get_events()
        results = replayer.replay(events, dry_run=False)

        assert len(handler_calls) == 1
        assert handler_calls[0] == "evt-1"
        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["event_id"] == "evt-1"
        assert results[0]["result"]["status"] == "replayed"

    def test_replay_handler_error(self, tmp_path):
        """Test replay handles handler errors gracefully."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="failing.handler", timestamp=1000.0, data={}))

        replayer = EventReplayer(store)

        def failing_handler(event):
            raise ValueError("Handler failed!")

        replayer.registry.register("failing.handler", failing_handler)

        events = store.get_events()
        results = replayer.replay(events, dry_run=False)

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert results[0]["event_id"] == "evt-1"
        assert "Handler failed!" in results[0]["error"]

    def test_replay_skip_unregistered(self, tmp_path):
        """Test replay skips unregistered event types by default."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="unregistered.type", timestamp=1000.0, data={}))
        store.append(Event(id="evt-2", type="registered.type", timestamp=2000.0, data={}))

        replayer = EventReplayer(store)
        replayer.registry.register("registered.type", lambda e: "handled")

        events = store.get_events(reverse=False)
        results = replayer.replay(events, dry_run=False, skip_unregistered=True)

        # Only the registered type should produce a result
        assert len(results) == 1
        assert results[0]["event_type"] == "registered.type"

    def test_replay_raise_on_unregistered(self, tmp_path):
        """Test replay raises for unregistered when skip_unregistered=False."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="unregistered.type", timestamp=1000.0, data={}))

        replayer = EventReplayer(store)

        events = store.get_events()

        with pytest.raises(ValueError, match="No handler registered"):
            replayer.replay(events, dry_run=False, skip_unregistered=False)

    def test_replay_with_callback_and_handler(self, tmp_path):
        """Test replay with both callback and handler."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="dual.test", timestamp=1000.0, data={}))

        replayer = EventReplayer(store)
        replayer.registry.register("dual.test", lambda e: "handler-result")

        callback_results = []

        def callback(event):
            callback_results.append(event.id)
            return f"callback-{event.id}"

        events = store.get_events()
        results = replayer.replay(events, on_event=callback, dry_run=False)

        # Callback should be called
        assert callback_results == ["evt-1"]
        # Results should include both callback result and handler result
        assert "callback-evt-1" in results
        handler_result = [r for r in results if isinstance(r, dict) and r.get("event_id") == "evt-1"][0]
        assert handler_result["result"] == "handler-result"

    def test_replay_correlation(self, tmp_path):
        """Test replay_correlation method."""
        store = EventStore(tmp_path / "events")

        corr_id = "batch-123"
        store.append(Event(id="evt-1", type="start", timestamp=1000.0, data={}, correlation_id=corr_id))
        store.append(Event(id="evt-2", type="other", timestamp=2000.0, data={}, correlation_id="different"))
        store.append(Event(id="evt-3", type="end", timestamp=3000.0, data={}, correlation_id=corr_id))

        replayer = EventReplayer(store)

        callback_events = []

        def callback(event):
            callback_events.append(event.id)

        results = replayer.replay_correlation(corr_id, on_event=callback)

        # Should only replay events with matching correlation ID
        assert set(callback_events) == {"evt-1", "evt-3"}

    def test_replay_preserves_system_exceptions(self, tmp_path):
        """Test replay re-raises KeyboardInterrupt and SystemExit."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="interrupt.test", timestamp=1000.0, data={}))

        replayer = EventReplayer(store)
        replayer.registry.register("interrupt.test", lambda e: (_ for _ in ()).throw(KeyboardInterrupt()))

        events = store.get_events()

        with pytest.raises(KeyboardInterrupt):
            replayer.replay(events, dry_run=False)


class TestReplayEventsFunction:
    """Tests for replay_events convenience function."""

    def test_replay_events_all(self, tmp_path, capsys):
        """Test replay_events replays all events."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="test.one", timestamp=1000.0, data={}))
        store.append(Event(id="evt-2", type="test.two", timestamp=2000.0, data={}))

        events = replay_events(store)

        assert len(events) == 2
        captured = capsys.readouterr()
        assert "Replaying:" in captured.out

    def test_replay_events_by_type(self, tmp_path, capsys):
        """Test replay_events filters by type."""
        store = EventStore(tmp_path / "events")
        store.append(Event(id="evt-1", type="target.type", timestamp=1000.0, data={}))
        store.append(Event(id="evt-2", type="other.type", timestamp=2000.0, data={}))
        store.append(Event(id="evt-3", type="target.type", timestamp=3000.0, data={}))

        events = replay_events(store, event_type="target.type")

        assert len(events) == 2
        assert all(e.type == "target.type" for e in events)

    def test_replay_events_with_limit(self, tmp_path):
        """Test replay_events respects limit."""
        store = EventStore(tmp_path / "events")
        for i in range(10):
            store.append(Event(id=f"evt-{i}", type="test", timestamp=float(i * 1000), data={}))

        events = replay_events(store, limit=3)

        assert len(events) == 3
