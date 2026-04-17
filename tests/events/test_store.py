"""Tests for events.store module.

Covers:
- Event dataclass creation and serialization
- EventStore append, query, persistence
- Global store singleton behavior
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.events.store import Event, EventStore, get_global_store

pytestmark = pytest.mark.unit


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation_minimal(self):
        """Test Event creation with minimal required fields."""
        event = Event(
            id="test-123",
            type="test.event",
            timestamp=1234567890.0,
            data={"key": "value"},
        )

        assert event.id == "test-123"
        assert event.type == "test.event"
        assert event.timestamp == 1234567890.0
        assert event.data == {"key": "value"}
        assert event.metadata == {}
        assert event.user is None
        assert event.correlation_id is None

    def test_event_creation_full(self):
        """Test Event creation with all fields."""
        event = Event(
            id="test-456",
            type="image.processed",
            timestamp=1234567890.123,
            data={"path": "image.jpg", "preset": "golden_hour"},
            metadata={"version": "1.0", "pipeline": "lux-depth-v3"},
            user="alice",
            correlation_id="batch-789",
        )

        assert event.id == "test-456"
        assert event.type == "image.processed"
        assert event.timestamp == 1234567890.123
        assert event.data == {"path": "image.jpg", "preset": "golden_hour"}
        assert event.metadata == {"version": "1.0", "pipeline": "lux-depth-v3"}
        assert event.user == "alice"
        assert event.correlation_id == "batch-789"

    def test_event_to_dict(self):
        """Test Event serialization to dict."""
        event = Event(
            id="test-789",
            type="depth.estimated",
            timestamp=1234567890.0,
            data={"model": "da3"},
            metadata={"quality": "apex"},
            user="bob",
            correlation_id="run-123",
        )

        d = event.to_dict()

        assert d["id"] == "test-789"
        assert d["type"] == "depth.estimated"
        assert d["timestamp"] == 1234567890.0
        assert d["data"] == {"model": "da3"}
        assert d["metadata"] == {"quality": "apex"}
        assert d["user"] == "bob"
        assert d["correlation_id"] == "run-123"

    def test_event_from_dict(self):
        """Test Event deserialization from dict."""
        d = {
            "id": "from-dict-123",
            "type": "from.dict",
            "timestamp": 9876543210.5,
            "data": {"key": "value"},
            "metadata": {"source": "test"},
            "user": "charlie",
            "correlation_id": "corr-456",
        }

        event = Event.from_dict(d)

        assert event.id == "from-dict-123"
        assert event.type == "from.dict"
        assert event.timestamp == 9876543210.5
        assert event.data == {"key": "value"}
        assert event.metadata == {"source": "test"}
        assert event.user == "charlie"
        assert event.correlation_id == "corr-456"

    def test_event_roundtrip(self):
        """Test Event roundtrip through to_dict/from_dict."""
        original = Event(
            id=str(uuid.uuid4()),
            type="roundtrip.test",
            timestamp=time.time(),
            data={"nested": {"data": [1, 2, 3]}},
            metadata={"test": True},
            user="roundtrip-user",
            correlation_id=str(uuid.uuid4()),
        )

        roundtripped = Event.from_dict(original.to_dict())

        assert roundtripped.id == original.id
        assert roundtripped.type == original.type
        assert roundtripped.timestamp == original.timestamp
        assert roundtripped.data == original.data
        assert roundtripped.metadata == original.metadata
        assert roundtripped.user == original.user
        assert roundtripped.correlation_id == original.correlation_id


class TestEventStore:
    """Tests for EventStore."""

    def test_store_creation_creates_directory(self, tmp_path):
        """Test EventStore creates storage directory."""
        storage_path = tmp_path / "my_events"
        assert not storage_path.exists()

        store = EventStore(storage_path)

        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_store_default_path(self, tmp_path, monkeypatch):
        """Test EventStore uses default .events path."""
        monkeypatch.chdir(tmp_path)

        store = EventStore()

        assert store.storage_path == Path(".events")
        assert store.storage_path.exists()

    def test_append_and_get_events(self, tmp_path):
        """Test append and retrieval of events."""
        store = EventStore(tmp_path / "events")

        event1 = Event(id="evt-1", type="test.one", timestamp=1000.0, data={"n": 1})
        event2 = Event(id="evt-2", type="test.two", timestamp=2000.0, data={"n": 2})

        store.append(event1)
        store.append(event2)

        events = store.get_events()

        assert len(events) == 2
        # Default is reverse=True (most recent first)
        assert events[0].id == "evt-2"
        assert events[1].id == "evt-1"

    def test_get_events_chronological(self, tmp_path):
        """Test get_events with reverse=False."""
        store = EventStore(tmp_path / "events")

        store.append(Event(id="evt-1", type="test", timestamp=1000.0, data={}))
        store.append(Event(id="evt-2", type="test", timestamp=2000.0, data={}))
        store.append(Event(id="evt-3", type="test", timestamp=3000.0, data={}))

        events = store.get_events(reverse=False)

        assert events[0].id == "evt-1"
        assert events[1].id == "evt-2"
        assert events[2].id == "evt-3"

    def test_get_events_with_limit(self, tmp_path):
        """Test get_events with limit."""
        store = EventStore(tmp_path / "events")

        for i in range(10):
            store.append(Event(id=f"evt-{i}", type="test", timestamp=float(i * 1000), data={}))

        events = store.get_events(limit=3)

        assert len(events) == 3
        # Most recent first (reverse=True by default)
        assert events[0].id == "evt-9"
        assert events[1].id == "evt-8"
        assert events[2].id == "evt-7"

    def test_get_events_with_offset(self, tmp_path):
        """Test get_events with offset."""
        store = EventStore(tmp_path / "events")

        for i in range(5):
            store.append(Event(id=f"evt-{i}", type="test", timestamp=float(i * 1000), data={}))

        events = store.get_events(offset=2, limit=2)

        assert len(events) == 2
        # With reverse=True: evt-4, evt-3, evt-2, evt-1, evt-0
        # offset=2 skips evt-4, evt-3
        # limit=2 returns evt-2, evt-1
        assert events[0].id == "evt-2"
        assert events[1].id == "evt-1"

    def test_get_events_by_type(self, tmp_path):
        """Test filtering events by type."""
        store = EventStore(tmp_path / "events")

        store.append(Event(id="img-1", type="image.processed", timestamp=1000.0, data={}))
        store.append(Event(id="depth-1", type="depth.estimated", timestamp=2000.0, data={}))
        store.append(Event(id="img-2", type="image.processed", timestamp=3000.0, data={}))
        store.append(Event(id="mask-1", type="mask.generated", timestamp=4000.0, data={}))

        image_events = store.get_events_by_type("image.processed")

        assert len(image_events) == 2
        assert all(e.type == "image.processed" for e in image_events)

    def test_get_events_by_type_with_limit(self, tmp_path):
        """Test filtering by type with limit."""
        store = EventStore(tmp_path / "events")

        for i in range(10):
            store.append(Event(id=f"img-{i}", type="image.processed", timestamp=float(i * 1000), data={}))

        events = store.get_events_by_type("image.processed", limit=3)

        assert len(events) == 3
        # Returns last 3 (most recent)
        assert events[0].id == "img-7"
        assert events[1].id == "img-8"
        assert events[2].id == "img-9"

    def test_get_events_by_correlation(self, tmp_path):
        """Test filtering events by correlation ID."""
        store = EventStore(tmp_path / "events")

        corr_id = "batch-123"
        store.append(Event(id="evt-1", type="start", timestamp=1000.0, data={}, correlation_id=corr_id))
        store.append(Event(id="evt-2", type="other", timestamp=2000.0, data={}, correlation_id="different"))
        store.append(Event(id="evt-3", type="process", timestamp=3000.0, data={}, correlation_id=corr_id))
        store.append(Event(id="evt-4", type="end", timestamp=4000.0, data={}, correlation_id=corr_id))

        correlated = store.get_events_by_correlation(corr_id)

        assert len(correlated) == 3
        assert all(e.correlation_id == corr_id for e in correlated)

    def test_get_events_in_range(self, tmp_path):
        """Test filtering events by time range."""
        store = EventStore(tmp_path / "events")

        store.append(Event(id="evt-1", type="test", timestamp=1000.0, data={}))
        store.append(Event(id="evt-2", type="test", timestamp=2000.0, data={}))
        store.append(Event(id="evt-3", type="test", timestamp=3000.0, data={}))
        store.append(Event(id="evt-4", type="test", timestamp=4000.0, data={}))
        store.append(Event(id="evt-5", type="test", timestamp=5000.0, data={}))

        events = store.get_events_in_range(2000.0, 4000.0)

        assert len(events) == 3
        ids = {e.id for e in events}
        assert ids == {"evt-2", "evt-3", "evt-4"}

    def test_clear(self, tmp_path):
        """Test clearing all events."""
        store = EventStore(tmp_path / "events")

        store.append(Event(id="evt-1", type="test", timestamp=1000.0, data={}))
        store.append(Event(id="evt-2", type="test", timestamp=2000.0, data={}))

        assert len(store.get_events()) == 2

        store.clear()

        assert len(store.get_events()) == 0

    def test_persistence_to_disk(self, tmp_path):
        """Test events are persisted to disk."""
        storage_path = tmp_path / "events"
        store = EventStore(storage_path)

        event = Event(
            id="persist-123",
            type="persistence.test",
            timestamp=time.time(),
            data={"persisted": True},
        )
        store.append(event)

        # Check files were created
        json_files = list(storage_path.rglob("*.json"))
        assert len(json_files) == 1

        # Verify content
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["id"] == "persist-123"
        assert data["type"] == "persistence.test"

    def test_load_events_on_init(self, tmp_path):
        """Test events are loaded from disk on initialization."""
        storage_path = tmp_path / "events"

        # Create first store and add events
        store1 = EventStore(storage_path)
        store1.append(Event(id="evt-1", type="test", timestamp=1000.0, data={"n": 1}))
        store1.append(Event(id="evt-2", type="test", timestamp=2000.0, data={"n": 2}))

        # Create new store instance - should load existing events
        store2 = EventStore(storage_path)

        events = store2.get_events(reverse=False)
        assert len(events) == 2
        assert events[0].id == "evt-1"
        assert events[1].id == "evt-2"

    def test_event_organized_by_date(self, tmp_path):
        """Test events are organized by date in storage."""
        storage_path = tmp_path / "events"
        store = EventStore(storage_path)

        # Use a known timestamp
        ts = 1609459200.0  # 2021-01-01 00:00:00 UTC

        event = Event(id="dated-event", type="test", timestamp=ts, data={})
        store.append(event)

        # Check date-based directory was created
        date_dirs = [d for d in storage_path.iterdir() if d.is_dir()]
        assert len(date_dirs) == 1
        assert "2021-01-01" in date_dirs[0].name or "2020-12-31" in date_dirs[0].name  # TZ dependent

    def test_load_handles_invalid_json(self, tmp_path, capsys):
        """Test store handles invalid JSON files gracefully."""
        storage_path = tmp_path / "events"
        storage_path.mkdir()

        # Create invalid JSON file
        invalid_file = storage_path / "invalid.json"
        invalid_file.write_text("not valid json {{{")

        # Should load without crashing
        store = EventStore(storage_path)

        # Should have printed error message
        captured = capsys.readouterr()
        assert "Failed to load event" in captured.out

        # Should have no events loaded
        assert len(store.get_events()) == 0


class TestGlobalStore:
    """Tests for global store singleton."""

    def test_get_global_store_returns_same_instance(self, tmp_path, monkeypatch):
        """Test get_global_store returns singleton."""
        # Reset global store
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        store1 = get_global_store()
        store2 = get_global_store()

        assert store1 is store2

    def test_global_store_thread_safety(self, tmp_path, monkeypatch):
        """Test global store creation is thread-safe."""
        import threading

        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        results = []
        errors = []

        def get_store():
            try:
                store = get_global_store()
                results.append(store)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should be the same instance
        assert all(r is results[0] for r in results)
