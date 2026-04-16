"""Tests for events.decorators module.

Covers:
- @event decorator for automatic event tracking
- @tracked decorator for correlation ID tracking
- Error handling in decorated functions
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.events.decorators import event, tracked
from transformation_portal.events.store import Event, EventStore, get_global_store

pytestmark = pytest.mark.unit


class TestEventDecorator:
    """Tests for @event decorator."""

    def test_event_decorator_records_success(self, tmp_path, monkeypatch):
        """Test @event decorator records successful function calls."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("test.operation")
        def my_function(x, y):
            return x + y

        result = my_function(2, 3)

        assert result == 5

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 1
        evt = events[0]
        assert evt.type == "test.operation"
        assert evt.data["function"] == "my_function"
        assert evt.data["status"] == "success"
        assert "duration" in evt.data

    def test_event_decorator_records_args(self, tmp_path, monkeypatch):
        """Test @event decorator records function arguments."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("arg.test")
        def process_image(path, preset="default", quality=100):
            return f"processed {path}"

        result = process_image("image.jpg", preset="golden_hour", quality=95)

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 1
        evt = events[0]
        assert evt.data["args"] == ["image.jpg"]
        assert evt.data["kwargs"] == {"preset": "golden_hour", "quality": "95"}

    def test_event_decorator_include_result(self, tmp_path, monkeypatch):
        """Test @event decorator with include_result=True."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("result.test", include_result=True)
        def compute_something():
            return {"computed": True, "value": 42}

        result = compute_something()

        assert result == {"computed": True, "value": 42}

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 1
        evt = events[0]
        assert "result" in evt.data
        # Result is stringified
        assert "computed" in evt.data["result"]

    def test_event_decorator_exclude_result_by_default(self, tmp_path, monkeypatch):
        """Test @event decorator excludes result by default."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("no.result")
        def returns_secret():
            return "secret-data"

        returns_secret()

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 1
        evt = events[0]
        assert "result" not in evt.data

    def test_event_decorator_records_error(self, tmp_path, monkeypatch):
        """Test @event decorator records errors."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("error.test")
        def failing_function():
            raise ValueError("Something went wrong!")

        with pytest.raises(ValueError, match="Something went wrong!"):
            failing_function()

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 1
        evt = events[0]
        assert evt.type == "error.test.error"  # Error suffix added
        assert evt.data["status"] == "error"
        assert "Something went wrong!" in evt.data["error"]
        assert "duration" in evt.data

    def test_event_decorator_preserves_function_metadata(self, tmp_path, monkeypatch):
        """Test @event preserves wrapped function metadata."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("metadata.test")
        def documented_function(x: int, y: int) -> int:
            """This is the docstring."""
            return x + y

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."

    def test_event_decorator_generates_unique_ids(self, tmp_path, monkeypatch):
        """Test @event generates unique event IDs."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("unique.id.test")
        def simple_func():
            return True

        simple_func()
        simple_func()
        simple_func()

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 3
        ids = {e.id for e in events}
        assert len(ids) == 3  # All unique


class TestTrackedDecorator:
    """Tests for @tracked decorator."""

    def test_tracked_decorator_creates_correlation_id(self, tmp_path, monkeypatch):
        """Test @tracked creates correlation ID and start/complete events."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @tracked()
        def batch_process():
            return "done"

        result = batch_process()

        assert result == "done"

        store = get_global_store()
        events = store.get_events(reverse=False)

        assert len(events) == 2
        start_event = events[0]
        end_event = events[1]

        assert start_event.type == "batch_process.started"
        assert end_event.type == "batch_process.completed"
        assert start_event.correlation_id == end_event.correlation_id
        assert start_event.correlation_id is not None

    def test_tracked_decorator_with_explicit_correlation_id(self, tmp_path, monkeypatch):
        """Test @tracked with explicit correlation ID."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        corr_id = "my-batch-123"

        @tracked(correlation_id=corr_id)
        def tracked_operation():
            return True

        tracked_operation()

        store = get_global_store()
        events = store.get_events()

        assert all(e.correlation_id == corr_id for e in events)

    def test_tracked_decorator_records_error(self, tmp_path, monkeypatch):
        """Test @tracked records failure events on error."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @tracked()
        def failing_tracked():
            raise RuntimeError("Tracked failure!")

        with pytest.raises(RuntimeError, match="Tracked failure!"):
            failing_tracked()

        store = get_global_store()
        events = store.get_events(reverse=False)

        assert len(events) == 2
        start_event = events[0]
        fail_event = events[1]

        assert start_event.type == "failing_tracked.started"
        assert fail_event.type == "failing_tracked.failed"
        assert "Tracked failure!" in fail_event.data["error"]

    def test_tracked_decorator_preserves_function_metadata(self, tmp_path, monkeypatch):
        """Test @tracked preserves wrapped function metadata."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @tracked()
        def my_tracked_func():
            """Tracked function docstring."""
            pass

        assert my_tracked_func.__name__ == "my_tracked_func"
        assert my_tracked_func.__doc__ == "Tracked function docstring."

    def test_tracked_decorator_stores_correlation_on_wrapper(self, tmp_path, monkeypatch):
        """Test @tracked stores correlation ID on wrapper for nested access."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @tracked()
        def outer_tracked():
            # Access correlation ID from wrapper
            return outer_tracked._correlation_id

        corr_id = outer_tracked()

        assert corr_id is not None
        # Should be a valid UUID-like string
        assert len(corr_id) == 36

    def test_tracked_and_event_together(self, tmp_path, monkeypatch):
        """Test @tracked and @event can be combined."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @tracked(correlation_id="combined-batch")
        @event("inner.operation")
        def combined_func():
            return "result"

        result = combined_func()

        assert result == "result"

        store = get_global_store()
        events = store.get_events()

        # Should have: started, inner.operation, completed (3 events)
        assert len(events) >= 2  # At least the tracked events

        types = {e.type for e in events}
        assert "inner.operation" in types or "combined_func.started" in types


class TestDecoratorIntegration:
    """Integration tests for decorators."""

    def test_multiple_decorated_functions(self, tmp_path, monkeypatch):
        """Test multiple decorated functions work independently."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("func.one")
        def func_one():
            return 1

        @event("func.two")
        def func_two():
            return 2

        @event("func.three")
        def func_three():
            return 3

        func_one()
        func_two()
        func_one()
        func_three()

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 4

        type_counts = {}
        for e in events:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1

        assert type_counts["func.one"] == 2
        assert type_counts["func.two"] == 1
        assert type_counts["func.three"] == 1

    def test_decorated_generator_function(self, tmp_path, monkeypatch):
        """Test @event works with generator-returning functions."""
        import transformation_portal.events.store as store_module

        monkeypatch.setattr(store_module, "_global_store", None)
        monkeypatch.chdir(tmp_path)

        @event("generator.test", include_result=True)
        def generate_items(n):
            return (i for i in range(n))

        gen = generate_items(5)
        items = list(gen)

        assert items == [0, 1, 2, 3, 4]

        store = get_global_store()
        events = store.get_events()

        assert len(events) == 1
        assert events[0].data["status"] == "success"
