#!/usr/bin/env python3
"""Negative-path coverage for EventStore loading.

`tests/events/test_store.py` covers the happy path plus a single
"invalid JSON" smoke test. This file targets the remaining ways an event
file can be malformed without taking the store down:

- Missing required fields (TypeError in Event.from_dict)
- JSON that decodes to a non-dict (list, string)
- Empty file
- Binary garbage / NUL bytes
- Mixed valid/invalid files: valid events must still load
- The error path uses ``print()`` to stdout (regression: this is the
  swallowed-error smell flagged in
  docs/testing/test_coverage_improvement_plan.md). If the implementation moves
  to logging, this test should be updated rather than silently passing.
"""

from __future__ import annotations

import json
import time

import pytest

from transformation_portal.events.store import Event, EventStore

pytestmark = pytest.mark.unit


def _write_event(storage_path, event):
    """Persist a valid event the same way EventStore does."""
    date_dir = storage_path / time.strftime("%Y-%m-%d", time.localtime(event.timestamp))
    date_dir.mkdir(parents=True, exist_ok=True)
    (date_dir / f"{event.id}.json").write_text(json.dumps(event.to_dict()))


def test_event_with_missing_required_fields_is_swallowed(tmp_path, capsys):
    storage_path = tmp_path / "events"
    storage_path.mkdir()

    # Drop the required `type` field — Event.from_dict will raise TypeError.
    bad = storage_path / "bad.json"
    bad.write_text(json.dumps({"id": "x", "timestamp": 1.0, "data": {}}))

    store = EventStore(storage_path)

    captured = capsys.readouterr()
    assert "Failed to load event" in captured.out
    assert store.get_events() == []


def test_event_payload_that_is_not_a_dict_is_swallowed(tmp_path, capsys):
    storage_path = tmp_path / "events"
    storage_path.mkdir()

    # Top-level JSON is a list, not the expected object — Event(**[...]) fails.
    (storage_path / "list.json").write_text(json.dumps(["a", "b"]))

    store = EventStore(storage_path)

    captured = capsys.readouterr()
    assert "Failed to load event" in captured.out
    assert store.get_events() == []


def test_empty_event_file_is_swallowed(tmp_path, capsys):
    storage_path = tmp_path / "events"
    storage_path.mkdir()
    (storage_path / "empty.json").write_text("")

    store = EventStore(storage_path)
    captured = capsys.readouterr()
    assert "Failed to load event" in captured.out
    assert store.get_events() == []


def test_binary_garbage_event_file_is_swallowed(tmp_path, capsys):
    storage_path = tmp_path / "events"
    storage_path.mkdir()
    (storage_path / "garbage.json").write_bytes(b"\x00\x01\x02\xff\xfeNot JSON")

    store = EventStore(storage_path)
    captured = capsys.readouterr()
    assert "Failed to load event" in captured.out
    assert store.get_events() == []


def test_mixed_valid_and_invalid_files_loads_only_the_valid_ones(tmp_path, capsys):
    """A single corrupt event file must NOT prevent loading the rest of the
    audit log. This is the highest-value invariant for an audit store."""
    storage_path = tmp_path / "events"
    storage_path.mkdir()

    good = Event(id="good-1", type="image.processed", timestamp=time.time(), data={"k": "v"})
    _write_event(storage_path, good)

    bad_dir = storage_path / "bad-day"
    bad_dir.mkdir()
    (bad_dir / "broken.json").write_text("not-json{{{")

    store = EventStore(storage_path)
    captured = capsys.readouterr()
    assert "Failed to load event" in captured.out

    events = store.get_events()
    assert len(events) == 1
    assert events[0].id == "good-1"


def test_corrupt_event_warning_is_written_to_stdout_not_stderr(tmp_path, capsys):
    """Regression / smell guard: load-failure currently uses ``print()``, which
    sends to stdout. If this is changed to a logger or stderr, update this test
    along with the call site — silent failures are a real audit-log risk."""
    storage_path = tmp_path / "events"
    storage_path.mkdir()
    (storage_path / "broken.json").write_text("not-json{{{")

    EventStore(storage_path)
    captured = capsys.readouterr()
    assert "Failed to load event" in captured.out
    assert "Failed to load event" not in captured.err


def test_event_files_with_non_json_extension_are_ignored(tmp_path, capsys):
    """rglob('*.json') must not pick up unrelated files in the storage tree."""
    storage_path = tmp_path / "events"
    storage_path.mkdir()
    # Decoy files that should not be considered events.
    (storage_path / "README.md").write_text("not an event")
    (storage_path / "junk.txt").write_text("definitely not json")

    store = EventStore(storage_path)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert store.get_events() == []
