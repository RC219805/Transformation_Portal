#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for progress tracking utilities."""

import time
import pytest

from transformation_portal.streaming.progress import (
    ProgressState,
    ProgressTracker,
    ProgressBar,
    MultiProgress,
    create_progress,
)


class TestProgressState:
    """Tests for ProgressState dataclass."""

    def test_initialization(self):
        """Test ProgressState initialization."""
        state = ProgressState(current=10, total=100, message="Processing")

        assert state.current == 10
        assert state.total == 100
        assert state.message == "Processing"
        assert not state.completed

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        state = ProgressState()
        time.sleep(0.1)

        elapsed = state.elapsed
        assert elapsed >= 0.1
        assert elapsed < 1.0

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        state = ProgressState(current=25, total=100)
        assert state.percentage == 25.0

    def test_percentage_with_no_total(self):
        """Test percentage when total is None."""
        state = ProgressState(current=10, total=None)
        assert state.percentage is None

    def test_percentage_with_zero_total(self):
        """Test percentage when total is zero."""
        state = ProgressState(current=0, total=0)
        assert state.percentage is None

    def test_eta_calculation(self):
        """Test ETA calculation."""
        state = ProgressState(current=50, total=100)

        # Mock elapsed time
        state.start_time = time.time() - 10  # Started 10 seconds ago

        eta = state.eta
        assert eta is not None
        assert eta > 0
        # Should take roughly same time for remaining half
        assert 8 < eta < 12

    def test_eta_with_no_progress(self):
        """Test ETA when no progress made."""
        state = ProgressState(current=0, total=100)
        assert state.eta is None

    def test_eta_with_no_total(self):
        """Test ETA when total is None."""
        state = ProgressState(current=10, total=None)
        assert state.eta is None


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ProgressTracker(total=100, description="Test task")

        assert tracker.state.total == 100
        assert tracker.state.message == "Test task"
        assert tracker.state.current == 0

    def test_update_progress(self):
        """Test updating progress."""
        tracker = ProgressTracker(total=100)

        tracker.update(10)
        assert tracker.state.current == 10

        tracker.update(5)
        assert tracker.state.current == 15

    def test_update_with_message(self):
        """Test updating with custom message."""
        tracker = ProgressTracker(total=100)

        tracker.update(10, message="Processing batch 1")
        assert tracker.state.message == "Processing batch 1"

    def test_auto_complete_on_reaching_total(self):
        """Test automatic completion when reaching total."""
        tracker = ProgressTracker(total=10)

        tracker.update(10)
        assert tracker.state.completed

    def test_manual_complete(self):
        """Test manual completion."""
        tracker = ProgressTracker(total=100)

        tracker.update(50)
        tracker.complete(message="Done early")

        assert tracker.state.completed
        assert tracker.state.message == "Done early"

    def test_callback_registration(self):
        """Test callback registration."""
        tracker = ProgressTracker(total=100, update_interval=0.0)
        callback_calls = []

        def on_update(state):
            callback_calls.append(state.current)

        tracker.on_update(on_update)
        tracker.update(10)
        tracker.update(20)

        assert len(callback_calls) >= 1

    def test_callback_error_handling(self):
        """Test that callback errors don't break tracker."""
        tracker = ProgressTracker(total=100, update_interval=0.0)

        def bad_callback(state):
            raise ValueError("Callback error")

        tracker.on_update(bad_callback)

        # Should not raise, just print error
        tracker.update(10)

    def test_update_interval_throttling(self):
        """Test that callbacks are throttled by update_interval."""
        tracker = ProgressTracker(total=100, update_interval=10.0)
        callback_calls = []

        def on_update(state):
            callback_calls.append(state.current)

        tracker.on_update(on_update)

        # Rapid updates should be throttled
        tracker.update(1)
        tracker.update(1)
        tracker.update(1)

        # Only first update should trigger callback (rest throttled)
        assert len(callback_calls) <= 1

    def test_get_state_thread_safe(self):
        """Test getting state copy."""
        tracker = ProgressTracker(total=100)
        tracker.update(50)

        state = tracker.get_state()

        # Should be a copy
        assert state.current == 50
        assert state.total == 100

        # Modifying copy shouldn't affect tracker
        tracker.update(10)
        assert state.current == 50  # Copy unchanged
        assert tracker.state.current == 60

    def test_indeterminate_progress(self):
        """Test tracker with no total (indeterminate)."""
        tracker = ProgressTracker(total=None)

        tracker.update(10)
        assert tracker.state.current == 10
        assert not tracker.state.completed


class TestProgressBar:
    """Tests for ProgressBar class."""

    def test_initialization(self):
        """Test progress bar initialization."""
        pbar = ProgressBar(total=100, description="Test", width=50)

        assert pbar.tracker.state.total == 100
        assert pbar.width == 50

    def test_update_delegates_to_tracker(self):
        """Test that update delegates to tracker."""
        pbar = ProgressBar(total=100)

        pbar.update(25)
        assert pbar.tracker.state.current == 25

    def test_context_manager(self):
        """Test using progress bar as context manager."""
        with ProgressBar(total=10) as pbar:
            pbar.update(5)
            assert pbar.tracker.state.current == 5

        # Should be completed after exiting context
        assert pbar.tracker.state.completed

    def test_context_manager_with_exception(self):
        """Test context manager still completes on exception."""
        pbar = None
        try:
            with ProgressBar(total=10) as pbar:
                pbar.update(5)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still be completed
        assert pbar.tracker.state.completed

    def test_render_with_total(self):
        """Test rendering with known total."""
        pbar = ProgressBar(total=100, width=10)
        pbar.update(50)

        # Just verify it doesn't crash
        pbar._render()

    def test_render_indeterminate(self):
        """Test rendering with no total."""
        pbar = ProgressBar(total=None)
        pbar.update(42)

        # Just verify it doesn't crash
        pbar._render()

    def test_render_only_when_changed(self):
        """Test that render only updates when line changes."""
        pbar = ProgressBar(total=100)
        pbar._last_render = "test"

        # Render shouldn't update if nothing changed
        # (This is more of an implementation detail test)
        pbar._render()


class TestMultiProgress:
    """Tests for MultiProgress class."""

    def test_initialization(self):
        """Test multi-progress initialization."""
        multi = MultiProgress()
        assert len(multi.tasks) == 0

    def test_add_task(self):
        """Test adding a task."""
        multi = MultiProgress()

        task_id = multi.add_task("Task 1", total=100)

        assert task_id in multi.tasks
        assert multi.tasks[task_id].state.total == 100

    def test_add_task_with_custom_id(self):
        """Test adding task with custom ID."""
        multi = MultiProgress()

        task_id = multi.add_task("Task 1", total=50, task_id="custom_id")

        assert task_id == "custom_id"
        assert "custom_id" in multi.tasks

    def test_auto_generated_task_ids(self):
        """Test auto-generated task IDs."""
        multi = MultiProgress()

        id1 = multi.add_task("Task 1", total=100)
        id2 = multi.add_task("Task 2", total=200)

        assert id1 != id2
        assert id1.startswith("task_")
        assert id2.startswith("task_")

    def test_update_task(self):
        """Test updating task progress."""
        multi = MultiProgress()

        task_id = multi.add_task("Task 1", total=100)
        multi.update(task_id, n=25)

        assert multi.tasks[task_id].state.current == 25

    def test_update_with_message(self):
        """Test updating task with message."""
        multi = MultiProgress()

        task_id = multi.add_task("Task 1", total=100)
        multi.update(task_id, n=10, message="Processing...")

        assert multi.tasks[task_id].state.message == "Processing..."

    def test_update_nonexistent_task(self):
        """Test updating non-existent task (should not error)."""
        multi = MultiProgress()

        # Should not raise
        multi.update("nonexistent", n=10)

    def test_get_summary(self):
        """Test getting summary of all tasks."""
        multi = MultiProgress()

        id1 = multi.add_task("Task 1", total=100)
        id2 = multi.add_task("Task 2", total=50)

        multi.update(id1, 25)
        multi.update(id2, 10)

        summary = multi.get_summary()

        assert id1 in summary
        assert id2 in summary
        assert summary[id1].current == 25
        assert summary[id2].current == 10


class TestCreateProgress:
    """Tests for create_progress factory function."""

    def test_create_basic_progress(self):
        """Test creating basic progress tracker."""
        tracker = create_progress(total=100, description="Test")

        assert isinstance(tracker, ProgressTracker)
        assert tracker.state.total == 100
        assert tracker.state.message == "Test"

    def test_create_indeterminate_progress(self):
        """Test creating indeterminate progress."""
        tracker = create_progress(description="Processing")

        assert tracker.state.total is None

    def test_create_with_rich_unavailable(self):
        """Test graceful fallback when rich is unavailable."""
        tracker = create_progress(total=50, use_rich=True)

        # Should still work even if rich is not installed
        assert isinstance(tracker, ProgressTracker)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
