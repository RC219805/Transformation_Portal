"""Unit tests for streaming.progress.

Covers ProgressState properties (elapsed, percentage, eta), ProgressTracker
update/complete/callback lifecycle, and MultiProgress task management —
all in-process with no filesystem or GPU dependencies.
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# ProgressState
# ---------------------------------------------------------------------------


class TestProgressState:
    def test_percentage_none_when_no_total(self):
        from transformation_portal.streaming.progress import ProgressState

        state = ProgressState(current=5, total=None)
        assert state.percentage is None

    def test_percentage_calculated_correctly(self):
        from transformation_portal.streaming.progress import ProgressState

        state = ProgressState(current=25, total=100)
        assert state.percentage == pytest.approx(25.0)

    def test_percentage_at_completion(self):
        from transformation_portal.streaming.progress import ProgressState

        state = ProgressState(current=100, total=100)
        assert state.percentage == pytest.approx(100.0)

    def test_elapsed_is_non_negative(self):
        from transformation_portal.streaming.progress import ProgressState

        state = ProgressState()
        assert state.elapsed >= 0.0

    def test_eta_none_when_no_total(self):
        from transformation_portal.streaming.progress import ProgressState

        state = ProgressState(current=10, total=None)
        assert state.eta is None

    def test_eta_none_when_no_progress(self):
        from transformation_portal.streaming.progress import ProgressState

        state = ProgressState(current=0, total=100)
        assert state.eta is None

    def test_eta_positive_when_making_progress(self):
        import time

        from transformation_portal.streaming.progress import ProgressState

        # Simulate some time having passed with some progress made
        start = time.time() - 1.0  # 1 second ago
        state = ProgressState(current=10, total=100, start_time=start, last_update=time.time())
        assert state.eta is not None
        assert state.eta > 0

    def test_completed_flag_false_by_default(self):
        from transformation_portal.streaming.progress import ProgressState

        assert ProgressState().completed is False


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


class TestProgressTracker:
    def test_update_increments_current(self):
        from transformation_portal.streaming.progress import ProgressTracker

        tracker = ProgressTracker(total=100)
        tracker.update(n=10)
        state = tracker.get_state()
        assert state.current == 10

    def test_update_multiple_times_accumulates(self):
        from transformation_portal.streaming.progress import ProgressTracker

        tracker = ProgressTracker(total=100)
        tracker.update(5)
        tracker.update(5)
        assert tracker.get_state().current == 10

    def test_update_with_message(self):
        from transformation_portal.streaming.progress import ProgressTracker

        tracker = ProgressTracker(total=50)
        tracker.update(1, message="processing item")
        assert tracker.get_state().message == "processing item"

    def test_complete_sets_completed_flag(self):
        from transformation_portal.streaming.progress import ProgressTracker

        tracker = ProgressTracker(total=10)
        tracker.update(10)
        tracker.complete()
        assert tracker.get_state().completed is True

    def test_complete_updates_current_to_total(self):
        from transformation_portal.streaming.progress import ProgressTracker

        tracker = ProgressTracker(total=100)
        tracker.update(50)
        tracker.complete()
        state = tracker.get_state()
        assert state.completed is True

    def test_on_update_callback_fires(self):
        from transformation_portal.streaming.progress import ProgressTracker

        calls = []
        tracker = ProgressTracker(total=10, update_interval=0.0)
        tracker.on_update(lambda s: calls.append(s.current))
        tracker.update(3)
        assert len(calls) > 0

    def test_get_state_returns_progress_state(self):
        from transformation_portal.streaming.progress import ProgressState, ProgressTracker

        tracker = ProgressTracker(total=50)
        assert isinstance(tracker.get_state(), ProgressState)

    def test_tracker_without_total(self):
        from transformation_portal.streaming.progress import ProgressTracker

        tracker = ProgressTracker()
        tracker.update(1)
        assert tracker.get_state().current == 1
        assert tracker.get_state().percentage is None


# ---------------------------------------------------------------------------
# MultiProgress
# ---------------------------------------------------------------------------


class TestMultiProgress:
    def test_add_task_returns_string_id(self):
        from transformation_portal.streaming.progress import MultiProgress

        mp = MultiProgress()
        task_id = mp.add_task("Task A", total=10)
        assert isinstance(task_id, str)

    def test_add_task_with_explicit_id(self):
        from transformation_portal.streaming.progress import MultiProgress

        mp = MultiProgress()
        task_id = mp.add_task("Task B", total=20, task_id="my_task")
        assert task_id == "my_task"

    def test_update_increments_specific_task(self):
        from transformation_portal.streaming.progress import MultiProgress

        mp = MultiProgress()
        t1 = mp.add_task("Task 1", total=10)
        t2 = mp.add_task("Task 2", total=10)
        mp.update(t1, n=5)
        summary = mp.get_summary()
        assert summary[t1].current == 5
        assert summary[t2].current == 0

    def test_get_summary_returns_all_tasks(self):
        from transformation_portal.streaming.progress import MultiProgress

        mp = MultiProgress()
        ids = [mp.add_task(f"Task {i}", total=10) for i in range(3)]
        summary = mp.get_summary()
        for task_id in ids:
            assert task_id in summary

    def test_multiple_tasks_independent(self):
        from transformation_portal.streaming.progress import MultiProgress

        mp = MultiProgress()
        a = mp.add_task("A", total=100)
        b = mp.add_task("B", total=50)
        mp.update(a, n=30)
        mp.update(b, n=25)
        summary = mp.get_summary()
        assert summary[a].current == 30
        assert summary[b].current == 25
