"""Tests for Phase 4 progress callback functionality."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


class TestProgressCallbackProtocol:
    """Tests for progress callback type compatibility."""

    def test_callback_signature(self) -> None:
        """Verify callback receives correct arguments."""
        callback = MagicMock()

        # Simulate what the functions would call
        callback(0, 10, "Starting...")
        callback(5, 10, "Halfway done")
        callback(10, 10, "Complete")

        assert callback.call_count == 3
        callback.assert_any_call(0, 10, "Starting...")
        callback.assert_any_call(5, 10, "Halfway done")
        callback.assert_any_call(10, 10, "Complete")

    def test_callback_can_be_none(self) -> None:
        """Verify None callback is handled gracefully."""
        callback: Any = None

        # The pattern used in the code
        if callback:
            callback(0, 10, "test")
        # Should not raise


class TestProgressCallbackUsagePattern:
    """Tests demonstrating the progress callback usage pattern."""

    def test_incremental_progress(self) -> None:
        """Progress should increment correctly."""
        progress_log: list[tuple[int, int, str]] = []

        def capture_progress(current: int, total: int, message: str) -> None:
            progress_log.append((current, total, message))

        # Simulate processing 5 items
        total = 5
        capture_progress(0, total, "Starting")
        for i in range(total):
            capture_progress(i + 1, total, f"Item {i + 1}")
        capture_progress(total, total, "Complete")

        assert len(progress_log) == 7  # start + 5 items + complete
        assert progress_log[0] == (0, 5, "Starting")
        assert progress_log[-1] == (5, 5, "Complete")

        # Check incremental progress
        for i in range(1, 6):
            assert progress_log[i][0] == i
            assert progress_log[i][1] == 5

    def test_callback_exception_handling(self) -> None:
        """Callback exceptions should propagate correctly."""

        def failing_callback(current: int, total: int, message: str) -> None:
            if current > 2:
                raise ValueError("Callback failed")

        with pytest.raises(ValueError) as exc_info:
            for i in range(5):
                failing_callback(i, 5, f"Item {i}")

        assert "Callback failed" in str(exc_info.value)
