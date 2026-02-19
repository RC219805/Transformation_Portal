"""Tests for orchestrator sequence lifecycle reset (ADR-026 §2.3 / §1.2).

Validates that:
- SpatialAIPipeline tracks stateful backends.
- reset_sequence() delegates to all registered backends.
- process() with sequence_id triggers a reset before execution.
- Non-stateful backends are skipped gracefully.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.spatial_ai.orchestration.pipeline import PipelineConfig, SpatialAIPipeline


def _make_pipeline(**kwargs) -> SpatialAIPipeline:
    """Create a minimal SpatialAIPipeline for testing."""
    config = PipelineConfig(tier="standard", stages=["ingest"], **kwargs)
    return SpatialAIPipeline(config)


class TestStatefulBackendRegistration:
    """Tests for register_stateful_backend()."""

    def test_register_backend_with_reset_state(self):
        """Backends with reset_state() should be tracked."""
        pipeline = _make_pipeline()
        mock_backend = MagicMock(spec=["reset_state"])

        pipeline.register_stateful_backend("depth_ensemble", mock_backend)

        assert "depth_ensemble" in pipeline._stateful_backends

    def test_skip_backend_without_reset_state(self):
        """Backends without reset_state() should NOT be registered."""
        pipeline = _make_pipeline()
        mock_backend = MagicMock(spec=["compute"])  # No reset_state

        pipeline.register_stateful_backend("plain_backend", mock_backend)

        assert "plain_backend" not in pipeline._stateful_backends

    def test_skip_backend_with_non_callable_reset_state(self):
        """Backends with non-callable reset_state attributes should be skipped."""
        pipeline = _make_pipeline()
        backend = type("Backend", (), {"reset_state": "not-callable"})()

        pipeline.register_stateful_backend("broken_backend", backend)

        assert "broken_backend" not in pipeline._stateful_backends


class TestResetSequence:
    """Tests for reset_sequence()."""

    def test_reset_calls_all_backends(self):
        """reset_sequence() should call reset_state() on every registered backend."""
        pipeline = _make_pipeline()

        backend_a = MagicMock(spec=["reset_state"])
        backend_b = MagicMock(spec=["reset_state"])

        pipeline.register_stateful_backend("a", backend_a)
        pipeline.register_stateful_backend("b", backend_b)

        pipeline.reset_sequence("seq_42")

        backend_a.reset_state.assert_called_once_with("seq_42")
        backend_b.reset_state.assert_called_once_with("seq_42")

    def test_reset_with_none_sequence_id(self):
        """reset_sequence(None) should still call reset_state(None)."""
        pipeline = _make_pipeline()
        backend = MagicMock(spec=["reset_state"])
        pipeline.register_stateful_backend("x", backend)

        pipeline.reset_sequence(None)

        backend.reset_state.assert_called_once_with(None)

    def test_reset_tolerates_failing_backend(self):
        """A failing backend should not prevent reset of others."""
        pipeline = _make_pipeline()

        backend_ok = MagicMock(spec=["reset_state"])
        backend_bad = MagicMock(spec=["reset_state"])
        backend_bad.reset_state.side_effect = RuntimeError("boom")

        pipeline.register_stateful_backend("ok", backend_ok)
        pipeline.register_stateful_backend("bad", backend_bad)

        # Should not raise
        pipeline.reset_sequence("seq_err")

        # Both should have been called
        backend_bad.reset_state.assert_called_once()
        backend_ok.reset_state.assert_called_once()


class TestProcessSequenceId:
    """Tests for process() with sequence_id parameter."""

    def test_process_with_sequence_id_triggers_reset(self, tmp_path):
        """Passing sequence_id to process() should call reset_sequence()."""
        pipeline = _make_pipeline()
        backend = MagicMock(spec=["reset_state"])
        pipeline.register_stateful_backend("depth", backend)

        # Create a dummy input file
        input_file = tmp_path / "test.tiff"
        input_file.write_bytes(b"dummy")
        output_dir = tmp_path / "output"

        # Mock the _run_ingest to avoid needing real ingest
        with patch.object(pipeline, "_run_ingest", return_value=MagicMock()):
            pipeline.process(input_file, output_dir, sequence_id="video_001")

        backend.reset_state.assert_called_once_with("video_001")

    def test_process_without_sequence_id_no_reset(self, tmp_path):
        """Not passing sequence_id should not call reset_sequence()."""
        pipeline = _make_pipeline()
        backend = MagicMock(spec=["reset_state"])
        pipeline.register_stateful_backend("depth", backend)

        input_file = tmp_path / "test.tiff"
        input_file.write_bytes(b"dummy")
        output_dir = tmp_path / "output"

        with patch.object(pipeline, "_run_ingest", return_value=MagicMock()):
            pipeline.process(input_file, output_dir)

        backend.reset_state.assert_not_called()

    def test_two_sequences_produce_independent_resets(self, tmp_path):
        """Two consecutive process() calls with different sequence_ids should reset each time."""
        pipeline = _make_pipeline()
        backend = MagicMock(spec=["reset_state"])
        pipeline.register_stateful_backend("depth", backend)

        input_file = tmp_path / "test.tiff"
        input_file.write_bytes(b"dummy")
        output_dir = tmp_path / "output"

        with patch.object(pipeline, "_run_ingest", return_value=MagicMock()):
            pipeline.process(input_file, output_dir, sequence_id="seq_A")
            pipeline.process(input_file, output_dir, sequence_id="seq_B")

        assert backend.reset_state.call_count == 2
        calls = [c.args[0] for c in backend.reset_state.call_args_list]
        assert calls == ["seq_A", "seq_B"]


# Pytest markers
pytestmark = [
    pytest.mark.unit,
]
