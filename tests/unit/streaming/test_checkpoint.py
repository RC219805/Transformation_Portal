"""Unit tests for streaming.checkpoint.

Covers EvolutionaryCheckpoint.evolve_or_alert() (stable / evolve / exact-horizon
boundary), Checkpoint save/load round-trip, and CheckpointManager lifecycle
(create, save, get_latest, list_checkpoints, clear) — all in-process with
tmp_path, no network or GPU required.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# EvolutionaryCheckpoint
# ---------------------------------------------------------------------------


class TestEvolutionaryCheckpoint:
    def test_stable_before_horizon_returns_ok_string(self):
        from transformation_portal.streaming.checkpoint import EvolutionaryCheckpoint

        yesterday = date.today() - timedelta(days=1)
        ec = EvolutionaryCheckpoint(horizon=date.today() + timedelta(days=30), mutation_path="migrate_v2")
        result = ec.evolve_or_alert(today=yesterday)
        # "stable" path — result is a human-readable string mentioning days remaining
        assert isinstance(result, str)
        assert len(result) > 0

    def test_evolve_required_after_horizon(self):
        from transformation_portal.streaming.checkpoint import EvolutionaryCheckpoint

        past_horizon = date.today() - timedelta(days=1)
        ec = EvolutionaryCheckpoint(horizon=past_horizon, mutation_path="migrate_v2")
        result = ec.evolve_or_alert(today=date.today())
        assert isinstance(result, str)
        assert "migrate_v2" in result

    def test_exact_horizon_date_is_stable_or_evolve(self):
        """On the exact horizon date the method must return a non-empty string."""
        from transformation_portal.streaming.checkpoint import EvolutionaryCheckpoint

        today = date.today()
        ec = EvolutionaryCheckpoint(horizon=today, mutation_path="migrate_v3")
        result = ec.evolve_or_alert(today=today)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_default_today_uses_system_date(self):
        """evolve_or_alert() with no 'today' argument should not raise."""
        from transformation_portal.streaming.checkpoint import EvolutionaryCheckpoint

        far_future = date.today() + timedelta(days=9999)
        ec = EvolutionaryCheckpoint(horizon=far_future, mutation_path="someday")
        result = ec.evolve_or_alert()  # uses real system date
        assert isinstance(result, str)

    def test_mutation_path_stored(self):
        from transformation_portal.streaming.checkpoint import EvolutionaryCheckpoint

        ec = EvolutionaryCheckpoint(horizon=date.today(), mutation_path="my_path")
        assert ec.mutation_path == "my_path"

    def test_frozen_dataclass_rejects_mutation(self):
        from transformation_portal.streaming.checkpoint import EvolutionaryCheckpoint

        ec = EvolutionaryCheckpoint(horizon=date.today(), mutation_path="path")
        with pytest.raises((AttributeError, TypeError)):
            ec.horizon = date.today() + timedelta(days=1)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Checkpoint dataclass — save / load round-trip
# ---------------------------------------------------------------------------


class TestCheckpointSaveLoad:
    def test_save_creates_file(self, tmp_path):
        from transformation_portal.streaming.checkpoint import Checkpoint

        ckpt = Checkpoint(
            id="ckpt-001",
            progress=0.5,
            state={"key": "value"},
            timestamp=time.time(),
            metadata={"op": "test"},
        )
        path = tmp_path / "checkpoint.json"
        ckpt.save(path)
        assert path.exists()

    def test_load_round_trip(self, tmp_path):
        from transformation_portal.streaming.checkpoint import Checkpoint

        ckpt = Checkpoint(
            id="round-trip",
            progress=0.75,
            state={"items_done": 75},
            timestamp=1234567890.0,
            metadata={},
        )
        path = tmp_path / "ckpt.json"
        ckpt.save(path)
        loaded = Checkpoint.load(path)
        assert loaded.id == "round-trip"
        assert loaded.progress == pytest.approx(0.75)
        assert loaded.state["items_done"] == 75

    def test_load_missing_file_raises(self, tmp_path):
        from transformation_portal.streaming.checkpoint import Checkpoint

        with pytest.raises(FileNotFoundError):
            Checkpoint.load(tmp_path / "nonexistent.json")

    def test_progress_stored_correctly(self, tmp_path):
        from transformation_portal.streaming.checkpoint import Checkpoint

        ckpt = Checkpoint(id="p", progress=0.333, state={}, timestamp=0.0, metadata={})
        path = tmp_path / "p.json"
        ckpt.save(path)
        loaded = Checkpoint.load(path)
        assert loaded.progress == pytest.approx(0.333, rel=1e-5)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_create_checkpoint_returns_checkpoint(self, tmp_path):
        from transformation_portal.streaming.checkpoint import Checkpoint, CheckpointManager

        mgr = CheckpointManager("op-1", checkpoint_dir=tmp_path)
        ckpt = mgr.create_checkpoint(progress=0.1, state={"step": 1})
        assert isinstance(ckpt, Checkpoint)

    def test_create_checkpoint_progress_stored(self, tmp_path):
        from transformation_portal.streaming.checkpoint import CheckpointManager

        mgr = CheckpointManager("op-2", checkpoint_dir=tmp_path)
        ckpt = mgr.create_checkpoint(progress=0.42, state={})
        assert ckpt.progress == pytest.approx(0.42)

    def test_save_creates_file(self, tmp_path):
        from transformation_portal.streaming.checkpoint import CheckpointManager

        mgr = CheckpointManager("op-3", checkpoint_dir=tmp_path)
        ckpt = mgr.create_checkpoint(progress=0.5, state={"x": 1})
        saved_path = mgr.save(ckpt)
        assert saved_path.exists()

    def test_get_latest_returns_most_recent(self, tmp_path):
        from transformation_portal.streaming.checkpoint import CheckpointManager

        mgr = CheckpointManager("op-4", checkpoint_dir=tmp_path)
        for i in range(3):
            ckpt = mgr.create_checkpoint(progress=i * 0.1, state={"step": i})
            mgr.save(ckpt)
        latest = mgr.get_latest()
        assert latest is not None
        assert latest.progress == pytest.approx(0.2, abs=0.01)

    def test_get_latest_returns_none_when_empty(self, tmp_path):
        from transformation_portal.streaming.checkpoint import CheckpointManager

        mgr = CheckpointManager("op-empty", checkpoint_dir=tmp_path / "empty")
        assert mgr.get_latest() is None

    def test_list_checkpoints_returns_correct_count(self, tmp_path):
        from transformation_portal.streaming.checkpoint import CheckpointManager

        mgr = CheckpointManager("op-list", checkpoint_dir=tmp_path)
        for i in range(4):
            ckpt = mgr.create_checkpoint(progress=i * 0.25, state={})
            mgr.save(ckpt)
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 4

    def test_clear_removes_all_checkpoints(self, tmp_path):
        from transformation_portal.streaming.checkpoint import CheckpointManager

        mgr = CheckpointManager("op-clear", checkpoint_dir=tmp_path)
        for i in range(3):
            mgr.save(mgr.create_checkpoint(progress=i * 0.1, state={}))
        mgr.clear()
        assert mgr.list_checkpoints() == []
        assert mgr.get_latest() is None
