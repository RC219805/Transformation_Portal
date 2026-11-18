"""Contracts of the legacy per-stage JSON/NPY cache in ``Stage.execute``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus

pytestmark = pytest.mark.unit


class _ArtifactStage(Stage):
    """Stage that returns whatever artifacts it is currently configured with."""

    def __init__(self, artifacts: Dict[str, Any]):
        super().__init__("artifact_stage")
        self.artifacts = artifacts
        self.calls = 0

    def get_cache_key(self, context: StageContext) -> str:
        return "fixed-key"

    def compute(self, context: StageContext) -> StageResult:
        self.calls += 1
        return StageResult(self.name, self.version, StageStatus.COMPLETED, artifacts=dict(self.artifacts))


def _context(cache_dir: Path) -> StageContext:
    return StageContext(cache_enabled=True, cache_dir=cache_dir)


def test_result_that_is_not_json_serializable_leaves_no_cache_entry(tmp_path: Path):
    """A failed cache write must not publish the part it managed to serialize."""
    stage = _ArtifactStage({"depth_map": np.ones((2, 2), dtype=np.float32), "written_to": tmp_path / "depth.npy"})

    first = stage.execute(_context(tmp_path / "cache"))
    second = stage.execute(_context(tmp_path / "cache"))

    assert first.is_success() and second.is_success()
    assert (first.cache_hit, second.cache_hit) == (False, False)
    assert stage.calls == 2
    assert [path.name for path in (tmp_path / "cache").iterdir() if path.suffix != ".npy"] == []


def test_failed_cache_write_keeps_the_entry_that_was_already_there(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    stage = _ArtifactStage({"score": 0.5})
    assert stage.execute(_context(cache_dir)).is_success()
    entry = cache_dir / "fixed-key.json"
    published = entry.read_text()

    unserializable = StageResult(stage.name, stage.version, StageStatus.COMPLETED, artifacts={"written_to": tmp_path})
    stage._save_to_cache(unserializable, "fixed-key", cache_dir)

    assert entry.read_text() == published
    assert json.loads(published)["artifacts"] == {"score": 0.5}
    assert [path.name for path in cache_dir.iterdir()] == ["fixed-key.json"]
    cached = stage.execute(_context(cache_dir))
    assert cached.cache_hit is True and cached.artifacts == {"score": 0.5}


def test_serializable_result_round_trips_through_the_cache(tmp_path: Path):
    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    stage = _ArtifactStage({"depth_map": depth, "stats": {"mean": 2.5}})

    first = stage.execute(_context(tmp_path))
    second = stage.execute(_context(tmp_path))

    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert stage.calls == 1
    assert second.status == StageStatus.CACHED
    assert np.array_equal(second.artifacts["depth_map"], depth)
    assert second.artifacts["stats"] == {"mean": 2.5}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fixed-key.json", "fixed-key_depth_map.npy"]
