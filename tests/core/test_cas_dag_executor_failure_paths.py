#!/usr/bin/env python3
"""Failure-path coverage for CASDAGExecutor.

`tests/determinism/test_execution_wrapper.py` covers the success path: simple
execution, full-cache reruns, partial reuse, and provenance. This file targets
the under-covered paths that decide what happens when things go wrong:

- A stage returns a failed StageResult
- A stage raises an exception mid-execution
- _check_cache encounters corrupted JSON, missing CAS objects, or schema mismatch
- invalidate() with no cache dir, with `before` time filter, and over-broad
- enable_provenance=False path (merkle_dag is None)
- enable_caching=False short-circuits cache reads
- _cache_path / _get_lock filename safety
"""

# pytest fixtures are passed by parameter name, so pylint flags every test
# function that takes the fixture as a parameter. Suppress at module scope.
# pylint: disable=redefined-outer-name

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import List
from unittest.mock import patch

import pytest

from transformation_portal.core.cas_dag_executor import (
    CASDAGConfig,
    CASDAGExecutor,
)
from transformation_portal.core.execution_wrapper import FileLock
from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import (
    Stage,
    StageContext,
    StageResult,
    StageStatus,
)
from transformation_portal.storage.cas_store import ArtifactStore

pytestmark = pytest.mark.unit


class _RecordingStage(Stage):
    """Configurable stage: success / explicit failure / raises exception."""

    def __init__(
        self,
        name: str,
        deps: List[str] | None = None,
        *,
        mode: str = "success",
        version: str = "1.0.0",
    ):
        super().__init__(name, version)
        self._deps = deps or []
        self.mode = mode
        self.calls = 0

    def get_dependencies(self) -> List[str]:
        return self._deps

    def get_cache_key(self, context: StageContext) -> str:
        return hashlib.sha256(f"{self.name}:{self.version}".encode()).hexdigest()

    def compute(self, context: StageContext) -> StageResult:
        self.calls += 1
        if self.mode == "raise":
            raise RuntimeError(f"intentional failure in {self.name}")
        if self.mode == "fail":
            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                error="boom",
            )
        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={f"{self.name}_out": self.calls},
        )


@pytest.fixture
def make_executor(tmp_path):
    def _make(**config_overrides):
        store = ArtifactStore(tmp_path / "cas")
        config = CASDAGConfig(**config_overrides)
        return CASDAGExecutor(store, tmp_path / "cache", config)

    return _make


@pytest.fixture(autouse=True)
def _stable_identity():
    """Pin code-hash and env-fingerprint so cache identities are deterministic."""
    with (
        patch(
            "transformation_portal.core.execution_identity.compute_code_hash",
            return_value="sha256:fixed_code",
        ),
        patch(
            "transformation_portal.core.execution_identity.get_env_fingerprint",
            return_value="sha256:fixed_env",
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Stage-failure propagation
# ---------------------------------------------------------------------------


def test_failed_stage_short_circuits_dag_and_reports_error(make_executor):
    executor = make_executor()
    graph = StageGraph("fail-pipeline")
    failing = _RecordingStage("first", mode="fail")
    downstream = _RecordingStage("second", deps=["first"])
    graph.add_stage(failing)
    graph.add_stage(downstream)

    result = executor.execute(graph, StageContext())

    assert result.success is False
    assert result.error is not None
    assert "first" in result.error
    # Downstream must not have been invoked.
    assert downstream.calls == 0
    # The failing stage's result is preserved for diagnostics.
    assert "first" in result.stage_results
    assert result.stage_results["first"].is_success() is False


def test_failed_stage_result_is_not_cached(make_executor):
    executor = make_executor()
    graph = StageGraph("fail-pipeline")
    failing = _RecordingStage("only", mode="fail")
    graph.add_stage(failing)

    result1 = executor.execute(graph, StageContext())
    assert result1.success is False
    assert failing.calls == 1

    # Re-running must execute the stage again — failures must never be cached.
    result2 = executor.execute(graph, StageContext())
    assert result2.success is False
    assert failing.calls == 2


def test_stage_compute_raise_is_wrapped_into_failed_result(make_executor):
    """A raise inside Stage.compute() is caught by Stage.execute() -> FAILED."""
    executor = make_executor()
    graph = StageGraph("raise-pipeline")
    raiser = _RecordingStage("raiser", mode="raise")
    graph.add_stage(raiser)

    result = executor.execute(graph, StageContext())

    assert result.success is False
    assert result.error is not None
    assert "intentional failure" in result.error
    # The stage was executed and counted as a miss before failing.
    assert result.cache_misses == 1
    assert result.stage_results["raiser"].is_success() is False


def test_executor_level_exception_is_caught_and_reported(make_executor):
    """An exception from _compute_stage_identity hits the executor's broad
    except block (cas_dag_executor.py:490) and must surface in the result."""
    executor = make_executor()
    graph = StageGraph("identity-fail")
    graph.add_stage(_RecordingStage("only"))

    def boom(*args, **kwargs):
        raise RuntimeError("identity computation imploded")

    with patch.object(executor, "_compute_stage_identity", side_effect=boom):
        result = executor.execute(graph, StageContext())

    assert result.success is False
    assert result.error is not None
    assert "identity computation imploded" in result.error
    # No stage was ever executed because identity failed first.
    assert result.cache_misses == 0
    assert result.cache_hits == 0
    assert result.stage_results == {}


# ---------------------------------------------------------------------------
# _check_cache failure modes
# ---------------------------------------------------------------------------


def test_corrupted_cache_file_falls_back_to_recompute(make_executor):
    executor = make_executor()
    graph = StageGraph("corrupt-pipeline")
    stage = _RecordingStage("only")
    graph.add_stage(stage)

    # First run populates the cache.
    result1 = executor.execute(graph, StageContext())
    assert result1.success is True
    assert stage.calls == 1

    # Corrupt every cache JSON file written for this stage.
    cache_root = executor.cache_dir / "dag_cache"
    cache_files = list(cache_root.rglob("*.json"))
    assert cache_files, "expected at least one cache file"
    for path in cache_files:
        path.write_text("not-json{{{")

    # Re-run must recover by recomputing rather than blowing up.
    result2 = executor.execute(graph, StageContext())
    assert result2.success is True
    assert stage.calls == 2  # recomputed
    assert result2.cache_misses == 1


def test_schema_version_mismatch_invalidates_cache_entry(make_executor):
    executor = make_executor()
    graph = StageGraph("schema-pipeline")
    stage = _RecordingStage("only")
    graph.add_stage(stage)

    result1 = executor.execute(graph, StageContext())
    assert result1.success is True
    cache_files = list((executor.cache_dir / "dag_cache").rglob("*.json"))
    assert cache_files

    # Tamper with schema_version to force a mismatch on next read.
    for path in cache_files:
        data = json.loads(path.read_text())
        data["schema_version"] = "totally-different"
        path.write_text(json.dumps(data))

    result2 = executor.execute(graph, StageContext())
    assert result2.success is True
    assert stage.calls == 2
    assert result2.cache_misses == 1


def test_missing_cache_file_returns_no_hit(make_executor):
    executor = make_executor()
    graph = StageGraph("nofile-pipeline")
    stage = _RecordingStage("only")
    graph.add_stage(stage)

    # Without ever populating the cache, _check_cache must return None and
    # downstream code must treat it as a miss.
    result = executor.execute(graph, StageContext())
    assert result.success is True
    assert result.cache_misses == 1
    assert result.cache_hits == 0


# ---------------------------------------------------------------------------
# invalidate()
# ---------------------------------------------------------------------------


def test_invalidate_returns_zero_when_cache_dir_absent(make_executor):
    executor = make_executor()
    # Remove the dag_cache directory before invalidate runs.
    cache_root = executor.cache_dir / "dag_cache"
    if cache_root.exists():
        for child in cache_root.rglob("*"):
            if child.is_file():
                child.unlink()
        cache_root.rmdir()
    assert not cache_root.exists()

    assert executor.invalidate() == 0


def test_invalidate_with_before_filter_only_removes_old_entries(make_executor, tmp_path):
    executor = make_executor()
    graph = StageGraph("before-pipeline")
    stage = _RecordingStage("only")
    graph.add_stage(stage)

    executor.execute(graph, StageContext())
    cache_files = list((executor.cache_dir / "dag_cache").rglob("*.json"))
    assert cache_files

    # `before` strictly in the past must NOT remove an entry whose cached_at is "now".
    way_back = _dt.datetime(2000, 1, 1, tzinfo=_dt.timezone.utc)
    assert executor.invalidate(before=way_back) == 0
    assert all(p.exists() for p in cache_files)

    # `before` in the far future MUST remove the entry.
    far_future = _dt.datetime(9999, 1, 1, tzinfo=_dt.timezone.utc)
    removed = executor.invalidate(before=far_future)
    assert removed == len(cache_files)
    assert all(not p.exists() for p in cache_files)


def test_invalidate_with_unknown_stage_name_is_no_op(make_executor):
    executor = make_executor()
    graph = StageGraph("unknown-pipeline")
    stage = _RecordingStage("only")
    graph.add_stage(stage)

    executor.execute(graph, StageContext())
    cache_files_before = list((executor.cache_dir / "dag_cache").rglob("*.json"))
    assert cache_files_before

    removed = executor.invalidate(stage_names=["never-existed"])
    assert removed == 0
    cache_files_after = list((executor.cache_dir / "dag_cache").rglob("*.json"))
    assert {p.name for p in cache_files_before} == {p.name for p in cache_files_after}


def test_invalidate_skips_unreadable_cache_files(make_executor):
    executor = make_executor()
    graph = StageGraph("unreadable-pipeline")
    stage = _RecordingStage("only")
    graph.add_stage(stage)

    executor.execute(graph, StageContext())
    cache_files = list((executor.cache_dir / "dag_cache").rglob("*.json"))
    assert cache_files
    for path in cache_files:
        path.write_text("not-json{{{")

    # Corrupted files are simply skipped; invalidate must not raise.
    removed = executor.invalidate()
    assert removed == 0


# ---------------------------------------------------------------------------
# Configuration toggles
# ---------------------------------------------------------------------------


def test_enable_provenance_false_returns_no_merkle_dag(make_executor):
    executor = make_executor(enable_provenance=False)
    graph = StageGraph("no-provenance")
    graph.add_stage(_RecordingStage("a"))
    graph.add_stage(_RecordingStage("b", deps=["a"]))

    result = executor.execute(graph, StageContext())
    assert result.success is True
    assert result.merkle_dag is None


def test_enable_caching_false_always_recomputes(make_executor):
    executor = make_executor(enable_caching=False)
    graph = StageGraph("nocache")
    stage = _RecordingStage("only")
    graph.add_stage(stage)

    executor.execute(graph, StageContext())
    executor.execute(graph, StageContext())
    executor.execute(graph, StageContext())
    assert stage.calls == 3  # cache short-circuit means every run executes


def test_lock_dir_is_created_on_init(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    executor = CASDAGExecutor(store, tmp_path / "cache", CASDAGConfig())
    assert executor.locks_dir.is_dir()
    assert executor.locks_dir == tmp_path / "cache" / ".locks"


def test_get_lock_sanitizes_cas_id_into_filename(tmp_path):
    """The cas_id portion of the lock filename is sanitized.

    Documents what is actually safe today: ``cas_id`` is run through
    ``_sanitize_cas_id_for_filename`` before being interpolated, so its
    value cannot introduce path separators into the lock filename. The
    ``stage_name`` portion is NOT sanitized (callers pass internal stage
    names that we control), which is reflected in
    ``test_get_lock_with_separator_in_stage_name_stays_under_locks_dir``.
    """
    store = ArtifactStore(tmp_path / "cas")
    executor = CASDAGExecutor(store, tmp_path / "cache", CASDAGConfig())

    # cas_id with characters that are unsafe as filename components.
    lock = executor._get_lock("stage", "sha256:" + "a" * 64 + "/../etc")
    assert isinstance(lock, FileLock)
    # Sanitization must keep the file inside locks_dir.
    resolved = lock.lock_path.resolve()
    assert resolved.is_relative_to(executor.locks_dir.resolve())
    # And the cas_id-derived suffix must not introduce a separator.
    name = lock.lock_path.name
    assert "/" not in name
    assert "\\" not in name


def test_get_lock_with_separator_in_stage_name_stays_under_locks_dir(tmp_path):
    """Regression: a slash-bearing stage_name must not carve a sub-directory
    out of ``locks_dir``. The sanitizer replaces ``/`` with ``_`` so the lock
    file lands as a regular file directly inside ``locks_dir``."""
    store = ArtifactStore(tmp_path / "cas")
    executor = CASDAGExecutor(store, tmp_path / "cache", CASDAGConfig())

    lock = executor._get_lock("stage_with/slash", "sha256:" + "a" * 64)
    resolved = lock.lock_path.resolve()
    # The full resolved path must stay inside locks_dir, AND must be a direct
    # child of it (no carved-out sub-directory).
    assert resolved.is_relative_to(executor.locks_dir.resolve())
    assert resolved.parent == executor.locks_dir.resolve()


def test_get_lock_rejects_dotdot_traversal_in_stage_name(tmp_path):
    """Regression: ``stage_name=".."`` (or any ``..``-prefixed value) must not
    let the lock path escape ``locks_dir``. Before the sanitizer was added,
    ``_get_lock("../escape", ...)`` resolved to a sibling of ``locks_dir``."""
    store = ArtifactStore(tmp_path / "cas")
    executor = CASDAGExecutor(store, tmp_path / "cache", CASDAGConfig())

    lock = executor._get_lock("../escape", "sha256:" + "a" * 64)
    resolved = lock.lock_path.resolve()
    assert resolved.is_relative_to(
        executor.locks_dir.resolve()
    ), f"lock escaped locks_dir: {resolved} not under {executor.locks_dir.resolve()}"
    assert resolved.parent == executor.locks_dir.resolve()


def test_sanitize_stage_name_handles_traversal_aliases():
    """Lock the sanitizer's contract: every traversal-style alias produces a
    name that cannot navigate the directory tree."""
    sanitize = CASDAGExecutor._sanitize_stage_name_for_filename
    # Pure traversal aliases are turned into names that start with ``_``,
    # which Path will treat as an ordinary segment.
    for raw in ("", ".", "..", "../escape", "../../etc/passwd"):
        assert not sanitize(raw).startswith("..")
        assert "/" not in sanitize(raw)
        assert "\\" not in sanitize(raw)
    # Normal identifier-shaped names pass through unchanged.
    assert sanitize("stage1") == "stage1"
    assert sanitize("depth.estimate") == "depth.estimate"
    assert sanitize("upscale-x4") == "upscale-x4"


def test_cache_path_is_partitioned_by_id_prefix(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    executor = CASDAGExecutor(store, tmp_path / "cache", CASDAGConfig())

    cas_id = "sha256:" + "abc" + "0" * 61
    path = executor._cache_path(cas_id)
    assert path.parent.parent == executor.cache_dir / "dag_cache"
    # The leaf directory must be a 2-character partition prefix of the
    # sanitized id — the sanitizer can transform but must produce something
    # non-empty and short.
    assert len(path.parent.name) == 2
    assert path.suffix == ".json"
