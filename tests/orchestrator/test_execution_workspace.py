"""Private workspace ownership and durable cleanup after interrupted consumers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
from transformation_portal.orchestrator.artifact_store.generation_cleanup import cleanup_terminal_generation
from transformation_portal.orchestrator.execution_workspace import (
    EXECUTION_ROOT_ENV,
    create_execution_workspace,
    execution_workspace_base,
    execution_workspace_path,
    remove_execution_workspace,
    validate_execution_workspace,
)

pytestmark = [pytest.mark.unit, pytest.mark.security]


@pytest.fixture(autouse=True)
def private_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "execution-storage"
    monkeypatch.setenv(EXECUTION_ROOT_ENV, str(base))
    return base


def test_distributed_configuration_is_explicit_and_admission_creates_no_directory(private_base: Path, monkeypatch):
    assert execution_workspace_base(require_configured=True) == private_base
    assert not private_base.exists()
    monkeypatch.delenv(EXECUTION_ROOT_ENV)
    with pytest.raises(ArtifactStoreError, match="protected shared execution storage"):
        execution_workspace_base(require_configured=True)


@pytest.mark.parametrize("kind", ["relative", "symlink", "public_permissions"])
def test_workspace_rejects_unprotected_storage(private_base: Path, monkeypatch, tmp_path: Path, kind: str):
    if kind == "relative":
        monkeypatch.setenv(EXECUTION_ROOT_ENV, "relative-path")
    elif kind == "symlink":
        destination = tmp_path / "elsewhere"
        destination.mkdir(mode=0o700)
        private_base.symlink_to(destination, target_is_directory=True)
    else:
        private_base.mkdir(mode=0o755)
    with pytest.raises(ArtifactStoreError):
        execution_workspace_base(require_configured=True)


def test_workspace_cannot_reuse_partial_execution_or_claim_another_output(tmp_path: Path):
    output = tmp_path / "attempt"
    workspace = create_execution_workspace(output)
    try:
        assert workspace == execution_workspace_path(output)
        validate_execution_workspace(workspace, output)
        with pytest.raises(FileExistsError):
            create_execution_workspace(output)
        with pytest.raises(ArtifactStoreError, match="does not match"):
            validate_execution_workspace(workspace, tmp_path / "other-attempt")
    finally:
        remove_execution_workspace(output)
    assert not workspace.exists()
    remove_execution_workspace(output)


def test_workspace_cleanup_does_not_follow_replacement(tmp_path: Path):
    output = tmp_path / "attempt"
    workspace = create_execution_workspace(output)
    workspace.rmdir()
    outside = tmp_path / "preserved"
    outside.mkdir()
    (outside / "file").write_text("preserve")
    workspace.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactStoreError, match="private directory"):
        remove_execution_workspace(output)
    assert (outside / "file").read_text() == "preserve"


@pytest.mark.asyncio
async def test_terminal_cleanup_removes_abrupt_exit_workspace_and_revisits_late_outputs(tmp_path: Path):
    output = tmp_path / "requested" / ".tp-attempts" / ("job_orphan-" + "a" * 32)
    command = [
        sys.executable,
        "-c",
        "import os,sys;from pathlib import Path;"
        "from transformation_portal.orchestrator.execution_workspace import create_execution_workspace;"
        "workspace=create_execution_workspace(Path(sys.argv[1]),require_configured=True);"
        "(workspace/'partial-output').write_bytes(b'partial');os._exit(9)",
        str(output),
    ]
    child = subprocess.run(command, check=False, timeout=10)
    assert child.returncode == 9
    workspace = execution_workspace_path(output)
    assert (workspace / "partial-output").read_bytes() == b"partial"
    records = AsyncMock()
    records.generation_cleanup_state.return_value = {
        "generation_id": "committed-generation",
        "output_root": str(output),
        "requested_output_root": str(output.parent.parent),
    }
    artifacts = AsyncMock()
    await cleanup_terminal_generation(record_store=records, artifact_store=artifacts, job_id="job_orphan")
    assert not workspace.exists()
    artifacts.delete.assert_not_called()
    # A delayed child may recreate files after the first terminal cleanup.
    recreated = create_execution_workspace(output)
    (recreated / "late-output").write_text("late")
    await cleanup_terminal_generation(record_store=records, artifact_store=artifacts, job_id="job_orphan")
    assert not recreated.exists()
    assert records.mark_generation_cleaned.await_count == 2


@pytest.mark.asyncio
async def test_active_dispatch_workspace_cannot_be_cleaned(tmp_path: Path):
    output = tmp_path / "attempt"
    workspace = create_execution_workspace(output)
    records = AsyncMock()
    records.generation_cleanup_state.return_value = None
    artifacts = AsyncMock()
    try:
        assert not await cleanup_terminal_generation(record_store=records, artifact_store=artifacts, job_id="active")
        assert workspace.is_dir()
        artifacts.delete.assert_not_called()
        records.mark_generation_cleaned.assert_not_called()
    finally:
        remove_execution_workspace(output)
