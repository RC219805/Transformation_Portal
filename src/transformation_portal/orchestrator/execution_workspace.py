"""Private native-execution storage with a durable, derivable cleanup locator.

Distributed hosts use the same protected shared root. The immutable admitted
output locator determines its private workspace, so terminal database records
can clean it after a consumer or worker exits without running Python cleanup.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path

from transformation_portal.orchestrator.artifact_store._filesystem import open_directory
from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError

EXECUTION_ROOT_ENV = "TP_ORCHESTRATOR_EXECUTION_ROOT"


def execution_workspace_base(*, require_configured: bool = False) -> Path:
    configured = os.environ.get(EXECUTION_ROOT_ENV, "").strip()
    if require_configured and not configured:
        raise ArtifactStoreError(f"{EXECUTION_ROOT_ENV} must name protected shared execution storage")
    base = Path(configured) if configured else Path(tempfile.gettempdir()).resolve() / f"tp-execution-{os.geteuid()}"
    if not base.is_absolute() or ".." in base.parts or base.resolve(strict=False) != base:
        raise ArtifactStoreError("execution storage must be an absolute canonical directory")
    if base.exists():
        descriptor = open_directory(base)
        try:
            _validate_private_directory(descriptor)
        finally:
            os.close(descriptor)
    else:
        # Validate the existing parent without creating storage during admission.
        descriptor = open_directory(base.parent)
        os.close(descriptor)
    return base


def _validate_private_directory(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ArtifactStoreError("execution storage must be owned by the service user with mode 0700")


def execution_workspace_path(output_root: Path, *, require_configured: bool = False) -> Path:
    if not output_root.is_absolute() or ".." in output_root.parts:
        raise ArtifactStoreError("execution workspace requires an absolute admitted output locator")
    digest = hashlib.sha256(b"tp.execution.workspace.v1\n" + str(output_root).encode()).hexdigest()
    return execution_workspace_base(require_configured=require_configured) / digest


def create_execution_workspace(output_root: Path, *, require_configured: bool = False) -> Path:
    workspace = execution_workspace_path(output_root, require_configured=require_configured)
    parent = open_directory(workspace.parent.parent)
    try:
        try:
            os.mkdir(workspace.parent.name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
        base = os.open(workspace.parent.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            _validate_private_directory(base)
            # Reusing a partial workspace would mix two executions. Existing
            # storage must be cleaned through its terminal authority first.
            os.mkdir(workspace.name, mode=0o700, dir_fd=base)
        finally:
            os.close(base)
    finally:
        os.close(parent)
    return workspace


def validate_execution_workspace(workspace: Path, output_root: Path) -> None:
    if workspace != execution_workspace_path(output_root):
        raise ArtifactStoreError("execution workspace does not match the admitted output locator")
    descriptor = open_directory(workspace)
    try:
        _validate_private_directory(descriptor)
    finally:
        os.close(descriptor)


def remove_execution_workspace(output_root: Path) -> None:
    workspace = execution_workspace_path(output_root)
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ArtifactStoreError("safe execution workspace cleanup is unavailable")
    try:
        parent = open_directory(workspace.parent)
    except FileNotFoundError:
        return
    try:
        _validate_private_directory(parent)
        try:
            entry = os.stat(workspace.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid():
            raise ArtifactStoreError("execution workspace cleanup requires its private directory")
        shutil.rmtree(workspace.name, dir_fd=parent)
    finally:
        os.close(parent)
