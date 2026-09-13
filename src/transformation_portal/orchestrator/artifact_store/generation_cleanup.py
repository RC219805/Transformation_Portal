"""Cleanup only recorded, private attempt directories and invisible staging."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

from transformation_portal.orchestrator.artifact_store._filesystem import open_directory
from transformation_portal.orchestrator.artifact_store.base import ArtifactStore, ArtifactStoreError
from transformation_portal.orchestrator.execution_workspace import remove_execution_workspace


def remove_owned_attempt_directory(*, job_id: str, output_root: str, requested_output_root: str) -> None:
    """Never recursively delete the requested root or a substituted symlink."""
    output = Path(output_root)
    requested = Path(requested_output_root)
    if (
        not output.is_absolute()
        or not requested.is_absolute()
        or output.parent != requested / ".tp-attempts"
        or re.fullmatch(re.escape(job_id) + r"-[0-9a-f]{32}", output.name) is None
        or ".." in output.parts
        or ".." in requested.parts
    ):
        raise ArtifactStoreError("attempt cleanup rejected an unowned output directory")
    # Pin every ancestor, then pass only the owned basename to rmtree. Its
    # symlink-safe traversal alone would not protect a mutable full-path parent.
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ArtifactStoreError("safe attempt cleanup is unavailable on this platform")
    try:
        parent = open_directory(output.parent)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ArtifactStoreError("attempt cleanup rejected a substituted ancestor") from exc
    try:
        try:
            entry = os.stat(output.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(entry.st_mode):
            raise ArtifactStoreError("attempt cleanup requires its original private directory")
        shutil.rmtree(output.name, dir_fd=parent)
    finally:
        os.close(parent)


async def cleanup_terminal_generation(*, record_store: Any, artifact_store: ArtifactStore, job_id: str) -> bool:
    state = await record_store.generation_cleanup_state(job_id)
    if state is None:
        return False
    # The database returned a terminal dispatch. It can never acquire again.
    # Check the committed pointer before touching staged objects after an
    # uncertain commit; committed objects always survive this cleanup.
    if state["generation_id"] is None:
        await artifact_store.delete(job_id)
    await asyncio.to_thread(remove_execution_workspace, Path(state["output_root"]))
    await asyncio.to_thread(
        remove_owned_attempt_directory,
        job_id=job_id,
        output_root=state["output_root"],
        requested_output_root=state["requested_output_root"],
    )
    await record_store.mark_generation_cleaned(job_id, expected_generation_id=state["generation_id"])
    return True
