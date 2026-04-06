"""Shared DA3 runtime contract helpers.

This module stays lightweight so config resolution and backend setup can
reference the repo-local DA3 runtime contract without importing heavier
pipeline modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_REPO_LOCAL_DA3_PYTHON_PARTS = (".venv-da3", "bin", "python")
REPO_LOCAL_DA3_PYTHON = f"./{'/'.join(_REPO_LOCAL_DA3_PYTHON_PARTS)}"


def find_repo_root(start: Path) -> Optional[Path]:
    """Find the repository root by walking upward from ``start``."""
    resolved_start = Path(start).expanduser().resolve()
    search_start = resolved_start if resolved_start.is_dir() else resolved_start.parent

    for candidate in [search_start, *search_start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return None


def repo_local_da3_python_path(start: Path) -> Optional[Path]:
    """Return the canonical repo-local DA3 interpreter path when in a checkout."""
    repo_root = find_repo_root(start)
    if repo_root is None:
        return None
    return repo_root.joinpath(*_REPO_LOCAL_DA3_PYTHON_PARTS)
