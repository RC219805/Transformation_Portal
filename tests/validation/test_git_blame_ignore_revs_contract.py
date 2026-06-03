"""Contract tests for the git blame ignore-revs file."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IGNORE_REVS_PATH = PROJECT_ROOT / ".git-blame-ignore-revs"

EXPECTED_FORMATTING_REVS = [
    "9af004eeb960249ca1f387fd8f5871a4bb958cbe",
]

REMOVED_PLACEHOLDER_REVS = {
    "d35d12479ee23f7af99da964d55f684e5e4108aa",
    "1073d409fafecb36af0c89f6daebcbc3aee8814d",
}


def _ignore_revs() -> list[str]:
    revs: list[str] = []
    for line in IGNORE_REVS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            revs.append(stripped)
    return revs


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_git_blame_ignore_revs_only_lists_known_formatting_commits() -> None:
    revs = _ignore_revs()

    assert revs == EXPECTED_FORMATTING_REVS
    assert REMOVED_PLACEHOLDER_REVS.isdisjoint(revs)
    assert len(revs) == len(set(revs))
    for rev in revs:
        assert re.fullmatch(r"[0-9a-f]{40}", rev), rev


def test_git_blame_ignore_revs_resolve_in_full_history_checkout() -> None:
    shallow_result = _git("rev-parse", "--is-shallow-repository")
    assert shallow_result.returncode == 0, shallow_result.stderr
    if shallow_result.stdout.strip() == "true":
        pytest.skip("shallow checkout cannot prove historical ignore-revs objects")

    for rev in _ignore_revs():
        result = _git("cat-file", "-e", f"{rev}^{{commit}}")
        assert result.returncode == 0, result.stderr
