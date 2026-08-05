"""Contracts for the paired CodeQL workflow actions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODEQL_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "codeql.yml"
CODEQL_ACTION_RE = re.compile(r"uses:\s+github/codeql-action/(init|analyze)@([0-9a-f]{40})")


def test_codeql_init_and_analyze_share_one_immutable_release() -> None:
    matches = CODEQL_ACTION_RE.findall(CODEQL_WORKFLOW.read_text(encoding="utf-8"))
    pins: dict[str, set[str]] = {"init": set(), "analyze": set()}
    for action, sha in matches:
        pins[action].add(sha)

    assert all(pins.values()), "CodeQL workflow must include both init and analyze actions"
    assert all(len(shas) == 1 for shas in pins.values()), "each CodeQL action must use one immutable SHA"
    assert pins["init"] == pins["analyze"], "CodeQL init and analyze must use the same release SHA"
