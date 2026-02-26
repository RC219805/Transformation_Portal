"""Tests for scripts/governance/check_docs_structure.py."""

from __future__ import annotations

import sys

from scripts.governance import check_docs_structure


def test_keyword_rule_allows_archived_paths() -> None:
    assert not check_docs_structure._keyword_violation("docs/historical/SUMMARY.md")
    assert not check_docs_structure._keyword_violation("docs/pr_archive/PR_123/REPORT.md")


def test_keyword_rule_rejects_non_archived_paths() -> None:
    assert check_docs_structure._keyword_violation("docs/STATUS.md")
    assert check_docs_structure._keyword_violation("docs/cli/FINAL_REPORT.md")


def test_root_topology_rule() -> None:
    assert not check_docs_structure._root_violation("docs/README.md")
    assert check_docs_structure._root_violation("docs/random.md")
    assert not check_docs_structure._root_violation("docs/cli/README.md")


def test_changed_docs_files_returns_none_when_git_diff_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(check_docs_structure, "_run_git", lambda _args: (1, "", "fatal: bad revision"))

    candidates, errors = check_docs_structure._changed_docs_files()

    assert candidates is None
    assert errors


def test_changed_docs_files_returns_empty_when_git_succeeds_without_docs(monkeypatch) -> None:
    monkeypatch.setattr(check_docs_structure, "_run_git", lambda _args: (0, "", ""))

    candidates, errors = check_docs_structure._changed_docs_files()

    assert candidates == []
    assert errors == []


def test_main_fails_closed_in_ci_when_changed_file_detection_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_changed_docs_files",
        lambda: (None, ["git diff failure"]),
    )
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(sys, "argv", ["check_docs_structure.py"])

    assert check_docs_structure.main() == 2


def test_main_falls_back_to_all_locally(monkeypatch) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_changed_docs_files",
        lambda: (None, ["git diff failure"]),
    )
    monkeypatch.setattr(
        check_docs_structure,
        "_all_docs_files",
        lambda: ["docs/README.md", "docs/cli/CLI_REFERENCE.md"],
    )
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(sys, "argv", ["check_docs_structure.py"])

    assert check_docs_structure.main() == 0


def test_main_all_mode_uses_all_docs_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_all_docs_files",
        lambda: ["docs/README.md", "docs/cli/CLI_REFERENCE.md"],
    )
    monkeypatch.setattr(sys, "argv", ["check_docs_structure.py", "--all"])

    assert check_docs_structure.main() == 0
