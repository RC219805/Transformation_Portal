"""Tests for scripts/governance/check_docs_structure.py."""

from __future__ import annotations

import sys

from scripts.governance import check_docs_structure


def test_topology_rule_allows_docs_readme_and_whitelisted_subdirs() -> None:
    assert not check_docs_structure._root_violation(check_docs_structure.DocChange(status="A", path="docs/README.md"))
    assert not check_docs_structure._root_violation(
        check_docs_structure.DocChange(status="A", path="docs/governance/DOCUMENTATION_POLICY.md")
    )
    assert not check_docs_structure._root_violation(
        check_docs_structure.DocChange(status="M", path="docs/historical/DELIVERABLES.md")
    )


def test_topology_rule_rejects_any_root_docs_file_other_than_readme() -> None:
    assert check_docs_structure._root_violation(check_docs_structure.DocChange(status="A", path="docs/random.md"))
    assert check_docs_structure._root_violation(
        check_docs_structure.DocChange(status="M", path="docs/ARCHITECTURAL_WORKFLOW.md")
    )
    assert check_docs_structure._root_violation(check_docs_structure.DocChange(status="R", path="docs/new_doc.md"))


def test_topology_rule_rejects_non_whitelisted_top_level_dir() -> None:
    assert check_docs_structure._root_violation(check_docs_structure.DocChange(status="A", path="docs/unapproved/new_doc.md"))


def test_root_topology_rule_for_added_files() -> None:
    assert not check_docs_structure._root_violation(check_docs_structure.DocChange(status="A", path="docs/README.md"))
    assert check_docs_structure._root_violation(check_docs_structure.DocChange(status="A", path="docs/random.md"))
    assert not check_docs_structure._root_violation(check_docs_structure.DocChange(status="A", path="docs/cli/README.md"))


def test_root_topology_rule_blocks_modifying_existing_legacy_root_docs() -> None:
    assert check_docs_structure._root_violation(
        check_docs_structure.DocChange(status="M", path="docs/ARCHITECTURAL_WORKFLOW.md")
    )


def test_root_topology_rule_blocks_rename_or_copy_into_root() -> None:
    assert check_docs_structure._root_violation(check_docs_structure.DocChange(status="R", path="docs/new_doc.md"))
    assert check_docs_structure._root_violation(check_docs_structure.DocChange(status="C", path="docs/new_doc.md"))


def test_parse_name_status_uses_rename_destination_path() -> None:
    changes = check_docs_structure._parse_name_status_output(
        "M\tdocs/ARCHITECTURAL_WORKFLOW.md\nR100\tdocs/cli/guide.md\tdocs/guide.md\nA\tdocs/new.md\n"
    )
    assert changes == [
        check_docs_structure.DocChange(status="M", path="docs/ARCHITECTURAL_WORKFLOW.md"),
        check_docs_structure.DocChange(status="R", path="docs/guide.md"),
        check_docs_structure.DocChange(status="A", path="docs/new.md"),
    ]


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


def test_all_docs_files_prefers_tracked_git_files(monkeypatch) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_run_git",
        lambda args: (
            (0, "docs/README.md\ndocs/quick_references/QUALITY_CONTROL_QUICKREF.md\n", "")
            if args == ["ls-files", "--", "docs"]
            else (1, "", "unexpected")
        ),
    )

    assert check_docs_structure._all_docs_files() == [
        "docs/README.md",
        "docs/quick_references/QUALITY_CONTROL_QUICKREF.md",
    ]


def test_all_docs_files_falls_back_to_filesystem_when_git_ls_files_fails(monkeypatch, tmp_path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "README.md").write_text("readme\n", encoding="utf-8")
    cli_dir = docs_root / "cli"
    cli_dir.mkdir()
    (cli_dir / "CLI_REFERENCE.md").write_text("cli\n", encoding="utf-8")
    monkeypatch.setattr(check_docs_structure, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        check_docs_structure,
        "_run_git",
        lambda _args: (1, "", "fatal: not a git repository"),
    )

    assert check_docs_structure._all_docs_files() == [
        "docs/README.md",
        "docs/cli/CLI_REFERENCE.md",
    ]


def test_main_fails_closed_in_ci_when_changed_file_detection_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_changed_docs_files",
        lambda: (None, ["git diff failure"]),
    )
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(check_docs_structure, "_all_docs_files", lambda: ["docs/README.md"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_docs_structure.py", "--legacy-allowlist", "/dev/null"],
    )

    assert check_docs_structure.main() == 0


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
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_docs_structure.py", "--legacy-allowlist", "/dev/null"],
    )

    assert check_docs_structure.main() == 0


def test_main_all_mode_uses_all_docs_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_all_docs_files",
        lambda: ["docs/README.md", "docs/cli/CLI_REFERENCE.md"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_docs_structure.py", "--all", "--legacy-allowlist", "/dev/null"],
    )

    assert check_docs_structure.main() == 0


def test_main_all_mode_fails_for_root_docs_violation(monkeypatch) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_all_docs_files",
        lambda: ["docs/README.md", "docs/ILLEGAL.md"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_docs_structure.py", "--all", "--legacy-allowlist", "/dev/null"],
    )

    assert check_docs_structure.main() == 1


def test_main_all_mode_allows_known_legacy_violation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_all_docs_files",
        lambda: ["docs/README.md", "docs/ILLEGAL.md"],
    )
    allowlist = tmp_path / "docs_legacy.txt"
    allowlist.write_text("docs/ILLEGAL.md\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_docs_structure.py", "--all", "--legacy-allowlist", str(allowlist)],
    )

    assert check_docs_structure.main() == 0


def test_main_changed_only_still_blocks_touched_legacy_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        check_docs_structure,
        "_changed_docs_files",
        lambda: ([check_docs_structure.DocChange(status="M", path="docs/ILLEGAL.md")], []),
    )
    allowlist = tmp_path / "docs_legacy.txt"
    allowlist.write_text("docs/ILLEGAL.md\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_docs_structure.py", "--changed-only", "--legacy-allowlist", str(allowlist)],
    )

    assert check_docs_structure.main() == 1
