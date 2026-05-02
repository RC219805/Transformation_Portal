from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_PATTERN = re.compile(r"^(MOVE (?P<src>.+) -> (?P<dest>.+)|REMOVE (?P<remove>.+))$")


def _copy_repo_script(repo_root: Path) -> Path:
    source = REPO_ROOT / "scripts" / "organize_docs.sh"
    destination = repo_root / "scripts" / "organize_docs.sh"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _init_repo(repo_root: Path) -> None:
    result = _run(["git", "init", "-q"], repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


def _write(path: Path, content: str = "test\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _track(repo_root: Path, *paths: str) -> None:
    result = _run(["git", "add", *paths], repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


def _run_organizer(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return _run(["bash", "scripts/organize_docs.sh", "--dry-run"], repo_root)


def _proposal_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(("MOVE ", "REMOVE "))]


def _proposal_map(output: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in _proposal_lines(output):
        match = PROPOSAL_PATTERN.match(line)
        assert match, f"Unexpected proposal format: {line}"
        if match.group("src") and match.group("dest"):
            mapping[match.group("src")] = match.group("dest")
    return mapping


def test_dry_run_uses_tracked_files_only_and_limits_docs_scan_to_depth_one(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "README.md")
    _write(repo_root / "legacy.md")
    _write(repo_root / "docs" / "README.md")
    _write(repo_root / "docs" / "ROOT.md")
    _write(repo_root / "docs" / "nested" / "INSIDE.md")
    _track(
        repo_root,
        "README.md",
        "legacy.md",
        "docs/README.md",
        "docs/ROOT.md",
        "docs/nested/INSIDE.md",
        "scripts/organize_docs.sh",
    )

    _write(repo_root / "docs" / ".DS_Store", "ignored\n")

    result = _run_organizer(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MOVE docs/ROOT.md -> docs/guides/ROOT.md" in result.stdout
    assert "MOVE legacy.md -> docs/guides/legacy.md" in result.stdout
    assert "docs/nested/INSIDE.md" not in result.stdout
    assert "docs/.DS_Store" not in result.stdout


def test_dry_run_output_is_stable_and_sorted(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "README.md")
    _write(repo_root / "zeta.txt")
    _write(repo_root / "alpha.md")
    _write(repo_root / "docs" / "README.md")
    _write(repo_root / "docs" / "beta.md")
    _track(repo_root, "README.md", "zeta.txt", "alpha.md", "docs/README.md", "docs/beta.md", "scripts/organize_docs.sh")

    first = _run_organizer(repo_root)
    second = _run_organizer(repo_root)

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert first.stdout == second.stdout

    sources = []
    for line in _proposal_lines(first.stdout):
        match = PROPOSAL_PATTERN.match(line)
        assert match, f"Unexpected proposal format: {line}"
        if match.group("src"):
            sources.append(match.group("src"))
        elif match.group("remove"):
            sources.append(match.group("remove"))
    assert sources == sorted(sources)


def test_claude_root_guidance_matches_root_file_policy(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "README.md")
    _write(repo_root / "CLAUDE.md")
    _write(repo_root / "docs" / "README.md")
    _track(repo_root, "README.md", "CLAUDE.md", "docs/README.md", "scripts/organize_docs.sh")

    result = _run_organizer(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLAUDE.md" not in result.stdout


def test_classifier_uses_tokens_not_substrings_and_preserves_positive_routes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "README.md")
    _write(repo_root / "CORE_DEPENDENCIES_SUMMARY.txt")
    _write(repo_root / "EFFICIENTSAM_REVIEW_SUMMARY.txt")
    _write(repo_root / "docs" / "README.md")
    _write(repo_root / "docs" / "ARCHITECT_DECISION.md")
    _write(repo_root / "docs" / "CI_WORKFLOW_GUIDE.md")
    _write(repo_root / "docs" / "CLI_REFERENCE.md")
    _write(repo_root / "docs" / "QUALITY_CONTRACT.md")
    _write(repo_root / "docs" / "METADATA_SCHEMA.md")
    _track(
        repo_root,
        "README.md",
        "CORE_DEPENDENCIES_SUMMARY.txt",
        "EFFICIENTSAM_REVIEW_SUMMARY.txt",
        "docs/README.md",
        "docs/ARCHITECT_DECISION.md",
        "docs/CI_WORKFLOW_GUIDE.md",
        "docs/CLI_REFERENCE.md",
        "docs/QUALITY_CONTRACT.md",
        "docs/METADATA_SCHEMA.md",
        "scripts/organize_docs.sh",
    )

    result = _run_organizer(repo_root)
    mapping = _proposal_map(result.stdout)

    assert result.returncode == 0, result.stdout + result.stderr
    assert mapping["docs/ARCHITECT_DECISION.md"] == "docs/architecture/ARCHITECT_DECISION.md"
    assert mapping["docs/CI_WORKFLOW_GUIDE.md"] == "docs/ci/CI_WORKFLOW_GUIDE.md"
    assert mapping["docs/CLI_REFERENCE.md"] == "docs/cli/CLI_REFERENCE.md"
    assert mapping["docs/QUALITY_CONTRACT.md"] == "docs/contracts/QUALITY_CONTRACT.md"
    assert mapping["docs/METADATA_SCHEMA.md"] == "docs/schemas/METADATA_SCHEMA.md"
    assert mapping["CORE_DEPENDENCIES_SUMMARY.txt"] != "docs/ci/CORE_DEPENDENCIES_SUMMARY.txt"
    assert mapping["EFFICIENTSAM_REVIEW_SUMMARY.txt"] != "docs/ci/EFFICIENTSAM_REVIEW_SUMMARY.txt"


def test_dry_run_uses_stable_move_and_remove_output_format(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "README.md")
    _write(repo_root / "docs" / "README.md")
    _write(repo_root / "docs" / "ROOT.md")
    _write(repo_root / "docs" / ".DS_Store", "tracked\n")
    _track(repo_root, "README.md", "docs/README.md", "docs/ROOT.md", "docs/.DS_Store", "scripts/organize_docs.sh")

    result = _run_organizer(repo_root)

    assert result.returncode == 0, result.stdout + result.stderr
    proposal_lines = _proposal_lines(result.stdout)
    assert proposal_lines
    for line in proposal_lines:
        assert PROPOSAL_PATTERN.match(line), f"Unexpected proposal format: {line}"
