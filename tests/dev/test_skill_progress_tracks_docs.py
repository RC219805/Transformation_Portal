"""Contract tests for skill progression track documentation."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKS_PATH = REPO_ROOT / "docs" / "guides" / "SKILL_PROGRESS_TRACKS.md"
AUTOMATION_GUIDE_PATH = REPO_ROOT / "docs" / "guides" / "SKILL_PROGRESSION_AUTOMATION.md"
GUIDES_README_PATH = REPO_ROOT / "docs" / "guides" / "README.md"
DOCS_README_PATH = REPO_ROOT / "docs" / "README.md"
DOCUMENTATION_MAP_PATH = REPO_ROOT / "docs" / "governance" / "DOCUMENTATION_MAP.md"

EXPECTED_TRACKS = {
    "API Contract Parity": ("PR #1567", "PR #1561"),
    "Fail-Closed Path Governance": ("PR #1555", "PR #1556"),
    "Deterministic CI And Docs Validation": ("PR #1560", "PR #1557"),
    "Runtime Bootstrap Determinism": ("PR #1565", "PR #1559"),
    "APEX Evidence Semantics": ("PR #1564", "PR #1556"),
}

EXPECTED_2026_05_02_TRACKS = {
    "Portal CSS Governance And Parity Isolation": ("PR #1608", "PR #1610"),
    "Deterministic Test And Lint Contracts": ("PR #1609", "PR #1605"),
    "Fail-Fast Input Validation And Path Containment": ("PR #1607", "PR #1609"),
    "Documentation Source-Of-Truth Governance": ("PR #1612", "PR #1611"),
    "Security And Coverage Evidence Honesty": ("PR #1604", "PR #1609"),
}

EXPECTED_2026_05_03_TRACKS = {
    "Coverage Governance": ("PR #1631", "check_per_package_coverage.py"),
    "Dependency Parser Contracts": ("PR #1629", "check_dependency_pinning.py"),
    "Public Surface Regression Testing": ("PR #1630", "depth_tools.py"),
    "CI Signal Efficiency": ("PR #1631", ".github/workflows/build.yml"),
    "Docs And Status Truthfulness": ("PR #1629", "PR #1628"),
}

EXPECTED_2026_05_05_TRACKS = {
    "Deterministic Validation-System Design": ("PR #1641", "asset_bundle.py"),
    "Documentation Governance Consistency": ("PR #1646", "AGENTS.md"),
    "Contract-Driven Portal And Frontdoor State Modeling": (
        "PR #1637",
        "review-surface-deferred.js",
    ),
    "Scripts Failure-Mode And Fixture Hygiene": ("PR #1634", "download_samples.py"),
    "Pipeline Re-Export And Import-Surface Discipline": ("PR #1645", "SciPy"),
}


def _read(path: Path) -> str:
    assert path.exists(), f"Missing expected documentation file: {path}"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}\n"
    start = text.index(marker) + len(marker)
    next_heading = text.find("\n## ", start)
    if next_heading == -1:
        return text[start:]
    return text[start:next_heading]


def _subsection(text: str, heading: str) -> str:
    marker = f"### {heading}\n"
    start = text.index(marker) + len(marker)
    next_heading = text.find("\n### ", start)
    if next_heading == -1:
        next_h2 = text.find("\n## ", start)
        return text[start:] if next_h2 == -1 else text[start:next_h2]
    return text[start:next_heading]


def test_skill_progress_tracks_document_exists_and_links_from_current_guides() -> None:
    tracks_text = _read(TRACKS_PATH)
    automation_text = _read(AUTOMATION_GUIDE_PATH)
    guides_readme = _read(GUIDES_README_PATH)
    docs_readme = _read(DOCS_README_PATH)
    documentation_map = _read(DOCUMENTATION_MAP_PATH)

    assert "# Skill Progress Tracks" in tracks_text
    assert "SKILL_PROGRESS_TRACKS.md" in automation_text
    assert "SKILL_PROGRESS_TRACKS.md" in guides_readme
    assert "SKILL_PROGRESS_TRACKS.md" in docs_readme
    assert "SKILL_PROGRESS_TRACKS.md" in documentation_map
    assert "## 2026-05-02 Review-Thread Refresh" in tracks_text
    assert "## 2026-05-03 Review-Thread Refresh" in tracks_text
    assert "## 2026-05-05 Review-Thread Refresh" in tracks_text


@pytest.mark.parametrize("heading, evidence", EXPECTED_TRACKS.items())
def test_skill_progress_track_sections_have_evidence_drills_acceptance_and_checklists(
    heading: str,
    evidence: tuple[str, str],
) -> None:
    section = _section(_read(TRACKS_PATH), heading)

    for pr_anchor in evidence:
        assert pr_anchor in section

    assert section.count("Drill 1 -") == 1
    assert section.count("Drill 2 -") == 1
    assert section.count("Acceptance tests:") == 2
    assert "Review checklist:" in section


def test_skill_progress_tracks_lock_reviewed_skill_names() -> None:
    text = _read(TRACKS_PATH)

    for heading in EXPECTED_TRACKS:
        assert f"## {heading}" in text


@pytest.mark.parametrize("heading, evidence", EXPECTED_2026_05_02_TRACKS.items())
def test_2026_05_02_skill_progress_refresh_tracks_have_evidence_drills_and_acceptance(
    heading: str,
    evidence: tuple[str, str],
) -> None:
    refresh = _section(_read(TRACKS_PATH), "2026-05-02 Review-Thread Refresh")
    section = _subsection(refresh, heading)

    for pr_anchor in evidence:
        assert pr_anchor in section

    assert section.count("Drill 1 -") == 1
    assert section.count("Drill 2 -") == 1
    assert section.count("Acceptance tests:") == 2
    assert "Expected behavior:" in section


def test_2026_05_02_refresh_locks_reviewed_skill_names() -> None:
    refresh = _section(_read(TRACKS_PATH), "2026-05-02 Review-Thread Refresh")

    for heading in EXPECTED_2026_05_02_TRACKS:
        assert f"### {heading}" in refresh


@pytest.mark.parametrize("heading, evidence", EXPECTED_2026_05_03_TRACKS.items())
def test_2026_05_03_skill_progress_refresh_tracks_have_evidence_drills_and_acceptance(
    heading: str,
    evidence: tuple[str, str],
) -> None:
    refresh = _section(_read(TRACKS_PATH), "2026-05-03 Review-Thread Refresh")
    section = _subsection(refresh, heading)

    for pr_anchor in evidence:
        assert pr_anchor in section

    assert section.count("Drill 1 -") == 1
    assert section.count("Drill 2 -") == 1
    assert section.count("Acceptance tests:") == 2
    assert "Expected behavior:" in section


def test_2026_05_03_refresh_locks_reviewed_skill_names() -> None:
    refresh = _section(_read(TRACKS_PATH), "2026-05-03 Review-Thread Refresh")

    for heading in EXPECTED_2026_05_03_TRACKS:
        assert f"### {heading}" in refresh


@pytest.mark.parametrize("heading, evidence", EXPECTED_2026_05_05_TRACKS.items())
def test_2026_05_05_skill_progress_refresh_tracks_have_evidence_drills_and_acceptance(
    heading: str,
    evidence: tuple[str, str],
) -> None:
    refresh = _section(_read(TRACKS_PATH), "2026-05-05 Review-Thread Refresh")
    section = _subsection(refresh, heading)

    for pr_anchor in evidence:
        assert pr_anchor in section

    assert section.count("Drill 1 -") == 1
    assert section.count("Drill 2 -") == 1
    assert section.count("Acceptance tests:") == 2
    assert "Expected behavior:" in section


def test_2026_05_05_refresh_locks_reviewed_skill_names() -> None:
    refresh = _section(_read(TRACKS_PATH), "2026-05-05 Review-Thread Refresh")

    for heading in EXPECTED_2026_05_05_TRACKS:
        assert f"### {heading}" in refresh


def test_apex_track_uses_canonical_apex_codes_import_path() -> None:
    section = _section(_read(TRACKS_PATH), "APEX Evidence Semantics")

    assert "`transformation_portal.lux_depth_v3.apex_codes`" in section
    assert "`src/transformation_portal/lux_depth_v3/apex_codes.py`" in section
    assert "`lux_depth_v3.apex_codes`" not in section
