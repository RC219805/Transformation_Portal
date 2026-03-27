"""Contract tests for live custom-agent configuration files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / ".github" / "agents"
AGENT_README = AGENTS_DIR / "README.md"
CUSTOM_AGENT_GUIDE = REPO_ROOT / "docs" / "guides" / "CUSTOM_AGENT_GUIDE.md"
DOCUMENTATION_MAP = REPO_ROOT / "docs" / "governance" / "DOCUMENTATION_MAP.md"

EXPECTED_PROFILES = {
    "transformation-portal-architect.md": {
        "name": "Transformation Portal Architect",
        "required_tools": {"read", "search", "agent"},
        "disable_model_invocation": True,
        "required_sections": [
            "## Role and Authority",
            "## Binding Governance Sources",
            "## Active Governed Surfaces",
            "## Hard Architectural Invariants",
            "## Security, Dependency, and Contract Rules",
            "## When Architect Must Be Consulted",
            "## Required Review Output",
        ],
        "required_references": [
            "docs/architecture/agent_governance.md",
            "docs/governance/DOCUMENTATION_MAP.md",
            "AGENTS.md",
            "docs/architecture/ARCHITECTURE.md",
            "docs/api/MACHINE_MODE_CONTRACT.md",
            "docs/apex/ingest_contract.md",
            "docs/architecture/ADR-043-orchestrator-decomposition.md",
        ],
        "required_terms": [
            "ci/cd",
            "dependency",
            "security",
            "public interface",
            "portal / orchestrator",
            "lux depth v3",
        ],
    },
    "transformation-portal-specialist.md": {
        "name": "Transformation Portal Specialist",
        "required_tools": {"read", "search", "edit", "execute"},
        "disable_model_invocation": False,
        "required_sections": [
            "## Governance References",
            "## Current Operational Scope",
            "## Authority Boundary",
            "## Repository-Grounded Work",
            "## Validation Expectations",
            "## Response Formats",
            "## Troubleshooting Guidance",
        ],
        "required_references": [
            "docs/architecture/agent_governance.md",
            "AGENTS.md",
            "docs/api/MACHINE_MODE_CONTRACT.md",
            "docs/apex/ingest_contract.md",
            "docs/architecture/ADR-043-orchestrator-decomposition.md",
        ],
        "required_terms": [
            "lux depth v3",
            "portal / orchestrator",
            "archive-gate",
            "machine-mode",
            "ingest",
            "dependency",
            "ci/cd",
            "security",
            "public interface",
        ],
    },
}

LEGACY_PATTERNS = [
    r"depth_pipeline/",
    r"lux_render_pipeline\.py",
    r"python\s+3\.10",
]

MAX_FRONTMATTER_LINES = 50
MAX_AGENT_FILE_SIZE = 50 * 1024
MAX_LINE_LENGTH = 200
MAX_LONG_LINE_RATIO = 0.1


def _read(path: Path) -> str:
    assert path.exists(), f"File not found: {path}"
    return path.read_text(encoding="utf-8")


def _profile_path(file_name: str) -> Path:
    return AGENTS_DIR / file_name


def _live_profile_paths() -> list[Path]:
    return sorted(path for path in AGENTS_DIR.glob("*.md") if _read(path).startswith("---\n"))


def _find_frontmatter_end(lines: list[str]) -> int:
    assert lines and lines[0] == "---", "Agent file must start with YAML frontmatter delimiter line (---)"

    for idx, line in enumerate(lines[1:MAX_FRONTMATTER_LINES], start=1):
        if line == "---":
            return idx

    raise AssertionError(
        "Agent file must include a closing YAML frontmatter delimiter ('---') "
        f"within the first {MAX_FRONTMATTER_LINES} lines"
    )


def _extract_frontmatter(content: str) -> tuple[dict[str, object], str]:
    lines = content.splitlines()
    end_index = _find_frontmatter_end(lines)
    frontmatter = "\n".join(lines[1:end_index])
    body = "\n".join(lines[idx] for idx in range(end_index + 1, len(lines)))
    parsed = yaml.safe_load(frontmatter)

    assert isinstance(parsed, dict), "Frontmatter must parse to a mapping"
    return parsed, body


def test_expected_profile_files_exist() -> None:
    for file_name in EXPECTED_PROFILES:
        assert _profile_path(file_name).exists(), f"Agent profile not found: {file_name}"


def test_no_untracked_live_agent_profiles() -> None:
    discovered = {path.name for path in _live_profile_paths()}
    expected = set(EXPECTED_PROFILES)
    assert discovered == expected, (
        "Live agent profile inventory changed. Update EXPECTED_PROFILES and the README "
        f"to cover the new or removed profiles. Discovered={sorted(discovered)!r}"
    )


@pytest.mark.parametrize("file_name, config", EXPECTED_PROFILES.items())
def test_profile_frontmatter_contract(file_name: str, config: dict[str, object]) -> None:
    frontmatter, body = _extract_frontmatter(_read(_profile_path(file_name)))

    assert frontmatter.get("name") == config["name"]
    assert isinstance(frontmatter.get("description"), str) and frontmatter["description"]
    assert (
        isinstance(frontmatter.get("tools"), list) and frontmatter["tools"]
    ), "Frontmatter must define an explicit non-empty tools list"
    assert set(frontmatter["tools"]) == config["required_tools"]
    assert "*" not in frontmatter["tools"], "Use least-privilege explicit tools, not '*'"
    assert frontmatter.get("disable-model-invocation", False) is config["disable_model_invocation"]
    assert frontmatter.get("user-invocable") is True
    assert re.search(r"^#\s+.+", body, re.MULTILINE), "Agent file must have a markdown H1 title"


@pytest.mark.parametrize("file_name, config", EXPECTED_PROFILES.items())
def test_profile_sections(file_name: str, config: dict[str, object]) -> None:
    _, body = _extract_frontmatter(_read(_profile_path(file_name)))
    for section in config["required_sections"]:
        assert section in body, f"{file_name} must include section: {section}"


@pytest.mark.parametrize("file_name, config", EXPECTED_PROFILES.items())
def test_profile_references_existing_docs(file_name: str, config: dict[str, object]) -> None:
    content = _read(_profile_path(file_name))
    for ref in config["required_references"]:
        assert ref in content, f"{file_name} must reference canonical file: {ref}"
        assert (REPO_ROOT / ref).exists(), f"Referenced file not found: {ref}"


@pytest.mark.parametrize("file_name, config", EXPECTED_PROFILES.items())
def test_profile_mentions_current_governed_surfaces(file_name: str, config: dict[str, object]) -> None:
    content_lower = _read(_profile_path(file_name)).lower()
    for term in config["required_terms"]:
        assert term in content_lower, f"{file_name} must mention required domain term: {term}"


@pytest.mark.parametrize("file_name", sorted(EXPECTED_PROFILES))
def test_profile_file_size_and_line_length(file_name: str) -> None:
    path = _profile_path(file_name)
    content = _read(path)
    assert path.stat().st_size < MAX_AGENT_FILE_SIZE, f"{file_name} is too large; keep live profiles concise and authoritative"
    long_lines = [
        number
        for number, line in enumerate(content.splitlines(), start=1)
        if len(line) > MAX_LINE_LENGTH and not line.strip().startswith("http")
    ]
    assert (
        len(long_lines) < len(content.splitlines()) * MAX_LONG_LINE_RATIO
    ), f"{file_name} has too many overly long lines: {long_lines[:10]!r}"


def test_agent_readme_exists_and_indexes_live_profiles() -> None:
    content = _read(AGENT_README)
    content_lower = content.lower()

    assert "# Transformation Portal Custom Agents" in content
    assert "live custom-agent configuration surface" in content_lower
    assert "authoritative" in content_lower

    for file_name, config in EXPECTED_PROFILES.items():
        assert file_name in content, f"README must index profile file: {file_name}"
        assert config["name"] in content, f"README must name profile: {config['name']}"

    assert "CUSTOM_AGENT_GUIDE.md" in content
    assert "DOCUMENTATION_MAP.md" in content
    assert "agent_governance.md" in content
    assert "AGENTS.md" in content


def test_canonical_docs_exist_and_reference_custom_agents() -> None:
    guide = _read(CUSTOM_AGENT_GUIDE)
    documentation_map = _read(DOCUMENTATION_MAP)

    assert "custom agent" in guide.lower()
    assert "Custom Agents" in documentation_map
    assert "CUSTOM_AGENT_GUIDE.md" in documentation_map


@pytest.mark.parametrize(
    "path",
    [
        AGENT_README,
        *(_profile_path(file_name) for file_name in EXPECTED_PROFILES),
    ],
)
def test_live_agent_docs_do_not_reintroduce_legacy_patterns(path: Path) -> None:
    content = _read(path).lower()
    for pattern in LEGACY_PATTERNS:
        assert not re.search(pattern, content), f"{path.relative_to(REPO_ROOT)} contains stale or legacy pattern: {pattern}"
