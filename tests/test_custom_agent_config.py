"""Contract tests for live custom agent profiles."""

import re
from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[1]
AGENTS_DIR = REPO_ROOT / ".github" / "agents"
AGENT_README = AGENTS_DIR / "README.md"
AGENT_QUICK_START = AGENTS_DIR / "QUICK_START_v2.md"
AGENT_RAG_SUMMARY = AGENTS_DIR / "RAG_IMPLEMENTATION_SUMMARY.md"
CUSTOM_AGENT_GUIDE = REPO_ROOT / "docs" / "guides" / "CUSTOM_AGENT_GUIDE.md"

LIVE_AGENT_PROFILES = {
    "architect": AGENTS_DIR / "transformation-portal-architect.md",
    "specialist": AGENTS_DIR / "transformation-portal-specialist.md",
}

MAX_FRONTMATTER_LINES = 50
MAX_AGENT_FILE_BYTES = 50 * 1024
SEP = r"\s*[-_/ ]\s*"

COMMON_FRONTMATTER_FIELDS = ("name", "description", "target", "tools")
ROLE_FRONTMATTER_FIELDS = {
    "architect": ("disable-model-invocation", "user-invocable"),
    "specialist": ("user-invocable",),
}
ROLE_FRONTMATTER_EXPECTATIONS = {
    "architect": {
        "target": "github-copilot",
        "tools": ["read", "search", "agent"],
        "disable-model-invocation": True,
        "user-invocable": True,
    },
    "specialist": {
        "target": "github-copilot",
        "tools": ["read", "search", "edit", "execute"],
        "user-invocable": True,
    },
}

ROLE_SECTION_INTENTS = {
    "architect": (
        "role and authority",
        "binding governance sources",
        "active governed surfaces",
        "when architect must be consulted",
        "required review output",
    ),
    "specialist": (
        "governance references",
        "current operational scope",
        "authority boundary",
        "repository grounded work",
        "validation expectations",
        "response formats",
        "troubleshooting guidance",
    ),
}

ROLE_SURFACE_PATTERNS = {
    "architect": {
        "Lux Depth V3": (rf"lux{SEP}depth{SEP}v3",),
        "portal/orchestrator": (rf"portal{SEP}orchestrator",),
        "ingest/evidence": (r"\bingest\b", r"\bevidence\b"),
        "dependency/governance": (r"\bdependency\b", r"\bci/cd\b", r"\bpackaging\b"),
    },
    "specialist": {
        "Lux Depth V3": (rf"lux{SEP}depth{SEP}v3",),
        "portal/orchestrator": (rf"portal{SEP}orchestrator",),
        "archive gates": (rf"archive{SEP}gate",),
        "machine-mode": (rf"machine{SEP}mode",),
        "ingest": (r"\bingest\b",),
    },
}

ROLE_CORE_REFERENCES = {
    "architect": (
        "docs/architecture/agent_governance.md",
        "docs/governance/DOCUMENTATION_MAP.md",
        "AGENTS.md",
        "docs/architecture/ARCHITECTURE.md",
    ),
    "specialist": (
        "docs/architecture/agent_governance.md",
        "AGENTS.md",
        "docs/api/MACHINE_MODE_CONTRACT.md",
        "docs/apex/ingest_contract.md",
        "docs/architecture/ADR-043-orchestrator-decomposition.md",
    ),
}


def _read_text(path: Path, label: str) -> str:
    assert path.exists(), f"{label} not found: {path}"
    return path.read_text(encoding="utf-8")


def _file_size_bytes(path: Path, label: str) -> int:
    assert path.exists(), f"{label} not found: {path}"
    return path.stat().st_size


def _read_profile(role: str) -> str:
    return _read_text(LIVE_AGENT_PROFILES[role], f"{role.title()} agent file")


def _read_agent_readme() -> str:
    return _read_text(AGENT_README, "Agent README")


def _read_custom_agent_guide() -> str:
    return _read_text(CUSTOM_AGENT_GUIDE, "Custom agent guide")


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalized_h2_headings(content: str) -> set[str]:
    headings = re.findall(r"^##\s+(.+?)\s*$", content, re.MULTILINE)
    return {_normalize_text(heading) for heading in headings}


def _matches_any_pattern(content: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)


def _find_frontmatter_end(lines: list[str]) -> int:
    assert lines and lines[0] == "---", "Agent file must start with YAML frontmatter delimiter line (---)"

    for idx, line in enumerate(lines[1:MAX_FRONTMATTER_LINES], start=1):
        if line == "---":
            return idx

    raise AssertionError(
        "Agent file must include a closing YAML frontmatter delimiter ('---') "
        f"within the first {MAX_FRONTMATTER_LINES} lines"
    )


def _extract_frontmatter(content: str) -> str:
    lines = content.splitlines()
    end_idx = _find_frontmatter_end(lines)
    return "\n".join(lines[1:end_idx])


def _parse_frontmatter(content: str) -> dict:
    frontmatter = _extract_frontmatter(content)
    parsed = yaml.safe_load(frontmatter)
    assert isinstance(parsed, dict), "Frontmatter must parse to a mapping"
    return parsed


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_profile_exists(role: str):
    assert LIVE_AGENT_PROFILES[role].exists(), f"{role.title()} agent file not found: {LIVE_AGENT_PROFILES[role]}"


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_has_frontmatter(role: str):
    _find_frontmatter_end(_read_profile(role).splitlines())


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_frontmatter_has_required_fields(role: str):
    frontmatter = _parse_frontmatter(_read_profile(role))
    for field in COMMON_FRONTMATTER_FIELDS + ROLE_FRONTMATTER_FIELDS[role]:
        assert field in frontmatter, f"{role.title()} frontmatter must include '{field}' field"

    assert isinstance(frontmatter["name"], str) and frontmatter["name"], f"{role.title()} frontmatter must set name"
    assert (
        isinstance(frontmatter["description"], str) and frontmatter["description"]
    ), f"{role.title()} frontmatter must set description"
    assert frontmatter["target"] == ROLE_FRONTMATTER_EXPECTATIONS[role]["target"]
    assert (
        isinstance(frontmatter["tools"], list) and frontmatter["tools"]
    ), f"{role.title()} frontmatter must include a non-empty tools list"
    assert frontmatter["tools"] == ROLE_FRONTMATTER_EXPECTATIONS[role]["tools"]

    for field, expected in ROLE_FRONTMATTER_EXPECTATIONS[role].items():
        assert frontmatter[field] == expected, f"{role.title()} frontmatter field '{field}' must equal {expected!r}"


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_has_title(role: str):
    lines = _read_profile(role).splitlines()
    frontmatter_end = _find_frontmatter_end(lines)
    body = "\n".join(lines[idx] for idx in range(frontmatter_end + 1, len(lines)))
    assert re.search(r"^# .+$", body, re.MULTILINE), f"{role.title()} agent file must have a markdown H1 title"


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_has_required_contract_sections(role: str):
    headings = _normalized_h2_headings(_read_profile(role))
    for section_intent in ROLE_SECTION_INTENTS[role]:
        assert section_intent in headings, f"{role.title()} agent file must include section intent: {section_intent}"


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_covers_role_surfaces(role: str):
    content = _read_profile(role)
    for label, patterns in ROLE_SURFACE_PATTERNS[role].items():
        assert _matches_any_pattern(content, patterns), f"{role.title()} agent file must cover surface: {label}"


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_references_existing_governance_and_contract_docs(role: str):
    content = _read_profile(role)
    for ref in ROLE_CORE_REFERENCES[role]:
        assert ref in content, f"{role.title()} agent file must reference canonical file: {ref}"
        assert (REPO_ROOT / ref).exists(), f"Referenced file not found: {ref}"


def test_agent_readme_exists_and_references_both_live_profiles():
    content = _read_agent_readme()
    assert "transformation-portal-architect.md" in content
    assert "transformation-portal-specialist.md" in content
    assert "Transformation Portal Architect" in content
    assert "Transformation Portal Specialist" in content
    assert "CUSTOM_AGENT_GUIDE.md" in content


def test_specialist_supporting_docs_exist():
    assert AGENT_QUICK_START.exists(), f"Quick start doc not found: {AGENT_QUICK_START}"
    assert AGENT_RAG_SUMMARY.exists(), f"Summary doc not found: {AGENT_RAG_SUMMARY}"
    assert CUSTOM_AGENT_GUIDE.exists(), f"Custom agent guide not found: {CUSTOM_AGENT_GUIDE}"


def test_custom_agent_guide_has_specialist_usage_examples():
    content = _read_custom_agent_guide()
    example_count = content.count("@transformation-portal-specialist")
    assert "@transformation-portal-specialist" in content
    assert example_count >= 5, f"Guide should include at least 5 usage examples, found {example_count}"


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_file_not_too_large(role: str):
    size_bytes = _file_size_bytes(LIVE_AGENT_PROFILES[role], f"{role.title()} agent file")
    assert (
        size_bytes < MAX_AGENT_FILE_BYTES
    ), f"{role.title()} agent file is {size_bytes} bytes, should be under {MAX_AGENT_FILE_BYTES} bytes"


@pytest.mark.parametrize("role", ("architect", "specialist"))
def test_agent_file_has_reasonable_line_length(role: str):
    lines = _read_profile(role).splitlines()
    long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 200 and not line.strip().startswith("http")]
    assert len(long_lines) < len(lines) * 0.1, f"Too many long lines in {role} agent file ({len(long_lines)})"
