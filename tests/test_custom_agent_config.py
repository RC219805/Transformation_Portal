"""Contract tests for custom agent configuration files."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[1]
AGENTS_DIR = REPO_ROOT / ".github" / "agents"
SPECIALIST_FILE = AGENTS_DIR / "transformation-portal-specialist.md"
AGENT_README = AGENTS_DIR / "README.md"
AGENT_QUICK_START = AGENTS_DIR / "QUICK_START_v2.md"
AGENT_RAG_SUMMARY = AGENTS_DIR / "RAG_IMPLEMENTATION_SUMMARY.md"
CUSTOM_AGENT_GUIDE = REPO_ROOT / "docs" / "guides" / "CUSTOM_AGENT_GUIDE.md"

REQUIRED_SECTIONS = [
    "## Governance References",
    "## Current Operational Scope",
    "## Authority Boundary",
    "## Repository-Grounded Work",
    "## Validation Expectations",
    "## Response Formats",
    "## Troubleshooting Guidance",
]

REQUIRED_SURFACE_PATTERNS = {
    "Lux Depth V3": [r"lux depth v3"],
    "portal/orchestrator": [r"portal\s*/\s*orchestrator", r"portal-orchestrator"],
    "archive gates": [r"archive[- ]gate"],
    "machine-mode": [r"machine[- ]mode"],
    "ingest": [r"\bingest\b"],
}

REQUIRED_ESCALATION_TERMS = [
    "dependency",
    "CI/CD",
    "security",
    "public interface",
]

REQUIRED_CORE_REFERENCES = [
    "docs/architecture/agent_governance.md",
    "AGENTS.md",
    "docs/api/MACHINE_MODE_CONTRACT.md",
    "docs/apex/ingest_contract.md",
    "docs/architecture/ADR-043-orchestrator-decomposition.md",
]


def _read_specialist() -> str:
    return SPECIALIST_FILE.read_text()


def _extract_frontmatter(content: str) -> str:
    lines = content.splitlines()
    frontmatter_lines = []
    in_frontmatter = False

    for line in lines:
        if line == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            break
        if in_frontmatter:
            frontmatter_lines.append(line)

    return "\n".join(frontmatter_lines)


def test_agent_file_exists():
    assert SPECIALIST_FILE.exists(), f"Agent file not found: {SPECIALIST_FILE}"


def test_agent_has_frontmatter():
    content = _read_specialist()
    assert content.startswith("---\n"), "Agent file must start with YAML frontmatter (---)"
    assert "\n---\n" in content, "Agent file must include a closing frontmatter delimiter (---)"


def test_agent_frontmatter_has_name_and_description():
    frontmatter = _extract_frontmatter(_read_specialist())
    assert re.search(r"^name:\s*.+", frontmatter, re.MULTILINE), "Frontmatter must include 'name:' field"
    assert re.search(
        r"^description:\s*.+", frontmatter, re.MULTILINE
    ), "Frontmatter must include 'description:' field"


def test_agent_has_title():
    content = _read_specialist()
    assert re.search(r"\n# .+\n", content), "Agent file must have a markdown H1 title"


def test_agent_has_required_contract_sections():
    content = _read_specialist()
    for section in REQUIRED_SECTIONS:
        assert section in content, f"Agent file must include section: {section}"


def test_agent_covers_current_operational_surfaces():
    content_lower = _read_specialist().lower()
    for label, patterns in REQUIRED_SURFACE_PATTERNS.items():
        assert any(re.search(pattern, content_lower) for pattern in patterns), (
            f"Agent file must cover operational surface: {label}"
        )


def test_agent_lists_mandatory_escalation_domains():
    content_lower = _read_specialist().lower()
    for term in REQUIRED_ESCALATION_TERMS:
        assert term.lower() in content_lower, f"Agent file must mention escalation domain: {term}"


def test_agent_references_existing_governance_and_contract_docs():
    content = _read_specialist()
    for ref in REQUIRED_CORE_REFERENCES:
        assert ref in content, f"Agent file must reference canonical file: {ref}"
        assert (REPO_ROOT / ref).exists(), f"Referenced file not found: {ref}"


def test_agent_readme_exists_and_references_specialist_scope():
    assert AGENT_README.exists(), f"Agent README not found: {AGENT_README}"
    content_lower = AGENT_README.read_text().lower()
    assert "transformation-portal-specialist" in content_lower
    assert "lux depth v3" in content_lower
    assert "portal/orchestrator" in content_lower or "portal / orchestrator" in content_lower


def test_agent_supporting_docs_exist():
    assert AGENT_QUICK_START.exists(), f"Quick start doc not found: {AGENT_QUICK_START}"
    assert AGENT_RAG_SUMMARY.exists(), f"Summary doc not found: {AGENT_RAG_SUMMARY}"
    assert CUSTOM_AGENT_GUIDE.exists(), f"Custom agent guide not found: {CUSTOM_AGENT_GUIDE}"


def test_custom_agent_guide_has_specialist_usage_examples():
    content = CUSTOM_AGENT_GUIDE.read_text()
    example_count = content.count("@transformation-portal-specialist")
    assert "@transformation-portal-specialist" in content
    assert example_count >= 5, f"Guide should include at least 5 usage examples, found {example_count}"


def test_agent_file_not_too_large():
    size_bytes = SPECIALIST_FILE.stat().st_size
    max_size = 50 * 1024
    assert size_bytes < max_size, f"Agent file is {size_bytes} bytes, should be under {max_size} bytes"


def test_agent_file_has_reasonable_line_length():
    lines = _read_specialist().splitlines()
    long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 200 and not line.strip().startswith("http")]
    assert len(long_lines) < len(lines) * 0.1, f"Too many long lines ({len(long_lines)}), check formatting"
