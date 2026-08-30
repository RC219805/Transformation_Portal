"""Doc-sync checks for the Lux Depth V3 CLI contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3.__main__ import app

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "src" / "transformation_portal" / "lux_depth_v3" / "README.md",
    REPO_ROOT / "docs" / "cli" / "LUX_DEPTH_V3_CLI_GUIDE.md",
    REPO_ROOT / "docs" / "guides" / "LUX_DEPTH_V3_TROUBLESHOOTING.md",
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from CLI output."""
    return re.sub(r"\x1b\[[0-9;]*[mGKHf]", "", text)


def _joined_docs_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOC_FILES)


def test_docs_use_canonical_backend_ids():
    """User-facing Lux Depth V3 docs should use canonical backend ids only."""
    docs_text = _joined_docs_text()

    assert '--depth-backend "depth_anything_v3"' not in docs_text
    assert "--depth-backend depth_anything_v3" not in docs_text
    assert '--depth-backend "da3"' in docs_text or "--depth-backend da3" in docs_text
    assert '--depth-backend "depth_pro"' in docs_text or "--depth-backend depth_pro" in docs_text


def test_docs_match_current_cli_help_for_backend_and_v2_controls():
    """Docs should reflect the current CLI help for backend and V2 flags."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0

    help_output = strip_ansi(result.stdout.lower())
    docs_text = _joined_docs_text().lower()

    assert "depth backend: da3" in help_output
    assert "enable-v2" in help_output
    assert "depth_pro" in docs_text
    assert "da3" in docs_text
    assert '--enable-v2 "off"' in docs_text


def test_docs_and_help_describe_presets_as_curated_or_metadata_labels():
    """Preset docs should not advertise fake curated presets."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0

    help_output = strip_ansi(result.stdout).lower()
    docs_text = _joined_docs_text().lower()

    assert '--preset "apple-depth-pro-research"' not in help_output
    assert "--preset apple-depth-pro-research" not in help_output
    assert '--preset "apple-depth-pro-research"' not in docs_text
    assert "--preset apple-depth-pro-research" not in docs_text
    assert "depth-anything-v3.1-research-m4" in docs_text
    assert "metadata label" in docs_text


def test_active_da3_research_examples_use_explicit_model_selector():
    """Research examples must not rely on the changed default or deprecated alias."""
    docs_text = _joined_docs_text()
    agent_text = (REPO_ROOT / ".github" / "apex-workflow-orchestrator.copilot-agent.yml").read_text(encoding="utf-8")

    assert '--model-key "da3" \\' not in docs_text
    assert 'model_key="da3"' not in docs_text
    assert '--model-key "da3-research"' in docs_text
    assert 'model_key="da3-research"' in docs_text
    assert 'model_key="da3"' not in agent_text
    assert 'model_key="da3-research"' in agent_text
