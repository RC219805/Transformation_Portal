"""Tests for markdown heading-link validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validation" / "check_doc_heading_links.py"
_spec = importlib.util.spec_from_file_location("check_doc_heading_links", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["check_doc_heading_links"] = _module
_spec.loader.exec_module(_module)


def test_heading_link_validator_accepts_existing_anchor(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    source = tmp_path / "source.md"
    target.write_text("# Target\n\n## Existing Heading\n", encoding="utf-8")
    source.write_text("[target](target.md#existing-heading)\n", encoding="utf-8")

    assert _module.check([source]) == []


def test_heading_link_validator_rejects_missing_anchor(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    source = tmp_path / "source.md"
    target.write_text("# Target\n\n## Different Heading\n", encoding="utf-8")
    source.write_text("[target](target.md#missing-heading)\n", encoding="utf-8")

    failures = _module.check([source])

    assert len(failures) == 1
    assert str(source.resolve()) in failures[0]
    assert str(target.resolve()) in failures[0]
    assert "#missing-heading" in failures[0]


def test_heading_link_validator_ignores_code_block_comment_lines(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    source = tmp_path / "source.md"
    target.write_text(
        "\n".join(
            [
                "# Target",
                "",
                "```bash",
                "# fake fenced heading",
                "```",
                "",
                "    # fake indented heading",
                "",
                "## Real Heading",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source.write_text(
        "[fenced](target.md#fake-fenced-heading)\n"
        "[indented](target.md#fake-indented-heading)\n"
        "[real](target.md#real-heading)\n",
        encoding="utf-8",
    )

    failures = _module.check([source])

    assert len(failures) == 2
    assert any("#fake-fenced-heading" in failure for failure in failures)
    assert any("#fake-indented-heading" in failure for failure in failures)
    assert all("#real-heading" not in failure for failure in failures)


def test_default_todo_quick_win_binary_cleanup_heading_references_are_current() -> None:
    assert _module.check([]) == []
