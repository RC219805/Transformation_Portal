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


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [
        ("M-1. Sandbox or sign plugins before broader use", "m-1-sandbox-or-sign-plugins-before-broader-use"),
        ("1.2.3. Numbered Heading", "123-numbered-heading"),
        ("What's new? (v2.0!)", "whats-new-v20"),
    ],
)
def test_heading_link_validator_removes_heading_punctuation(tmp_path: Path, heading: str, anchor: str) -> None:
    target = tmp_path / "target.md"
    source = tmp_path / "source.md"
    target.write_text(f"# Target\n\n## {heading}\n", encoding="utf-8")
    source.write_text(f"[section](target.md#{anchor})\n", encoding="utf-8")

    assert _module.check([source]) == []


def test_heading_link_validator_numbers_duplicates_after_punctuation_removal(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    source = tmp_path / "source.md"
    target.write_text("## 1. Introduction\n\n## 1 Introduction\n", encoding="utf-8")
    source.write_text(
        "[first](target.md#1-introduction)\n"
        "[second](target.md#1-introduction-1)\n"
        "[invalid punctuation](target.md#1.-introduction)\n",
        encoding="utf-8",
    )

    failures = _module.check([source])

    assert len(failures) == 1
    assert "#1.-introduction" in failures[0]
