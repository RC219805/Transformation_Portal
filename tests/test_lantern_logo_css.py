"""Tests guarding Lantern logo CSS architectural decisions."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = ROOT / "09_Client_Deliverables" / "Lantern_Logo_Implementation_Kit" / "lantern_logo.css"

from .documentation import documents, valid_until


def _extract_hover_block(css: str) -> str:
    pattern = (
        r"@media\s*\(hover:\s*hover\)\s*and\s*\(pointer:\s*fine\)\s*\{\s*"
        r"\.lantern-logo:hover\s+svg,\s*"
        r"\.lantern-logo:focus-visible\s+svg\s*\{([^{}]+)\}\s*"
        r"\}"
    )
    match = re.search(pattern, css, flags=re.S)
    if not match:
        raise AssertionError("Could not locate hover/focus block in Lantern logo CSS")
    return match.group(1)


def _extract_stroke_value(block: str) -> str:
    stroke_match = re.search(r"stroke:\s*([^;]+);", block)
    if not stroke_match:
        raise AssertionError("Hover/focus block did not define a stroke declaration")
    return stroke_match.group(1).strip()


@documents("Logo uses SVG gradient on hover, not brand gradient")
@valid_until("2026-01-01", reason="Review after brand refresh cycle")
def test_hover_gradient_source() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    hover_block = _extract_hover_block(css)
    hover_stroke = _extract_stroke_value(hover_block)

    assert hover_stroke == "url(#lantern-gradient)"
    assert hover_stroke != "var(--brand-gradient)"
