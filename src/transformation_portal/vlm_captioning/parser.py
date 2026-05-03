"""Conservative parser for FastVLM flat caption output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CAPTION_KEYS = ("scene", "materials", "features", "natural", "lighting", "issues", "uncertain")
LIST_KEYS = {"materials", "features", "natural", "issues", "uncertain"}
KEY_ALIASES = {
    "scene": "scene",
    "materials": "materials",
    "material": "materials",
    "features": "features",
    "feature": "features",
    "architectural features": "features",
    "natural": "natural",
    "natural elements": "natural",
    "lighting": "lighting",
    "light": "lighting",
    "issues": "issues",
    "issue": "issues",
    "uncertain": "uncertain",
    "uncertainty": "uncertain",
}
_KEY_PATTERN = re.compile(
    r"\b("
    r"SCENE|MATERIALS?|FEATURES?|ARCHITECTURAL\s+FEATURES|NATURAL(?:\s+ELEMENTS)?|"
    r"LIGHTING|LIGHT|ISSUES?|UNCERTAIN|UNCERTAINTY"
    r")\s*[:=]\s*",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class FastVLMCaptionParse:
    """Parsed FastVLM caption payload."""

    raw_text: str
    caption: dict[str, Any]
    validated: bool
    missing_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "validated": self.validated,
            "missing_keys": list(self.missing_keys),
            "warnings": list(self.warnings),
        }


def _normalize_key(raw_key: str) -> str | None:
    key = re.sub(r"\s+", " ", raw_key.strip().lower())
    return KEY_ALIASES.get(key)


def _strip_value(value: str) -> str:
    cleaned = value.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" ;.\t\r\n")
    return cleaned


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in (_strip_value(part) for part in value.split(",")) if item]


def _caption_candidate(raw_text: str) -> str:
    no_tags = _TAG_RE.sub(" ", raw_text or "")
    no_tags = no_tags.replace("|", ";")
    lines = [line.strip() for line in no_tags.splitlines() if line.strip()]
    keyed_lines = [line for line in lines if _KEY_PATTERN.search(line)]
    if keyed_lines:
        return "; ".join(keyed_lines)
    return no_tags


def parse_fastvlm_caption(raw_text: str) -> FastVLMCaptionParse:
    """Parse FastVLM flat caption output.

    The parser accepts ``KEY=value`` or ``KEY: value`` pairs and never
    fabricates missing fields. Missing required keys make ``validated`` false.
    """
    candidate = _caption_candidate(raw_text)
    matches = list(_KEY_PATTERN.finditer(candidate))
    caption: dict[str, Any] = {}
    warnings: list[str] = []
    seen_keys: set[str] = set()

    for index, match in enumerate(matches):
        normalized_key = _normalize_key(match.group(1))
        if normalized_key is None:
            continue
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(candidate)
        value = _strip_value(candidate[value_start:value_end])
        if normalized_key in LIST_KEYS:
            caption[normalized_key] = _split_list(value)
        elif value:
            caption[normalized_key] = value
        else:
            caption[normalized_key] = ""
        seen_keys.add(normalized_key)

    if not matches:
        warnings.append("No FastVLM caption keys were found.")

    missing_keys = [key for key in CAPTION_KEYS if key not in seen_keys]
    if missing_keys:
        warnings.append("FastVLM caption is missing required keys: " + ", ".join(missing_keys))

    return FastVLMCaptionParse(
        raw_text=raw_text,
        caption=caption,
        validated=not missing_keys,
        missing_keys=missing_keys,
        warnings=warnings,
    )
