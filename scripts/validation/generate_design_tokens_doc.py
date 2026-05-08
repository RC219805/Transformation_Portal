#!/usr/bin/env python3
"""Generate ``docs/design/tokens.md`` from the canonical CSS token sources.

Reads:
    - ``web/shared/shared-ui-tokens.css`` (canonical ``--ux-*`` tokens)
    - ``web/secure-landing/portal-src/styles/tokens.css``
      (``--ux-panel-*``, ``--shell-*``, ``--ambient-*`` tokens)

Writes (default mode):
    - ``docs/design/tokens.md``

Modes:
    --check   Regenerate the document in-memory and diff against the
              committed file; exit non-zero with a unified diff on drift.
    (default) Write the document (idempotent).

Output is fully deterministic: no timestamps, no machine-local paths,
stable ordering by category then source-order within category.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

SHARED_UI_TOKENS_PATH = REPO_ROOT / "web" / "shared" / "shared-ui-tokens.css"
PORTAL_SHELL_TOKENS_PATH = (
    REPO_ROOT / "web" / "secure-landing" / "portal-src" / "styles" / "tokens.css"
)
GENERATED_DOC_PATH = REPO_ROOT / "docs" / "design" / "tokens.md"

REGENERATE_COMMAND = "python3 scripts/validation/generate_design_tokens_doc.py"

SCOPE_LIGHT = "light"
SCOPE_DARK = "dark"
SCOPE_REDUCED_MOTION = "reduced_motion"


@dataclass(frozen=True)
class TokenDeclaration:
    name: str
    value: str
    scope: str
    source: str


@dataclass
class TokenEntry:
    name: str
    light: Optional[str] = None
    dark: Optional[str] = None
    reduced_motion: Optional[str] = None
    sources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CSS parsing
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_DECL_RE = re.compile(r"(--[A-Za-z0-9_-]+)\s*:\s*([^;]+?)\s*(?:;|$)", re.DOTALL)
_REDUCED_MOTION_RE = re.compile(
    r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)", re.IGNORECASE
)
_DARK_SELECTORS = frozenset({":root.dark", ".dark:root"})
_ROOT_SELECTOR = ":root"


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _split_top_level_blocks(text: str) -> List[Tuple[str, str]]:
    """Return ``[(preamble, body)]`` for each balanced ``{...}`` block."""
    blocks: List[Tuple[str, str]] = []
    depth = 0
    preamble_start = 0
    block_start = 0
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                preamble = text[preamble_start:index].strip()
                block_start = index + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blocks.append((preamble, text[block_start:index]))
                preamble_start = index + 1
            elif depth < 0:
                raise ValueError(
                    f"unbalanced braces at offset {index}: '}}' without matching '{{'"
                )
    if depth != 0:
        raise ValueError("unbalanced braces: unterminated block")
    return blocks


def _resolve_scope(outer: Optional[str], inner: str) -> Optional[str]:
    """Determine the token-value scope for a (media, selector) pair.

    Returns ``None`` when the selector is not token-bearing (e.g. ``html``,
    ``body``, ``*``, or any non-:root rule).
    """
    selectors = {part.strip() for part in inner.split(",")}
    if outer is not None:
        if not _REDUCED_MOTION_RE.search(outer):
            return None
        if _ROOT_SELECTOR in selectors:
            return SCOPE_REDUCED_MOTION
        return None
    if selectors & _DARK_SELECTORS:
        return SCOPE_DARK
    if _ROOT_SELECTOR in selectors:
        return SCOPE_LIGHT
    return None


def _extract_declarations(
    body: str, scope: str, source: str
) -> List[TokenDeclaration]:
    declarations: List[TokenDeclaration] = []
    for match in _DECL_RE.finditer(body):
        name = match.group(1).strip()
        value = " ".join(match.group(2).split())
        declarations.append(
            TokenDeclaration(name=name, value=value, scope=scope, source=source)
        )
    return declarations


def parse_token_declarations(text: str, source: str) -> List[TokenDeclaration]:
    """Parse a CSS file and return token declarations grouped by scope."""
    cleaned = _strip_comments(text)
    declarations: List[TokenDeclaration] = []
    for preamble, body in _split_top_level_blocks(cleaned):
        if preamble.lstrip().startswith("@media"):
            for nested_preamble, nested_body in _split_top_level_blocks(body):
                scope = _resolve_scope(preamble, nested_preamble)
                if scope is None:
                    continue
                declarations.extend(_extract_declarations(nested_body, scope, source))
            continue
        scope = _resolve_scope(None, preamble)
        if scope is None:
            continue
        declarations.extend(_extract_declarations(body, scope, source))
    return declarations


def merge_declarations(
    declarations_by_source: List[Tuple[str, List[TokenDeclaration]]],
) -> List[TokenEntry]:
    """Merge per-source declarations into a stable list of TokenEntry rows.

    Order is preserved by first-appearance across the supplied sources, so the
    output reflects how an author reading the CSS files top-to-bottom would
    encounter each token.
    """
    entries: Dict[str, TokenEntry] = {}
    order: List[str] = []
    for source, declarations in declarations_by_source:
        for declaration in declarations:
            entry = entries.get(declaration.name)
            if entry is None:
                entry = TokenEntry(name=declaration.name)
                entries[declaration.name] = entry
                order.append(declaration.name)
            if source not in entry.sources:
                entry.sources.append(source)
            if declaration.scope == SCOPE_LIGHT and entry.light is None:
                entry.light = declaration.value
            elif declaration.scope == SCOPE_DARK and entry.dark is None:
                entry.dark = declaration.value
            elif declaration.scope == SCOPE_REDUCED_MOTION and entry.reduced_motion is None:
                entry.reduced_motion = declaration.value
    return [entries[name] for name in order]


# ---------------------------------------------------------------------------
# Description derivation
# ---------------------------------------------------------------------------

_RADIUS_LABELS = {"sm": "Small", "md": "Medium", "lg": "Large", "pill": "Pill"}


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


_DESCRIPTION_RULES: List[Tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    (re.compile(r"^--ux-target-min-size$"), lambda _m: "Minimum interactive target size"),
    (re.compile(r"^--ux-body-size$"), lambda _m: "Body text size"),
    (re.compile(r"^--ux-label-size$"), lambda _m: "Label text size"),
    (re.compile(r"^--ux-meta-size$"), lambda _m: "Meta text size"),
    (re.compile(r"^--ux-meta-tracking$"), lambda _m: "Meta text letter-spacing"),
    (re.compile(r"^--ux-space-(\d+)$"), lambda m: f"Spacing scale step {m.group(1)}"),
    (
        re.compile(r"^--ux-radius-(sm|md|lg|pill)$"),
        lambda m: f"{_RADIUS_LABELS[m.group(1)]} radius",
    ),
    (re.compile(r"^--ux-focus-ring$"), lambda _m: "Focus ring color"),
    (re.compile(r"^--ux-focus-shadow$"), lambda _m: "Focus glow shadow"),
    (
        re.compile(r"^--ux-shadow-(surface|overlay)$"),
        lambda m: f"{_capitalize(m.group(1))} shadow",
    ),
    (
        re.compile(r"^--ux-border-(subtle|strong)$"),
        lambda m: f"{_capitalize(m.group(1))} border color",
    ),
    (
        re.compile(r"^--ux-text-(strong|primary|muted|soft)$"),
        lambda m: f"{_capitalize(m.group(1))} text color",
    ),
    (
        re.compile(r"^--ux-surface-(canvas|elevated|muted|overlay)$"),
        lambda m: f"{_capitalize(m.group(1))} surface color",
    ),
    (
        re.compile(r"^--ux-accent-(primary|secondary)$"),
        lambda m: f"{_capitalize(m.group(1))} accent color",
    ),
    (
        re.compile(r"^--ux-status-(ready|warning|blocked)$"),
        lambda m: f"{_capitalize(m.group(1))} status color",
    ),
    (
        re.compile(r"^--ux-motion-(fast|normal)$"),
        lambda m: f"{_capitalize(m.group(1))} motion duration",
    ),
    (re.compile(r"^--ux-panel-border$"), lambda _m: "Panel border color"),
    (re.compile(r"^--ux-panel-border-strong$"), lambda _m: "Strong panel border color"),
    # Shell
    (re.compile(r"^--shell-ink$"), lambda _m: "Ink (foreground text)"),
    (re.compile(r"^--shell-muted$"), lambda _m: "Muted text"),
    (re.compile(r"^--shell-border$"), lambda _m: "Border color"),
    (re.compile(r"^--shell-panel$"), lambda _m: "Panel background"),
    (re.compile(r"^--shell-panel-strong$"), lambda _m: "Strong panel background"),
    (re.compile(r"^--shell-accent-text$"), lambda _m: "Accent text color"),
    (re.compile(r"^--shell-accent-fill$"), lambda _m: "Accent fill color"),
    (re.compile(r"^--shell-accent-fill-strong$"), lambda _m: "Strong accent fill color"),
    (re.compile(r"^--shell-on-accent$"), lambda _m: "Text color on accent surfaces"),
    (re.compile(r"^--shell-accent-soft$"), lambda _m: "Soft accent tint"),
    (re.compile(r"^--shell-accent$"), lambda _m: "Accent color"),
    (re.compile(r"^--shell-signal$"), lambda _m: "Signal color"),
    (re.compile(r"^--shell-danger$"), lambda _m: "Danger color"),
    (re.compile(r"^--shell-veil-soft$"), lambda _m: "Soft veil overlay"),
    (re.compile(r"^--shell-veil-strong$"), lambda _m: "Strong veil overlay"),
    (re.compile(r"^--shell-veil$"), lambda _m: "Veil overlay"),
    (re.compile(r"^--shell-tint-faint$"), lambda _m: "Faint tint"),
    # Ambient
    (
        re.compile(r"^--ambient-shift-([xy])$"),
        lambda m: f"Ambient shift {m.group(1).upper()} (offset)",
    ),
    (re.compile(r"^--ambient-focus-color$"), lambda _m: "Ambient focal color"),
    (
        re.compile(r"^--ambient-focus-([xy])$"),
        lambda m: f"Ambient focal point {m.group(1).upper()}",
    ),
    (
        re.compile(r"^--ambient-stage-([xy])$"),
        lambda m: f"Ambient stage {m.group(1).upper()}",
    ),
    (re.compile(r"^--ambient-stage-scale$"), lambda _m: "Ambient stage scale"),
    (re.compile(r"^--ambient-stage-rotate$"), lambda _m: "Ambient stage rotation"),
    (
        re.compile(r"^--ambient-color-([abc])$"),
        lambda m: f"Ambient color {m.group(1).upper()}",
    ),
]


def derive_description(name: str) -> str:
    for pattern, formatter in _DESCRIPTION_RULES:
        match = pattern.fullmatch(name)
        if match is not None:
            return formatter(match)
    stripped = name.lstrip("-")
    return _capitalize(stripped.replace("-", " "))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def _classify_ux(name: str) -> Optional[str]:
    if name.startswith("--ux-panel-"):
        return None
    if name.startswith(("--ux-target-", "--ux-body-", "--ux-label-", "--ux-meta-")):
        return "Typography"
    if re.match(r"^--ux-space-\d+$", name):
        return "Spacing"
    if name.startswith("--ux-radius-"):
        return "Radii"
    if name.startswith(("--ux-focus-", "--ux-shadow-")):
        return "Focus & shadow"
    if name.startswith("--ux-border-"):
        return "Borders"
    if name.startswith("--ux-text-"):
        return "Text colors"
    if name.startswith("--ux-surface-"):
        return "Surface colors"
    if name.startswith("--ux-accent-"):
        return "Accents"
    if name.startswith("--ux-status-"):
        return "Status"
    if name.startswith("--ux-motion-"):
        return "Motion"
    return None


def _classify_shell(name: str) -> Optional[str]:
    if name in ("--shell-ink", "--shell-muted"):
        return "Ink & text"
    if name in ("--shell-border", "--shell-panel", "--shell-panel-strong"):
        return "Borders & surfaces"
    if name.startswith("--shell-accent"):
        return "Accents"
    if name in ("--shell-signal", "--shell-danger"):
        return "Signals"
    if name.startswith("--shell-veil") or name.startswith("--shell-tint"):
        return "Veils & tints"
    if name == "--shell-on-accent":
        return "Accents"
    return None


def _classify_ambient(name: str) -> Optional[str]:
    if name.startswith(("--ambient-shift-", "--ambient-stage-")):
        return "Stage transforms"
    if name.startswith("--ambient-focus-"):
        return "Focal point"
    if name.startswith("--ambient-color-"):
        return "Color stops"
    return None


_UX_CATEGORY_ORDER = (
    "Typography",
    "Spacing",
    "Radii",
    "Focus & shadow",
    "Borders",
    "Text colors",
    "Surface colors",
    "Accents",
    "Status",
    "Motion",
)
_SHELL_CATEGORY_ORDER = (
    "Ink & text",
    "Borders & surfaces",
    "Accents",
    "Signals",
    "Veils & tints",
)
_AMBIENT_CATEGORY_ORDER = ("Stage transforms", "Focal point", "Color stops")


@dataclass
class NamespaceSection:
    title: str
    namespace_label: str
    source_repo_path: str
    classify: Callable[[str], Optional[str]]
    category_order: Tuple[str, ...]
    name_filter: Callable[[str], bool]


_PANEL_NAME_FILTER: Callable[[str], bool] = lambda name: name.startswith("--ux-panel-")
_UX_NAME_FILTER: Callable[[str], bool] = lambda name: (
    name.startswith("--ux-") and not name.startswith("--ux-panel-")
)
_SHELL_NAME_FILTER: Callable[[str], bool] = lambda name: name.startswith("--shell-")
_AMBIENT_NAME_FILTER: Callable[[str], bool] = lambda name: name.startswith("--ambient-")


_NAMESPACE_SECTIONS: Tuple[NamespaceSection, ...] = (
    NamespaceSection(
        title="Shared UI tokens (`--ux-*`)",
        namespace_label="--ux-",
        source_repo_path="web/shared/shared-ui-tokens.css",
        classify=_classify_ux,
        category_order=_UX_CATEGORY_ORDER,
        name_filter=_UX_NAME_FILTER,
    ),
    NamespaceSection(
        title="Panel tokens (`--ux-panel-*`)",
        namespace_label="--ux-panel-",
        source_repo_path="web/secure-landing/portal-src/styles/tokens.css",
        classify=lambda _name: None,
        category_order=(),
        name_filter=_PANEL_NAME_FILTER,
    ),
    NamespaceSection(
        title="Shell tokens (`--shell-*`)",
        namespace_label="--shell-",
        source_repo_path="web/secure-landing/portal-src/styles/tokens.css",
        classify=_classify_shell,
        category_order=_SHELL_CATEGORY_ORDER,
        name_filter=_SHELL_NAME_FILTER,
    ),
    NamespaceSection(
        title="Ambient tokens (`--ambient-*`)",
        namespace_label="--ambient-",
        source_repo_path="web/secure-landing/portal-src/styles/tokens.css",
        classify=_classify_ambient,
        category_order=_AMBIENT_CATEGORY_ORDER,
        name_filter=_AMBIENT_NAME_FILTER,
    ),
)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _format_value(value: Optional[str]) -> str:
    if value is None or value == "":
        return "—"
    return f"`{value}`"


def _render_table(entries: List[TokenEntry]) -> List[str]:
    lines = ["| Token | Light | Dark | Description |", "|---|---|---|---|"]
    for entry in entries:
        lines.append(
            f"| `{entry.name}` | {_format_value(entry.light)}"
            f" | {_format_value(entry.dark)}"
            f" | {derive_description(entry.name)} |"
        )
    return lines


def _render_namespace_section(
    section: NamespaceSection, entries: List[TokenEntry]
) -> List[str]:
    selected = [entry for entry in entries if section.name_filter(entry.name)]
    if not selected:
        return []

    lines = [f"## {section.title}", "", f"Source: `{section.source_repo_path}`", ""]

    if section.category_order:
        grouped: Dict[str, List[TokenEntry]] = {
            category: [] for category in section.category_order
        }
        unclassified: List[TokenEntry] = []
        for entry in selected:
            category = section.classify(entry.name)
            if category is None or category not in grouped:
                unclassified.append(entry)
            else:
                grouped[category].append(entry)
        for category in section.category_order:
            members = grouped[category]
            if not members:
                continue
            lines.append(f"### {category}")
            lines.append("")
            lines.extend(_render_table(members))
            lines.append("")
        if unclassified:
            lines.append("### Other")
            lines.append("")
            lines.extend(_render_table(unclassified))
            lines.append("")
    else:
        lines.extend(_render_table(selected))
        lines.append("")

    return lines


def _render_reduced_motion_section(entries: List[TokenEntry]) -> List[str]:
    affected = [
        entry
        for entry in entries
        if entry.reduced_motion is not None and entry.reduced_motion != entry.light
    ]
    if not affected:
        return []
    lines = [
        "## Reduced motion",
        "",
        "The following tokens have different values under "
        "`@media (prefers-reduced-motion: reduce)`:",
        "",
        "| Token | Default | Reduced motion |",
        "|---|---|---|",
    ]
    for entry in affected:
        lines.append(
            f"| `{entry.name}` | {_format_value(entry.light)}"
            f" | {_format_value(entry.reduced_motion)} |"
        )
    lines.append("")
    return lines


def render_document(entries: List[TokenEntry]) -> str:
    header = [
        "<!--",
        "This file is generated by scripts/validation/generate_design_tokens_doc.py.",
        "To refresh after editing the source CSS, run:",
        "",
        f"    {REGENERATE_COMMAND}",
        "",
        "Drift is enforced by `make check-design-tokens-doc`, a pre-commit hook,",
        'and the "Validate generated design tokens reference" step in',
        ".github/workflows/build.yml.",
        "Do not edit by hand.",
        "-->",
        "",
        "# Design tokens",
        "",
        "The Transformation Portal design system is a small set of CSS custom",
        "properties shared between the portal shell and the managed front door.",
        "This catalog is generated from two source files; if the values below",
        "differ from those files, the drift gate will block CI.",
        "",
        "Each table shows the token name, light-mode value, dark-mode value,",
        "and an auto-derived description. Tokens that change under",
        "`prefers-reduced-motion: reduce` are summarized in a final section.",
        "",
    ]

    body: List[str] = []
    for section in _NAMESPACE_SECTIONS:
        body.extend(_render_namespace_section(section, entries))

    body.extend(_render_reduced_motion_section(entries))

    footer = [
        "## Regeneration",
        "",
        "This file is generated. To refresh:",
        "",
        f"    {REGENERATE_COMMAND}",
        "",
        "The drift gate is enforced via `make check-design-tokens-doc`, a",
        "pre-commit hook, and a step in `.github/workflows/build.yml`.",
        "",
    ]

    return "\n".join(header + body + footer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _read_sources() -> List[Tuple[str, List[TokenDeclaration]]]:
    pairs: List[Tuple[str, List[TokenDeclaration]]] = []
    for repo_path, absolute_path in (
        ("web/shared/shared-ui-tokens.css", SHARED_UI_TOKENS_PATH),
        (
            "web/secure-landing/portal-src/styles/tokens.css",
            PORTAL_SHELL_TOKENS_PATH,
        ),
    ):
        text = absolute_path.read_text(encoding="utf-8")
        pairs.append((repo_path, parse_token_declarations(text, repo_path)))
    return pairs


def generate_document() -> str:
    sources = _read_sources()
    entries = merge_declarations(sources)
    return render_document(entries)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify docs/design/tokens.md matches the generated output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    generated = generate_document()

    if args.check:
        if not GENERATED_DOC_PATH.exists():
            print(
                f"design tokens doc missing at {GENERATED_DOC_PATH.relative_to(REPO_ROOT)}; "
                f"run: {REGENERATE_COMMAND}",
                file=sys.stderr,
            )
            return 1
        committed = GENERATED_DOC_PATH.read_text(encoding="utf-8")
        if committed == generated:
            print("design tokens doc is up to date.")
            return 0
        diff = difflib.unified_diff(
            committed.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile="committed/docs/design/tokens.md",
            tofile="regenerated/docs/design/tokens.md",
        )
        sys.stderr.writelines(diff)
        print(
            f"\ndesign tokens doc has drifted; run: {REGENERATE_COMMAND}",
            file=sys.stderr,
        )
        return 1

    GENERATED_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_DOC_PATH.write_text(generated, encoding="utf-8")
    print(
        f"wrote {GENERATED_DOC_PATH.relative_to(REPO_ROOT)} "
        f"({len(generated.splitlines())} lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
