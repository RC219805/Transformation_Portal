#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for the portal.html static scaffolding.

Pins three contracts that the portal shell must satisfy before any
JavaScript hydration runs:

* Item 12: the default-selected options on the preset, quality tier,
  and compute device ``<select>`` elements agree with the backend's
  default preset (``PRESET_CATALOG`` + ``LUX_PORTAL_DEFAULT_ARGS``),
  so the pre-hydration render does not misrepresent the effective
  config.
* Item 13: every ``<button>`` carries an explicit ``type`` attribute
  (relocated verbatim — the regex whole-file contract was already
  strong).
* Item 14: the workspace rail uses ``<nav>`` + ``aria-current="page"``
  semantics, not ``role="tablist"`` / ``aria-selected``.

Parser: stdlib ``html.parser`` only — no new dependency.
"""

from __future__ import annotations

import importlib
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")

PORTAL_HTML_PATH = Path(orchestrator_app.REPO_ROOT) / "portal.html"


def _read_portal_markup() -> str:
    return PORTAL_HTML_PATH.read_text(encoding="utf-8")


class _SelectOptionParser(HTMLParser):
    """Collect ``<option>`` entries keyed by parent ``<select id=...>``.

    For each recorded select, stores a list of ``(value, selected)``
    tuples in document order. Nested selects are not expected in
    portal.html; the tracker clears on any ``</select>`` it sees.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.selects: Dict[str, List[Tuple[str, bool]]] = {}
        self._current_select_id: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "select":
            attr_map = {k: v for k, v in attrs}
            select_id = attr_map.get("id")
            if select_id:
                self._current_select_id = select_id
                self.selects.setdefault(select_id, [])
        elif tag == "option" and self._current_select_id is not None:
            attr_keys = {k for k, _ in attrs}
            attr_map = {k: v for k, v in attrs}
            self.selects[self._current_select_id].append((attr_map.get("value", ""), "selected" in attr_keys))

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self._current_select_id = None


def _default_preset_recommended_args() -> Dict[str, object]:
    default_preset_name = orchestrator_app.LUX_PORTAL_DEFAULT_ARGS["preset"]
    for preset in orchestrator_app.PRESET_CATALOG["lux-depth-v3"]:
        if preset["name"] == default_preset_name:
            return dict(preset.get("recommended_args") or {})
    raise AssertionError(f"default preset {default_preset_name!r} not found in PRESET_CATALOG['lux-depth-v3']")


def test_portal_html_defaults_match_default_preset_recommended_args() -> None:
    parser = _SelectOptionParser()
    parser.feed(_read_portal_markup())

    for select_id in ("presetSelect", "qualityTier", "depthDevice"):
        assert select_id in parser.selects, f"<select id={select_id!r}> not found in portal.html"
        options = parser.selects[select_id]
        selected_values = [value for value, is_selected in options if is_selected]
        assert len(selected_values) == 1, (
            f"<select id={select_id!r}> must have exactly one selected option, " f"got {selected_values!r}"
        )

    recommended = _default_preset_recommended_args()
    defaults = orchestrator_app.LUX_PORTAL_DEFAULT_ARGS

    preset_selected = next(value for value, is_selected in parser.selects["presetSelect"] if is_selected)
    quality_selected = next(value for value, is_selected in parser.selects["qualityTier"] if is_selected)
    device_selected = next(value for value, is_selected in parser.selects["depthDevice"] if is_selected)

    assert (
        preset_selected == defaults["preset"]
    ), f"presetSelect default {preset_selected!r} does not match LUX_PORTAL_DEFAULT_ARGS['preset']"
    # Quality tier is preset-scoped: the static HTML reflects the post-hydration
    # state of the default preset's recommended_args, not LUX_PORTAL_DEFAULT_ARGS.
    assert quality_selected == recommended["quality_tier"], (
        f"qualityTier default {quality_selected!r} does not match default preset's "
        f"recommended_args.quality_tier={recommended['quality_tier']!r}"
    )
    assert (
        device_selected == defaults["depth_device"]
    ), f"depthDevice default {device_selected!r} does not match LUX_PORTAL_DEFAULT_ARGS['depth_device']"


def test_portal_html_buttons_all_have_type_attribute() -> None:
    markup = _read_portal_markup()
    missing_type: List[str] = []
    for match in re.finditer(r"<button\b([^>]*)>", markup):
        attrs = match.group(1)
        if "type=" not in attrs:
            missing_type.append(attrs.strip())
    assert missing_type == [], f"buttons without type attribute: {missing_type}"


def test_portal_html_exposes_capability_matrix_contract_hooks() -> None:
    markup = _read_portal_markup()
    assert 'id="capabilityMatrix"' in markup
    assert 'data-ui="capability-matrix"' in markup
    assert 'id="capabilitySummaryBadge"' in markup
    assert 'id="capabilitySummaryDetail"' in markup


class _WorkspaceRailParser(HTMLParser):
    """Scan the element tagged ``data-ui="view-switcher"`` and its subtree.

    Records the outer element's tag name, its attributes, and, for every
    descendant, any ``role`` / ``aria-selected`` / ``aria-current``
    attributes seen. Depth-counting handles any nested elements (though
    the rail itself is flat).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.outer_tag: Optional[str] = None
        self.outer_attrs: Dict[str, Optional[str]] = {}
        self.descendant_roles: List[str] = []
        self.descendant_aria_selected: List[str] = []
        self.aria_current_elements: List[Tuple[str, Dict[str, Optional[str]]]] = []
        self._depth = 0  # 0 = outside, 1 = on the outer element, >1 = inside

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k: v for k, v in attrs}
        if self._depth == 0:
            if attr_map.get("data-ui") == "view-switcher":
                self.outer_tag = tag
                self.outer_attrs = attr_map
                self._depth = 1
            return
        # Already inside the rail subtree (depth >= 1); the outer element's
        # own start tag was consumed above.
        self._depth += 1
        if "role" in attr_map and attr_map["role"] is not None:
            self.descendant_roles.append(attr_map["role"])
        if "aria-selected" in attr_map:
            self.descendant_aria_selected.append(attr_map.get("aria-selected") or "")
        if attr_map.get("aria-current") == "page":
            self.aria_current_elements.append((tag, attr_map))

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        # Self-closing tags inside the rail (e.g. <br/>) should not bump depth
        # but must still be inspected for attributes.
        if self._depth == 0:
            return
        attr_map = {k: v for k, v in attrs}
        if "role" in attr_map and attr_map["role"] is not None:
            self.descendant_roles.append(attr_map["role"])
        if "aria-selected" in attr_map:
            self.descendant_aria_selected.append(attr_map.get("aria-selected") or "")
        if attr_map.get("aria-current") == "page":
            self.aria_current_elements.append((tag, attr_map))

    def handle_endtag(self, tag: str) -> None:
        if self._depth == 0:
            return
        self._depth -= 1


def test_workspace_rail_uses_nav_semantics() -> None:
    parser = _WorkspaceRailParser()
    parser.feed(_read_portal_markup())

    assert parser.outer_tag is not None, 'element with data-ui="view-switcher" not found'
    assert parser.outer_tag == "nav", f"workspace rail must be a <nav>, got <{parser.outer_tag}>"
    aria_label = parser.outer_attrs.get("aria-label")
    assert aria_label and aria_label.strip(), f"workspace rail <nav> must have a non-empty aria-label, got {aria_label!r}"

    # The rail root itself must also be free of tab semantics; banning them
    # only on descendants would let ``<nav data-ui="view-switcher" role="tablist">``
    # slip through exactly the regression this test is meant to catch.
    outer_role = parser.outer_attrs.get("role")
    assert outer_role is None, (
        f"workspace rail <nav> must not carry a role attribute (would re-introduce tab semantics), "
        f"found role={outer_role!r}"
    )
    assert "aria-selected" not in parser.outer_attrs, (
        'workspace rail <nav> must not carry aria-selected (replaced by aria-current="page" on the active link), '
        f"found aria-selected={parser.outer_attrs.get('aria-selected')!r}"
    )

    assert parser.descendant_roles == [], (
        f'workspace rail must not use role="..." on descendants (tablist/tab were replaced), '
        f"found {parser.descendant_roles!r}"
    )
    assert parser.descendant_aria_selected == [], (
        f'workspace rail must not use aria-selected (replaced by aria-current="page"), '
        f"found {parser.descendant_aria_selected!r}"
    )

    assert len(parser.aria_current_elements) == 1, (
        f'workspace rail must mark exactly one active link with aria-current="page", '
        f"found {len(parser.aria_current_elements)}"
    )
    active_tag, active_attrs = parser.aria_current_elements[0]
    assert active_tag == "a", f'aria-current="page" must be on an <a>, got <{active_tag}>'
    href = active_attrs.get("href")
    assert href and href.strip(), f"active workspace link must have a non-empty href, got {href!r}"
