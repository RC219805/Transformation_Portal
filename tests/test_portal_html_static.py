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
* UX/accessibility hardening: navigation precedes view content, Build-step
  ownership is explicit, dynamic field messages are associated with their
  controls, modal/inert hooks are stable, and high-churn queue content is not
  itself a live region.

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


class _PortalContractParser(HTMLParser):
    """Collect id/data-ui elements and explicit label associations."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: List[Tuple[str, Dict[str, Optional[str]]]] = []
        self.by_id: Dict[str, Tuple[str, Dict[str, Optional[str]]]] = {}
        self.labels_for: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {key: value for key, value in attrs}
        self.elements.append((tag, attr_map))
        element_id = attr_map.get("id")
        if element_id:
            self.by_id[element_id] = (tag, attr_map)
        if tag == "label" and attr_map.get("for"):
            self.labels_for.append(str(attr_map["for"]))


def _portal_contract_parser() -> _PortalContractParser:
    parser = _PortalContractParser()
    parser.feed(_read_portal_markup())
    return parser


def test_workspace_navigation_precedes_overview_in_dom_order() -> None:
    parser = _portal_contract_parser()
    nav_index = next(
        index for index, (tag, attrs) in enumerate(parser.elements) if tag == "nav" and attrs.get("data-ui") == "view-switcher"
    )
    overview_index = next(index for index, (_tag, attrs) in enumerate(parser.elements) if attrs.get("id") == "overview-shell")
    assert nav_index < overview_index


def test_capability_matrix_is_progressively_disclosed_without_losing_hooks() -> None:
    parser = _portal_contract_parser()
    tag, attrs = parser.by_id["overviewCapabilityRow"]
    assert tag == "details"
    assert "open" not in attrs

    matrix_tag, matrix_attrs = parser.by_id["capabilityMatrix"]
    assert matrix_tag == "div"
    assert matrix_attrs.get("data-ui") == "capability-matrix"
    assert matrix_attrs.get("role") == "list"


def test_build_step_toolbar_controls_peer_sections_without_false_tabpanels() -> None:
    parser = _portal_contract_parser()
    expected_controls = {
        "buildStepTab1": {"presetBuilderShell"},
        "buildStepTab2": {"buildPathsShell", "fieldsArchiveGate"},
        "buildStepTab3": {"fieldsLuxDepth", "flagsShellPanel", "governance-shell"},
        "buildStepTab4": {"cli-shell"},
    }

    toolbar_tag, toolbar_attrs = parser.by_id["buildStepTabs"]
    assert toolbar_tag == "div"
    assert toolbar_attrs.get("role") == "toolbar"
    assert toolbar_attrs.get("aria-label") == "Build steps"

    active_steps = []
    for step_id, panel_ids in expected_controls.items():
        step_tag, step_attrs = parser.by_id[step_id]
        assert step_tag == "button"
        assert "role" not in step_attrs
        assert "aria-selected" not in step_attrs
        assert step_attrs.get("aria-pressed") in {"true", "false"}
        if step_attrs.get("aria-current") == "step":
            active_steps.append(step_id)
        assert set(str(step_attrs.get("aria-controls") or "").split()) == panel_ids
        for panel_id in panel_ids:
            _panel_tag, panel_attrs = parser.by_id[panel_id]
            assert panel_attrs.get("role") != "tabpanel"
            assert "aria-labelledby" not in panel_attrs
            assert "tabindex" not in panel_attrs

    assert active_steps == ["buildStepTab1"]

    flags_tag, flags_attrs = parser.by_id["flags-shell"]
    assert flags_tag == "fieldset"
    assert "role" not in flags_attrs


def test_inspector_tablist_and_panels_have_complete_associations() -> None:
    parser = _portal_contract_parser()
    expected = {
        "inspectorOverviewTab": "selectedJobOverviewPanel",
        "inspectorTimelineTab": "selectedJobTimelinePanel",
        "inspectorLogsTab": "selectedJobLogsPanel",
    }
    for tab_id, panel_id in expected.items():
        _tab_tag, tab_attrs = parser.by_id[tab_id]
        _panel_tag, panel_attrs = parser.by_id[panel_id]
        assert tab_attrs.get("role") == "tab"
        assert tab_attrs.get("aria-controls") == panel_id
        assert panel_attrs.get("role") == "tabpanel"
        assert panel_attrs.get("aria-labelledby") == tab_id


def test_dynamic_build_field_messages_are_associated_and_fixed_values_are_labeled() -> None:
    parser = _portal_contract_parser()
    status_by_control = {
        "inputDir": "inputDirStatus",
        "outputDir": "outputDirStatus",
        "archiveIndexPath": "archiveIndexStatus",
        "rightsManifestPath": "rightsManifestStatus",
        "groupingMode": "groupingModeStatus",
        "reconstructionIterations": "reconstructionIterationsStatus",
        "camerasSidecarPath": "camerasSidecarStatus",
        "reconstructionTier": "reconstructionTierStatus",
        "rawIngestMode": "rawIngestModeStatus",
        "maxWorkersMode": "maxWorkersStatus",
        "maxWorkers": "maxWorkersStatus",
        "maxGpuWorkersMode": "maxGpuWorkersStatus",
        "maxGpuWorkers": "maxGpuWorkersStatus",
        "logLevel": "logLevelStatus",
    }
    for control_id, status_id in status_by_control.items():
        _tag, attrs = parser.by_id[control_id]
        assert status_id in str(attrs.get("aria-describedby") or "").split()
        assert attrs.get("aria-errormessage") == status_id
        assert status_id in parser.by_id

    for control_id in ("v2Preset", "maxWorkers", "maxGpuWorkers"):
        assert control_id in parser.labels_for

    switch_attrs = [attrs for _tag, attrs in parser.elements if attrs.get("role") == "switch"]
    assert switch_attrs
    assert all(str(attrs.get("aria-label") or "").strip() for attrs in switch_attrs)


def test_modal_inert_hooks_review_empty_state_and_queue_announcements_are_stable() -> None:
    parser = _portal_contract_parser()
    modal_shells = [attrs for _tag, attrs in parser.elements if attrs.get("data-modal-shell")]
    modal_dialogs = [attrs for _tag, attrs in parser.elements if "data-modal-dialog" in attrs]
    inert_targets = [attrs for _tag, attrs in parser.elements if "data-modal-inert-target" in attrs]
    assert {attrs["data-modal-shell"] for attrs in modal_shells} == {
        "legacy-draft-recovery",
        "shortcuts",
        "effective-config",
        "artifact-viewer",
    }
    assert len(modal_dialogs) == 4
    assert all(attrs.get("role") == "dialog" and attrs.get("aria-modal") == "true" for attrs in modal_dialogs)
    assert len(inert_targets) == 2

    _shell_tag, recovery_shell = parser.by_id["legacyDraftRecoveryModal"]
    _panel_tag, recovery_panel = parser.by_id["legacyDraftRecoveryPanel"]
    _status_tag, recovery_status = parser.by_id["legacyDraftRecoveryStatus"]
    _claim_tag, recovery_claim = parser.by_id["claimLegacyDraftBtn"]
    _discard_tag, recovery_discard = parser.by_id["discardLegacyDraftBtn"]
    assert recovery_shell.get("data-ui") == "legacy-draft-recovery-dialog"
    assert recovery_panel.get("aria-labelledby") == "legacyDraftRecoveryTitle"
    assert set(str(recovery_panel.get("aria-describedby") or "").split()) == {
        "legacyDraftRecoveryDescription",
        "legacyDraftRecoveryStatus",
    }
    assert recovery_status.get("role") == "alert"
    assert recovery_claim.get("data-draft-recovery-action") == "claim"
    assert recovery_discard.get("data-draft-recovery-action") == "discard"

    health_attrs = next(attrs for _tag, attrs in parser.elements if "topbar-status" in str(attrs.get("class") or "").split())
    assert health_attrs.get("role") == "group"
    assert "aria-live" not in health_attrs
    assert health_attrs.get("aria-label") == "Backend health"

    _empty_tag, empty_attrs = parser.by_id["emptyArtifactState"]
    assert empty_attrs.get("role") == "status"
    assert empty_attrs.get("aria-live") == "polite"
    assert empty_attrs.get("aria-labelledby") == "emptyArtifactTitle"
    assert set(str(empty_attrs.get("aria-describedby") or "").split()) == {
        "emptyArtifactDetail",
        "emptyArtifactAction",
    }

    _list_tag, list_attrs = parser.by_id["jobList"]
    _delta_tag, delta_attrs = parser.by_id["queueDeltaStatus"]
    assert "aria-live" not in list_attrs
    assert delta_attrs.get("role") == "status"
    assert delta_attrs.get("aria-live") == "polite"
    assert delta_attrs.get("aria-atomic") == "true"
