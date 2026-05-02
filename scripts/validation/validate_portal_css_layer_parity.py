#!/usr/bin/env python3
"""Compare production portal CSS computed styles against the committed layer baseline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote

from validate_portal_browser_smoke import (  # type: ignore
    DevToolsConnection,
    LocalRuntimeHandle,
    SmokeFailure,
    _base_url,
    _default_profile_dir,
    _find_free_port,
    _portal_shell_ready,
    _poll,
    _resolve_chrome_binary,
    _spawn_local_backend,
    _terminate_runtime,
    _wait_for_devtools,
    _wait_for_page_target,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _layer_parity_contract_path() -> Path:
    return _repo_root() / "tests" / "fixtures" / "portal-css" / "layer-parity-contract.json"


def _layer_parity_baseline_path() -> Path:
    return _repo_root() / "tests" / "fixtures" / "portal-css" / "layer-parity-baseline.json"


def _utility_ownership_path() -> Path:
    return _repo_root() / "web" / "secure-landing" / "portal-src" / "styles" / "utility-ownership.json"


def _load_layer_parity_contract() -> tuple[list[str], list[str], list[str], list[str]]:
    contract = json.loads(_layer_parity_contract_path().read_text(encoding="utf-8"))
    return (
        list(contract.get("states", [])),
        list(contract.get("views", [])),
        list(contract["representativeStyleSelectors"]),
        list(contract["representativeStyleProperties"]),
    )


(
    LAYER_PARITY_STATES,
    LAYER_PARITY_VIEWS,
    REPRESENTATIVE_STYLE_SELECTORS,
    REPRESENTATIVE_STYLE_PROPERTIES,
) = _load_layer_parity_contract()
UTILITY_EXACT_CLASSES = {
    "absolute",
    "antialiased",
    "block",
    "flex",
    "fixed",
    "grid",
    "group",
    "hidden",
    "inline",
    "inline-block",
    "inline-flex",
    "peer",
    "relative",
    "sr-only",
    "sticky",
    "transform",
    "truncate",
    "uppercase",
}
UTILITY_OWNER_ALLOWLIST = {"dark", "light", "performance-lite"}
UTILITY_VARIANTS = {
    "active",
    "dark",
    "disabled",
    "focus",
    "focus-visible",
    "group-hover",
    "hover",
    "lg",
    "md",
    "peer-checked",
    "peer-focus-visible",
    "selection",
    "sm",
    "xl",
}
UTILITY_PREFIXES = (
    "m",
    "mt",
    "mr",
    "mb",
    "ml",
    "mx",
    "my",
    "-m",
    "-mt",
    "-mr",
    "-mb",
    "-ml",
    "-mx",
    "-my",
    "p",
    "pt",
    "pr",
    "pb",
    "pl",
    "px",
    "py",
    "gap",
    "grid-cols",
    "col-span",
    "row-span",
    "flex",
    "items",
    "justify",
    "self",
    "place",
    "w",
    "h",
    "min-w",
    "min-h",
    "max-w",
    "max-h",
    "rounded",
    "border",
    "bg",
    "from",
    "via",
    "to",
    "text",
    "font",
    "tracking",
    "leading",
    "shadow",
    "ring",
    "opacity",
    "overflow",
    "object",
    "inset",
    "top",
    "right",
    "bottom",
    "left",
    "z",
    "cursor",
    "pointer-events",
    "select",
    "resize",
    "whitespace",
    "break",
    "duration",
    "ease",
    "transition",
    "translate",
    "scale",
    "backdrop",
    "animate",
    "outline",
    "fill",
    "stroke",
    "order",
    "basis",
    "shrink",
    "grow",
    "space-x",
    "space-y",
)
REVIEW_STATUS_TONES = ("ready", "warning", "error", "info")
REVIEW_STATUS_THEMES = ("light", "dark")
INTERACTION_OUTLINE_PROBES = (
    {
        "name": "build-step-tab",
        "selector": ".build-step-tab:not(.is-disabled)",
        "sharedProperties": ("transform", "borderColor"),
        "combinedOutlineStyle": "solid",
    },
    {
        "name": "dispatch-tool-btn",
        "selector": ".dispatch-tool-btn:not(.dispatch-tool-btn-primary):not([disabled])",
        "sharedProperties": ("transform", "borderColor", "backgroundColor", "color"),
        "combinedOutlineStyle": "none",
    },
    {
        "name": "workspace-link",
        "selector": ".workspace-link:not(.is-active)",
        "sharedProperties": ("transform", "borderColor", "backgroundColor", "boxShadow"),
        "combinedOutlineStyle": "none",
    },
)
SKELETON_STATE_IDS = (
    "missionShellSkeletonState",
    "intelligenceShellSkeletonState",
    "overviewStatsSkeletonState",
    "profileShellSkeletonState",
    "buildStepperSkeletonState",
    "parametersShellSkeletonState",
    "selectedJobSkeletonState",
    "queueSkeletonState",
    "artifactSkeletonState",
)
SKELETON_STYLE_PROBES = (
    {
        "name": "skeleton-line",
        "selector": ".skeleton-line:not(.skeleton-line-short):not(.skeleton-line-medium):not(.skeleton-line-tiny)",
        "height": "12px",
        "borderRadius": "999px",
    },
    {
        "name": "skeleton-line-short",
        "selector": ".skeleton-line.skeleton-line-short",
        "height": "12px",
        "borderRadius": "999px",
    },
    {
        "name": "skeleton-line-medium",
        "selector": ".skeleton-line.skeleton-line-medium",
        "height": "12px",
        "borderRadius": "999px",
    },
    {
        "name": "skeleton-line-tiny",
        "selector": ".skeleton-line.skeleton-line-tiny",
        "height": "8.8px",
        "borderRadius": "999px",
    },
    {
        "name": "skeleton-block",
        "selector": ".skeleton-block:not(.skeleton-block-compact)",
        "minHeight": "216px",
        "borderRadius": "20px",
    },
    {
        "name": "skeleton-block-compact",
        "selector": ".skeleton-block.skeleton-block-compact",
        "minHeight": "72px",
        "borderRadius": "20px",
    },
    {
        "name": "skeleton-pill",
        "selector": ".skeleton-pill:not(.skeleton-pill-short)",
        "width": "76px",
        "height": "25.6px",
        "borderRadius": "999px",
    },
    {
        "name": "skeleton-pill-short",
        "selector": ".skeleton-pill.skeleton-pill-short",
        "width": "52px",
        "height": "25.6px",
        "borderRadius": "999px",
    },
)
PORTAL_PARITY_FEATURE_ENV = {
    "TP_PORTAL_UPLOAD_STAGING_ENABLED": "1",
    "TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT": "100",
    "TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT": "100",
    "TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT": "100",
}


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("TP_ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("TP_API_KEY", "contract-secret"))
    parser.add_argument("--baseline-path", default=str(_layer_parity_baseline_path()))
    parser.add_argument("--chrome-binary", default=os.getenv("TP_PORTAL_BROWSER_BINARY", ""))
    parser.add_argument("--debugging-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--spawn-local-backend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend-startup-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument(
        "--disable-compat-overrides",
        action="store_true",
        default=os.getenv("PORTAL_CSS_DISABLE_COMPAT_OVERRIDES") == "1",
        help=(
            "Rebuild portal.css with overrides.compat.css emitted as empty, run parity, then restore the "
            "original asset. Proves component-layer semantic rules own the style without the compat override."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _portal_css_asset_path() -> Path:
    return _repo_root() / "public" / "portal-assets" / "portal.css"


def _portal_bundle_script_path() -> Path:
    return _repo_root() / "web" / "secure-landing" / "scripts" / "build-portal-bundle.mjs"


def _build_portal_css_with_compat_disabled() -> None:
    """Rebuild portal.css with overrides.compat.css emitted as empty.

    Uses ``--css-only`` so the bundler does not also rewrite portal.js,
    portal-review.js, or shared-token assets. The probe path's ``finally``
    block only restores portal.css, so widening the rebuild beyond CSS
    would leave unrelated generated files mutated on the filesystem.
    """
    script = _portal_bundle_script_path()
    env = {**os.environ, "PORTAL_CSS_DISABLE_COMPAT_OVERRIDES": "1"}
    try:
        subprocess.run(
            ["node", str(script), "--css-only"],
            check=True,
            env=env,
            cwd=str(_repo_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SmokeFailure(f"node binary not found while rebuilding portal.css: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise SmokeFailure(
            "PORTAL_CSS_DISABLE_COMPAT_OVERRIDES=1 build failed; "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc


def _portal_shell_probe_expression() -> str:
    return r"""
(() => {
  const visible = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  };
  return {
    title: document.title,
    readyState: document.readyState,
    bootstrapStatus: document.body ? String(document.body.dataset.bootstrapStatus || '') : '',
    overviewViewVisible: visible('#overview-shell')
  };
})()
"""


def _portal_document_ready(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return value.get("readyState") == "complete" and value.get("bootstrapStatus") == "ready"


def _style_snapshot_expression() -> str:
    selectors = json.dumps(REPRESENTATIVE_STYLE_SELECTORS)
    properties = json.dumps(REPRESENTATIVE_STYLE_PROPERTIES)
    return f"""
(() => {{
  const selectors = {selectors};
  const properties = {properties};
  const snapshot = {{}};
  for (const selector of selectors) {{
    const el = document.querySelector(selector);
    if (!el) {{
      snapshot[selector] = {{ present: false }};
      continue;
    }}
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    const values = {{
      present: true,
      rect: {{
        width: Number(rect.width.toFixed(2)),
        height: Number(rect.height.toFixed(2)),
        top: Number(rect.top.toFixed(2)),
        left: Number(rect.left.toFixed(2))
      }}
    }};
    for (const property of properties) {{
      values[property] = style.getPropertyValue(property);
    }}
    snapshot[selector] = values;
  }}
  return snapshot;
}})()
"""


def _class_census_expression() -> str:
    return r"""
(() => {
  const classes = new Set();
  for (const node of document.querySelectorAll("*")) {
    for (const className of Array.from(node.classList || [])) {
      classes.add(className);
    }
  }
  return Array.from(classes).sort();
})()
"""


def _style_settle_expression() -> str:
    return "new Promise((resolve) => window.setTimeout(() => resolve(true), 450))"


def _portal_parity_probe_guard_source() -> str:
    return r"""
const createPortalParityProbeGuard = () => {
  const root = document.documentElement;
  const rootClassSnapshot = {
    present: root.hasAttribute('class'),
    value: root.getAttribute('class')
  };
  const storageSnapshots = new Map();
  const nodeSnapshots = new Map();
  const propertySnapshots = [];

  const captureStorage = (...keys) => {
    for (const key of keys) {
      if (storageSnapshots.has(key)) continue;
      try {
        storageSnapshots.set(key, {
          present: window.localStorage.getItem(key) !== null,
          value: window.localStorage.getItem(key)
        });
      } catch (_error) {}
    }
  };

  const captureNode = (node) => {
    if (!node || nodeSnapshots.has(node)) return;
    const attributes = {};
    for (const attribute of Array.from(node.attributes || [])) {
      attributes[attribute.name] = attribute.value;
    }
    nodeSnapshots.set(node, {
      attributes,
      hidden: Boolean(node.hidden),
      styleCssText: node.style ? node.style.cssText : ''
    });
  };

  const captureNodeAndAncestors = (node) => {
    let current = node;
    while (current && current !== document.body) {
      captureNode(current);
      current = current.parentElement;
    }
  };

  const captureProperty = (owner, property) => {
    if (!owner || typeof owner !== 'object') return;
    propertySnapshots.push({
      owner,
      property,
      present: Object.prototype.hasOwnProperty.call(owner, property),
      value: owner[property]
    });
  };

  const restore = () => {
    for (const snapshot of propertySnapshots.slice().reverse()) {
      if (snapshot.present) {
        snapshot.owner[snapshot.property] = snapshot.value;
      } else {
        delete snapshot.owner[snapshot.property];
      }
    }
    for (const [node, snapshot] of Array.from(nodeSnapshots.entries()).reverse()) {
      for (const attribute of Array.from(node.attributes || [])) {
        if (!Object.prototype.hasOwnProperty.call(snapshot.attributes, attribute.name)) {
          node.removeAttribute(attribute.name);
        }
      }
      for (const [name, value] of Object.entries(snapshot.attributes)) {
        node.setAttribute(name, value);
      }
      node.hidden = snapshot.hidden;
      if (node.style) {
        node.style.cssText = snapshot.styleCssText;
      }
    }
    for (const [key, snapshot] of storageSnapshots.entries()) {
      try {
        if (snapshot.present) {
          window.localStorage.setItem(key, snapshot.value || '');
        } else {
          window.localStorage.removeItem(key);
        }
      } catch (_error) {}
    }
    if (rootClassSnapshot.present) {
      root.setAttribute('class', rootClassSnapshot.value || '');
    } else {
      root.removeAttribute('class');
    }
  };

  return { captureStorage, captureNode, captureNodeAndAncestors, captureProperty, restore };
};
"""


def _portal_parity_probe_restore_expression() -> str:
    return r"""
(() => {
  const guard = window.__portalParityProbeGuard;
  window.__portalParityProbeGuard = null;
  if (guard && typeof guard.restore === 'function') {
    guard.restore();
  }
  return true;
})()
"""


def _review_status_tone_probe_expression() -> str:
    tones = json.dumps(REVIEW_STATUS_TONES)
    themes = json.dumps(REVIEW_STATUS_THEMES)
    probe_guard = _portal_parity_probe_guard_source()
    return f"""
(async () => {{
  {probe_guard}
  const tones = {tones};
  const themes = {themes};
  const results = [];
  const root = document.documentElement;
  const banner = document.getElementById('reviewStatusBanner');
  const guard = createPortalParityProbeGuard();
  guard.captureStorage('tp_theme', 'tp_theme_version');
  if (banner) {{
    guard.captureNodeAndAncestors(banner);
  }}
  try {{
    for (const theme of themes) {{
      try {{
        window.localStorage.setItem('tp_theme', theme);
        window.localStorage.setItem('tp_theme_version', '2');
      }} catch (_error) {{}}
      root.classList.toggle('light', theme === 'light');
      root.classList.toggle('dark', theme === 'dark');
      root.classList.remove('performance-lite');
      if (typeof updateUIFromState === 'function') {{
        updateUIFromState();
      }}
      for (const tone of tones) {{
        if (!banner) {{
          results.push({{ theme, tone, present: false }});
          continue;
        }}
      let current = banner;
      while (current && current !== document.body) {{
        current.classList.remove('hidden');
        current.hidden = false;
        current.removeAttribute('hidden');
        if (window.getComputedStyle(current).display === 'none') {{
          current.style.display = 'block';
        }}
        if (window.getComputedStyle(current).visibility === 'hidden') {{
          current.style.visibility = 'visible';
        }}
        current = current.parentElement;
      }}
      banner.dataset.tone = tone;
      banner.dataset.ui = 'review-status-banner';
      banner.setAttribute('aria-hidden', 'false');
      await new Promise((resolve) => window.requestAnimationFrame(() => resolve(true)));
      const style = window.getComputedStyle(banner);
      const rect = banner.getBoundingClientRect();
      results.push({{
        theme,
        tone,
        present: true,
        visible: style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0,
        backgroundColor: style.backgroundColor,
        borderColor: style.borderColor
      }});
    }}
    }}
  }} finally {{
    guard.restore();
  }}
  return results;
}})()
"""


def _validate_review_status_tone_states(connection: DevToolsConnection) -> None:
    results = connection.evaluate(_review_status_tone_probe_expression(), timeout_seconds=20.0)
    if not isinstance(results, list):
        raise SmokeFailure("Review status tone parity probe did not return results")
    expected = {(theme, tone) for theme in REVIEW_STATUS_THEMES for tone in REVIEW_STATUS_TONES}
    seen: set[tuple[str, str]] = set()
    failures: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            failures.append(f"invalid probe result {result!r}")
            continue
        theme = str(result.get("theme") or "")
        tone = str(result.get("tone") or "")
        seen.add((theme, tone))
        if not result.get("present"):
            failures.append(f"{theme}/{tone}: #reviewStatusBanner missing")
            continue
        if not result.get("visible"):
            failures.append(f"{theme}/{tone}: #reviewStatusBanner not visible")
        for property_name in ("backgroundColor", "borderColor"):
            value = str(result.get(property_name) or "").strip()
            if value in {"", "transparent", "rgba(0, 0, 0, 0)"}:
                failures.append(f"{theme}/{tone}: #reviewStatusBanner {property_name} unresolved")
    missing = expected - seen
    for theme, tone in sorted(missing):
        failures.append(f"{theme}/{tone}: review status tone state was not probed")
    if failures:
        raise SmokeFailure("Review status tone parity probe failed:\n" + "\n".join(failures))


def _overview_mobile_probe_expression() -> str:
    return r"""
(() => {
  const read = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return { present: false };
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return {
      present: true,
      display: style.display,
      justifyContent: style.justifyContent,
      gridTemplateColumns: style.gridTemplateColumns,
      width: Number(rect.width.toFixed(2)),
      parentWidth: el.parentElement ? Number(el.parentElement.getBoundingClientRect().width.toFixed(2)) : 0
    };
  };
  return {
    overviewActions: read('[data-ui="overview-actions-cluster"]'),
    buildStepperActions: read('.build-stepper-actions-inline'),
    heroAction: read('[data-ui="overview-new-run"]')
  };
})()
"""


def _grid_track_count(value: str) -> int:
    return len([part for part in value.split(" ") if part.strip()])


def _css_px_value(value: object) -> float:
    text = str(value or "").strip()
    if not text.endswith("px"):
        return -1.0
    try:
        return float(text[:-2])
    except ValueError:
        return -1.0


def _css_px_matches(actual: object, expected: object, tolerance: float = 0.05) -> bool:
    return abs(_css_px_value(actual) - _css_px_value(expected)) <= tolerance


def _validate_overview_mobile_states(connection: DevToolsConnection) -> None:
    failures: list[str] = []
    probes = [
        {"width": 767, "height": 900, "mobile": False},
        {"width": 375, "height": 900, "mobile": True},
    ]
    for probe in probes:
        width = int(probe["width"])
        connection.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": width,
                "height": int(probe["height"]),
                "deviceScaleFactor": 1,
                "mobile": bool(probe["mobile"]),
            },
            timeout_seconds=20.0,
        )
        connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
        result = connection.evaluate(_overview_mobile_probe_expression(), timeout_seconds=20.0)
        if not isinstance(result, dict):
            failures.append(f"{width}px: overview mobile probe did not return an object")
            continue
        overview = result.get("overviewActions")
        build_stepper = result.get("buildStepperActions")
        hero = result.get("heroAction")
        if not isinstance(overview, dict) or not overview.get("present"):
            failures.append(f"{width}px: overview actions cluster missing")
        else:
            if overview.get("display") != "grid":
                failures.append(f"{width}px: overview actions display is {overview.get('display')!r}")
            if overview.get("justifyContent") != "stretch":
                failures.append(f"{width}px: overview actions justify-content is {overview.get('justifyContent')!r}")
            if _grid_track_count(str(overview.get("gridTemplateColumns") or "")) != 1:
                failures.append(f"{width}px: overview actions must resolve to one grid track")
        if not isinstance(build_stepper, dict) or not build_stepper.get("present"):
            failures.append(f"{width}px: build stepper actions missing")
        elif build_stepper.get("justifyContent") != "stretch":
            failures.append(f"{width}px: build stepper actions justify-content is {build_stepper.get('justifyContent')!r}")
        if width <= 479:
            if not isinstance(hero, dict) or not hero.get("present"):
                failures.append(f"{width}px: overview hero action missing")
            else:
                if hero.get("justifyContent") != "center":
                    failures.append(f"{width}px: hero action justify-content is {hero.get('justifyContent')!r}")
                if abs(float(hero.get("width") or 0) - float(hero.get("parentWidth") or 0)) > 2:
                    failures.append(f"{width}px: hero action is not full width")
    connection.call("Emulation.clearDeviceMetricsOverride", {}, timeout_seconds=20.0)
    if failures:
        raise SmokeFailure("Overview mobile parity probe failed:\n" + "\n".join(failures))


def _interaction_outline_setup_expression() -> str:
    probe_guard = _portal_parity_probe_guard_source()
    return (
        r"""
(() => {
  """
        + probe_guard
        + r"""
  if (window.__portalParityProbeGuard && typeof window.__portalParityProbeGuard.restore === 'function') {
    window.__portalParityProbeGuard.restore();
  }
  const guard = createPortalParityProbeGuard();
  window.__portalParityProbeGuard = guard;
  guard.captureStorage('tp_theme', 'tp_theme_version');
  try {
    window.localStorage.setItem('tp_theme', 'dark');
    window.localStorage.setItem('tp_theme_version', '2');
  } catch (_error) {}
  document.documentElement.classList.remove('light');
  document.documentElement.classList.add('dark');
  document.documentElement.classList.remove('performance-lite');
  if (typeof state === 'object' && state?.portalUi) {
    guard.captureProperty(state.portalUi, 'buildStep');
    guard.captureProperty(state.portalUi, 'disclosurePrefs');
    state.portalUi.buildStep = 4;
    const disclosurePrefs = state.portalUi.disclosurePrefs || {};
    guard.captureProperty(disclosurePrefs, 'dispatchTools');
    state.portalUi.disclosurePrefs = disclosurePrefs;
    state.portalUi.disclosurePrefs.dispatchTools = true;
  }
  if (typeof updateUIFromState === 'function') {
    updateUIFromState();
  }
  if (typeof setBuildStep === 'function') {
    setBuildStep(4, { silent: true });
  }
  const dispatchTools = document.getElementById('dispatchToolsDetails');
  if (dispatchTools) {
    guard.captureNode(dispatchTools);
    dispatchTools.open = true;
    dispatchTools.classList.remove('hidden');
    dispatchTools.removeAttribute('hidden');
  }
  const dispatchTool = document.querySelector('.dispatch-tool-btn:not(.dispatch-tool-btn-primary):not([disabled])');
  guard.captureNodeAndAncestors(dispatchTool);
  let current = dispatchTool;
  while (current && current !== document.body) {
    current.classList.remove('hidden');
    current.hidden = false;
    current.removeAttribute('hidden');
    current = current.parentElement;
  }
  return {
    buildStepTab: Boolean(document.querySelector('.build-step-tab:not(.is-disabled)')),
    dispatchToolButton: Boolean(dispatchTool),
    workspaceLink: Boolean(document.querySelector('.workspace-link:not(.is-active)')),
    activeWorkspaceLink: Boolean(document.querySelector('.workspace-link.is-active')),
    dispatchToolsOpen: Boolean(dispatchTools?.open)
  };
})()
"""
    )


def _node_id_for_selector(connection: DevToolsConnection, selector: str) -> int:
    document = connection.call("DOM.getDocument", {"depth": -1, "pierce": True}, timeout_seconds=20.0)
    root = document.get("root") or {}
    root_node_id = int(root.get("nodeId") or 0)
    if root_node_id <= 0:
        raise SmokeFailure("DevTools DOM root node is unavailable")
    result = connection.call("DOM.querySelector", {"nodeId": root_node_id, "selector": selector}, timeout_seconds=20.0)
    node_id = int(result.get("nodeId") or 0)
    if node_id <= 0:
        raise SmokeFailure(f"Interaction outline probe target missing: {selector}")
    return node_id


def _interaction_outline_read_expression(selector: str) -> str:
    selector_json = json.dumps(selector)
    return f"""
(() => {{
  const el = document.querySelector({selector_json});
  if (!el) return {{ present: false }};
  const style = window.getComputedStyle(el);
  return {{
    present: true,
    hover: el.matches(':hover'),
    focusVisible: el.matches(':focus-visible'),
    outlineStyle: style.outlineStyle,
    outlineWidth: style.outlineWidth,
    outlineColor: style.outlineColor,
    transform: style.transform,
    borderColor: style.borderColor,
    backgroundColor: style.backgroundColor,
    color: style.color,
    boxShadow: style.boxShadow
  }};
}})()
"""


def _validate_interaction_outline_states(connection: DevToolsConnection) -> None:
    try:
        _validate_interaction_outline_states_with_guard(connection)
    finally:
        connection.evaluate(_portal_parity_probe_restore_expression(), timeout_seconds=20.0)


def _validate_interaction_outline_states_with_guard(connection: DevToolsConnection) -> None:
    setup = connection.evaluate(_interaction_outline_setup_expression(), timeout_seconds=20.0)
    if not isinstance(setup, dict):
        raise SmokeFailure("Interaction outline probe setup did not return a status object")
    setup_expectations = {
        "buildStepTab": ".build-step-tab:not(.is-disabled)",
        "dispatchToolButton": ".dispatch-tool-btn:not(.dispatch-tool-btn-primary):not([disabled])",
        "workspaceLink": ".workspace-link:not(.is-active)",
        "activeWorkspaceLink": ".workspace-link.is-active",
        "dispatchToolsOpen": "#dispatchToolsDetails[open]",
    }
    failures: list[str] = []
    for key, label in setup_expectations.items():
        if not setup.get(key):
            failures.append(f"setup: expected {label} to be available")
    if failures:
        raise SmokeFailure("Interaction outline parity probe failed:\n" + "\n".join(failures))

    connection.call("DOM.enable", {}, timeout_seconds=20.0)
    connection.call("CSS.enable", {}, timeout_seconds=20.0)
    for probe in INTERACTION_OUTLINE_PROBES:
        name = str(probe["name"])
        selector = str(probe["selector"])
        shared_properties = tuple(str(property_name) for property_name in probe["sharedProperties"])
        combined_outline_style = str(probe.get("combinedOutlineStyle") or "none")
        node_id = _node_id_for_selector(connection, selector)

        def force_and_read(pseudo_classes: list[str]) -> dict[str, Any]:
            connection.call(
                "CSS.forcePseudoState",
                {"nodeId": node_id, "forcedPseudoClasses": pseudo_classes},
                timeout_seconds=20.0,
            )
            connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
            result = connection.evaluate(_interaction_outline_read_expression(selector), timeout_seconds=20.0)
            if not isinstance(result, dict) or not result.get("present"):
                raise SmokeFailure(f"Interaction outline probe target disappeared: {selector}")
            return result

        try:
            force_and_read([])
            hover = force_and_read(["hover"])
            focus_visible = force_and_read(["focus-visible"])
            combined = force_and_read(["hover", "focus-visible"])
        finally:
            connection.call(
                "CSS.forcePseudoState",
                {"nodeId": node_id, "forcedPseudoClasses": []},
                timeout_seconds=20.0,
            )

        if hover.get("hover") is not True:
            failures.append(f"{name}: hover pseudo-state was not applied")
        if hover.get("focusVisible"):
            failures.append(f"{name}: hover-only probe unexpectedly matched :focus-visible")
        if focus_visible.get("focusVisible") is not True:
            failures.append(f"{name}: focus-visible pseudo-state was not applied")
        if focus_visible.get("hover"):
            failures.append(f"{name}: focus-visible-only probe unexpectedly matched :hover")
        if combined.get("hover") is not True or combined.get("focusVisible") is not True:
            failures.append(f"{name}: combined hover + focus-visible state was not applied")

        if hover.get("outlineStyle") != "none":
            failures.append(f"{name}: hover outlineStyle is {hover.get('outlineStyle')!r}, expected 'none'")
        if focus_visible.get("outlineStyle") == "none":
            failures.append(f"{name}: focus-visible-only outline was suppressed")
        if combined.get("outlineStyle") != combined_outline_style:
            failures.append(
                f"{name}: combined hover + focus-visible outlineStyle is {combined.get('outlineStyle')!r}, expected {combined_outline_style!r}"
            )

        for property_name in shared_properties:
            hover_value = str(hover.get(property_name) or "")
            focus_value = str(focus_visible.get(property_name) or "")
            combined_value = str(combined.get(property_name) or "")
            if focus_value != hover_value:
                failures.append(f"{name}: focus-visible {property_name} drifted from hover value")
            if combined_value != hover_value:
                failures.append(f"{name}: combined {property_name} drifted from hover value")

    if failures:
        raise SmokeFailure("Interaction outline parity probe failed:\n" + "\n".join(failures))


def _skeleton_visibility_probe_expression(parity_state: str) -> str:
    state_json = json.dumps(parity_state)
    skeleton_ids = json.dumps(SKELETON_STATE_IDS)
    probes = json.dumps(SKELETON_STYLE_PROBES)
    probe_guard = _portal_parity_probe_guard_source()
    return f"""
(() => {{
  {probe_guard}
  const parityState = {state_json};
  const skeletonIds = {skeleton_ids};
  const probes = {probes};
  const theme = parityState === 'light' ? 'light' : 'dark';
  const guard = createPortalParityProbeGuard();
  guard.captureStorage('tp_theme', 'tp_theme_version');
  try {{
  try {{
    window.localStorage.setItem('tp_theme', theme);
    window.localStorage.setItem('tp_theme_version', '2');
  }} catch (_error) {{}}
  const root = document.documentElement;
  root.classList.toggle('light', theme === 'light');
  root.classList.toggle('dark', theme === 'dark');
  root.classList.remove('performance-lite');
  if (typeof updateUIFromState === 'function') {{
    updateUIFromState();
  }}
  const skeletonStates = {{}};
  const skeletonRoots = [];
  for (const id of skeletonIds) {{
    const node = document.getElementById(id);
    if (!node) {{
      skeletonStates[id] = {{ present: false }};
      continue;
    }}
    guard.captureNodeAndAncestors(node);
    let current = node;
    while (current && current !== document.body) {{
      current.classList.remove('hidden');
      current.hidden = false;
      current.removeAttribute('hidden');
      if (current.style && current.style.display === 'none') {{
        current.style.display = '';
      }}
      current = current.parentElement;
    }}
    node.setAttribute('aria-hidden', 'false');
    skeletonRoots.push(node);
    const rect = node.getBoundingClientRect();
    skeletonStates[id] = {{
      present: true,
      hidden: node.classList.contains('hidden') || node.hidden || node.hasAttribute('hidden'),
      width: Number(rect.width.toFixed(2)),
      height: Number(rect.height.toFixed(2))
    }};
  }}
  const styles = {{}};
  for (const probe of probes) {{
    let el = null;
    for (const skeletonRoot of skeletonRoots) {{
      el = skeletonRoot.querySelector(probe.selector);
      if (el) break;
    }}
    if (!el) {{
      styles[probe.name] = {{ present: false }};
      continue;
    }}
    const style = window.getComputedStyle(el);
    const before = window.getComputedStyle(el, '::before');
    const rect = el.getBoundingClientRect();
    styles[probe.name] = {{
      present: true,
      display: style.display,
      overflow: style.overflow,
      position: style.position,
      backgroundColor: style.backgroundColor,
      height: style.height,
      minHeight: style.minHeight,
      width: style.width,
      borderRadius: style.borderRadius,
      animationName: style.animationName,
      transitionDuration: style.transitionDuration,
      transform: style.transform,
      rect: {{
        width: Number(rect.width.toFixed(2)),
        height: Number(rect.height.toFixed(2))
      }},
      before: {{
        content: before.content,
        position: before.position,
        pointerEvents: before.pointerEvents,
        backgroundImage: before.backgroundImage,
        animationName: before.animationName
      }}
    }};
  }}
  const result = {{ parityState, skeletonStates, styles }};
  return result;
  }} finally {{
    guard.restore();
  }}
}})()
"""


def _validate_skeleton_primitive_states(connection: DevToolsConnection) -> None:
    failures: list[str] = []
    expected_backgrounds = {
        "light": "rgba(226, 232, 240, 0.72)",
        "dark": "rgba(51, 65, 85, 0.72)",
        "reduced-motion": "rgba(51, 65, 85, 0.72)",
    }
    for state in ("light", "dark", "reduced-motion"):
        if state == "reduced-motion":
            connection.call(
                "Emulation.setEmulatedMedia",
                {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]},
                timeout_seconds=20.0,
            )
        else:
            connection.call("Emulation.setEmulatedMedia", {"features": []}, timeout_seconds=20.0)
        connection.evaluate(_skeleton_visibility_probe_expression(state), timeout_seconds=20.0)
        connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
        result = connection.evaluate(_skeleton_visibility_probe_expression(state), timeout_seconds=20.0)
        if not isinstance(result, dict):
            failures.append(f"{state}: skeleton probe did not return an object")
            continue
        skeleton_states = result.get("skeletonStates")
        styles = result.get("styles")
        if not isinstance(skeleton_states, dict) or not isinstance(styles, dict):
            failures.append(f"{state}: skeleton probe returned malformed state")
            continue
        for skeleton_id in SKELETON_STATE_IDS:
            status = skeleton_states.get(skeleton_id)
            if not isinstance(status, dict) or not status.get("present"):
                failures.append(f"{state}: #{skeleton_id} missing")
                continue
            if status.get("hidden"):
                failures.append(f"{state}: #{skeleton_id} remained hidden")
            if float(status.get("height") or 0) <= 0:
                failures.append(f"{state}: #{skeleton_id} has no layout height")
        expected_background = expected_backgrounds[state]
        for probe in SKELETON_STYLE_PROBES:
            name = str(probe["name"])
            style = styles.get(name)
            if not isinstance(style, dict) or not style.get("present"):
                failures.append(f"{state}: {name} missing")
                continue
            if style.get("display") != "block":
                failures.append(f"{state}: {name} display is {style.get('display')!r}")
            if style.get("overflow") != "hidden":
                failures.append(f"{state}: {name} overflow is {style.get('overflow')!r}")
            if style.get("position") != "relative":
                failures.append(f"{state}: {name} position is {style.get('position')!r}")
            if style.get("backgroundColor") != expected_background:
                failures.append(f"{state}: {name} background is {style.get('backgroundColor')!r}")
            if "height" in probe and not _css_px_matches(style.get("height"), probe["height"]):
                failures.append(f"{state}: {name} height is {style.get('height')!r}")
            if "minHeight" in probe and not _css_px_matches(style.get("minHeight"), probe["minHeight"]):
                failures.append(f"{state}: {name} min-height is {style.get('minHeight')!r}")
            if "width" in probe and not _css_px_matches(style.get("width"), probe["width"]):
                failures.append(f"{state}: {name} width is {style.get('width')!r}")
            if style.get("borderRadius") != probe["borderRadius"]:
                failures.append(f"{state}: {name} border-radius is {style.get('borderRadius')!r}")
            before = style.get("before") or {}
            if before.get("content") != '""':
                failures.append(f"{state}: {name} shimmer content is {before.get('content')!r}")
            if before.get("position") != "absolute":
                failures.append(f"{state}: {name} shimmer position is {before.get('position')!r}")
            if before.get("pointerEvents") != "none":
                failures.append(f"{state}: {name} shimmer pointer-events is {before.get('pointerEvents')!r}")
            if "linear-gradient" not in str(before.get("backgroundImage") or ""):
                failures.append(f"{state}: {name} shimmer background is missing")
        connection.call("Emulation.setEmulatedMedia", {"features": []}, timeout_seconds=20.0)

    if failures:
        raise SmokeFailure("Skeleton primitive parity probe failed:\n" + "\n".join(failures))


def _surface_loading_probe_expression(parity_state: str) -> str:
    state_json = json.dumps(parity_state)
    probe_guard = _portal_parity_probe_guard_source()
    return f"""
(() => {{
  {probe_guard}
  const parityState = {state_json};
  const theme = parityState === 'light' ? 'light' : 'dark';
  const guard = createPortalParityProbeGuard();
  let probe = null;
  guard.captureStorage('tp_theme', 'tp_theme_version');
  try {{
  try {{
    window.localStorage.setItem('tp_theme', theme);
    window.localStorage.setItem('tp_theme_version', '2');
  }} catch (_error) {{}}
  const root = document.documentElement;
  root.classList.toggle('light', theme === 'light');
  root.classList.toggle('dark', theme === 'dark');
  root.classList.remove('performance-lite');
  if (typeof updateUIFromState === 'function') {{
    updateUIFromState();
  }}
  probe = document.createElement('section');
  probe.className = 'surface-loading';
  probe.setAttribute('data-ui', 'surface-loading-phase15-probe');
  probe.setAttribute('aria-busy', 'true');
  probe.style.width = '320px';
  probe.style.height = '96px';
  probe.style.margin = '0';
  probe.style.padding = '0';
  document.body.appendChild(probe);
  const style = window.getComputedStyle(probe);
  const after = window.getComputedStyle(probe, '::after');
  const rect = probe.getBoundingClientRect();
  return {{
    parityState,
    surface: {{
      present: true,
      rect: {{
        width: Number(rect.width.toFixed(2)),
        height: Number(rect.height.toFixed(2))
      }},
      style: {{
        position: style.position,
        borderTopWidth: style.borderTopWidth,
        borderTopStyle: style.borderTopStyle,
        borderTopColor: style.borderTopColor,
        backgroundImage: style.backgroundImage,
        backgroundColor: style.backgroundColor,
        boxShadow: style.boxShadow,
        transitionProperty: style.transitionProperty,
        transitionDuration: style.transitionDuration,
        transform: style.transform
      }},
      after: {{
        content: after.content,
        position: after.position,
        top: after.top,
        height: after.height,
        borderRadius: after.borderRadius,
        pointerEvents: after.pointerEvents,
        opacity: after.opacity,
        backgroundImage: after.backgroundImage,
        transitionDuration: after.transitionDuration,
        transform: after.transform
      }}
    }}
  }};
  }} finally {{
    if (probe && probe.parentNode) {{
      probe.parentNode.removeChild(probe);
    }}
    guard.restore();
  }}
}})()
"""


def _validate_surface_loading_states(connection: DevToolsConnection) -> None:
    failures: list[str] = []
    expected_border_colors = {
        "light": "rgba(148, 163, 184, 0.16)",
        "dark": "rgba(71, 85, 105, 0.48)",
        "reduced-motion": "rgba(71, 85, 105, 0.48)",
    }
    for state in ("light", "dark", "reduced-motion"):
        if state == "reduced-motion":
            connection.call(
                "Emulation.setEmulatedMedia",
                {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]},
                timeout_seconds=20.0,
            )
        else:
            connection.call("Emulation.setEmulatedMedia", {"features": []}, timeout_seconds=20.0)
        connection.evaluate(_surface_loading_probe_expression(state), timeout_seconds=20.0)
        connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
        result = connection.evaluate(_surface_loading_probe_expression(state), timeout_seconds=20.0)
        if not isinstance(result, dict):
            failures.append(f"{state}: surface-loading probe did not return an object")
            continue
        surface = result.get("surface")
        if not isinstance(surface, dict):
            failures.append(f"{state}: surface-loading probe returned malformed state")
            continue
        rect = surface.get("rect") or {}
        if float(rect.get("height") or 0) <= 0:
            failures.append(f"{state}: surface-loading probe has no layout height")
        style = surface.get("style") or {}
        if style.get("position") != "relative":
            failures.append(f"{state}: surface-loading position is {style.get('position')!r}")
        if style.get("borderTopWidth") != "1px" or style.get("borderTopStyle") != "solid":
            failures.append(
                f"{state}: surface-loading border is {style.get('borderTopWidth')!r} {style.get('borderTopStyle')!r}"
            )
        if style.get("borderTopColor") != expected_border_colors[state]:
            failures.append(f"{state}: surface-loading border color is {style.get('borderTopColor')!r}")
        if "linear-gradient" not in str(style.get("backgroundImage") or ""):
            failures.append(f"{state}: surface-loading background gradient is missing")
        if style.get("boxShadow") in {"", "none"}:
            failures.append(f"{state}: surface-loading box-shadow is missing")
        after = surface.get("after") or {}
        if after.get("content") != '""':
            failures.append(f"{state}: surface-loading ::after content is {after.get('content')!r}")
        if after.get("position") != "absolute":
            failures.append(f"{state}: surface-loading ::after position is {after.get('position')!r}")
        if after.get("top") != "12px":
            failures.append(f"{state}: surface-loading ::after top is {after.get('top')!r}")
        if after.get("height") != "2px":
            failures.append(f"{state}: surface-loading ::after height is {after.get('height')!r}")
        if after.get("borderRadius") != "999px":
            failures.append(f"{state}: surface-loading ::after border-radius is {after.get('borderRadius')!r}")
        if after.get("pointerEvents") != "none":
            failures.append(f"{state}: surface-loading ::after pointer-events is {after.get('pointerEvents')!r}")
        if after.get("opacity") != "0.82":
            failures.append(f"{state}: surface-loading ::after opacity is {after.get('opacity')!r}")
        if "linear-gradient" not in str(after.get("backgroundImage") or ""):
            failures.append(f"{state}: surface-loading ::after background gradient is missing")
        if state == "reduced-motion":
            if style.get("transitionDuration") not in {"0s", "1e-05s"}:
                failures.append(f"{state}: surface-loading transition duration is {style.get('transitionDuration')!r}")
            if style.get("transform") != "none":
                failures.append(f"{state}: surface-loading transform is {style.get('transform')!r}")
            if after.get("transitionDuration") not in {"0s", "1e-05s"}:
                failures.append(f"{state}: surface-loading ::after transition duration is {after.get('transitionDuration')!r}")
            if after.get("transform") != "none":
                failures.append(f"{state}: surface-loading ::after transform is {after.get('transform')!r}")
        connection.call("Emulation.setEmulatedMedia", {"features": []}, timeout_seconds=20.0)

    if failures:
        raise SmokeFailure("Surface loading parity probe failed:\n" + "\n".join(failures))


def _enable_portal_feature_rollouts_for_spawn() -> dict[str, Optional[str]]:
    previous: dict[str, Optional[str]] = {}
    if not {
        "staged-uploads",
        "artifact-viewer-modal",
        "review-surface-defer",
    }.intersection(LAYER_PARITY_STATES):
        return previous
    for name, value in PORTAL_PARITY_FEATURE_ENV.items():
        previous[name] = os.environ.get(name)
        os.environ[name] = value
    return previous


def _restore_env(previous: dict[str, Optional[str]]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _force_snapshot_state_expression() -> str:
    probe_guard = _portal_parity_probe_guard_source()
    return (
        r"""
(() => {
  """
        + probe_guard
        + r"""
  if (window.__portalParityProbeGuard && typeof window.__portalParityProbeGuard.restore === 'function') {
    window.__portalParityProbeGuard.restore();
  }
  const guard = createPortalParityProbeGuard();
  window.__portalParityProbeGuard = guard;
  guard.captureStorage('tp_theme', 'tp_theme_version');
  try {
    window.localStorage.setItem('tp_theme', 'dark');
    window.localStorage.setItem('tp_theme_version', '2');
  } catch (_error) {}
  document.documentElement.classList.remove('light');
  document.documentElement.classList.add('dark');
  document.documentElement.classList.remove('performance-lite');
  const portalState = typeof state === 'object' && state ? state : null;
  if (portalState?.auth?.features) {
    guard.captureProperty(portalState.auth.features, 'artifactViewerModal');
    guard.captureProperty(portalState.auth.features, 'reviewSurfaceDeferred');
    guard.captureProperty(portalState.auth.features, 'stagedUploads');
    portalState.auth.features.artifactViewerModal = false;
    portalState.auth.features.reviewSurfaceDeferred = false;
    portalState.auth.features.stagedUploads = false;
  }
  if (typeof updateUIFromState === 'function') {
    updateUIFromState();
  }
  const artifactViewerModal = document.getElementById('artifactViewerModal');
  if (artifactViewerModal) {
    guard.captureNode(artifactViewerModal);
    artifactViewerModal.classList.add('hidden');
    artifactViewerModal.classList.remove('flex');
    artifactViewerModal.setAttribute('aria-hidden', 'true');
    artifactViewerModal.dataset.overlayOpen = 'false';
  }
  return true;
})()
"""
    )


def _force_census_state_expression(state: str) -> str:
    state_json = json.dumps(state)
    probe_guard = _portal_parity_probe_guard_source()
    return f"""
(async () => {{
  {probe_guard}
  if (window.__portalParityProbeGuard && typeof window.__portalParityProbeGuard.restore === 'function') {{
    window.__portalParityProbeGuard.restore();
  }}
  const guard = createPortalParityProbeGuard();
  window.__portalParityProbeGuard = guard;
  guard.captureStorage('tp_theme', 'tp_theme_version');
  const parityState = {state_json};
  const theme = parityState === 'light' ? 'light' : 'dark';
  try {{
    window.localStorage.setItem('tp_theme', theme);
    window.localStorage.setItem('tp_theme_version', '2');
  }} catch (_error) {{}}
  const root = document.documentElement;
  root.classList.toggle('light', theme === 'light');
  root.classList.toggle('dark', theme === 'dark');
  root.classList.toggle('performance-lite', parityState === 'performance-lite');

  const portalState = typeof state === 'object' && state ? state : null;
  if (portalState?.auth?.features) {{
    guard.captureProperty(portalState.auth.features, 'stagedUploads');
    guard.captureProperty(portalState.auth.features, 'artifactViewerModal');
    guard.captureProperty(portalState.auth.features, 'reviewSurfaceDeferred');
    guard.captureProperty(portalState, 'pipeline');
    if (parityState === 'staged-uploads') {{
      portalState.auth.features.stagedUploads = true;
      portalState.pipeline = 'lux-depth-v3';
    }}
    if (parityState === 'artifact-viewer-modal') {{
      portalState.auth.features.artifactViewerModal = true;
    }}
    if (parityState === 'review-surface-defer') {{
      portalState.auth.features.reviewSurfaceDeferred = true;
    }}
  }}
  if (typeof updateUIFromState === 'function') {{
    updateUIFromState();
  }}
  if (parityState === 'artifact-viewer-modal') {{
    const modal = document.getElementById('artifactViewerModal');
    if (modal) {{
      guard.captureNode(modal);
      modal.classList.remove('hidden');
      modal.classList.add('flex');
      modal.setAttribute('aria-hidden', 'false');
      modal.dataset.overlayOpen = 'true';
    }}
  }}
  if (parityState === 'review-surface-defer') {{
    if (typeof _primeDeferredReviewSurface === 'function') {{
      _primeDeferredReviewSurface('parity-census');
    }}
    if (typeof _loadDeferredReviewSurface === 'function') {{
      try {{
        await _loadDeferredReviewSurface();
      }} catch (_error) {{}}
    }}
  }}
  return true;
}})()
"""


def _portal_view_url(base_url: str, view: str) -> str:
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}view={quote(view)}"


def _collect_runtime_utility_classes(
    connection: DevToolsConnection,
    base_url: str,
    timeout_seconds: float,
) -> set[str]:
    runtime_utility_classes: set[str] = set()
    views = LAYER_PARITY_VIEWS or ["overview"]
    states = LAYER_PARITY_STATES or ["dark"]

    for view in views:
        connection.call("Page.navigate", {"url": _portal_view_url(base_url, view)}, timeout_seconds=20.0)
        _poll(
            connection,
            _portal_shell_probe_expression(),
            predicate=_portal_document_ready,
            timeout_seconds=timeout_seconds,
            description=f"portal document ready for {view} view",
        )
        for state in states:
            try:
                if state == "reduced-motion":
                    connection.call(
                        "Emulation.setEmulatedMedia",
                        {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]},
                        timeout_seconds=20.0,
                    )
                else:
                    connection.call("Emulation.setEmulatedMedia", {"features": []}, timeout_seconds=20.0)
                connection.evaluate(_force_census_state_expression(state), timeout_seconds=20.0)
                connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
                runtime_classes = connection.evaluate(_class_census_expression(), timeout_seconds=20.0)
                if not isinstance(runtime_classes, list):
                    raise SmokeFailure("Runtime class census did not return a class list")
                runtime_utility_classes.update(
                    str(class_name)
                    for class_name in runtime_classes
                    if _is_utility_like_class_token(str(class_name))
                )
            finally:
                connection.evaluate(_portal_parity_probe_restore_expression(), timeout_seconds=20.0)
                connection.call("Emulation.setEmulatedMedia", {"features": []}, timeout_seconds=20.0)

    return runtime_utility_classes


def _diff_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> list[str]:
    differences: list[str] = []
    for selector in REPRESENTATIVE_STYLE_SELECTORS:
        before_values = before.get(selector)
        after_values = after.get(selector)
        if before_values != after_values:
            if not isinstance(before_values, dict) or not isinstance(after_values, dict):
                differences.append(f"{selector}: snapshot changed")
                continue
            keys = sorted(set(before_values) | set(after_values))
            for key in keys:
                if before_values.get(key) != after_values.get(key):
                    differences.append(
                        f"{selector} {key}: {before_values.get(key)!r} -> {after_values.get(key)!r}"
                    )
    return differences


def _read_baseline(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SmokeFailure(f"Layer parity baseline is missing: {path}")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    selectors = baseline.get("representativeStyleSelectors")
    properties = baseline.get("representativeStyleProperties")
    if selectors != REPRESENTATIVE_STYLE_SELECTORS or properties != REPRESENTATIVE_STYLE_PROPERTIES:
        raise SmokeFailure("Layer parity baseline contract does not match layer-parity-contract.json")
    snapshot = baseline.get("snapshot")
    if not isinstance(snapshot, dict):
        raise SmokeFailure(f"Layer parity baseline has no snapshot object: {path}")
    return snapshot


def _class_token_base(token: str) -> str:
    parts = token.split(":")
    while len(parts) > 1 and parts[0] in UTILITY_VARIANTS:
        parts.pop(0)
    return ":".join(parts)


def _is_utility_like_class_token(token: str) -> bool:
    if not token or token in UTILITY_OWNER_ALLOWLIST:
        return False
    if token.startswith(":") or token.endswith(":") or any(character in token for character in "$\"'`;=<>?{}()"):
        return False
    base = _class_token_base(token)
    if base in UTILITY_EXACT_CLASSES or base in {"underline", "no-underline"}:
        return True
    return any(base.startswith(f"{prefix}-") for prefix in UTILITY_PREFIXES)


def _read_utility_ownership_classes() -> set[str]:
    path = _utility_ownership_path()
    if not path.exists():
        raise SmokeFailure(f"Utility ownership manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    utilities = manifest.get("utilities")
    if not isinstance(utilities, dict):
        raise SmokeFailure(f"Utility ownership manifest has no utilities object: {path}")
    return set(str(token) for token in utilities)


def _write_baseline(path: Path, snapshot: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "representativeStyleSelectors": REPRESENTATIVE_STYLE_SELECTORS,
                "representativeStyleProperties": REPRESENTATIVE_STYLE_PROPERTIES,
                "snapshot": snapshot,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _launch_chrome(chrome_binary: str, port: int, profile_dir: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            chrome_binary,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-sync",
            "--disable-extensions",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    runtime_handle: Optional[LocalRuntimeHandle] = None
    chrome_process: Optional[subprocess.Popen[str]] = None
    connection: Optional[DevToolsConnection] = None
    profile_dir: Optional[Path] = None
    portal_css_backup: Optional[bytes] = None
    portal_css_path = _portal_css_asset_path()
    probe_mode = bool(args.disable_compat_overrides)

    try:
        if probe_mode:
            if args.write_baseline:
                raise SmokeFailure(
                    "--write-baseline must not be combined with --disable-compat-overrides; "
                    "the probe build is not a valid baseline source."
                )
            if not portal_css_path.exists():
                raise SmokeFailure(
                    f"Cannot enter compat-disabled probe mode: {portal_css_path} is missing. "
                    "Run npm run build:portal first."
                )
            portal_css_backup = portal_css_path.read_bytes()
            print(
                "portal-css-layer-parity: probe mode active "
                "(PORTAL_CSS_DISABLE_COMPAT_OVERRIDES=1); rebuilding portal.css without overrides.compat.css",
                flush=True,
            )
            _build_portal_css_with_compat_disabled()

        base_url = _base_url(str(args.base_url))
        if args.spawn_local_backend:
            print("portal-css-layer-parity: launching isolated local backend", flush=True)
            previous_env = _enable_portal_feature_rollouts_for_spawn()
            try:
                runtime_handle = _spawn_local_backend(
                    str(args.api_key),
                    timeout_seconds=float(args.backend_startup_timeout_seconds),
                )
            finally:
                _restore_env(previous_env)
            base_url = runtime_handle.base_url
            print(f"portal-css-layer-parity: isolated backend ready at {base_url}", flush=True)

        profile_dir = _default_profile_dir()
        port = int(args.debugging_port or _find_free_port())
        chrome_binary = _resolve_chrome_binary(str(args.chrome_binary))
        print("portal-css-layer-parity: launching chrome", flush=True)
        chrome_process = _launch_chrome(chrome_binary, port, profile_dir)

        _wait_for_devtools(port)
        target = _wait_for_page_target(port)
        websocket_url = str(target.get("webSocketDebuggerUrl") or "").strip()
        if not websocket_url.startswith("ws://"):
            raise SmokeFailure(f"Invalid DevTools websocket URL: {websocket_url!r}")

        connection = DevToolsConnection(websocket_url)
        connection.call("Page.enable")
        connection.call("Page.setBypassCSP", {"enabled": True})
        connection.call("Runtime.enable")
        connection.call("Page.navigate", {"url": base_url}, timeout_seconds=20.0)

        _poll(
            connection,
            _portal_shell_probe_expression(),
            predicate=_portal_shell_ready,
            timeout_seconds=float(args.timeout_seconds),
            description="portal document ready",
        )
        try:
            connection.evaluate(_force_snapshot_state_expression(), timeout_seconds=20.0)
            connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
            current = connection.evaluate(_style_snapshot_expression(), timeout_seconds=20.0)
        finally:
            connection.evaluate(_portal_parity_probe_restore_expression(), timeout_seconds=20.0)

        baseline_path = Path(str(args.baseline_path))
        if args.write_baseline:
            _write_baseline(baseline_path, current)
            print(
                "portal-css-layer-parity: wrote baseline "
                f"{baseline_path} ({len(REPRESENTATIVE_STYLE_SELECTORS)} selectors, "
                f"{len(REPRESENTATIVE_STYLE_PROPERTIES)} properties)",
                flush=True,
            )
            return 0

        expected = _read_baseline(baseline_path)
        differences = _diff_snapshots(expected, current)
        if differences:
            detail = "\n".join(differences[:40])
            suffix = f"\n... {len(differences) - 40} additional differences" if len(differences) > 40 else ""
            mode_label = (
                "compat-disabled probe parity failed: a candidate semantic rule does not own this style "
                "without overrides.compat.css"
                if probe_mode
                else "Layered CSS computed-style parity failed against baseline"
            )
            raise SmokeFailure(f"{mode_label}:\n{detail}{suffix}")

        _validate_review_status_tone_states(connection)
        _validate_overview_mobile_states(connection)
        _validate_interaction_outline_states(connection)
        _validate_skeleton_primitive_states(connection)
        _validate_surface_loading_states(connection)

        runtime_utility_classes = _collect_runtime_utility_classes(
            connection,
            base_url,
            float(args.timeout_seconds),
        )
        utility_ownership_classes = _read_utility_ownership_classes()
        missing_runtime_owners = sorted(runtime_utility_classes - utility_ownership_classes)
        if missing_runtime_owners:
            detail = ", ".join(missing_runtime_owners[:40])
            suffix = (
                f"; {len(missing_runtime_owners) - 40} additional missing utility owners"
                if len(missing_runtime_owners) > 40
                else ""
            )
            raise SmokeFailure(f"Runtime class census found utility classes missing ownership: {detail}{suffix}")

        print(
            "portal-css-layer-parity: ok "
            f"({'compat-disabled probe; ' if probe_mode else ''}"
            f"{len(REPRESENTATIVE_STYLE_SELECTORS)} selectors, "
            f"{len(REPRESENTATIVE_STYLE_PROPERTIES)} properties, "
            f"{len(runtime_utility_classes)} runtime utility classes)",
            flush=True,
        )
        return 0
    except SmokeFailure as exc:
        print(f"portal-css-layer-parity: failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()
        if chrome_process is not None and chrome_process.poll() is None:
            chrome_process.terminate()
            try:
                chrome_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome_process.kill()
                chrome_process.wait(timeout=5)
        if profile_dir is not None:
            shutil.rmtree(profile_dir, ignore_errors=True)
        if runtime_handle is not None:
            _terminate_runtime(runtime_handle)
        if portal_css_backup is not None:
            try:
                portal_css_path.write_bytes(portal_css_backup)
                print(
                    f"portal-css-layer-parity: restored {portal_css_path} from pre-probe backup",
                    flush=True,
                )
            except OSError as exc:
                print(
                    f"portal-css-layer-parity: WARNING failed to restore {portal_css_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    raise SystemExit(main())
