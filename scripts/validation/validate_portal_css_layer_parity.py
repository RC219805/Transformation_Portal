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
    return parser.parse_args(list(argv) if argv is not None else None)


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
    return r"""
(() => {
  try {
    window.localStorage.setItem('tp_theme', 'dark');
    window.localStorage.setItem('tp_theme_version', '2');
  } catch (_error) {}
  document.documentElement.classList.remove('light');
  document.documentElement.classList.add('dark');
  document.documentElement.classList.remove('performance-lite');
  const portalState = typeof state === 'object' && state ? state : null;
  if (portalState?.auth?.features) {
    portalState.auth.features.artifactViewerModal = false;
    portalState.auth.features.reviewSurfaceDeferred = false;
    portalState.auth.features.stagedUploads = false;
  }
  if (typeof updateUIFromState === 'function') {
    updateUIFromState();
  }
  const artifactViewerModal = document.getElementById('artifactViewerModal');
  if (artifactViewerModal) {
    artifactViewerModal.classList.add('hidden');
    artifactViewerModal.classList.remove('flex');
    artifactViewerModal.setAttribute('aria-hidden', 'true');
    artifactViewerModal.dataset.overlayOpen = 'false';
  }
  return true;
})()
"""


def _force_census_state_expression(state: str) -> str:
    state_json = json.dumps(state)
    return f"""
(async () => {{
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

    try:
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
        connection.evaluate(_force_snapshot_state_expression(), timeout_seconds=20.0)
        connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
        current = connection.evaluate(_style_snapshot_expression(), timeout_seconds=20.0)

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
            raise SmokeFailure(f"Layered CSS computed-style parity failed against baseline:\n{detail}{suffix}")

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
            f"({len(REPRESENTATIVE_STYLE_SELECTORS)} selectors, "
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


if __name__ == "__main__":
    raise SystemExit(main())
