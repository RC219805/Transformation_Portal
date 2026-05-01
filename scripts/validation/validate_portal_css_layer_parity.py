#!/usr/bin/env python3
"""Compare current portal CSS against the proposed layered CSS graph in Chrome."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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


def _frontdoor_root() -> Path:
    return _repo_root() / "web" / "secure-landing"


def _layer_parity_contract_path() -> Path:
    return _frontdoor_root() / "portal-src" / "styles" / "layer-parity-contract.json"


def _load_layer_parity_contract() -> tuple[list[str], list[str]]:
    contract = json.loads(_layer_parity_contract_path().read_text(encoding="utf-8"))
    return (
        list(contract["representativeStyleSelectors"]),
        list(contract["representativeStyleProperties"]),
    )


REPRESENTATIVE_STYLE_SELECTORS, REPRESENTATIVE_STYLE_PROPERTIES = _load_layer_parity_contract()


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("TP_ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("TP_API_KEY", "contract-secret"))
    parser.add_argument("--chrome-binary", default=os.getenv("TP_PORTAL_BROWSER_BINARY", ""))
    parser.add_argument("--debugging-port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--spawn-local-backend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend-startup-timeout-seconds", type=float, default=30.0)
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


def _apply_layered_css_expression(layered_css: str) -> str:
    encoded = base64.b64encode(layered_css.encode("utf-8")).decode("ascii")
    return f"""
(() => {{
  for (const link of Array.from(document.querySelectorAll('link[rel="stylesheet"]'))) {{
    if (String(link.href || '').includes('/portal/assets/portal.css')) {{
      link.disabled = true;
      link.setAttribute('data-layer-parity-disabled', 'true');
    }}
  }}
  const previous = document.getElementById('portal-layer-parity-css');
  if (previous) previous.remove();
  const style = document.createElement('style');
  style.id = 'portal-layer-parity-css';
  style.textContent = atob('{encoded}');
  document.head.appendChild(style);
  return new Promise((resolve) => {{
    requestAnimationFrame(() => {{
      requestAnimationFrame(() => {{
        window.setTimeout(() => resolve(true), 450);
      }});
    }});
  }});
}})()
"""


def _style_settle_expression() -> str:
    return "new Promise((resolve) => window.setTimeout(() => resolve(true), 450))"


def _write_layered_css(temp_dir: Path) -> Path:
    output_path = temp_dir / "portal-layered.css"
    subprocess.run(
        [
            "node",
            "./scripts/check-portal-css-layer-dry-run.mjs",
            "--write-css",
            str(output_path),
        ],
        cwd=_frontdoor_root(),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return output_path


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
    temp_dir = Path(tempfile.mkdtemp(prefix="tp-portal-layer-parity-"))

    try:
        base_url = _base_url(str(args.base_url))
        if args.spawn_local_backend:
            print("portal-css-layer-parity: launching isolated local backend", flush=True)
            runtime_handle = _spawn_local_backend(
                str(args.api_key),
                timeout_seconds=float(args.backend_startup_timeout_seconds),
            )
            base_url = runtime_handle.base_url
            print(f"portal-css-layer-parity: isolated backend ready at {base_url}", flush=True)

        layered_css_path = _write_layered_css(temp_dir)
        layered_css = layered_css_path.read_text(encoding="utf-8")

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
        connection.evaluate(_style_settle_expression(), timeout_seconds=20.0)
        before = connection.evaluate(_style_snapshot_expression(), timeout_seconds=20.0)
        connection.evaluate(_apply_layered_css_expression(layered_css), timeout_seconds=20.0)
        after = connection.evaluate(_style_snapshot_expression(), timeout_seconds=20.0)
        differences = _diff_snapshots(before, after)
        if differences:
            detail = "\n".join(differences[:40])
            suffix = f"\n... {len(differences) - 40} additional differences" if len(differences) > 40 else ""
            raise SmokeFailure(f"Layered CSS computed-style parity failed:\n{detail}{suffix}")

        print(
            "portal-css-layer-parity: ok "
            f"({len(REPRESENTATIVE_STYLE_SELECTORS)} selectors, "
            f"{len(REPRESENTATIVE_STYLE_PROPERTIES)} properties)",
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
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
