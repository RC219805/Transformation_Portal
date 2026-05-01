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


def _style_settle_expression() -> str:
    return "new Promise((resolve) => window.setTimeout(() => resolve(true), 450))"


def _force_snapshot_state_expression() -> str:
    return r"""
(() => {
  try {
    window.localStorage.setItem('portal-theme', 'dark');
  } catch (_error) {}
  document.documentElement.classList.remove('light');
  document.documentElement.classList.add('dark');
  if (document.body) {
    document.body.classList.remove('performance-lite');
  }
  return true;
})()
"""


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
            runtime_handle = _spawn_local_backend(
                str(args.api_key),
                timeout_seconds=float(args.backend_startup_timeout_seconds),
            )
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


if __name__ == "__main__":
    raise SystemExit(main())
