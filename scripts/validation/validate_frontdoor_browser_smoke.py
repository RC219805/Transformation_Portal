#!/usr/bin/env python3
"""
Browser smoke validation for the managed secure front door.

This script launches a disposable Chrome instance with the DevTools protocol
enabled, exercises the public homepage and login in a real browser, and then
verifies portal entry after authentication against a running front-door app.

Coverage:
1. Homepage loads and renders the public DNA hero.
2. Login loads with the operator form and front-door video shell.
3. Username/password authentication succeeds.
4. Managed portal entry lands on `/portal` with browser-side API key input hidden.

Run via:
    python scripts/validation/validate_frontdoor_browser_smoke.py

Environment overrides:
    TP_FRONTDOOR_BASE_URL    Front-door URL (default: http://127.0.0.1:3000)
    TP_FRONTDOOR_USERNAME    Front-door username
    TP_FRONTDOOR_PASSWORD    Front-door password
    TP_PORTAL_BROWSER_BINARY Chrome binary path override
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_portal_browser_smoke import (  # noqa: E402
    DevToolsConnection,
    SmokeFailure,
    _default_profile_dir,
    _expect,
    _find_free_port,
    _poll,
    _resolve_chrome_binary,
    _wait_for_devtools,
    _wait_for_page_target,
)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontdoor-base-url",
        dest="frontdoor_base_url",
        default=None,
        help="Front-door base URL (default: TP_FRONTDOOR_BASE_URL or http://127.0.0.1:3000)",
    )
    parser.add_argument(
        "--username",
        default="",
        help="Front-door username (default: TP_FRONTDOOR_USERNAME)",
    )
    parser.add_argument(
        "--password",
        default="",
        help="Front-door password (default: TP_FRONTDOOR_PASSWORD)",
    )
    parser.add_argument(
        "--chrome-binary",
        default="",
        help="Chrome executable path (default: TP_PORTAL_BROWSER_BINARY or auto-detect)",
    )
    parser.add_argument(
        "--debugging-port",
        type=int,
        default=0,
        help="Chrome remote debugging port (default: auto-select free port)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Overall wait budget for front-door transitions (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-profile",
        action="store_true",
        help="Preserve the temporary Chrome profile for debugging",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_base_url(args: argparse.Namespace) -> str:
    import os

    raw = args.frontdoor_base_url or os.getenv("TP_FRONTDOOR_BASE_URL", "http://127.0.0.1:3000")
    return str(raw).strip().rstrip("/")


def _resolve_username(args: argparse.Namespace) -> str:
    import os

    return str(args.username or os.getenv("TP_FRONTDOOR_USERNAME", "")).strip()


def _resolve_password(args: argparse.Namespace) -> str:
    import os

    return str(args.password or os.getenv("TP_FRONTDOOR_PASSWORD", ""))


def _frontdoor_state_probe_expression() -> str:
    return r"""
(() => {
  const text = (selector) => {
    const el = document.querySelector(selector);
    return el ? String(el.textContent || '').trim() : '';
  };
  const value = (selector) => {
    const el = document.querySelector(selector);
    return el ? String(el.value || '') : '';
  };
  const hidden = (id) => {
    const el = document.getElementById(id);
    return !!(el && el.classList.contains('hidden'));
  };
  return {
    title: document.title,
    readyState: document.readyState,
    pathname: window.location.pathname,
    homepageHeading: text('main h1'),
    loginHeading: text('.card h1'),
    hasHeroVideo: !!document.querySelector('.hero-video'),
    usernameValue: value('input[name="username"]'),
    passwordPresent: !!document.querySelector('input[name="password"]'),
    authModeBadge: text('#authModeBadge'),
    currentView: document.body ? String(document.body.dataset.consoleView || '') : '',
    apiKeySectionHidden: hidden('apiKeySection')
  };
})()
"""


def _navigate_expression(pathname: str) -> str:
    encoded = json.dumps(pathname)
    return f"""
(() => {{
  const url = new URL(window.location.href);
  url.pathname = {encoded};
  url.search = '';
  window.location.assign(url.toString());
  return url.toString();
}})()
"""


def _submit_login_expression(username: str, password: str) -> str:
    payload = json.dumps({"username": username, "password": password})
    return f"""
(() => {{
  const cfg = {payload};
  const form = document.querySelector('form[action="/login"]');
  const usernameInput = document.querySelector('input[name="username"]');
  const passwordInput = document.querySelector('input[name="password"]');
  if (!form || !usernameInput || !passwordInput) {{
    throw new Error('login form is unavailable');
  }}
  usernameInput.value = cfg.username;
  usernameInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
  usernameInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
  passwordInput.value = cfg.password;
  passwordInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
  passwordInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
  form.requestSubmit();
  return true;
}})()
"""


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    base_url = _resolve_base_url(args)
    _expect(base_url, "Front-door base URL cannot be empty")
    username = _resolve_username(args)
    password = _resolve_password(args)
    if not username or not password:
        raise SmokeFailure(
            "Front-door username and password are required. Set TP_FRONTDOOR_USERNAME and TP_FRONTDOOR_PASSWORD or pass flags."
        )

    profile_dir = _default_profile_dir()
    port = int(args.debugging_port or _find_free_port())
    chrome_binary = _resolve_chrome_binary(args.chrome_binary)
    _expect(Path(chrome_binary).exists(), f"Chrome binary does not exist: {chrome_binary}")

    chrome_process: Optional[subprocess.Popen[str]] = None
    connection: Optional[DevToolsConnection] = None

    try:
        print("frontdoor-browser-smoke: launching chrome", flush=True)
        command = [
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
            "--disable-popup-blocking",
            "about:blank",
        ]
        chrome_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("frontdoor-browser-smoke: connecting devtools", flush=True)
        _wait_for_devtools(port)
        target = _wait_for_page_target(port)
        websocket_url = str(target.get("webSocketDebuggerUrl") or "").strip()
        _expect(websocket_url.startswith("ws://"), f"Invalid DevTools websocket URL: {websocket_url!r}")

        connection = DevToolsConnection(websocket_url)
        connection.call("Page.enable")
        connection.call("Runtime.enable")
        connection.call("Page.navigate", {"url": base_url}, timeout_seconds=20.0)

        print("frontdoor-browser-smoke: waiting for homepage", flush=True)
        homepage_state = _poll(
            connection,
            _frontdoor_state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and value.get("readyState") == "complete"
                and "Certified Premium Media for the AI Era" in str(value.get("homepageHeading", ""))
            ),
            timeout_seconds=args.timeout_seconds,
            description="front-door homepage to render",
        )
        _expect(str(homepage_state.get("pathname", "")) == "/", f"Front-door homepage did not load at root: {homepage_state}")
        _expect(bool(homepage_state.get("hasHeroVideo")), f"Homepage video canvas was not rendered: {homepage_state}")

        print("frontdoor-browser-smoke: opening login", flush=True)
        connection.evaluate(_navigate_expression("/login"))
        login_state = _poll(
            connection,
            _frontdoor_state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pathname", "")) == "/login"
                and "Operator Login" in str(value.get("loginHeading", ""))
                and bool(value.get("passwordPresent"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="front-door login to render",
        )
        _expect(bool(login_state.get("hasHeroVideo")), f"Login page lost hero video shell: {login_state}")

        print("frontdoor-browser-smoke: submitting operator login", flush=True)
        connection.evaluate(_submit_login_expression(username, password))
        portal_state = _poll(
            connection,
            _frontdoor_state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pathname", "")) == "/portal"
                and str(value.get("currentView", "")) == "overview"
                and bool(value.get("apiKeySectionHidden"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="managed portal entry after login",
        )
        _expect(
            str(portal_state.get("authModeBadge", "")).lower() in {"managed", ""},
            f"Managed portal did not hide browser API key workflow cleanly: {portal_state}",
        )

        print("frontdoor-browser-smoke: ok")
        print(f"base_url: {base_url}")
        print(f"portal_path: {portal_state.get('pathname')}")
        print(f"view: {portal_state.get('currentView')}")
        return 0
    finally:
        if connection is not None:
            connection.close()
        if chrome_process is not None:
            try:
                chrome_process.terminate()
                chrome_process.wait(timeout=5)
            except Exception:
                try:
                    chrome_process.kill()
                except Exception:
                    pass
        if not args.keep_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"frontdoor-browser-smoke: failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
