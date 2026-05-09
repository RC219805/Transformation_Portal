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
4. Managed portal entry honors a validated `/portal?view=build` returnTo and keeps browser-side API key input hidden.

Run via:
    python scripts/validation/validate_frontdoor_browser_smoke.py

Environment overrides:
    TP_FRONTDOOR_BASE_URL    Front-door URL (default: http://localhost:3000)
    TP_FRONTDOOR_USERNAME    Front-door username
    TP_FRONTDOOR_PASSWORD    Front-door password
    TP_FRONTDOOR_ACCESS_EMAIL Optional access email for locally seeded fixtures
    TP_PORTAL_BROWSER_BINARY Chrome binary path override
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_portal_browser_smoke import (  # noqa: E402
    DevToolsConnection,
    LocalRuntimeHandle,
    SmokeFailure,
    _base_url,
    _default_profile_dir,
    _expect,
    _find_free_port,
    _poll,
    _request_json,
    _resolve_chrome_binary,
    _spawn_local_backend,
    _tail_text,
    _terminate_runtime,
    _wait_for_devtools,
    _wait_for_page_target,
)

DEFAULT_FRONTDOOR_BASE_URL = "http://localhost:3000"
DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_FRONTDOOR_USERS_FILE = "/tmp/tp-frontdoor-users.json"
DEFAULT_FRONTDOOR_USERNAME = "smoke-admin"
DEFAULT_FRONTDOOR_PASSWORD = "correct horse battery staple"
DEFAULT_FRONTDOOR_ROLE = "admin"
FRONTDOOR_ROOT = SCRIPT_DIR.parent.parent / "web" / "secure-landing"
FRONTDOOR_SEED_SCRIPT = FRONTDOOR_ROOT / "scripts" / "seed-frontdoor-user.mjs"
FRONTDOOR_SMOKE_DIST_DIR_PREFIX = ".next-smoke-"
FRONTDOOR_STALE_DIST_DIR_MIN_AGE_SECONDS = 5 * 60


def _request_frontdoor_health(base_url: str) -> tuple[int, dict]:
    status, body = _request_json(base_url, "/healthz")
    if not isinstance(body, dict):
        raise SmokeFailure("Front-door health probe returned an invalid JSON body.", kind="contract")
    return status, body


def _wait_for_frontdoor_ready(
    base_url: str,
    *,
    timeout_seconds: float,
    process: Optional[subprocess.Popen[str]] = None,
    log_path: Optional[Path] = None,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            break
        try:
            status, body = _request_frontdoor_health(base_url)
            if status == 200 and body.get("ok") is True:
                return body
            last_error = f"status={status} body={body}"
        except SmokeFailure as exc:
            last_error = str(exc)
        time.sleep(0.25)

    if process is not None and process.poll() is not None:
        exit_code = process.returncode
        log_tail = _tail_text(log_path) if log_path is not None else ""
        detail = f"isolated front-door exited before readiness (code {exit_code})"
        if log_tail:
            detail = f"{detail}. Recent log output:\n{log_tail}"
        raise SmokeFailure(detail, kind="runtime")

    detail = last_error or "timed out waiting for /healthz"
    raise SmokeFailure(
        f"Front-door did not become ready at {base_url}/healthz within {timeout_seconds:.1f}s ({detail}).",
        kind="runtime",
    )


def _default_frontdoor_access_email(username: str) -> str:
    return f"{username}@local.invalid"


def _seed_frontdoor_users_file(
    *,
    output_path: Path,
    username: str,
    password: str,
    access_email: str,
    role: str = DEFAULT_FRONTDOOR_ROLE,
) -> Path:
    resolved_output_path = output_path.resolve()
    resolved_access_email = str(access_email).strip() or _default_frontdoor_access_email(username)
    command = [
        "node",
        str(FRONTDOOR_SEED_SCRIPT),
        "--output",
        str(resolved_output_path),
        "--username",
        str(username),
        "--password",
        str(password),
        "--access-email",
        resolved_access_email,
        "--role",
        str(role),
        "--quiet",
    ]
    try:
        subprocess.run(
            command,
            cwd=str(FRONTDOOR_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() or str(exc)
        raise SmokeFailure(
            f"Could not seed isolated front-door credential fixture under {resolved_output_path}: {detail}",
            kind="runtime",
        ) from exc
    return resolved_output_path


def _generate_frontdoor_users_file(username: str, password: str, runtime_root: Path, *, access_email: str) -> Path:
    users_file = runtime_root / "frontdoor-users.json"
    return _seed_frontdoor_users_file(
        output_path=users_file,
        username=username,
        password=password,
        access_email=access_email,
    )


def _prune_stale_frontdoor_dist_dirs(
    active_dist_dir: Path,
    *,
    now: Optional[float] = None,
    min_age_seconds: float = FRONTDOOR_STALE_DIST_DIR_MIN_AGE_SECONDS,
) -> None:
    active_dist_dir = active_dist_dir.resolve()
    current_time = time.time() if now is None else now
    for candidate in FRONTDOOR_ROOT.glob(f"{FRONTDOOR_SMOKE_DIST_DIR_PREFIX}*"):
        if candidate.resolve() == active_dist_dir:
            continue
        if candidate.is_dir():
            try:
                candidate_mtime = candidate.stat().st_mtime
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise SmokeFailure(
                    f"Could not inspect front-door smoke distDir {candidate}: {exc}",
                    kind="runtime",
                ) from exc
            if current_time - candidate_mtime < min_age_seconds:
                continue
            try:
                shutil.rmtree(candidate)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise SmokeFailure(
                    f"Could not remove stale front-door smoke distDir {candidate}: {exc}",
                    kind="runtime",
                ) from exc


def _spawn_local_frontdoor(
    *,
    username: str,
    password: str,
    access_email: str,
    backend_base_url: str,
    backend_api_key: str,
    timeout_seconds: float,
) -> LocalRuntimeHandle:
    runtime_root = Path(
        tempfile.mkdtemp(
            prefix="tp-frontdoor-browser-runtime-",
            dir="/tmp" if os.name != "nt" and Path("/tmp").exists() else None,
        )
    )
    users_file = _generate_frontdoor_users_file(
        username,
        password,
        runtime_root,
        access_email=access_email,
    )
    session_db = runtime_root / "sessions.sqlite"
    log_path = runtime_root / "frontdoor.log"
    port = _find_free_port()
    base_url = f"http://localhost:{port}"
    dist_dir_name = f"{FRONTDOOR_SMOKE_DIST_DIR_PREFIX}{port}"
    dist_dir_path = FRONTDOOR_ROOT / dist_dir_name
    _prune_stale_frontdoor_dist_dirs(dist_dir_path)

    env = os.environ.copy()
    env.update(
        {
            "NEXT_TELEMETRY_DISABLED": "1",
            "WATCHPACK_POLLING": "true",
            "TP_FRONTDOOR_HOST": "127.0.0.1",
            "TP_FRONTDOOR_PORT": str(port),
            "TP_FRONTDOOR_USERS_FILE": str(users_file),
            "TP_FRONTDOOR_SESSION_DB": str(session_db),
            "TP_FRONTDOOR_DIST_DIR": dist_dir_name,
            "TP_FASTAPI_ORIGIN": backend_base_url,
            "TP_ALLOW_LOCAL_ACCESS_BYPASS": "1",
        }
    )
    if backend_api_key:
        env["TP_BACKEND_API_KEY"] = backend_api_key
    else:
        env.pop("TP_BACKEND_API_KEY", None)

    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(SCRIPT_DIR.parent / "setup" / "run_frontdoor_local.sh")],
            cwd=str(SCRIPT_DIR.parent.parent),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()

    handle = LocalRuntimeHandle(
        process=process,
        base_url=base_url,
        log_path=log_path,
        temp_paths=(runtime_root, dist_dir_path),
    )
    try:
        _wait_for_frontdoor_ready(
            base_url,
            timeout_seconds=timeout_seconds,
            process=process,
            log_path=log_path,
        )
    except Exception:
        _terminate_runtime(handle)
        raise
    return handle


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontdoor-base-url",
        dest="frontdoor_base_url",
        default=None,
        help=f"Front-door base URL (default: TP_FRONTDOOR_BASE_URL or {DEFAULT_FRONTDOOR_BASE_URL})",
    )
    parser.add_argument(
        "--backend-base-url",
        default="",
        help=f"Backend base URL for isolated front-door launches (default: TP_FASTAPI_ORIGIN or {DEFAULT_BACKEND_BASE_URL})",
    )
    parser.add_argument(
        "--backend-api-key",
        default="",
        help="Backend API key for isolated front-door launches (default: TP_BACKEND_API_KEY or TP_API_KEY)",
    )
    parser.add_argument(
        "--spawn-local-frontdoor",
        action="store_true",
        help="Launch an isolated local managed front-door on a free port before the browser smoke",
    )
    parser.add_argument(
        "--spawn-local-backend",
        action="store_true",
        help="Launch an isolated local backend for portal/front-door validation",
    )
    parser.add_argument(
        "--username",
        default="",
        help=(
            "Front-door username " "(default: TP_FRONTDOOR_USERNAME; falls back to smoke-admin for --spawn-local-frontdoor)"
        ),
    )
    parser.add_argument(
        "--password",
        default="",
        help=(
            "Front-door password "
            "(default: TP_FRONTDOOR_PASSWORD; falls back to the canonical smoke password for --spawn-local-frontdoor)"
        ),
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
        "--backend-startup-timeout-seconds",
        type=float,
        default=30.0,
        help="Wait budget for an auto-launched local backend to become ready (default: %(default)s)",
    )
    parser.add_argument(
        "--frontdoor-startup-timeout-seconds",
        type=float,
        default=45.0,
        help="Wait budget for an auto-launched local front-door to become ready (default: %(default)s)",
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
    raw = args.frontdoor_base_url or os.getenv("TP_FRONTDOOR_BASE_URL", DEFAULT_FRONTDOOR_BASE_URL)
    return _base_url(str(raw))


def _resolve_username(args: argparse.Namespace) -> str:
    explicit_username = str(args.username or os.getenv("TP_FRONTDOOR_USERNAME", "")).strip()
    if explicit_username:
        return explicit_username
    if args.spawn_local_frontdoor:
        return DEFAULT_FRONTDOOR_USERNAME
    return ""


def _resolve_password(args: argparse.Namespace) -> str:
    explicit_password = str(args.password or os.getenv("TP_FRONTDOOR_PASSWORD", ""))
    if explicit_password:
        return explicit_password
    if args.spawn_local_frontdoor:
        return DEFAULT_FRONTDOOR_PASSWORD
    return ""


def _resolve_access_email(username: str) -> str:
    explicit_access_email = str(os.getenv("TP_FRONTDOOR_ACCESS_EMAIL", "")).strip()
    if explicit_access_email:
        return explicit_access_email
    return _default_frontdoor_access_email(username)


def _resolve_backend_base_url(args: argparse.Namespace) -> str:
    raw = args.backend_base_url or os.getenv("TP_FASTAPI_ORIGIN", DEFAULT_BACKEND_BASE_URL)
    return _base_url(str(raw))


def _resolve_backend_api_key(args: argparse.Namespace) -> str:
    return str(args.backend_api_key or os.getenv("TP_BACKEND_API_KEY", "") or os.getenv("TP_API_KEY", "")).strip()


def _format_frontdoor_health_failure(status: int, body: dict) -> str:
    checks = body.get("checks") if isinstance(body.get("checks"), dict) else {}
    failing_checks = []
    for key in ("backend", "access_config", "user_source", "session_store", "session_scaling"):
        check = checks.get(key)
        if isinstance(check, dict) and check.get("required") and not check.get("ok"):
            reason = str(check.get("reason") or "unknown").strip() or "unknown"
            failing_checks.append(f"{key}={reason}")
    failure_summary = ", ".join(failing_checks) if failing_checks else f"status={status}"
    return f"Front-door readiness preflight failed at /healthz ({failure_summary})."


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
  const attr = (selector, name) => {
    const el = document.querySelector(selector);
    return el ? String(el.getAttribute(name) || '') : '';
  };
  const visibleById = (id) => {
    const el = document.getElementById(id);
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  };
  const hiddenById = (id) => {
    const el = document.getElementById(id);
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    const isVisible = (
      style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      rect.width > 0 &&
      rect.height > 0
    );
    return !isVisible;
  };
  return {
    title: document.title,
    readyState: document.readyState,
    pathname: window.location.pathname,
    locationSearch: window.location.search,
    homepageHeroReady: !!document.querySelector('[data-ui="homepage-hero-title"]'),
    homepageEntryRailReady: !!document.querySelector('[data-ui="homepage-entry-rail"]'),
    homepageLearnLinkReady: !!document.querySelector('[data-ui="homepage-learn-link"]'),
    homepagePrimaryCtaHref: attr('[data-ui="homepage-primary-cta"]', 'href'),
    loginTitleReady: !!document.querySelector('[data-ui="login-title"]'),
    loginEntryStateReady: !!document.querySelector('[data-ui="login-entry-state"]'),
    loginSequenceReady: !!document.querySelector('[data-ui="login-sequence"]'),
    brandAssetPresent: !!document.querySelector('.brand-asset'),
    hasHeroVideo: !!document.querySelector('.hero-video, .homepage-video'),
    loginFormPresent: !!document.querySelector('[data-ui="login-form"]'),
    usernamePresent: !!document.querySelector('input[name="username"]'),
    usernameValue: value('input[name="username"]'),
    passwordPresent: !!document.querySelector('input[name="password"]'),
    authModeBadge: text('#authModeBadge'),
    currentView: document.body ? String(document.body.dataset.consoleView || '') : '',
    apiKeySectionHidden: hiddenById('apiKeySection'),
    buildViewVisible: visibleById('build-shell'),
    overviewViewVisible: visibleById('overview-shell'),
    operateViewVisible: visibleById('jobs-shell'),
    portalAccessStateReady: !!document.querySelector('[data-ui="portal-access-state"]')
  };
})()
"""


def _frontdoor_accessibility_probe_expression() -> str:
    return r"""
(() => {
  const visible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  };
  const minTarget = (selector) => {
    const el = document.querySelector(selector);
    if (!visible(el)) return false;
    const rect = el.getBoundingClientRect();
    return rect.width >= 44 && rect.height >= 44;
  };
  const maxDisclosureDepth = (() => {
    const detailsNodes = Array.from(document.querySelectorAll('details'));
    if (!detailsNodes.length) return 0;
    const depthFor = (node) => {
      let depth = 1;
      let current = node.parentElement;
      while (current) {
        if (current.tagName === 'DETAILS') depth += 1;
        current = current.parentElement;
      }
      return depth;
    };
    return Math.max(...detailsNodes.map(depthFor));
  })();
  const focusVisibleWithStickyHeader = (() => {
    const header = document.querySelector('.site-header');
    const target = document.querySelector('[data-ui="homepage-primary-cta"]');
    if (!header || !visible(target)) return false;
    target.scrollIntoView({ block: 'center' });
    target.focus();
    const headerRect = header.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    return targetRect.top >= headerRect.bottom - 2 && targetRect.bottom <= window.innerHeight + 2;
  })();
  return {
    pathname: window.location.pathname,
    readyState: document.readyState,
    homepagePrimaryMinTarget: minTarget('[data-ui="homepage-primary-cta"]'),
    homepageSecondaryMinTarget: minTarget('[data-ui="homepage-secondary-cta"]'),
    homepageLearnMinTarget: minTarget('[data-ui="homepage-learn-link"]'),
    loginSubmitMinTarget: minTarget('[data-ui="login-submit"]'),
    loginSecondaryMinTarget: minTarget('[data-ui="login-secondary-link"]'),
    focusVisibleWithStickyHeader,
    maxDisclosureDepth,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    decorativeMotionStatic: (() => {
      const video = document.querySelector('.hero-video, .homepage-video');
      if (!video) return true;
      const style = window.getComputedStyle(video);
      if (style.display === 'none') return true;
      const hasSource = Array.from(video.querySelectorAll('source'))
        .some((source) => Boolean(source.getAttribute('src')));
      const hasPlayableMedia = Boolean(video.currentSrc) || hasSource;
      return (
        video.paused
        || video.ended
        || !hasPlayableMedia
        || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
      );
    })()
  };
})()
"""


def _frontdoor_accessibility_snapshot(result: object, *, page: str) -> dict[str, object]:
    payload = result if isinstance(result, dict) else {}
    scoped_keys = {
        "homepage": (
            "pathname",
            "readyState",
            "homepagePrimaryMinTarget",
            "homepageSecondaryMinTarget",
            "homepageLearnMinTarget",
            "focusVisibleWithStickyHeader",
            "maxDisclosureDepth",
        ),
        "login": (
            "pathname",
            "readyState",
            "loginSubmitMinTarget",
            "loginSecondaryMinTarget",
            "maxDisclosureDepth",
            "reducedMotion",
            "decorativeMotionStatic",
        ),
    }
    keys = scoped_keys.get(page)
    if keys is None:
        raise ValueError(f"Unsupported frontdoor accessibility page scope: {page}")
    return {key: payload.get(key) for key in keys}


def _navigate_expression(pathname: str) -> str:
    encoded = json.dumps(pathname)
    return f"""
(() => {{
  const url = new URL({encoded}, window.location.origin);
  window.location.assign(url.toString());
  return `${{url.pathname}}${{url.search}}`;
}})()
"""


def _populate_login_expression(username: str, password: str) -> str:
    payload = json.dumps({"username": username, "password": password})
    return f"""
(() => {{
  const cfg = {payload};
  const usernameInput = document.querySelector('input[name="username"]');
  const passwordInput = document.querySelector('input[name="password"]');
  if (!usernameInput || !passwordInput) {{
    throw new Error('login form is unavailable');
  }}
  usernameInput.value = cfg.username;
  usernameInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
  usernameInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
  passwordInput.value = cfg.password;
  passwordInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
  passwordInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
  return 'prepared';
}})()
"""


def _click_expression(selector: str) -> str:
    encoded = json.dumps(selector)
    return f"""
(() => {{
  const el = document.querySelector({encoded});
  if (!el) throw new Error('missing element for selector ' + {encoded});
  el.click();
  return true;
}})()
"""


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    base_url = _resolve_base_url(args)
    _expect(base_url, "Front-door base URL cannot be empty")
    backend_base_url = _resolve_backend_base_url(args)
    backend_api_key = _resolve_backend_api_key(args)
    username = _resolve_username(args)
    password = _resolve_password(args)
    if not username or not password:
        raise SmokeFailure(
            "Front-door username and password are required. Set TP_FRONTDOOR_USERNAME and TP_FRONTDOOR_PASSWORD or pass flags."
        )
    if args.spawn_local_frontdoor and args.spawn_local_backend and not backend_api_key:
        backend_api_key = "contract-secret"

    backend_runtime: Optional[LocalRuntimeHandle] = None
    frontdoor_runtime: Optional[LocalRuntimeHandle] = None
    chrome_process: Optional[subprocess.Popen[str]] = None
    connection: Optional[DevToolsConnection] = None
    profile_dir: Optional[Path] = None

    try:
        if args.spawn_local_backend:
            print("frontdoor-browser-smoke: launching isolated local backend", flush=True)
            backend_runtime = _spawn_local_backend(
                backend_api_key,
                timeout_seconds=args.backend_startup_timeout_seconds,
            )
            backend_base_url = backend_runtime.base_url
            print(f"frontdoor-browser-smoke: isolated backend ready at {backend_base_url}", flush=True)

        if args.spawn_local_frontdoor:
            if not backend_api_key:
                raise SmokeFailure(
                    "An isolated front-door launch requires TP_BACKEND_API_KEY or TP_API_KEY unless --spawn-local-backend is also enabled.",
                    kind="environment",
                )
            access_email = _resolve_access_email(username)
            print("frontdoor-browser-smoke: launching isolated managed front-door", flush=True)
            frontdoor_runtime = _spawn_local_frontdoor(
                username=username,
                password=password,
                access_email=access_email,
                backend_base_url=backend_base_url,
                backend_api_key=backend_api_key,
                timeout_seconds=args.frontdoor_startup_timeout_seconds,
            )
            base_url = frontdoor_runtime.base_url
            print(f"frontdoor-browser-smoke: isolated front-door ready at {base_url}", flush=True)
        else:
            print("frontdoor-browser-smoke: preflighting /healthz", flush=True)
            status, body = _request_frontdoor_health(base_url)
            if status != 200 or body.get("ok") is not True:
                raise SmokeFailure(_format_frontdoor_health_failure(status, body), kind="environment")

        profile_dir = _default_profile_dir()
        port = int(args.debugging_port or _find_free_port())
        chrome_binary = _resolve_chrome_binary(args.chrome_binary)
        _expect(Path(chrome_binary).exists(), f"Chrome binary does not exist: {chrome_binary}")

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
                and bool(value.get("homepageHeroReady"))
                and bool(value.get("homepageEntryRailReady"))
                and bool(value.get("homepageLearnLinkReady"))
                and str(value.get("homepagePrimaryCtaHref", "")) == "/login"
            ),
            timeout_seconds=args.timeout_seconds,
            description="front-door homepage to render",
        )
        _expect(str(homepage_state.get("pathname", "")) == "/", f"Front-door homepage did not load at root: {homepage_state}")
        _expect(bool(homepage_state.get("hasHeroVideo")), f"Homepage video canvas was not rendered: {homepage_state}")
        homepage_accessibility = _frontdoor_accessibility_snapshot(
            connection.evaluate(_frontdoor_accessibility_probe_expression()),
            page="homepage",
        )
        _expect(
            bool(homepage_accessibility.get("homepagePrimaryMinTarget"))
            and bool(homepage_accessibility.get("homepageSecondaryMinTarget"))
            and bool(homepage_accessibility.get("homepageLearnMinTarget")),
            f"Homepage interactive targets fell below the 44px contract: {homepage_accessibility}",
        )
        _expect(
            bool(homepage_accessibility.get("focusVisibleWithStickyHeader")),
            f"Homepage sticky header obscured focused actions: {homepage_accessibility}",
        )
        _expect(
            int(homepage_accessibility.get("maxDisclosureDepth", 0)) <= 1,
            f"Homepage disclosure depth exceeded the single-level contract: {homepage_accessibility}",
        )

        print("frontdoor-browser-smoke: opening login", flush=True)
        connection.evaluate(_navigate_expression("/login?returnTo=%2Fportal%3Fview%3Dbuild"))
        login_state = _poll(
            connection,
            _frontdoor_state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("readyState", "")) == "complete"
                and str(value.get("pathname", "")) == "/login"
                and "returnTo=%2Fportal%3Fview%3Dbuild" in str(value.get("locationSearch", ""))
                and bool(value.get("loginTitleReady"))
                and bool(value.get("loginEntryStateReady"))
                and bool(value.get("loginSequenceReady"))
                and bool(value.get("brandAssetPresent"))
                and bool(value.get("loginFormPresent"))
                and bool(value.get("usernamePresent"))
                and bool(value.get("passwordPresent"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="front-door login to render",
        )
        _expect(bool(login_state.get("hasHeroVideo")), f"Login page lost hero video shell: {login_state}")
        login_accessibility = _frontdoor_accessibility_snapshot(
            connection.evaluate(_frontdoor_accessibility_probe_expression()),
            page="login",
        )
        _expect(
            bool(login_accessibility.get("loginSubmitMinTarget")) and bool(login_accessibility.get("loginSecondaryMinTarget")),
            f"Login interactive targets fell below the 44px contract: {login_accessibility}",
        )
        _expect(
            int(login_accessibility.get("maxDisclosureDepth", 0)) <= 1,
            f"Login disclosure depth exceeded the single-level contract: {login_accessibility}",
        )

        print("frontdoor-browser-smoke: checking reduced motion", flush=True)
        connection.call(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]},
        )
        reduced_motion_state = _poll(
            connection,
            _frontdoor_accessibility_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("readyState", "")) == "complete"
                and str(value.get("pathname", "")) == "/login"
                and bool(value.get("reducedMotion"))
                and bool(value.get("decorativeMotionStatic"))
                and bool(value.get("loginSubmitMinTarget"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="front-door reduced-motion login shell",
        )
        reduced_motion_state = _frontdoor_accessibility_snapshot(reduced_motion_state, page="login")
        _expect(
            bool(reduced_motion_state.get("decorativeMotionStatic")),
            f"Reduced-motion mode left decorative front-door motion active: {reduced_motion_state}",
        )
        connection.call(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-reduced-motion", "value": "no-preference"}]},
        )

        print("frontdoor-browser-smoke: submitting operator login", flush=True)
        connection.evaluate(_populate_login_expression(username, password))
        connection.evaluate(_click_expression('[data-ui="login-submit"]'))
        portal_state = _poll(
            connection,
            _frontdoor_state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("readyState", "")) == "complete"
                and str(value.get("pathname", "")) == "/portal"
                and str(value.get("currentView", "")) == "build"
                and "view=build" in str(value.get("locationSearch", ""))
                and bool(value.get("apiKeySectionHidden"))
                and bool(value.get("buildViewVisible"))
                and not bool(value.get("overviewViewVisible"))
                and not bool(value.get("operateViewVisible"))
                and bool(value.get("portalAccessStateReady"))
                and str(value.get("authModeBadge", "")).lower() == "managed"
            ),
            timeout_seconds=args.timeout_seconds,
            description="managed portal entry after login",
        )
        _expect(
            str(portal_state.get("authModeBadge", "")).lower() == "managed",
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
        if frontdoor_runtime is not None:
            _terminate_runtime(frontdoor_runtime)
        if backend_runtime is not None:
            _terminate_runtime(backend_runtime)
        if profile_dir is not None and not args.keep_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"frontdoor-browser-smoke: failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
