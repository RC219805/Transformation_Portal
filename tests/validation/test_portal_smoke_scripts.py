"""Unit tests for portal smoke validation scripts."""

from __future__ import annotations

import importlib.util
import sys
import urllib.error
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTAL_BROWSER_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_portal_browser_smoke.py"
FRONTDOOR_BROWSER_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "validate_frontdoor_browser_smoke.py"
ORCHESTRATOR_HTTP_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_orchestrator_http_smoke.py"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_portal_browser_parse_args_does_not_probe_chrome_for_explicit_override(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke")

    def _boom() -> str:
        raise AssertionError("_default_chrome_binary should not be called while parsing args")

    monkeypatch.setattr(module, "_default_chrome_binary", _boom)

    args = module._parse_args(["--chrome-binary", "/custom/chrome"])

    assert args.chrome_binary == "/custom/chrome"


def test_portal_browser_explicit_output_dirs_are_not_auto_cleaned(tmp_path: Path):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_cleanup")
    explicit_output_dir = tmp_path / "browser-output"

    resolved_output_dir, output_dir_is_temp = module._resolve_output_dir(str(explicit_output_dir))

    assert resolved_output_dir == explicit_output_dir.resolve()
    assert output_dir_is_temp is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=output_dir_is_temp) is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=True) is True


def test_orchestrator_http_request_json_wraps_transport_failures(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(ORCHESTRATOR_HTTP_SCRIPT_PATH, "tests_validate_orchestrator_http_smoke")

    def _raise_url_error(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(module.urllib.request, "urlopen", _raise_url_error)

    with pytest.raises(module.SmokeFailure, match="GET /ready request failed: connection refused"):
        module._request_json("http://127.0.0.1:8000", "/ready")


def test_orchestrator_http_explicit_output_dirs_are_not_auto_cleaned(tmp_path: Path):
    module = _load_module(ORCHESTRATOR_HTTP_SCRIPT_PATH, "tests_validate_orchestrator_http_smoke_cleanup")
    explicit_output_dir = tmp_path / "http-output"

    resolved_output_dir, output_dir_is_temp = module._resolve_output_dir(str(explicit_output_dir))

    assert resolved_output_dir == explicit_output_dir.resolve()
    assert output_dir_is_temp is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=output_dir_is_temp) is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=True) is True


def test_frontdoor_browser_parse_args_does_not_probe_chrome_for_explicit_override(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke")

    def _boom() -> str:
        raise AssertionError("_resolve_chrome_binary should not be called while parsing args")

    monkeypatch.setattr(module, "_resolve_chrome_binary", _boom)

    args = module._parse_args(
        [
            "--chrome-binary",
            "/custom/chrome",
            "--frontdoor-base-url",
            "http://127.0.0.1:3000",
            "--username",
            "admin",
            "--password",
            "secret",
        ]
    )

    assert args.chrome_binary == "/custom/chrome"
    assert args.frontdoor_base_url == "http://127.0.0.1:3000"
    assert args.username == "admin"
    assert args.password == "secret"
