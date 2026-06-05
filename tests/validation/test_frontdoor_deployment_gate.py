"""Unit tests for the frontdoor shared-deployment gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_frontdoor_deployment_gate.py"


def _load_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _probe(
    module,
    *,
    path: str,
    status: int | None,
    headers: dict[str, str] | None = None,
    body: str = "",
    location: str = "",
    transport_error: str | None = None,
    transport_detail: str | None = None,
):
    return module.ProbeResponse(
        url=f"https://example.test{path}",
        path=path,
        status=status,
        headers={key.lower(): value for key, value in (headers or {}).items()},
        body=body,
        location=location,
        transport_error=transport_error,
        transport_detail=transport_detail,
    )


def _fake_request_factory(module, mapping: dict[tuple[str, str], object]):
    def _fake_request(*, base_url: str, path: str, accept: str, timeout_seconds: float, user_agent: str):
        del accept, timeout_seconds, user_agent
        response = mapping[(base_url, path)]
        if callable(response):
            return response()
        return response

    return _fake_request


def test_normalize_base_url_preserves_ipv6_literals():
    module = _load_module("tests_frontdoor_gate_ipv6_url")

    assert (
        module._normalize_base_url(
            "https://[::1]:8443",
            require_https=True,
            label="frontdoor URL",
        )
        == "https://[::1]:8443"
    )


@pytest.mark.parametrize(
    ("module_name", "value"),
    [
        ("tests_frontdoor_gate_team_domain_path_host_only", "team.cloudflareaccess.com/cdn-cgi/access/login/app.example.com"),
        ("tests_frontdoor_gate_team_domain_path_https", "https://team.cloudflareaccess.com/cdn-cgi/access/login"),
        ("tests_frontdoor_gate_team_domain_query", "https://team.cloudflareaccess.com?next=/login"),
        ("tests_frontdoor_gate_team_domain_fragment", "https://team.cloudflareaccess.com#fragment"),
    ],
)
def test_normalize_access_team_domain_rejects_non_base_url(module_name: str, value: str):
    module = _load_module(module_name)

    with pytest.raises(ValueError):
        module._normalize_access_team_domain(value)


def test_surface_verdict_render_escapes_control_characters():
    module = _load_module("tests_frontdoor_gate_render_detail")
    verdict = module.SurfaceVerdict(
        surface="frontdoor",
        verdict="FAIL",
        code="frontdoor_unclassified",
        detail='line one\nline two\rline three\t"quoted"',
    )

    rendered = verdict.render()
    assert "\n" not in rendered
    assert "\r" not in rendered
    assert "\\n" in rendered
    assert "\\r" in rendered
    assert "\\t" in rendered


def test_cloudflare_access_redirect_to_configured_team_domain_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_cf_redirect")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module,
            path="/",
            status=302,
            headers={"Location": "https://team.cloudflareaccess.com/cdn-cgi/access/login/portal.example.com"},
            location="https://team.cloudflareaccess.com/cdn-cgi/access/login/portal.example.com",
        ),
        ("https://portal.example.com", "/login"): _probe(
            module,
            path="/login",
            status=302,
            headers={"Location": "https://team.cloudflareaccess.com/cdn-cgi/access/login/portal.example.com"},
            location="https://team.cloudflareaccess.com/cdn-cgi/access/login/portal.example.com",
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "staging",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "surface=frontdoor verdict=PASS code=cf_access_redirect" in output
    assert "overall=PASS" in output


def test_cloudflare_access_interstitial_passes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_cf_interstitial")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module,
            path="/",
            status=403,
            headers={"CF-Ray": "abc", "Server": "cloudflare"},
            body="Cloudflare Access required. Continue at /cdn-cgi/access/login",
        ),
        ("https://portal.example.com", "/login"): _probe(
            module,
            path="/login",
            status=403,
            headers={"Server": "cloudflare"},
            body="cloudflareaccess.com challenge",
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "staging",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "surface=frontdoor verdict=PASS code=cf_access_interstitial" in output
    assert "overall=PASS" in output


def test_vercel_auth_protected_deployment_url_passes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_vercel_protected")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=302, location="https://team.cloudflareaccess.com/cdn-cgi/access/login"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=302, location="https://team.cloudflareaccess.com/cdn-cgi/access/login"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(
            module,
            path="/",
            status=401,
            headers={"X-Vercel-Id": "iad1::abc", "Server": "Vercel"},
        ),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module,
            path="/login",
            status=302,
            headers={"X-Vercel-Id": "iad1::def", "Location": "https://vercel.com/sso-login"},
            location="https://vercel.com/sso-login",
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "staging",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "surface=vercel_deployment verdict=PASS code=vercel_protected_status" in output
    assert "surface=fastapi_public_probe verdict=PASS code=fastapi_non_public_confirmed" in output
    assert "overall=PASS" in output


def test_frontdoor_serving_real_homepage_shell_fails(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_frontdoor_homepage_exposed")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module,
            path="/",
            status=200,
            body='<title>Dynamic Neural Access</title><body data-ui="homepage-shell"><h1 data-ui="homepage-hero-title"></h1>',
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=frontdoor verdict=FAIL code=frontdoor_app_shell_exposed" in output


def test_frontdoor_serving_real_login_shell_fails(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_frontdoor_login_exposed")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module,
            path="/login",
            status=200,
            body=(
                "<title>Dynamic Neural Access | Transformation Portal</title>"
                '<main data-ui="login-shell"><form data-ui="login-form"></form>'
                "Transformation Portal operator console"
            ),
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=frontdoor verdict=FAIL code=frontdoor_app_shell_exposed" in output


def test_vercel_deployment_serving_real_homepage_shell_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_vercel_homepage_exposed")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(
            module,
            path="/",
            status=200,
            headers={"Server": "Vercel"},
            body='<title>Dynamic Neural Access</title><body data-ui="homepage-shell"><div data-ui="homepage-hero-title"></div>',
        ),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::abc"}
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=vercel_deployment verdict=FAIL code=vercel_app_shell_exposed" in output


def test_vercel_deployment_serving_real_login_shell_fails(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_vercel_login_exposed")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module,
            path="/login",
            status=200,
            headers={"Server": "Vercel"},
            body=(
                "<title>Dynamic Neural Access | Transformation Portal</title>"
                '<section data-ui="login-shell"></section>'
                '<form data-ui="login-form"></form>'
            ),
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=vercel_deployment verdict=FAIL code=vercel_app_shell_exposed" in output


def test_public_fastapi_ready_returning_healthy_json_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_fastapi_ready_exposed")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
        ("https://api.example.com", "/ready"): _probe(
            module, path="/ready", status=200, body='{"ok": true, "time": "2026-04-09T00:00:00Z"}'
        ),
        ("https://api.example.com", "/healthz"): _probe(module, path="/healthz", status=403, body=""),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--fastapi-public-url",
            "https://api.example.com",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=fastapi_public_probe verdict=FAIL code=fastapi_public_health_exposed" in output


def test_public_fastapi_healthz_returning_healthy_json_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_fastapi_healthz_exposed")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
        ("https://api.example.com", "/ready"): _probe(module, path="/ready", status=403, body=""),
        ("https://api.example.com", "/healthz"): _probe(
            module, path="/healthz", status=200, body='{"ok": true, "time": "2026-04-09T00:00:00Z"}'
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--fastapi-public-url",
            "https://api.example.com",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=fastapi_public_probe verdict=FAIL code=fastapi_public_health_exposed" in output


def test_explicit_fastapi_attestation_missing_fails(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_fastapi_attestation_missing")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=fastapi_public_probe verdict=FAIL code=fastapi_attestation_missing" in output


def test_ambiguous_vercel_404_without_vercel_markers_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_vercel_404_ambiguous")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=404, body="not found"),
        ("https://portal-preview.vercel.app", "/login"): _probe(module, path="/login", status=404, body="not found"),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "staging",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=vercel_deployment verdict=FAIL code=vercel_unclassified" in output


def test_ambiguous_frontdoor_html_without_cloudflare_markers_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_frontdoor_ambiguous")
    mapping = {
        ("https://portal.example.com", "/"): _probe(module, path="/", status=403, body="<html>please sign in</html>"),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, body="<html>please sign in</html>"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "staging",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=frontdoor verdict=FAIL code=frontdoor_unclassified" in output


def test_blocked_fastapi_probe_outcomes_pass(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_fastapi_blocked")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
        ("https://api.example.com", "/ready"): _probe(
            module, path="/ready", status=None, transport_error="dns_failure", transport_detail="host not found"
        ),
        ("https://api.example.com", "/healthz"): _probe(module, path="/healthz", status=404, body=""),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--fastapi-public-url",
            "https://api.example.com",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "surface=fastapi_public_probe verdict=PASS code=fastapi_probe_blocked" in output


def test_ambiguous_fastapi_2xx_non_health_response_fails(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_fastapi_ambiguous_2xx")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal-preview.vercel.app", "/"): _probe(module, path="/", status=401, headers={"X-Vercel-Id": "iad1::abc"}),
        ("https://portal-preview.vercel.app", "/login"): _probe(
            module, path="/login", status=401, headers={"X-Vercel-Id": "iad1::def"}
        ),
        ("https://api.example.com", "/ready"): _probe(module, path="/ready", status=200, body='{"status": "ok"}'),
        ("https://api.example.com", "/healthz"): _probe(module, path="/healthz", status=404, body=""),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--vercel-deployment-url",
            "https://portal-preview.vercel.app",
            "--fastapi-public-url",
            "https://api.example.com",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=fastapi_public_probe verdict=FAIL code=fastapi_probe_unclassified" in output


def test_cloudflare_worker_deployment_url_passes_when_access_protected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_worker_protected")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://transformationportal.example.workers.dev", "/"): _probe(
            module,
            path="/",
            status=302,
            headers={"Location": "https://team.cloudflareaccess.com/cdn-cgi/access/login/worker"},
            location="https://team.cloudflareaccess.com/cdn-cgi/access/login/worker",
        ),
        ("https://transformationportal.example.workers.dev", "/login"): _probe(
            module,
            path="/login",
            status=302,
            headers={"Location": "https://team.cloudflareaccess.com/cdn-cgi/access/login/worker"},
            location="https://team.cloudflareaccess.com/cdn-cgi/access/login/worker",
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--deployment-target",
            "cloudflare-worker",
            "--deployment-url",
            "https://transformationportal.example.workers.dev",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "surface=worker_deployment verdict=PASS code=cf_access_redirect" in output
    assert "overall=PASS" in output


def test_cloudflare_worker_deployment_serving_real_shell_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_module("tests_frontdoor_gate_worker_shell_exposed")
    mapping = {
        ("https://portal.example.com", "/"): _probe(
            module, path="/", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://portal.example.com", "/login"): _probe(
            module, path="/login", status=403, headers={"Server": "cloudflare"}, body="cloudflare access"
        ),
        ("https://transformationportal.example.workers.dev", "/"): _probe(
            module,
            path="/",
            status=200,
            headers={"Server": "cloudflare"},
            body='<title>Dynamic Neural Access</title><body data-ui="homepage-shell"><h1 data-ui="homepage-hero-title"></h1>',
        ),
        ("https://transformationportal.example.workers.dev", "/login"): _probe(
            module,
            path="/login",
            status=302,
            headers={"Location": "https://team.cloudflareaccess.com/cdn-cgi/access/login/worker"},
            location="https://team.cloudflareaccess.com/cdn-cgi/access/login/worker",
        ),
    }
    monkeypatch.setattr(module, "_perform_request", _fake_request_factory(module, mapping))

    exit_code = module.main(
        [
            "--environment",
            "production",
            "--frontdoor-url",
            "https://portal.example.com",
            "--cf-access-team-domain",
            "https://team.cloudflareaccess.com",
            "--deployment-target",
            "cloudflare_worker",
            "--deployment-url",
            "https://transformationportal.example.workers.dev",
            "--confirm-fastapi-non-public",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "surface=worker_deployment verdict=FAIL code=frontdoor_app_shell_exposed" in output


def test_generic_deployment_url_requires_explicit_target(capsys: pytest.CaptureFixture[str]):
    module = _load_module("tests_frontdoor_gate_generic_deployment_target_required")

    with pytest.raises(SystemExit):
        module._parse_args(
            [
                "--environment",
                "staging",
                "--frontdoor-url",
                "https://portal.example.com",
                "--cf-access-team-domain",
                "https://team.cloudflareaccess.com",
                "--deployment-url",
                "https://transformationportal.example.workers.dev",
                "--confirm-fastapi-non-public",
            ]
        )

    assert "--deployment-target is required" in capsys.readouterr().err
