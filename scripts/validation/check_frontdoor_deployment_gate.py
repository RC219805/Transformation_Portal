#!/usr/bin/env python3
"""Manual predeploy gate for shared frontdoor rollouts."""

from __future__ import annotations

import argparse
import errno
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_USER_AGENT = "TransformationPortalFrontdoorDeploymentGate/1.0"
MAX_BODY_BYTES = 64 * 1024
HTML_ACCEPT = "text/html,application/xhtml+xml"
JSON_ACCEPT = "application/json"
DEPLOYMENT_TARGETS = ("cloudflare-worker", "vercel")
FRONTDOOR_PATHS = ("/", "/login")
FASTAPI_PATHS = ("/ready", "/healthz")

HOMEPAGE_MARKERS = (
    "<title>Dynamic Neural Access</title>",
    'data-ui="homepage-shell"',
    'data-ui="homepage-hero-title"',
)
LOGIN_MARKERS = (
    "<title>Dynamic Neural Access | Transformation Portal</title>",
    'data-ui="login-shell"',
    'data-ui="login-form"',
    "Transformation Portal operator console",
)
APP_SHELL_MARKERS = {
    "/": HOMEPAGE_MARKERS,
    "/login": LOGIN_MARKERS,
}
CLOUDFLARE_ACCESS_INDICATORS = (
    "cloudflareaccess.com",
    "cdn-cgi/access",
    "cloudflare access",
)
VERCEL_AUTH_PATH_MARKERS = (
    "/_vercel",
    "deployment-protection",
    "password-protection",
    "/auth",
    "sso",
    "protected",
    "protection",
)
BLOCKED_TRANSPORT_ERRORS = {"timeout", "dns_failure", "connection_refused"}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from automatically following redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


@dataclass(frozen=True)
class ProbeResponse:
    """Snapshot of a single HTTP probe."""

    url: str
    path: str
    status: int | None
    headers: Dict[str, str]
    body: str
    location: str
    transport_error: str | None = None
    transport_detail: str | None = None


@dataclass(frozen=True)
class PathOutcome:
    """Classification outcome for a single probed path."""

    ok: bool
    code: str
    detail: str


@dataclass(frozen=True)
class SurfaceVerdict:
    """User-facing verdict for one gate surface."""

    surface: str
    verdict: str
    code: str
    detail: str

    def render(self) -> str:
        return f"surface={self.surface} verdict={self.verdict} code={self.code} detail={json.dumps(self.detail)}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("staging", "production"), required=True)
    parser.add_argument("--frontdoor-url", required=True)
    parser.add_argument("--cf-access-team-domain", required=True)
    parser.add_argument(
        "--deployment-target",
        choices=DEPLOYMENT_TARGETS + ("cloudflare_worker",),
        help="Deployment surface to validate. Use cloudflare-worker for the current Worker frontdoor rollout.",
    )
    parser.add_argument("--deployment-url", help="Cloudflare Worker or Vercel deployment base URL.")
    parser.add_argument(
        "--vercel-deployment-url",
        help="Deprecated alias for --deployment-url with --deployment-target=vercel.",
    )
    parser.add_argument("--fastapi-public-url")
    parser.add_argument("--confirm-fastapi-non-public", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def _normalize_base_url(value: str, *, require_https: bool, label: str) -> str:
    trimmed = str(value or "").strip()
    if not trimmed:
        raise ValueError(f"{label} cannot be empty.")
    parsed = urllib.parse.urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute URL.")
    if require_https and parsed.scheme.lower() != "https":
        raise ValueError(f"{label} must use https.")
    if not require_https and parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"{label} must use http or https.")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not include userinfo.")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not include a query string or fragment.")
    if parsed.path not in ("", "/"):
        raise ValueError(f"{label} must be a base URL with an empty path or '/'.")
    scheme = parsed.scheme.lower()
    return f"{scheme}://{_normalized_authority(parsed, label=label)}"


def _normalized_authority(parsed: urllib.parse.ParseResult, *, label: str) -> str:
    host = parsed.hostname
    if not host:
        raise ValueError(f"{label} must include a host.")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} contains an invalid port.") from exc
    host_display = f"[{host}]" if ":" in host else host
    port = f":{parsed_port}" if parsed_port is not None else ""
    return f"{host_display}{port}"


def _normalize_access_team_domain(value: str) -> str:
    trimmed = str(value or "").strip()
    if not trimmed:
        raise ValueError("Cloudflare Access team domain cannot be empty.")
    normalized = trimmed if "://" in trimmed else f"https://{trimmed}"
    return _normalize_base_url(
        normalized,
        require_https=True,
        label="Cloudflare Access team domain",
    )


def _normalized_headers(headers: Iterable[tuple[str, str]]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in headers:
        lower_key = str(key).lower()
        if lower_key in normalized:
            normalized[lower_key] = f"{normalized[lower_key]}, {value}"
        else:
            normalized[lower_key] = value
    return normalized


def _read_body(stream) -> str:
    raw = stream.read(MAX_BODY_BYTES)
    return raw.decode("utf-8", errors="replace")


def _classify_transport_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, TimeoutError):
        return "timeout", "timed out"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
    else:
        reason = exc
    if isinstance(reason, socket.timeout):
        return "timeout", "timed out"
    if isinstance(reason, socket.gaierror):
        return "dns_failure", str(reason)
    if isinstance(reason, ConnectionRefusedError):
        return "connection_refused", str(reason)
    if isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
        return "connection_refused", str(reason)
    return "transport_error", str(reason)


def _perform_request(
    *,
    base_url: str,
    path: str,
    accept: str,
    timeout_seconds: float,
    user_agent: str,
) -> ProbeResponse:
    url = f"{base_url}{path}"
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": user_agent},
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler)
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            headers = _normalized_headers(response.headers.items())
            return ProbeResponse(
                url=url,
                path=path,
                status=response.status,
                headers=headers,
                body=_read_body(response),
                location=headers.get("location", ""),
            )
    except urllib.error.HTTPError as exc:
        headers = _normalized_headers(exc.headers.items()) if exc.headers is not None else {}
        return ProbeResponse(
            url=url,
            path=path,
            status=exc.code,
            headers=headers,
            body=_read_body(exc),
            location=headers.get("location", ""),
        )
    except (TimeoutError, urllib.error.URLError) as exc:
        transport_error, transport_detail = _classify_transport_error(exc)
        return ProbeResponse(
            url=url,
            path=path,
            status=None,
            headers={},
            body="",
            location="",
            transport_error=transport_error,
            transport_detail=transport_detail,
        )


def _matched_app_shell_markers(path: str, body: str) -> list[str]:
    haystack = body.lower()
    matched: list[str] = []
    for marker in APP_SHELL_MARKERS[path]:
        if marker.lower() in haystack:
            matched.append(marker)
    return matched


def _cloudflare_platform_markers(probe: ProbeResponse) -> list[str]:
    matched: list[str] = []
    if "cf-ray" in probe.headers:
        matched.append("cf-ray")
    if "cloudflare" in probe.headers.get("server", "").lower():
        matched.append("server=cloudflare")
    return matched


def _cloudflare_access_indicators(probe: ProbeResponse) -> list[str]:
    haystack = probe.body.lower()
    return [marker for marker in CLOUDFLARE_ACCESS_INDICATORS if marker in haystack]


def _vercel_platform_markers(probe: ProbeResponse) -> list[str]:
    matched: list[str] = []
    if "x-vercel-id" in probe.headers:
        matched.append("x-vercel-id")
    if "x-vercel-error" in probe.headers:
        matched.append("x-vercel-error")
    if "vercel" in probe.headers.get("server", "").lower():
        matched.append("server=vercel")
    return matched


def _location_host_and_path(probe: ProbeResponse) -> tuple[str, str]:
    if not probe.location:
        return "", ""
    resolved = urllib.parse.urljoin(probe.url, probe.location)
    parsed = urllib.parse.urlparse(resolved)
    return parsed.hostname or "", parsed.path or "/"


def _is_redirect_status(status: int | None) -> bool:
    return status in {301, 302, 303, 307, 308}


def _format_marker_list(markers: Sequence[str]) -> str:
    return ",".join(markers) if markers else "none"


def _classify_frontdoor_probe(probe: ProbeResponse, *, access_team_domain: str) -> PathOutcome:
    shell_markers = _matched_app_shell_markers(probe.path, probe.body)
    if len(shell_markers) >= 2:
        return PathOutcome(
            ok=False,
            code="frontdoor_app_shell_exposed",
            detail=(f"path={probe.path} status={probe.status or 0} " f"app_markers={_format_marker_list(shell_markers)}"),
        )

    if probe.transport_error:
        return PathOutcome(
            ok=False,
            code="frontdoor_unclassified",
            detail=f"path={probe.path} transport_error={probe.transport_error} detail={probe.transport_detail}",
        )

    location_host, location_path = _location_host_and_path(probe)
    if _is_redirect_status(probe.status) and location_host == urllib.parse.urlparse(access_team_domain).hostname:
        return PathOutcome(
            ok=True,
            code="cf_access_redirect",
            detail=(
                f"path={probe.path} status={probe.status} "
                f"redirect_host={location_host} redirect_path={location_path or '/'}"
            ),
        )

    cloudflare_markers = _cloudflare_platform_markers(probe)
    access_indicators = _cloudflare_access_indicators(probe)
    if cloudflare_markers and access_indicators:
        return PathOutcome(
            ok=True,
            code="cf_access_interstitial",
            detail=(
                f"path={probe.path} status={probe.status or 0} "
                f"cf_markers={_format_marker_list(cloudflare_markers)} "
                f"access_indicators={_format_marker_list(access_indicators)}"
            ),
        )

    return PathOutcome(
        ok=False,
        code="frontdoor_unclassified",
        detail=(
            f"path={probe.path} status={probe.status or 0} "
            f"redirect_host={location_host or 'none'} "
            f"cf_markers={_format_marker_list(cloudflare_markers)} "
            f"access_indicators={_format_marker_list(access_indicators)}"
        ),
    )


def _redirect_indicates_vercel_auth(probe: ProbeResponse) -> bool:
    location_host, location_path = _location_host_and_path(probe)
    if location_host.endswith(".vercel.com"):
        return True
    lowered_path = location_path.lower()
    return any(marker in lowered_path for marker in VERCEL_AUTH_PATH_MARKERS)


def _classify_vercel_probe(probe: ProbeResponse) -> PathOutcome:
    shell_markers = _matched_app_shell_markers(probe.path, probe.body)
    if len(shell_markers) >= 2:
        return PathOutcome(
            ok=False,
            code="vercel_app_shell_exposed",
            detail=(f"path={probe.path} status={probe.status or 0} " f"app_markers={_format_marker_list(shell_markers)}"),
        )

    if probe.transport_error:
        return PathOutcome(
            ok=False,
            code="vercel_unclassified",
            detail=f"path={probe.path} transport_error={probe.transport_error} detail={probe.transport_detail}",
        )

    vercel_markers = _vercel_platform_markers(probe)
    location_host, location_path = _location_host_and_path(probe)
    if _is_redirect_status(probe.status) and vercel_markers and _redirect_indicates_vercel_auth(probe):
        return PathOutcome(
            ok=True,
            code="vercel_auth_redirect",
            detail=(
                f"path={probe.path} status={probe.status} "
                f"redirect_host={location_host or 'none'} redirect_path={location_path or '/'} "
                f"vercel_markers={_format_marker_list(vercel_markers)}"
            ),
        )

    if probe.status in {401, 403, 404} and vercel_markers:
        return PathOutcome(
            ok=True,
            code="vercel_protected_status",
            detail=(f"path={probe.path} status={probe.status} " f"vercel_markers={_format_marker_list(vercel_markers)}"),
        )

    return PathOutcome(
        ok=False,
        code="vercel_unclassified",
        detail=(
            f"path={probe.path} status={probe.status or 0} "
            f"redirect_host={location_host or 'none'} redirect_path={location_path or 'none'} "
            f"vercel_markers={_format_marker_list(vercel_markers)}"
        ),
    )


def _surface_verdict_from_path_outcomes(
    *,
    surface: str,
    outcomes: Sequence[PathOutcome],
    preferred_success_code: str,
    fallback_success_code: str,
) -> SurfaceVerdict:
    for outcome in outcomes:
        if not outcome.ok:
            return SurfaceVerdict(surface=surface, verdict="FAIL", code=outcome.code, detail=outcome.detail)
    code = (
        preferred_success_code
        if all(outcome.code == preferred_success_code for outcome in outcomes)
        else fallback_success_code
    )
    detail = "; ".join(outcome.detail for outcome in outcomes)
    return SurfaceVerdict(surface=surface, verdict="PASS", code=code, detail=detail)


def _probe_frontdoor(
    *,
    frontdoor_url: str,
    access_team_domain: str,
    timeout_seconds: float,
    user_agent: str,
) -> SurfaceVerdict:
    outcomes = [
        _classify_frontdoor_probe(
            _perform_request(
                base_url=frontdoor_url,
                path=path,
                accept=HTML_ACCEPT,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            ),
            access_team_domain=access_team_domain,
        )
        for path in FRONTDOOR_PATHS
    ]
    return _surface_verdict_from_path_outcomes(
        surface="frontdoor",
        outcomes=outcomes,
        preferred_success_code="cf_access_redirect",
        fallback_success_code="cf_access_interstitial",
    )


def _probe_vercel_deployment(
    *,
    vercel_url: str,
    timeout_seconds: float,
    user_agent: str,
) -> SurfaceVerdict:
    outcomes = [
        _classify_vercel_probe(
            _perform_request(
                base_url=vercel_url,
                path=path,
                accept=HTML_ACCEPT,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            )
        )
        for path in FRONTDOOR_PATHS
    ]
    return _surface_verdict_from_path_outcomes(
        surface="vercel_deployment",
        outcomes=outcomes,
        preferred_success_code="vercel_auth_redirect",
        fallback_success_code="vercel_protected_status",
    )


def _probe_worker_deployment(
    *,
    worker_url: str,
    access_team_domain: str,
    timeout_seconds: float,
    user_agent: str,
) -> SurfaceVerdict:
    outcomes = [
        _classify_frontdoor_probe(
            _perform_request(
                base_url=worker_url,
                path=path,
                accept=HTML_ACCEPT,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            ),
            access_team_domain=access_team_domain,
        )
        for path in FRONTDOOR_PATHS
    ]
    return _surface_verdict_from_path_outcomes(
        surface="worker_deployment",
        outcomes=outcomes,
        preferred_success_code="cf_access_redirect",
        fallback_success_code="cf_access_interstitial",
    )


def _probe_deployment_target(
    *,
    deployment_target: str,
    deployment_url: str,
    access_team_domain: str,
    timeout_seconds: float,
    user_agent: str,
) -> SurfaceVerdict:
    if deployment_target == "vercel":
        return _probe_vercel_deployment(
            vercel_url=deployment_url,
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
        )
    if deployment_target == "cloudflare-worker":
        return _probe_worker_deployment(
            worker_url=deployment_url,
            access_team_domain=access_team_domain,
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
        )
    raise ValueError(f"Unsupported deployment target: {deployment_target}")


def _healthy_json_payload(probe: ProbeResponse) -> bool:
    if probe.status is None or not (200 <= probe.status < 300):
        return False
    try:
        import json

        payload = json.loads(probe.body)
    except Exception:
        return False
    return payload.get("ok") is True


def _probe_fastapi_public_surface(
    *,
    fastapi_public_url: str | None,
    confirm_fastapi_non_public: bool,
    timeout_seconds: float,
    user_agent: str,
) -> SurfaceVerdict:
    if confirm_fastapi_non_public:
        return SurfaceVerdict(
            surface="fastapi_public_probe",
            verdict="PASS",
            code="fastapi_non_public_confirmed",
            detail="operator confirmed no public FastAPI URL exists",
        )

    if not fastapi_public_url:
        return SurfaceVerdict(
            surface="fastapi_public_probe",
            verdict="FAIL",
            code="fastapi_attestation_missing",
            detail="supply --fastapi-public-url or --confirm-fastapi-non-public",
        )

    blocked_details: list[str] = []
    ambiguous_details: list[str] = []
    for path in FASTAPI_PATHS:
        probe = _perform_request(
            base_url=fastapi_public_url,
            path=path,
            accept=JSON_ACCEPT,
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
        )
        if _healthy_json_payload(probe):
            return SurfaceVerdict(
                surface="fastapi_public_probe",
                verdict="FAIL",
                code="fastapi_public_health_exposed",
                detail=f"path={path} status={probe.status} returned ok=true",
            )
        if probe.transport_error:
            if probe.transport_error in BLOCKED_TRANSPORT_ERRORS:
                blocked_details.append(f"path={path} blocked={probe.transport_error}")
                continue
            ambiguous_details.append(f"path={path} transport_error={probe.transport_error}")
            continue
        if probe.status in {401, 403, 404}:
            blocked_details.append(f"path={path} status={probe.status}")
            continue
        ambiguous_details.append(f"path={path} status={probe.status or 0}")

    if len(blocked_details) == len(FASTAPI_PATHS):
        return SurfaceVerdict(
            surface="fastapi_public_probe",
            verdict="PASS",
            code="fastapi_probe_blocked",
            detail="; ".join(blocked_details),
        )
    return SurfaceVerdict(
        surface="fastapi_public_probe",
        verdict="FAIL",
        code="fastapi_probe_unclassified",
        detail="; ".join(ambiguous_details + blocked_details),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.confirm_fastapi_non_public and args.fastapi_public_url:
        parser.error("Use either --fastapi-public-url or --confirm-fastapi-non-public, not both.")
    if args.deployment_target:
        args.deployment_target = args.deployment_target.replace("_", "-")
    if args.deployment_url and args.vercel_deployment_url:
        parser.error("Use either --deployment-url or --vercel-deployment-url, not both.")
    if args.vercel_deployment_url:
        if args.deployment_target and args.deployment_target != "vercel":
            parser.error("--vercel-deployment-url requires --deployment-target=vercel.")
        args.deployment_target = "vercel"
        args.deployment_url = args.vercel_deployment_url
    if not args.deployment_url:
        parser.error("Supply --deployment-url, or the legacy --vercel-deployment-url alias.")
    if not args.deployment_target:
        parser.error("--deployment-target is required when --deployment-url is used.")
    try:
        args.frontdoor_url = _normalize_base_url(args.frontdoor_url, require_https=True, label="frontdoor URL")
        args.deployment_url = _normalize_base_url(
            args.deployment_url,
            require_https=True,
            label=f"{args.deployment_target} deployment URL",
        )
        if args.fastapi_public_url:
            args.fastapi_public_url = _normalize_base_url(
                args.fastapi_public_url,
                require_https=False,
                label="FastAPI public URL",
            )
        args.cf_access_team_domain = _normalize_access_team_domain(args.cf_access_team_domain)
    except ValueError as exc:
        parser.error(str(exc))
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero.")
    return args


def run_validation(argv: Sequence[str] | None = None) -> list[SurfaceVerdict]:
    args = _parse_args(argv)
    return [
        _probe_frontdoor(
            frontdoor_url=args.frontdoor_url,
            access_team_domain=args.cf_access_team_domain,
            timeout_seconds=args.timeout_seconds,
            user_agent=args.user_agent,
        ),
        _probe_deployment_target(
            deployment_target=args.deployment_target,
            deployment_url=args.deployment_url,
            access_team_domain=args.cf_access_team_domain,
            timeout_seconds=args.timeout_seconds,
            user_agent=args.user_agent,
        ),
        _probe_fastapi_public_surface(
            fastapi_public_url=args.fastapi_public_url,
            confirm_fastapi_non_public=args.confirm_fastapi_non_public,
            timeout_seconds=args.timeout_seconds,
            user_agent=args.user_agent,
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    verdicts = run_validation(argv)
    overall_pass = all(verdict.verdict == "PASS" for verdict in verdicts)
    for verdict in verdicts:
        print(verdict.render(), flush=True)
    print(f"overall={'PASS' if overall_pass else 'FAIL'}", flush=True)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
