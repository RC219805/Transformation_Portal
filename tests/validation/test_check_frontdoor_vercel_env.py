from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validation" / "check_frontdoor_vercel_env.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_frontdoor_vercel_env", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_user_json() -> str:
    return json.dumps(
        [
            {
                "username": "admin",
                "password_hash": "hash",
                "access_email": "admin@example.com",
                "role": "admin",
            }
        ]
    )


@pytest.mark.unit
def test_vercel_env_accepts_runtime_fastapi_origin_without_backend_alias() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "single_instance",
        },
        production=False,
    )

    assert ok is True
    assert all(row[1] != "TP_BACKEND_ORIGIN" for row in rows)
    assert (
        "optional",
        "TP_FRONTDOOR_RUM_ENABLED",
        "Independent landing/login/logout RUM flag",
    ) in rows
    assert (
        "optional",
        "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT",
        "Independent front-door RUM sampling percent",
    ) in rows


@pytest.mark.unit
def test_vercel_env_reports_frontdoor_rum_knobs_when_configured() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "single_instance",
            "TP_PORTAL_RUM_ENABLED": "1",
            "TP_PORTAL_RUM_ROLLOUT_PERCENT": "100",
            "TP_FRONTDOOR_RUM_ENABLED": "1",
            "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT": "25",
        },
        production=False,
    )

    assert ok is True
    assert ("ok", "TP_PORTAL_RUM_ENABLED", "set via TP_PORTAL_RUM_ENABLED") in rows
    assert ("ok", "TP_PORTAL_RUM_ROLLOUT_PERCENT", "set via TP_PORTAL_RUM_ROLLOUT_PERCENT") in rows
    assert ("ok", "TP_FRONTDOOR_RUM_ENABLED", "set via TP_FRONTDOOR_RUM_ENABLED") in rows
    assert (
        "ok",
        "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT",
        "set via TP_FRONTDOOR_RUM_ROLLOUT_PERCENT",
    ) in rows


@pytest.mark.unit
def test_vercel_env_rejects_empty_user_json() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": "[]",
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "single_instance",
        },
        production=False,
    )

    assert ok is False
    assert (
        "missing",
        "TP_FRONTDOOR_USERS_JSON|TP_FRONTDOOR_USERS_FILE",
        "TP_FRONTDOOR_USERS_JSON contains zero valid users",
    ) in rows


@pytest.mark.unit
def test_vercel_env_snapshot_accepts_file_backed_user_source_without_local_file() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_FILE": "/vercel/runtime/users.json",
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "single_instance",
        },
        production=False,
        validate_user_file_contents=False,
    )

    assert ok is True
    assert (
        "ok",
        "TP_FRONTDOOR_USERS_JSON|TP_FRONTDOOR_USERS_FILE",
        "declared via TP_FRONTDOOR_USERS_FILE (file contents not available in env snapshot)",
    ) in rows


@pytest.mark.unit
def test_vercel_env_rejects_runtime_unsupported_session_scaling_mode() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "planet-scale",
        },
        production=False,
    )

    assert ok is False
    assert ("missing", "TP_FRONTDOOR_SESSION_SCALING_MODE", "unsupported session scaling mode: planet_scale") in rows


@pytest.mark.unit
def test_vercel_env_accepts_redis_backed_multi_instance_sessions() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "multi-instance",
            "TP_FRONTDOOR_SESSION_STORE": "redis",
            "TP_FRONTDOOR_REDIS_URL": "rediss://session.example.com:6380/0",
        },
        production=False,
    )

    assert ok is True
    assert (
        "ok",
        "TP_FRONTDOOR_SESSION_SCALING_MODE",
        "set via TP_FRONTDOOR_SESSION_SCALING_MODE (multi_instance) with Redis session store",
    ) in rows
    assert ("ok", "TP_FRONTDOOR_SESSION_STORE", "set via TP_FRONTDOOR_SESSION_STORE (redis)") in rows
    assert ("ok", "TP_FRONTDOOR_REDIS_URL", "set via TP_FRONTDOOR_REDIS_URL") in rows


@pytest.mark.unit
def test_vercel_env_rejects_external_scaling_without_redis_store() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "ephemeral_runtime",
        },
        production=False,
    )

    assert ok is False
    assert (
        "missing",
        "TP_FRONTDOOR_SESSION_SCALING_MODE",
        "ephemeral_runtime requires TP_FRONTDOOR_SESSION_STORE=redis",
    ) in rows


@pytest.mark.unit
def test_vercel_env_rejects_redis_store_without_redis_url() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "single_instance",
            "TP_FRONTDOOR_SESSION_STORE": "redis",
        },
        production=False,
    )

    assert ok is False
    assert (
        "missing",
        "TP_FRONTDOOR_SESSION_STORE",
        "TP_FRONTDOOR_SESSION_STORE=redis requires TP_FRONTDOOR_REDIS_URL",
    ) in rows
    assert (
        "missing",
        "TP_FRONTDOOR_REDIS_URL",
        "TP_FRONTDOOR_REDIS_URL is required for Redis-backed sessions",
    ) in rows


@pytest.mark.unit
def test_vercel_env_rejects_invalid_redis_url_scheme() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "multi_instance",
            "TP_FRONTDOOR_SESSION_STORE": "redis",
            "TP_FRONTDOOR_REDIS_URL": "https://session.example.com",
        },
        production=False,
    )

    assert ok is False
    assert (
        "missing",
        "TP_FRONTDOOR_REDIS_URL",
        "TP_FRONTDOOR_REDIS_URL must be an absolute redis:// or rediss:// URL",
    ) in rows


@pytest.mark.unit
def test_vercel_env_rejects_unsupported_session_store_backend() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        {
            "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
            "TP_BACKEND_API_KEY": "secret",
            "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
            "TP_FRONTDOOR_SESSION_SCALING_MODE": "single_instance",
            "TP_FRONTDOOR_SESSION_STORE": "memcached",
        },
        production=False,
    )

    assert ok is False
    assert ("missing", "TP_FRONTDOOR_SESSION_STORE", "unsupported session store backend: memcached") in rows


PUBLIC_ORIGIN_VECTORS_PATH = (
    Path(__file__).resolve().parents[2] / "web" / "secure-landing" / "tests" / "fixtures" / "public-origin-vectors.json"
)
PUBLIC_ORIGIN_UNSET_DETAIL = "only required when browsers reach the frontdoor through an edge proxy or tunnel"
PUBLIC_ORIGIN_FORM_DETAIL = (
    "TP_FRONTDOOR_PUBLIC_ORIGIN must be exactly https://<host>[:<port>] in lower case, "
    "as a browser sends it in the Origin header"
)


def _public_origin_env(**overrides: str) -> dict[str, str]:
    env = {
        "TP_FASTAPI_ORIGIN": "https://fastapi.example.com",
        "TP_BACKEND_API_KEY": "secret",
        "TP_FRONTDOOR_USERS_JSON": _valid_user_json(),
        "TP_FRONTDOOR_SESSION_SCALING_MODE": "single_instance",
    }
    env.update(overrides)
    return env


@pytest.mark.unit
def test_vercel_env_public_origin_validator_matches_the_shared_vectors() -> None:
    """The frontdoor's JavaScript validator is driven by the same file, so the two cannot drift."""
    module = _load_module()
    vectors = json.loads(PUBLIC_ORIGIN_VECTORS_PATH.read_text(encoding="utf-8"))

    assert vectors["schema"] == "tp.frontdoor.public_origin_vectors.v1"
    assert vectors["accepted"] and vectors["rejected"] and vectors["trimmed"]

    for value in vectors["accepted"]:
        assert module._is_canonical_public_origin(value) is True, value
    for vector in vectors["rejected"]:
        assert module._is_canonical_public_origin(vector["value"]) is False, vector["why"]
    for vector in vectors["trimmed"]:
        valid, detail, configured = module._evaluate_public_origin({"TP_FRONTDOOR_PUBLIC_ORIGIN": vector["value"]})
        assert (valid, configured) == (True, True), repr(vector["value"])
        assert detail == f"redirects and same-origin checks use {vector['origin']}"


@pytest.mark.unit
def test_vercel_env_public_origin_bounds_host_length() -> None:
    module = _load_module()
    label = "a" * 63
    longest_host = f"{label}.{label}.{label}.{'a' * 57}.com"

    assert len(longest_host) == 253
    assert module._is_canonical_public_origin(f"https://{longest_host}") is True
    assert module._is_canonical_public_origin(f"https://{label}.{label}.{label}.{'a' * 58}.com") is False


@pytest.mark.unit
def test_vercel_env_treats_public_origin_as_optional_when_unset() -> None:
    module = _load_module()

    for env in (_public_origin_env(), _public_origin_env(TP_FRONTDOOR_PUBLIC_ORIGIN=" \t\r\n")):
        ok, rows = module._evaluate(env, production=True)
        assert ("optional", "TP_FRONTDOOR_PUBLIC_ORIGIN", PUBLIC_ORIGIN_UNSET_DETAIL) in rows
        # Production still fails here, but only for the Cloudflare Access variables.
        assert ok is False
        assert all(row[1] != "TP_FRONTDOOR_PUBLIC_ORIGIN" for row in rows if row[0] == "missing")


@pytest.mark.unit
def test_vercel_env_reports_the_accepted_public_origin_for_review() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        _public_origin_env(TP_FRONTDOOR_PUBLIC_ORIGIN="https://portal.example.com"),
        production=False,
    )

    assert ok is True
    assert (
        "ok",
        "TP_FRONTDOOR_PUBLIC_ORIGIN",
        "redirects and same-origin checks use https://portal.example.com",
    ) in rows


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "portal.example.com",
        "http://portal.example.com",
        "https://portal.example.com/",
        "https://Portal.example.com",
        "https://portal.example.com:443",
        "https://127.0.0.1",
        "https://localhost:3000",
    ],
)
def test_vercel_env_rejects_malformed_public_origin(value: str) -> None:
    module = _load_module()
    ok, rows = module._evaluate(_public_origin_env(TP_FRONTDOOR_PUBLIC_ORIGIN=value), production=False)

    assert ok is False
    assert ("missing", "TP_FRONTDOOR_PUBLIC_ORIGIN", PUBLIC_ORIGIN_FORM_DETAIL) in rows


@pytest.mark.unit
def test_vercel_env_rejects_public_origin_combined_with_local_access_bypass() -> None:
    module = _load_module()
    ok, rows = module._evaluate(
        _public_origin_env(
            TP_FRONTDOOR_PUBLIC_ORIGIN="https://portal.example.com",
            TP_ALLOW_LOCAL_ACCESS_BYPASS="1",
        ),
        production=False,
    )

    assert ok is False
    assert (
        "missing",
        "TP_FRONTDOOR_PUBLIC_ORIGIN",
        "TP_FRONTDOOR_PUBLIC_ORIGIN cannot be combined with TP_ALLOW_LOCAL_ACCESS_BYPASS=1",
    ) in rows
