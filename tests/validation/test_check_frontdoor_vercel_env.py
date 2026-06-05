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
