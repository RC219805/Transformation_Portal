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
