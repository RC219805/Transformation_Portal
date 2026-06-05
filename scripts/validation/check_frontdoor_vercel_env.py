#!/usr/bin/env python3
"""Validate the managed frontdoor's Vercel/production environment.

Reads the variables documented in docs/operations/frontdoor_vercel_env.md and
prints a green/red table. Intended to run against a `vercel env pull` snapshot
or against the operator's current shell.

Exit codes:
    0 - all required variables present (production-mode requirements satisfied)
    1 - one or more required variables missing
    2 - usage error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional


RequiredScope = Literal["all", "production", "optional"]


@dataclass(frozen=True)
class Variable:
    name: str
    required_in: RequiredScope
    description: str


VARIABLES: tuple[Variable, ...] = (
    Variable("TP_FASTAPI_ORIGIN", "all", "Origin used by /healthz and the v1 proxy"),
    Variable("TP_BACKEND_API_KEY", "all", "Frontdoor-presented API key (must equal backend TP_API_KEY)"),
    Variable("TP_FRONTDOOR_USERS_JSON|TP_FRONTDOOR_USERS_FILE", "all", "User source (JSON array or file path)"),
    Variable("TP_FRONTDOOR_SESSION_SCALING_MODE", "all", "single_instance or external-store-backed mode"),
    Variable("TP_FRONTDOOR_SESSION_STORE", "optional", "sqlite default or redis for external session storage"),
    Variable("TP_FRONTDOOR_REDIS_URL", "optional", "Redis URL required when TP_FRONTDOOR_SESSION_STORE=redis"),
    Variable("TP_CF_ACCESS_TEAM_DOMAIN", "production", "Cloudflare Access team domain"),
    Variable("TP_CF_ACCESS_AUD", "production", "Cloudflare Access JWT audience"),
    Variable("TP_PORTAL_RUM_ENABLED", "optional", "Shared portal/frontdoor RUM kill switch"),
    Variable("TP_PORTAL_RUM_ROLLOUT_PERCENT", "optional", "Managed portal/bootstrap RUM rollout percent"),
    Variable("TP_FRONTDOOR_RUM_ENABLED", "optional", "Independent landing/login/logout RUM flag"),
    Variable("TP_FRONTDOOR_RUM_ROLLOUT_PERCENT", "optional", "Independent front-door RUM sampling percent"),
)

SUPPORTED_SESSION_SCALING_MODES = frozenset({"single_instance"})
EXTERNAL_SESSION_SCALING_MODES = frozenset({"multi_instance", "ephemeral_runtime"})
SUPPORTED_SESSION_STORE_BACKENDS = frozenset({"sqlite", "redis"})
SUPPORTED_REDIS_URL_SCHEMES = frozenset({"redis", "rediss"})


def _resolve(value_or_alias: str, env: dict[str, str]) -> Optional[str]:
    for name in value_or_alias.split("|"):
        if env.get(name, "").strip():
            return name
    return None


def _valid_user_count_from_json(raw: str) -> int:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if not isinstance(parsed, list):
        return 0
    valid = 0
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("username", "")).strip()
            and str(item.get("password_hash", "")).strip()
            and str(item.get("access_email", "")).strip()
        ):
            valid += 1
    return valid


def _evaluate_user_source(env: dict[str, str], *, validate_user_file_contents: bool = True) -> tuple[bool, str]:
    users_file = env.get("TP_FRONTDOOR_USERS_FILE", "").strip()
    if users_file:
        if not validate_user_file_contents:
            return True, "declared via TP_FRONTDOOR_USERS_FILE (file contents not available in env snapshot)"
        try:
            with open(users_file, "r", encoding="utf-8") as handle:
                count = _valid_user_count_from_json(handle.read())
        except OSError as exc:
            return False, f"TP_FRONTDOOR_USERS_FILE unreadable: {exc}"
        if count <= 0:
            return False, "TP_FRONTDOOR_USERS_FILE contains zero valid users"
        return True, f"set via TP_FRONTDOOR_USERS_FILE ({count} valid user(s))"

    users_json = env.get("TP_FRONTDOOR_USERS_JSON", "").strip()
    if not users_json:
        return False, "User source (JSON array or file path)"
    count = _valid_user_count_from_json(users_json)
    if count <= 0:
        return False, "TP_FRONTDOOR_USERS_JSON contains zero valid users"
    return True, f"set via TP_FRONTDOOR_USERS_JSON ({count} valid user(s))"


def _normalize_session_scaling_mode(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _normalize_session_store_backend(value: str) -> str:
    return value.strip().lower() if value.strip() else "sqlite"


def _validate_redis_url(value: str) -> tuple[bool, str]:
    raw = value.strip()
    if not raw:
        return False, "TP_FRONTDOOR_REDIS_URL is required for Redis-backed sessions"
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme.lower() not in SUPPORTED_REDIS_URL_SCHEMES or not parsed.netloc:
        return False, "TP_FRONTDOOR_REDIS_URL must be an absolute redis:// or rediss:// URL"
    return True, "set via TP_FRONTDOOR_REDIS_URL"


def _evaluate_session_store_backend(env: dict[str, str]) -> tuple[bool, str]:
    raw = env.get("TP_FRONTDOOR_SESSION_STORE", "").strip()
    backend = _normalize_session_store_backend(raw)
    if backend not in SUPPORTED_SESSION_STORE_BACKENDS:
        return False, f"unsupported session store backend: {backend}"
    if backend == "redis":
        redis_url = env.get("TP_FRONTDOOR_REDIS_URL", "")
        redis_ok, redis_detail = _validate_redis_url(redis_url)
        if not redis_ok:
            if not redis_url.strip():
                return False, "TP_FRONTDOOR_SESSION_STORE=redis requires TP_FRONTDOOR_REDIS_URL"
            return False, redis_detail
        return True, "set via TP_FRONTDOOR_SESSION_STORE (redis)"
    if raw:
        return True, "set via TP_FRONTDOOR_SESSION_STORE (sqlite)"
    return True, "default sqlite session store"


def _evaluate_redis_url(env: dict[str, str]) -> tuple[bool, str, bool]:
    scaling_mode = _normalize_session_scaling_mode(env.get("TP_FRONTDOOR_SESSION_SCALING_MODE", ""))
    backend = _normalize_session_store_backend(env.get("TP_FRONTDOOR_SESSION_STORE", ""))
    required = backend == "redis" or scaling_mode in EXTERNAL_SESSION_SCALING_MODES
    raw = env.get("TP_FRONTDOOR_REDIS_URL", "").strip()
    if not raw and not required:
        return True, "only required when TP_FRONTDOOR_SESSION_STORE=redis", False
    valid, detail = _validate_redis_url(raw)
    return valid, detail, bool(raw)


def _evaluate_session_scaling_mode(env: dict[str, str]) -> tuple[bool, str]:
    raw = env.get("TP_FRONTDOOR_SESSION_SCALING_MODE", "").strip()
    if not raw:
        return False, "single_instance or external-store-backed mode"
    mode = _normalize_session_scaling_mode(raw)
    if mode in SUPPORTED_SESSION_SCALING_MODES:
        return True, f"set via TP_FRONTDOOR_SESSION_SCALING_MODE ({mode})"
    if mode in EXTERNAL_SESSION_SCALING_MODES:
        backend = _normalize_session_store_backend(env.get("TP_FRONTDOOR_SESSION_STORE", ""))
        if backend != "redis":
            return False, f"{mode} requires TP_FRONTDOOR_SESSION_STORE=redis"
        redis_url = env.get("TP_FRONTDOOR_REDIS_URL", "")
        redis_ok, redis_detail = _validate_redis_url(redis_url)
        if not redis_ok:
            if not redis_url.strip():
                return False, f"{mode} requires TP_FRONTDOOR_REDIS_URL"
            return False, f"{mode} requires {redis_detail}"
        return True, f"set via TP_FRONTDOOR_SESSION_SCALING_MODE ({mode}) with Redis session store"
    return False, f"unsupported session scaling mode: {mode}"


def _load_env_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'") and len(value) >= 2:
                value = value[1:-1]
            out[key.strip()] = value
    return out


def _evaluate(
    env: dict[str, str],
    production: bool,
    *,
    validate_user_file_contents: bool = True,
) -> tuple[bool, List[tuple[str, str, str]]]:
    rows: List[tuple[str, str, str]] = []
    ok = True
    for var in VARIABLES:
        required = var.required_in == "all" or (production and var.required_in == "production")
        if var.name == "TP_FRONTDOOR_USERS_JSON|TP_FRONTDOOR_USERS_FILE":
            valid, detail = _evaluate_user_source(
                env,
                validate_user_file_contents=validate_user_file_contents,
            )
            if valid:
                rows.append(("ok", var.name, detail))
            elif required:
                ok = False
                rows.append(("missing", var.name, detail))
            else:
                rows.append(("optional", var.name, detail))
            continue
        if var.name == "TP_FRONTDOOR_SESSION_SCALING_MODE":
            valid, detail = _evaluate_session_scaling_mode(env)
            if valid:
                rows.append(("ok", var.name, detail))
            elif required:
                ok = False
                rows.append(("missing", var.name, detail))
            else:
                rows.append(("optional", var.name, detail))
            continue
        if var.name == "TP_FRONTDOOR_SESSION_STORE":
            valid, detail = _evaluate_session_store_backend(env)
            if valid:
                rows.append(("ok", var.name, detail))
            else:
                ok = False
                rows.append(("missing", var.name, detail))
            continue
        if var.name == "TP_FRONTDOOR_REDIS_URL":
            valid, detail, configured = _evaluate_redis_url(env)
            if valid and configured:
                rows.append(("ok", var.name, detail))
            elif valid:
                rows.append(("optional", var.name, detail))
            else:
                ok = False
                rows.append(("missing", var.name, detail))
            continue
        resolved = _resolve(var.name, env)
        if resolved:
            rows.append(("ok", var.name, f"set via {resolved}"))
            continue
        if required:
            ok = False
            rows.append(("missing", var.name, var.description))
        else:
            rows.append(("optional", var.name, var.description))
    return ok, rows


def _format(rows: Iterable[tuple[str, str, str]], color: bool) -> str:
    def fmt(state: str) -> str:
        if not color:
            return f"[{state}]".ljust(11)
        if state == "ok":
            return "[\x1b[32mok\x1b[0m]      "
        if state == "missing":
            return "[\x1b[31mmissing\x1b[0m] "
        return f"[{state}]".ljust(11)

    lines = []
    for state, name, detail in rows:
        lines.append(f"{fmt(state)} {name:<48} {detail}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        help="Optional path to a dotenv-style file (e.g. output of `vercel env pull`).",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Apply production-mode requirements (Cloudflare Access).",
    )
    parser.add_argument(
        "--validate-user-file",
        action="store_true",
        help=(
            "When --env-file is used, also require TP_FRONTDOOR_USERS_FILE to be "
            "readable on this machine and contain at least one valid user."
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output.",
    )
    args = parser.parse_args(argv)

    env: dict[str, str] = dict(os.environ)
    if args.env_file:
        try:
            env.update(_load_env_file(args.env_file))
        except OSError as exc:
            print(f"check_frontdoor_vercel_env: cannot read {args.env_file}: {exc}", file=sys.stderr)
            return 2

    ok, rows = _evaluate(
        env,
        production=args.production,
        validate_user_file_contents=args.validate_user_file or not bool(args.env_file),
    )
    print(_format(rows, color=sys.stdout.isatty() and not args.no_color))
    if not ok:
        print(
            "\nSome required variables are missing. " "See docs/operations/frontdoor_vercel_env.md.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
