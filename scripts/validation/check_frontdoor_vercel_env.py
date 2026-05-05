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
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Variable:
    name: str
    required_in: str  # "all" | "production"
    description: str


VARIABLES: tuple[Variable, ...] = (
    Variable("TP_BACKEND_ORIGIN", "all", "Upstream FastAPI origin URL"),
    Variable("TP_FASTAPI_ORIGIN", "all", "Origin alias used by /healthz and the v1 proxy"),
    Variable("TP_BACKEND_API_KEY", "all", "Frontdoor-presented API key (must equal backend TP_API_KEY)"),
    Variable("TP_FRONTDOOR_USERS_JSON|TP_FRONTDOOR_USERS_FILE", "all", "User source (JSON array or file path)"),
    Variable("TP_FRONTDOOR_SESSION_SCALING_MODE", "all", "single_instance or external-store-backed mode"),
    Variable("TP_CF_ACCESS_TEAM_DOMAIN", "production", "Cloudflare Access team domain"),
    Variable("TP_CF_ACCESS_AUD", "production", "Cloudflare Access JWT audience"),
)


def _resolve(value_or_alias: str, env: dict[str, str]) -> Optional[str]:
    for name in value_or_alias.split("|"):
        if env.get(name, "").strip():
            return name
    return None


def _load_env_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
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


def _evaluate(env: dict[str, str], production: bool) -> tuple[bool, List[tuple[str, str, str]]]:
    rows: List[tuple[str, str, str]] = []
    ok = True
    for var in VARIABLES:
        required = var.required_in == "all" or (production and var.required_in == "production")
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

    ok, rows = _evaluate(env, production=args.production)
    print(_format(rows, color=sys.stdout.isatty() and not args.no_color))
    if not ok:
        print(
            "\nSome required variables are missing. "
            "See docs/operations/frontdoor_vercel_env.md.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
