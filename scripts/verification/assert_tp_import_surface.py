#!/usr/bin/env python3
"""Assert explicit importability intent for a module name."""

from __future__ import annotations

import argparse
import importlib.util
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert expected importability for a module.")
    parser.add_argument("--module", default="tp", help="Module name to validate.")
    parser.add_argument(
        "--expect",
        choices=("importable", "not-importable"),
        required=True,
        help="Expected importability state.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = importlib.util.find_spec(args.module)
    is_importable = spec is not None
    expected = args.expect == "importable"

    if is_importable != expected:
        print(
            f"Import surface assertion failed for '{args.module}': "
            f"expected={args.expect}, actual={'importable' if is_importable else 'not-importable'}",
            file=sys.stderr,
        )
        return 1

    location = spec.origin if spec is not None else "N/A"
    print(
        f"Import surface assertion passed for '{args.module}': "
        f"{'importable' if is_importable else 'not-importable'} (origin={location})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
