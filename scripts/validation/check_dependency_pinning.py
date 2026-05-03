#!/usr/bin/env python3
"""Validate that compiled requirements lockfiles use exact (==) version pins.

Policy:
- Every requirement line in ``requirements/*.txt`` must specify an exact
  ``==`` version pin (``package==version``) so installations are reproducible.
  PEP 440 arbitrary-equality (``===``) and any range/inequality operator
  (``>=``, ``<=``, ``~=``, ``!=``, ``>``, ``<``) are rejected.
- ``constraints.txt`` is always exempt — both when scanning the default
  ``requirements/`` directory and when callers pass it explicitly via the
  CLI (e.g. ``check_dependency_pinning.py requirements/*.txt``). The file
  intentionally uses ``>=9999.0.0`` to hard-block banned packages.
- Hashes, comments, blank lines, options (``-r``, ``-c``, ``--hash``, etc.),
  and pip-compile continuation lines are ignored.

This complements ``check_requirements_lock_contract.py`` (which validates
header metadata and security baselines) by enforcing the broader supply-chain
invariant that no transitive dependency drifts onto a floating range.

Closes TODO_INVENTORY.md §5.7 (Dependency Pinning Validation).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_DIR = PROJECT_ROOT / "requirements"

EXEMPT_FILES = frozenset({"constraints.txt"})

# ``==`` only — explicitly reject ``===`` (PEP 440 arbitrary equality) and any
# range/inequality operator. The ``(?!=)`` lookahead is the trip-wire that
# stops ``pkg===1.2.3`` from masquerading as a ``==`` pin.
PINNED_LINE_PATTERN = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[A-Za-z0-9_.,\-\s]+\])?==(?!=)[^\s;#]+")
# Order matters: longer operators must precede shorter ones so substring
# detection reports ``===`` rather than the empty-string match it contains.
UNPINNED_OPERATORS = ("===", ">=", "<=", "~=", "!=", ">", "<")


def iter_lockfiles(requirements_dir: Path = REQUIREMENTS_DIR) -> list[Path]:
    """Return compiled lockfiles to validate, in deterministic order."""
    if not requirements_dir.is_dir():
        return []
    return sorted(path for path in requirements_dir.glob("*.txt") if path.name not in EXEMPT_FILES)


def _partition_exempt(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Split ``paths`` into (lockfiles_to_scan, exempt_files_to_skip)."""
    keep: list[Path] = []
    exempt: list[Path] = []
    for path in paths:
        if path.name in EXEMPT_FILES:
            exempt.append(path)
        else:
            keep.append(path)
    return keep, exempt


def _is_skippable(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    # pip-compile continuation lines (``    # via ...``) are stripped above.
    # Pip option lines (``-r foo.in``, ``--hash=sha256:...``) are not pins.
    if stripped.startswith("-") or stripped.startswith("--"):
        return True
    return False


def _is_hash_continuation(line: str) -> bool:
    """Return True for ``--hash=...`` continuation fragments on a wrapped line."""
    return line.lstrip().startswith("--hash=")


def find_violations(paths: Iterable[Path]) -> list[str]:
    """Return human-readable violations for any non-pinned requirement line."""
    violations: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(f"{path}: could not read file ({exc})")
            continue

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.rstrip("\\").rstrip()
            if _is_skippable(line) or _is_hash_continuation(line):
                continue

            # Strip inline environment markers / comments so we focus on the
            # ``package<op>version`` segment.
            head = re.split(r"[;#]", line, maxsplit=1)[0].strip()
            if not head:
                continue
            # Continuation lines without a leading package name (e.g. wrapped
            # extras in pip-compile output) do not carry a version operator.
            if not re.match(r"^[A-Za-z0-9_.\-]", head):
                continue

            if PINNED_LINE_PATTERN.match(head):
                continue

            offending_op = next((op for op in UNPINNED_OPERATORS if op in head), None)
            if offending_op is None:
                # Bare package name with no version (e.g. ``somepkg``) is also
                # unpinned and should be flagged.
                violations.append(f"{path}:{line_number}: requirement {head!r} has no exact (==) pin")
                continue

            violations.append(
                f"{path}:{line_number}: requirement {head!r} uses unpinned operator "
                f"{offending_op!r}; lockfiles must use '==' pins"
            )
    return violations


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Validate requirements lockfiles use exact (==) version pins.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional lockfile paths to scan instead of requirements/*.txt",
    )
    args = parser.parse_args()

    if args.paths:
        raw_paths = sorted({path.resolve() for path in args.paths})
    else:
        raw_paths = iter_lockfiles()

    paths, exempt = _partition_exempt(raw_paths)
    for skipped in exempt:
        # ``constraints.txt`` intentionally uses ``>=9999.0.0`` to ban
        # packages, so it can never satisfy the ``==`` policy. Make the
        # exemption visible rather than silent so callers passing globs
        # like ``requirements/*.txt`` understand why it isn't scanned.
        print(f"NOTE: skipping exempt file {skipped}", file=sys.stderr)

    if not paths:
        print(
            f"WARNING: no lockfiles found under {REQUIREMENTS_DIR}",
            file=sys.stderr,
        )
        return 0

    violations = find_violations(paths)
    if violations:
        print("ERROR: dependency pinning validation failed:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    scanned = ", ".join(path.name for path in paths)
    print(f"dependency pinning passed: {len(paths)} lockfile(s) verified ({scanned})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
