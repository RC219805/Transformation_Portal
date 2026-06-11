#!/usr/bin/env python3
"""Validate governed dependency pinning contracts.

Policy:
- Every requirement line in ``requirements/*.txt`` must specify an exact
  ``==`` version pin (``package==version``) so installations are reproducible.
  PEP 440 arbitrary-equality (``===``) and any range/inequality operator
  (``>=``, ``<=``, ``~=``, ``!=``, ``>``, ``<``) are rejected.
- ``constraints.txt`` is exempt from the lockfile ``==`` policy because it
  intentionally uses ``>=9999.0.0`` to hard-block banned packages. Any normal
  ``constraints.txt`` pin must use exact ``==`` syntax and match the version
  installed in the governed Python environment.
- Hashes, comments, blank lines, options (``-r``, ``-c``, ``--hash``, etc.),
  and pip-compile continuation lines are ignored.

This complements ``check_requirements_lock_contract.py`` (which validates
header metadata and security baselines) by enforcing the broader supply-chain
invariant that no transitive dependency drifts onto a floating range.

Implements the core local-enforcement piece of TODO_INVENTORY.md §5.7
(Dependency Pinning Validation). A dedicated GitHub Actions workflow
(``.github/workflows/dependency-pinning-check.yml``) runs this same check as an
isolated PR/push signal.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_DIR = PROJECT_ROOT / "requirements"

EXEMPT_FILES = frozenset({"constraints.txt"})

# ``==`` only — explicitly reject ``===`` (PEP 440 arbitrary equality) and any
# range/inequality operator. The ``(?!=)`` lookahead is the trip-wire that
# stops ``pkg===1.2.3`` from masquerading as a ``==`` pin.
PINNED_LINE_PATTERN = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[A-Za-z0-9_.,\-\s]+\])?==(?!=)[^\s;#]+")
CONSTRAINT_PIN_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)(?:\[[A-Za-z0-9_.,\-\s]+\])?==(?!=)(?P<version>[^\s;#]+)$")
HARD_BLOCK_CONSTRAINT_PATTERN = re.compile(r">=\s*9999(?:\.0\.0)?\b")
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


def _requirement_head(line: str) -> str:
    """Return the requirement segment before comments or environment markers."""
    return re.split(r"[;#]", line, maxsplit=1)[0].strip()


def _collapsed_requirement_head(line: str) -> str:
    """Return a whitespace-normalized requirement segment for policy matching."""
    return re.sub(r"\s+", "", _requirement_head(line))


def _is_hard_block_constraint(line: str) -> bool:
    """Return True when a constraints entry intentionally blocks a package."""
    return bool(HARD_BLOCK_CONSTRAINT_PATTERN.search(_requirement_head(line)))


def _iter_logical_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(start_line_number, joined_line)`` for each logical requirement.

    Handles two physical-line wrapping conventions that show up in
    requirements files:

    1. Trailing backslash continuation (``pkg \\`` on one line, ``--hash=...``
       on the next). These are normalised to a single logical line so the
       continuation doesn't get treated as a stand-alone requirement.
    2. Bracket-balanced extras wraps (``pkg[extra1,`` on one line,
       ``    extra2]==1.0.0`` on the next). pip-compile usually keeps these
       on a single line, but hand-edited inputs often split them; without
       rejoining, the trailing fragment looks like an unpinned requirement.

    Comment-only and ``--hash`` continuations remain ignored downstream and
    are not eagerly joined into the previous logical line.
    """

    physical_lines = text.splitlines()
    index = 0
    total = len(physical_lines)
    while index < total:
        start_line = index + 1
        accumulated = physical_lines[index]
        index += 1

        # Backslash continuation: drop the trailing slash and pull the next
        # physical line in, separated by a single space so tokens don't merge.
        while accumulated.rstrip().endswith("\\") and index < total:
            accumulated = accumulated.rstrip()[:-1].rstrip()
            accumulated = f"{accumulated} {physical_lines[index].strip()}"
            index += 1

        # Bracket continuation: only join when the open bracket precedes the
        # first comment/marker character on the line, so we don't misread a
        # bracket inside a ``# via foo[bar]`` comment as an open extras list.
        while _has_unbalanced_extras(accumulated) and index < total:
            accumulated = f"{accumulated.rstrip()} {physical_lines[index].strip()}"
            index += 1

        yield start_line, accumulated


def _has_unbalanced_extras(line: str) -> bool:
    """Return True if ``line`` opens an extras list that doesn't close on it."""
    head = re.split(r"[;#]", line, maxsplit=1)[0]
    return head.count("[") > head.count("]")


def find_violations(paths: Iterable[Path]) -> list[str]:
    """Return human-readable violations for any non-pinned requirement line."""
    violations: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(f"{path}: could not read file ({exc})")
            continue

        for line_number, logical_line in _iter_logical_lines(text):
            line = logical_line
            if _is_skippable(line) or _is_hash_continuation(line):
                continue

            # Strip inline environment markers / comments so we focus on the
            # ``package<op>version`` segment.
            head = _requirement_head(line)
            if not head:
                continue
            # Stray fragments without a leading package name (the bracket
            # joiner above should normally absorb these) carry no version
            # operator and are not actionable.
            if not re.match(r"^[A-Za-z0-9_.\-]", head):
                continue

            # Collapse internal whitespace so ``pkg[a, b]==1.0.0`` (joined
            # from a wrapped extras list) is matched the same way pip would
            # canonicalise it.
            collapsed = _collapsed_requirement_head(line)

            if PINNED_LINE_PATTERN.match(collapsed):
                continue

            offending_op = next((op for op in UNPINNED_OPERATORS if op in collapsed), None)
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


def find_constraint_install_violations(
    constraints_path: Path,
    *,
    version_lookup: Callable[[str], str] = importlib_metadata.version,
) -> list[str]:
    """Return violations for normal constraints that drift from installed packages.

    ``constraints.txt`` also carries intentional hard-blocks such as
    ``realesrgan>=9999.0.0``. Those entries are supply-chain bans, not install
    pins, so they are skipped here and enforced by the banned-dependencies
    validator.
    """
    violations: list[str] = []
    try:
        text = constraints_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{constraints_path}: could not read file ({exc})"]

    for line_number, logical_line in _iter_logical_lines(text):
        if _is_skippable(logical_line) or _is_hash_continuation(logical_line):
            continue
        if _is_hard_block_constraint(logical_line):
            continue

        head = _requirement_head(logical_line)
        if not head:
            continue
        if not re.match(r"^[A-Za-z0-9_.\-]", head):
            continue

        collapsed = _collapsed_requirement_head(logical_line)
        match = CONSTRAINT_PIN_PATTERN.match(collapsed)
        if match is None:
            offending_op = next((op for op in UNPINNED_OPERATORS if op in collapsed), None)
            if offending_op is None:
                violations.append(f"{constraints_path}:{line_number}: constraint {head!r} has no exact (==) pin")
            else:
                violations.append(
                    f"{constraints_path}:{line_number}: constraint {head!r} uses operator "
                    f"{offending_op!r}; constraints must either hard-block with '>=9999.0.0' "
                    "or use an exact installed-version '==' pin"
                )
            continue

        name = match.group("name")
        expected = match.group("version")
        try:
            installed = version_lookup(name)
        except importlib_metadata.PackageNotFoundError:
            violations.append(
                f"{constraints_path}:{line_number}: constraint pins {name}=={expected} "
                "but that distribution is not installed in the governed environment"
            )
            continue

        if installed != expected:
            violations.append(
                f"{constraints_path}:{line_number}: constraint pins {name}=={expected} "
                f"but the governed environment has {name}=={installed}"
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

    explicit = bool(args.paths)
    if explicit:
        raw_paths = sorted({path.resolve() for path in args.paths})
    else:
        raw_paths = iter_lockfiles()

    paths, constraints_paths = _partition_exempt(raw_paths)
    for skipped in constraints_paths:
        # ``constraints.txt`` intentionally uses ``>=9999.0.0`` to ban
        # packages, so it can never satisfy the ``==`` policy. Make the
        # exemption visible rather than silent so callers passing globs
        # like ``requirements/*.txt`` understand why it isn't scanned.
        print(f"NOTE: skipping constraints lockfile pin scan for {skipped}", file=sys.stderr)

    if not explicit:
        constraints_paths = [REQUIREMENTS_DIR / "constraints.txt"]

    if not paths and not constraints_paths:
        # Distinguish "default scan turned up nothing" from "every explicit
        # path was exempt" so the message describes the actual input mode.
        if explicit:
            print(
                "WARNING: no lockfiles supplied to scan.",
                file=sys.stderr,
            )
        else:
            print(
                f"WARNING: no lockfiles found under {REQUIREMENTS_DIR}",
                file=sys.stderr,
            )
        return 0

    violations = find_violations(paths)
    for constraints_path in constraints_paths:
        violations.extend(find_constraint_install_violations(constraints_path))

    if violations:
        print("ERROR: dependency pinning validation failed:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    scanned = ", ".join(path.name for path in paths) if paths else "none"
    audited = ", ".join(path.name for path in constraints_paths) if constraints_paths else "none"
    print(
        "dependency pinning passed: "
        f"{len(paths)} lockfile(s) verified ({scanned}); "
        f"{len(constraints_paths)} constraints file(s) audited ({audited})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
