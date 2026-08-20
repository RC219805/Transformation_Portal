#!/usr/bin/env python3
"""Restore governed platform-marker pins after host-side pip-compile output."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PACKAGE_LINE_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s;#]+)")


@dataclass(frozen=True)
class MarkerPin:
    package_name: str
    marker: str


GOVERNED_MARKER_PINS = (
    MarkerPin("opencv-python", 'platform_system != "Linux"'),
    MarkerPin("opencv-python-headless", 'platform_system == "Linux"'),
)


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _package_name_for_line(line: str) -> str | None:
    match = PACKAGE_LINE_RE.match(line.strip())
    if match is None:
        return None
    return _normalize_package_name(match.group("name"))


def _requirement_blocks(lines: Sequence[str]) -> dict[str, tuple[int, int, list[str]]]:
    blocks: dict[str, tuple[int, int, list[str]]] = {}
    index = 0
    while index < len(lines):
        package_name = _package_name_for_line(lines[index])
        if package_name is None:
            index += 1
            continue

        end = index + 1
        while end < len(lines) and (lines[end].startswith(" ") or lines[end].startswith("\t")):
            end += 1
        blocks[package_name] = (index, end, list(lines[index:end]))
        index = end
    return blocks


def _validated_previous_blocks(previous_lines: Sequence[str]) -> dict[str, list[str]]:
    previous_blocks = _requirement_blocks(previous_lines)
    restored_blocks: dict[str, list[str]] = {}

    for pin in GOVERNED_MARKER_PINS:
        package_name = _normalize_package_name(pin.package_name)
        block = previous_blocks.get(package_name)
        if block is None:
            raise ValueError(f"previous lockfile is missing governed marker pin {pin.package_name!r}")

        first_line = block[2][0]
        if pin.marker not in first_line:
            raise ValueError(f"previous lockfile pin for {pin.package_name!r} must retain marker {pin.marker!r}")
        restored_blocks[package_name] = block[2]

    return restored_blocks


def _block_version(package_name: str, block: Sequence[str]) -> str:
    match = PACKAGE_LINE_RE.match(block[0].strip())
    if match is None:
        raise ValueError(f"lockfile block for {package_name!r} does not start with a pinned requirement")
    return match.group("version")


def _resolved_current_versions(current_blocks: dict[str, tuple[int, int, list[str]]]) -> dict[str, str]:
    resolved_versions: dict[str, str] = {}

    for pin in GOVERNED_MARKER_PINS:
        package_name = _normalize_package_name(pin.package_name)
        block = current_blocks.get(package_name)
        if block is not None:
            resolved_versions[package_name] = _block_version(package_name, block[2])

    if not resolved_versions:
        governed_names = ", ".join(pin.package_name for pin in GOVERNED_MARKER_PINS)
        raise ValueError(f"current lockfile is missing newly resolved governed marker pins: {governed_names}")

    fallback_version = next(iter(resolved_versions.values()))
    return {
        _normalize_package_name(pin.package_name): resolved_versions.get(
            _normalize_package_name(pin.package_name),
            fallback_version,
        )
        for pin in GOVERNED_MARKER_PINS
    }


def _governed_marker_blocks(
    previous_blocks: dict[str, list[str]],
    current_lines: Sequence[str],
) -> dict[str, list[str]]:
    current_blocks = _requirement_blocks(current_lines)
    resolved_versions = _resolved_current_versions(current_blocks)
    marker_blocks: dict[str, list[str]] = {}

    for pin in GOVERNED_MARKER_PINS:
        package_name = _normalize_package_name(pin.package_name)
        current_block = current_blocks.get(package_name)
        previous_block = previous_blocks.get(package_name)
        if current_block is not None:
            continuation_lines = current_block[2][1:]
        elif previous_block is not None:
            continuation_lines = previous_block[1:]
        else:
            continuation_lines = ["    # via -r base.in"]
        marker_blocks[package_name] = [
            f"{pin.package_name}=={resolved_versions[package_name]} ; {pin.marker}",
            *continuation_lines,
        ]

    return marker_blocks


def _without_governed_blocks(lines: Sequence[str]) -> list[str]:
    governed_packages = {_normalize_package_name(pin.package_name) for pin in GOVERNED_MARKER_PINS}
    target_blocks = _requirement_blocks(lines)
    skip_ranges = [
        (start, end) for package_name, (start, end, _block) in target_blocks.items() if package_name in governed_packages
    ]

    if not skip_ranges:
        return list(lines)

    filtered: list[str] = []
    index = 0
    for start, end in sorted(skip_ranges):
        filtered.extend(lines[index:start])
        index = end
    filtered.extend(lines[index:])
    return filtered


def _insertion_index(lines: Sequence[str]) -> int:
    first_governed_name = _normalize_package_name(GOVERNED_MARKER_PINS[0].package_name)
    fallback = len(lines)

    for index, line in enumerate(lines):
        package_name = _package_name_for_line(line)
        if package_name is None:
            continue
        if package_name > first_governed_name:
            return index
        fallback = index + 1

    return fallback


def restore_marker_pins(previous_lockfile: Path, lockfile: Path) -> bool:
    """Restore governed marker-pin blocks into ``lockfile``.

    When ``previous_lockfile`` does not exist (for example after ``make clean``),
    synthesize the missing platform block from the version resolved for the
    other governed OpenCV package. Existing snapshots remain the preferred
    source for pip-compile provenance continuation lines.

    Returns ``True`` when the target lockfile changed.
    """
    current_lines = lockfile.read_text(encoding="utf-8").splitlines()
    if previous_lockfile.is_symlink():
        raise ValueError(f"previous lockfile must not be a symlink: {previous_lockfile}")
    if previous_lockfile.exists() and not previous_lockfile.is_file():
        raise ValueError(f"previous lockfile must be a regular file: {previous_lockfile}")

    previous_blocks: dict[str, list[str]] = {}
    if previous_lockfile.is_file():
        previous_lines = previous_lockfile.read_text(encoding="utf-8").splitlines()
        previous_blocks = _validated_previous_blocks(previous_lines)
    marker_blocks = _governed_marker_blocks(previous_blocks, current_lines)

    filtered_lines = _without_governed_blocks(current_lines)
    insertion_index = _insertion_index(filtered_lines)
    restored_lines: list[str] = []
    for pin in GOVERNED_MARKER_PINS:
        package_name = _normalize_package_name(pin.package_name)
        restored_lines.extend(marker_blocks[package_name])

    next_lines = filtered_lines[:insertion_index] + restored_lines + filtered_lines[insertion_index:]
    if next_lines == current_lines:
        return False

    lockfile.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous",
        required=True,
        type=Path,
        help="pre-update lockfile snapshot (may be absent after make clean)",
    )
    parser.add_argument("--lockfile", required=True, type=Path, help="lockfile to repair")
    args = parser.parse_args(argv)

    restore_marker_pins(args.previous, args.lockfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
