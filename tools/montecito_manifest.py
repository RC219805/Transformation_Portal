"""Utilities for generating file manifests for Montecito batch outputs.

This module provides a small CLI that mirrors the ad-hoc shell pipeline
previously embedded in the local ``.bash_history``.  It walks a directory,
collects file sizes and MD5 checksums, and writes a CSV manifest that matches
what the one-off shell command produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable, Tuple


def iter_files(root: Path) -> Iterable[Tuple[Path, int, str]]:
    """Yield ``(relative_path, size_bytes, md5_hash)`` for each file under ``root``.

    The MD5 hash matches the historical ``md5 -q`` invocation that appeared in
    the shell pipeline, ensuring drop-in compatibility with existing tooling.
    """

    for path in sorted(root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            md5_hash = _md5(path)
            yield path.relative_to(root), size, md5_hash


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, destination: Path) -> None:
    """Write a CSV manifest for ``root`` to ``destination``.

    The CSV contains headers ``filename,bytes,md5`` and uses UTF-8 encoding.
    Paths are stored relative to ``root`` to keep the manifest portable.
    """

    rows = list(iter_files(root))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "bytes", "md5"])
        writer.writerows((str(path), size, md5) for path, size, md5 in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV manifest (filename, bytes, md5) for an output directory."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to scan for files."
    )
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path("manifest.csv"),
        help="Where to write the manifest (defaults to ./manifest.csv)."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root: Path = args.root.expanduser().resolve()
    destination: Path = args.destination.expanduser().resolve()

    if not root.exists():
        raise SystemExit(
            f"Root path '{root}' does not exist. Please check the path for typos and verify that you have the necessary permissions."
        )
    if not root.is_dir():
        raise SystemExit(
            f"Root path '{root}' exists but is not a directory. Please check that the path is correct and that you have the necessary permissions."
        )

    write_manifest(root, destination)


if __name__ == "__main__":
    main()
