#!/usr/bin/env python3
"""
Workspace cleanup utility for Transformation_Portal.

Safely removes local-only artifacts:
- logs
- cache directories
- benchmark/output folders
- temporary reports

Default mode is DRY-RUN. Use --apply to actually delete.
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]


# Directories to exclude from cleanup traversal
# These are heavy/critical dirs that should never be cleaned
EXCLUDE_DIRS: Set[str] = {".git", ".venv", "weights"}


FILE_PATTERNS: List[str] = [
    "*.log",
    "safety-report.json",
    "lux_depth_v2_changes.txt",
]

DIR_PATTERNS: List[str] = [
    # Root-level patterns
    "output_*",
    "benchmarks_*",
    "benchmarks_*_*",
    "temp_greatroom",
    "build",
    "dist",
    # Recursive patterns
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.hypothesis",
    "**/*.egg-info",
]

RECURSIVE_FILE_PATTERNS: List[str] = [
    "**/*.pyc",
    "**/*.pyo",
    "**/.DS_Store",
]


def is_excluded(path: Path) -> bool:
    """
    Exclude heavy / non-project dirs from cleanup traversal.
    
    Excludes:
    - .git (version control)
    - .venv (virtual environment - not a workspace artifact)
    - weights (model weights - large and intentional)
    """
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True  # already excluded by ROOT guard anyway

    parts = set(rel.parts)
    return bool(parts & EXCLUDE_DIRS)


def get_tracked_files() -> Set[Path]:
    """
    Return a set of tracked file paths under ROOT.

    If git is unavailable (or this isn't a git repo), returns an empty set and
    cleanup behaves as before (pattern-only).
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--full-name"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return set()

    if result.returncode != 0:
        return set()

    tracked: Set[Path] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tracked.add((ROOT / line).resolve())
        except Exception:
            continue
    return tracked


def iter_paths() -> Iterable[Path]:
    # Root-level files
    for pattern in FILE_PATTERNS:
        yield from ROOT.glob(pattern)

    # Root-level + recursive dirs/files
    for pattern in DIR_PATTERNS:
        yield from ROOT.glob(pattern)

    for pattern in RECURSIVE_FILE_PATTERNS:
        yield from ROOT.glob(pattern)


def classify_path(p: Path) -> str:
    if p.is_dir():
        return "DIR "
    if p.is_file():
        return "FILE"
    return "UNK "


def cleanup(apply: bool, verbose: bool) -> None:
    tracked_files = get_tracked_files()
    
    seen = set()
    to_delete: List[Path] = []
    skipped_tracked = 0
    skipped_excluded = 0

    for p in iter_paths():
        try:
            p = p.resolve()
        except Exception:
            continue
        if not p.exists():
            continue
        if p in seen:
            continue
        # Skip paths outside the repository (e.g., symlinks to external storage)
        try:
            p.relative_to(ROOT)
        except ValueError:
            continue
        
        # Skip excluded directories (.venv, weights, .git)
        if is_excluded(p):
            skipped_excluded += 1
            if verbose:
                rel = p.relative_to(ROOT)
                print(f"  SKIP excluded: {rel}")
            continue
        
        # Protect tracked files
        if p.is_file() and p in tracked_files:
            skipped_tracked += 1
            if verbose:
                rel = p.relative_to(ROOT)
                print(f"  SKIP tracked file: {rel}")
            continue
        
        seen.add(p)
        to_delete.append(p)

    to_delete.sort()

    if not to_delete:
        print("Nothing to clean. Workspace already tidy.")
        if skipped_tracked > 0 or skipped_excluded > 0:
            print(f"(Skipped {skipped_tracked} tracked, {skipped_excluded} excluded)")
        return

    print(f"Found {len(to_delete)} paths to remove:")
    if skipped_tracked > 0 or skipped_excluded > 0:
        print(f"(Skipped {skipped_tracked} tracked, {skipped_excluded} excluded)")
    for p in to_delete:
        rel = p.relative_to(ROOT)
        print(f"  {classify_path(p)}  {rel}")

    if not apply:
        print("\nDRY-RUN ONLY. No files were deleted.")
        print("Re-run with --apply to perform the cleanup.")
        return

    print("\nApplying cleanup...")
    for p in to_delete:
        rel = p.relative_to(ROOT)
        if p.is_dir():
            if verbose:
                print(f"  rm -rf {rel}")
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            if verbose:
                print(f"  rm -f {rel}")
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    print("\nCleanup complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean local-only artifacts from the Transformation_Portal workspace."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the files/directories (otherwise dry-run).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print individual delete operations.",
    )
    args = parser.parse_args()

    print(f"Workspace root: {ROOT}")
    cleanup(apply=args.apply, verbose=args.verbose)


if __name__ == "__main__":
    main()
