#!/usr/bin/env python3
"""Repo-owned transport adapter for the pre-commit-hooks v4.5.0 policy.

Policy bodies are retained from check_added_large_files.py. Only Git attribute
stdout becomes a UTF-8 TemporaryFile; complete response validation precedes any
LFS removal. The existing remote hook environment supplies pre_commit_hooks.

Copyright (c) 2014 pre-commit dev team: Anthony Sottile, Ken Struys

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from tempfile import TemporaryFile
from typing import Sequence

from pre_commit_hooks.util import added_files


def filter_lfs_files(filenames: set[str]) -> None:
    """Retain policy while avoiding simultaneous stdin/stdout data pipes."""
    if not filenames:
        return
    with TemporaryFile(mode="w+", encoding="utf-8", newline="") as output:
        subprocess.run(
            ("git", "check-attr", "filter", "-z", "--stdin"),
            stdout=output,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            check=True,
            input="\0".join(filenames),
        )
        output.seek(0)
        response = output.read()
    if not response.endswith("\0"):
        raise ValueError("Truncated Git filter attribute response")
    fields = response[:-1].split("\0")
    if len(fields) != 3 * len(filenames):
        raise ValueError("Incomplete Git filter attribute response")
    seen, lfs = set(), set()
    for i in range(0, len(fields), 3):
        filename, attribute, value = fields[i : i + 3]
        if attribute != "filter" or filename not in filenames or filename in seen:
            raise ValueError("Mismatched Git filter attribute response")
        seen.add(filename)
        if value == "lfs":
            lfs.add(filename)
    if seen != filenames:
        raise ValueError("Incomplete Git filter attribute filename coverage")
    filenames.difference_update(lfs)


def find_large_added_files(
    filenames: Sequence[str],
    maxkb: int,
    *,
    enforce_all: bool = False,
) -> int:
    # Find all added files that are also in the list of files pre-commit tells
    # us about
    retv = 0
    filenames_filtered = set(filenames)
    filter_lfs_files(filenames_filtered)

    if not enforce_all:
        filenames_filtered &= added_files()

    for filename in filenames_filtered:
        kb = math.ceil(os.stat(filename).st_size / 1024)
        if kb > maxkb:
            print(f"{filename} ({kb} KB) exceeds {maxkb} KB.")
            retv = 1

    return retv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Filenames pre-commit believes are changed.",
    )
    parser.add_argument(
        "--enforce-all",
        action="store_true",
        help="Enforce all files are checked, not just staged files.",
    )
    parser.add_argument(
        "--maxkb",
        type=int,
        default=500,
        help="Maximum allowable KB for added files",
    )
    args = parser.parse_args(argv)

    return find_large_added_files(
        args.filenames,
        args.maxkb,
        enforce_all=args.enforce_all,
    )


if __name__ == "__main__":
    raise SystemExit(main())
