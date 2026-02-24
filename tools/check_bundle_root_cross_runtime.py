#!/usr/bin/env python3
"""
Verify bundle-root digest parity across two Python runtimes.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from bundle_root_fixture import EXPECTED_BUNDLE_ROOT_SHA256, write_bundle_fixture_artifacts

EXIT_RUNTIME_FAILURE = 31
EXIT_ROOT_MISMATCH = 32
DEFAULT_EXPECTED_ROOT = EXPECTED_BUNDLE_ROOT_SHA256


def _run_checked(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _compute_bundle_root(
    *,
    python_executable: str,
    project_root: Path,
    artifacts: dict[str, Path],
) -> str:
    tools_dir = project_root / "tools"
    generate_tool = tools_dir / "generate_evidence_bundle_manifest.py"
    compute_tool = tools_dir / "compute_bundle_root.py"

    _run_checked(
        [
            python_executable,
            str(generate_tool),
            "--roots",
            str(artifacts["roots"]),
            "--hash-manifest",
            str(artifacts["hash_manifest"]),
            "--hash-summary",
            str(artifacts["hash_summary"]),
            "--signature",
            str(artifacts["signature"]),
            "--timestamp-target",
            "signature",
            "--timestamp",
            str(artifacts["timestamp"]),
            "--out",
            str(artifacts["out"]),
        ],
        cwd=project_root,
    )
    root = _run_checked(
        [
            python_executable,
            str(compute_tool),
            "--bundle-manifest",
            str(artifacts["out"]),
        ],
        cwd=project_root,
    )
    return root.splitlines()[-1].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-a", required=True, help="First Python interpreter path")
    parser.add_argument("--python-b", required=True, help="Second Python interpreter path")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root containing tools/",
    )
    parser.add_argument(
        "--expected-root",
        default=DEFAULT_EXPECTED_ROOT,
        help="Expected deterministic root digest",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = write_bundle_fixture_artifacts(Path(tmp), timestamp_target="signature")
        try:
            root_a = _compute_bundle_root(
                python_executable=args.python_a,
                project_root=project_root,
                artifacts=artifacts,
            )
            root_b = _compute_bundle_root(
                python_executable=args.python_b,
                project_root=project_root,
                artifacts=artifacts,
            )
        except RuntimeError as exc:
            print(f"Cross-runtime parity check failed: {exc}")
            return EXIT_RUNTIME_FAILURE

    print(f"python_a root: {root_a}")
    print(f"python_b root: {root_b}")
    if root_a != root_b:
        print("Cross-runtime parity check failed: root mismatch between interpreters")
        return EXIT_ROOT_MISMATCH
    if root_a != args.expected_root:
        print("Cross-runtime parity check failed: computed root does not match expected golden value")
        return EXIT_ROOT_MISMATCH

    print("Cross-runtime parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
