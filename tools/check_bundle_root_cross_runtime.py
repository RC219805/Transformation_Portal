#!/usr/bin/env python3
"""
Verify bundle-root digest parity across two Python runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

EXIT_RUNTIME_FAILURE = 31
EXIT_ROOT_MISMATCH = 32
DEFAULT_EXPECTED_ROOT = "47c09af843470b891e8c33d614fb5b4a4399218fdc7b8461f5c6b11d1fa000ce"


def _write_fixture_bundle(bundle_dir: Path) -> Path:
    roots_path = bundle_dir / "merkle_roots.json"
    roots_path.write_text(
        json.dumps(
            {
                "hash_algorithm": "sha256",
                "leaf_hash_algorithm": "sha256",
                "leaf_format_version": "1",
                "leaf_format": "v1",
                "tree_method_version": "1",
                "tree_method": "duplicate_last",
                "partitions": [],
                "global": {"leaf_count": 3, "root_sha256": "0" * 64},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    hash_manifest_path = bundle_dir / "hash_manifest.csv.gz"
    hash_manifest_path.write_bytes(
        b"# hash_algorithm=sha256\n"
        b"origin_drive,partition,relpath,filesize_bytes,sha256,hash_status,error\n"
        b"driveA,partA,a.jpg,5,abc,ok,\n"
    )

    hash_summary_path = bundle_dir / "hash_summary.json"
    hash_summary_path.write_text(
        json.dumps(
            {
                "hash_algorithm": "sha256",
                "hash_manifest_schema_version": "1",
                "rows_total": 1,
                "hashed_ok": 1,
                "missing": 0,
                "unreadable": 0,
                "skipped": 0,
                "total_bytes_hashed": 5,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    signature_path = bundle_dir / "merkle_roots.sig.json"
    signature_path.write_text(
        json.dumps(
            {
                "envelope_version": "1",
                "signature_algorithm": "ed25519",
                "artifact_digest_algorithm": "sha256",
                "signed_artifact": "merkle_roots.json",
                "signed_artifact_sha256": hashlib.sha256(roots_path.read_bytes()).hexdigest(),
                "signature_base64": "c2ln",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    timestamp_path = bundle_dir / "merkle_roots.sig.tsr"
    timestamp_path.write_bytes(b"\x30\x03\x30\x01\x00")

    return bundle_dir / "evidence_bundle_manifest.json"


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


def _compute_bundle_root(*, python_executable: str, project_root: Path, manifest_path: Path) -> str:
    tools_dir = project_root / "tools"
    generate_tool = tools_dir / "generate_evidence_bundle_manifest.py"
    compute_tool = tools_dir / "compute_bundle_root.py"

    roots_path = manifest_path.parent / "merkle_roots.json"
    hash_manifest_path = manifest_path.parent / "hash_manifest.csv.gz"
    hash_summary_path = manifest_path.parent / "hash_summary.json"
    signature_path = manifest_path.parent / "merkle_roots.sig.json"
    timestamp_path = manifest_path.parent / "merkle_roots.sig.tsr"

    _run_checked(
        [
            python_executable,
            str(generate_tool),
            "--roots",
            str(roots_path),
            "--hash-manifest",
            str(hash_manifest_path),
            "--hash-summary",
            str(hash_summary_path),
            "--signature",
            str(signature_path),
            "--timestamp-target",
            "signature",
            "--timestamp",
            str(timestamp_path),
            "--out",
            str(manifest_path),
        ],
        cwd=project_root,
    )
    root = _run_checked(
        [
            python_executable,
            str(compute_tool),
            "--bundle-manifest",
            str(manifest_path),
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
        manifest_path = _write_fixture_bundle(Path(tmp))
        try:
            root_a = _compute_bundle_root(
                python_executable=args.python_a,
                project_root=project_root,
                manifest_path=manifest_path,
            )
            root_b = _compute_bundle_root(
                python_executable=args.python_b,
                project_root=project_root,
                manifest_path=manifest_path,
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
