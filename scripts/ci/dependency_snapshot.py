#!/usr/bin/env python3
"""Submit verified committed lock inventories, without resolving Python environments.

Flat locks establish package/version membership, not dependency edges or scope.
The detector/correlator pair intentionally supersedes the previous collector.
Local evidence errors fail closed; the workflow may retry advisory API failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.check_dependency_pinning import _iter_logical_lines  # noqa: E402
from scripts.validation.check_lock_ownership import GOVERNED_LOCK_FILES  # noqa: E402

LOCK_FILES = (
    *(f"requirements/{name}" for name in GOVERNED_LOCK_FILES),
    "config/fastvlm_runtime_requirements.txt",
    "requirements/locks/editorial-darwin-arm64-py312.txt",
)
DETECTOR_NAME = "Component Detection"
DETECTOR_VERSION = "1.0.0"
CORRELATOR = "submit-pypi"
MAX_FILE_BYTES = 2 * 1024 * 1024
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[A-Za-z0-9_.,\s-]+\])?==(?!=)([A-Za-z0-9][A-Za-z0-9.!+_-]*)")
# Canonical generated-lock PEP 440 subset: release, epoch, prerelease,
# post/dev release and local version. Unsupported spellings fail closed.
VERSION = re.compile(
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?"
)
HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}")
INDEX_OPTION = re.compile(r"--(?:extra-index-url|index-url|trusted-host)\s+\S+")
MARKER = re.compile(
    r"(?:python_version|python_full_version|os_name|sys_platform|platform_release|platform_system|"
    r"platform_version|platform_machine|platform_python_implementation|implementation_name|implementation_version)"
    r"\s*(?:==|!=|<=|>=|<|>|~=|in|not\s+in)\s*(['\"])[^'\"\\]*\1"
)


class EvidenceError(ValueError):
    """The payload cannot authorize a dependency graph submission."""


def read_bounded(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise EvidenceError(f"{path}: evidence must be a bounded regular file")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise EvidenceError(f"{path}: file exceeds evidence size limit")
    return data


def parse_pins(data: bytes, path: str) -> set[str]:
    """Parse exact pins and known lock syntax; retain all platform inventories."""
    purls: set[str] = set()
    names: set[str] = set()
    for line_number, logical_line in _iter_logical_lines(data.decode("utf-8")):
        line = logical_line.split("#", 1)[0].strip()
        if not line or INDEX_OPTION.fullmatch(line):
            continue
        # Environment markers describe the committed target; never evaluate
        # Darwin locks against the hosted Linux runner's environment.
        requirement, separator, marker = line.partition(";")
        if separator and not MARKER.fullmatch(marker.strip()):
            raise EvidenceError(f"{path}:{line_number}: unsupported marker syntax")
        tokens = requirement.split()
        hashes = [token for token in tokens if token.startswith("--hash")]
        if any(not HASH.fullmatch(token) for token in hashes):
            raise EvidenceError(f"{path}:{line_number}: invalid hash syntax")
        head = " ".join(token for token in tokens if token not in hashes)
        match = PIN.fullmatch(head)
        if not match:
            raise EvidenceError(f"{path}:{line_number}: unsupported lock syntax")
        if not VERSION.fullmatch(match[2]):
            raise EvidenceError(f"{path}:{line_number}: unsupported pinned version")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in names:
            raise EvidenceError(f"{path}:{line_number}: duplicate package {name}")
        names.add(name)
        purls.add(f"pkg:pypi/{name}@{quote(match[2], safe='')}")
    if not purls:
        raise EvidenceError(f"{path}: empty locked inventory")
    return purls


def context_from_env() -> dict[str, str]:
    context = {name: os.environ.get(f"GITHUB_{name.upper()}", "") for name in ("sha", "ref", "run_id", "repository")}
    if not re.fullmatch(r"[0-9a-f]{40}", context["sha"]):
        raise EvidenceError("GITHUB_SHA must be an exact commit")
    if not re.fullmatch(r"refs/(?:heads|tags|pull)/[^\s]+", context["ref"]):
        raise EvidenceError("GITHUB_REF must be a full Git ref")
    if not re.fullmatch(r"[0-9]+", context["run_id"]):
        raise EvidenceError("GITHUB_RUN_ID must be numeric")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", context["repository"]):
        raise EvidenceError("GITHUB_REPOSITORY must identify owner/repository")
    return context


def build_snapshot(repo_root: Path, context: dict[str, str], scanned: str) -> dict:
    """Bind every inventory to the exact commit and its checked-out bytes."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    if head != context["sha"]:
        raise EvidenceError("checkout HEAD does not match GITHUB_SHA")
    manifests = {}
    for path in LOCK_FILES:
        data = read_bounded(repo_root / path)
        committed = subprocess.check_output(["git", "cat-file", "blob", f"{head}:{path}"], cwd=repo_root)
        if data != committed:
            raise EvidenceError(f"{path}: working bytes differ from the submitted commit")
        purls = parse_pins(data, path)
        manifests[path] = {
            "name": path,
            "file": {"source_location": path},
            "metadata": {"lock_sha256": hashlib.sha256(data).hexdigest()},
            "resolved": {purl: {"package_url": purl} for purl in sorted(purls)},
        }
    return {
        "version": 0,
        "sha": context["sha"],
        "ref": context["ref"],
        "job": {"id": context["run_id"], "correlator": CORRELATOR},
        "detector": {
            "name": DETECTOR_NAME,
            "version": DETECTOR_VERSION,
            "url": f"https://github.com/{context['repository']}/blob/{context['sha']}/scripts/ci/dependency_snapshot.py",
        },
        "scanned": scanned,
        "manifests": manifests,
    }


def canonical_bytes(snapshot: dict) -> bytes:
    return (json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_payload(data: bytes, repo_root: Path, context: dict[str, str], digest: str = "") -> dict:
    """Validate the exact serialized bytes against all committed lock sets."""
    if len(data) > MAX_FILE_BYTES:
        raise EvidenceError("snapshot exceeds evidence size limit")
    if digest and hashlib.sha256(data).hexdigest() != digest:
        raise EvidenceError("snapshot SHA256 differs from the prepared payload")
    snapshot = json.loads(data, object_pairs_hook=unique_object)
    if not isinstance(snapshot, dict):
        raise EvidenceError("snapshot must be an object")
    scanned = snapshot.get("scanned")
    if not isinstance(scanned, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", scanned):
        raise EvidenceError("snapshot scanned timestamp must be RFC3339 UTC")
    try:
        datetime.strptime(scanned, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise EvidenceError("invalid snapshot scanned timestamp") from error
    expected = build_snapshot(repo_root, context, scanned)
    if snapshot != expected or data != canonical_bytes(expected):
        raise EvidenceError("snapshot does not exactly match commit identity and all governed lock inventories")
    return snapshot


def append_output(path: str | None, name: str, value: str) -> None:
    if path:
        with Path(path).open("a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


def validate_attempts(attempts: list[str]) -> None:
    """An invalid attempted payload cannot be hidden by another API success."""
    if len(attempts) != 2:
        raise EvidenceError("both submission attempt outcomes are required")
    for attempt in attempts:
        outcome, _, evidence_valid = attempt.partition(":")
        if outcome not in {"success", "failure", "skipped"}:
            raise EvidenceError("unexpected submission attempt outcome")
        if outcome != "skipped" and evidence_valid != "true":
            raise EvidenceError("submission attempt lacked validated local evidence")
    if attempts[0].startswith("skipped:"):
        raise EvidenceError("the first submission attempt did not execute")
    if attempts[1].startswith("skipped:") and not attempts[0].startswith("success:true"):
        raise EvidenceError("retry skipped without an accepted first submission")


def submit_payload(data: bytes, context: dict[str, str]) -> None:
    result = subprocess.run(
        ["gh", "api", "--method", "POST", f"repos/{context['repository']}/dependency-graph/snapshots", "--input", "-"],
        input=data,
        stdout=subprocess.PIPE,
        check=True,
    )
    receipt = json.loads(result.stdout)
    if (
        not isinstance(receipt, dict)
        or receipt.get("result") != "SUCCESS"
        or type(receipt.get("id")) is not int
        or receipt["id"] <= 0
    ):
        raise EvidenceError("GitHub did not return an accepted snapshot receipt")
    print(f"GitHub accepted dependency snapshot {receipt['id']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "submit"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--digest", default="")
    parser.add_argument("--attempt", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        context = context_from_env()
        if args.command == "build":
            scanned = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            data = canonical_bytes(build_snapshot(REPO_ROOT, context, scanned))
            if args.snapshot.is_symlink():
                raise EvidenceError("snapshot output must not be a symlink")
            args.snapshot.write_bytes(data)
        else:
            if not re.fullmatch(r"[0-9a-f]{64}", args.digest):
                raise EvidenceError("a prepared snapshot SHA256 is required")
            data = read_bounded(args.snapshot)
        snapshot = validate_payload(data, REPO_ROOT, context, args.digest)
        if args.command == "build":
            append_output(os.environ.get("GITHUB_OUTPUT"), "snapshot_sha256", hashlib.sha256(data).hexdigest())
            print(f"Verified commit {context['sha']} with {len(snapshot['manifests'])} governed lock inventories")
            for path, manifest in snapshot["manifests"].items():
                print(f"{path}: {len(manifest['resolved'])} pins; sha256={manifest['metadata']['lock_sha256']}")
        elif args.command == "submit":
            append_output(os.environ.get("GITHUB_OUTPUT"), "evidence_valid", "true")
            submit_payload(data, context)
        elif args.attempt:
            validate_attempts(args.attempt)
    except (EvidenceError, OSError, UnicodeError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"Dependency snapshot rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
