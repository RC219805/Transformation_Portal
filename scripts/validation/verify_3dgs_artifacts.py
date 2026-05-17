#!/usr/bin/env python3
"""
3DGS Artifact Attestation Verification Script

Validates 3DGS (3D Gaussian Splatting) artifact attestation entries in the
model lock manifest. Unlike HuggingFace model revisions which use commit SHAs,
3DGS artifacts require source-attestation verification including:

  - Code revision (from source repository)
  - Rasterizer backend revision
  - Optimizer behavior
  - Serialization format of splat parameters

This script enforces that all 3DGS artifact entries have verified attestation
data before being used in production workflows.

Usage:
    python scripts/validation/verify_3dgs_artifacts.py
    python scripts/validation/verify_3dgs_artifacts.py --strict
    python scripts/validation/verify_3dgs_artifacts.py --check-files

Exit Codes:
    0 - All attestation fields verified (or --check-files passes)
    1 - Warnings: pending attestation fields found (expected during setup)
    2 - Failures: manifest missing, malformed, or invalid attestation structure

ADR: ADR-032 (Dependency Pinning Strategy)
Related: config/model_lock_manifest.yaml (artifact_attestation section)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

# Attempt to import yaml; provide clear guidance if missing
try:
    import yaml
except ImportError:
    print("❌ Error: PyYAML not installed")
    print()
    print("Install with:")
    print("  pip install pyyaml")
    print()
    print("Or install full development dependencies:")
    print("  pip install -r requirements-dev.txt")
    sys.exit(2)


# Constants
MANIFEST_PATH = Path("config/model_lock_manifest.yaml")
PENDING_MARKERS = frozenset(
    {
        "PENDING_CANONICAL_URL",
        "PENDING_VERIFICATION",
        "PENDING",
        "TBD",
        "TODO",
    }
)


def load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Load and parse the model lock manifest.

    Returns:
        Parsed YAML mapping, or None if loading fails or the root is invalid.
    """
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        print()
        print("Expected location: config/model_lock_manifest.yaml")
        print("Ensure you are running from the repository root.")
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ Failed to parse manifest YAML: {e}")
        return None

    if not isinstance(parsed, dict):
        print(f"❌ Manifest root must be a mapping/dict, got: {type(parsed).__name__}")
        return None

    return parsed


def is_pending(value: Any) -> bool:
    """Check if a value is a pending/placeholder marker."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in PENDING_MARKERS or value.strip() == ""
    return False


def verify_attestation_entry(name: str, entry: dict[str, Any], strict: bool = False) -> tuple[list[str], list[str]]:
    """Verify a single artifact attestation entry.

    Args:
        name: The artifact backend name (e.g., "gaussian_splatting")
        entry: The attestation entry dict
        strict: If True, treat pending fields as errors instead of warnings

    Returns:
        Tuple of (errors, warnings) lists
    """
    errors: list[str] = []
    warnings: list[str] = []

    def add_issue(msg: str) -> None:
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Source-only attestations (git_release with no binary artifacts) are the
    # legitimate shape for upstreams like Inria graphdeco-inria/gaussian-splatting
    # that distribute source code, not weights. They pin a commit but carry
    # `artifacts: []` and `verification.method: source_commit`.
    source_type = entry.get("source_type")
    is_source_only = source_type == "git_release"

    # Check source_url
    source_url = entry.get("source_url")
    if is_pending(source_url):
        add_issue(f"[{name}] source_url is pending: {source_url!r}")
    elif not isinstance(source_url, str):
        errors.append(f"[{name}] source_url must be a string, got: {type(source_url).__name__}")

    # Check source_commit_or_tag
    source_commit = entry.get("source_commit_or_tag")
    if is_pending(source_commit):
        add_issue(f"[{name}] source_commit_or_tag is pending: {source_commit!r}")
    elif not isinstance(source_commit, str):
        errors.append(f"[{name}] source_commit_or_tag must be a string, got: {type(source_commit).__name__}")

    # Check artifacts list. For source-only (git_release) attestations, an
    # empty list is the canonical shape — no binary artifacts to verify.
    artifacts = entry.get("artifacts")
    if artifacts is None:
        errors.append(f"[{name}] 'artifacts' list is missing")
    elif not isinstance(artifacts, list):
        errors.append(f"[{name}] 'artifacts' must be a list, got: {type(artifacts).__name__}")
    elif len(artifacts) == 0:
        if not is_source_only:
            add_issue(f"[{name}] 'artifacts' list is empty")
    else:
        for i, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                errors.append(f"[{name}] artifacts[{i}] must be a dict, got: {type(artifact).__name__}")
                continue

            filename = artifact.get("filename", f"<unnamed artifact {i}>")

            # Check SHA256
            sha256 = artifact.get("sha256")
            if is_pending(sha256):
                add_issue(f"[{name}] {filename}: sha256 is pending: {sha256!r}")
            elif isinstance(sha256, str):
                # Validate SHA256 format (64 hex characters)
                if len(sha256) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha256):
                    errors.append(f"[{name}] {filename}: sha256 is not a valid 64-character hex string: {sha256!r}")
            else:
                errors.append(f"[{name}] {filename}: sha256 must be a string, got: {type(sha256).__name__}")

            # Check filesize_bytes
            filesize = artifact.get("filesize_bytes")
            if filesize is None:
                add_issue(f"[{name}] {filename}: filesize_bytes is null (not yet verified)")
            elif not isinstance(filesize, int):
                errors.append(f"[{name}] {filename}: filesize_bytes must be an integer, got: {type(filesize).__name__}")
            elif filesize <= 0:
                errors.append(f"[{name}] {filename}: filesize_bytes must be positive, got: {filesize}")

    # Check verification section. `source_commit` is the method used by
    # source-only attestations (git_release shape, no binary artifacts).
    verification = entry.get("verification")
    if verification is not None:
        if not isinstance(verification, dict):
            errors.append(f"[{name}] 'verification' must be a dict, got: {type(verification).__name__}")
        else:
            method = verification.get("method")
            valid_methods = {
                "sha256_only",
                "sha256+source_commit",
                "source_commit",
                "reproducibility_trial",
            }
            if method not in valid_methods:
                errors.append(f"[{name}] verification.method must be one of {valid_methods}, got: {method!r}")

    return errors, warnings


def compute_sha256(filepath: Path) -> str | None:
    """Compute SHA256 hash of a file.

    Returns:
        Hex digest string, or None if file cannot be read.
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except OSError as e:
        print(f"  ⚠️  Cannot read file: {filepath} ({e})")
        return None


def verify_actual_files(attestation: dict[str, dict[str, Any]], checkpoint_dir: Path) -> tuple[int, int, int]:
    """Verify actual checkpoint files against manifest attestation.

    Args:
        attestation: The artifact_attestation dict from manifest
        checkpoint_dir: Path to the checkpoints directory

    Returns:
        Tuple of (verified, mismatched, missing) counts
    """
    verified = 0
    mismatched = 0
    missing = 0

    for name, entry in attestation.items():
        artifacts = entry.get("artifacts", [])

        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue

            filename = artifact.get("filename")
            expected_sha = artifact.get("sha256")

            if not filename:
                continue

            filepath = checkpoint_dir / filename

            if not filepath.exists():
                print(f"  📦 {filename}: not present (skip)")
                missing += 1
                continue

            if is_pending(expected_sha):
                print(f"  📦 {filename}: expected hash is pending (skip verification)")
                print(f"      Actual SHA256: {compute_sha256(filepath) or 'N/A'}")
                missing += 1
                continue

            actual_sha = compute_sha256(filepath)
            if actual_sha is None:
                print(f"  📦 {filename}: ❌ cannot read file")
                mismatched += 1
                continue

            if actual_sha.lower() == expected_sha.lower():
                print(f"  📦 {filename}: ✅ VERIFIED")
                verified += 1
            else:
                print(f"  📦 {filename}: ❌ MISMATCH")
                print(f"      Expected: {expected_sha}")
                print(f"      Actual:   {actual_sha}")
                mismatched += 1

    return verified, mismatched, missing


def print_guidance() -> None:
    """Print guidance for resolving pending attestation."""
    print()
    print("━" * 70)
    print("  Guidance: Resolving 3DGS Artifact Attestation")
    print("━" * 70)
    print()
    print("3DGS artifacts require source-attestation verification, not just")
    print("HuggingFace model locks. The full revision tuple must be verified:")
    print()
    print("  1. Code revision      - Git commit/tag from source repository")
    print("  2. Rasterizer backend - diff-gaussian-rasterization version")
    print("  3. Optimizer behavior - Training hyperparameters and seed")
    print("  4. Serialization      - .pt/.pth format and tensor layout")
    print()
    print("Steps to resolve PENDING attestation:")
    print()
    print("  a) Identify canonical source:")
    print("     - Inria GraphDeco: https://github.com/graphdeco-inria/gaussian-splatting")
    print("     - Official releases, forks, or HuggingFace conversions")
    print()
    print("  b) Obtain checkpoints from verified source:")
    print("     - Download directly from release artifacts")
    print("     - Record exact URL, commit/tag, and download date")
    print()
    print("  c) Compute checksums:")
    print("     shasum -a 256 <checkpoint.pt>")
    print("     stat -f '%z' <checkpoint.pt>  # macOS")
    print("     stat --printf='%s' <checkpoint.pt>  # Linux")
    print()
    print("  d) Update config/model_lock_manifest.yaml:")
    print("     - Set source_url to canonical download location")
    print("     - Set source_commit_or_tag to verified revision")
    print("     - Set sha256 and filesize_bytes for each artifact")
    print()
    print("  e) Re-run this script to verify:")
    print("     python scripts/validation/verify_3dgs_artifacts.py --strict")
    print()
    print("━" * 70)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point.

    Returns:
        Exit code: 0 (success), 1 (warnings), 2 (errors/failures)
    """
    parser = argparse.ArgumentParser(
        description="Verify 3DGS artifact attestation in model lock manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic verification (warnings for pending fields)
    python scripts/validation/verify_3dgs_artifacts.py

    # Strict mode (pending fields are errors)
    python scripts/validation/verify_3dgs_artifacts.py --strict

    # Also verify actual checkpoint files
    python scripts/validation/verify_3dgs_artifacts.py --check-files

    # Specify custom checkpoint directory
    python scripts/validation/verify_3dgs_artifacts.py --check-files --checkpoint-dir ./models/3dgs
        """,
    )
    parser.add_argument("--strict", action="store_true", help="Treat pending/unverified fields as errors (exit code 2)")
    parser.add_argument(
        "--check-files", action="store_true", help="Also verify actual checkpoint files against manifest hashes"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoint files (default: checkpoints/)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_PATH, help=f"Path to model lock manifest (default: {MANIFEST_PATH})"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress guidance output on warnings")
    args = parser.parse_args(argv)

    print("━" * 70)
    print("  3DGS Artifact Attestation Verification")
    print("━" * 70)
    print()

    # Load manifest
    manifest = load_manifest(args.manifest)
    if manifest is None:
        return 2

    # Extract artifact_attestation section
    attestation = manifest.get("artifact_attestation")
    if attestation is None:
        print("❌ No 'artifact_attestation' section found in manifest")
        print()
        print("Expected structure in config/model_lock_manifest.yaml:")
        print()
        print("  artifact_attestation:")
        print("    gaussian_splatting:")
        print("      backend: inria_graphdeco")
        print('      source_url: "https://..."')
        print('      source_commit_or_tag: "v1.0"')
        print("      artifacts:")
        print('        - filename: "model.pt"')
        print('          sha256: "abc123..."')
        print("          filesize_bytes: 12345678")
        return 2

    if not isinstance(attestation, dict):
        print(f"❌ 'artifact_attestation' must be a dict, got: {type(attestation).__name__}")
        return 2

    if len(attestation) == 0:
        print("⚠️  'artifact_attestation' section is empty")
        print()
        print("No 3DGS artifacts to verify.")
        return 0

    # Verify each attestation entry
    all_errors: list[str] = []
    all_warnings: list[str] = []

    print(f"Verifying {len(attestation)} artifact attestation entry(ies)...")
    print()

    for name, entry in attestation.items():
        print(f"📋 {name}")

        if not isinstance(entry, dict):
            all_errors.append(f"[{name}] Entry must be a dict, got: {type(entry).__name__}")
            print(f"   ❌ Invalid entry type: {type(entry).__name__}")
            continue

        backend = entry.get("backend", "unknown")
        source_type = entry.get("source_type", "unknown")
        print(f"   Backend: {backend}")
        print(f"   Source type: {source_type}")

        errors, warnings = verify_attestation_entry(name, entry, strict=args.strict)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

        if errors:
            print(f"   ❌ {len(errors)} error(s)")
        if warnings:
            print(f"   ⚠️  {len(warnings)} warning(s)")
        if not errors and not warnings:
            print("   ✅ All attestation fields verified")

        print()

    # Optionally verify actual files
    file_verified = 0
    file_mismatched = 0
    file_missing = 0
    checkpoint_dir_missing = False

    if args.check_files:
        print("━" * 70)
        print("  Checkpoint File Verification")
        print("━" * 70)
        print()

        if not args.checkpoint_dir.exists():
            checkpoint_dir_missing = True
            all_errors.append(f"Checkpoint directory not found: {args.checkpoint_dir}")
            print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
            print()
            print("File verification cannot proceed. To verify files:")
            print(f"  1. Create directory: mkdir -p {args.checkpoint_dir}")
            print("  2. Place checkpoint files in the directory")
            print("  3. Re-run with --check-files")
        else:
            print(f"Scanning: {args.checkpoint_dir}")
            print()
            file_verified, file_mismatched, file_missing = verify_actual_files(attestation, args.checkpoint_dir)
            print()

    # Print summary
    print("━" * 70)
    print("  Summary")
    print("━" * 70)
    print()
    print(f"  Attestation entries:  {len(attestation)}")
    print(f"  Errors:               {len(all_errors)}")
    print(f"  Warnings:             {len(all_warnings)}")

    if args.check_files and not checkpoint_dir_missing:
        print()
        print(f"  Files verified:       {file_verified}")
        print(f"  Files mismatched:     {file_mismatched}")
        print(f"  Files missing/skip:   {file_missing}")

    print()

    # Print detailed issues
    if all_errors:
        print("Errors:")
        for err in all_errors:
            print(f"  ❌ {err}")
        print()

    if all_warnings:
        print("Warnings:")
        for warn in all_warnings:
            print(f"  ⚠️  {warn}")
        print()

    # Determine exit code
    if all_errors or (args.check_files and (file_mismatched > 0 or file_missing > 0)):
        print("━" * 70)
        print("❌ VERIFICATION FAILED")
        print("━" * 70)
        if not args.quiet:
            print_guidance()
        return 2

    if all_warnings:
        print("━" * 70)
        print("⚠️  VERIFICATION INCOMPLETE (pending attestation)")
        print("━" * 70)
        if not args.quiet:
            print_guidance()
        return 1

    print("━" * 70)
    print("✅ ALL 3DGS ARTIFACT ATTESTATION VERIFIED")
    print("━" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
