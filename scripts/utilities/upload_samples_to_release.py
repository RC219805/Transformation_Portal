#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Release Sample Upload Automation

Automates the process of uploading sample images to GitHub Releases
and updating download_samples.py with verified URLs and SHA256 hashes.

Prerequisites:
1. GitHub CLI (gh) installed and authenticated
2. Sample images available in local directory
3. Write permissions to repository

Usage:
    # Upload all samples from a directory
    python scripts/utilities/upload_samples_to_release.py \
        --samples-dir data/sample_images/ \
        --release-tag samples-v1.0.0 \
        --repo RC219805/Transformation_Portal

    # Generate SHA256 hashes for existing files
    python scripts/utilities/upload_samples_to_release.py \
        --samples-dir data/sample_images/ \
        --generate-hashes-only

    # Update download_samples.py with computed URLs
    python scripts/utilities/upload_samples_to_release.py \
        --samples-dir data/sample_images/ \
        --release-tag samples-v1.0.0 \
        --repo RC219805/Transformation_Portal \
        --update-registry

Workflow:
1. Prepare sample images (downscale if needed for reasonable file sizes)
2. Run script with --generate-hashes-only to verify files
3. Run script with --upload to create release and upload files
4. Run script with --update-registry to update download_samples.py
5. Commit updated download_samples.py to repository

Author: Transformation Portal Team
License: MIT
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def generate_sample_manifest(samples_dir: Path) -> List[Dict]:
    """
    Generate manifest of sample files with SHA256 and size.

    Returns:
        List of sample metadata dicts
    """
    samples = []

    for file_path in sorted(samples_dir.glob("**/*")):
        if not file_path.is_file():
            continue

        # Skip hidden files and non-image files
        if file_path.name.startswith("."):
            continue

        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            continue

        sha256 = compute_sha256(file_path)
        size_mb = get_file_size_mb(file_path)

        sample = {
            "name": file_path.name,
            "path": str(file_path.relative_to(samples_dir)),
            "sha256": sha256,
            "size_mb": f"{size_mb:.1f}",
            "format": file_path.suffix.upper().replace(".", ""),
        }

        samples.append(sample)

    return samples


def create_github_release(repo: str, tag: str, title: str, description: str) -> bool:
    """
    Create GitHub release using gh CLI.

    Returns:
        True if successful
    """
    try:
        cmd = [
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--title",
            title,
            "--notes",
            description,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Created release: {tag}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create release: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ GitHub CLI (gh) not found. Install from https://cli.github.com/")
        return False


def upload_file_to_release(repo: str, tag: str, file_path: Path) -> Optional[str]:
    """
    Upload file to GitHub release.

    Returns:
        Download URL if successful, None otherwise
    """
    try:
        cmd = [
            "gh",
            "release",
            "upload",
            tag,
            str(file_path),
            "--repo",
            repo,
            "--clobber",  # Overwrite if exists
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Construct download URL
        url = f"https://github.com/{repo}/releases/download/{tag}/{file_path.name}"
        print(f"  ✓ Uploaded: {file_path.name}")
        return url
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to upload {file_path.name}: {e.stderr}")
        return None


def generate_registry_updates(samples: List[Dict], repo: str, tag: str) -> str:
    """
    Generate Python code snippets for updating SAMPLE_REGISTRY.

    Returns:
        Python code as string
    """
    lines = []
    lines.append("# Generated registry updates for download_samples.py")
    lines.append("# Copy these into SAMPLE_REGISTRY dict in scripts/download_samples.py\n")

    for sample in samples:
        url = f"https://github.com/{repo}/releases/download/{tag}/{sample['name']}"

        lines.append(f'"{sample["name"]}": {{')
        lines.append(f'    "url": "{url}",')
        lines.append(f'    "sha256": "{sample["sha256"]}",')
        lines.append(f'    "size": "{sample["size_mb"]} MB",')
        lines.append(f'    "path": "data/sample_images/{sample["path"]}",')
        lines.append('    "category": "full",  # Adjust as needed')
        lines.append(f'    "description": "Sample image - {sample["name"]}",')
        lines.append("},\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Upload sample images to GitHub Releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=True,
        help="Directory containing sample images",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default="samples-v1.0.0",
        help="GitHub release tag (default: samples-v1.0.0)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="RC219805/Transformation_Portal",
        help="GitHub repository (default: RC219805/Transformation_Portal)",
    )
    parser.add_argument(
        "--generate-hashes-only",
        action="store_true",
        help="Only generate SHA256 hashes without uploading",
    )
    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Generate registry update code after upload",
    )
    parser.add_argument(
        "--create-release",
        action="store_true",
        help="Create GitHub release (only needed once)",
    )

    args = parser.parse_args()

    # Validate samples directory
    if not args.samples_dir.exists():
        print(f"✗ Samples directory not found: {args.samples_dir}")
        return 1

    # Generate manifest
    print(f"Scanning {args.samples_dir} for sample images...\n")
    samples = generate_sample_manifest(args.samples_dir)

    if not samples:
        print("✗ No sample images found in directory")
        return 1

    print(f"Found {len(samples)} sample image(s):\n")
    for sample in samples:
        print(f"  • {sample['name']}")
        print(f"    SHA256: {sample['sha256']}")
        print(f"    Size: {sample['size_mb']} MB\n")

    # Generate hashes only
    if args.generate_hashes_only:
        manifest_path = args.samples_dir / "sample_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"\n✓ Manifest saved to: {manifest_path}")
        return 0

    # Create release if requested
    if args.create_release:
        title = f"Sample Images {args.release_tag}"
        description = (
            f"Sample images for Transformation Portal development and testing.\n\n"
            f"Total samples: {len(samples)}\n"
            f"Total size: {sum(float(s['size_mb']) for s in samples):.1f} MB"
        )

        if not create_github_release(args.repo, args.release_tag, title, description):
            return 1

    # Upload samples
    print(f"\nUploading {len(samples)} sample(s) to release {args.release_tag}...\n")

    uploaded_count = 0
    for sample in samples:
        file_path = args.samples_dir / sample["path"]
        url = upload_file_to_release(args.repo, args.release_tag, file_path)

        if url:
            sample["url"] = url
            uploaded_count += 1

    print(f"\n✓ Uploaded {uploaded_count}/{len(samples)} samples")

    # Generate registry updates
    if args.update_registry or uploaded_count > 0:
        registry_code = generate_registry_updates(samples, args.repo, args.release_tag)

        output_path = args.samples_dir / "registry_updates.py"
        with open(output_path, "w") as f:
            f.write(registry_code)

        print(f"\n✓ Registry updates saved to: {output_path}")
        print("\nNext steps:")
        print("1. Copy the generated code into scripts/download_samples.py")
        print("2. Update category and description fields as needed")
        print("3. Commit changes to repository")

    return 0


if __name__ == "__main__":
    sys.exit(main())
