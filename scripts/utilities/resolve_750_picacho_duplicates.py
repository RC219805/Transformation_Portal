#!/usr/bin/env python3
"""
Resolution script for 750 Picacho Lane duplicate file issue.

This script:
1. Identifies duplicate source files (different versions of same scene)
2. Determines the canonical version to use
3. Consolidates processing to use only canonical sources
4. Creates a clean batch processing manifest
5. Removes or archives duplicate outputs

Author: Transformation Portal
Date: 2025-11-08
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class DuplicateResolver:
    """Resolves duplicate file issues in 750 Picacho project."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.source_dirs = {
            "exr": self.base_dir / "16-Bit_EXRs",
            "tif": self.base_dir / "TIFFs" / "_TIFFs",
            "lightroom": self.base_dir / "LightRoom_TiFFs",
        }
        self.output_dirs = ["Maximum_Quality_Final", "Phase3_Refined", "Processed_Output", "Ultimate_Quality"]

    def find_duplicates(self) -> Dict[str, List[Path]]:
        """Find all duplicate source files grouped by base name."""
        duplicates = {}

        for source_type, source_dir in self.source_dirs.items():
            if not source_dir.exists():
                continue

            for file_path in source_dir.glob("*"):
                if not file_path.is_file():
                    continue

                # Extract base name (remove version suffixes and extensions)
                name = file_path.stem
                # Remove version indicators like "2-" prefix or "-2" suffix
                base_name = name.replace("2-", "").replace("-2", "")

                if base_name not in duplicates:
                    duplicates[base_name] = []

                duplicates[base_name].append(file_path)

        # Filter to only actual duplicates
        return {k: v for k, v in duplicates.items() if len(v) > 1}

    def determine_canonical_version(self, files: List[Path]) -> Tuple[Path, str]:
        """
        Determine which version should be the canonical source.

        Priority:
        1. Newest file (most recent modification time)
        2. Largest file size (more data retained)
        3. Simplest filename (no version suffix)

        Returns:
            Tuple of (canonical_file, reason)
        """
        if len(files) == 1:
            return files[0], "only_version"

        # Check modification times
        files_with_stats = [(f, f.stat()) for f in files]

        # Sort by: newer first, larger first, simpler name first
        def sort_key(item):
            file_path, stat = item
            # Prefer files without version numbers in name
            has_version = any(x in file_path.stem for x in ["2-", "-2", "_v2", "_V2"])
            return (
                -stat.st_mtime,  # Newer first (negative for descending)
                -stat.st_size,  # Larger first
                has_version,  # No version suffix preferred
                len(file_path.stem),  # Shorter name preferred
            )

        sorted_files = sorted(files_with_stats, key=sort_key)
        canonical = sorted_files[0][0]

        # Determine reason
        reasons = []
        if sorted_files[0][1].st_mtime > sorted_files[1][1].st_mtime:
            reasons.append("newest_modification")
        if sorted_files[0][1].st_size > sorted_files[1][1].st_size:
            reasons.append("largest_size")
        if not any(x in canonical.stem for x in ["2-", "-2"]):
            reasons.append("clean_filename")

        return canonical, "_and_".join(reasons) if reasons else "first_in_list"

    def create_canonical_manifest(self) -> Dict[str, Dict]:
        """Create manifest of canonical source files for batch processing."""
        duplicates = self.find_duplicates()
        manifest = {
            "created": datetime.now().isoformat(),
            "base_directory": str(self.base_dir),
            "canonical_sources": {},
            "duplicates_found": {},
            "resolution_summary": {},
        }

        # Find all unique scenes
        all_scenes = set()
        for source_dir in self.source_dirs.values():
            if source_dir.exists():
                for f in source_dir.glob("*"):
                    if f.is_file():
                        base_name = f.stem.replace("2-", "").replace("-2", "")
                        all_scenes.add(base_name)

        # For each scene, determine canonical version
        for scene in sorted(all_scenes):
            # Find all versions of this scene
            versions = []
            for source_dir in self.source_dirs.values():
                if not source_dir.exists():
                    continue
                # Look for exact match and versioned variants
                patterns = [scene, f"2-{scene}", f"{scene}-2", f"2-{scene}-2"]
                for pattern in patterns:
                    for ext in [".exr", ".ti", ".tiff"]:
                        candidate = source_dir / f"{pattern}{ext}"
                        if candidate.exists():
                            versions.append(candidate)

            if versions:
                canonical, reason = self.determine_canonical_version(versions)
                manifest["canonical_sources"][scene] = {
                    "path": str(canonical),
                    "size_mb": canonical.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(canonical.stat().st_mtime).isoformat(),
                    "selection_reason": reason,
                }

                if len(versions) > 1:
                    manifest["duplicates_found"][scene] = {
                        "canonical": str(canonical),
                        "alternates": [str(v) for v in versions if v != canonical],
                        "count": len(versions),
                    }

        manifest["resolution_summary"] = {
            "total_scenes": len(all_scenes),
            "scenes_with_duplicates": len(manifest["duplicates_found"]),
            "canonical_sources_identified": len(manifest["canonical_sources"]),
        }

        return manifest

    def cleanup_duplicate_outputs(self, manifest: Dict, dry_run: bool = True) -> Dict:
        """
        Clean up output files from duplicate/non-canonical sources.

        Args:
            manifest: Canonical source manifest
            dry_run: If True, only report what would be done

        Returns:
            Summary of cleanup actions
        """
        cleanup_summary = {"files_to_archive": [], "files_to_keep": [], "actions_taken": []}

        duplicates = manifest.get("duplicates_found", {})

        for scene, dup_info in duplicates.items():
            # Find output files from non-canonical sources
            for output_dir in self.output_dirs:
                output_path = self.base_dir / output_dir
                if not output_path.exists():
                    continue

                # Look for files matching non-canonical patterns
                for alt_source in dup_info["alternates"]:
                    alt_stem = Path(alt_source).stem
                    # Find all outputs derived from this non-canonical source
                    for output_file in output_path.glob(f"{alt_stem}*"):
                        if dry_run:
                            cleanup_summary["files_to_archive"].append(str(output_file))
                        else:
                            # Archive to subdirectory
                            archive_dir = output_path / "_archived_duplicates"
                            archive_dir.mkdir(exist_ok=True)
                            archived_path = archive_dir / output_file.name
                            output_file.rename(archived_path)
                            cleanup_summary["actions_taken"].append(
                                {"action": "archived", "from": str(output_file), "to": str(archived_path)}
                            )

        return cleanup_summary

    def generate_processing_list(self, manifest: Dict) -> List[Path]:
        """Generate clean list of files for batch processing."""
        return [Path(info["path"]) for scene, info in sorted(manifest["canonical_sources"].items())]


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Resolve duplicate file issues in 750 Picacho project")
    parser.add_argument("base_dir", type=Path, help="Base directory containing 750 Picacho files")
    parser.add_argument("--cleanup", action="store_true", help="Actually cleanup duplicates (default is dry-run)")
    parser.add_argument("--output-manifest", type=Path, default=None, help="Path to save manifest JSON")

    args = parser.parse_args()

    # Validate base directory
    if not args.base_dir.exists():
        print(f"❌ Base directory not found: {args.base_dir}")
        sys.exit(1)

    print("=" * 80)
    print("750 Picacho Lane - Duplicate Resolution Tool")
    print("=" * 80)
    print()

    # Initialize resolver
    resolver = DuplicateResolver(args.base_dir)

    # Create manifest
    print("📋 Analyzing source files...")
    manifest = resolver.create_canonical_manifest()

    print("\n✅ Analysis complete:")
    print(f"   Total scenes: {manifest['resolution_summary']['total_scenes']}")
    print(f"   Scenes with duplicates: {manifest['resolution_summary']['scenes_with_duplicates']}")
    print(f"   Canonical sources identified: {manifest['resolution_summary']['canonical_sources_identified']}")

    # Show duplicates
    if manifest["duplicates_found"]:
        print("\n🔍 Duplicate sources found:")
        for scene, dup_info in manifest["duplicates_found"].items():
            print(f"\n   Scene: {scene}")
            print(f"   ✓ Canonical: {Path(dup_info['canonical']).name}")
            print(f"   ⚠ Alternates ({len(dup_info['alternates'])}):")
            for alt in dup_info["alternates"]:
                print(f"      - {Path(alt).name}")

    # Cleanup analysis
    print("\n🧹 Analyzing output cleanup...")
    cleanup_summary = resolver.cleanup_duplicate_outputs(manifest, dry_run=not args.cleanup)

    if cleanup_summary["files_to_archive"]:
        print(f"\n   Files to archive: {len(cleanup_summary['files_to_archive'])}")
        if not args.cleanup:
            print("   (Dry run - use --cleanup to actually archive)")
        else:
            print(f"   ✓ Archived {len(cleanup_summary['actions_taken'])} files")

    # Save manifest
    if args.output_manifest:
        with open(args.output_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n💾 Manifest saved to: {args.output_manifest}")
    else:
        # Default location
        default_manifest = args.base_dir / "canonical_sources_manifest.json"
        with open(default_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n💾 Manifest saved to: {default_manifest}")

    # Generate processing list
    processing_list = resolver.generate_processing_list(manifest)
    processing_list_file = args.base_dir / "batch_processing_list.txt"
    with open(processing_list_file, "w") as f:
        for file_path in processing_list:
            f.write(f"{file_path}\n")
    print(f"📝 Processing list saved to: {processing_list_file}")

    print("\n✨ Resolution complete!")
    print("\nNext steps:")
    print("1. Review the canonical_sources_manifest.json")
    print("2. Use batch_processing_list.txt for clean pipeline execution")
    if not args.cleanup:
        print("3. Run with --cleanup to archive duplicate outputs")
    print()


if __name__ == "__main__":
    main()
