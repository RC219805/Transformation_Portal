#!/usr/bin/env python3
"""
Migration helper script for Transformation Portal refactoring.

This script helps update import statements in user code to use the new
modular structure introduced in v0.1.0.

Usage:
    python scripts/migrate_imports.py [file_or_directory]

Examples:
    # Update a single file
    python scripts/migrate_imports.py my_script.py

    # Update all Python files in a directory
    python scripts/migrate_imports.py my_project/

    # Dry run (show changes without applying)
    python scripts/migrate_imports.py --dry-run my_script.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Processors
    'from material_response import': 'from transformation_portal.processors.material_response.core import',
    'import material_response': 'from transformation_portal.processors import material_response',
    'from material_response_optimizer import': 'from transformation_portal.processors.material_response.optimizer import',
    'from luxury_video_master_grader import': 'from transformation_portal.processors.luxury_video_master_grader import',

    # Pipelines
    'from lux_render_pipeline import': 'from transformation_portal.pipelines.lux_render_pipeline import',
    'from depth_tools import': 'from transformation_portal.pipelines.depth_tools import',
    'from dreaming_pipeline import': 'from transformation_portal.pipelines.dreaming_pipeline import',

    # Enhancers
    'from enhance_aerial import': 'from transformation_portal.enhancers.enhance_aerial import',
    'from enhance_pool_aerial import': 'from transformation_portal.enhancers.enhance_pool_aerial import',
    'from board_material_aerial_enhancer import': 'from transformation_portal.enhancers.board_material_aerial_enhancer import',
    'from update_enhance_aerial import': 'from transformation_portal.enhancers.update_enhance_aerial import',

    # Analyzers
    'from decision_decay_dashboard import': 'from transformation_portal.analyzers.decision_decay_dashboard import',
    'from codebase_philosophy_auditor import': 'from transformation_portal.analyzers.codebase_philosophy_auditor import',
    'from parse_workflows import': 'from transformation_portal.analyzers.parse_workflows import',

    # Rendering
    'from coastal_estate_render import': 'from transformation_portal.rendering.coastal_estate_render import',
    'from golden_hour_courtyard_workflow import': 'from transformation_portal.rendering.golden_hour_courtyard_workflow import',
    'from process_renderings_750 import': 'from transformation_portal.rendering.process_renderings_750 import',

    # Utils
    'from color_science import': 'from transformation_portal.utils.color_science import',
    'from helpers import': 'from transformation_portal.utils.helpers import',
}


def migrate_imports(content: str) -> Tuple[str, List[str]]:
    """
    Migrate import statements in content to new structure.

    Args:
        content: File content as string

    Returns:
        Tuple of (updated_content, list_of_changes)
    """
    changes = []
    updated = content

    for old_import, new_import in IMPORT_MAPPINGS.items():
        if old_import in updated:
            count = updated.count(old_import)
            updated = updated.replace(old_import, new_import)
            changes.append(f"  - Replaced '{old_import}' with '{new_import}' ({count} occurrence(s))")

    return updated, changes


def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Process a single Python file.

    Args:
        file_path: Path to Python file
        dry_run: If True, show changes without writing

    Returns:
        True if changes were made, False otherwise
    """
    try:
        content = file_path.read_text()
        updated, changes = migrate_imports(content)

        if changes:
            print(f"\n{file_path}:")
            for change in changes:
                print(change)

            if not dry_run:
                file_path.write_text(updated)
                print("  ✓ File updated")
            else:
                print("  (dry run - no changes written)")
            return True
        return False
    except Exception as e:
        print(f"\n✗ Error processing {file_path}: {e}", file=sys.stderr)
        return False


def process_directory(dir_path: Path, dry_run: bool = False) -> int:
    """
    Process all Python files in a directory recursively.

    Args:
        dir_path: Path to directory
        dry_run: If True, show changes without writing

    Returns:
        Number of files changed
    """
    python_files = list(dir_path.rglob("*.py"))
    changed_count = 0

    print(f"Found {len(python_files)} Python files in {dir_path}")

    for py_file in python_files:
        if process_file(py_file, dry_run):
            changed_count += 1

    return changed_count


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate import statements to new Transformation Portal structure"
    )
    parser.add_argument(
        "path",
        help="File or directory to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without applying them"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"✗ Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("Transformation Portal Import Migration Tool")
    print("=" * 80)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")

    if path.is_file():
        if path.suffix == ".py":
            changed = process_file(path, args.dry_run)
            if changed:
                print("\n✓ Migration complete - 1 file updated")
            else:
                print("\n✓ No changes needed")
        else:
            print(f"✗ Error: {path} is not a Python file", file=sys.stderr)
            sys.exit(1)
    elif path.is_dir():
        changed_count = process_directory(path, args.dry_run)
        print(f"\n✓ Migration complete - {changed_count} file(s) updated")
    else:
        print(f"✗ Error: {path} is neither a file nor directory", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and changed_count > 0:
        print("\n⚠️  Remember to:")
        print("  1. Test your code after migration")
        print("  2. Update your requirements to use: transformation-portal>=0.1.0")
        print("  3. Run your test suite to verify everything works")


if __name__ == "__main__":
    main()
