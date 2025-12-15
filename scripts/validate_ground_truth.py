#!/usr/bin/env python3
"""Validate ground truth JSON against schema for water detection dataset.

Usage:
    python scripts/validate_ground_truth.py data/water_v0/ground_truth.json

Returns:
    0 on success, 1 on validation failure
"""

import argparse
import json
import sys
from pathlib import Path


def load_schema(schema_path: Path) -> dict:
    """Load JSON schema from file."""
    with open(schema_path) as f:
        return json.load(f)


def load_ground_truth(gt_path: Path) -> dict:
    """Load ground truth JSON."""
    with open(gt_path) as f:
        return json.load(f)


def validate_schema(data: dict, schema: dict) -> list[str]:
    """Validate data against schema (manual validation).
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Check required top-level fields
    if "root" not in data:
        errors.append("Missing required field: 'root'")
    elif not isinstance(data["root"], str):
        errors.append("Field 'root' must be a string")
    
    if "images" not in data:
        errors.append("Missing required field: 'images'")
        return errors  # Can't proceed without images
    
    if not isinstance(data["images"], dict):
        errors.append("Field 'images' must be an object")
        return errors
    
    # Validate each image entry
    for img_path, img_data in data["images"].items():
        prefix = f"images['{img_path}']"
        
        # Check image path format
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            errors.append(f"{prefix}: Invalid image extension (must be jpg/jpeg/png)")
        
        # Check required fields
        if not isinstance(img_data, dict):
            errors.append(f"{prefix}: Must be an object")
            continue
        
        if "label" not in img_data:
            errors.append(f"{prefix}: Missing required field 'label'")
        elif img_data["label"] not in ["pool", "ocean"]:
            errors.append(f"{prefix}: Invalid label '{img_data['label']}' (must be 'pool' or 'ocean')")
        
        if "should_detect" not in img_data:
            errors.append(f"{prefix}: Missing required field 'should_detect'")
        elif not isinstance(img_data["should_detect"], bool):
            errors.append(f"{prefix}: Field 'should_detect' must be boolean")
        
        # Check optional fields
        if "difficulty" in img_data:
            if img_data["difficulty"] not in ["easy", "medium", "hard"]:
                errors.append(f"{prefix}: Invalid difficulty '{img_data['difficulty']}' (must be easy/medium/hard)")
        
        if "tags" in img_data:
            if not isinstance(img_data["tags"], list):
                errors.append(f"{prefix}: Field 'tags' must be an array")
            elif not all(isinstance(t, str) for t in img_data["tags"]):
                errors.append(f"{prefix}: All tags must be strings")
        
        if "bbox" in img_data:
            bbox = img_data["bbox"]
            if not isinstance(bbox, dict):
                errors.append(f"{prefix}: Field 'bbox' must be an object")
            else:
                for field in ["x", "y", "width", "height"]:
                    if field not in bbox:
                        errors.append(f"{prefix}.bbox: Missing required field '{field}'")
                    elif not isinstance(bbox[field], (int, float)):
                        errors.append(f"{prefix}.bbox.{field}: Must be a number")
                    elif bbox[field] < 0:
                        errors.append(f"{prefix}.bbox.{field}: Must be >= 0")
        
        if "expected_mask_coverage" in img_data:
            coverage = img_data["expected_mask_coverage"]
            if not isinstance(coverage, (int, float)):
                errors.append(f"{prefix}: Field 'expected_mask_coverage' must be a number")
            elif not 0 <= coverage <= 1:
                errors.append(f"{prefix}: Field 'expected_mask_coverage' must be in range [0, 1]")
    
    return errors


def check_path_consistency(data: dict, gt_file: Path) -> list[str]:
    """Check that image paths are relative and consistent."""
    errors = []
    root = data.get("root", "")
    base_dir = gt_file.parent / root
    
    for img_path in data.get("images", {}).keys():
        # Check if path is relative
        if Path(img_path).is_absolute():
            errors.append(f"Image path '{img_path}' is absolute (should be relative to root)")
        
        # Check if file exists (if base_dir exists)
        if base_dir.exists():
            full_path = base_dir / img_path
            if not full_path.exists():
                errors.append(f"Image file not found: {full_path}")
    
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate water detection ground truth JSON against schema"
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to ground_truth.json file"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Path to schema file (default: auto-detect in same directory)"
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Skip checking if image files exist (useful for CI with synthetic data)"
    )
    
    args = parser.parse_args()
    
    # Load ground truth
    if not args.ground_truth.exists():
        print(f"❌ Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        return 1
    
    try:
        gt_data = load_ground_truth(args.ground_truth)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}", file=sys.stderr)
        return 1
    
    # Load schema
    if args.schema:
        schema_path = args.schema
    else:
        schema_path = args.ground_truth.parent / "ground_truth.schema.json"
    
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}", file=sys.stderr)
        return 1
    
    try:
        schema = load_schema(schema_path)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid schema JSON: {e}", file=sys.stderr)
        return 1
    
    # Validate against schema
    errors = validate_schema(gt_data, schema)
    
    # Check path consistency
    if not args.skip_file_check:
        path_errors = check_path_consistency(gt_data, args.ground_truth)
        errors.extend(path_errors)
    
    # Report results
    if errors:
        print(f"❌ Validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  • {error}", file=sys.stderr)
        return 1
    else:
        num_images = len(gt_data.get("images", {}))
        print(f"✅ Validation passed ({num_images} images)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
