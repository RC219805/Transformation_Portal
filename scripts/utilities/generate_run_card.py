#!/usr/bin/env python3
"""
Generate Run Card for processed images.

Automates creation of run cards with pre-filled technical fields.
Human review required for: human_rating, decision, notes, lessons, tags.

Usage:
    python scripts/utilities/generate_run_card.py \\
        path/to/image.jpg \\
        --baseline-score 58.3 \\
        --processed-score 54.1 \\
        --recipe signature_estate_gentle \\
        --project project_name

    # With recipe settings
    python scripts/utilities/generate_run_card.py \\
        input_images/750picacho/kitchen.jpg \\
        --baseline-score 58.3 \\
        --processed-score 54.1 \\
        --recipe interior_luxury \\
        --recipe-settings '{"clarity": 0.2, "glow": 0.1}' \\
        --project 750_picacho_lane

For integration with pipeline, see ROADMAP_NEXT_THREE.md
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

try:
    import yaml
except ImportError:
    print("⚠️  PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)

# Import scene type taxonomy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
try:
    from transformation_portal.scene_types import normalize_scene_type, validate_scene_type
except ImportError:
    print("⚠️  Could not import scene_types module")
    print("   Run from repository root or check src/transformation_portal/scene_types.py")
    sys.exit(1)


def infer_scene_type(image_path: str) -> str:
    """
    Infer scene type from filename or folder structure.

    Args:
        image_path: Path to image file

    Returns:
        Inferred scene type (may need manual review)

    Examples:
        >>> infer_scene_type("renders/kitchen.jpg")
        'interior_kitchen'
        >>> infer_scene_type("750picacho/pool_exterior.jpg")
        'exterior_pool'
    """
    path_lower = str(image_path).lower()

    # Interior detection
    if any(x in path_lower for x in ["bedroom", "bed", "master", "suite"]):
        return "interior_bedroom"
    elif any(x in path_lower for x in ["kitchen", "kit"]):
        return "interior_kitchen"
    elif any(x in path_lower for x in ["bathroom", "bath", "powder"]):
        return "interior_bathroom"
    elif any(x in path_lower for x in ["great", "living", "family", "lounge"]):
        return "interior_great_room"
    elif any(x in path_lower for x in ["dining", "breakfast"]):
        return "interior_dining_room"
    elif any(x in path_lower for x in ["office", "study", "library"]):
        return "interior_office"
    elif any(x in path_lower for x in ["closet", "wardrobe", "dressing"]):
        return "interior_closet"
    elif any(x in path_lower for x in ["hallway", "corridor", "foyer", "entry"]):
        return "interior_hallway"

    # Exterior detection
    elif any(x in path_lower for x in ["pool", "spa", "water", "jacuzzi"]):
        return "exterior_pool"
    elif any(x in path_lower for x in ["aerial", "drone", "overhead"]):
        return "aerial_exterior"
    elif any(x in path_lower for x in ["garden", "yard", "landscape"]):
        return "exterior_garden"
    elif any(x in path_lower for x in ["courtyard", "patio", "terrace", "deck"]):
        return "exterior_courtyard"
    elif any(x in path_lower for x in ["facade", "front", "elevation"]):
        return "exterior_facade"

    # Special conditions
    elif any(x in path_lower for x in ["twilight", "dusk", "golden_hour"]):
        return "twilight_exterior"
    elif any(x in path_lower for x in ["night"]):
        if any(x in path_lower for x in ["interior", "inside"]):
            return "night_interior"
        else:
            return "night_exterior"

    # Default - needs manual specification
    return "TODO: Specify scene type"


def generate_run_card(
    image_path: str,
    baseline_score: float,
    processed_score: float,
    recipe_name: str,
    project_name: str,
    recipe_settings: Optional[Dict[str, Any]] = None,
    processing_time: Optional[float] = None,
    output_dir: str = "docs/runs",
    scene_type_override: Optional[str] = None
) -> Path:
    """
    Generate draft run card from pipeline output.

    Human fills in: human_rating, decision, notes, lessons, tags

    Args:
        image_path: Path to source image
        baseline_score: Quality score of baseline/source image
        processed_score: Quality score after processing
        recipe_name: Name of recipe used
        project_name: Project identifier
        recipe_settings: Recipe configuration dict (optional)
        processing_time: Processing time in seconds (optional)
        output_dir: Directory for run card output
        scene_type_override: Manual scene type (skips inference)

    Returns:
        Path to generated run card YAML file

    Raises:
        ValueError: If scene type cannot be normalized
    """
    image_id = Path(image_path).stem

    # Infer or use override scene type
    if scene_type_override:
        raw_scene_type = scene_type_override
    else:
        raw_scene_type = infer_scene_type(image_path)

    # Normalize to canonical taxonomy
    try:
        scene_type = normalize_scene_type(raw_scene_type)
    except ValueError as e:
        print(f"⚠️  {e}")
        scene_type = "TODO: Specify valid scene type"

    # Calculate delta
    delta_score = round(processed_score - baseline_score, 2)

    # Build run card structure
    run_card = {
        "image_id": image_id,
        "image_path": str(image_path),
        "project": project_name,
        "scene_type": scene_type,
        "scene_features": ["TODO: Review and add specific features"],

        "source_baseline_score": round(baseline_score, 2),
        "processed_score": round(processed_score, 2),
        "delta_score": delta_score,
        "targets_met": "TODO: Review quality report and specify",

        "recipe": recipe_name,
        "recipe_path": f"config/recipes/{recipe_name}.yaml",
        "recipe_settings": recipe_settings or {},

        "processing_time_seconds": processing_time if processing_time else "TODO: From pipeline log",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_by": "generate_run_card.py",

        "# HUMAN REVIEW REQUIRED": "Complete the following fields after visual review",
        "human_rating": "TODO: [clearly_better|acceptable_but_unnecessary|worse_than_source|significantly_worse]",
        "decision": "TODO: [recipe_recommended|recipe_acceptable|recipe_avoid]",
        "notes": ["TODO: Add observations about quality, artifacts, strengths, weaknesses"],
        "lessons": ["TODO: Add learnings for future processing"],
        "tags": ["TODO: Add tags for retrieval (e.g., high_contrast, warm_tones, sharp_details)"]
    }

    # Write to project subdirectory
    project_dir = Path(output_dir) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    output_file = project_dir / f"{image_id}_{recipe_name}.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(run_card, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Draft run card created: {output_file}")
    print(f"   Image: {image_id}")
    print(f"   Scene type: {scene_type}")
    print(f"   Delta: {delta_score:+.1f}%")
    print(f"⚠️  REVIEW REQUIRED: Complete human assessment fields")

    return output_file


def main():
    """Command-line interface for run card generation."""
    parser = argparse.ArgumentParser(
        description="Generate run card with pre-filled technical fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s image.jpg --baseline-score 58.3 --processed-score 54.1 \\
    --recipe signature_estate --project my_project

  # With recipe settings
  %(prog)s kitchen.jpg --baseline-score 60.0 --processed-score 62.5 \\
    --recipe interior_luxury --project 750_picacho \\
    --recipe-settings '{"clarity": 0.2, "glow": 0.1}'

  # With scene type override
  %(prog)s pool.jpg --baseline-score 55.0 --processed-score 58.0 \\
    --recipe pool_estate --project villa_project \\
    --scene-type exterior_pool
        """
    )

    parser.add_argument("image_path", help="Path to source image")
    parser.add_argument("--baseline-score", "-b", type=float, required=True,
                        help="Quality score of baseline image")
    parser.add_argument("--processed-score", "-p", type=float, required=True,
                        help="Quality score after processing")
    parser.add_argument("--recipe", "-r", required=True,
                        help="Recipe name used for processing")
    parser.add_argument("--project", required=True,
                        help="Project name/identifier")
    parser.add_argument("--recipe-settings", "-s",
                        help="Recipe settings as JSON string")
    parser.add_argument("--processing-time", "-t", type=float,
                        help="Processing time in seconds")
    parser.add_argument("--output-dir", "-o", default="docs/runs",
                        help="Output directory for run cards (default: docs/runs)")
    parser.add_argument("--scene-type",
                        help="Override inferred scene type")

    args = parser.parse_args()

    # Parse recipe settings if provided
    recipe_settings = None
    if args.recipe_settings:
        try:
            recipe_settings = json.loads(args.recipe_settings)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in --recipe-settings: {e}")
            sys.exit(1)

    # Generate run card
    try:
        output_path = generate_run_card(
            image_path=args.image_path,
            baseline_score=args.baseline_score,
            processed_score=args.processed_score,
            recipe_name=args.recipe,
            project_name=args.project,
            recipe_settings=recipe_settings,
            processing_time=args.processing_time,
            output_dir=args.output_dir,
            scene_type_override=args.scene_type
        )
        print(f"\n✅ Success! Edit {output_path} to complete human review.")
        return 0
    except Exception as e:
        print(f"❌ Error generating run card: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
