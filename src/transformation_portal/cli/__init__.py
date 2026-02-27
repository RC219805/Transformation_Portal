"""Command-line interface entry points for Transformation Portal.

This module provides a unified CLI for accessing transformation portal
functionalities including rendering, processing, and analysis tools.

The CLI is structured with three main subcommands:
- render: AI-powered rendering and enhancement pipelines
- process: Image and video processing operations
- analyze: Codebase and workflow analysis tools

Entry Points:
    transform-render: Main rendering CLI (calls render_cli)
    transform-process: Main processing CLI (calls process_cli)
    transform-analyze: Main analysis CLI (calls analyze_cli)

Example Usage:
    # Render with Lux Render Pipeline
    transform-render lux --input image.jpg --output enhanced/

    # Process with Material Response
    transform-process material --input image.tiff --strength 0.7

    # Analyze codebase philosophy
    transform-analyze philosophy --path ./src/

Note:
    This CLI is under active development. Additional subcommands and options
    will be added as the transformation portal evolves.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError as e:
    raise ImportError(
        "typer is required for the CLI. Install it with:\n"
        "  pip install typer\n"
        "or install the full package with:\n"
        "  pip install -e '.[dev]'"
    ) from e


def check_module_availability(module_path: str, module_name: str) -> bool:
    """Check if a module is available for import.

    Args:
        module_path: Full import path (e.g., 'transformation_portal.pipelines.lux_render_pipeline')
        module_name: Human-readable module name for error messages

    Returns:
        True if module can be imported, False otherwise

    Raises:
        typer.Exit: If module cannot be imported
    """
    try:
        # Attempt to import the module
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            from_module, import_name = parts
            __import__(from_module, fromlist=[import_name])
        else:
            __import__(module_path)
        return True
    except ImportError as e:
        typer.echo(f"❌ Error loading {module_name}: {e}", err=True)
        raise typer.Exit(code=1)


# Main application instances
app = typer.Typer(
    name="transformation-portal",
    help="Professional image and video processing toolkit",
    no_args_is_help=True,
    add_completion=False,
)

render_app = typer.Typer(
    name="render",
    help="AI-powered rendering and enhancement pipelines",
    no_args_is_help=True,
)

process_app = typer.Typer(
    name="process",
    help="Image and video processing operations",
    no_args_is_help=True,
)

analyze_app = typer.Typer(
    name="analyze",
    help="Codebase and workflow analysis tools",
    no_args_is_help=True,
)


# ============================================================================
# RENDER SUBCOMMANDS
# ============================================================================


@render_app.command("lux")
def render_lux(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input image path"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Enhancement prompt"),
    strength: float = typer.Option(0.7, "--strength", "-s", help="Enhancement strength (0.0-1.0)"),
    upscale: bool = typer.Option(False, "--upscale", "-u", help="Apply 4x upscaling"),
):
    """Run Lux Render Pipeline for AI-powered enhancement.

    The Lux Render Pipeline uses Stable Diffusion XL, ControlNet, and Real-ESRGAN
    for intelligent enhancement of architectural and real estate imagery.
    """
    typer.echo("🎨 Running Lux Render Pipeline...")
    typer.echo(f"   Input: {input_path}")
    typer.echo(f"   Output: {output_dir}")
    typer.echo(f"   Strength: {strength}")

    if not input_path.exists():
        typer.echo(f"❌ Error: Input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    # Verify pipeline module is available
    check_module_availability("transformation_portal.pipelines.lux_render_pipeline", "Lux Render Pipeline")
    typer.echo("✅ Pipeline module loaded successfully")
    typer.echo("⚠️  Note: Full pipeline execution requires ML dependencies")
    typer.echo("   Install with: pip install -e '.[ml]'")


@render_app.command("depth")
def render_depth(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input image path"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    preset: str = typer.Option("interior", "--preset", "-p", help="Processing preset"),
):
    """Run Depth Pipeline for depth-aware processing.

    The Depth Pipeline uses Depth Anything V2 for monocular depth estimation
    and applies depth-aware enhancements for architectural rendering.
    """
    typer.echo("🌊 Running Depth Pipeline...")
    typer.echo(f"   Input: {input_path}")
    typer.echo(f"   Output: {output_dir}")
    typer.echo(f"   Preset: {preset}")

    if not input_path.exists():
        typer.echo(f"❌ Error: Input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    # Verify depth tools module is available
    check_module_availability("transformation_portal.pipelines.depth_tools", "Depth Tools")
    typer.echo("✅ Depth tools module loaded successfully")


# ============================================================================
# PROCESS SUBCOMMANDS
# ============================================================================


@process_app.command("material")
def process_material(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input image path"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output image path"),
    strength: float = typer.Option(0.7, "--strength", "-s", help="Enhancement strength (0.0-1.0)"),
    surfaces: Optional[str] = typer.Option(
        None,
        "--surfaces",
        help="Comma-separated surface types (wood,metal,glass,fabric,stone)",
    ),
):
    """Apply Material Response Technology for surface enhancement.

    Material Response analyzes and enhances material surfaces with physics-based
    rendering techniques for wood, metal, glass, fabric, and stone.
    """
    typer.echo("💎 Running Material Response...")
    typer.echo(f"   Input: {input_path}")
    typer.echo(f"   Output: {output_path}")
    typer.echo(f"   Strength: {strength}")

    if not input_path.exists():
        typer.echo(f"❌ Error: Input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    # Verify Material Response module is available
    check_module_availability("transformation_portal.processors.material_response.core", "Material Response")
    typer.echo("✅ Material Response module loaded successfully")


@process_app.command("video")
def process_video(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input video path"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output video path"),
    preset: str = typer.Option("signature_estate", "--preset", "-p", help="Grading preset"),
    lut_strength: float = typer.Option(0.7, "--lut-strength", help="LUT strength (0.0-1.0)"),
):
    """Process video with Luxury Video Master Grader.

    Apply professional color grading, LUTs, and HDR tone mapping to video content
    using FFmpeg-based processing pipelines.
    """
    typer.echo("🎬 Running Video Master Grader...")
    typer.echo(f"   Input: {input_path}")
    typer.echo(f"   Output: {output_path}")
    typer.echo(f"   Preset: {preset}")

    if not input_path.exists():
        typer.echo(f"❌ Error: Input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    # Verify Video Master Grader module is available
    check_module_availability(
        "transformation_portal.processors.luxury_video_master_grader",
        "Video Master Grader",
    )
    typer.echo("✅ Video Master Grader module loaded successfully")
    typer.echo("⚠️  Note: FFmpeg is required for video processing")


@process_app.command("tif")
def process_tiff(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Input directory"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    preset: str = typer.Option("signature", "--preset", "-p", help="Processing preset"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process subdirectories"),
):
    """Process TIFF images with Luxury TIFF Batch Processor.

    Batch process 16-bit TIFF images with professional color grading, LUTs,
    and metadata preservation for luxury real estate workflows.
    """
    typer.echo("📸 Running TIFF Batch Processor...")
    typer.echo(f"   Input: {input_dir}")
    typer.echo(f"   Output: {output_dir}")
    typer.echo(f"   Preset: {preset}")
    typer.echo(f"   Recursive: {recursive}")

    if not input_dir.exists():
        typer.echo(f"❌ Error: Input directory not found: {input_dir}", err=True)
        raise typer.Exit(code=1)

    typer.echo("⚠️  Note: Full TIFF support requires tifffile")
    typer.echo("   Install with: pip install -e '.[tiff]'")


# ============================================================================
# ANALYZE SUBCOMMANDS
# ============================================================================


@analyze_app.command("philosophy")
def analyze_philosophy(
    path: Path = typer.Option(".", "--path", "-p", help="Path to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output report path"),
):
    """Run codebase philosophy auditor.

    Analyze codebase for adherence to architectural principles, design patterns,
    and coding standards specific to the Transformation Portal.
    """
    typer.echo("🔍 Running Codebase Philosophy Auditor...")
    typer.echo(f"   Path: {path}")

    if not path.exists():
        typer.echo(f"❌ Error: Path not found: {path}", err=True)
        raise typer.Exit(code=1)

    # Verify auditor module is available
    check_module_availability(
        "transformation_portal.analyzers.codebase_philosophy_auditor",
        "Codebase Philosophy Auditor",
    )
    typer.echo("✅ Auditor module loaded successfully")


@analyze_app.command("decay")
def analyze_decay(
    path: Path = typer.Option(".", "--path", "-p", help="Path to analyze"),
    threshold_days: int = typer.Option(90, "--threshold", "-t", help="Decay threshold in days"),
):
    """Run decision decay dashboard.

    Analyze temporal contracts and decision age to identify technical debt
    and outdated architectural decisions.
    """
    typer.echo("⏰ Running Decision Decay Dashboard...")
    typer.echo(f"   Path: {path}")
    typer.echo(f"   Threshold: {threshold_days} days")

    if not path.exists():
        typer.echo(f"❌ Error: Path not found: {path}", err=True)
        raise typer.Exit(code=1)

    # Verify dashboard module is available
    check_module_availability(
        "transformation_portal.analyzers.decision_decay_dashboard",
        "Decision Decay Dashboard",
    )
    typer.echo("✅ Dashboard module loaded successfully")


@analyze_app.command("workflow")
def analyze_workflow(
    path: Path = typer.Option(".github/workflows", "--path", "-p", help="Workflows directory"),
):
    """Parse and analyze GitHub Actions workflows.

    Analyze GitHub Actions workflow files for optimization opportunities,
    security issues, and best practices.
    """
    typer.echo("⚙️  Running Workflow Analyzer...")
    typer.echo(f"   Path: {path}")

    if not path.exists():
        typer.echo(f"❌ Error: Path not found: {path}", err=True)
        raise typer.Exit(code=1)

    # Verify workflow parser module is available
    check_module_availability("transformation_portal.analyzers.parse_workflows", "Workflow Parser")
    typer.echo("✅ Workflow parser module loaded successfully")


# ============================================================================
# MAIN CLI FUNCTIONS (Entry Points)
# ============================================================================


def render_cli():
    """Entry point for transform-render command."""
    render_app()


def process_cli():
    """Entry point for transform-process command."""
    process_app()


def analyze_cli():
    """Entry point for transform-analyze command."""
    analyze_app()


# ============================================================================
# UNIFIED PIPELINE COMMANDS
# ============================================================================

pipeline_app = typer.Typer(
    name="pipeline",
    help="Unified pipeline operations with YAML recipes",
    no_args_is_help=True,
)


@pipeline_app.command("process")
def process_command(
    input_glob: str = typer.Option(..., "--input", "-i", help="Input glob pattern (e.g., 'inputs/*.jpg')"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    recipe: Path = typer.Option(..., "--recipe", "-r", help="Recipe YAML file path"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Preview processing plan without executing"),
):
    """Run unified enhancement pipeline with recipe.

    Process images using the unified pipeline with a YAML recipe configuration.
    Supports batch processing with dry-run preview.

    Example:
        transform-process pipeline process -i "renders/*.exr" -o outputs/ -r config/recipes/signature_estate.yaml
    """
    typer.echo("🚀 Running Unified Pipeline...")
    typer.echo(f"   Recipe: {recipe}")
    typer.echo(f"   Input: {input_glob}")
    typer.echo(f"   Output: {output_dir}")
    typer.echo(f"   Dry run: {dry_run}")

    if not recipe.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    try:
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(recipe)
        result = pipeline.process_batch(input_glob, output_dir, dry_run=dry_run)

        if not dry_run:
            typer.echo(f"\n✅ Processed {result.successful_count} images successfully")
            if result.failed_count > 0:
                typer.echo(f"⚠️  {result.failed_count} images failed", err=True)

    except ImportError as e:
        typer.echo(f"❌ Error loading pipeline: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Pipeline error: {e}", err=True)
        raise typer.Exit(code=1)


@pipeline_app.command("list-recipes")
def pipeline_list_recipes(
    recipes_dir: Path = typer.Option(Path("config/recipes"), "--dir", "-d", help="Recipes directory path"),
):
    """List all available recipe presets.

    Scans the recipes directory and displays available pipeline recipes
    with their descriptions.

    Example:
        transform-process pipeline list-recipes
        transform-process pipeline list-recipes -d custom/recipes/
    """
    typer.echo("📋 Available Pipeline Recipes\n")

    if not recipes_dir.exists():
        typer.echo(f"⚠️  Recipes directory not found: {recipes_dir}", err=True)
        typer.echo("   Create recipes in config/recipes/ directory")
        raise typer.Exit(code=1)

    try:
        from transformation_portal.config_loader import list_recipes

        recipes = list_recipes(recipes_dir)

        if not recipes:
            typer.echo("No recipes found in directory")
            raise typer.Exit(code=0)

        for recipe in recipes:
            if "error" in recipe:
                typer.echo(f"  ❌ {recipe['path']}: {recipe['error']}")
            else:
                name = recipe.get("name", "Unknown")
                description = recipe.get("description", "No description")
                stages = recipe.get("stages", [])
                output_format = recipe.get("output_format", "tiff")

                typer.echo(f"  📄 {name}")
                typer.echo(f"     {description}")
                typer.echo(f"     Stages: {', '.join(stages)}")
                typer.echo(f"     Output: {output_format}")
                typer.echo(f"     Path: {recipe['path']}")
                typer.echo()

    except ImportError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


@pipeline_app.command("validate-recipe")
def pipeline_validate_recipe(
    recipe_path: Path = typer.Argument(..., help="Recipe YAML file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
):
    """Validate a recipe configuration.

    Validates a recipe YAML file against the schema and reports any errors.

    Example:
        transform-process pipeline validate-recipe config/recipes/signature_estate.yaml
        transform-process pipeline validate-recipe custom_recipe.yaml -v
    """
    typer.echo(f"🔍 Validating recipe: {recipe_path}\n")

    if not recipe_path.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe_path}", err=True)
        raise typer.Exit(code=1)

    try:
        from transformation_portal.config_loader import get_recipe_info, load_recipe, validate_recipe

        # Load the recipe
        recipe = load_recipe(recipe_path, expand_env=False, resolve_paths=False)

        # Validate
        is_valid, errors = validate_recipe(recipe)

        if verbose:
            info = get_recipe_info(recipe)
            typer.echo("Recipe Information:")
            typer.echo(f"  Name: {info.get('name', 'Unknown')}")
            typer.echo(f"  Description: {info.get('description', 'None')}")
            typer.echo(f"  Stages: {', '.join(info.get('stages', []))}")
            typer.echo(f"  Has Depth: {info.get('has_depth', False)}")
            typer.echo(f"  Has Material Response: {info.get('has_material_response', False)}")
            typer.echo(f"  Has Color Grading: {info.get('has_color_grading', False)}")
            typer.echo(f"  Output Format: {info.get('output_format', 'unknown')}")
            typer.echo()

        if is_valid:
            typer.echo("✅ Recipe is valid!")
        else:
            typer.echo("❌ Recipe validation failed:")
            for error in errors:
                typer.echo(f"   - {error}")
            raise typer.Exit(code=1)

    except ImportError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error validating recipe: {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# INGEST END-TO-END COMMANDS
# ============================================================================

ingest_app = typer.Typer(
    name="ingest",
    help="RAW file ingest and provenance capture operations",
    no_args_is_help=True,
)


@ingest_app.command("e2e")
def ingest_e2e(
    input_path: Path = typer.Option(
        ...,
        "--input", "-i",
        exists=True,
        help="Input file or directory containing RAW/TIFF images",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output directory for all artifacts",
    ),
    contract: str = typer.Option(
        "legacy_linear_srgb",
        "--contract", "-c",
        help="Ingest contract: 'camera_native_linear' or 'legacy_linear_srgb'",
    ),
    enable_depth: bool = typer.Option(
        False,
        "--enable-depth/--no-depth",
        help="Enable depth estimation (DA3) phase",
    ),
    enable_evidence: bool = typer.Option(
        False,
        "--enable-evidence/--no-evidence",
        help="Enable evidence bundle generation phase",
    ),
    depth_device: str = typer.Option(
        "cpu",
        "--depth-device",
        help="Device for depth estimation: cpu, mps, or cuda",
    ),
    generate_pbr: bool = typer.Option(
        False,
        "--generate-pbr/--no-pbr",
        help="Generate PBR maps during depth phase",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Search subdirectories for images",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Fail on validation errors",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run", "-n",
        help="Preview plan without executing",
    ),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Output machine-readable JSON",
    ),
):
    """Run end-to-end RAW file ingest through integrated phases.

    This command orchestrates the complete ingest pipeline:

    1. INGEST: Extract metadata and generate provenance sidecars
    2. DEPTH (optional): Run depth estimation with DA3
    3. EVIDENCE (optional): Generate Merkle-backed evidence bundle

    Examples:

        # Basic ingest with provenance capture
        transformation-portal ingest e2e -i /path/to/images -o /output

        # Full pipeline with depth and evidence
        transformation-portal ingest e2e -i /path/to/images -o /output \\
            --enable-depth --enable-evidence --depth-device mps

        # Dry run to preview plan
        transformation-portal ingest e2e -i /path/to/images -o /output --dry-run
    """
    from transformation_portal.cli.ingest_e2e import run_e2e_ingest
    import json

    # Validate inputs
    valid_contracts = ("camera_native_linear", "legacy_linear_srgb")
    if contract not in valid_contracts:
        if json_output:
            typer.echo(json.dumps({
                "success": False,
                "error": f"Invalid contract: {contract}. Valid: {valid_contracts}",
            }))
        else:
            typer.echo(f"❌ Invalid contract: {contract}", err=True)
        raise typer.Exit(code=2)

    valid_devices = ("cpu", "mps", "cuda")
    if depth_device not in valid_devices:
        if json_output:
            typer.echo(json.dumps({
                "success": False,
                "error": f"Invalid device: {depth_device}. Valid: {valid_devices}",
            }))
        else:
            typer.echo(f"❌ Invalid device: {depth_device}", err=True)
        raise typer.Exit(code=2)

    if not json_output:
        typer.echo("🚀 End-to-End RAW Ingest Pipeline")
        typer.echo(f"   Input: {input_path}")
        typer.echo(f"   Output: {output_dir}")
        typer.echo(f"   Contract: {contract}")
        phases_str = "ingest"
        if enable_depth:
            phases_str += " + depth"
        if enable_evidence:
            phases_str += " + evidence"
        typer.echo(f"   Phases: {phases_str}")
        if dry_run:
            typer.echo("   Mode: DRY RUN")
        typer.echo()

    result = run_e2e_ingest(
        input_path=input_path,
        output_dir=output_dir,
        contract=contract,
        enable_depth=enable_depth,
        enable_evidence=enable_evidence,
        depth_device=depth_device,
        generate_pbr=generate_pbr,
        recursive=recursive,
        strict=strict,
        dry_run=dry_run,
    )

    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        if dry_run:
            typer.echo("📋 Execution Plan:")
            for phase in result.phases:
                plan = phase.artifacts.get("plan", "Process items")
                typer.echo(f"   • {phase.phase.upper()}: {plan}")
            typer.echo(f"\nTotal images: {result.input_count}")
        else:
            typer.echo("📊 Results:")
            for phase in result.phases:
                status = "✅" if phase.success else "❌"
                typer.echo(f"   {status} {phase.phase.upper()}: "
                           f"{phase.items_processed} processed "
                           f"({phase.elapsed_seconds:.2f}s)")
                if phase.error:
                    typer.echo(f"      Error: {phase.error}")

            if result.success:
                typer.echo(f"\n✅ Pipeline completed ({result.total_elapsed_seconds:.2f}s)")
            else:
                typer.echo(f"\n❌ Pipeline failed: {result.error}")

    raise typer.Exit(code=0 if result.success else 3)


@ingest_app.command("info")
def ingest_info():
    """Show information about ingest phases and dependencies."""
    from transformation_portal.cli.ingest_e2e import SUPPORTED_RAW_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS

    typer.echo("🔧 End-to-End Ingest Pipeline Information\n")

    typer.echo("Available Phases:")
    typer.echo("  1. INGEST - Metadata extraction and provenance capture")
    typer.echo("  2. DEPTH  - Depth estimation (DA3) with optional PBR")
    typer.echo("  3. EVIDENCE - Merkle-backed evidence bundle generation")
    typer.echo()

    typer.echo("Supported Contracts:")
    typer.echo("  • legacy_linear_srgb - Phase I (default)")
    typer.echo("  • camera_native_linear - Phase II (requires rawpy)")
    typer.echo()

    typer.echo("Supported Formats:")
    raw_exts = ", ".join(sorted(SUPPORTED_RAW_EXTENSIONS))
    img_exts = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS - SUPPORTED_RAW_EXTENSIONS))
    typer.echo(f"  RAW: {raw_exts}")
    typer.echo(f"  Other: {img_exts}")


# ============================================================================
# MAIN UNIFIED CLI (For development/testing)
# ============================================================================

# Register subcommands with main app (for unified CLI during development)
app.add_typer(render_app, name="render")
app.add_typer(process_app, name="process")
app.add_typer(analyze_app, name="analyze")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(ingest_app, name="ingest")


@app.command()
def version():
    """Show version information."""
    try:
        from transformation_portal import __version__

        typer.echo(f"Transformation Portal v{__version__}")
    except ImportError:
        typer.echo("Transformation Portal (version unknown)")


@app.command()
def info():
    """Show system and dependency information."""
    typer.echo("Transformation Portal - System Information\n")

    # Python version
    typer.echo(f"Python: {sys.version.split()[0]}")

    # Check key dependencies
    dependencies = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("typer", "Typer"),
    ]

    typer.echo("\nDependencies:")
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            typer.echo(f"  ✅ {display_name}: {version}")
        except ImportError:
            typer.echo(f"  ❌ {display_name}: not installed")

    # Optional dependencies
    optional_deps = [
        ("tifffile", "TIFF support"),
        ("transformers", "ML models"),
        ("cv2", "OpenCV"),
    ]

    typer.echo("\nOptional Dependencies:")
    for module_name, feature in optional_deps:
        try:
            __import__(module_name)
            typer.echo(f"  ✅ {feature}")
        except ImportError:
            typer.echo(f"  ⚠️  {feature}: not installed")


def main():
    """Main entry point for unified CLI (for development/testing)."""
    app()


# Export all CLI entry points
__all__ = [
    "app",
    "render_app",
    "process_app",
    "analyze_app",
    "pipeline_app",
    "ingest_app",
    "render_cli",
    "process_cli",
    "analyze_cli",
    "main",
    "version",
    "info",
]


if __name__ == "__main__":
    main()
