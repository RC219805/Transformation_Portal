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

from transformation_portal.cli_support import (
    list_recipe_summaries,
    probe_dependency_versions,
    probe_pipeline_features,
    validate_recipe_file,
)


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


def _emit_dependency_group(title: str, dependency_specs: tuple[tuple[str, str], ...], unavailable_prefix: str) -> None:
    """Render a dependency status group."""

    typer.echo(f"\n{title}:")
    for status in probe_dependency_versions(dependency_specs):
        if status.available:
            typer.echo(f"  ✅ {status.display_name}: {status.version}")
            continue

        line = f"  {unavailable_prefix} {status.display_name}: not installed"
        if status.reason:
            line = f"{line} ({status.reason})"
        typer.echo(line)


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
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Enable parallel processing"),
):
    """Run unified enhancement pipeline with recipe.

    Process images using the unified pipeline with a YAML recipe configuration.
    Supports batch processing with dry-run preview.

    Example:
        transform-process pipeline process -i "renders/*.exr" -o outputs/ -r path/to/recipe.yaml
    """
    typer.echo("🚀 Running Unified Pipeline...")
    typer.echo(f"   Recipe: {recipe}")
    typer.echo(f"   Input: {input_glob}")
    typer.echo(f"   Output: {output_dir}")
    typer.echo(f"   Dry run: {dry_run}")
    typer.echo(f"   Parallel: {parallel}")

    if not recipe.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    try:
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(recipe)
        result = pipeline.process_batch(
            input_glob,
            output_dir,
            dry_run=dry_run,
            parallel=parallel,
        )

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
    recipes_dir: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Recipe directory path (defaults to config/recipes, then config/ recursion)",
    ),
):
    """List all available recipe presets.

    Scans the recipes directory and displays available pipeline recipes
    with their descriptions.

    Example:
        transform-process pipeline list-recipes
        transform-process pipeline list-recipes -d path/to/recipes/
    """
    typer.echo("📋 Available Pipeline Recipes\n")

    try:
        recipes = list_recipe_summaries(recipes_dir)

        if not recipes:
            if recipes_dir is not None:
                typer.echo(f"No recipe presets found in {recipes_dir}")
            else:
                typer.echo("No recipe presets found under config/recipes or config/")
            raise typer.Exit(code=0)

        for recipe in recipes:
            typer.echo(f"  📄 {recipe.get('name', 'Unknown')}")
            typer.echo(f"     {recipe.get('description', 'No description')}")
            typer.echo(f"     Stages: {', '.join(recipe.get('stages', []))}")
            typer.echo(f"     Output: {recipe.get('output_format', 'tiff')}")
            typer.echo(f"     Path: {recipe['path']}")
            typer.echo()

        typer.echo(f"Total: {len(recipes)} recipes found")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error loading recipe presets: {e}", err=True)
        raise typer.Exit(code=1)


@pipeline_app.command("validate-recipe")
def pipeline_validate_recipe(
    recipe_path: Path = typer.Argument(..., help="Recipe YAML file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
):
    """Validate a recipe configuration.

    Validates a recipe YAML file against the schema and reports any errors.

    Example:
        transform-process pipeline validate-recipe path/to/recipe.yaml
        transform-process pipeline validate-recipe custom_recipe.yaml -v
    """
    typer.echo(f"🔍 Validating recipe: {recipe_path}\n")

    if not recipe_path.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe_path}", err=True)
        raise typer.Exit(code=1)

    try:
        validation_result = validate_recipe_file(recipe_path)

        if verbose:
            info = validation_result.info
            typer.echo("Recipe Information:")
            typer.echo(f"  Name: {info.get('name', 'Unknown')}")
            typer.echo(f"  Description: {info.get('description', 'None')}")
            typer.echo(f"  Stages: {', '.join(info.get('stages', []))}")
            typer.echo(f"  Has Depth: {info.get('has_depth', False)}")
            typer.echo(f"  Has Material Response: {info.get('has_material_response', False)}")
            typer.echo(f"  Has Color Grading: {info.get('has_color_grading', False)}")
            typer.echo(f"  Has Quality Feedback: {info.get('has_quality_feedback', False)}")
            typer.echo(f"  Output Format: {info.get('output_format', 'unknown')}")
            typer.echo()

        if validation_result.is_valid:
            typer.echo("✅ Recipe is valid!")
        else:
            typer.echo("❌ Recipe validation failed:")
            for error in validation_result.errors:
                typer.echo(f"   - {error}")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error validating recipe: {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# MAIN UNIFIED CLI (For development/testing)
# ============================================================================

# Register subcommands with main app (for unified CLI during development)
app.add_typer(render_app, name="render")
app.add_typer(process_app, name="process")
app.add_typer(analyze_app, name="analyze")
app.add_typer(pipeline_app, name="pipeline")


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

    typer.echo(f"Python: {sys.version.split()[0]}")

    typer.echo("\nPipeline Features:")
    for status in probe_pipeline_features():
        line = f"  {'✅' if status.available else '⚠️ '} {status.display_name}"
        if status.reason:
            line = f"{line}: {status.reason}"
        typer.echo(line)

    _emit_dependency_group(
        "Core Dependencies",
        (
            ("NumPy", "numpy"),
            ("Pillow", "Pillow"),
            ("PyTorch", "torch"),
            ("PyYAML", "PyYAML"),
            ("Typer", "typer"),
        ),
        unavailable_prefix="❌",
    )
    _emit_dependency_group(
        "Optional Dependencies",
        (
            ("16-bit TIFF support", "tifffile"),
            ("Advanced image processing", "scipy"),
            ("LPIPS perceptual metrics", "lpips"),
            ("ML models", "transformers"),
        ),
        unavailable_prefix="⚠️ ",
    )


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
    "render_cli",
    "process_cli",
    "analyze_cli",
    "main",
    "version",
    "info",
]


if __name__ == "__main__":
    main()
