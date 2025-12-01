#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformation Portal - Main CLI Entry Point

This module provides the primary command-line interface for the Transformation Portal,
enabling unified access to all pipeline and processing capabilities.

Usage:
    python -m transformation_portal --help
    python -m transformation_portal process -i "inputs/*.jpg" -r config/recipes/signature_estate.yaml -o output/
    python -m transformation_portal list-recipes
    python -m transformation_portal validate-recipe config/recipes/signature_estate.yaml

The CLI supports:
    - Recipe-driven batch processing
    - Dry-run mode for previewing processing plans
    - Quality feedback with RAG integration
    - 4K upscaling with Rendering 4K Pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import typer
except ImportError as e:
    raise ImportError(
        "typer is required for the CLI. Install it with:\n"
        "  pip install typer\n"
        "or install the full package with:\n"
        "  pip install -e '.[dev]'"
    ) from e


# Create the main application
app = typer.Typer(
    name="transformation-portal",
    help="Professional image and video processing toolkit for luxury real estate rendering",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def process(
    input_glob: str = typer.Option(
        ..., "--input", "-i",
        help="Input glob pattern (e.g., 'inputs/*.jpg')"
    ),
    recipe: Path = typer.Option(
        ..., "--recipe", "-r",
        help="Recipe YAML file path"
    ),
    output: Path = typer.Option(
        Path("./final"), "--output", "-o",
        help="Output directory"
    ),
    mode: str = typer.Option(
        "auto", "--mode", "-m",
        help="Processing mode: auto|image|video"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n",
        help="Preview processing plan without executing"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", "-p",
        help="Enable parallel processing"
    ),
    log_level: str = typer.Option(
        "info", "--log-level", "-l",
        help="Logging level: debug|info|warning|error"
    ),
):
    """Run unified enhancement pipeline.

    Process images through the unified pipeline using a YAML recipe configuration.
    Supports batch processing with dry-run preview and quality feedback.

    Example:
        transformation-portal process -i "renders/*.exr" -r config/recipes/signature_estate.yaml -o output/
    """
    import logging

    # Configure logging
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(
        level=log_levels.get(log_level.lower(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    typer.echo("🚀 Transformation Portal - Unified Pipeline")
    typer.echo(f"   Recipe: {recipe}")
    typer.echo(f"   Input: {input_glob}")
    typer.echo(f"   Output: {output}")
    typer.echo(f"   Mode: {mode}")
    typer.echo(f"   Dry run: {dry_run}")
    typer.echo()

    if not recipe.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    try:
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(recipe)
        result = pipeline.process_batch(input_glob, output, mode=mode, dry_run=dry_run)

        if not dry_run:
            typer.echo()
            typer.echo(f"✅ Processed {result.successful_count} images successfully")
            if result.failed_count > 0:
                typer.echo(f"⚠️  {result.failed_count} images failed", err=True)
            typer.echo(f"📊 Total time: {result.total_time:.2f}s")

    except ImportError as e:
        typer.echo(f"❌ Error loading pipeline: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Pipeline error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("list-recipes")
def list_recipes(
    recipes_dir: Path = typer.Option(
        Path("config/recipes"),
        "--dir", "-d",
        help="Recipes directory path"
    ),
):
    """List all available recipe presets.

    Scans the recipes directory and displays available pipeline recipes
    with their descriptions and configurations.

    Example:
        transformation-portal list-recipes
        transformation-portal list-recipes -d custom/recipes/
    """
    typer.echo("📋 Available Pipeline Recipes\n")

    if not recipes_dir.exists():
        typer.echo(f"⚠️  Recipes directory not found: {recipes_dir}", err=True)
        typer.echo("   Create recipes in config/recipes/ directory")
        raise typer.Exit(code=1)

    try:
        from transformation_portal.config_loader import list_recipes as get_recipes

        recipes = get_recipes(recipes_dir)

        if not recipes:
            typer.echo("No recipes found in directory")
            raise typer.Exit(code=0)

        for recipe in recipes:
            if 'error' in recipe:
                typer.echo(f"  ❌ {recipe['path']}: {recipe['error']}")
            else:
                name = recipe.get('name', 'Unknown')
                description = recipe.get('description', 'No description')
                stages = recipe.get('stages', [])
                output_format = recipe.get('output_format', 'tiff')

                typer.echo(f"  📄 {name}")
                typer.echo(f"     {description}")
                typer.echo(f"     Stages: {', '.join(stages)}")
                typer.echo(f"     Output: {output_format}")
                typer.echo(f"     Path: {recipe['path']}")
                typer.echo()

        typer.echo(f"Total: {len(recipes)} recipes found")

    except ImportError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("validate-recipe")
def validate_recipe(
    recipe: Path = typer.Argument(..., help="Recipe YAML file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
):
    """Validate recipe configuration.

    Validates a recipe YAML file against the schema and reports any errors
    or warnings.

    Example:
        transformation-portal validate-recipe config/recipes/signature_estate.yaml
        transformation-portal validate-recipe custom_recipe.yaml -v
    """
    typer.echo(f"🔍 Validating recipe: {recipe}\n")

    if not recipe.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    try:
        from transformation_portal.config_loader import load_recipe, validate_recipe as validate, get_recipe_info

        # Load the recipe
        loaded = load_recipe(recipe, expand_env=False, resolve_paths=False)

        # Validate
        is_valid, errors = validate(loaded)

        if verbose:
            info = get_recipe_info(loaded)
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

    # Check pipeline availability
    try:
        from transformation_portal.pipeline_unified import HAS_QUALITY_BRIDGE, HAS_4K_PIPELINE
        typer.echo("\nPipeline Features:")
        typer.echo(f"  {'✅' if HAS_QUALITY_BRIDGE else '⚠️ '} RAG Quality Feedback")
        typer.echo(f"  {'✅' if HAS_4K_PIPELINE else '⚠️ '} 4K Rendering Pipeline")
    except ImportError:
        typer.echo("\n⚠️  Pipeline modules not available")

    # Check key dependencies
    dependencies = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
        ("typer", "Typer"),
    ]

    typer.echo("\nCore Dependencies:")
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            typer.echo(f"  ✅ {display_name}: {version}")
        except ImportError:
            typer.echo(f"  ❌ {display_name}: not installed")

    # Optional dependencies
    optional_deps = [
        ("tifffile", "16-bit TIFF support"),
        ("scipy", "Advanced image processing"),
        ("lpips", "LPIPS perceptual metrics"),
        ("transformers", "ML models"),
    ]

    typer.echo("\nOptional Dependencies:")
    for module_name, feature in optional_deps:
        try:
            __import__(module_name)
            typer.echo(f"  ✅ {feature}")
        except ImportError:
            typer.echo(f"  ⚠️  {feature}: not installed")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
