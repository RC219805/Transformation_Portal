#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformation Portal - Main CLI Entry Point

This module provides the primary command-line interface for the Transformation
Portal, enabling unified access to all pipeline and processing capabilities.

Usage:
    python -m transformation_portal --help
    python -m transformation_portal process -i "inputs/*.jpg" -r path/to/recipe.yaml -o output/
    python -m transformation_portal list-recipes
    python -m transformation_portal validate-recipe path/to/recipe.yaml

The CLI supports:
    - Recipe-driven batch processing
    - Dry-run mode for previewing processing plans
    - Quality feedback with RAG integration
    - 4K upscaling with Rendering 4K Pipeline
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

app = typer.Typer(
    name="transformation-portal",
    help="Professional image and video processing toolkit for luxury real estate rendering",
    no_args_is_help=True,
    add_completion=False,
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


@app.command()
def process(
    input_glob: str = typer.Option(..., "--input", "-i", help="Input glob pattern (e.g., 'inputs/*.jpg')"),
    recipe: Path = typer.Option(..., "--recipe", "-r", help="Recipe YAML file path"),
    output: Path = typer.Option(Path("./final"), "--output", "-o", help="Output directory"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Processing mode: auto|image|video"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Preview processing plan without executing"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Enable parallel processing"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Logging level: debug|info|warning|error"),
):
    """Run unified enhancement pipeline.

    Process images through the unified pipeline using a YAML recipe
    configuration. Supports batch processing with dry-run preview, quality
    feedback, and bounded parallel execution.

    Example:
        transformation-portal process -i "renders/*.exr" -r path/to/recipe.yaml -o output/
    """
    import logging

    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(
        level=log_levels.get(log_level.lower(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    typer.echo("🚀 Transformation Portal - Unified Pipeline")
    typer.echo(f"   Recipe: {recipe}")
    typer.echo(f"   Input: {input_glob}")
    typer.echo(f"   Output: {output}")
    typer.echo(f"   Mode: {mode}")
    typer.echo(f"   Dry run: {dry_run}")
    typer.echo(f"   Parallel: {parallel}")
    typer.echo()

    if not recipe.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    try:
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(recipe)
        result = pipeline.process_batch(
            input_glob,
            output,
            mode=mode,
            dry_run=dry_run,
            parallel=parallel,
        )

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
    recipes_dir: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Recipe directory path (defaults to config/recipes, then config/ recursion)",
    ),
):
    """List all available recipe presets.

    Scans the preferred recipe locations and displays available UnifiedPipeline
    recipes with their descriptions and configurations.

    Example:
        transformation-portal list-recipes
        transformation-portal list-recipes -d path/to/recipes/
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


@app.command("validate-recipe")
def validate_recipe(
    recipe: Path = typer.Argument(..., help="Recipe YAML file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
):
    """Validate recipe configuration.

    Validates a recipe YAML file against the schema and reports any errors or
    warnings.

    Example:
        transformation-portal validate-recipe path/to/recipe.yaml
        transformation-portal validate-recipe custom_recipe.yaml -v
    """
    typer.echo(f"🔍 Validating recipe: {recipe}\n")

    if not recipe.exists():
        typer.echo(f"❌ Error: Recipe file not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    try:
        validation_result = validate_recipe_file(recipe)

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
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
