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
except ImportError:
    print(
        "Error: typer is required for the CLI. Install it with:\n"
        "  pip install typer\n"
        "or install the full package with:\n"
        "  pip install -e '.[dev]'",
        file=sys.stderr
    )
    sys.exit(1)


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

    # Import and run pipeline
    try:
        from transformation_portal.pipelines import lux_render_pipeline  # noqa: F401
        typer.echo("✅ Pipeline module loaded successfully")
        typer.echo("⚠️  Note: Full pipeline execution requires ML dependencies")
        typer.echo("   Install with: pip install -e '.[ml]'")
    except ImportError as e:
        typer.echo(f"❌ Error loading pipeline: {e}", err=True)
        typer.echo("   Install dependencies: pip install -e '.[ml]'")
        raise typer.Exit(code=1)


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

    try:
        from transformation_portal.pipelines import depth_tools  # noqa: F401
        typer.echo("✅ Depth tools module loaded successfully")
    except ImportError as e:
        typer.echo(f"❌ Error loading depth tools: {e}", err=True)
        raise typer.Exit(code=1)


# ============================================================================
# PROCESS SUBCOMMANDS
# ============================================================================

@process_app.command("material")
def process_material(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input image path"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output image path"),
    strength: float = typer.Option(0.7, "--strength", "-s", help="Enhancement strength (0.0-1.0)"),
    surfaces: Optional[str] = typer.Option(
        None, "--surfaces", help="Comma-separated surface types (wood,metal,glass,fabric,stone)"
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

    try:
        from transformation_portal.processors.material_response import core  # noqa: F401
        typer.echo("✅ Material Response module loaded successfully")
    except ImportError as e:
        typer.echo(f"❌ Error loading Material Response: {e}", err=True)
        raise typer.Exit(code=1)


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

    try:
        from transformation_portal.processors import luxury_video_master_grader  # noqa: F401
        typer.echo("✅ Video Master Grader module loaded successfully")
        typer.echo("⚠️  Note: FFmpeg is required for video processing")
    except ImportError as e:
        typer.echo(f"❌ Error loading Video Master Grader: {e}", err=True)
        raise typer.Exit(code=1)


@process_app.command("tiff")
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

    try:
        from transformation_portal.analyzers import codebase_philosophy_auditor  # noqa: F401
        typer.echo("✅ Auditor module loaded successfully")
    except ImportError as e:
        typer.echo(f"❌ Error loading auditor: {e}", err=True)
        raise typer.Exit(code=1)


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

    try:
        from transformation_portal.analyzers import decision_decay_dashboard  # noqa: F401
        typer.echo("✅ Dashboard module loaded successfully")
    except ImportError as e:
        typer.echo(f"❌ Error loading dashboard: {e}", err=True)
        raise typer.Exit(code=1)


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

    try:
        from transformation_portal.analyzers import parse_workflows  # noqa: F401
        typer.echo("✅ Workflow parser module loaded successfully")
    except ImportError as e:
        typer.echo(f"❌ Error loading workflow parser: {e}", err=True)
        raise typer.Exit(code=1)


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
# MAIN UNIFIED CLI (For development/testing)
# ============================================================================

# Register subcommands with main app (for unified CLI during development)
app.add_typer(render_app, name="render")
app.add_typer(process_app, name="process")
app.add_typer(analyze_app, name="analyze")


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
    "render_cli",
    "process_cli",
    "analyze_cli",
    "main",
    "version",
    "info",
]


if __name__ == "__main__":
    main()
