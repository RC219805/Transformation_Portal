#!/usr/bin/env python3
"""PBR CLI - Generate PBR maps from cached depth files.

This CLI provides a standalone entry point for PBR generation
without requiring the full enhancement orchestrator pipeline.

Usage:
    # Basic usage with preset
    python -m transformation_portal.lux_depth_v3.pbr_cli \\
        --depth output/scene1_depth.npy \\
        --preset premium \\
        --output output/pbr/

    # Batch processing
    python -m transformation_portal.lux_depth_v3.pbr_cli \\
        --depth-dir output/depth/ \\
        --preset wood \\
        --output output/pbr/

    # Custom parameters
    python -m transformation_portal.lux_depth_v3.pbr_cli \\
        --depth output/scene1_depth.npy \\
        --normal-strength 1.8 \\
        --roughness-strength 1.5 \\
        --ao-strength 1.3 \\
        --output output/pbr/
"""

import sys
from pathlib import Path
from typing import Optional
import logging

try:
    import typer
except ImportError:
    print("Error: typer not installed. Install with: pip install typer", file=sys.stderr)
    sys.exit(1)

from .pbr_presets import get_preset, list_presets
from .pbr import PBRConfig
from .pbr_processor import PBRProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="pbr",
    help="Generate PBR maps from cached depth files",
    add_completion=False,
)


@app.command()
def generate(
    depth: Optional[Path] = typer.Option(
        None,
        "--depth",
        "-d",
        help="Path to single depth file (.npy or .png)"
    ),
    depth_dir: Optional[Path] = typer.Option(
        None,
        "--depth-dir",
        help="Directory containing depth files (batch mode)"
    ),
    output: Path = typer.Option(
        Path("./pbr"),
        "--output",
        "-o",
        help="Output directory for PBR maps"
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="PBR preset name (premium, wood, metal, glass, stone, fabric)"
    ),
    base_name: Optional[str] = typer.Option(
        None,
        "--base-name",
        "-n",
        help="Base name for output files (auto-derived from depth filename if omitted)"
    ),
    # Custom parameters (override preset)
    normal_strength: Optional[float] = typer.Option(
        None,
        "--normal-strength",
        help="Normal map strength multiplier (overrides preset)"
    ),
    roughness_strength: Optional[float] = typer.Option(
        None,
        "--roughness-strength",
        help="Roughness map strength multiplier (overrides preset)"
    ),
    ao_strength: Optional[float] = typer.Option(
        None,
        "--ao-strength",
        help="Ambient occlusion strength multiplier (overrides preset)"
    ),
    ao_bias: Optional[float] = typer.Option(
        None,
        "--ao-bias",
        help="AO bias (0.0-1.0, lower=darker) (overrides preset)"
    ),
    list_presets_flag: bool = typer.Option(
        False,
        "--list-presets",
        help="List available presets and exit"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
):
    """Generate PBR maps from cached depth file(s).

    Requires either --depth (single file) or --depth-dir (batch mode).
    """
    # Setup logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List presets and exit
    if list_presets_flag:
        typer.echo("\nAvailable PBR Presets:\n")
        typer.echo("  premium  - Maximum quality (hero shots, client deliverables)")
        typer.echo("  standard - Balanced quality/speed (typical batch processing)")
        typer.echo("  draft    - Fast preview (quick iteration)")
        typer.echo("  wood     - Optimized for hardwood surfaces")
        typer.echo("  metal    - Optimized for metal surfaces")
        typer.echo("  glass    - Optimized for glass/reflective surfaces")
        typer.echo("  stone    - Optimized for stone/tile surfaces")
        typer.echo("  fabric   - Optimized for textile surfaces")
        typer.echo()
        return

    # Validate inputs
    if not depth and not depth_dir:
        typer.echo("Error: Either --depth or --depth-dir required", err=True)
        raise typer.Exit(1)

    if depth and depth_dir:
        typer.echo("Error: Cannot specify both --depth and --depth-dir", err=True)
        raise typer.Exit(1)

    # Load preset or create custom config
    if preset:
        try:
            enhance_config = get_preset(preset)
            config = enhance_config.to_pbr_config()
            typer.echo(f"Using preset: {preset}")
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            typer.echo(f"\nAvailable presets: {', '.join(list_presets())}")
            raise typer.Exit(1)
    else:
        # Create default config
        config = PBRConfig()
        typer.echo("Using default PBR configuration")

    # Apply parameter overrides
    if normal_strength is not None:
        config = PBRConfig(
            normal_strength=normal_strength,
            normal_blur_radius=config.normal_blur_radius,
            roughness_strength=config.roughness_strength,
            roughness_blur_radius=config.roughness_blur_radius,
            ao_strength=config.ao_strength,
            ao_blur_radius=config.ao_blur_radius,
            ao_bias=config.ao_bias,
        )

    # Similar for other overrides (frozen dataclass requires recreation)
    overrides = {}
    if normal_strength is not None:
        overrides['normal_strength'] = normal_strength
    if roughness_strength is not None:
        overrides['roughness_strength'] = roughness_strength
    if ao_strength is not None:
        overrides['ao_strength'] = ao_strength
    if ao_bias is not None:
        overrides['ao_bias'] = ao_bias

    if overrides:
        from dataclasses import replace
        config = replace(config, **overrides)
        typer.echo(f"Applied {len(overrides)} parameter override(s)")

    # Single file mode
    if depth:
        if not depth.exists():
            typer.echo(f"Error: Depth file not found: {depth}", err=True)
            raise typer.Exit(1)

        base = base_name or depth.stem.replace("_depth", "")

        typer.echo(f"\nProcessing: {depth.name}")
        try:
            paths = PBRProcessor.from_cached_depth(
                depth_path=depth,
                config=config,
                output_dir=output,
                base_name=base
            )

            typer.echo(f"✓ Generated PBR maps in {output}/")
            for map_type, path in paths.items():
                typer.echo(f"  • {map_type:10s}: {path.name}")

        except Exception as e:
            typer.echo(f"✗ Error: {e}", err=True)
            raise typer.Exit(1)

    # Batch mode
    else:
        if not depth_dir.exists():
            typer.echo(f"Error: Directory not found: {depth_dir}", err=True)
            raise typer.Exit(1)

        # Find all depth files
        depth_files = []
        for ext in ['.npy', '.png']:
            depth_files.extend(depth_dir.glob(f"*{ext}"))

        if not depth_files:
            typer.echo(f"Warning: No depth files (.npy, .png) found in {depth_dir}", err=True)
            raise typer.Exit(1)

        typer.echo(f"\nBatch processing {len(depth_files)} depth file(s)...\n")

        success_count = 0
        error_count = 0

        for depth_file in sorted(depth_files):
            base = depth_file.stem.replace("_depth", "")

            try:
                typer.echo(f"Processing: {depth_file.name}...", nl=False)
                paths = PBRProcessor.from_cached_depth(
                    depth_path=depth_file,
                    config=config,
                    output_dir=output,
                    base_name=base
                )
                typer.echo(" ✓")
                success_count += 1

            except Exception as e:
                typer.echo(f" ✗ Error: {e}")
                error_count += 1

        # Summary
        typer.echo(f"\nBatch complete:")
        typer.echo(f"  Success: {success_count}")
        typer.echo(f"  Errors:  {error_count}")
        typer.echo(f"  Output:  {output}/")

        if error_count > 0:
            raise typer.Exit(1)


@app.command()
def info():
    """Show PBR configuration information."""
    typer.echo("\nPBR Map Generation Parameters:\n")
    typer.echo("Normal Map:")
    typer.echo("  • Encodes surface gradients as RGB")
    typer.echo("  • Strength: multiplier for gradient magnitude")
    typer.echo("  • Blur: pre-smoothing before gradient (0=sharp)")
    typer.echo()
    typer.echo("Roughness Map:")
    typer.echo("  • Encodes surface micro-detail (Laplacian)")
    typer.echo("  • Strength: multiplier for detail sensitivity")
    typer.echo("  • Blur: smoothing kernel size")
    typer.echo()
    typer.echo("Ambient Occlusion:")
    typer.echo("  • Approximates indirect lighting/shadows")
    typer.echo("  • Strength: multiplier for occlusion intensity")
    typer.echo("  • Bias: brightness offset (0.0=dark, 1.0=bright)")
    typer.echo()


if __name__ == "__main__":
    app()
