#!/usr/bin/env python3
"""PBR CLI - Generate PBR maps from cached depth files.

This CLI provides a standalone entry point for PBR generation
without requiring the full enhancement orchestrator pipeline.

Usage:
    # Basic usage with preset
    python -m transformation_portal.lux_depth_v3.pbr_cli \
        --depth output/scene1_depth.npy \
        --preset premium \
        --output output/pbr/

    # Batch processing
    python -m transformation_portal.lux_depth_v3.pbr_cli \
        --depth-dir output/depth/ \
        --preset wood \
        --output output/pbr/

    # Custom parameters
    python -m transformation_portal.lux_depth_v3.pbr_cli \
        --depth output/scene1_depth.npy \
        --normal-strength 1.8 \
        --roughness-strength 1.5 \
        --ao-strength 1.3 \
        --output output/pbr/
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Optional
import logging
import time
from dataclasses import asdict, replace

try:
    import typer
except ImportError:
    print("Error: typer not installed. Install with: pip install typer", file=sys.stderr)
    sys.exit(1)

from .pbr_presets import get_preset, list_presets
from .pbr import PBRConfig
from .pbr_processor import PBRProcessor

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="pbr",
    help="Generate PBR maps from cached depth files",
    add_completion=False,
)


def _configure_logging(verbose: bool = False, quiet: bool = False, log_level: Optional[str] = None):
    """Configure logging at CLI entrypoint (not at import time)."""
    if quiet:
        level = logging.ERROR
    elif log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        force=True
    )


def _compute_config_fingerprint(config: PBRConfig) -> str:
    """Compute deterministic fingerprint of config for reproducibility tracking."""
    config_dict = asdict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


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
        help=f"PBR preset name. Available: {', '.join(list_presets())}"
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
    # Batch file selection controls
    pattern: str = typer.Option(
        "*_depth.*",
        "--pattern",
        help="Glob pattern for batch depth file selection (default: *_depth.*)"
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively search for depth files in subdirectories"
    ),
    # Information/listing
    list_presets_flag: bool = typer.Option(
        False,
        "--list-presets",
        help="List available presets and exit"
    ),
    # Logging/verbosity
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output"
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Explicit log level: DEBUG, INFO, WARNING, ERROR"
    ),
    # Output modes
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON (for automation/scripting)"
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        help="Write manifest of generated files to specified path"
    ),
    # Safety guardrails
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would be processed without actually running"
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Exit on first error (vs. continue on error)"
    ),
    max_files: Optional[int] = typer.Option(
        None,
        "--max-files",
        "--limit",
        help="Maximum number of files to process (safety limit)"
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Overwrite existing output files"
    ),
):
    """Generate PBR maps from cached depth file(s).

    Requires either --depth (single file) or --depth-dir (batch mode).
    """
    # Configure logging at entrypoint
    _configure_logging(verbose=verbose, quiet=quiet, log_level=log_level)

    # List presets and exit
    if list_presets_flag:
        available_presets = list_presets()
        typer.echo("\nAvailable PBR Presets:\n")
        for preset_name in available_presets:
            preset_config = get_preset(preset_name)
            typer.echo(f"  {preset_name:10s} - {_get_preset_description(preset_name)}")
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
            if not quiet:
                typer.echo(f"Using preset: {preset}")
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            typer.echo(f"\nAvailable presets: {', '.join(list_presets())}")
            raise typer.Exit(1)
    else:
        # Create default config
        config = PBRConfig()
        if not quiet:
            typer.echo("Using default PBR configuration")

    # Apply parameter overrides using replace() pattern
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
        config = replace(config, **overrides)
        if not quiet:
            typer.echo(f"Applied {len(overrides)} parameter override(s)")

    # Compute config fingerprint for reproducibility
    config_fingerprint = _compute_config_fingerprint(config)
    logger.debug(f"Config fingerprint: {config_fingerprint}")

    # Single file mode
    if depth:
        if not depth.exists():
            typer.echo(f"Error: Depth file not found: {depth}", err=True)
            raise typer.Exit(1)

        # Ensure output directory exists
        output.mkdir(parents=True, exist_ok=True)

        base = base_name or depth.stem.replace("_depth", "")

        if dry_run:
            typer.echo(f"\n[DRY RUN] Would process: {depth.name}")
            typer.echo(f"  Output directory: {output}/")
            typer.echo(f"  Base name: {base}")
            typer.echo(f"  Config fingerprint: {config_fingerprint}")
            return

        start_time = time.time()
        if not quiet:
            typer.echo(f"\nProcessing: {depth.name}")

        try:
            paths = PBRProcessor.from_cached_depth(
                depth_path=depth,
                config=config,
                output_dir=output,
                base_name=base
            )

            elapsed = time.time() - start_time

            if json_output:
                result = {
                    "status": "success",
                    "input": str(depth),
                    "output_dir": str(output),
                    "files": {k: str(v) for k, v in paths.items()},
                    "preset": preset,
                    "config_fingerprint": config_fingerprint,
                    "elapsed_seconds": round(elapsed, 3)
                }
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(f"✓ Generated PBR maps in {output}/ ({elapsed:.2f}s)")
                for map_type, path in paths.items():
                    typer.echo(f"  • {map_type:10s}: {path.name}")

            if manifest:
                _write_manifest(manifest, [paths], config_fingerprint, preset)

        except Exception as e:
            if json_output:
                result = {
                    "status": "error",
                    "input": str(depth),
                    "error": str(e)
                }
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(f"✗ Error: {e}", err=True)
            raise typer.Exit(1)

    # Batch mode
    else:
        if not depth_dir.exists():
            typer.echo(f"Error: Directory not found: {depth_dir}", err=True)
            raise typer.Exit(1)

        # Ensure output directory exists
        output.mkdir(parents=True, exist_ok=True)

        # Find depth files using restrictive pattern
        depth_files = []
        if recursive:
            depth_files = list(depth_dir.rglob(pattern))
        else:
            depth_files = list(depth_dir.glob(pattern))

        # Sort for deterministic ordering
        depth_files = sorted(depth_files)

        # Apply max_files limit
        if max_files and len(depth_files) > max_files:
            logger.warning(f"Limiting to {max_files} files (found {len(depth_files)})")
            depth_files = depth_files[:max_files]

        if not depth_files:
            typer.echo(f"Warning: No depth files matching '{pattern}' found in {depth_dir}", err=True)
            raise typer.Exit(1)

        if dry_run:
            typer.echo(f"\n[DRY RUN] Would process {len(depth_files)} file(s):")
            for depth_file in depth_files:
                typer.echo(f"  • {depth_file.relative_to(depth_dir)}")
            typer.echo(f"\nOutput directory: {output}/")
            typer.echo(f"Config fingerprint: {config_fingerprint}")
            return

        if not quiet:
            typer.echo(f"\nBatch processing {len(depth_files)} depth file(s)...\n")

        success_count = 0
        error_count = 0
        failed_files = []
        all_paths = []
        start_time = time.time()

        for depth_file in depth_files:
            base = depth_file.stem.replace("_depth", "")

            try:
                if not quiet:
                    typer.echo(f"Processing: {depth_file.name}...", nl=False)

                paths = PBRProcessor.from_cached_depth(
                    depth_path=depth_file,
                    config=config,
                    output_dir=output,
                    base_name=base
                )

                if not quiet:
                    typer.echo(" ✓")
                success_count += 1
                all_paths.append(paths)

            except Exception as e:
                if not quiet:
                    typer.echo(f" ✗ Error: {e}")
                logger.error(f"Failed to process {depth_file.name}: {e}")
                error_count += 1
                failed_files.append((depth_file.name, str(e)))

                if fail_fast:
                    typer.echo("\n[FAIL FAST] Aborting on first error", err=True)
                    raise typer.Exit(1)

        elapsed = time.time() - start_time

        # Summary
        if json_output:
            result = {
                "status": "partial" if error_count > 0 else "success",
                "input_dir": str(depth_dir),
                "output_dir": str(output),
                "total_files": len(depth_files),
                "success_count": success_count,
                "error_count": error_count,
                "failed_files": [{"file": f, "error": e} for f, e in failed_files],
                "preset": preset,
                "config_fingerprint": config_fingerprint,
                "elapsed_seconds": round(elapsed, 3)
            }
            typer.echo(json.dumps(result, indent=2))
        else:
            typer.echo(f"\nBatch complete ({elapsed:.2f}s):")
            typer.echo(f"  Success: {success_count}")
            typer.echo(f"  Errors:  {error_count}")
            typer.echo(f"  Output:  {output}/")

            # Show details of failures if any
            if failed_files:
                typer.echo(f"\nFailed files:")
                for filename, error in failed_files:
                    typer.echo(f"  • {filename}: {error}")

        if manifest:
            _write_manifest(manifest, all_paths, config_fingerprint, preset)

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


def _get_preset_description(preset_name: str) -> str:
    """Get human-readable description of preset."""
    descriptions = {
        "premium": "Maximum quality (hero shots, client deliverables)",
        "standard": "Balanced quality/speed (typical batch processing)",
        "draft": "Fast preview (quick iteration)",
        "wood": "Optimized for hardwood surfaces",
        "metal": "Optimized for metal surfaces",
        "glass": "Optimized for glass/reflective surfaces",
        "stone": "Optimized for stone/tile surfaces",
        "fabric": "Optimized for textile surfaces",
    }
    return descriptions.get(preset_name, "Custom preset")


def _write_manifest(manifest_path: Path, all_paths: list, config_fingerprint: str, preset: Optional[str]):
    """Write manifest of generated files."""
    manifest_data = {
        "config_fingerprint": config_fingerprint,
        "preset": preset,
        "generated_files": []
    }

    for paths in all_paths:
        manifest_data["generated_files"].append({k: str(v) for k, v in paths.items()})

    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    app()
