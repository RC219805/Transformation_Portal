#!/usr/bin/env python3
"""Lux Depth V3 CLI - Full orchestrator pipeline with APEX quality tiers.

This CLI provides a complete command-line interface for the lux_depth_v3 pipeline,
including depth generation, PBR mapping, Materials V3, and V2 enhancement.

Usage:
    # APEX quality with all features (commercial-safe)
    lux-depth-v3 \\
        --input-dir ./input_images \\
        --output-dir ./output/apex \\
        --preset premium \\
        --quality-tier apex \\
        --depth-backend depth_anything_v3 \\
        --materials-v3 on \\
        --pbr on \\
        --cache-depth on \\
        --emit-master16 on \\
        --emit-upscaled16 on \\
        --emit-marketing on \\
        --emit-report on \\
        --emit-run-card on \\
        --overwrite

    # Research-only APEX+ with Depth Anything V3.1
    lux-depth-v3 \\
        --input-dir ./input_images \\
        --output-dir ./output/apex_da31 \\
        --preset depth-anything-v3.1-research-m4 \\
        --quality-tier apex \\
        --non-commercial-ok true \\
        --depth-device mps \\
        --materials-v3 on \\
        --pbr on \\
        --overwrite

    # Research-only APEX+ with Apple Depth Pro
    lux-depth-v3 \\
        --input-dir ./input_images \\
        --output-dir ./output/apex_depthpro \\
        --preset premium \\
        --quality-tier apex \\
        --depth-backend depth_pro \\
        --non-commercial-ok true \\
        --accept-apple-depth-pro-research-license true \\
        --depth-device mps \\
        --materials-v3 on \\
        --pbr on \\
        --overwrite
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import replace
from enum import Enum

try:
    import typer
except ImportError:
    print("Error: typer not installed. Install with: pip install typer", file=sys.stderr)
    sys.exit(1)

from .config import EnhanceConfig, ModelVariant, Preset
from .orchestrator import EnhanceOrchestrator
from .input_manager import ImageInput
from .security import HashMode

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="lux-depth-v3",
    help="Lux Depth V3 Pipeline - Depth + PBR + Materials V3 + Enhancement",
    add_completion=False,
)


class QualityTier(str, Enum):
    """Quality tiers for processing."""
    DRAFT = "draft"
    STANDARD = "standard"
    PREMIUM = "premium"
    APEX = "apex"


class OnOffAuto(str, Enum):
    """On/Off/Auto toggle."""
    ON = "on"
    OFF = "off"
    AUTO = "auto"


def _configure_logging(verbose: bool = False, quiet: bool = False, log_level: Optional[str] = None):
    """Configure logging at CLI entrypoint."""
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )


def _resolve_preset(preset_name: str) -> Optional[Preset]:
    """Resolve preset name to Preset enum."""
    preset_map = {
        "architectural_interior": Preset.ARCHITECTURAL_INTERIOR,
        "architectural_exterior": Preset.ARCHITECTURAL_EXTERIOR,
        "luxury_estate": Preset.LUXURY_ESTATE,
        "premium": Preset.LUXURY_ESTATE,  # Alias for premium quality
        "default": Preset.DEFAULT,
    }
    
    # Special presets for research models
    if "depth-anything-v3.1" in preset_name or "da3.1" in preset_name:
        # Use luxury estate as the base for research models
        return Preset.LUXURY_ESTATE
    
    return preset_map.get(preset_name.lower())


def _resolve_model_variant(preset_name: str, depth_backend: Optional[str]) -> Optional[ModelVariant]:
    """Resolve model variant from preset or backend."""
    # Depth Pro is handled separately via depth_backend
    if depth_backend == "depth_pro":
        return None
    
    # Check for DA3.1 research model in preset name
    if "depth-anything-v3.1" in preset_name.lower() or "da3.1" in preset_name.lower():
        # DA3.1 uses the large model variant
        return ModelVariant.METRIC_LARGE
    
    # Default to large for apex/premium quality
    return ModelVariant.METRIC_LARGE


def _apply_quality_tier(config: EnhanceConfig, tier: QualityTier) -> EnhanceConfig:
    """Apply quality tier settings to configuration."""
    if tier == QualityTier.DRAFT:
        # Fast preview - lower quality, faster processing
        return replace(
            config,
            model_variant=ModelVariant.METRIC_SMALL,
            pbr_normal_strength=0.8,
            pbr_roughness_strength=0.8,
            pbr_ao_strength=0.8,
            save_float_depth=False,
        )
    elif tier == QualityTier.STANDARD:
        # Balanced quality/speed
        return replace(
            config,
            model_variant=ModelVariant.METRIC_BASE,
            pbr_normal_strength=1.0,
            pbr_roughness_strength=1.0,
            pbr_ao_strength=1.0,
            save_float_depth=True,
        )
    elif tier == QualityTier.PREMIUM:
        # High quality
        return replace(
            config,
            model_variant=ModelVariant.METRIC_LARGE,
            pbr_normal_strength=1.2,
            pbr_roughness_strength=1.2,
            pbr_ao_strength=1.2,
            save_float_depth=True,
        )
    elif tier == QualityTier.APEX:
        # Maximum quality - all features enabled
        return replace(
            config,
            model_variant=ModelVariant.METRIC_LARGE,
            pbr_normal_strength=1.5,
            pbr_normal_blur_radius=0,  # Sharp normals for APEX
            pbr_roughness_strength=1.5,
            pbr_roughness_blur_radius=3,
            pbr_ao_strength=1.5,
            pbr_ao_blur_radius=5,
            pbr_ao_bias=0.5,
            save_float_depth=True,
            verify_depth_writes=True,
            enable_manifest_cache=True,
            chunked_hashing=True,
        )
    
    return config


def _parse_bool_flag(value: str) -> bool:
    """Parse boolean flag from string."""
    return value.lower() in ("true", "yes", "on", "1")


@app.command()
def process(
    # Input/Output
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        help="Input directory containing images to process"
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Output directory for all results"
    ),
    
    # Quality and Presets
    preset: str = typer.Option(
        "premium",
        "--preset",
        help="Processing preset (premium, architectural_interior, luxury_estate, etc.)"
    ),
    quality_tier: QualityTier = typer.Option(
        QualityTier.PREMIUM,
        "--quality-tier",
        help="Quality tier (draft, standard, premium, apex)"
    ),
    
    # Depth Backend
    depth_backend: Optional[str] = typer.Option(
        None,
        "--depth-backend",
        help="Depth backend (depth_anything_v3, depth_pro)"
    ),
    depth_device: str = typer.Option(
        "cpu",
        "--depth-device",
        help="Device for depth inference (cpu, cuda, mps)"
    ),
    
    # Feature Toggles
    materials_v3: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--materials-v3",
        help="Enable Materials V3 (on, off, auto)"
    ),
    pbr: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--pbr",
        help="Enable PBR map generation (on, off, auto)"
    ),
    cache_depth: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--cache-depth",
        help="Enable depth caching (on, off, auto)"
    ),
    
    # Output Deliverables
    emit_master16: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--emit-master16",
        help="Emit master 16-bit output (on, off, auto)"
    ),
    emit_upscaled16: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--emit-upscaled16",
        help="Emit upscaled 16-bit output (on, off, auto)"
    ),
    emit_marketing: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--emit-marketing",
        help="Emit marketing deliverables (on, off, auto)"
    ),
    emit_report: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--emit-report",
        help="Emit processing report (on, off, auto)"
    ),
    emit_run_card: OnOffAuto = typer.Option(
        OnOffAuto.AUTO,
        "--emit-run-card",
        help="Emit run card for reproducibility (on, off, auto)"
    ),
    
    # License Acknowledgments
    non_commercial_ok: str = typer.Option(
        "false",
        "--non-commercial-ok",
        help="Acknowledge non-commercial license restrictions (true/false)"
    ),
    accept_apple_depth_pro_research_license: str = typer.Option(
        "false",
        "--accept-apple-depth-pro-research-license",
        help="Accept Apple Depth Pro research license (true/false)"
    ),
    
    # Processing Options
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing outputs"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Limit number of images to process (for testing)"
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on first error"
    ),
    
    # Logging
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress all output except errors"
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Set log level (DEBUG, INFO, WARNING, ERROR)"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON"
    ),
):
    """Process images with lux_depth_v3 pipeline.
    
    This command runs the full orchestrator pipeline including:
    - Depth estimation (Depth Anything V3 or Depth Pro)
    - PBR map generation (normal, roughness, AO)
    - Materials V3 processing
    - V2 enhancement (if enabled)
    - Multiple output formats and deliverables
    """
    # Configure logging
    _configure_logging(verbose, quiet, log_level)
    
    # Validate inputs
    if not input_dir.exists():
        typer.echo(f"Error: Input directory not found: {input_dir}", err=True)
        raise typer.Exit(1)
    
    # Parse boolean flags
    non_commercial = _parse_bool_flag(non_commercial_ok)
    accept_depth_pro = _parse_bool_flag(accept_apple_depth_pro_research_license)
    
    # Validate research model usage
    if depth_backend == "depth_pro":
        if not non_commercial:
            typer.echo(
                "Error: Apple Depth Pro is research-only (AMLR license).\n"
                "You must set --non-commercial-ok true to use this backend.",
                err=True
            )
            raise typer.Exit(1)
        if not accept_depth_pro:
            typer.echo(
                "Error: Apple Depth Pro requires explicit license acceptance.\n"
                "You must set --accept-apple-depth-pro-research-license true to use this backend.",
                err=True
            )
            raise typer.Exit(1)
    
    # Resolve preset and model
    preset_enum = _resolve_preset(preset)
    model_variant = _resolve_model_variant(preset, depth_backend)
    
    # Create base configuration
    config = EnhanceConfig(
        preset=preset_enum,
        model_variant=model_variant,
        depth_device=depth_device,
        depth_backend=depth_backend,
        non_commercial_ok=non_commercial,
        accept_apple_depth_pro_research_license=accept_depth_pro,
    )
    
    # Apply quality tier
    config = _apply_quality_tier(config, quality_tier)
    
    # Apply feature toggles
    if pbr != OnOffAuto.AUTO:
        config = replace(config, generate_pbr=(pbr == OnOffAuto.ON))
    elif quality_tier == QualityTier.APEX:
        # APEX always enables PBR
        config = replace(config, generate_pbr=True)
    
    if cache_depth != OnOffAuto.AUTO:
        config = replace(config, enable_depth_cache=(cache_depth == OnOffAuto.ON))
    elif quality_tier == QualityTier.APEX:
        # APEX enables depth caching by default
        config = replace(config, enable_depth_cache=True)
    
    # Handle overwrite
    if overwrite:
        config = replace(config, force_depth=True, force_v2=True)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    image_paths.sort()
    
    if not image_paths:
        typer.echo(f"Error: No images found in {input_dir}", err=True)
        typer.echo(f"Looking for: {', '.join(image_extensions)}", err=True)
        raise typer.Exit(1)
    
    # Apply limit
    if limit:
        image_paths = image_paths[:limit]
    
    # Print configuration
    if not quiet:
        typer.echo(f"\n{'='*70}")
        typer.echo(f"Lux Depth V3 - {quality_tier.value.upper()} Quality Tier")
        typer.echo(f"{'='*70}")
        typer.echo(f"Input:          {input_dir}")
        typer.echo(f"Output:         {output_dir}")
        typer.echo(f"Images:         {len(image_paths)}")
        typer.echo(f"Preset:         {preset}")
        typer.echo(f"Quality Tier:   {quality_tier.value}")
        typer.echo(f"Depth Backend:  {depth_backend or 'depth_anything_v3 (default)'}")
        typer.echo(f"Depth Device:   {depth_device}")
        typer.echo(f"PBR:            {config.generate_pbr}")
        typer.echo(f"Materials V3:   {materials_v3.value}")
        typer.echo(f"Depth Cache:    {config.enable_depth_cache}")
        typer.echo(f"V2 Enhancement: {config.enable_v2}")
        if non_commercial:
            typer.echo(f"\n⚠️  NON-COMMERCIAL LICENSE MODE")
        typer.echo(f"{'='*70}\n")
    
    # Initialize orchestrator
    try:
        orchestrator = EnhanceOrchestrator(config, output_dir)
    except Exception as e:
        typer.echo(f"Error: Failed to initialize orchestrator: {e}", err=True)
        logger.exception("Orchestrator initialization failed")
        raise typer.Exit(1)
    
    # Process images
    successful = 0
    failed = 0
    failed_files = []
    
    for i, img_path in enumerate(image_paths, 1):
        if not quiet:
            typer.echo(f"[{i}/{len(image_paths)}] Processing: {img_path.name}...", nl=False)
        
        try:
            image_input = ImageInput(path=img_path)
            result = orchestrator.enhance_image(image_input, input_root=input_dir)
            
            if not quiet:
                typer.echo(" ✓")
            successful += 1
            
        except Exception as e:
            if not quiet:
                typer.echo(f" ✗ Error: {e}")
            logger.error(f"Failed to process {img_path.name}: {e}")
            failed += 1
            failed_files.append((img_path.name, str(e)))
            
            if fail_fast:
                typer.echo("\n[FAIL FAST] Aborting on first error", err=True)
                raise typer.Exit(1)
    
    # Summary
    if json_output:
        result = {
            "status": "partial" if failed > 0 else "success",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "total_files": len(image_paths),
            "success_count": successful,
            "error_count": failed,
            "failed_files": [{"file": f, "error": e} for f, e in failed_files],
            "quality_tier": quality_tier.value,
            "preset": preset,
        }
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"\n{'='*70}")
        typer.echo(f"Processing Complete")
        typer.echo(f"{'='*70}")
        typer.echo(f"Successful: {successful}/{len(image_paths)}")
        if failed > 0:
            typer.echo(f"Failed:     {failed}/{len(image_paths)}")
        typer.echo(f"Output:     {output_dir}/")
        typer.echo(f"{'='*70}\n")
        
        if failed_files and not quiet:
            typer.echo("Failed files:")
            for filename, error in failed_files:
                typer.echo(f"  • {filename}: {error}")
            typer.echo()
    
    # Exit with error if any files failed
    if failed > 0:
        raise typer.Exit(1)


@app.command()
def info():
    """Show lux_depth_v3 configuration and available options."""
    typer.echo("\nLux Depth V3 Pipeline Information\n")
    
    typer.echo("Quality Tiers:")
    typer.echo("  draft    - Fast preview (500-700 img/hr, small model)")
    typer.echo("  standard - Balanced (200-250 img/hr, base model)")
    typer.echo("  premium  - High quality (100-150 img/hr, large model)")
    typer.echo("  apex     - Maximum quality (50-100 img/hr, all features)")
    typer.echo()
    
    typer.echo("Presets:")
    typer.echo("  premium                  - Alias for luxury_estate")
    typer.echo("  luxury_estate           - Premium quality for luxury real estate")
    typer.echo("  architectural_interior  - Optimized for interior scenes")
    typer.echo("  architectural_exterior  - Optimized for exterior scenes")
    typer.echo("  default                 - Standard balanced configuration")
    typer.echo()
    
    typer.echo("Depth Backends:")
    typer.echo("  depth_anything_v3  - Depth Anything V3 (commercial-safe, default)")
    typer.echo("  depth_pro          - Apple Depth Pro (research-only, requires license)")
    typer.echo()
    
    typer.echo("Features:")
    typer.echo("  PBR Maps        - Normal, Roughness, Ambient Occlusion")
    typer.echo("  Materials V3    - Surface-aware enhancement")
    typer.echo("  Depth Cache     - Content-addressable caching")
    typer.echo("  V2 Enhancement  - AI-powered upscaling and refinement")
    typer.echo()
    
    typer.echo("For detailed documentation, see:")
    typer.echo("  docs/architecture/ADR-001-PBR-Integration-Architecture.md")
    typer.echo()


if __name__ == "__main__":
    app()
