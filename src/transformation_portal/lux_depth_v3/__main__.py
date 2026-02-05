#!/usr/bin/env python3
"""Lux Depth V3 Pipeline - Main CLI Entry Point.

APEX command variants for the lux_depth_v3 pipeline supporting:
- Commercial-safe APEX mode (default)
- Research-only APEX+ variants (explicit opt-in)

Usage:
    # Commercial-safe APEX (default, no research opt-ins)
    lux-depth-v3 \\
        --input-dir "./input_images" \\
        --output-dir "./output/lux_depth_v3_apex" \\
        --quality-tier "apex" \\
        --depth-backend "depth_anything_v3" \\
        --depth-device "mps" \\
        --materials-v3 "on" \\
        --pbr "on" \\
        --cache-depth "on" \\
        --emit-master16 "on" \\
        --emit-upscaled16 "on" \\
        --emit-marketing "on" \\
        --emit-report "on" \\
        --emit-run-card "on" \\
        --overwrite

    # Research-only: Depth Anything V3.1 (explicit opt-in)
    lux-depth-v3 \\
        --input-dir "./input_images" \\
        --output-dir "./output/lux_depth_v3_apex_da31" \\
        --quality-tier "apex" \\
        --preset "depth-anything-v3.1-research-m4" \\
        --non-commercial-ok "true" \\
        --depth-device "mps" \\
        --materials-v3 "on" \\
        --pbr "on" \\
        --cache-depth "on" \\
        --emit-master16 "on" \\
        --emit-upscaled16 "on" \\
        --emit-marketing "on" \\
        --emit-report "on" \\
        --emit-run-card "on" \\
        --overwrite

    # Research-only: Apple Depth Pro (explicit opt-in)
    lux-depth-v3 \\
        --input-dir "./input_images" \\
        --output-dir "./output/lux_depth_v3_apex_depthpro" \\
        --quality-tier "apex" \\
        --preset "apple-depth-pro-research" \\
        --depth-backend "depth_pro" \\
        --non-commercial-ok "true" \\
        --accept-apple-depth-pro-research-license "true" \\
        --depth-device "mps" \\
        --materials-v3 "on" \\
        --pbr "on" \\
        --cache-depth "on" \\
        --emit-master16 "on" \\
        --emit-upscaled16 "on" \\
        --emit-marketing "on" \\
        --emit-report "on" \\
        --emit-run-card "on" \\
        --overwrite

    # PBR-only workflow (skip V2 enhancement)
    lux-depth-v3 \\
        --input-dir "./input_images" \\
        --output-dir "./output/lux_depth_v3_pbr_only" \\
        --quality-tier "apex" \\
        --enable-v2 "off" \\
        --pbr "on" \\
        --depth-device "mps"

    # Module invocation (if console script not on PATH)
    python -m transformation_portal.lux_depth_v3 [args]

Notes:
    - --quality-tier: standard | premium | apex (quality level)
    - --preset: named preset like "default", "depth-anything-v3.1-research-m4", etc.
    - --enable-v2: on | off (control V2 enhancement stage, default: on)
    - --v2-preset: preset name or "none" to skip V2 (default: "default")
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("Error: typer not installed. Install with: pip install typer", file=sys.stderr)
    sys.exit(1)

from .config import EnhanceConfig, Preset
from .orchestrator import EnhanceOrchestrator

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="lux-depth-v3",
    help="Lux Depth V3 Pipeline - Orchestrated depth + enhancement with APEX quality tier support",
    add_completion=False,
)


def _parse_bool_flag(value: str) -> bool:
    """Parse string boolean flags (on/off, true/false, yes/no, 1/0)."""
    if isinstance(value, bool):
        return value
    normalized = value.lower().strip()
    return normalized in ("on", "true", "yes", "1")


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

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


@app.command()
def main(
    # I/O Paths
    input_dir: Path = typer.Option(..., "--input-dir", help="Input directory containing images to process"),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Output directory for all artifacts (depth, PBR, enhanced images, manifests)"
    ),
    # Preset and Quality
    preset: str = typer.Option(
        "premium", "--preset", help="Pipeline preset (premium, depth-anything-v3.1-research-m4, default, etc.)"
    ),
    quality_tier: str = typer.Option("standard", "--quality-tier", help="Quality tier: standard, premium, or apex"),
    # Depth Backend Configuration
    depth_backend: Optional[str] = typer.Option(
        None, "--depth-backend", help="Depth backend: depth_anything_v3 (default), depth_pro (research-only)"
    ),
    depth_device: str = typer.Option("cpu", "--depth-device", help="Device for depth inference: cpu, cuda, mps"),
    # Materials V3 and PBR
    materials_v3: str = typer.Option("off", "--materials-v3", help="Enable Materials V3 surface-aware finishing: on/off"),
    pbr: str = typer.Option("off", "--pbr", help="Enable PBR map generation (normal, roughness, AO): on/off"),
    # Caching
    cache_depth: str = typer.Option("off", "--cache-depth", help="Enable content-addressable depth cache: on/off"),
    # V2 Enhancement Stage
    enable_v2: str = typer.Option("on", "--enable-v2", help="Enable V2 enhancement stage: on/off (default: on)"),
    v2_preset: Optional[str] = typer.Option(
        "default", "--v2-preset", help="V2 enhancement preset (use 'none' to skip V2, default: default)"
    ),
    # Emit Options (Deliverables)
    emit_master16: str = typer.Option("off", "--emit-master16", help="Emit master 16-bit output: on/off"),
    emit_upscaled16: str = typer.Option("off", "--emit-upscaled16", help="Emit upscaled 16-bit output: on/off"),
    emit_marketing: str = typer.Option("off", "--emit-marketing", help="Emit marketing-ready output: on/off"),
    emit_report: str = typer.Option("on", "--emit-report", help="Emit processing report: on/off"),
    emit_run_card: str = typer.Option("on", "--emit-run-card", help="Emit run card for reproducibility: on/off"),
    # License and Research Acknowledgements
    non_commercial_ok: str = typer.Option(
        "false", "--non-commercial-ok", help="Acknowledge non-commercial license restrictions (CC BY-NC 4.0): true/false"
    ),
    accept_apple_depth_pro_research_license: str = typer.Option(
        "false", "--accept-apple-depth-pro-research-license", help="Accept Apple Depth Pro research license (AMLR): true/false"
    ),
    # Processing Flags
    overwrite: bool = typer.Option(False, "--overwrite", help="Force reprocessing even if outputs exist"),
    force_depth: bool = typer.Option(False, "--force-depth", help="Force depth recomputation (ignore cache)"),
    # Logging
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress all output except errors"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Set log level: DEBUG, INFO, WARNING, ERROR"),
):
    """Process images through the Lux Depth V3 pipeline with APEX quality tier support.

    This CLI provides orchestrated depth estimation + V2 enhancement with support for:
    - Commercial-safe APEX mode (default)
    - Research-only APEX+ variants (explicit opt-in with license acknowledgements)
    - Materials V3 surface-aware finishing
    - PBR map generation
    - Multiple output formats and deliverables
    """
    _configure_logging(verbose, quiet, log_level)

    # Parse boolean flags
    enable_materials_v3 = _parse_bool_flag(materials_v3)
    enable_pbr = _parse_bool_flag(pbr)
    enable_cache_depth = _parse_bool_flag(cache_depth)
    enable_v2_bool = _parse_bool_flag(enable_v2)
    enable_emit_master16 = _parse_bool_flag(emit_master16)
    enable_emit_upscaled16 = _parse_bool_flag(emit_upscaled16)
    enable_emit_marketing = _parse_bool_flag(emit_marketing)
    enable_emit_report = _parse_bool_flag(emit_report)
    enable_emit_run_card = _parse_bool_flag(emit_run_card)
    enable_non_commercial = _parse_bool_flag(non_commercial_ok)
    enable_apple_license = _parse_bool_flag(accept_apple_depth_pro_research_license)

    # Parse V2 preset (convert "none" string to None for skipping V2)
    v2_preset_value = None if (v2_preset and v2_preset.lower() == "none") else v2_preset

    # Validate input directory
    if not input_dir.exists():
        error_msg = f"Input directory does not exist: {input_dir}"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    # Validate non-commercial usage
    if depth_backend == "depth_pro" and not enable_non_commercial:
        error_msg = "Depth Pro backend requires --non-commercial-ok true (AMLR research-only license)"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    if depth_backend == "depth_pro" and not enable_apple_license:
        error_msg = "Depth Pro backend requires --accept-apple-depth-pro-research-license true (Apple research-only)"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    if "v3.1" in preset.lower() and not enable_non_commercial:
        error_msg = f"Preset '{preset}' requires --non-commercial-ok true (CC BY-NC 4.0 non-commercial license)"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate quality tier
    valid_quality_tiers = ["standard", "premium", "apex"]
    if quality_tier.lower() not in valid_quality_tiers:
        error_msg = f"Invalid quality tier '{quality_tier}'. Must be one of: {', '.join(valid_quality_tiers)}"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    # Build configuration
    logger.info(f"Configuring pipeline with quality tier: {quality_tier}")

    # Map preset to Preset enum if possible
    preset_enum = None
    preset_lower = preset.lower().replace("-", "_")
    for p in Preset:
        if p.value.lower() == preset_lower:
            preset_enum = p
            break

    # Log warning if preset doesn't map to enum
    if preset_enum is None and preset != "premium":  # "premium" is default string, doesn't have enum
        logger.warning(f"Preset '{preset}' does not map to a known Preset enum. Continuing with string value.")

    config = EnhanceConfig(
        preset=preset_enum,
        depth_device=depth_device,
        depth_backend=depth_backend,
        non_commercial_ok=enable_non_commercial,
        accept_apple_depth_pro_research_license=enable_apple_license,
        force_depth=force_depth or overwrite,
        enable_depth_cache=enable_cache_depth,
        enable_v2=enable_v2_bool,
        v2_preset=v2_preset_value,
        generate_pbr=enable_pbr,
        quality_tier=quality_tier,
        enable_materials_v3=enable_materials_v3,
        emit_master16=enable_emit_master16,
        emit_upscaled16=enable_emit_upscaled16,
        emit_marketing=enable_emit_marketing,
        emit_report=enable_emit_report,
        emit_run_card=enable_emit_run_card,
    )

    # Create orchestrator
    logger.info(f"Initializing orchestrator with output dir: {output_dir}")
    orchestrator = EnhanceOrchestrator(config=config, output_root=output_dir)

    # Discover images
    logger.info(f"Discovering images in: {input_dir}")
    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"**/*{ext}"))
        image_files.extend(input_dir.glob(f"**/*{ext.upper()}"))

    if not image_files:
        error_msg = f"No images found in {input_dir}"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    logger.info(f"Found {len(image_files)} images to process")

    # Process batch
    try:
        results = orchestrator.enhance_batch(input_dir=input_dir, image_extensions=image_extensions)

        # Summary
        successful = sum(1 for r in results if r.get("status") == "success")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        failed = sum(1 for r in results if r.get("status") == "error")

        logger.info("\nProcessing complete:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Failed: {failed}")

        if failed > 0:
            raise typer.Exit(code=1)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(code=1)

    logger.info("✅ All processing complete")


if __name__ == "__main__":
    app()
