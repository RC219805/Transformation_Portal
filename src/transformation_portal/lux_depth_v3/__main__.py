#!/usr/bin/env python3
"""Lux Depth V3 Pipeline - Main CLI Entry Point.

APEX command variants for the lux_depth_v3 pipeline supporting:
- Commercial-safe APEX mode (default)
- Research-only APEX+ variants (explicit opt-in)
- PBR-only workflows (skip V2 enhancement)
- Material segmentation (stub, EfficientSAM, or SAM2 backends)

Usage:
    # Commercial-safe APEX (default, no research opt-ins)
    lux-depth-v3 \\
        --input-dir "./input_images" \\
        --output-dir "./output/lux_depth_v3_apex" \\
        --quality-tier "apex" \\
        --depth-backend "da3" \\
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
        --depth-device "mps" \\
        --emit-master16 "on"

    # Material segmentation with EfficientSAM backend
    lux-depth-v3 \\
        --input-dir "./input_images" \\
        --output-dir "./output/lux_depth_v3_segmented" \\
        --quality-tier "apex" \\
        --materials-v3 "on" \\
        --enable-segmentation "on" \\
        --segmentation-backend "efficientsam" \\
        --depth-device "mps"

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
        --depth-pro-python "./.venv-depth-pro/bin/python" \\
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

    # Scene-level reconstruction (research-only, grouped by parent folder)
    lux-depth-v3 \\
        --input-dir "./input_images" \\
        --output-dir "./output/lux_depth_v3_with_reconstruction" \\
        --quality-tier "apex" \\
        --depth-backend "depth_pro" \\
        --depth-pro-python "./.venv-depth-pro/bin/python" \\
        --materials-v3 "on" \\
        --enable-segmentation "on" \\
        --segmentation-backend "sam2" \\
        --strict-segmentation \\
        --enable-reconstruction "on" \\
        --grouping-mode "parent_dir" \\
        --non-commercial-ok "true" \\
        --accept-research-tools-license "true" \\
        --accept-apple-depth-pro-research-license "true"

    # Module invocation (if console script not on PATH)
    python -m transformation_portal.lux_depth_v3 [args]

Key Concepts:
    - V2 Enhancement: Optional AI-powered refinement stage (enabled by default)
      * Use --enable-v2 "off" to skip V2 entirely (PBR-only workflows)
      * Use --v2-preset "none" to skip V2 preset while keeping validation

    - Quality vs Preset:
      * --quality-tier: Controls output quality
        (standard|premium|apex) - use for most
        workflows
      * --preset: Named configuration for
        specialized scenarios - overrides
        quality-tier

    - PBR-Only Workflows:
      * Add --enable-v2 "off" to skip V2 enhancement validation and execution
      * Faster processing, still produces high-quality depth and PBR maps

    - Material Segmentation:
      * Use --enable-segmentation "on" to enable
        automatic material detection
      * --segmentation-backend: Choose "stub"
        (default), "efficientsam", or "sam2"
      * --strict-segmentation: Fail on backend
        errors instead of falling back to stub

    - Research Models:
      * Require explicit license acknowledgement
        (--non-commercial-ok "true")
      * Depth Pro also requires
        --accept-apple-depth-pro-research-license
        "true"

Troubleshooting:
    - "Script not found" error: Add --enable-v2
      "off" to disable V2 enhancement
    - See docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md
      for complete troubleshooting guide
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print(
        "Error: typer not installed." + " Install with: pip install typer",
        file=sys.stderr,
    )
    sys.exit(1)

from ._backend_contract import backend_alias_warning, is_legacy_backend_alias, normalize_backend_id
from .config import EnhanceConfig, Preset
from .orchestrator import EnhanceOrchestrator

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="lux-depth-v3",
    help=("Lux Depth V3 Pipeline - Orchestrated depth + enhancement with APEX " "quality tier support"),
    add_completion=False,
)


def _parse_bool_flag(value: str) -> bool:
    """Parse string boolean flags (on/off, true/false, yes/no, 1/0)."""
    if isinstance(value, bool):
        return value
    normalized = value.lower().strip()
    return normalized in ("on", "true", "yes", "1")


def _configure_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_level: Optional[str] = None,
) -> None:
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
        format="%(levelname)s: %(message)s",
        force=True,
    )


@app.command()
def main(
    # I/O Paths
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        help="Input directory containing images to process",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help=("Output directory for all artifacts " "(depth, PBR, enhanced images, manifests)"),
    ),
    # Preset and Quality
    preset: str = typer.Option(
        "premium",
        "--preset",
        help=(
            "Named pipeline configuration"
            " (premium,"
            " depth-anything-v3.1-research-m4,"
            " etc.). Optional - use"
            " --quality-tier for most workflows."
        ),
    ),
    quality_tier: str = typer.Option(
        "standard",
        "--quality-tier",
        help=(
            "Output quality level: standard"
            " (fast/draft), premium (balanced),"
            " or apex (maximum quality)."
            " Controls processing resolution"
            " and features."
        ),
    ),
    # Depth Backend Configuration
    depth_backend: Optional[str] = typer.Option(
        None,
        "--depth-backend",
        help=("Depth backend: da3 (default, commercial), depth_pro " "(research-only, metric depth)"),
    ),
    depth_pro_python: Optional[str] = typer.Option(
        None,
        "--depth-pro-python",
        help=(
            "Optional Python executable for an isolated Depth Pro environment. "
            "Use this to keep depth_pro out of the main Transformation Portal venv."
        ),
    ),
    depth_device: str = typer.Option(
        "cpu",
        "--depth-device",
        help="Device for depth inference:" " cpu, cuda, mps",
    ),
    # Materials V3 and PBR
    materials_v3: str = typer.Option(
        "off",
        "--materials-v3",
        help="Enable Materials V3" " surface-aware finishing: on/off",
    ),
    pbr: str = typer.Option(
        "off",
        "--pbr",
        help="Enable PBR map generation" " (normal, roughness, AO): on/off",
    ),
    save_float_depth: str = typer.Option(
        "off",
        "--save-float-depth",
        help=("Save canonical float depth artifact (.npy) alongside preview PNG " "depth: on/off"),
    ),
    # Material Segmentation
    enable_segmentation: str = typer.Option(
        "off",
        "--enable-segmentation",
        help="Enable automatic material segmentation: on/off (default: off)",
    ),
    segmentation_backend: str = typer.Option(
        "stub",
        "--segmentation-backend",
        help=("Segmentation backend: stub (default, no segmentation), " + "efficientsam, sam2"),
    ),
    sam2_model_size: str = typer.Option(
        "base",
        "--sam2-model-size",
        help="SAM2 model size (base|large) when --segmentation-backend sam2",
    ),
    sam2_checkpoint_path: Optional[Path] = typer.Option(
        None,
        "--sam2-checkpoint-path",
        help=("Optional path to SAM2 checkpoint (.pt) when " "--segmentation-backend sam2"),
    ),
    strict_segmentation: bool = typer.Option(
        False,
        "--strict-segmentation",
        help=("Fail on segmentation backend errors instead of " + "falling back to stub"),
    ),
    # Caching
    cache_depth: str = typer.Option(
        "off",
        "--cache-depth",
        help="Enable content-addressable" " depth cache: on/off",
    ),
    # V2 Enhancement Stage
    enable_v2: str = typer.Option(
        "on",
        "--enable-v2",
        help=(
            "Enable V2 AI-powered enhancement"
            " stage: on/off (default: on)."
            " Set to 'off' for PBR-only"
            " workflows or when enhancement"
            " script is unavailable."
        ),
    ),
    v2_preset: Optional[str] = typer.Option(
        "default",
        "--v2-preset",
        help=(
            "V2 enhancement preset name or"
            " 'none' to skip enhancement"
            " (default: default). Only used"
            " when --enable-v2 is on."
        ),
    ),
    # Emit Options (Deliverables)
    emit_master16: str = typer.Option(
        "off",
        "--emit-master16",
        help="Emit master 16-bit output: on/off",
    ),
    emit_upscaled16: str = typer.Option(
        "off",
        "--emit-upscaled16",
        help="Emit upscaled 16-bit output: on/off",
    ),
    emit_marketing: str = typer.Option(
        "off",
        "--emit-marketing",
        help="Emit marketing-ready output: on/off",
    ),
    emit_report: str = typer.Option(
        "on",
        "--emit-report",
        help="Emit processing report: on/off",
    ),
    emit_run_card: str = typer.Option(
        "on",
        "--emit-run-card",
        help="Emit run card for reproducibility: on/off",
    ),
    # License and Research Acknowledgements
    non_commercial_ok: str = typer.Option(
        "false",
        "--non-commercial-ok",
        help=("Acknowledge non-commercial license restrictions " + "(CC BY-NC 4.0): true/false"),
    ),
    accept_apple_depth_pro_research_license: str = typer.Option(
        "false",
        "--accept-apple-depth-pro-research-license",
        help="Accept Apple Depth Pro research license (AMLR): true/false",
    ),
    accept_research_tools_license: str = typer.Option(
        "false",
        "--accept-research-tools-license",
        help=(
            "Accept research-tools license"
            " required for scene reconstruction"
            " and other research-only tooling:"
            " true/false"
        ),
    ),
    # Scene-Level Reconstruction
    enable_reconstruction: str = typer.Option(
        "off",
        "--enable-reconstruction",
        help=("Enable scene-level reconstruction stage " + "(requires research license acknowledgements): on/off"),
    ),
    grouping_mode: str = typer.Option(
        "single",
        "--grouping-mode",
        help=("Scene grouping strategy: single (default) or parent_dir " + "(recommended for multi-view reconstruction)"),
    ),
    cameras_sidecar_path: Optional[Path] = typer.Option(
        None,
        "--cameras-sidecar-path",
        help=("Optional path to a tp.scene_cameras.v1 sidecar " + "JSON file for scene camera metadata"),
    ),
    reconstruction_iterations: int = typer.Option(
        1000,
        "--reconstruction-iterations",
        min=1,
        help=("Iteration budget for reconstruction optimization " "(default: 1000)"),
    ),
    reconstruction_tier: str = typer.Option(
        "apex_research",
        "--reconstruction-tier",
        help=("Reconstruction policy tier label forwarded to the " + "reconstruction backend (default: apex_research)"),
    ),
    emit_scene_debug_bundle: str = typer.Option(
        "off",
        "--emit-scene-debug-bundle",
        help=("Emit reconstruction debug bundle artifacts " + "(scene manifest, cameras, reprojection preview): on/off"),
    ),
    # Processing Flags
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Force reprocessing even if outputs exist",
    ),
    force_depth: bool = typer.Option(
        False,
        "--force-depth",
        help="Force depth recomputation (ignore cache)",
    ),
    strict_inputs: bool = typer.Option(
        False,
        "--strict-inputs",
        help=("Fail if depth artifacts or derived outputs " + "found in input directory (validation mode)"),
    ),
    raw_ingest_mode: str = typer.Option(
        "auto",
        "--raw-ingest-mode",
        help=("RAW decode mode: auto, force_rawpy, or force_preview " + "(preview requires TP_ALLOW_RAW_PREVIEW=1)."),
    ),
    raw_wb_mode: str = typer.Option(
        "camera",
        "--raw-wb-mode",
        help=("RAW white-balance mode for legacy_linear_srgb ingest " + "contract (currently only 'camera' is supported)."),
    ),
    raw_demosaic: str = typer.Option(
        "AHD",
        "--raw-demosaic",
        help=("RAW demosaic algorithm for legacy_linear_srgb ingest " + "contract (currently only 'AHD' is supported)."),
    ),
    # Performance Tuning (Forward-Compatible)
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        help=("Max CPU/I/O worker threads for parallel processing " + "(default: auto-detect based on CPU count)"),
    ),
    max_gpu_workers: Optional[int] = typer.Option(
        None,
        "--max-gpu-workers",
        help=("Max GPU workers for inference " + "(default: 2 for GPU/MPS, auto for CPU)"),
    ),
    verify_images: bool = typer.Option(
        False,
        "--verify-images",
        help=("Strict image verification via PIL.verify() " + "- useful for CI/ingest validation"),
    ),
    allow_semantic_fallback: bool = typer.Option(
        False,
        "--allow-semantic-fallback",
        help=("Allow fallback to secondary depth backends " + "when APEX semantic gate fails"),
    ),
    # Logging
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress all output except errors",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Set log level: DEBUG, INFO," " WARNING, ERROR",
    ),
) -> None:
    """Process images through Lux Depth V3.

    Orchestrated depth estimation + V2
    enhancement with APEX quality tier support.

    This CLI supports:
    - Commercial-safe APEX mode (default)
    - Research-only APEX+ variants
      (explicit opt-in with license acks)
    - Materials V3 surface-aware finishing
    - PBR map generation
    - Multiple output formats / deliverables

    Input Discovery:
    The pipeline automatically excludes derived
    artifacts from input discovery:
    - Depth maps (*_depth.png,
      *_depthpro_depth16.png)
    - PBR maps (*_normal.png,
      *_roughness.png, *_ao.png)
    - Output directories
      (depth/, pbr/, v2/, manifests/, logs/)
    - Hidden files/dirs (.DS_Store, .cache/)

    Use --strict-inputs to fail on excluded
    files (validation mode).
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
    enable_apple_license = _parse_bool_flag(
        accept_apple_depth_pro_research_license,
    )
    enable_research_tools_license = _parse_bool_flag(
        accept_research_tools_license,
    )
    enable_material_segmentation = _parse_bool_flag(enable_segmentation)
    enable_save_float_depth = _parse_bool_flag(save_float_depth)
    enable_reconstruction_bool = _parse_bool_flag(
        enable_reconstruction,
    )
    enable_emit_scene_debug_bundle = _parse_bool_flag(
        emit_scene_debug_bundle,
    )

    # Parse V2 preset (convert "none" string to None for skipping V2)
    v2_preset_value = None if (v2_preset and v2_preset.lower() == "none") else v2_preset
    legacy_depth_backend = depth_backend if is_legacy_backend_alias(depth_backend) else None
    depth_backend = normalize_backend_id(depth_backend)
    if legacy_depth_backend:
        logger.warning(backend_alias_warning(str(legacy_depth_backend).strip(), str(depth_backend)))

    # Validate input directory
    if not input_dir.exists():
        error_msg = f"Input directory does not exist: {input_dir}"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    # Validate non-commercial usage
    if depth_backend == "depth_pro" and not enable_non_commercial:
        error_msg = "Depth Pro backend requires --non-commercial-ok true " "(AMLR research-only license)"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    if depth_backend == "depth_pro" and not enable_apple_license:
        error_msg = "Depth Pro backend requires --accept-apple-depth-pro-research-license " "true (Apple research-only)"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    if "v3.1" in preset.lower() and not enable_non_commercial:
        error_msg = f"Preset '{preset}' requires --non-commercial-ok true " "(CC BY-NC 4.0 non-commercial license)"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate quality tier
    valid_quality_tiers = ["standard", "premium", "apex"]
    if quality_tier.lower() not in valid_quality_tiers:
        error_msg = f"Invalid quality tier '{quality_tier}'. Must be one of: " f"{', '.join(valid_quality_tiers)}"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    # Validate scene grouping mode
    grouping_mode_normalized = grouping_mode.strip().lower()
    valid_grouping_modes = ["single", "parent_dir"]
    if grouping_mode_normalized not in valid_grouping_modes:
        error_msg = f"Invalid grouping mode '{grouping_mode}'. Must be one of: " f"{', '.join(valid_grouping_modes)}"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    reconstruction_tier_value = reconstruction_tier.strip()
    if not reconstruction_tier_value:
        error_msg = "Invalid --reconstruction-tier. Value must be non-empty."
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    if cameras_sidecar_path is not None:
        if not cameras_sidecar_path.exists():
            error_msg = f"Camera sidecar file does not exist: " f"{cameras_sidecar_path}"
            logger.error(error_msg)
            print(error_msg, file=sys.stdout)
            raise typer.Exit(code=1)
        if not cameras_sidecar_path.is_file():
            error_msg = f"Camera sidecar path is not a file: " f"{cameras_sidecar_path}"
            logger.error(error_msg)
            print(error_msg, file=sys.stdout)
            raise typer.Exit(code=1)

    # Validate segmentation backend
    valid_segmentation_backends = ["stub", "efficientsam", "sam2"]
    if segmentation_backend.lower() not in valid_segmentation_backends:
        error_msg = (
            "Invalid segmentation backend"
            f" '{segmentation_backend}'."
            " Must be one of:"
            f" {', '.join(valid_segmentation_backends)}"
        )
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    # Validate SAM2 size option only when SAM2 backend is selected.
    if segmentation_backend.lower() == "sam2" and sam2_model_size.lower() not in ["base", "large"]:
        error_msg = "Invalid --sam2-model-size. Must be one of: base, large"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    # APEX strict gate: do not allow Materials V3 no-op configurations
    if quality_tier.lower() == "apex" and enable_materials_v3:
        if not enable_material_segmentation:
            error_msg = (
                "APEX strict gate: Materials V3"
                " in apex tier requires"
                " --enable-segmentation on"
                " (segmentation must be explicit)."
            )
            logger.error(error_msg)
            print(error_msg, file=sys.stdout)
            raise typer.Exit(code=1)

        if segmentation_backend.lower() == "stub":
            error_msg = (
                "APEX strict gate: Materials V3"
                " in apex tier cannot use stub"
                " segmentation backend. Use"
                " --segmentation-backend"
                " efficientsam or sam2."
            )
            logger.error(error_msg)
            print(error_msg, file=sys.stdout)
            raise typer.Exit(code=1)

        if not strict_segmentation:
            error_msg = (
                "APEX strict gate: Materials V3"
                " in apex tier requires"
                " --strict-segmentation to prevent"
                " silent backend fallback."
            )
            logger.error(error_msg)
            print(error_msg, file=sys.stdout)
            raise typer.Exit(code=1)

    # Phase C1 contract guardrails for legacy_linear_srgb ingest contract.
    raw_wb_mode_normalized = raw_wb_mode.strip().lower()
    if raw_wb_mode_normalized != "camera":
        error_msg = f"Invalid --raw-wb-mode" f" '{raw_wb_mode}'." " legacy_linear_srgb currently" " supports only: camera"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    raw_demosaic_normalized = raw_demosaic.strip().upper()
    if raw_demosaic_normalized != "AHD":
        error_msg = f"Invalid --raw-demosaic" f" '{raw_demosaic}'." " legacy_linear_srgb currently" " supports only: AHD"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    raw_ingest_mode_normalized = raw_ingest_mode.strip().lower()
    valid_raw_ingest_modes = ("auto", "force_rawpy", "force_preview")
    if raw_ingest_mode_normalized not in valid_raw_ingest_modes:
        error_msg = (
            f"Invalid --raw-ingest-mode" f" '{raw_ingest_mode}'." " Supported modes are:" " auto|force_rawpy|force_preview"
        )
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    raw_ingest_mode = raw_ingest_mode_normalized

    if enable_reconstruction_bool and not enable_non_commercial:
        error_msg = (
            "Scene reconstruction requires"
            " --non-commercial-ok true"
            " (Inria 3D Gaussian Splatting is"
            " restricted to non-commercial use)."
        )
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    if enable_reconstruction_bool and not enable_research_tools_license:
        error_msg = (
            "Scene reconstruction requires"
            " --accept-research-tools-license"
            " true (research-only tooling"
            " acknowledgement)."
        )
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)
        raise typer.Exit(code=1)

    if enable_reconstruction_bool and grouping_mode_normalized == "single":
        logger.warning(
            "Reconstruction is enabled with"
            " grouping_mode=single; only scenes"
            " with >=2 images are eligible."
            " Use --grouping-mode parent_dir"
            " for typical multi-view datasets."
        )

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
    if preset_enum is None and preset != "premium":
        logger.warning(f"Preset '{preset}' does not map" " to a known Preset enum." " Continuing with string value.")

    config = EnhanceConfig(
        preset=preset_enum,
        preset_requested=preset,
        depth_device=depth_device,
        depth_backend=depth_backend,
        non_commercial_ok=enable_non_commercial,
        accept_apple_depth_pro_research_license=enable_apple_license,
        accept_research_tools_license=enable_research_tools_license,
        depth_pro_python_executable=depth_pro_python,
        force_depth=force_depth or overwrite,
        enable_depth_cache=enable_cache_depth,
        enable_v2=enable_v2_bool,
        v2_preset=v2_preset_value,
        generate_pbr=enable_pbr,
        save_float_depth=enable_save_float_depth,
        quality_tier=quality_tier,
        enable_materials_v3=enable_materials_v3,
        enable_material_segmentation=enable_material_segmentation,
        material_segmentation_backend=segmentation_backend.lower(),
        sam2_model_size=sam2_model_size.lower(),
        sam2_checkpoint_path=(str(sam2_checkpoint_path) if sam2_checkpoint_path else None),
        strict_backend=strict_segmentation,
        emit_master16=enable_emit_master16,
        emit_upscaled16=enable_emit_upscaled16,
        emit_marketing=enable_emit_marketing,
        emit_report=enable_emit_report,
        emit_run_card=enable_emit_run_card,
        enable_reconstruction=enable_reconstruction_bool,
        grouping_mode=grouping_mode_normalized,
        cameras_sidecar_path=(str(cameras_sidecar_path) if cameras_sidecar_path else None),
        reconstruction_iterations=reconstruction_iterations,
        reconstruction_tier=reconstruction_tier_value,
        emit_scene_debug_bundle=enable_emit_scene_debug_bundle,
        strict_inputs=strict_inputs,
        raw_ingest_mode=raw_ingest_mode,
        raw_wb_mode=raw_wb_mode_normalized,
        raw_demosaic=raw_demosaic_normalized,
        allow_semantic_fallback=allow_semantic_fallback,
    )

    # Forward-compatible knobs: apply via setattr
    # for non-breaking config evolution.
    # These are read via getattr in orchestrator,
    # so no config schema changes needed.
    if max_workers is not None:
        setattr(config, "max_workers", max_workers)

    if max_gpu_workers is not None:
        setattr(config, "max_gpu_workers", max_gpu_workers)

    if verify_images:
        setattr(config, "verify_images", verify_images)

    # Create orchestrator
    logger.info(
        "Initializing orchestrator with" f" output dir: {output_dir}",
    )
    orchestrator = EnhanceOrchestrator(config=config, output_root=output_dir)

    # Discover images using same hygiene filters as orchestrator
    from .input_discovery import DiscoveryConfig, discover_images
    from .raw_loader import RAW_EXTENSIONS

    logger.info(f"Discovering images in: {input_dir}")
    # Standard image formats + RAW camera formats (CR2, NEF, ARW, DNG, etc.)
    standard_exts = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"]
    raw_exts = sorted(RAW_EXTENSIONS)
    image_extensions = sorted(set(standard_exts + raw_exts))

    discovery_config = DiscoveryConfig(strict_mode=strict_inputs)
    try:
        image_files = discover_images(
            input_dir,
            discovery_config,
            image_extensions,
        )
    except ValueError as e:
        # Strict mode validation failed
        logger.error(str(e))
        print(str(e), file=sys.stdout)
        raise typer.Exit(code=1)

    if not image_files:
        error_msg = f"No images found in {input_dir}"
        logger.error(error_msg)
        print(error_msg, file=sys.stdout)  # Also print to stdout for CLI tests
        raise typer.Exit(code=1)

    logger.info(f"Found {len(image_files)} images to process")

    # Process batch
    try:
        results = orchestrator.enhance_batch(
            input_dir=input_dir,
            image_extensions=image_extensions,
        )

        # Summary (Note: orchestrator returns "ok" not "success")
        successful = sum(1 for r in results if r.get("status") == "ok")
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
