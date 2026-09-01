#!/usr/bin/env python3
"""V2 Enhancement Script Entrypoint.

Real V2 enhancement implementation:
- Applies depth-aware perceptual finishing
- Material-specific processing
- Clarity and tone mapping enhancements
- Writes comprehensive report JSON

Design goals:
- Fail-fast validation (no silent errors)
- Safer path handling (avoid traversal + collisions)
- Atomic report writing
- Clear status indication
- Reuses existing EnhancementStage for core logic

Architecture: V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Import V2 enhancement implementation
from transformation_portal.lux_depth_v3.v2_enhance import (
    V2EnhancementError,
    enhance_image,
    find_depth_map,
    infer_v2_output_bit_depth,
    resolve_v2_emitted_artifact_path,
)
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig

logger = logging.getLogger("lux_depth_v2_enhance")

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}


def _resolve_path(p: Path) -> Path:
    # expanduser handles "~" properly; resolve normalizes ".." and symlinks
    return p.expanduser().resolve()


def validate_input_path(path: Path) -> Path:
    path = _resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        logger.warning(
            "Input file has unexpected extension: %s (expected one of: %s)",
            path.suffix,
            ", ".join(sorted(VALID_IMAGE_EXTENSIONS)),
        )

    return path


def validate_depth_dir(depth_dir: Path | None) -> Path | None:
    if depth_dir is None:
        return None
    depth_dir = _resolve_path(depth_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth dir not found: {depth_dir}")
    if not depth_dir.is_dir():
        raise ValueError(f"Depth dir is not a directory: {depth_dir}")
    return depth_dir


def validate_output_dir(output_dir: Path) -> Path:
    output_dir = _resolve_path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output dir path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_join_under(base_dir: Path, filename: str) -> Path:
    """
    Join a filename under base_dir and ensure the result stays within base_dir
    after resolution (prevents traversal via weird filename tricks).
    """
    candidate = (base_dir / filename).resolve()
    base_resolved = base_dir.resolve()
    if base_resolved not in candidate.parents and candidate != base_resolved:
        raise ValueError(f"Unsafe output path (escapes output dir): {candidate}")
    return candidate


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def resolve_asset_key(asset_key: str | None, fallback_stem: str) -> str:
    """Normalize and validate canonical asset key.

    Validates that asset_key is a stem-like identifier, not a path-like string
    that could enable directory traversal attacks.

    Args:
        asset_key: Provided asset key (may be None or blank)
        fallback_stem: Fallback value to use if asset_key is None/blank

    Returns:
        Normalized, validated asset key string

    Raises:
        ValueError: If asset_key contains path separators, NUL bytes, or is "." or ".."
    """
    if asset_key is None:
        return fallback_stem

    normalized = str(asset_key).strip()
    if not normalized:
        return fallback_stem

    if "\x00" in normalized:
        raise ValueError("asset_key contains NUL byte")

    if normalized in {".", ".."}:
        raise ValueError(f"asset_key must be a stem-like identifier (no path separators), got: {asset_key!r}")

    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"asset_key must be a stem-like identifier (no path separators), got: {asset_key!r}")

    return normalized


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lux Depth V2 Enhancement Script - Depth-Aware Perceptual Finishing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input_path", type=Path, help="Input image path")
    parser.add_argument("--depth-dir", type=Path, default=None, help="Directory containing depth maps")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for enhanced images and reports")

    parser.add_argument("--preset", default="default", help="Enhancement preset (default: %(default)s)")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Processing device (default: %(default)s)"
    )
    parser.add_argument("--upscaler", default="default", help="Upscaler backend (default: %(default)s)")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    parser.add_argument(
        "--masks-file",
        type=Path,
        default=None,
        help="Explicit path to material masks NPZ file for Materials V3 integration",
    )
    parser.add_argument(
        "--allow-8bit",
        action="store_true",
        help="Allow 16-bit → 8-bit downgrade (bypasses Quality Firewall)",
    )
    parser.add_argument(
        "--output-bit-depth",
        type=int,
        choices=(8, 16),
        default=None,
        help="Explicit enhanced-image encoding depth (8-bit PNG or 16-bit TIFF)",
    )
    parser.add_argument(
        "--asset-key",
        default=None,
        help="Canonical asset key for depth/report resolution; defaults to input_path.stem",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    verbosity.add_argument("--quiet", "-q", action="store_true", help="Suppress all output except errors")

    return parser.parse_args()


def configure_logging(verbose: bool, quiet: bool, log_file: Path | None) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handlers: list[logging.Handler] = []

    # stderr is safer for logs; stdout can be reserved for machine-readable output later
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    handlers.append(console)

    if log_file:
        log_file = _resolve_path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def load_material_masks(masks_file: Path | None) -> dict[str, Any] | None:
    """Load material masks from explicit NPZ file path.

    Args:
        masks_file: Explicit path to material masks NPZ file

    Returns:
        Dictionary mapping material names to numpy arrays, or None if no masks found

    Raises:
        ValueError: If mask file is invalid or corrupted
    """
    if not masks_file or not masks_file.exists():
        return None

    # Security: Check file size BEFORE loading (DoS protection)
    file_size = masks_file.stat().st_size
    if file_size > 100 * 1024 * 1024:  # 100MB limit
        logger.warning(
            f"Material masks file too large: {file_size / (1024 * 1024):.1f}MB " f"(limit: 100MB). Rejecting for safety."
        )
        return None

    try:
        # Load NPZ file with explicit pickle=False for security
        import numpy as np

        with np.load(masks_file, allow_pickle=False) as data:
            masks = {key: data[key] for key in data.files}

        if not masks:
            logger.warning(f"Material mask file is empty: {masks_file}")
            return None

        # Validate loaded masks
        for mat_name, mask in masks.items():
            if not isinstance(mask, np.ndarray):
                logger.warning(f"Invalid mask type for {mat_name}: {type(mask)}")
                return None
            if mask.ndim != 2:
                logger.warning(f"Invalid mask shape for {mat_name}: {mask.shape} (expected 2D)")
                return None

        logger.info(f"Loaded {len(masks)} material masks from {masks_file.name}: {list(masks.keys())}")
        return masks

    except Exception as e:
        logger.warning(f"Failed to load material masks from {masks_file}: {e}")
        return None


def run_v2_enhancement(
    input_path: Path,
    depth_dir: Path | None,
    output_dir: Path,
    preset: str,
    device: str,
    upscaler: str,
    allow_8bit: bool = False,
    output_bit_depth: int | None = None,
    masks_file: Path | None = None,
    asset_key: str | None = None,
) -> dict[str, Any]:
    """Run V2 depth-aware enhancement.

    Args:
        input_path: Input image path
        depth_dir: Directory containing depth maps (optional)
        output_dir: Output directory
        preset: Enhancement preset name
        device: Processing device (cpu/cuda/mps)
        upscaler: Upscaler backend (currently unused, reserved for future)
        allow_8bit: Allow 16-bit → 8-bit downgrade (Quality Firewall bypass)
        output_bit_depth: Explicit enhanced-image encoding depth (8 or 16)
        masks_file: Explicit path to material masks NPZ file (optional, Materials V3 integration)
        asset_key: Canonical asset key for depth/report resolution (optional,
            defaults to input_path.stem if not provided)

    Returns:
        Dict containing enhancement report

    Raises:
        FileNotFoundError: If input not found
        ValueError: If validation fails
        V2EnhancementError: If enhancement fails
    """
    input_path = validate_input_path(input_path)
    depth_dir = validate_depth_dir(depth_dir)
    output_dir = validate_output_dir(output_dir)

    # Build emitted artifact path from canonical identity rather than source suffix.
    lookup_key = resolve_asset_key(asset_key, input_path.stem)
    candidate_bit_depth = output_bit_depth or infer_v2_output_bit_depth(
        input_path,
        allow_8bit_output=allow_8bit,
    )
    output_path = resolve_v2_emitted_artifact_path(
        safe_join_under(output_dir, lookup_key),
        bit_depth=candidate_bit_depth,
        identity=lookup_key,
        materials_enabled=masks_file is not None,
    )

    # Prevent no-op/self-copy edge case
    if input_path == output_path:
        raise ValueError(f"Input and output paths are identical: {input_path}")

    logger.info("V2 Enhancement: %s with preset '%s'", input_path.name, preset)

    # Load enhancement configuration from preset
    try:
        config = V2EnhancementConfig.from_preset(preset)
    except ValueError as e:
        logger.error("Invalid preset: %s", e)
        raise

    # Find depth map if depth_dir provided
    # Use canonical asset key for depth lookup to align with orchestrator naming
    # Validate asset_key to prevent path traversal (important for direct CLI invocation)
    depth_map_path = None
    if depth_dir:
        depth_map_path = find_depth_map(depth_dir, lookup_key)
        if depth_map_path:
            logger.info("Using depth map: %s", depth_map_path.name)
        else:
            logger.warning("No depth map found in %s for %s", depth_dir, lookup_key)

    # Load material masks if masks_file provided (Materials V3 integration)
    material_masks = load_material_masks(masks_file) if masks_file else None

    # Run enhancement
    report = enhance_image(
        input_path=input_path,
        output_path=output_path,
        depth_map_path=depth_map_path,
        material_masks=material_masks,  # Pass through to v2_enhance.py
        config=config,
        device=device,
        allow_8bit_output=allow_8bit,
        output_bit_depth=output_bit_depth,
    )

    # Add upscaler info to report (currently unused but maintained for compatibility)
    report["upscaler"] = upscaler

    # Add identity metadata for provenance/debugging
    report["asset_key"] = lookup_key
    report["input_stem"] = input_path.stem

    # Enrich depth block with lookup_key (computed at this layer)
    if "depth" in report:
        report["depth"]["lookup_key"] = lookup_key
        report["depth"]["depth_dir"] = str(depth_dir) if depth_dir else None
    else:
        # Ensure depth block exists even if enhance_image didn't create one
        report["depth"] = {
            "requested": depth_dir is not None,
            "lookup_key": lookup_key,
            "depth_dir": str(depth_dir) if depth_dir else None,
            "resolved_path": str(depth_map_path) if depth_map_path else None,
            "loaded": depth_map_path is not None,
            "supplied_to_stage": depth_map_path is not None,
            "consumed": report.get("depth_consumed", False),
            "consumption_source": "unknown",
            "stage_has_depth": None,
        }

    return report


def main() -> int:
    args = parse_arguments()
    configure_logging(args.verbose, args.quiet, args.log_file)

    # Resolve canonical asset key for consistent depth/report resolution
    # Validate asset_key to prevent path traversal (important for direct CLI invocation)
    resolved_asset_key = resolve_asset_key(args.asset_key, Path(args.input_path).stem)

    # Always create output dir early so we can write an error report if needed
    try:
        out_dir = validate_output_dir(args.output_dir)
    except Exception:
        # If output dir can't be created, we can't report there
        out_dir = None

    report_path = None
    if out_dir is not None:
        # Use canonical asset key for report naming to align with orchestrator
        # SECURITY: safe_join_under prevents traversal attacks when asset_key
        # is provided via CLI, even though resolve_asset_key already validates
        report_path = safe_join_under(out_dir, f"{resolved_asset_key}_report.json")

    try:
        report = run_v2_enhancement(
            input_path=args.input_path,
            depth_dir=args.depth_dir,
            output_dir=args.output_dir,
            preset=args.preset,
            device=args.device,
            upscaler=args.upscaler,
            allow_8bit=args.allow_8bit,
            output_bit_depth=args.output_bit_depth,
            masks_file=args.masks_file,  # Pass through Materials V3 explicit mask file
            asset_key=resolved_asset_key,
        )

        if report_path:
            logger.info("Writing report: %s", report_path)
            atomic_write_json(report_path, report)

        # Log success
        if report.get("status") == "passthrough":
            logger.info("Enhancement skipped (preset='none' or zero strength)")
        else:
            logger.info("Enhancement completed successfully in %.2fs", report.get("runtime_s", 0))

        return 0

    except V2EnhancementError as e:
        logger.error("V2 enhancement failed: %s", e)

        if report_path:
            # Use canonical asset key for depth lookup consistency
            lookup_key = resolved_asset_key
            error_report = {
                "status": "error",
                "implementation": "v2_enhance",
                "input": str(_resolve_path(Path(args.input_path))),
                "output": None,
                "depth_dir": str(_resolve_path(args.depth_dir)) if args.depth_dir else None,
                "preset": args.preset,
                "device": args.device,
                "upscaler": args.upscaler,
                "timestamp": time.time(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                # Include identity metadata for provenance/debugging
                "asset_key": resolved_asset_key,
                "input_stem": Path(args.input_path).stem,
                # Include structured depth block for observability
                "depth": {
                    "requested": args.depth_dir is not None,
                    "lookup_key": lookup_key,
                    "depth_dir": str(_resolve_path(args.depth_dir)) if args.depth_dir else None,
                    "resolved_path": None,
                    "loaded": False,
                    "supplied_to_stage": False,
                    "consumed": False,
                    "consumption_source": "error_before_processing",
                    "stage_has_depth": None,
                },
            }
            try:
                atomic_write_json(report_path, error_report)
            except Exception:
                # Last resort: don't mask the original failure
                pass

        return 1

    except Exception as e:
        logger.exception("Enhancement failed: %s", e)

        if report_path:
            # Use canonical asset key for depth lookup consistency
            lookup_key = resolved_asset_key
            error_report = {
                "status": "error",
                "implementation": "placeholder",
                "input": str(_resolve_path(Path(args.input_path))),
                "output": None,
                "depth_dir": str(_resolve_path(args.depth_dir)) if args.depth_dir else None,
                "preset": args.preset,
                "device": args.device,
                "upscaler": args.upscaler,
                "timestamp": time.time(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                # Include identity metadata for provenance/debugging
                "asset_key": resolved_asset_key,
                "input_stem": Path(args.input_path).stem,
                # Include structured depth block for observability
                "depth": {
                    "requested": args.depth_dir is not None,
                    "lookup_key": lookup_key,
                    "depth_dir": str(_resolve_path(args.depth_dir)) if args.depth_dir else None,
                    "resolved_path": None,
                    "loaded": False,
                    "supplied_to_stage": False,
                    "consumed": False,
                    "consumption_source": "error_before_processing",
                    "stage_has_depth": None,
                },
            }
            try:
                atomic_write_json(report_path, error_report)
            except Exception:
                # Last resort: don't mask the original failure
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
