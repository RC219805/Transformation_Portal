#!/usr/bin/env python3
"""V2 Enhancement Script Entrypoint (Placeholder).

Pass-through implementation:
- Copies input image to output directory
- Writes report JSON for pipeline continuity

Design goals:
- Fail-fast validation (no silent errors)
- Safer path handling (avoid traversal + collisions)
- Atomic report writing
- Clear status indication
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lux Depth V2 Enhancement Script (Placeholder Implementation)",
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
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def enhance_image_passthrough(
    input_path: Path,
    depth_dir: Path | None,
    output_dir: Path,
    preset: str,
    device: str,
    upscaler: str,
) -> dict[str, Any]:
    start = time.perf_counter()

    input_path = validate_input_path(input_path)
    depth_dir = validate_depth_dir(depth_dir)
    output_dir = validate_output_dir(output_dir)

    # Use only filename to avoid leaking directory structure into output paths
    output_path = safe_join_under(output_dir, input_path.name)

    # Prevent no-op/self-copy edge case
    if input_path == output_path:
        raise ValueError(f"Input and output paths are identical: {input_path}")

    logger.info("Passthrough copy: %s -> %s", input_path.name, output_dir)

    # Copy (overwrites if exists)
    shutil.copy2(input_path, output_path)

    runtime_s = time.perf_counter() - start

    return {
        "status": "passthrough",
        "implementation": "placeholder",
        "input": str(input_path),
        "output": str(output_path),
        "depth_dir": str(depth_dir) if depth_dir else None,
        "preset": preset,
        "device": device,
        "upscaler": upscaler,
        "runtime_s": runtime_s,
        "timestamp": time.time(),
        "message": "Placeholder implementation: input copied to output.",
    }


def main() -> int:
    args = parse_arguments()
    configure_logging(args.verbose, args.quiet, args.log_file)

    # Always create output dir early so we can write an error report if needed
    try:
        out_dir = validate_output_dir(args.output_dir)
    except Exception:
        # If output dir can't be created, we can't report there
        out_dir = None

    report_path = None
    if out_dir is not None:
        # Keep your naming convention for compatibility
        report_path = out_dir / f"{Path(args.input_path).stem}_report.json"

    try:
        report = enhance_image_passthrough(
            input_path=args.input_path,
            depth_dir=args.depth_dir,
            output_dir=args.output_dir,
            preset=args.preset,
            device=args.device,
            upscaler=args.upscaler,
        )

        if report_path:
            logger.info("Writing report: %s", report_path)
            atomic_write_json(report_path, report)

        logger.warning("This is a PLACEHOLDER implementation (passthrough).")
        return 0

    except Exception as e:
        logger.exception("Enhancement failed: %s", e)

        if report_path:
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
            }
            try:
                atomic_write_json(report_path, error_report)
            except Exception:
                # Last resort: don't mask the original failure
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
