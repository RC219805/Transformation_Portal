"""CLI entry point for Lux Depth V3 pipeline.

Provides command-line interface for running depth estimation pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point for Lux Depth V3 pipeline.

    Args:
        args: Optional command line arguments (for testing)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Lux Depth V3 - Depth estimation pipeline for architectural visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input image path",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "-m", "--model-variant",
        choices=["large", "base", "small"],
        default="large",
        help="Model variant: large, base, or small (default: large)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto, cpu, cuda, mps (default: auto)",
    )
    parser.add_argument(
        "--v2-fallback",
        action="store_true",
        help="Enable V2 fallback if V3 inference fails",
    )
    parser.add_argument(
        "--skip-v2",
        action="store_true",
        help="Skip V2 enhancement stage",
    )
    parser.add_argument(
        "--v2-preset",
        default="default",
        help="V2 enhancement preset (default: default)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    parsed = parser.parse_args(args)

    # Setup logging
    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate input
    if not parsed.input.exists():
        logger.error(f"Input file not found: {parsed.input}")
        return 1

    # Import here to avoid slow startup for --help
    try:
        from .config import EnhanceConfig, ModelVariant
        from .orchestrator import EnhanceOrchestrator
    except ImportError as e:
        logger.error(f"Failed to import lux_depth_v3 modules: {e}")
        return 1

    # Map model variant
    model_map = {
        "large": ModelVariant.METRIC_LARGE,
        "base": ModelVariant.METRIC_BASE,
        "small": ModelVariant.METRIC_SMALL,
    }
    model_variant = model_map[parsed.model_variant]

    # Configure
    config = EnhanceConfig(
        model_variant=model_variant,
        depth_device=parsed.device if parsed.device != "auto" else "cpu",
        depth_fallback="v2-auto" if parsed.v2_fallback else "fail",
        force_v2=not parsed.skip_v2,
        v2_preset=parsed.v2_preset,
    )

    logger.info(f"Processing: {parsed.input}")
    logger.info(f"Output dir: {parsed.output_dir}")
    logger.info(f"Model: {model_variant.value.display_name}")

    try:
        # Initialize orchestrator
        orchestrator = EnhanceOrchestrator(
            config=config,
            output_root=parsed.output_dir,
        )

        # Run pipeline
        result = orchestrator.run_pipeline(parsed.input)

        # Output result
        if parsed.json:
            print(json.dumps(result, indent=2))
        else:
            status = result.get("status", "unknown")
            depth_path = result.get("depth_path")
            runtime = result.get("runtime_s", 0)

            if status == "ok":
                print(f"✓ Success: {parsed.input}")
                if depth_path:
                    print(f"  Depth: {depth_path}")
                print(f"  Runtime: {runtime:.2f}s")
            else:
                print(f"✗ {status}: {result.get('reason', 'Unknown error')}")

        return 0 if result.get("status") == "ok" else 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if parsed.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
