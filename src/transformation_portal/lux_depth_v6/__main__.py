"""Standalone, opt-in V6 finishing of a retained verified V5 generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformation_portal.ingest.canonical_json import dumps_json

from .color import GradeRecipe, RenderRecipe
from .evidence import verify_execution_evidence
from .pipeline import run
from .plan import LuxDepthV6Request, OutputLimits, prepare
from .source import SourceLimits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lux-depth-v6",
        description="Verify V5 evidence, reconstruct conservatively, grade the float master, and render SDR.",
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path, help="Completed V5 output directory; retain it for verification"
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="New V6 output directory (existing only with --verify)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="Emit canonical grade plan without creating output")
    action.add_argument("--verify", action="store_true", help="Verify existing V6 output against its retained V5 parent")
    parser.add_argument("--expected-plan-sha256", help="Expected V6 canonical-byte digest, for --verify")
    parser.add_argument("--exposure-stops", type=float, default=0.0)
    parser.add_argument("--white-balance", nargs=3, type=float, metavar=("R", "G", "B"), default=(1.0, 1.0, 1.0))
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--pivot", type=float, default=0.18)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--render", choices=("perceptual_srgb", "soft_srgb", "clip_srgb"), default="perceptual_srgb")
    parser.add_argument("--shoulder", type=float, default=0.8)
    parser.add_argument("--max-input-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--max-pixels", type=int, default=100_000_000)
    parser.add_argument("--memory-mib", type=int, default=16384)
    parser.add_argument("--wall-time-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        limits = SourceLimits(args.max_input_bytes, args.max_pixels, args.memory_mib)
        if args.verify:
            verified = verify_execution_evidence(
                args.output_dir,
                source_root=args.input_dir,
                expected_plan_sha256=args.expected_plan_sha256,
                source_limits=limits,
            )
            print(
                dumps_json(
                    {"verified": True, "plan_sha256": verified.plan_sha256, "production_acceptance": "not_established"},
                    sort_keys=True,
                )
            )
            return 0
        if args.expected_plan_sha256 is not None:
            raise ValueError("--expected-plan-sha256 is only valid with --verify")
        prepared = prepare(
            LuxDepthV6Request(
                args.input_dir,
                args.output_dir,
                GradeRecipe(args.exposure_stops, tuple(args.white_balance), args.contrast, args.pivot, args.saturation),
                RenderRecipe(args.render, args.shoulder),
                limits,
                OutputLimits(args.max_output_bytes, args.wall_time_seconds),
            )
        )
        if args.plan:
            sys.stdout.write(prepared.canonical_plan_bytes.decode("utf-8") + "\n")
            return 0
        result = run(prepared)
        print(
            dumps_json(
                {
                    "output_root": str(result.output_root),
                    "plan_sha256": result.plan_sha256,
                    "input_count": result.input_count,
                    "production_acceptance": "not_established",
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"lux-depth-v6: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("lux-depth-v6: cancelled; no unverified completion is accepted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
