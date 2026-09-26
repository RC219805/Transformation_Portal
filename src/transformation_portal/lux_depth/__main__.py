"""One explicit CLI for governed Lux inference, finishing, research, and replay."""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--plan", action="store_true", help="Emit the exact canonical plan without creating outputs")


def _limits(parser: argparse.ArgumentParser, *, max_input_bytes: int) -> None:
    parser.add_argument("--max-input-bytes", type=int, default=max_input_bytes)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--max-pixels", type=int, default=100_000_000)
    parser.add_argument("--memory-mib", type=int, default=16384)
    parser.add_argument("--wall-time-seconds", type=int, default=3600)


def _grade_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exposure-stops", type=float, default=0.0)
    parser.add_argument("--white-balance", nargs=3, type=float, metavar=("R", "G", "B"), default=(1.0, 1.0, 1.0))
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--pivot", type=float, default=0.18)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--render", choices=("perceptual_srgb", "soft_srgb", "clip_srgb"), default="perceptual_srgb")
    parser.add_argument("--shoulder", type=float, default=0.8)


def _inference_options(parser: argparse.ArgumentParser, *, materials: bool) -> None:
    _paths(parser)
    _limits(parser, max_input_bytes=1024**3)
    parser.add_argument("--model-key", default="da3-metric")
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--input-color", choices=("auto", "srgb", "linear_srgb"), default="auto")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--refinement", choices=("bilinear", "guided_bilinear"), default="guided_bilinear")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--clarity", type=float, default=0.0)
    parser.add_argument("--preview-maps", action="store_true", help="Include explicitly nonphysical depth-derived previews")
    parser.add_argument("--runtime-python")
    parser.add_argument("--raw-python")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--companions-manifest", type=Path, help="Source-bound camera calibration manifest")
    if materials:
        parser.add_argument("--materials-manifest", type=Path, help="Source-bound Materials V4 evidence")
        parser.add_argument("--materials-policy", type=Path, help="Complete ResponsePolicy JSON; requires materials evidence")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lux-depth",
        description="Unified explicit Lux workflows. Successor execution does not establish production acceptance.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    process = commands.add_parser("process", help="DA3 originals through verified V5 inference and V6 finishing")
    _inference_options(process, materials=False)
    _grade_options(process)
    process.add_argument(
        "--depth-refinement", choices=("bilinear", "guided_bilinear_v3", "guided_bilinear_v4"), default="guided_bilinear_v4"
    )
    infer = commands.add_parser("infer", help="V5 inference with optional evidence-bound Materials V4 response")
    _inference_options(infer, materials=True)
    finish = commands.add_parser("finish", help="Finish a complete verified V5 generation")
    _paths(finish)
    _limits(finish, max_input_bytes=64 * 1024**3)
    _grade_options(finish)
    finish.add_argument("--depth-maps", action=argparse.BooleanOptionalAction, default=True)
    finish.add_argument("--depth-refinement", choices=("bilinear", "guided_bilinear_v3", "guided_bilinear_v4"))
    research = commands.add_parser("depth-pro", help="Explicit non-commercial native Depth Pro research")
    _paths(research)
    _limits(research, max_input_bytes=64 * 1024**3)
    _grade_options(research)
    research.add_argument("--depth-pro-python", type=Path, required=True)
    research.add_argument("--depth-pro-checkpoint", type=Path, required=True)
    research.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    research.add_argument("--input-color", choices=("auto", "srgb", "linear_srgb"), default="auto")
    research.add_argument("--non-commercial-ok", action="store_true")
    research.add_argument("--accept-apple-depth-pro-research-license", action="store_true")
    verify = commands.add_parser("verify", help="Verify a recorded generation using its exact recorded schema and recipe")
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument(
        "--source-root", "--input-dir", dest="source_root", type=Path, help="Retained source required by V6 replay"
    )
    verify.add_argument("--expected-plan-sha256", help="SHA-256 of the complete canonical plan bytes")
    commands.add_parser("legacy", add_help=False, help="Pass remaining arguments directly to the established V3 CLI")
    return parser


def _material_policy(path: Path | None) -> Any:
    if path is None:
        return None
    from transformation_portal.core.execution_plan import decode_bounded_json_object
    from transformation_portal.lux_depth_v4.io import directory_path, snapshot
    from transformation_portal.materials_v4.engine import ResponsePolicy

    path = path.expanduser().absolute()
    root = directory_path(path.parent)
    raw, _ = snapshot(root, root / path.name, maximum_bytes=65536)
    return ResponsePolicy.from_payload(decode_bounded_json_object(raw))


def _inference_request(args: argparse.Namespace) -> Any:
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request

    fields = (
        "input_dir",
        "output_dir",
        "model_key",
        "device",
        "input_color",
        "target_size",
        "precision",
        "refinement",
        "strength",
        "clarity",
        "preview_maps",
        "max_pixels",
        "max_input_bytes",
        "max_output_bytes",
        "wall_time_seconds",
        "memory_mib",
        "runtime_python",
        "raw_python",
        "cache_dir",
        "companions_manifest",
    )
    values = {name: getattr(args, name) for name in fields}
    if args.command == "infer":
        values["materials_manifest"] = args.materials_manifest
        values["materials_policy"] = _material_policy(args.materials_policy)
    return LuxDepthV5Request(**values)


def _request(args: argparse.Namespace) -> Any:
    if args.command == "infer":
        return _inference_request(args)
    from transformation_portal.lux_depth_v6.color import GradeRecipe, RenderRecipe

    grade = GradeRecipe(args.exposure_stops, tuple(args.white_balance), args.contrast, args.pivot, args.saturation)
    render = RenderRecipe(args.render, args.shoulder)
    if args.command == "process":
        from transformation_portal.lux_depth_v6.depth_maps import DepthMapRecipe
        from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request

        return ManagedLuxDepthV6Request(_inference_request(args), grade, render, DepthMapRecipe(args.depth_refinement))
    from transformation_portal.lux_depth_v6.plan import OutputLimits
    from transformation_portal.lux_depth_v6.source import SourceLimits

    source_limits = SourceLimits(args.max_input_bytes, args.max_pixels, args.memory_mib)
    output_limits = OutputLimits(args.max_output_bytes, args.wall_time_seconds)
    if args.command == "depth-pro":
        from transformation_portal.lux_depth_v6.depth_pro import NativeDepthProRequest

        return NativeDepthProRequest(
            args.input_dir,
            args.output_dir,
            args.depth_pro_python,
            args.depth_pro_checkpoint,
            device=args.device,
            non_commercial_ok=args.non_commercial_ok,
            accept_license=args.accept_apple_depth_pro_research_license,
            input_color=args.input_color,
            grade=grade,
            render=render,
            source_limits=source_limits,
            output_limits=output_limits,
        )
    from transformation_portal.lux_depth_v6.depth_maps import DepthMapRecipe
    from transformation_portal.lux_depth_v6.plan import LuxDepthV6Request

    if args.depth_refinement is not None and not args.depth_maps:
        raise ValueError("--depth-refinement requires --depth-maps")
    depth_maps = DepthMapRecipe(args.depth_refinement or "guided_bilinear_v4") if args.depth_maps else None
    return LuxDepthV6Request(args.input_dir, args.output_dir, grade, render, source_limits, output_limits, depth_maps)


def _legacy(arguments: list[str]) -> int:
    import click

    from transformation_portal.lux_depth_v3.__main__ import app

    try:
        result = app(args=arguments, prog_name="lux-depth legacy", standalone_mode=False)
        return result if type(result) is int else 0
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        print("lux-depth legacy: cancelled", file=sys.stderr)
        return 130


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["legacy"]:
        return _legacy(arguments[1:])
    args = _parser().parse_args(arguments)
    from transformation_portal.ingest.canonical_json import dumps_json
    from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError

    from . import lifecycle

    received_signal = 0

    def cancel(signum: int, _frame: FrameType | None) -> None:
        nonlocal received_signal
        received_signal = signum

    previous = {signum: signal.signal(signum, cancel) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Native libraries and progress output must never contaminate canonical stdout.
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "verify":
                verified = lifecycle.verify(
                    args.output_dir,
                    source_root=args.source_root,
                    expected_plan_sha256=args.expected_plan_sha256,
                    cancellation=lambda: bool(received_signal),
                )
                summary = {
                    "verified": True,
                    "output_root": str(verified.output_root),
                    "plan_sha256": verified.plan_sha256,
                    "production_acceptance": "not_established",
                }
            else:
                prepared = lifecycle.prepare(_request(args))
                if received_signal:
                    raise RuntimeError("Cancelled during preparation")
                if not args.plan:
                    result = lifecycle.run(prepared, cancellation=lambda: bool(received_signal))
                    summary = lifecycle.result_summary(result)
            if received_signal:
                raise RuntimeError("Cancelled before completion reporting")
        if args.command != "verify" and args.plan:
            sys.stdout.write(prepared.canonical_plan_bytes.decode("utf-8"))
        else:
            print(dumps_json(summary, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ArtifactEvidenceError) as exc:
        print(f"lux-depth: {exc}", file=sys.stderr)
        return 128 + received_signal if received_signal else 1
    except KeyboardInterrupt:
        print("lux-depth: cancelled", file=sys.stderr)
        return 130
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
