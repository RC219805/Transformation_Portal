"""Explicit, resolver-only planning and photographic V4 execution CLI."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from types import FrameType

from transformation_portal.ingest.canonical_json import dumps_json

from .lifecycle import LuxDepthV4Request, prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LuxDepthV4 photography candidate: canonical geometry, native depth, 16-bit TIFF."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="da3-metric")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--input-color", choices=("auto", "srgb", "linear_srgb"), default="auto")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--clarity", type=float, default=0.0)
    parser.add_argument("--preview-maps", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=100_000_000)
    parser.add_argument("--max-input-bytes", type=int, default=1024**3)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--wall-time-seconds", type=int, default=3600)
    parser.add_argument("--memory-mib", type=int, default=16384)
    parser.add_argument("--runtime-python")
    parser.add_argument("--raw-python")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--companions-manifest", type=Path, help="Immutable per-source calibration and material-mask bindings")
    parser.add_argument(
        "--materials-manifest", type=Path, help="Opt in to source-bound Materials V4 evidence and conservative response"
    )
    parser.add_argument(
        "--materials-policy", type=Path, help="Complete Materials V4 ResponsePolicy JSON; requires --materials-manifest"
    )
    parser.add_argument(
        "--plan", action="store_true", help="Print exact canonical execution bytes without loading models or writing outputs"
    )
    args = vars(parser.parse_args(argv))
    planning = args.pop("plan")
    try:
        policy_path = args.pop("materials_policy")
        if policy_path is not None:
            from transformation_portal.core.execution_plan import decode_bounded_json_object
            from transformation_portal.lux_depth_v4.io import directory_path, snapshot
            from transformation_portal.materials_v4.engine import ResponsePolicy

            policy_path = policy_path.expanduser().absolute()
            policy_root = directory_path(policy_path.parent)
            policy_bytes, _ = snapshot(policy_root, policy_path, maximum_bytes=65536)
            args["materials_policy"] = ResponsePolicy.from_payload(decode_bounded_json_object(policy_bytes))
        prepared = prepare(LuxDepthV4Request(**args))
        if planning:
            sys.stdout.buffer.write(prepared.canonical_plan_bytes)
            sys.stdout.buffer.flush()
            return 0
        from .pipeline import run

        cancelled = False

        def cancel(_signum: int, _frame: FrameType | None) -> None:
            nonlocal cancelled
            cancelled = True

        previous = {signum: signal.signal(signum, cancel) for signum in (signal.SIGTERM, signal.SIGINT)}
        try:
            result = run(prepared, cancellation=lambda: cancelled)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        print(
            dumps_json(
                {
                    "complete": True,
                    "plan_fingerprint_sha256": result.plan_fingerprint_sha256,
                    "evidence": str(result.evidence_path),
                    "input_count": result.input_count,
                    "depth_cache_hits": result.depth_cache_hits,
                },
                sort_keys=True,
            )
        )
        return 0
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        print(f"lux-depth-v4: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
