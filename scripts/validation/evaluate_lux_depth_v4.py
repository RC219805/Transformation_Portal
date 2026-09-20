#!/usr/bin/env python3
"""Freeze private photographic inputs and evaluate instrumented Lux V3/V4 candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformation_portal.ingest.canonical_json import dumps_json  # noqa: E402
from transformation_portal.lux_depth_v4.evaluation import (  # noqa: E402
    EvaluationError,
    _write_private,
    command_runner,
    compare_run,
    file_sha256,
    freeze_corpus,
    run_evaluation,
    validate_spec,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    freeze = subparsers.add_parser("freeze", help="Freeze a private corpus; does not run models")
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--root", action="append", required=True, help="Relative corpus directory; repeatable")
    freeze.add_argument("--baseline-commit", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run", help="Run cold paired commands with explicit evaluation receipts")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="Validate every observation and compare local p95 values")
    compare.add_argument("--run", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "freeze":
            manifest = freeze_corpus(args.source_root, args.root, args.output, args.baseline_commit)
            print(dumps_json({"files": len(manifest["files"]), "manifest_sha256": file_sha256(args.output)}))
        elif args.operation == "run":
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
            validate_spec(spec)
            output = run_evaluation(args.corpus, spec, args.output_dir, command_runner(spec))
            print(output)
        else:
            comparison = compare_run(args.run)
            _write_private(args.output, comparison)
            print(args.output)
            if any(row["performance_verdict"] == "fail" for row in comparison["scenarios"].values()):
                return 2
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation incomplete: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
