#!/usr/bin/env python3
"""Emit an offline APEX model-family characterization matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformation_portal.evals.apex_model_family import (
    DuplicateFamilyError,
    ObservationBindingError,
    ObservationValidationError,
    ReconciliationError,
    build_apex_model_family_characterization_report,
    collect_family_specs,
    validate_now,
)

EXIT_SPEC_INVALID = 2
EXIT_OBSERVATION_INVALID = 3
EXIT_BINDING_INVALID = 4
EXIT_SELF_CHECK_FAILED = 5
EXIT_USAGE = 64


class CharacterizationArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean-like value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = CharacterizationArgumentParser(
        description=(
            "Emit a declarative, offline-only APEX model-family characterization matrix. "
            "This command does not run models or read raw evidence artifacts."
        )
    )
    parser.add_argument("--family", action="append", default=[], help="Explicit comma-separated family key=value spec.")
    parser.add_argument("--matrix", action="append", default=[], help="Semicolon-separated matrix key=value spec.")
    parser.add_argument("--family-file", action="append", default=[], help="JSON apex_family_matrix.v1 spec file.")
    parser.add_argument("--observation", action="append", default=[], help="Mock observation bound by candidate_family.")
    parser.add_argument("--redacted-summary", action="append", default=[], help="Redacted summary binding: candidate_family=...,path=...")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--now", default="1970-01-01T00:00:00Z", help="UTC timestamp YYYY-MM-DDTHH:MM:SSZ.")
    parser.add_argument("--allow-spec-invalid", type=_bool_flag, default=False)
    parser.add_argument("--allow-observation-invalid", type=_bool_flag, default=False)
    parser.add_argument("--non-commercial-ok", type=_bool_flag, default=None)
    parser.add_argument("--accept-depth-pro-license", type=_bool_flag, default=None)
    parser.add_argument("--output", required=True, help="Output report path.")
    return parser


def _print_error(message: str) -> None:
    print(f"APEX model-family characterization error: {message}", file=sys.stderr)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_now(args.now)
    except ValueError as exc:
        _print_error(str(exc))
        return EXIT_USAGE

    if not (args.family or args.matrix or args.family_file):
        _print_error("at least one of --family, --matrix, or --family-file is required")
        return EXIT_USAGE

    try:
        specs, file_governance = collect_family_specs(
            families=args.family,
            matrices=args.matrix,
            family_files=[Path(path) for path in args.family_file],
        )
        non_commercial_ok = (
            bool(args.non_commercial_ok)
            if args.non_commercial_ok is not None
            else bool(file_governance.get("non_commercial_ok", False))
        )
        accept_depth_pro_license = (
            bool(args.accept_depth_pro_license)
            if args.accept_depth_pro_license is not None
            else bool(file_governance.get("accept_depth_pro_license", False))
        )
        report = build_apex_model_family_characterization_report(
            family_specs=specs,
            observations=args.observation,
            redacted_summaries=args.redacted_summary,
            output_path=Path(args.output),
            non_commercial_ok=non_commercial_ok,
            accept_depth_pro_license=accept_depth_pro_license,
            output_format=args.format,
            created_at=args.now,
            allow_observation_invalid=args.allow_observation_invalid,
        )
    except DuplicateFamilyError as exc:
        _print_error(str(exc))
        return EXIT_BINDING_INVALID
    except ObservationBindingError as exc:
        _print_error(str(exc))
        return EXIT_BINDING_INVALID
    except ObservationValidationError as exc:
        _print_error(str(exc))
        return EXIT_OBSERVATION_INVALID
    except ReconciliationError as exc:
        _print_error(str(exc))
        return EXIT_SELF_CHECK_FAILED
    except (OSError, ValueError) as exc:
        _print_error(str(exc))
        return EXIT_USAGE

    spec_failures = [
        row["family_spec"]["candidate_family"]
        for row in report["families"]
        if row["spec_validation"]["status"] != "ok"
    ]
    if spec_failures and not args.allow_spec_invalid:
        _print_error("spec validation failed for: " + ", ".join(spec_failures))
        return EXIT_SPEC_INVALID
    if report["self_check"]["status"] != "ok":
        _print_error("self-check failed")
        return EXIT_SELF_CHECK_FAILED

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
