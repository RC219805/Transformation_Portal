#!/usr/bin/env python3
"""Reference parser for tp.meta.machine.v1 JSON output.

This is a minimal reference implementation showing how to consume
machine-mode JSON output from the metadata extraction CLI.

This parser is illustrative only. The stable API surface is the JSON
contract itself (schema + field semantics), not this parser's stdout text.

Usage:
    # Parse from file
    .venv/bin/python tools/parse_machine_json.py result.json

    # Parse from stdin
    .venv/bin/python scripts/test_metadata_extraction.py --json extract input_images/image.CR2 | .venv/bin/python tools/parse_machine_json.py

    # Exit code forwarding
    .venv/bin/python scripts/test_metadata_extraction.py --json validate sidecar.json | .venv/bin/python tools/parse_machine_json.py
    echo $?  # Parser exits with same code as CLI

See docs/api/MACHINE_MODE_CONTRACT.md for full contract specification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, NoReturn


def parse_machine_json(json_str: str) -> Dict[str, Any]:
    """Parse and validate machine JSON envelope.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed payload dictionary

    Raises:
        ValueError: If schema version is unsupported
        json.JSONDecodeError: If JSON is invalid
    """
    payload = json.loads(json_str)

    # Validate schema version
    schema = payload.get("schema")
    if schema != "tp.meta.machine.v1":
        raise ValueError(f"Unsupported schema: {schema} " f"(this parser only supports tp.meta.machine.v1)")

    return payload


def route_by_command(payload: Dict[str, Any]) -> NoReturn:
    """Route by command and handle typed results.

    Args:
        payload: Parsed machine JSON payload

    Exits:
        With the exit code from the payload
    """
    command = payload["command"]
    exit_code = payload["exit_code"]
    success = payload["success"]
    data = payload["data"]

    if command == "check-system":
        handle_check_system(success, data)

    elif command == "extract":
        handle_extract(success, data)

    elif command == "validate":
        handle_validate(success, data)

    elif command == "extract-batch":
        handle_extract_batch(success, data)

    elif command == "summarize":
        handle_summarize(success, data)

    else:
        print(f"❓ Unknown command: {command}", file=sys.stderr)

    sys.exit(exit_code)


def handle_check_system(success: bool, data: Dict[str, Any]) -> None:
    """Handle check-system command result."""
    if success:
        print(f"✅ System check passed (all_required_ok={data['all_required_ok']})")
        print("\nTool availability:")
        if data.get("exiftool_available"):
            print(f"  exiftool: {data.get('exiftool_version', 'unknown')}")
        if data.get("git_available"):
            print(f"  git: {data.get('git_version', 'unknown')}")
        if data.get("rawpy_available"):
            print(f"  rawpy: {data.get('rawpy_version', 'unknown')}")
    else:
        print("❌ System check failed", file=sys.stderr)
        if data.get("errors"):
            print("\nErrors:", file=sys.stderr)
            for error in data["errors"]:
                print(f"  - {error}", file=sys.stderr)


def handle_extract(success: bool, data: Dict[str, Any]) -> None:
    """Handle extract command result."""
    input_path = data["input_path"]

    if success:
        output_path = data["output_path"]
        elapsed = data["elapsed_seconds"]
        print(f"✅ Extracted: {input_path}")
        print(f"   Output: {output_path}")
        print(f"   Elapsed: {elapsed:.3f}s")
        if data.get("preset"):
            print(f"   Preset: {data['preset']}")
    else:
        error = data["error"]
        print(f"❌ Extract failed: {input_path}", file=sys.stderr)
        print(f"   Error type: {error['type']}", file=sys.stderr)
        print(f"   Exit code: {error['exit_code']['name']} ({error['exit_code']['value']})", file=sys.stderr)
        print(f"   Message: {error['message']}", file=sys.stderr)


def handle_validate(success: bool, data: Dict[str, Any]) -> None:
    """Handle validate command result."""
    sidecar_path = data["sidecar_path"]

    if success:
        print(f"✅ Validation passed: {sidecar_path}")
        if data.get("strict"):
            print("   Mode: strict")
    else:
        print(f"❌ Validation failed: {sidecar_path}", file=sys.stderr)

        dominant = data.get("dominant_error")
        if dominant:
            print(f"   Dominant error: {dominant['type']} ({dominant['exit_code']['name']})", file=sys.stderr)

        errors = data.get("errors", [])
        if errors:
            print(f"   Total errors: {len(errors)}", file=sys.stderr)
            print("\n   Error breakdown:", file=sys.stderr)
            for error in errors:
                print(f"     - {error['type']}: {error['message']}", file=sys.stderr)


def handle_extract_batch(success: bool, data: Dict[str, Any]) -> None:
    """Handle extract-batch command result."""
    counts = data["summary_counts"]
    total = counts["total"]
    succeeded = counts["success"]
    failed = counts["failure"]

    print(f"📦 Batch result: {succeeded}/{total} succeeded, {failed} failed")
    print(f"   Input root: {data['input_root']}")
    print(f"   Output dir: {data['output_dir']}")

    if not success:
        dominant = data.get("dominant_error")
        if dominant:
            print(f"\n   Dominant error: {dominant['type']}", file=sys.stderr)

        by_exit_code = counts.get("by_exit_code", {})
        if by_exit_code:
            print("\n   Failed by exit code:", file=sys.stderr)
            for code_name, count in by_exit_code.items():
                print(f"     {code_name}: {count}", file=sys.stderr)

    # Show first few failures for debugging
    if failed > 0 and data.get("items"):
        failed_items = [item for item in data["items"] if not item["success"]]
        print(f"\n   First {min(3, len(failed_items))} failures:", file=sys.stderr)
        for item in failed_items[:3]:
            error = item.get("error", {})
            print(f"     - {item['path']}: {error.get('type', 'unknown')}", file=sys.stderr)


def handle_summarize(success: bool, data: Dict[str, Any]) -> None:
    """Handle summarize command result."""
    total = data["total_sidecars"]
    valid = data["valid"]
    invalid = data["invalid"]

    print(f"📊 Summary result: {valid}/{total} valid, {invalid} invalid")
    print(f"   Sidecar dir: {data['sidecar_dir']}")

    errors = data.get("errors", [])
    if not success and errors:
        print("\n   Errors:", file=sys.stderr)
        for error in errors:
            path = error.get("path")
            message = error.get("message", "")
            if path:
                print(f"     - {path}: {message}", file=sys.stderr)
            else:
                print(f"     - {message}", file=sys.stderr)


def main() -> NoReturn:
    """Parse machine JSON from stdin or file argument."""
    try:
        if len(sys.argv) > 1:
            json_path = Path(sys.argv[1])
            if not json_path.exists():
                print(f"ERROR: File not found: {json_path}", file=sys.stderr)
                sys.exit(99)
            json_str = json_path.read_text(encoding="utf-8")
        else:
            json_str = sys.stdin.read()

        if not json_str.strip():
            print("ERROR: No input provided", file=sys.stderr)
            sys.exit(99)

        payload = parse_machine_json(json_str)
        route_by_command(payload)

    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(99)

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(99)

    except KeyError as e:
        print(f"ERROR: Missing required field in JSON: {e}", file=sys.stderr)
        sys.exit(99)

    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(99)


if __name__ == "__main__":
    main()
