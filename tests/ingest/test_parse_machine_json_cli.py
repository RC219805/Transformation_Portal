"""Subprocess contracts for the illustrative machine-output consumer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.ingest.errors import OtherIngestFailure, SchemaValidationFailure
from transformation_portal.ingest.machine_output import error_to_dict, extract_result_to_dict
from transformation_portal.ingest.metadata_service import ExtractResult

pytestmark = pytest.mark.unit

PARSER = Path(__file__).resolve().parents[2] / "tools" / "parse_machine_json.py"


@pytest.fixture(params=["stdin", "file"])
def parse_output(request: pytest.FixtureRequest, tmp_path: Path):
    """Exercise both supported transports without importing the CLI's main."""

    def run(text: str) -> subprocess.CompletedProcess[str]:
        args = [sys.executable, str(PARSER)]
        if request.param == "file":
            source = tmp_path / "result.json"
            source.write_text(text, encoding="utf-8")
            args.append(str(source))
            return subprocess.run(args, capture_output=True, text=True, check=False)
        return subprocess.run(args, input=text, capture_output=True, text=True, check=False)

    return run


def envelope(data: dict[str, Any], *, exit_code: int = 0, error: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {
            "schema": "tp.meta.machine.v1",
            "command": "extract",
            "success": exit_code == 0,
            "exit_code": exit_code,
            "data": data,
            "error": error,
        }
    )


@pytest.mark.parametrize("producer_data", [False, True], ids=["empty-data", "producer-data"])
@pytest.mark.parametrize(
    "error,exit_code",
    [
        (error_to_dict(SchemaValidationFailure("invalid extraction input")), 1),
        ({"type": "RuntimeError", "message": "extraction command failed"}, 5),
    ],
    ids=["typed", "generic"],
)
def test_top_level_command_error_precedes_data_routing(parse_output, producer_data, error, exit_code) -> None:
    data = (
        extract_result_to_dict(
            ExtractResult(path=Path("input.CR2"), success=False, output_path=None, elapsed_seconds=0.0, error=None)
        )
        if producer_data
        else {}
    )

    result = parse_output(envelope(data, exit_code=exit_code, error=error))

    assert result.returncode == exit_code
    assert result.stdout == ""
    assert error["type"] in result.stderr
    assert error["message"] in result.stderr
    assert "Traceback" not in result.stderr


def test_success_routes_command_output_to_stdout(parse_output) -> None:
    data = extract_result_to_dict(
        ExtractResult(path=Path("input.CR2"), success=True, output_path=Path("output.json"), elapsed_seconds=0.25)
    )

    result = parse_output(envelope(data))

    assert result.returncode == 0
    assert "input.CR2" in result.stdout
    assert "output.json" in result.stdout
    assert result.stderr == ""


def test_domain_error_keeps_existing_stderr_routing(parse_output) -> None:
    error = OtherIngestFailure("metadata unavailable")
    data = extract_result_to_dict(
        ExtractResult(path=Path("input.CR2"), success=False, output_path=None, elapsed_seconds=0.0, error=error)
    )

    result = parse_output(envelope(data, exit_code=int(error.exit_code)))

    assert result.returncode == int(error.exit_code)
    assert result.stdout == ""
    assert "OtherIngestFailure" in result.stderr
    assert "metadata unavailable" in result.stderr
    assert "OTHER_FAILURE" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "text,diagnostic",
    [
        ("{invalid", "Invalid JSON"),
        ('{"schema":"unsupported"}', "Unsupported schema"),
        ('{"schema":"tp.meta.machine.v1"}', "Missing required field"),
        ("", "No input"),
    ],
)
def test_malformed_input_remains_controlled(parse_output, text, diagnostic) -> None:
    result = parse_output(text)

    assert result.returncode == 99
    assert result.stdout == ""
    assert diagnostic in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(shutil.which("jq") is None, reason="bash reference examples require jq")
@pytest.mark.parametrize(
    "function,args",
    [
        ("extract_with_routing", ["input.CR2"]),
        ("validate_with_error_handling", ["sidecar.json"]),
        ("batch_extract_with_summary", ["inputs", "outputs"]),
        ("check_system_readiness", []),
        ("ci_safe_validate", ["sidecar.json"]),
        ("compact_status_check", ["sidecar.json"]),
    ],
)
def test_shell_examples_handle_command_errors_with_errexit(tmp_path: Path, function: str, args: list[str]) -> None:
    producer = tmp_path / ".venv" / "bin" / "python"
    producer.parent.mkdir(parents=True)
    producer.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_ENVELOPE"\nexit 5\n', encoding="utf-8")
    producer.chmod(0o755)
    script = PARSER.with_name("parse_machine_json_examples.sh")
    env = {
        **os.environ,
        "TEST_ENVELOPE": envelope({}, exit_code=5, error={"type": "RuntimeError", "message": "producer unavailable"}),
    }

    result = subprocess.run(
        ["bash", "-c", 'source "$1"; shift; "$@"', "bash", str(script), function, *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 5
    assert result.stdout == ""
    assert "RuntimeError" in result.stderr
    assert "producer unavailable" in result.stderr
    assert "jq: error" not in result.stderr
