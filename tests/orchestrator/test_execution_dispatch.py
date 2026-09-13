"""Canonical dispatch rejects executable data and preserves archive/Lux intent."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import app
from transformation_portal.core.archive_execution_plan import ARCHIVE_OPERATIONS, archive_configuration_fingerprint
from transformation_portal.core.execution_plan import (
    CanonicalExecutionPlan,
    ExecutionPlanError,
    parse_execution_plan_json,
    with_execution_plan_fingerprint,
)
from transformation_portal.orchestrator.execution_dispatch import (
    _archive_command,
    command_from_dispatch_plan,
    execute_dispatch_plan,
    prepare_dispatch_plan,
)

pytestmark = pytest.mark.unit


def _request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation_name: str) -> tuple[dict, list[str]]:
    operation = ARCHIVE_OPERATIONS[operation_name]
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    output = tmp_path / "requested"
    args: dict = {"input_dir": str(inputs), "output_dir": str(output), "archive_command": operation_name}
    for field in operation.files:
        path = inputs / f"{field}.json"
        path.write_text('{"fixture":true}')
        args[field] = str(path)
    for field in operation.directories:
        args[field] = str(inputs)
    for field in operation.outputs:
        args[field] = str(output / field)
    for field in operation.integers:
        args[field] = 1
    for field in operation.strings:
        args[field] = "contract-value"
    for field in operation.flags:
        args[field] = True
    for name in ("ALLOWED_INPUT_ROOTS", "ALLOWED_OUTPUT_ROOTS", "ALLOWED_PATH_ROOTS"):
        monkeypatch.setattr(app, name, [tmp_path])
    request = {"pipeline": operation.pipeline, "args": args, "tenant_id": "tenant-one"}
    command = app._archive_gate_argv(operation.pipeline, args, str(inputs), str(output))
    return request, command


@pytest.mark.parametrize("operation_name", tuple(ARCHIVE_OPERATIONS))
def test_every_existing_archive_command_has_closed_canonical_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation_name: str
) -> None:
    request, trusted_argv = _request(tmp_path, monkeypatch, operation_name)
    data = prepare_dispatch_plan(request, trusted_argv=trusted_argv)
    plan = parse_execution_plan_json(data)
    assert data == plan.to_canonical_json().encode()
    assert plan.planned_backend == "archive"
    assert plan.nodes[0].configuration["operation"] == operation_name
    assert not Path(request["args"]["output_dir"]).exists()
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    staging = tmp_path / "attempt"
    command = _archive_command(plan, staging, snapshots)
    assert command[:4] == [sys.executable, str(app.ARCHIVE_GOVERNANCE_SCRIPT), "--json", operation_name]
    for field in ARCHIVE_OPERATIONS[operation_name].outputs:
        assert f"--{field.replace('_', '-')}={staging / field}" in command
    assert set(plan.nodes[0].configuration) == {
        "schema",
        "configuration_completeness",
        "pipeline",
        "operation",
        "parameters",
        "input_fingerprints",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("executable", "/bin/sh"),
        ("module", "os"),
        ("argv", ["-c", "id"]),
        ("out_xml", "../../outside"),
        ("pipeline", "archive-gate-a"),
    ],
)
def test_rehashed_archive_plan_cannot_add_executable_or_escape_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    request, argv = _request(tmp_path, monkeypatch, "mets-export")
    payload = json.loads(prepare_dispatch_plan(request, trusted_argv=argv))
    configuration = payload["nodes"][0]["configuration"]
    if field == "pipeline":
        configuration[field] = value
    else:
        configuration["parameters"][field] = value
    with pytest.raises((ValueError, ExecutionPlanError)):
        payload["config_fingerprint_sha256"] = archive_configuration_fingerprint(configuration)
        CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))


def test_changed_input_cannot_execute_or_create_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, argv = _request(tmp_path, monkeypatch, "mets-export")
    data = prepare_dispatch_plan(request, trusted_argv=argv)
    Path(request["args"]["manifest_jsonl"]).write_text("changed after admission")
    output = tmp_path / "attempt"
    with pytest.raises(ExecutionPlanError, match="no longer matches"):
        execute_dispatch_plan(data, output_root=output)
    assert not output.exists()


def test_rehashed_archive_plan_cannot_make_its_bundle_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, argv = _request(tmp_path, monkeypatch, "mets-export")
    payload = json.loads(prepare_dispatch_plan(request, trusted_argv=argv))
    payload["nodes"][0]["outputs"][0]["required"] = False
    with pytest.raises(ExecutionPlanError, match="bundle must be required"):
        CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))


def test_archive_backend_cannot_be_used_as_a_lux_ensemble_model() -> None:
    from tests.core.test_execution_plan import _valid_payload, _with_backend_shape

    payload = _with_backend_shape(_valid_payload(), ["ensemble"])
    payload["backend_candidates"][0]["model_contracts"][0]["backend_id"] = "archive"
    with pytest.raises(ExecutionPlanError, match="concrete non-synthetic backends"):
        CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO regression")
def test_special_file_replacement_rejects_without_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, argv = _request(tmp_path, monkeypatch, "mets-export")
    source = Path(request["args"]["manifest_jsonl"])
    source.unlink()
    os.mkfifo(source)
    with pytest.raises(ExecutionPlanError, match="regular file"):
        prepare_dispatch_plan(request, trusted_argv=argv)


def test_private_snapshot_preserves_admitted_input_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, argv = _request(tmp_path, monkeypatch, "mets-export")
    plan = parse_execution_plan_json(prepare_dispatch_plan(request, trusted_argv=argv))
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    command = _archive_command(plan, tmp_path / "attempt", snapshots)
    captured = Path(next(item.partition("=")[2] for item in command if item.startswith("--manifest-jsonl=")))
    source = Path(request["args"]["manifest_jsonl"])
    expected = source.read_bytes()
    source.write_text("mutated original")
    assert captured.read_bytes() == expected
    assert captured != source


def test_native_lux_preparation_uses_exact_cli_plan_without_outputs(tmp_path: Path) -> None:
    from PIL import Image

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(inputs / "sample.png")
    output = tmp_path / "output"
    argv = [
        sys.executable,
        "-m",
        "transformation_portal.lux_depth_v3",
        "--input-dir",
        str(inputs),
        "--output-dir",
        str(output),
        "--model-key",
        "da3-metric",
    ]
    data = prepare_dispatch_plan({"pipeline": "lux-depth-v3"}, trusted_argv=argv)
    direct = subprocess.run([*argv, "--plan"], capture_output=True, check=True)
    assert data == direct.stdout.removesuffix(b"\n")
    assert not output.exists()
    with pytest.raises(ExecutionPlanError, match="trusted Lux CLI"):
        prepare_dispatch_plan({"pipeline": "lux-depth-v3"}, trusted_argv=[sys.executable, "-c", "raise SystemExit(0)"])


def test_dispatch_consumer_rejects_tampered_carrier_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, argv = _request(tmp_path, monkeypatch, "mets-export")
    data = prepare_dispatch_plan(request, trusted_argv=argv)
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "attempt"
    command = command_from_dispatch_plan(data, output_root=output, plan_path=plan_path)
    plan_path.write_bytes(data + b" ")
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"digest does not match" in result.stderr
    assert not output.exists()


def test_real_archive_fixity_verify_publishes_only_to_attempt(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests" / "fixtures" / "archive_small"
    requested = tmp_path / "requested"
    attempt = tmp_path / "attempt"
    request = {
        "pipeline": "archive-gate-a",
        "args": {"input_dir": str(fixture / "archive_root"), "output_dir": str(requested)},
    }
    argv = [
        sys.executable,
        str(repo / "tools" / "archive_governance.py"),
        "--json",
        "fixity-verify",
        "--hash-manifest",
        str(fixture / "golden" / "hash_manifest.csv.gz"),
        "--archive-root",
        str(fixture / "archive_root"),
        "--report-path",
        str(requested / "verification_report.json"),
        "--verify-sample",
        "0",
        "--workers",
        "1",
    ]
    plan_bytes = prepare_dispatch_plan(request, trusted_argv=argv)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(plan_bytes)
    result = subprocess.run(
        command_from_dispatch_plan(plan_bytes, output_root=attempt, plan_path=plan_path), capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr.decode()
    assert (attempt / "verification_report.json").is_file()
    assert json.loads(result.stdout)["command"] == "fixity-verify"
    assert not requested.exists()


def test_archive_cannot_be_injected_into_lux_fallback_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, argv = _request(tmp_path, monkeypatch, "mets-export")
    payload = json.loads(prepare_dispatch_plan(request, trusted_argv=argv))
    broken = copy.deepcopy(payload)
    broken["planned_backend"] = "synthetic"
    broken["candidate_fallback_chain"] = ["synthetic", "archive"]
    broken["backend_candidates"].insert(0, {"backend_id": "synthetic", "model_contracts": []})
    with pytest.raises(ExecutionPlanError, match="not a Lux fallback"):
        CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(broken))


@pytest.mark.parametrize("swap_ancestor", [False, True])
def test_real_archive_output_cannot_follow_replaced_admitted_directory(tmp_path: Path, monkeypatch, swap_ancestor):
    from transformation_portal.orchestrator import execution_dispatch as dispatch

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests" / "fixtures" / "archive_small"
    requested = tmp_path / "requested"
    mutable = tmp_path / "mutable"
    attempt = mutable / "attempt"
    mutable.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim_output = outside / "attempt" if swap_ancestor else outside
    victim_output.mkdir(exist_ok=True)
    protected = victim_output / "verification_report.json"
    protected.write_text("preserve outside file")
    request = {
        "pipeline": "archive-gate-a",
        "args": {"input_dir": str(fixture / "archive_root"), "output_dir": str(requested)},
    }
    argv = [
        sys.executable,
        str(repo / "tools/archive_governance.py"),
        "--json",
        "fixity-verify",
        "--hash-manifest",
        str(fixture / "golden/hash_manifest.csv.gz"),
        "--archive-root",
        str(fixture / "archive_root"),
        "--report-path",
        str(requested / "verification_report.json"),
        "--verify-sample",
        "0",
        "--workers",
        "1",
    ]
    data = prepare_dispatch_plan(request, trusted_argv=argv)
    original_run = dispatch.subprocess.run
    native_result = []

    def swap_then_run(command, **kwargs):
        target = mutable if swap_ancestor else attempt
        target.rename(tmp_path / "original-owned-directory")
        target.symlink_to(outside, target_is_directory=True)
        result = original_run(command, capture_output=True, timeout=10, **kwargs)
        native_result.append(result.returncode)
        return result

    monkeypatch.setattr(dispatch.subprocess, "run", swap_then_run)
    with pytest.raises((OSError, ExecutionPlanError)):
        execute_dispatch_plan(data, output_root=attempt)
    assert native_result == [0]  # actual archive operation completed in private storage
    assert protected.read_text() == "preserve outside file"


@pytest.mark.parametrize("replace_output", [False, True])
def test_lux_runs_privately_before_descriptor_export(tmp_path: Path, monkeypatch, replace_output):
    from types import SimpleNamespace

    from tests.core.test_execution_plan import _valid_payload
    from transformation_portal.lux_depth_v3 import execution_lifecycle
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    data = CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()
    output = tmp_path / "attempt"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "result.txt").write_text("preserve")
    private_roots = []
    prepared = SimpleNamespace(input_root=tmp_path)
    monkeypatch.setattr(execution_lifecycle, "consume_lux_execution_plan", lambda *args, **kwargs: prepared)

    def from_prepared(_prepared, *, output_root):
        assert _prepared is prepared
        assert not output_root.is_relative_to(output)
        private_roots.append(output_root)

        def enhance_batch(**kwargs):
            if replace_output:
                output.rename(tmp_path / "original-attempt")
                output.symlink_to(victim, target_is_directory=True)
            (output_root / "result.txt").write_text("private Lux result")
            return [{"status": "ok"}]

        return SimpleNamespace(enhance_batch=enhance_batch)

    monkeypatch.setattr(EnhanceOrchestrator, "from_prepared", from_prepared)
    if replace_output:
        with pytest.raises((OSError, ExecutionPlanError)):
            execute_dispatch_plan(data, output_root=output)
    else:
        assert execute_dispatch_plan(data, output_root=output) == 0
        assert (output / "result.txt").read_text() == "private Lux result"
    assert (victim / "result.txt").read_text() == "preserve"
    assert private_roots and not private_roots[0].exists()


def test_export_stays_pinned_when_ancestor_changes_after_final_identity_check(tmp_path: Path, monkeypatch):
    from transformation_portal.orchestrator import execution_dispatch as dispatch

    source = tmp_path / "private"
    source.mkdir()
    (source / "result.txt").write_text("owned output")
    output = tmp_path / "attempt"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.txt").write_text("preserve")
    descriptor = dispatch._pin_output_directory(output)
    try:
        output.rename(tmp_path / "original-attempt")
        output.symlink_to(outside, target_is_directory=True)
        dispatch._export_execution_outputs(source, descriptor)
    finally:
        os.close(descriptor)
    assert (tmp_path / "original-attempt/result.txt").read_text() == "owned output"
    assert (outside / "result.txt").read_text() == "preserve"


@pytest.mark.parametrize("violation", ["symlink", "fifo", "file_size", "total_size", "file_count"])
def test_private_output_export_enforces_regular_file_and_resource_bounds(tmp_path: Path, monkeypatch, violation: str):
    from transformation_portal.orchestrator import execution_dispatch as dispatch

    source = tmp_path / "private"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    data = source / "result.txt"
    if violation == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("must not be copied")
        data.symlink_to(outside)
    elif violation == "fifo":
        os.mkfifo(data)
    else:
        data.write_bytes(b"bounded content")
        limit = {
            "file_size": "MAX_GENERATION_FILE_BYTES",
            "total_size": "MAX_GENERATION_BYTES",
            "file_count": "MAX_GENERATION_FILES",
        }
        monkeypatch.setattr(dispatch, limit[violation], 0)
    descriptor = dispatch._pin_output_directory(output)
    try:
        with pytest.raises(ExecutionPlanError, match="bounded regular files|publication bounds"):
            dispatch._export_execution_outputs(source, descriptor)
    finally:
        os.close(descriptor)
    assert list(output.iterdir()) == []
