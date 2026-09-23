"""Prepare and consume canonical plans behind the trusted job boundary.

Only the API's existing allowlisted builder supplies ``trusted_argv`` during
preparation. Commands are never persisted in a broker or operational record.
Workers execute this fixed entry point with exact, digest-bound plan bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformation_portal.core.archive_execution_plan import (
    ARCHIVE_CONFIGURATION_SCHEMA,
    ARCHIVE_OPERATIONS,
    archive_configuration_fingerprint,
    validate_archive_configuration,
)
from transformation_portal.core.execution_plan import (
    EXECUTION_COMPLETE,
    EXECUTION_PLAN_SCHEMA,
    MAX_DECODED_PIXELS_PER_INPUT,
    MAX_INPUT_DECOMPRESSION_RATIO,
    MAX_PLAN_BODY_BYTES,
    MAX_TOTAL_DECODED_PIXELS,
    CanonicalExecutionPlan,
    ExecutionPlanError,
    decode_bounded_json_object,
    parse_execution_plan_json,
    with_execution_plan_fingerprint,
)
from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4
from transformation_portal.core.execution_plan_v5 import ExecutionPlanV5
from transformation_portal.ingest.canonical_json import TP_CANONICAL_JSON_PROFILE
from transformation_portal.orchestrator.artifact_store._filesystem import open_directory, open_source_file
from transformation_portal.orchestrator.artifact_store.generation import (
    MAX_GENERATION_BYTES,
    MAX_GENERATION_FILE_BYTES,
    MAX_GENERATION_FILES,
)
from transformation_portal.orchestrator.execution_workspace import (
    create_execution_workspace,
    remove_execution_workspace,
    validate_execution_workspace,
)
from transformation_portal.orchestrator.photography_adapter import (
    MAX_PHOTOGRAPHY_BINDINGS_BYTES,
    consume_photography_dispatch,
    validate_photography_dispatch,
)
from transformation_portal.stage_graph.registry import StageRegistryIdentifier, get_stage_definition

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARCHIVE_RUNNER = _REPO_ROOT / "tools" / "archive_governance.py"
_LUX_MODULE = "transformation_portal.lux_depth_v3"
_CHUNK_BYTES = 1024 * 1024
_MAX_ARCHIVE_INPUT_BYTES = 1024**4


def _canonical_plan(data: bytes) -> CanonicalExecutionPlan:
    plan = parse_execution_plan_json(data)
    if data != plan.to_canonical_json().encode("utf-8") or plan.configuration_completeness != EXECUTION_COMPLETE:
        raise ExecutionPlanError("Dispatch requires exact execution-complete canonical plan bytes")
    return plan


def validate_dispatch_plan(
    plan_bytes: bytes, execution_bindings: bytes | None = None
) -> CanonicalExecutionPlan | ExecutionPlanV4 | ExecutionPlanV5:
    """Statically allowlist plan families and their exact physical-binding carrier."""
    schema = decode_bounded_json_object(plan_bytes).get("schema")
    if schema == "tp.execution.plan.v5":
        from transformation_portal.orchestrator.photography_v6_adapter import validate_v6_dispatch

        if execution_bindings is None:
            raise ExecutionPlanError("V6 dispatch requires immutable photography bindings")
        return validate_v6_dispatch(plan_bytes, execution_bindings)
    if schema == "tp.execution.plan.v4":
        if execution_bindings is None:
            raise ExecutionPlanError("V5 dispatch requires immutable photography bindings")
        return validate_photography_dispatch(plan_bytes, execution_bindings)
    if execution_bindings is not None:
        raise ExecutionPlanError("Legacy dispatch does not accept photography bindings")
    return _canonical_plan(plan_bytes)


def _regular_file_digest(path: Path, *, snapshot: Path | None = None) -> str:
    """Hash/copy the same nonblocking descriptor, rejecting special files."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_ARCHIVE_INPUT_BYTES:
            raise ExecutionPlanError("Archive input must be a bounded regular file")
        digest = hashlib.sha256()
        copied = 0
        target = snapshot.open("xb") if snapshot is not None else None
        try:
            while chunk := source.read(_CHUNK_BYTES):
                copied += len(chunk)
                if copied > before.st_size:
                    raise ExecutionPlanError("Archive input changed during snapshot")
                digest.update(chunk)
                if target is not None:
                    target.write(chunk)
            after = os.fstat(source.fileno())
            if copied != before.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns):
                raise ExecutionPlanError("Archive input changed during snapshot")
        finally:
            if target is not None:
                target.close()
    return digest.hexdigest()


def _absolute_path(value: str, *, directory: bool = False, must_exist: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=must_exist) != path:
        raise ExecutionPlanError("Dispatch paths must be absolute canonical paths without symlinks")
    if directory and must_exist and not path.is_dir():
        raise ExecutionPlanError("Dispatch directory is unavailable")
    return path


def _archive_parameters(trusted_argv: Sequence[str], output_root: Path) -> tuple[str, dict[str, Any]]:
    if len(trusted_argv) < 4 or list(trusted_argv[:3]) != [sys.executable, str(_ARCHIVE_RUNNER), "--json"]:
        raise ExecutionPlanError("Archive preparation requires the trusted archive runner")
    operation_name = trusted_argv[3]
    operation = ARCHIVE_OPERATIONS.get(operation_name)
    if operation is None:
        raise ExecutionPlanError("Unknown archive operation")
    parameters: dict[str, Any] = {}
    index = 4
    while index < len(trusted_argv):
        flag = trusted_argv[index]
        if not flag.startswith("--"):
            raise ExecutionPlanError("Unexpected archive positional argument")
        negative = flag.startswith("--no-")
        name = flag[5 if negative else 2 :].replace("-", "_")
        if name not in operation.parameters or name in parameters:
            raise ExecutionPlanError("Unknown or repeated archive parameter")
        if name in operation.flags:
            parameters[name] = not negative
            index += 1
            continue
        if negative or index + 1 >= len(trusted_argv):
            raise ExecutionPlanError("Invalid archive parameter value")
        raw = trusted_argv[index + 1]
        if name in operation.integers:
            parameters[name] = int(raw)
        elif name in operation.outputs:
            try:
                parameters[name] = _absolute_path(raw, must_exist=False).relative_to(output_root).as_posix()
            except ValueError as exc:
                raise ExecutionPlanError("Archive outputs must stay inside the requested job output root") from exc
        elif name in operation.files + operation.directories:
            parameters[name] = str(_absolute_path(raw, directory=name in operation.directories))
        else:
            parameters[name] = raw
        index += 2
    return operation_name, parameters


def prepare_dispatch_plan(job_request: Mapping[str, Any], *, trusted_argv: Sequence[str]) -> bytes:
    """Freeze the API's validated request before admission or output creation."""

    pipeline = job_request.get("pipeline")
    if pipeline == "lux-depth-v3":
        if list(trusted_argv[:3]) != [sys.executable, "-m", _LUX_MODULE] or "--plan" in trusted_argv:
            raise ExecutionPlanError("Lux preparation requires the trusted Lux CLI")
        # Reuse the exact CLI configuration resolver. Temporary streams bound
        # memory even when an invalid local runtime emits excessive diagnostics.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            result = subprocess.run([*trusted_argv, "--plan"], stdout=output, stderr=errors, timeout=120, check=False)
            if result.returncode:
                raise ExecutionPlanError("Native Lux plan preparation failed")
            output.seek(0)
            data = output.read(MAX_PLAN_BODY_BYTES + 2).removesuffix(b"\n")
        return _canonical_plan(data).to_canonical_json().encode("utf-8")
    arguments = job_request.get("args")
    if not isinstance(arguments, Mapping):
        raise ExecutionPlanError("Archive request requires normalized arguments")
    output_root = _absolute_path(str(arguments.get("output_dir", "")), must_exist=False)
    input_root = _absolute_path(str(arguments.get("input_dir", "")), directory=True)
    operation_name, parameters = _archive_parameters(trusted_argv, output_root)
    operation = ARCHIVE_OPERATIONS[operation_name]
    if pipeline != operation.pipeline:
        raise ExecutionPlanError("Archive operation does not match the requested pipeline")
    fingerprints = {name: _regular_file_digest(Path(parameters[name])) for name in operation.files if name in parameters}
    configuration = {
        "schema": ARCHIVE_CONFIGURATION_SCHEMA,
        "configuration_completeness": EXECUTION_COMPLETE,
        "pipeline": pipeline,
        "operation": operation_name,
        "parameters": parameters,
        "input_fingerprints": fingerprints,
    }
    fingerprint = archive_configuration_fingerprint(configuration)
    input_paths: set[str] = set()
    for name in fingerprints:
        try:
            input_paths.add(Path(parameters[name]).relative_to(input_root).as_posix())
        except ValueError:
            # Archive gate inputs may reference prior outputs. Their absolute
            # locators and digests remain bound by the closed configuration.
            pass
    definition = get_stage_definition(StageRegistryIdentifier.ARCHIVE_OPERATION)
    payload = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "canonicalization": TP_CANONICAL_JSON_PROFILE,
        "configuration_completeness": EXECUTION_COMPLETE,
        "planned_backend": "archive",
        "candidate_fallback_chain": ["archive"],
        "backend_candidates": [{"backend_id": "archive", "model_contracts": []}],
        "resolved_model": None,
        "license_acknowledgements": {"non_commercial_ok": False, "apple_depth_pro_research": False, "research_tools": False},
        "license_evaluation": {"enforced": True, "status": "allowed"},
        "quality_tier": "archive",
        "preset_requested": None,
        "preset_resolved": None,
        "input_selection": {
            "root": str(input_root),
            "files": [{"id": f"archive.input.{i}", "path": path} for i, path in enumerate(sorted(input_paths))],
        },
        "input_limits": {
            "max_decoded_pixels_per_input": MAX_DECODED_PIXELS_PER_INPUT,
            "max_total_decoded_pixels": MAX_TOTAL_DECODED_PIXELS,
            "max_decompression_ratio": MAX_INPUT_DECOMPRESSION_RATIO,
        },
        "config_fingerprint_sha256": fingerprint,
        "nodes": [
            {
                "id": "archive.operation",
                "stage_registry_id": definition.identifier.value,
                "configuration": configuration,
                "resources": definition.resources.to_payload(),
                "optional": False,
                "failure_policy": "abort_plan",
                "outputs": [
                    {
                        "id": "archive.bundle",
                        "artifact_kind": "archive_bundle",
                        "scope": "per_run",
                        "cardinality": "many",
                        "required": True,
                        "disposition": "requested",
                    }
                ],
            }
        ],
        "edges": [],
        "requested_outputs": ["archive_bundle"],
        "warnings": [],
    }
    return CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload)).to_canonical_json().encode("utf-8")


def command_from_dispatch_plan(
    plan_bytes: bytes,
    *,
    output_root: Path,
    plan_path: Path,
    execution_workspace: Path | None = None,
    execution_bindings: bytes | None = None,
    bindings_path: Path | None = None,
) -> list[str]:
    """Return only the fixed local consumer, never a command from stored data."""

    validate_dispatch_plan(plan_bytes, execution_bindings)
    if (execution_bindings is None) != (bindings_path is None):
        raise ExecutionPlanError("Dispatch bindings require both carrier bytes and a private file")
    root = _absolute_path(str(output_root), must_exist=False)
    path = _absolute_path(str(plan_path), must_exist=False)
    command = [
        sys.executable,
        "-m",
        __name__,
        "--plan-file",
        str(path),
        "--plan-sha256",
        hashlib.sha256(plan_bytes).hexdigest(),
        "--output-root",
        str(root),
    ]
    if execution_bindings is not None:
        command.extend(
            [
                "--bindings-file",
                str(_absolute_path(str(bindings_path), must_exist=False)),
                "--bindings-sha256",
                hashlib.sha256(execution_bindings).hexdigest(),
            ]
        )
    if execution_workspace is not None:
        validate_execution_workspace(execution_workspace, root)
        command.extend(["--execution-workspace", str(execution_workspace)])
    return command


def _archive_command(plan: CanonicalExecutionPlan, output_root: Path, snapshot_root: Path) -> list[str]:
    configuration = plan.to_payload()["nodes"][0]["configuration"]
    validate_archive_configuration(configuration)
    operation_name = configuration["operation"]
    operation = ARCHIVE_OPERATIONS[operation_name]
    command = [sys.executable, str(_ARCHIVE_RUNNER), "--json", operation_name]
    for name, value in configuration["parameters"].items():
        flag = "--" + name.replace("_", "-")
        if name in operation.flags:
            if value:
                command.append(flag)
            elif name in {"validate_schemas", "require_stac"}:
                command.append("--no-" + name.replace("_", "-"))
            continue
        if name in operation.files:
            source = _absolute_path(value)
            snapshot = snapshot_root / (name + "-" + source.name)
            actual = _regular_file_digest(source, snapshot=snapshot)
            if actual != configuration["input_fingerprints"][name]:
                raise ExecutionPlanError("Archive input no longer matches the admitted plan")
            value = str(snapshot)
        elif name in operation.directories:
            value = str(_absolute_path(value, directory=True))
        elif name in operation.outputs:
            value = str(_absolute_path(str(output_root / value), must_exist=False))
        command.append(f"{flag}={value}")
    return command


def _pin_output_directory(path: Path) -> int:
    """Create missing directories relative to pinned parents, never symlinks."""
    try:
        return open_directory(path)
    except FileNotFoundError:
        parent = _pin_output_directory(path.parent)
        try:
            with suppress(FileExistsError):
                os.mkdir(path.name, mode=0o700, dir_fd=parent)
            return os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)


def _export_execution_outputs(source_root: Path, output_descriptor: int) -> None:
    """Copy bounded regular outputs into the pinned destination atomically.

    Native tools may resolve their output paths internally, so they operate
    exclusively in a server-owned private directory. Only this exporter touches
    the mutable requested namespace, with every write relative to pinned fds.
    """
    total = 0
    file_count = 0
    entry_count = 0

    def export(source_dir: Path, destination: int, depth: int = 0) -> None:
        nonlocal total, file_count, entry_count
        if depth > 64:
            raise ExecutionPlanError("Execution output nesting exceeds publication bounds")
        with os.scandir(source_dir) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_GENERATION_FILES * 16:
                    raise ExecutionPlanError("Execution output directory count exceeds publication bounds")
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    with suppress(FileExistsError):
                        os.mkdir(entry.name, mode=0o700, dir_fd=destination)
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=destination)
                    try:
                        export(Path(entry.path), child, depth + 1)
                    finally:
                        os.close(child)
                    continue
                file_count += 1
                if file_count > MAX_GENERATION_FILES or not stat.S_ISREG(metadata.st_mode):
                    raise ExecutionPlanError("Execution outputs must be bounded regular files")
                temporary = ".tp-export-" + uuid.uuid4().hex
                try:
                    with os.fdopen(open_source_file(Path(entry.path)), "rb") as source:
                        before = os.fstat(source.fileno())
                        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_GENERATION_FILE_BYTES:
                            raise ExecutionPlanError("Execution output exceeds publication bounds")
                        fd = os.open(
                            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=destination
                        )
                        with os.fdopen(fd, "wb") as target:
                            copied = 0
                            while chunk := source.read(_CHUNK_BYTES):
                                copied += len(chunk)
                                total += len(chunk)
                                if copied > before.st_size or total > MAX_GENERATION_BYTES:
                                    raise ExecutionPlanError("Execution output exceeds publication bounds")
                                target.write(chunk)
                            after = os.fstat(source.fileno())
                            if copied != before.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (
                                after.st_mtime_ns,
                                after.st_ctime_ns,
                            ):
                                raise ExecutionPlanError("Execution output changed during export")
                            target.flush()
                            os.fsync(target.fileno())
                            os.fchmod(target.fileno(), stat.S_IMODE(before.st_mode))
                        os.utime(
                            temporary, ns=(before.st_atime_ns, before.st_mtime_ns), dir_fd=destination, follow_symlinks=False
                        )
                    os.replace(temporary, entry.name, src_dir_fd=destination, dst_dir_fd=destination)
                finally:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary, dir_fd=destination)
        os.fsync(destination)

    export(source_root, output_descriptor)


def execute_dispatch_plan(
    plan_bytes: bytes,
    *,
    output_root: Path,
    execution_workspace: Path | None = None,
    execution_bindings: bytes | None = None,
    managed_process_group: bool = False,
) -> int:
    """Consume frozen intent privately, then export through pinned authority."""

    plan = validate_dispatch_plan(plan_bytes, execution_bindings)
    root = _absolute_path(str(output_root), must_exist=False)
    output_descriptor = None
    owned_workspace = execution_workspace is None
    workspace = execution_workspace if execution_workspace is not None else create_execution_workspace(root)
    validate_execution_workspace(workspace, root)
    try:
        execution_root = workspace / "outputs"
        if isinstance(plan, ExecutionPlanV5):
            from transformation_portal.lux_depth_v6.managed import run as run_managed_v6
            from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits
            from transformation_portal.orchestrator.photography_v6_adapter import consume_v6_dispatch

            assert execution_bindings is not None
            prepared_v6 = consume_v6_dispatch(plan_bytes, execution_bindings, execution_root=execution_root)
            limits = GenerationPublicationLimits.from_payload(plan.to_payload()["publication"])
            output_descriptor = _pin_output_directory(root)
            run_managed_v6(prepared_v6, publication_limits=limits, managed_process_group=managed_process_group)
            return_code = 0
        elif isinstance(plan, ExecutionPlanV4):
            from transformation_portal.lux_depth_v5.pipeline import run
            from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits

            assert execution_bindings is not None
            prepared_photography = consume_photography_dispatch(plan_bytes, execution_bindings, execution_root=execution_root)
            limits = GenerationPublicationLimits.from_payload(plan.to_payload()["publication"])
            output_descriptor = _pin_output_directory(root)
            run(prepared_photography, publication_limits=limits, managed_process_group=managed_process_group)
            return_code = 0
        elif plan.planned_backend == "archive":
            snapshot_root = workspace / "inputs"
            snapshot_root.mkdir(mode=0o700)
            command = _archive_command(plan, execution_root, snapshot_root)
            output_descriptor = _pin_output_directory(root)
            execution_root.mkdir()
            return_code = subprocess.run(command, check=False).returncode
        else:
            from transformation_portal.lux_depth_v3.execution_lifecycle import consume_lux_execution_plan
            from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

            prepared = consume_lux_execution_plan(plan_bytes, authorized_input_root=Path(plan.input_root))
            output_descriptor = _pin_output_directory(root)
            execution_root.mkdir()
            orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root=execution_root)
            results = orchestrator.enhance_batch(input_dir=prepared.input_root)
            return_code = 1 if any(result.get("status") == "error" for result in results) else 0
        # A renamed/replaced path is no longer the admitted locator. Even if
        # changed after this check, descriptor-relative export stays pinned.
        current = open_directory(root)
        try:
            pinned = os.fstat(output_descriptor)
            observed = os.fstat(current)
            if (pinned.st_dev, pinned.st_ino) != (observed.st_dev, observed.st_ino):
                raise ExecutionPlanError("Admitted output directory changed during execution")
        finally:
            os.close(current)
        _export_execution_outputs(execution_root, output_descriptor)
        return return_code
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if owned_workspace:
            remove_execution_workspace(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consume an admitted, canonical execution plan.")
    parser.add_argument("--plan-file", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-workspace", type=Path)
    parser.add_argument("--bindings-file", type=Path)
    parser.add_argument("--bindings-sha256")
    args = parser.parse_args()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(args.plan_file, flags), "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PLAN_BODY_BYTES:
            raise ExecutionPlanError("Dispatch plan carrier is not a bounded regular file")
        data = source.read(MAX_PLAN_BODY_BYTES + 1)
    if hashlib.sha256(data).hexdigest() != args.plan_sha256:
        raise ExecutionPlanError("Dispatch plan carrier digest does not match the claim")
    if (args.bindings_file is None) != (args.bindings_sha256 is None):
        raise ExecutionPlanError("Dispatch bindings require both file and digest")
    bindings = None
    if args.bindings_file is not None:
        with os.fdopen(os.open(args.bindings_file, flags), "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PHOTOGRAPHY_BINDINGS_BYTES:
                raise ExecutionPlanError("Dispatch bindings carrier is not a bounded regular file")
            bindings = source.read(MAX_PHOTOGRAPHY_BINDINGS_BYTES + 1)
        if hashlib.sha256(bindings).hexdigest() != args.bindings_sha256:
            raise ExecutionPlanError("Dispatch bindings carrier digest does not match the claim")

    def terminate_dispatch(signum: int, _frame: Any) -> None:
        # The isolated photography session owns another process group. Unwind
        # its context manager on normal managed cancellation so it is reaped.
        raise SystemExit(128 + signum)

    previous_handler = signal.signal(signal.SIGTERM, terminate_dispatch)
    try:
        return execute_dispatch_plan(
            data,
            output_root=args.output_root,
            execution_workspace=args.execution_workspace,
            execution_bindings=bindings,
            managed_process_group=True,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
