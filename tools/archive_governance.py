#!/usr/bin/env python3
"""Archive governance orchestration CLI (tp.archive.machine.v1)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import (  # pylint: disable=wrong-import-position
    CANONICAL_PROFILES,
    emit_machine_payload,
    make_machine_envelope,
    make_typed_error,
)
from premis_events import append_event, build_premis_event  # pylint: disable=wrong-import-position

EXIT_SUCCESS = 0
EXIT_OTHER_FAILURE = 5


def _run_tool(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _normalize_process_returncode(return_code: int) -> tuple[int, str | None]:
    if 0 <= return_code <= 255:
        return return_code, None
    if return_code < 0:
        return EXIT_OTHER_FAILURE, f"terminated by signal {abs(return_code)}"
    return EXIT_OTHER_FAILURE, f"out-of-range exit code {return_code}"


def _emit_result(
    *,
    args: argparse.Namespace,
    command_name: str,
    exit_code: int,
    data: dict[str, Any],
    error: dict[str, Any] | None = None,
) -> int:
    if args.json:
        envelope = make_machine_envelope(
            command=command_name,
            exit_code=exit_code,
            data=data,
            error=error,
        )
        emit_machine_payload(
            envelope=envelope,
            pretty=bool(args.json_pretty),
            json_output=Path(args.json_output) if args.json_output else None,
            canonical_profile=args.json_canonical_profile,
        )
        return exit_code

    if data.get("stdout"):
        print(data["stdout"])
    if data.get("stderr"):
        print(data["stderr"], file=sys.stderr)
    if error is not None:
        print(error.get("message", "Command failed"), file=sys.stderr)
    return exit_code


def _record_premis(
    *,
    premis_log: str | None,
    premis_agent_id: str,
    event_type: str,
    event_detail: str,
    success: bool,
    object_ids: list[str],
) -> None:
    if not premis_log:
        return

    payload = build_premis_event(
        event_type=event_type,
        event_detail=event_detail,
        event_outcome="success" if success else "failure",
        agent_id=premis_agent_id,
        object_ids=object_ids,
    )
    append_event(Path(premis_log), payload)


def _tool_failure_error(command_name: str, return_code: int, stderr: str) -> dict[str, Any]:
    normalized_exit_code, normalization_note = _normalize_process_returncode(return_code)
    detail = _first_nonempty_line(stderr) or "tool exited non-zero"
    if normalization_note:
        message = f"{command_name} failed with return {return_code} ({normalization_note}): {detail}"
    else:
        message = f"{command_name} failed with exit {normalized_exit_code}: {detail}"
    return make_typed_error(
        type_name="ToolExecutionError",
        message=message,
        exit_code=normalized_exit_code,
        exit_name="OTHER_FAILURE",
        priority=20,
    )


def _tool_unavailable_error(command_name: str, script_path: Path) -> dict[str, Any]:
    return make_typed_error(
        type_name="ToolUnavailableError",
        message=f"{command_name} unavailable: missing tool script at {script_path}",
        exit_code=EXIT_OTHER_FAILURE,
        exit_name="OTHER_FAILURE",
        priority=20,
    )


def _handle_fixity_scan(args: argparse.Namespace) -> int:
    script_path = PROJECT_ROOT / "tools" / "archive_hash_manifest.py"
    out_dir = Path(args.out_dir)
    if not script_path.is_file():
        return _emit_result(
            args=args,
            command_name="fixity-scan",
            exit_code=EXIT_OTHER_FAILURE,
            data={
                "tool": "archive_hash_manifest.py",
                "archive_index": args.archive_index,
                "archive_root": args.archive_root,
                "out_dir": args.out_dir,
                "workers": int(args.workers),
                "strict": bool(args.strict),
                "strict_identity": bool(args.strict_identity),
                "validate_schemas": bool(args.validate_schemas),
                "artifacts": {
                    "hash_manifest": str(out_dir / "hash_manifest.csv.gz"),
                    "hash_summary": str(out_dir / "hash_summary.json"),
                    "merkle_roots": str(out_dir / "merkle_roots.json"),
                },
                "stdout": "",
                "stderr": "",
                "missing_tool": str(script_path),
            },
            error=_tool_unavailable_error("fixity-scan", script_path),
        )

    command = [
        sys.executable,
        str(script_path),
        "--archive-index",
        args.archive_index,
        "--archive-root",
        args.archive_root,
        "--out-dir",
        args.out_dir,
        "--workers",
        str(args.workers),
    ]
    if args.strict:
        command.append("--strict")
    if args.strict_identity:
        command.append("--strict-identity")
    if args.validate_schemas:
        command.append("--validate-schemas")

    result = _run_tool(command)
    data = {
        "tool": "archive_hash_manifest.py",
        "archive_index": args.archive_index,
        "archive_root": args.archive_root,
        "out_dir": args.out_dir,
        "workers": int(args.workers),
        "strict": bool(args.strict),
        "strict_identity": bool(args.strict_identity),
        "validate_schemas": bool(args.validate_schemas),
        "artifacts": {
            "hash_manifest": str(out_dir / "hash_manifest.csv.gz"),
            "hash_summary": str(out_dir / "hash_summary.json"),
            "merkle_roots": str(out_dir / "merkle_roots.json"),
        },
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

    error = None
    if result.returncode != 0:
        error = _tool_failure_error("fixity-scan", result.returncode, result.stderr)
    normalized_exit_code, _ = _normalize_process_returncode(result.returncode)

    _record_premis(
        premis_log=args.premis_log,
        premis_agent_id=args.premis_agent_id,
        event_type="fixityGeneration",
        event_detail="archive_hash_manifest execution",
        success=result.returncode == 0,
        object_ids=[data["artifacts"]["hash_manifest"], data["artifacts"]["hash_summary"], data["artifacts"]["merkle_roots"]],
    )
    return _emit_result(
        args=args,
        command_name="fixity-scan",
        exit_code=normalized_exit_code,
        data=data,
        error=error,
    )


def _handle_fixity_verify(args: argparse.Namespace) -> int:
    script_path = PROJECT_ROOT / "tools" / "verify_hash_manifest.py"
    report_path = Path(args.report_path) if args.report_path else Path(args.hash_manifest).parent / "verification_report.json"
    if not script_path.is_file():
        return _emit_result(
            args=args,
            command_name="fixity-verify",
            exit_code=EXIT_OTHER_FAILURE,
            data={
                "tool": "verify_hash_manifest.py",
                "hash_manifest": args.hash_manifest,
                "archive_root": args.archive_root,
                "report_path": str(report_path),
                "workers": int(args.workers),
                "verify_sample": int(args.verify_sample),
                "stdout": "",
                "stderr": "",
                "missing_tool": str(script_path),
            },
            error=_tool_unavailable_error("fixity-verify", script_path),
        )

    command = [
        sys.executable,
        str(script_path),
        "--hash-manifest",
        args.hash_manifest,
        "--archive-root",
        args.archive_root,
        "--report-path",
        str(report_path),
        "--workers",
        str(args.workers),
    ]
    if args.verify_sample > 0:
        command.extend(["--verify-sample", str(args.verify_sample)])
    else:
        command.append("--verify-all")

    result = _run_tool(command)
    data = {
        "tool": "verify_hash_manifest.py",
        "hash_manifest": args.hash_manifest,
        "archive_root": args.archive_root,
        "report_path": str(report_path),
        "workers": int(args.workers),
        "verify_sample": int(args.verify_sample),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

    error = None
    if result.returncode != 0:
        error = _tool_failure_error("fixity-verify", result.returncode, result.stderr)
    normalized_exit_code, _ = _normalize_process_returncode(result.returncode)

    _record_premis(
        premis_log=args.premis_log,
        premis_agent_id=args.premis_agent_id,
        event_type="fixityCheck",
        event_detail="verify_hash_manifest execution",
        success=result.returncode == 0,
        object_ids=[args.hash_manifest, str(report_path)],
    )
    return _emit_result(
        args=args,
        command_name="fixity-verify",
        exit_code=normalized_exit_code,
        data=data,
        error=error,
    )


def _run_wrapped_tool(
    *,
    args: argparse.Namespace,
    command_name: str,
    script_name: str,
    tool_args: list[str],
    premis_event_type: str | None,
    premis_event_detail: str,
    premis_object_ids: list[str],
) -> int:
    script_path = PROJECT_ROOT / "tools" / script_name
    if not script_path.is_file():
        return _emit_result(
            args=args,
            command_name=command_name,
            exit_code=EXIT_OTHER_FAILURE,
            data={
                "tool": script_name,
                "arguments": tool_args,
                "stdout": "",
                "stderr": "",
                "missing_tool": str(script_path),
            },
            error=_tool_unavailable_error(command_name, script_path),
        )

    command = [sys.executable, str(script_path), *tool_args]
    result = _run_tool(command)

    data = {
        "tool": script_name,
        "arguments": tool_args,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

    error = None
    if result.returncode != 0:
        error = _tool_failure_error(command_name, result.returncode, result.stderr)
    normalized_exit_code, _ = _normalize_process_returncode(result.returncode)

    if premis_event_type is not None:
        _record_premis(
            premis_log=args.premis_log,
            premis_agent_id=args.premis_agent_id,
            event_type=premis_event_type,
            event_detail=premis_event_detail,
            success=result.returncode == 0,
            object_ids=premis_object_ids,
        )

    return _emit_result(
        args=args,
        command_name=command_name,
        exit_code=normalized_exit_code,
        data=data,
        error=error,
    )


def _handle_manifest_build(args: argparse.Namespace) -> int:
    tool_args = [
        "--archive-index",
        args.archive_index,
        "--hash-manifest",
        args.hash_manifest,
        "--archive-root",
        args.archive_root,
        "--out-jsonl",
        args.out_jsonl,
        "--out-summary",
        args.out_summary,
        "--collection-id",
        args.collection_id,
        "--owner",
        args.owner,
    ]
    if args.rights_jsonl:
        tool_args.extend(["--rights-jsonl", args.rights_jsonl])

    return _run_wrapped_tool(
        args=args,
        command_name="manifest-build",
        script_name="build_archive_manifest_v2.py",
        tool_args=tool_args,
        premis_event_type="metadataExtraction",
        premis_event_detail="build_archive_manifest_v2 execution",
        premis_object_ids=[args.out_jsonl, args.out_summary],
    )


def _handle_rights_apply(args: argparse.Namespace) -> int:
    return _run_wrapped_tool(
        args=args,
        command_name="rights-apply",
        script_name="apply_rights_policy.py",
        tool_args=[
            "--manifest-jsonl",
            args.manifest_jsonl,
            "--policy-yaml",
            args.policy_yaml,
            "--out-jsonl",
            args.out_jsonl,
            "--out-summary",
            args.out_summary,
        ],
        premis_event_type="rightsModification",
        premis_event_detail="apply_rights_policy execution",
        premis_object_ids=[args.out_jsonl, args.out_summary],
    )


def _handle_bag_build(args: argparse.Namespace) -> int:
    tool_args = [
        "build",
        "--manifest-jsonl",
        args.manifest_jsonl,
        "--archive-root",
        args.archive_root,
        "--bag-dir",
        args.bag_dir,
        "--source-organization",
        args.source_organization,
    ]
    if args.report_json:
        tool_args.extend(["--report-json", args.report_json])
    if args.validate_with_bagit_python:
        tool_args.append("--validate-with-bagit-python")

    object_ids = [args.bag_dir]
    if args.report_json:
        object_ids.append(args.report_json)

    return _run_wrapped_tool(
        args=args,
        command_name="bag-build",
        script_name="archive_bagit.py",
        tool_args=tool_args,
        premis_event_type="ingestion",
        premis_event_detail="archive_bagit build execution",
        premis_object_ids=object_ids,
    )


def _handle_bag_validate(args: argparse.Namespace) -> int:
    tool_args = [
        "validate",
        "--bag-dir",
        args.bag_dir,
        "--report-json",
        args.report_json,
    ]
    if args.validate_with_bagit_python:
        tool_args.append("--validate-with-bagit-python")

    return _run_wrapped_tool(
        args=args,
        command_name="bag-validate",
        script_name="archive_bagit.py",
        tool_args=tool_args,
        premis_event_type="validation",
        premis_event_detail="archive_bagit validate execution",
        premis_object_ids=[args.bag_dir, args.report_json],
    )


def _handle_premis_export(args: argparse.Namespace) -> int:
    tool_args = [
        "emit",
        "--out-jsonl",
        args.out_jsonl,
        "--event-type",
        args.event_type,
        "--event-detail",
        args.event_detail,
        "--event-outcome",
        args.event_outcome,
        "--agent-id",
        args.agent_id,
    ]
    if args.event_datetime:
        tool_args.extend(["--event-datetime", args.event_datetime])
    if args.event_id:
        tool_args.extend(["--event-id", args.event_id])
    for object_id in args.object_id:
        tool_args.extend(["--object-id", object_id])

    return _run_wrapped_tool(
        args=args,
        command_name="premis-export",
        script_name="premis_events.py",
        tool_args=tool_args,
        premis_event_type=None,
        premis_event_detail="",
        premis_object_ids=[],
    )


def _handle_dedup_plan(args: argparse.Namespace) -> int:
    return _run_wrapped_tool(
        args=args,
        command_name="dedup-plan",
        script_name="build_dedup_ledger.py",
        tool_args=[
            "--manifest-jsonl",
            args.manifest_jsonl,
            "--out-ledger",
            args.out_ledger,
            "--out-summary",
            args.out_summary,
            "--approver",
            args.approver,
        ],
        premis_event_type="metadataModification",
        premis_event_detail="build_dedup_ledger execution",
        premis_object_ids=[args.out_ledger, args.out_summary],
    )


def _handle_mets_export(args: argparse.Namespace) -> int:
    return _run_wrapped_tool(
        args=args,
        command_name="mets-export",
        script_name="build_mets_structmap.py",
        tool_args=[
            "--manifest-jsonl",
            args.manifest_jsonl,
            "--out-xml",
            args.out_xml,
            "--out-summary",
            args.out_summary,
            "--href-prefix",
            args.href_prefix,
        ],
        premis_event_type="metadataModification",
        premis_event_detail="build_mets_structmap execution",
        premis_object_ids=[args.out_xml, args.out_summary],
    )


def _handle_prov_export(args: argparse.Namespace) -> int:
    return _run_wrapped_tool(
        args=args,
        command_name="prov-export",
        script_name="export_prov_stac.py",
        tool_args=[
            "--manifest-jsonl",
            args.manifest_jsonl,
            "--out-prov-jsonld",
            args.out_prov_jsonld,
            "--out-summary",
            args.out_summary,
            "--datetime-field",
            args.datetime_field,
            "--no-require-stac",
        ],
        premis_event_type="metadataExtraction",
        premis_event_detail="export_prov_stac prov-only execution",
        premis_object_ids=[args.out_prov_jsonld, args.out_summary],
    )


def _handle_stac_export(args: argparse.Namespace) -> int:
    tool_args = [
        "--manifest-jsonl",
        args.manifest_jsonl,
        "--out-prov-jsonld",
        args.out_prov_jsonld,
        "--out-stac-catalog",
        args.out_stac_catalog,
        "--out-summary",
        args.out_summary,
        "--datetime-field",
        args.datetime_field,
    ]
    if args.out_stac_items_dir:
        tool_args.extend(["--out-stac-items-dir", args.out_stac_items_dir])
    if args.require_stac:
        tool_args.append("--require-stac")

    object_ids = [args.out_prov_jsonld, args.out_summary, args.out_stac_catalog]
    if args.out_stac_items_dir:
        object_ids.append(args.out_stac_items_dir)

    return _run_wrapped_tool(
        args=args,
        command_name="stac-export",
        script_name="export_prov_stac.py",
        tool_args=tool_args,
        premis_event_type="metadataExtraction",
        premis_event_detail="export_prov_stac stac execution",
        premis_object_ids=object_ids,
    )


def _handle_sealed_eval_run(args: argparse.Namespace) -> int:
    script_path = PROJECT_ROOT / "scripts" / "pipelines" / "run_sealed_eval_72h.sh"
    if not script_path.is_file():
        return _emit_result(
            args=args,
            command_name="sealed-eval-run",
            exit_code=EXIT_OTHER_FAILURE,
            data={
                "tool": "run_sealed_eval_72h.sh",
                "archive_index": args.archive_index,
                "archive_root": args.archive_root,
                "out_root": args.out_root,
                "subset_root": args.subset_root or args.archive_root,
                "eval_command": args.eval_command,
                "validate_schemas": bool(args.validate_schemas),
                "allow_writable_subset": bool(args.allow_writable_subset),
                "stdout": "",
                "stderr": "",
                "missing_tool": str(script_path),
            },
            error=_tool_unavailable_error("sealed-eval-run", script_path),
        )

    command = [
        str(script_path),
        "--archive-index",
        args.archive_index,
        "--archive-root",
        args.archive_root,
        "--out-root",
        args.out_root,
    ]
    if args.subset_root:
        command.extend(["--subset-root", args.subset_root])
    if args.eval_command:
        command.extend(["--eval-command", args.eval_command])
    if not args.validate_schemas:
        command.append("--no-validate-schemas")
    if args.allow_writable_subset:
        command.append("--allow-writable-subset")

    result = _run_tool(command)
    data = {
        "tool": "run_sealed_eval_72h.sh",
        "archive_index": args.archive_index,
        "archive_root": args.archive_root,
        "out_root": args.out_root,
        "subset_root": args.subset_root or args.archive_root,
        "eval_command": args.eval_command,
        "validate_schemas": bool(args.validate_schemas),
        "allow_writable_subset": bool(args.allow_writable_subset),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    error = None
    if result.returncode != 0:
        error = _tool_failure_error("sealed-eval-run", result.returncode, result.stderr)
    normalized_exit_code, _ = _normalize_process_returncode(result.returncode)

    _record_premis(
        premis_log=args.premis_log,
        premis_agent_id=args.premis_agent_id,
        event_type="fixityCheck",
        event_detail="sealed evaluation pre/post fixity run",
        success=result.returncode == 0,
        object_ids=[args.out_root],
    )

    return _emit_result(
        args=args,
        command_name="sealed-eval-run",
        exit_code=normalized_exit_code,
        data=data,
        error=error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit tp.archive.machine.v1 envelope JSON")
    parser.add_argument("--json-pretty", action="store_true", help="Pretty print machine JSON (requires --json)")
    parser.add_argument("--json-output", default=None, help="Write machine JSON payload to file")
    parser.add_argument(
        "--json-canonical-profile",
        default="canonical_v1",
        choices=CANONICAL_PROFILES,
        help="Canonical serialization profile",
    )
    parser.add_argument("--premis-log", default=None, help="Optional PREMIS JSONL append path")
    parser.add_argument("--premis-agent-id", default="tp.archive.governance.v1", help="PREMIS software agent identifier")

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_fixity_scan = subparsers.add_parser("fixity-scan", help="Generate archive hash/fixity artifacts")
    parser_fixity_scan.add_argument("--archive-index", required=True)
    parser_fixity_scan.add_argument("--archive-root", required=True)
    parser_fixity_scan.add_argument("--out-dir", required=True)
    parser_fixity_scan.add_argument("--workers", type=int, default=1)
    parser_fixity_scan.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    parser_fixity_scan.add_argument("--strict-identity", action=argparse.BooleanOptionalAction, default=False)
    parser_fixity_scan.add_argument("--validate-schemas", action=argparse.BooleanOptionalAction, default=True)
    parser_fixity_scan.set_defaults(func=_handle_fixity_scan)

    parser_fixity_verify = subparsers.add_parser("fixity-verify", help="Verify archive against hash manifest")
    parser_fixity_verify.add_argument("--hash-manifest", required=True)
    parser_fixity_verify.add_argument("--archive-root", required=True)
    parser_fixity_verify.add_argument("--report-path", default=None)
    parser_fixity_verify.add_argument("--verify-sample", type=int, default=0)
    parser_fixity_verify.add_argument("--workers", type=int, default=1)
    parser_fixity_verify.set_defaults(func=_handle_fixity_verify)

    parser_manifest = subparsers.add_parser("manifest-build", help="Build tp.archive.manifest.v2 artifacts")
    parser_manifest.add_argument("--archive-index", required=True)
    parser_manifest.add_argument("--hash-manifest", required=True)
    parser_manifest.add_argument("--archive-root", required=True)
    parser_manifest.add_argument("--out-jsonl", required=True)
    parser_manifest.add_argument("--out-summary", required=True)
    parser_manifest.add_argument("--rights-jsonl", default=None)
    parser_manifest.add_argument("--collection-id", default="UNSPECIFIED")
    parser_manifest.add_argument("--owner", default="UNSPECIFIED")
    parser_manifest.set_defaults(func=_handle_manifest_build)

    parser_rights = subparsers.add_parser("rights-apply", help="Apply rights policy over manifest entries")
    parser_rights.add_argument("--manifest-jsonl", required=True)
    parser_rights.add_argument("--policy-yaml", required=True)
    parser_rights.add_argument("--out-jsonl", required=True)
    parser_rights.add_argument("--out-summary", required=True)
    parser_rights.set_defaults(func=_handle_rights_apply)

    parser_bag_build = subparsers.add_parser("bag-build", help="Build deterministic BagIt package")
    parser_bag_build.add_argument("--manifest-jsonl", required=True)
    parser_bag_build.add_argument("--archive-root", required=True)
    parser_bag_build.add_argument("--bag-dir", required=True)
    parser_bag_build.add_argument("--report-json", default=None)
    parser_bag_build.add_argument("--source-organization", default="UNSPECIFIED")
    parser_bag_build.add_argument("--validate-with-bagit-python", action=argparse.BooleanOptionalAction, default=False)
    parser_bag_build.set_defaults(func=_handle_bag_build)

    parser_bag_validate = subparsers.add_parser("bag-validate", help="Validate deterministic BagIt package")
    parser_bag_validate.add_argument("--bag-dir", required=True)
    parser_bag_validate.add_argument("--report-json", required=True)
    parser_bag_validate.add_argument("--validate-with-bagit-python", action=argparse.BooleanOptionalAction, default=False)
    parser_bag_validate.set_defaults(func=_handle_bag_validate)

    parser_premis = subparsers.add_parser("premis-export", help="Append PREMIS event JSONL record")
    parser_premis.add_argument("--out-jsonl", required=True)
    parser_premis.add_argument("--event-type", required=True)
    parser_premis.add_argument("--event-detail", required=True)
    parser_premis.add_argument("--event-outcome", required=True, choices=["success", "failure"])
    parser_premis.add_argument("--agent-id", default="tp.archive.governance.v1")
    parser_premis.add_argument("--object-id", action="append", default=[])
    parser_premis.add_argument("--event-datetime", default=None)
    parser_premis.add_argument("--event-id", default=None)
    parser_premis.set_defaults(func=_handle_premis_export)

    parser_dedup = subparsers.add_parser("dedup-plan", help="Generate checksum dedup planning ledger")
    parser_dedup.add_argument("--manifest-jsonl", required=True)
    parser_dedup.add_argument("--out-ledger", required=True)
    parser_dedup.add_argument("--out-summary", required=True)
    parser_dedup.add_argument("--approver", default="UNSPECIFIED")
    parser_dedup.set_defaults(func=_handle_dedup_plan)

    parser_mets = subparsers.add_parser("mets-export", help="Export METS fileSec + structMap")
    parser_mets.add_argument("--manifest-jsonl", required=True)
    parser_mets.add_argument("--out-xml", required=True)
    parser_mets.add_argument("--out-summary", required=True)
    parser_mets.add_argument("--href-prefix", default="data")
    parser_mets.set_defaults(func=_handle_mets_export)

    parser_prov = subparsers.add_parser("prov-export", help="Export PROV JSON-LD")
    parser_prov.add_argument("--manifest-jsonl", required=True)
    parser_prov.add_argument("--out-prov-jsonld", required=True)
    parser_prov.add_argument("--out-summary", required=True)
    parser_prov.add_argument("--datetime-field", default="modified_utc")
    parser_prov.set_defaults(func=_handle_prov_export)

    parser_stac = subparsers.add_parser("stac-export", help="Export STAC catalog when geometry/timestamps exist")
    parser_stac.add_argument("--manifest-jsonl", required=True)
    parser_stac.add_argument("--out-prov-jsonld", required=True)
    parser_stac.add_argument("--out-stac-catalog", required=True)
    parser_stac.add_argument("--out-stac-items-dir", default=None)
    parser_stac.add_argument("--out-summary", required=True)
    parser_stac.add_argument("--datetime-field", default="modified_utc")
    parser_stac.add_argument("--require-stac", action=argparse.BooleanOptionalAction, default=False)
    parser_stac.set_defaults(func=_handle_stac_export)

    parser_sealed = subparsers.add_parser("sealed-eval-run", help="Run sealed 72-hour evaluation harness")
    parser_sealed.add_argument("--archive-index", required=True)
    parser_sealed.add_argument("--archive-root", required=True)
    parser_sealed.add_argument("--out-root", default="archive_reports/sealed_eval")
    parser_sealed.add_argument("--subset-root", default=None)
    parser_sealed.add_argument("--eval-command", default=None)
    parser_sealed.add_argument("--validate-schemas", action=argparse.BooleanOptionalAction, default=True)
    parser_sealed.add_argument("--allow-writable-subset", action=argparse.BooleanOptionalAction, default=False)
    parser_sealed.set_defaults(func=_handle_sealed_eval_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.json_pretty or args.json_output) and not args.json:
        parser.error("--json-pretty and --json-output require --json")

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
