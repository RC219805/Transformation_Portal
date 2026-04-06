#!/usr/bin/env python3
"""Safe local execution-readiness audit for the four portal pipelines."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as orchestrator_app


class AuditFailure(RuntimeError):
    """Raised when the readiness audit cannot complete safely."""


ARCHIVE_TOOL = PROJECT_ROOT / "tools" / "archive_governance.py"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "archive_small"
FIXTURE_ARCHIVE_ROOT = FIXTURE_DIR / "archive_root"
FIXTURE_ARCHIVE_INDEX = FIXTURE_DIR / "archive_index_normalized.csv.gz"
FIXTURE_HASH_MANIFEST = FIXTURE_DIR / "golden" / "hash_manifest.csv.gz"
RIGHTS_POLICY = PROJECT_ROOT / "policy" / "archive" / "rights_flags.yml"


def _default_output_dir() -> Path:
    kwargs: Dict[str, Any] = {"prefix": "tp-pipeline-readiness-audit-"}
    if os.name != "nt" and Path("/tmp").exists():
        kwargs["dir"] = "/tmp"
    return Path(tempfile.mkdtemp(**kwargs))


def _resolve_output_dir(raw_output_dir: str) -> tuple[Path, bool]:
    candidate = str(raw_output_dir).strip()
    if candidate:
        return Path(candidate).resolve(), False
    return _default_output_dir(), True


def _should_cleanup_output_dir(*, keep_output: bool, output_dir_is_temp: bool) -> bool:
    return output_dir_is_temp and not keep_output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional working directory for generated audit artifacts (defaults to a temp dir).",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional file path to write the final readiness matrix JSON.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Preserve the generated artifact directory instead of deleting it.",
    )
    return parser.parse_args(argv)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_fixture(path: Path, kind: str) -> None:
    if kind == "dir":
        if not path.is_dir():
            raise AuditFailure(f"Required fixture directory is missing: {path}")
        return
    if not path.is_file():
        raise AuditFailure(f"Required fixture file is missing: {path}")


def _run_archive_governance(*args: str) -> Dict[str, Any]:
    command = [sys.executable, str(ARCHIVE_TOOL), "--json", *args]
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    payload: Dict[str, Any] = {}
    raw_stdout = result.stdout.strip()
    if raw_stdout:
        try:
            payload = json.loads(raw_stdout)
        except json.JSONDecodeError as exc:
            raise AuditFailure(
                f"Archive governance command returned non-JSON stdout for {' '.join(args)!r}: {raw_stdout[:400]!r}"
            ) from exc
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "payload": payload,
    }


def _require_command_success(label: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if int(result["returncode"]) != 0:
        raise AuditFailure(f"{label} failed with exit code {result['returncode']}: {str(result.get('stderr') or '').strip()}")
    return result


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()).as_posix())
    except ValueError:
        return str(path.resolve())


def _readiness_snapshot(
    pipeline: str,
    args: Dict[str, Any] | None = None,
    *,
    require_dispatch_inputs: bool,
) -> Dict[str, Any]:
    raw_args = dict(args or {})
    return orchestrator_app._evaluate_pipeline_readiness(
        pipeline,
        raw_args,
        require_dispatch_inputs=require_dispatch_inputs,
    )


def _lux_depth_audit_entry() -> Dict[str, Any]:
    readiness = _readiness_snapshot("lux-depth-v3", require_dispatch_inputs=False)
    return {
        "canonical_command": readiness.get("canonical_command"),
        "base_status": readiness.get("status"),
        "canary_status": readiness.get("canary_status"),
        "missing_prerequisites": readiness.get("missing_prerequisites") or [],
        "runner_details": readiness.get("runner_details") or {},
        "notes": readiness.get("notes") or [],
    }


def run_audit(output_dir: Path) -> Dict[str, Any]:
    _ensure_fixture(ARCHIVE_TOOL, "file")
    _ensure_fixture(FIXTURE_ARCHIVE_ROOT, "dir")
    _ensure_fixture(FIXTURE_ARCHIVE_INDEX, "file")
    _ensure_fixture(FIXTURE_HASH_MANIFEST, "file")
    _ensure_fixture(RIGHTS_POLICY, "file")

    output_dir.mkdir(parents=True, exist_ok=True)

    hash_manifest = output_dir / "hash_manifest.csv.gz"
    hash_summary = output_dir / "hash_summary.json"
    merkle_roots = output_dir / "merkle_roots.json"
    manifest_jsonl = output_dir / "archive_manifest_v2.jsonl"
    manifest_summary = output_dir / "archive_manifest_v2.summary.json"
    rights_jsonl = output_dir / "archive_manifest_v2.rights.jsonl"
    rights_summary = output_dir / "asset_rights.summary.json"
    bag_dir = output_dir / "bag"
    bag_report = output_dir / "bag_build_report.json"
    mets_xml = output_dir / "mets_export.xml"
    mets_summary = output_dir / "mets_summary.json"

    gate_a_baseline = _readiness_snapshot("archive-gate-a", require_dispatch_inputs=False)
    gate_b_baseline = _readiness_snapshot("archive-gate-b", require_dispatch_inputs=False)
    gate_c_baseline = _readiness_snapshot("archive-gate-c", require_dispatch_inputs=False)

    fixity_result = _require_command_success(
        "archive-gate-a/fixity-scan",
        _run_archive_governance(
            "fixity-scan",
            "--archive-index",
            str(FIXTURE_ARCHIVE_INDEX),
            "--archive-root",
            str(FIXTURE_ARCHIVE_ROOT),
            "--out-dir",
            str(output_dir),
            "--workers",
            "1",
        ),
    )

    manifest_result = _require_command_success(
        "archive-gate-a/manifest-build",
        _run_archive_governance(
            "manifest-build",
            "--archive-index",
            str(FIXTURE_ARCHIVE_INDEX),
            "--hash-manifest",
            str(FIXTURE_HASH_MANIFEST),
            "--archive-root",
            str(FIXTURE_ARCHIVE_ROOT),
            "--out-jsonl",
            str(manifest_jsonl),
            "--out-summary",
            str(manifest_summary),
            "--collection-id",
            "readiness_audit",
            "--owner",
            "transformation_portal",
        ),
    )

    rights_result = _require_command_success(
        "archive-gate-a/rights-apply",
        _run_archive_governance(
            "rights-apply",
            "--manifest-jsonl",
            str(manifest_jsonl),
            "--policy-yaml",
            str(RIGHTS_POLICY),
            "--out-jsonl",
            str(rights_jsonl),
            "--out-summary",
            str(rights_summary),
        ),
    )

    gate_b_blocked = _readiness_snapshot(
        "archive-gate-b",
        {
            "input_dir": str(FIXTURE_ARCHIVE_ROOT),
            "output_dir": str(output_dir),
            "archive_command": "bag-build",
        },
        require_dispatch_inputs=True,
    )
    gate_c_blocked = _readiness_snapshot(
        "archive-gate-c",
        {
            "input_dir": str(FIXTURE_ARCHIVE_ROOT),
            "output_dir": str(output_dir),
            "archive_command": "mets-export",
        },
        require_dispatch_inputs=True,
    )

    gate_b_ready = _readiness_snapshot(
        "archive-gate-b",
        {
            "input_dir": str(FIXTURE_ARCHIVE_ROOT),
            "output_dir": str(output_dir),
            "archive_command": "bag-build",
            "manifest_jsonl": str(rights_jsonl),
        },
        require_dispatch_inputs=True,
    )
    gate_c_ready = _readiness_snapshot(
        "archive-gate-c",
        {
            "input_dir": str(FIXTURE_ARCHIVE_ROOT),
            "output_dir": str(output_dir),
            "archive_command": "mets-export",
            "manifest_jsonl": str(rights_jsonl),
        },
        require_dispatch_inputs=True,
    )

    bag_result = _require_command_success(
        "archive-gate-b/bag-build",
        _run_archive_governance(
            "bag-build",
            "--manifest-jsonl",
            str(rights_jsonl),
            "--archive-root",
            str(FIXTURE_ARCHIVE_ROOT),
            "--bag-dir",
            str(bag_dir),
            "--report-json",
            str(bag_report),
        ),
    )

    mets_result = _require_command_success(
        "archive-gate-c/mets-export",
        _run_archive_governance(
            "mets-export",
            "--manifest-jsonl",
            str(rights_jsonl),
            "--out-xml",
            str(mets_xml),
            "--out-summary",
            str(mets_summary),
        ),
    )

    payload = {
        "schema": "tp.orchestrator.pipeline_readiness_audit.v1",
        "success": True,
        "data": {
            "generated_at": _now_iso(),
            "output_dir": str(output_dir),
            "fixtures": {
                "archive_root": str(FIXTURE_ARCHIVE_ROOT),
                "archive_index": str(FIXTURE_ARCHIVE_INDEX),
                "hash_manifest": str(FIXTURE_HASH_MANIFEST),
                "rights_policy": str(RIGHTS_POLICY),
            },
            "pipelines": {
                "lux-depth-v3": _lux_depth_audit_entry(),
                "archive-gate-a": {
                    "canonical_command": "fixity-scan",
                    "baseline_status": gate_a_baseline.get("status"),
                    "dispatch_readiness": _readiness_snapshot(
                        "archive-gate-a",
                        {
                            "input_dir": str(FIXTURE_ARCHIVE_ROOT),
                            "output_dir": str(output_dir),
                            "archive_command": "fixity-scan",
                            "archive_index": str(FIXTURE_ARCHIVE_INDEX),
                        },
                        require_dispatch_inputs=True,
                    ),
                    "command_exit_code": fixity_result["returncode"],
                    "artifacts": [
                        _relative_or_absolute(hash_manifest, output_dir),
                        _relative_or_absolute(hash_summary, output_dir),
                        _relative_or_absolute(merkle_roots, output_dir),
                    ],
                    "manifest_chain": {
                        "manifest_build_exit_code": manifest_result["returncode"],
                        "rights_apply_exit_code": rights_result["returncode"],
                        "manifest_jsonl": _relative_or_absolute(manifest_jsonl, output_dir),
                        "rights_manifest_jsonl": _relative_or_absolute(rights_jsonl, output_dir),
                    },
                },
                "archive-gate-b": {
                    "canonical_command": "bag-build",
                    "baseline_status": gate_b_baseline.get("status"),
                    "blocked_without_manifest": gate_b_blocked,
                    "dispatch_readiness": gate_b_ready,
                    "command_exit_code": bag_result["returncode"],
                    "artifacts": [
                        _relative_or_absolute(bag_dir / "bagit.txt", output_dir),
                        _relative_or_absolute(bag_report, output_dir),
                    ],
                },
                "archive-gate-c": {
                    "canonical_command": "mets-export",
                    "baseline_status": gate_c_baseline.get("status"),
                    "blocked_without_manifest": gate_c_blocked,
                    "dispatch_readiness": gate_c_ready,
                    "command_exit_code": mets_result["returncode"],
                    "artifacts": [
                        _relative_or_absolute(mets_xml, output_dir),
                        _relative_or_absolute(mets_summary, output_dir),
                    ],
                },
            },
        },
    }

    lux_entry = payload["data"]["pipelines"]["lux-depth-v3"]
    archive_entries = (
        payload["data"]["pipelines"]["archive-gate-a"],
        payload["data"]["pipelines"]["archive-gate-b"],
        payload["data"]["pipelines"]["archive-gate-c"],
    )
    payload["success"] = (
        lux_entry.get("base_status") == "ready"
        and all(int(entry["command_exit_code"]) == 0 for entry in archive_entries)
        and payload["data"]["pipelines"]["archive-gate-b"]["dispatch_readiness"].get("status") == "ready"
        and payload["data"]["pipelines"]["archive-gate-c"]["dispatch_readiness"].get("status") == "ready"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir, output_dir_is_temp = _resolve_output_dir(args.output_dir)
    cleanup_output_dir = _should_cleanup_output_dir(
        keep_output=bool(args.keep_output),
        output_dir_is_temp=output_dir_is_temp,
    )
    json_output = Path(args.json_output).resolve() if str(args.json_output).strip() else None

    try:
        payload = run_audit(output_dir)
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        print(rendered)
        if json_output is not None:
            json_output.parent.mkdir(parents=True, exist_ok=True)
            json_output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if payload.get("success") else 1
    finally:
        if cleanup_output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as exc:
        print(f"pipeline-readiness-audit: failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
