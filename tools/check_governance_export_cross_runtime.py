#!/usr/bin/env python3
"""
Verify governance export determinism and cross-runtime parity.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

from bundle_root_fixture import write_bundle_fixture_artifacts

# Reuse Phase 3.x cross-runtime tier exit codes for CI/tooling consistency.
EXIT_RUNTIME_FAILURE = 31
EXIT_GOVERNANCE_MISMATCH = 32


def _run_checked(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _write_hash_manifest(path: Path) -> None:
    rows = [
        ["driveA", "partA", "images/a.jpg", "20", "a" * 64, "ok", ""],
        ["driveA", "partA", "images/b.png", "40", "b" * 64, "ok", ""],
        ["driveB", "partB", "docs/readme.txt", "0", "", "missing", "missing"],
        ["driveB", "partB", "data/no_ext", "0", "", "skipped", "invalid_relpath_empty"],
    ]
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as text:
                text.write("# hash_algorithm=sha256\n")
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(
                    [
                        "origin_drive",
                        "partition",
                        "relpath",
                        "filesize_bytes",
                        "sha256",
                        "hash_status",
                        "error",
                    ]
                )
                writer.writerows(rows)


def _write_hash_summary(path: Path) -> None:
    payload = {
        "hash_algorithm": "sha256",
        "hash_manifest_schema_version": "1",
        "rows_total": 4,
        "hashed_ok": 2,
        "missing": 1,
        "unreadable": 0,
        "skipped": 1,
        "total_bytes_hashed": 60,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_risk_metadata(path: Path) -> None:
    payload = {
        "schema_version": "1",
        "risk_profile_id": "gpaiv1-2026-q1",
        "regulatory_regime": "EU_AI_ACT_ART53",
        "content_rights": {
            "policy_id": "CR-001",
            "policy_version": "2026.1",
            "policy_url": "https://example.invalid/policies/cr-001",
        },
        "copyright_compliance": {
            "tdm_opt_out_detection": True,
            "signals_supported": ["http_headers", "robots_txt"],
            "removal_process_documented": True,
            "removal_deltas_affect_root": True,
        },
        "risk_controls": [
            {
                "control_id": "RC-101",
                "description": "Schema validation gate",
                "status": "implemented",
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_source_taxonomy(path: Path) -> None:
    payload = {
        "schema_version": "1",
        "catalog_id": "training-catalog-2026-q1",
        "sources": [
            {
                "source_id": "web_scrape_2024_q1",
                "category": "web_scraped",
                "provenance_type": "third_party",
                "license_class": "public_web",
                "synthetic": False,
                "tdm_compliance_note": "robots and header signals honored",
                "crawler": "PortalBot/1.2",
                "collection_period": {"start": "2024-01-01", "end": "2024-03-31"},
                "top_domains": ["example.com", "example.org"],
                "processing_disclosure": "Duplicate and corruption filtering applied.",
            },
            {
                "source_id": "licensed_partner_set",
                "category": "licensed_private_dataset",
                "provenance_type": "third_party",
                "license_class": "contractual",
                "synthetic": False,
                "tdm_compliance_note": "License terms audited.",
                "record_count_estimate": 1200,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_risk_assessment_report(path: Path) -> None:
    payload = {
        "schema_version": "1",
        "assessment_id": "risk-assessment-2026-q1",
        "regulatory_regime": "CPPA_CCPA_2026",
        "assessment_version": "1",
        "purpose_specificity": "Assess significant-risk processing controls for model training data.",
        "categories_processed": ["web_scraped_text", "licensed_partner_data"],
        "sensitive_categories": ["biometric_templates"],
        "operational_elements": {
            "collection_methods": ["web_crawl", "contractual_transfer"],
            "retention_policy": "Retain for 5 years unless legal hold applies.",
            "recipients": ["security_team", "compliance_team"],
            "population_scale_estimate": {
                "estimated_records": 12345,
                "estimated_data_subjects": 6789,
            },
        },
        "benefits": "Improves reliability and defensibility of governance records.",
        "negative_impacts": ["profiling risk", "sensitive data leakage risk"],
        "safeguards": ["data minimization", "role-based access controls"],
        "review_approval": {
            "reviewer": "Jane Reviewer",
            "role": "Privacy Counsel",
            "date": "2026-02-24",
            "approval_status": "approved",
        },
        "next_review_due": "2026-08-24",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_cybersecurity_audit_record(path: Path) -> None:
    payload = {
        "schema_version": "1",
        "audit_record_id": "cyber-audit-2026",
        "regulatory_regime": "CPPA_CCPA_2026",
        "audit_standard": "NIST CSF 2.0",
        "auditor_name": "Independent Controls LLC",
        "auditor_independence_attested": True,
        "audit_period_start": "2026-01-01",
        "audit_period_end": "2026-12-31",
        "report_sha256": "a" * 64,
        "certification_attestation": {
            "signer": "Alex Signer",
            "date": "2027-01-15",
            "penalty_of_perjury": True,
        },
        "retention": {
            "retention_years": 5,
            "retention_basis": "CPPA five-year retention requirement",
            "records_location": "compliance/audits/2026",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_admt_governance(path: Path) -> None:
    payload = {
        "schema_version": "1",
        "governance_record_id": "admt-governance-2026-q1",
        "regulatory_regime": "CPPA_CCPA_2026",
        "admt_significant_decision_use": True,
        "pre_use_notice_template_version": "v1",
        "opt_out_mechanism_url": "https://example.invalid/privacy/admt-opt-out",
        "human_review_available": True,
        "appeal_process_documented": True,
        "access_explanation_available": True,
        "request_verification_rules": {
            "opt_out_requires_verification": False,
            "access_requires_verification": True,
            "rule_reference": "POL-ADMT-ACCESS-001",
        },
        "review_date": "2026-02-24",
        "owner_role": "AI Governance Lead",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_bundle_fixture(
    *,
    python_executable: str,
    project_root: Path,
    workspace: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    tools_dir = project_root / "tools"
    generate_tool = tools_dir / "generate_evidence_bundle_manifest.py"
    compute_tool = tools_dir / "compute_bundle_root.py"

    artifacts = write_bundle_fixture_artifacts(workspace / "bundle", timestamp_target="signature")
    _write_hash_manifest(artifacts["hash_manifest"])
    _write_hash_summary(artifacts["hash_summary"])

    _run_checked(
        [
            python_executable,
            str(generate_tool),
            "--roots",
            str(artifacts["roots"]),
            "--hash-manifest",
            str(artifacts["hash_manifest"]),
            "--hash-summary",
            str(artifacts["hash_summary"]),
            "--signature",
            str(artifacts["signature"]),
            "--timestamp-target",
            "signature",
            "--timestamp",
            str(artifacts["timestamp"]),
            "--out",
            str(artifacts["out"]),
        ],
        cwd=project_root,
    )
    _run_checked(
        [
            python_executable,
            str(compute_tool),
            "--bundle-manifest",
            str(artifacts["out"]),
            "--write",
        ],
        cwd=project_root,
    )

    risk_metadata = workspace / "risk_metadata.json"
    source_taxonomy = workspace / "source_taxonomy.json"
    risk_assessment = workspace / "risk_assessment_report.json"
    cybersecurity = workspace / "cybersecurity_audit_record.json"
    admt_governance = workspace / "admt_governance.json"

    _write_risk_metadata(risk_metadata)
    _write_source_taxonomy(source_taxonomy)
    _write_risk_assessment_report(risk_assessment)
    _write_cybersecurity_audit_record(cybersecurity)
    _write_admt_governance(admt_governance)
    return artifacts["out"], risk_metadata, source_taxonomy, risk_assessment, cybersecurity, admt_governance


def _generate_governance_export_bytes(
    *,
    python_executable: str,
    project_root: Path,
    bundle_manifest: Path,
    risk_metadata: Path,
    source_taxonomy: Path,
    risk_assessment: Path,
    cybersecurity: Path,
    admt_governance: Path,
    out_json: Path,
    out_md: Path,
    governance_export: Path,
) -> bytes:
    tool = project_root / "tools" / "regulatory_export.py"
    _run_checked(
        [
            python_executable,
            str(tool),
            "--bundle-manifest",
            str(bundle_manifest),
            "--risk-metadata",
            str(risk_metadata),
            "--source-taxonomy",
            str(source_taxonomy),
            "--risk-assessment-report",
            str(risk_assessment),
            "--cybersecurity-audit-record",
            str(cybersecurity),
            "--admt-governance",
            str(admt_governance),
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_md),
            "--governance-export",
            str(governance_export),
        ],
        cwd=project_root,
    )
    return governance_export.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-a", required=True, help="First Python interpreter path")
    parser.add_argument("--python-b", required=True, help="Second Python interpreter path")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root containing tools/",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (
                bundle_manifest,
                risk_metadata,
                source_taxonomy,
                risk_assessment,
                cybersecurity,
                admt_governance,
            ) = _prepare_bundle_fixture(
                python_executable=args.python_a,
                project_root=project_root,
                workspace=workspace,
            )

            manifest_payload = json.loads(bundle_manifest.read_text(encoding="utf-8"))
            bundle_root = str(manifest_payload["bundle_root_sha256"])

            bytes_a_1 = _generate_governance_export_bytes(
                python_executable=args.python_a,
                project_root=project_root,
                bundle_manifest=bundle_manifest,
                risk_metadata=risk_metadata,
                source_taxonomy=source_taxonomy,
                risk_assessment=risk_assessment,
                cybersecurity=cybersecurity,
                admt_governance=admt_governance,
                out_json=workspace / "regulatory_a_1.json",
                out_md=workspace / "regulatory_a_1.md",
                governance_export=workspace / "governance_a_1.json",
            )
            bytes_a_2 = _generate_governance_export_bytes(
                python_executable=args.python_a,
                project_root=project_root,
                bundle_manifest=bundle_manifest,
                risk_metadata=risk_metadata,
                source_taxonomy=source_taxonomy,
                risk_assessment=risk_assessment,
                cybersecurity=cybersecurity,
                admt_governance=admt_governance,
                out_json=workspace / "regulatory_a_2.json",
                out_md=workspace / "regulatory_a_2.md",
                governance_export=workspace / "governance_a_2.json",
            )
            bytes_b = _generate_governance_export_bytes(
                python_executable=args.python_b,
                project_root=project_root,
                bundle_manifest=bundle_manifest,
                risk_metadata=risk_metadata,
                source_taxonomy=source_taxonomy,
                risk_assessment=risk_assessment,
                cybersecurity=cybersecurity,
                admt_governance=admt_governance,
                out_json=workspace / "regulatory_b.json",
                out_md=workspace / "regulatory_b.md",
                governance_export=workspace / "governance_b.json",
            )
    except RuntimeError as exc:
        print(f"Governance cross-runtime parity check failed: {exc}")
        return EXIT_RUNTIME_FAILURE

    if bytes_a_1 != bytes_a_2:
        print("Governance determinism check failed: repeated run under python-a produced different bytes")
        return EXIT_GOVERNANCE_MISMATCH
    if bytes_a_1 != bytes_b:
        print("Governance cross-runtime parity check failed: python-a and python-b outputs differ")
        return EXIT_GOVERNANCE_MISMATCH

    governance_payload = json.loads(bytes_a_1.decode("utf-8"))
    payload_root = governance_payload["bundle_binding"]["bundle_root_sha256"]
    if payload_root != bundle_root:
        print("Governance root-binding check failed: governance export bundle_root_sha256 mismatch")
        return EXIT_GOVERNANCE_MISMATCH

    sha = hashlib.sha256(bytes_a_1).hexdigest()
    print(f"governance export sha256: {sha}")
    print("Governance determinism and cross-runtime parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
