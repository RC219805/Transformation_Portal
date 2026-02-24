"""Tests for Phase 3.5 regulatory export mode."""

from __future__ import annotations

import csv
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
GENERATE_TOOL = TOOLS_DIR / "generate_evidence_bundle_manifest.py"
COMPUTE_ROOT_TOOL = TOOLS_DIR / "compute_bundle_root.py"
REGULATORY_EXPORT_TOOL = TOOLS_DIR / "regulatory_export.py"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from bundle_root_fixture import write_bundle_fixture_artifacts  # noqa: E402

pytestmark = [pytest.mark.regression]


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


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
    path.write_text(
        json.dumps(
            {
                "hash_algorithm": "sha256",
                "hash_manifest_schema_version": "1",
                "rows_total": 4,
                "hashed_ok": 2,
                "missing": 1,
                "unreadable": 0,
                "skipped": 1,
                "total_bytes_hashed": 60,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_bundle_with_root(bundle_dir: Path) -> Path:
    artifacts = write_bundle_fixture_artifacts(bundle_dir, timestamp_target="signature")
    _write_hash_manifest(artifacts["hash_manifest"])
    _write_hash_summary(artifacts["hash_summary"])

    generated = _run_cli(
        [
            sys.executable,
            str(GENERATE_TOOL),
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
        ]
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr

    with_root = _run_cli(
        [
            sys.executable,
            str(COMPUTE_ROOT_TOOL),
            "--bundle-manifest",
            str(artifacts["out"]),
            "--write",
        ]
    )
    assert with_root.returncode == 0, with_root.stdout + with_root.stderr
    return artifacts["out"]


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
            {
                "source_id": "synthetic_aug_01",
                "category": "synthetic",
                "provenance_type": "first_party",
                "license_class": "internal",
                "synthetic": True,
                "tdm_compliance_note": "Generated from licensed base assets.",
                "byte_count_estimate": 1024,
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
            "geographic_scope": ["US", "EU"],
        },
        "benefits": "Improves reliability and defensibility of governance records.",
        "negative_impacts": ["profiling risk", "sensitive data leakage risk"],
        "safeguards": ["data minimization", "role-based access controls"],
        "pets_used": ["sampling"],
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
        "findings_summary": "No critical findings.",
        "corrective_actions": ["improve key rotation cadence"],
        "threshold_tier": "tier_1",
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
        "exception_paths": ["fraud_prevention_exception"],
        "review_date": "2026-02-24",
        "owner_role": "AI Governance Lead",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_regulatory_export(
    manifest_path: Path,
    risk_path: Path,
    taxonomy_path: Path,
    out_json: Path,
    out_md: Path,
    *,
    top_n: int = 10,
    risk_assessment_path: Path | None = None,
    cybersecurity_audit_path: Path | None = None,
    admt_governance_path: Path | None = None,
    governance_export_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(REGULATORY_EXPORT_TOOL),
        "--bundle-manifest",
        str(manifest_path),
        "--risk-metadata",
        str(risk_path),
        "--source-taxonomy",
        str(taxonomy_path),
        "--out-json",
        str(out_json),
        "--out-markdown",
        str(out_md),
        "--top-n",
        str(top_n),
    ]
    if risk_assessment_path is not None:
        command.extend(["--risk-assessment-report", str(risk_assessment_path)])
    if cybersecurity_audit_path is not None:
        command.extend(["--cybersecurity-audit-record", str(cybersecurity_audit_path)])
    if admt_governance_path is not None:
        command.extend(["--admt-governance", str(admt_governance_path)])
    if governance_export_path is not None:
        command.extend(["--governance-export", str(governance_export_path)])
    return _run_cli(command)


def test_regulatory_export_generates_bound_outputs(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)

    out_json = tmp_path / "regulatory_export.json"
    out_md = tmp_path / "regulatory_export.md"
    result = _run_regulatory_export(bundle_manifest, risk_path, taxonomy_path, out_json, out_md, top_n=5)

    assert result.returncode == 0, result.stdout + result.stderr
    assert out_json.exists()
    assert out_md.exists()

    export_payload = json.loads(out_json.read_text(encoding="utf-8"))
    manifest_payload = json.loads(bundle_manifest.read_text(encoding="utf-8"))

    assert export_payload["compliance_profile_id"] == "EU-AI-ACT-ART53-GPAI-V1"
    assert export_payload["bundle_binding"]["bundle_root_sha256"] == manifest_payload["bundle_root_sha256"]
    assert export_payload["training_data_summary"]["hash_manifest_summary"]["rows_total"] == 4
    assert export_payload["training_data_summary"]["source_taxonomy_summary"]["source_count_total"] == 3
    commands = export_payload["verification_commands"]
    assert "<BUNDLE_DIR>" in "\n".join(commands)
    assert str(tmp_path) not in "\n".join(commands)

    markdown = out_md.read_text(encoding="utf-8")
    assert "# Regulatory Export Summary" in markdown
    assert "## Verification Commands" in markdown


def test_regulatory_export_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)

    out_json_a = tmp_path / "export_a.json"
    out_md_a = tmp_path / "export_a.md"
    out_json_b = tmp_path / "export_b.json"
    out_md_b = tmp_path / "export_b.md"

    first = _run_regulatory_export(bundle_manifest, risk_path, taxonomy_path, out_json_a, out_md_a)
    second = _run_regulatory_export(bundle_manifest, risk_path, taxonomy_path, out_json_b, out_md_b)

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert out_json_a.read_bytes() == out_json_b.read_bytes()
    assert out_md_a.read_bytes() == out_md_b.read_bytes()


def test_regulatory_export_rejects_unknown_risk_metadata_fields(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)

    risk_payload = json.loads(risk_path.read_text(encoding="utf-8"))
    risk_payload["unexpected"] = "field"
    risk_path.write_text(json.dumps(risk_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_json = tmp_path / "export_fail.json"
    out_md = tmp_path / "export_fail.md"
    result = _run_regulatory_export(bundle_manifest, risk_path, taxonomy_path, out_json, out_md)

    assert result.returncode == 31
    assert "unexpected field(s): unexpected" in result.stdout


def test_regulatory_export_rejects_empty_web_scraped_top_domains(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)

    taxonomy_payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy_payload["sources"][0]["top_domains"] = []
    taxonomy_path.write_text(json.dumps(taxonomy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_json = tmp_path / "export_fail.json"
    out_md = tmp_path / "export_fail.md"
    result = _run_regulatory_export(bundle_manifest, risk_path, taxonomy_path, out_json, out_md)

    assert result.returncode == 31
    assert "top_domains must include at least one domain" in result.stdout


def test_regulatory_export_returns_write_failure_exit_code(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)

    out_json = tmp_path / "occupied_path"
    out_json.mkdir()
    out_md = tmp_path / "export_fail.md"
    result = _run_regulatory_export(bundle_manifest, risk_path, taxonomy_path, out_json, out_md)

    assert result.returncode == 32
    assert "Regulatory export write failed" in result.stdout


def test_regulatory_export_generates_governance_export(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    risk_assessment_path = tmp_path / "risk_assessment_report.json"
    cybersecurity_audit_path = tmp_path / "cybersecurity_audit_record.json"
    admt_path = tmp_path / "admt_governance.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)
    _write_risk_assessment_report(risk_assessment_path)
    _write_cybersecurity_audit_record(cybersecurity_audit_path)
    _write_admt_governance(admt_path)

    out_json = tmp_path / "regulatory_export.json"
    out_md = tmp_path / "regulatory_export.md"
    governance_export = tmp_path / "governance_export.json"
    result = _run_regulatory_export(
        bundle_manifest,
        risk_path,
        taxonomy_path,
        out_json,
        out_md,
        risk_assessment_path=risk_assessment_path,
        cybersecurity_audit_path=cybersecurity_audit_path,
        admt_governance_path=admt_path,
        governance_export_path=governance_export,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert governance_export.exists()

    governance_payload = json.loads(governance_export.read_text(encoding="utf-8"))
    manifest_payload = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    assert governance_payload["governance_profile_id"] == "CA-CPPA-CCPA-2026-ACCOUNTABILITY-V1"
    assert governance_payload["bundle_binding"]["bundle_root_sha256"] == manifest_payload["bundle_root_sha256"]
    assert "admt_governance_sha256" in governance_payload["governance_artifact_digests"]
    assert "Governance export written to" in result.stdout


def test_regulatory_export_rejects_governance_inputs_without_output(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    risk_assessment_path = tmp_path / "risk_assessment_report.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)
    _write_risk_assessment_report(risk_assessment_path)

    out_json = tmp_path / "regulatory_export.json"
    out_md = tmp_path / "regulatory_export.md"
    result = _run_regulatory_export(
        bundle_manifest,
        risk_path,
        taxonomy_path,
        out_json,
        out_md,
        risk_assessment_path=risk_assessment_path,
    )

    assert result.returncode == 31
    assert "governance input files require --governance-export" in result.stdout


def test_regulatory_export_rejects_admt_opt_out_verification_for_governance_export(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    risk_assessment_path = tmp_path / "risk_assessment_report.json"
    cybersecurity_audit_path = tmp_path / "cybersecurity_audit_record.json"
    admt_path = tmp_path / "admt_governance.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)
    _write_risk_assessment_report(risk_assessment_path)
    _write_cybersecurity_audit_record(cybersecurity_audit_path)
    _write_admt_governance(admt_path)

    admt_payload = json.loads(admt_path.read_text(encoding="utf-8"))
    admt_payload["request_verification_rules"]["opt_out_requires_verification"] = True
    admt_path.write_text(json.dumps(admt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_json = tmp_path / "regulatory_export.json"
    out_md = tmp_path / "regulatory_export.md"
    governance_export = tmp_path / "governance_export.json"
    result = _run_regulatory_export(
        bundle_manifest,
        risk_path,
        taxonomy_path,
        out_json,
        out_md,
        risk_assessment_path=risk_assessment_path,
        cybersecurity_audit_path=cybersecurity_audit_path,
        admt_governance_path=admt_path,
        governance_export_path=governance_export,
    )

    assert result.returncode == 31
    assert "opt_out_requires_verification must be false" in result.stdout


def test_regulatory_export_returns_write_failure_for_governance_export(tmp_path: Path) -> None:
    bundle_manifest = _write_bundle_with_root(tmp_path / "bundle")
    risk_path = tmp_path / "risk_metadata.json"
    taxonomy_path = tmp_path / "source_taxonomy.json"
    risk_assessment_path = tmp_path / "risk_assessment_report.json"
    cybersecurity_audit_path = tmp_path / "cybersecurity_audit_record.json"
    _write_risk_metadata(risk_path)
    _write_source_taxonomy(taxonomy_path)
    _write_risk_assessment_report(risk_assessment_path)
    _write_cybersecurity_audit_record(cybersecurity_audit_path)

    out_json = tmp_path / "regulatory_export.json"
    out_md = tmp_path / "regulatory_export.md"
    governance_export = tmp_path / "occupied_path"
    governance_export.mkdir()
    result = _run_regulatory_export(
        bundle_manifest,
        risk_path,
        taxonomy_path,
        out_json,
        out_md,
        risk_assessment_path=risk_assessment_path,
        cybersecurity_audit_path=cybersecurity_audit_path,
        governance_export_path=governance_export,
    )

    assert result.returncode == 32
    assert "Regulatory export write failed" in result.stdout
