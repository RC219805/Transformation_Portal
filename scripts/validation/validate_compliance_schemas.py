#!/usr/bin/env python3
"""Validate compliance schema files and canonical fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - exercised in CI
    print(f"❌ Missing dependency 'jsonschema': {exc}", file=sys.stderr)
    raise SystemExit(2) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "docs" / "compliance" / "schemas"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _make_validator(schema_name: str) -> jsonschema.Draft202012Validator:
    schema_path = SCHEMA_DIR / schema_name
    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _assert_valid(
    *,
    validator: jsonschema.Draft202012Validator,
    payload: dict[str, Any],
    context: str,
) -> None:
    errors = sorted(validator.iter_errors(payload), key=str)
    if errors:
        message = "; ".join(error.message for error in errors)
        raise ValueError(f"{context} expected valid payload, got: {message}")


def _assert_invalid(
    *,
    validator: jsonschema.Draft202012Validator,
    payload: dict[str, Any],
    context: str,
) -> None:
    errors = list(validator.iter_errors(payload))
    if not errors:
        raise ValueError(f"{context} expected invalid payload, but validation passed")


def main() -> int:
    risk_assessment_validator = _make_validator("risk_assessment_report.schema.json")
    cybersecurity_validator = _make_validator("cybersecurity_audit_record.schema.json")
    admt_validator = _make_validator("admt_governance.schema.json")

    risk_assessment_payload = {
        "schema_version": "1",
        "assessment_id": "ra-2026-q1",
        "regulatory_regime": "CPPA_CCPA_2026",
        "assessment_version": "1",
        "purpose_specificity": "Evaluate significant-risk processing posture for model training.",
        "categories_processed": ["web_scraped_text", "licensed_partner_media"],
        "sensitive_categories": ["biometric_templates"],
        "operational_elements": {
            "collection_methods": ["web_crawl", "contractual_transfer"],
            "retention_policy": "Retain for 5 years unless legal hold applies.",
            "recipients": ["model_safety_team", "compliance_team"],
            "population_scale_estimate": {
                "estimated_records": 1200000,
                "estimated_data_subjects": 75000,
            },
            "geographic_scope": ["US", "EU"],
        },
        "benefits": "Supports reproducible model safety analysis and documentation quality.",
        "negative_impacts": ["profiling risk", "sensitive data exposure risk"],
        "safeguards": ["data minimization", "access control", "content filtering"],
        "pets_used": ["sampling", "token-level redaction"],
        "review_approval": {
            "reviewer": "Jane Reviewer",
            "role": "Privacy Counsel",
            "date": "2026-02-24",
            "approval_status": "approved_with_conditions",
        },
        "next_review_due": "2026-08-24",
    }

    cybersecurity_payload = {
        "schema_version": "1",
        "audit_record_id": "cyber-audit-2026-annual",
        "regulatory_regime": "CPPA_CCPA_2026",
        "audit_standard": "NIST CSF 2.0",
        "auditor_name": "Independent Controls LLC",
        "auditor_independence_attested": True,
        "audit_period_start": "2026-01-01",
        "audit_period_end": "2026-12-31",
        "report_sha256": "a" * 64,
        "findings_summary": "No high-severity control failures detected.",
        "corrective_actions": ["improve key rotation cadence"],
        "threshold_tier": "tier_1",
        "certification_attestation": {
            "signer": "Alex Signer",
            "date": "2027-01-15",
            "penalty_of_perjury": True,
        },
        "retention": {
            "retention_years": 5,
            "retention_basis": "CPPA cybersecurity audit retention requirement",
            "records_location": "compliance/audits/2026",
        },
    }

    admt_payload = {
        "schema_version": "1",
        "governance_record_id": "admt-gov-2026-q1",
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

    _assert_valid(validator=risk_assessment_validator, payload=risk_assessment_payload, context="risk_assessment")
    _assert_valid(validator=cybersecurity_validator, payload=cybersecurity_payload, context="cybersecurity_audit")
    _assert_valid(validator=admt_validator, payload=admt_payload, context="admt_governance")

    risk_assessment_invalid = dict(risk_assessment_payload)
    risk_assessment_invalid["unexpected"] = True
    _assert_invalid(
        validator=risk_assessment_validator,
        payload=risk_assessment_invalid,
        context="risk_assessment.additional_properties",
    )

    cybersecurity_invalid = dict(cybersecurity_payload)
    cybersecurity_invalid["retention"] = dict(cybersecurity_payload["retention"])
    cybersecurity_invalid["retention"]["retention_years"] = 4
    _assert_invalid(
        validator=cybersecurity_validator,
        payload=cybersecurity_invalid,
        context="cybersecurity.retention_years",
    )

    admt_invalid = dict(admt_payload)
    admt_invalid["request_verification_rules"] = dict(admt_payload["request_verification_rules"])
    admt_invalid["request_verification_rules"]["opt_out_requires_verification"] = True
    _assert_invalid(
        validator=admt_validator,
        payload=admt_invalid,
        context="admt.opt_out_requires_verification",
    )

    print("✅ Compliance schemas validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
