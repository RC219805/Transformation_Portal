"""Tests for APEX redacted evidence fixture policy documentation."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "docs" / "validation" / "APEX_REDACTED_EVIDENCE_FIXTURE_POLICY.md"
RUNBOOK_PATH = REPO_ROOT / "docs" / "validation" / "APEX_REAL_CANONICAL_EVIDENCE_RUNBOOK.md"
REAL_ESTATE_README_PATH = REPO_ROOT / "evalsets" / "apex_real_estate_v1" / "README.md"
POLICY_LINK = "docs/validation/APEX_REDACTED_EVIDENCE_FIXTURE_POLICY.md"


def _policy_text() -> str:
    return POLICY_PATH.read_text(encoding="utf-8")


def test_apex_redacted_evidence_fixture_policy_exists():
    assert POLICY_PATH.exists()


def test_apex_redacted_evidence_fixture_policy_locks_non_promotional_contract():
    text = _policy_text()
    normalized = text.lower()
    collapsed = " ".join(normalized.split())

    assert "synthetic_data=true" in text
    assert "schema regression" in normalized
    assert "non-promotional" in normalized
    assert "never real apex quality evidence" in normalized
    assert "must never satisfy promotion eligibility" in collapsed


def test_apex_redacted_evidence_fixture_policy_prohibits_real_artifacts():
    text = _policy_text()
    prohibited_terms = {
        "TIFF/TIF",
        "RAW/DNG/CR2/CR3/NEF/ARW/RAF/ORF/RW2",
        "ICC/profile binaries",
        "candidate outputs",
        "generated `output/` artifacts",
        "private property imagery",
    }

    for term in prohibited_terms:
        assert term in text


def test_apex_redacted_evidence_fixture_policy_has_future_fixture_pr_checklist():
    normalized = _policy_text().lower()

    for phrase in (
        "fixture purpose",
        "fixture size",
        "fixture provenance",
        "synthetic or redaction method",
        "why docs-only coverage is insufficient",
        "cannot expose private real-estate imagery",
        "cannot satisfy promotion eligibility",
        "promotion eligibility remains blocked",
    ):
        assert phrase in normalized


def test_real_canonical_docs_link_to_redacted_fixture_policy():
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
    readme = REAL_ESTATE_README_PATH.read_text(encoding="utf-8")

    assert POLICY_LINK in runbook
    assert POLICY_LINK in readme
