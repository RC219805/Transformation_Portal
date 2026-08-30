"""Unit tests for ``assess_run_card_release``.

The release assessor orchestrates run-card integrity, version, and three
attestation surfaces (native detached, DSSE in-toto, Sigstore bundle)
into a single normalized PASS/FAIL report. These tests pin down its
control flow by mocking ``verify_run_card_integrity`` and the
attestation helpers, so the assessor's policy logic is covered without
requiring fully signed fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.lux_depth_v3.validators import release_assessment
from transformation_portal.lux_depth_v3.validators.release_assessment import (
    assess_run_card_release,
)

pytestmark = [pytest.mark.unit]


def _write_run_card(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "run_card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _check(report: dict, name: str) -> dict:
    matches = [c for c in report["checks"] if c["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} check, got {len(matches)}"
    return matches[0]


class TestIntegrityShortCircuit:
    def test_integrity_failure_skips_downstream_checks(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        with patch.object(
            release_assessment,
            "verify_run_card_integrity",
            return_value=["artifact hash mismatch", "missing manifest"],
        ):
            report = assess_run_card_release(run_card_path=run_card)
        assert report["status"] == "FAIL"
        assert {c["name"] for c in report["checks"]} == {"run_card_integrity"}
        integrity = _check(report, "run_card_integrity")
        assert integrity["status"] == "FAIL"
        assert integrity["details"] == ["artifact hash mismatch", "missing manifest"]


class TestVersionCheck:
    def test_v2_required_v1_payload_fails(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v1"})
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(run_card_path=run_card, require_v2=True)
        assert report["status"] == "FAIL"
        version_check = _check(report, "run_card_version")
        assert version_check["status"] == "FAIL"
        assert version_check["details"] == {"detected": "v1", "required": "v2"}

    def test_v2_required_v2_payload_passes(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(run_card_path=run_card, require_v2=True)
        assert _check(report, "run_card_version")["status"] == "PASS"
        assert report["run_card_version"] == "v2"

    def test_v1_payload_with_v2_not_required_passes(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v1"})
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(run_card_path=run_card, require_v2=False)
        version_check = _check(report, "run_card_version")
        assert version_check["status"] == "PASS"
        assert version_check["details"] == {"detected": "v1", "required": "any"}


class TestNativeAttestation:
    def test_required_but_missing_yields_failure(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_native_attestation=True,
            )
        native = _check(report, "native_attestation")
        assert native["status"] == "FAIL"
        assert any("missing native detached attestation" in e for e in native["details"]["errors"])
        assert report["status"] == "FAIL"

    def test_skipped_when_neither_required_nor_present(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(run_card_path=run_card)
        assert "native_attestation" not in {c["name"] for c in report["checks"]}

    def test_present_but_malformed_yields_failure(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        # Sidecar lives at <run_card>.attestation.native.json
        native_path = run_card.with_suffix(".attestation.native.json")
        native_path.write_text(json.dumps({"signature": {}}), encoding="utf-8")
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(run_card_path=run_card)
        native = _check(report, "native_attestation")
        assert native["status"] == "FAIL"
        assert native["details"]["errors"], "expected at least one error from validator chain"

    def test_full_chain_passes_when_helpers_succeed(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        native_path = run_card.with_suffix(".attestation.native.json")
        native_path.write_text(
            json.dumps(
                {
                    "signature": {
                        "algorithm": "openpgp-clearsign",
                        "key_id": "A" * 40,
                        "signature": "-----BEGIN PGP-----\n...",
                    }
                }
            ),
            encoding="utf-8",
        )
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(release_assessment, "validate_run_card_detached_attestation_surface"),
            patch.object(release_assessment, "bind_run_card_detached_attestation"),
            patch.object(release_assessment, "verify_run_card_attestation_self_hash"),
            patch.object(
                release_assessment,
                "canonical_run_card_attestation_preimage_bytes",
                return_value=b"preimage",
            ),
            patch.object(release_assessment, "gpg_verify_clearsign") as mock_gpg,
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_native_attestation=True,
                verify_gpg=True,
            )
        mock_gpg.assert_called_once_with(
            "-----BEGIN PGP-----\n...",
            expected_payload=b"preimage",
            key_id="A" * 40,
        )
        native = _check(report, "native_attestation")
        assert native["status"] == "PASS"
        assert native["details"]["errors"] == []

    def test_chain_passes_without_gpg_skips_clearsign(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        native_path = run_card.with_suffix(".attestation.native.json")
        native_path.write_text(json.dumps({"signature": {}}), encoding="utf-8")
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(release_assessment, "validate_run_card_detached_attestation_surface"),
            patch.object(release_assessment, "bind_run_card_detached_attestation"),
            patch.object(release_assessment, "verify_run_card_attestation_self_hash"),
            patch.object(release_assessment, "gpg_verify_clearsign") as mock_gpg,
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_native_attestation=True,
                verify_gpg=False,
            )
        assert mock_gpg.call_count == 0
        assert _check(report, "native_attestation")["status"] == "PASS"


class TestDsseAttestation:
    def test_required_but_missing_yields_failure(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_dsse_attestation=True,
            )
        dsse = _check(report, "dsse_attestation")
        assert dsse["status"] == "FAIL"
        assert any("missing DSSE attestation" in e for e in dsse["details"]["errors"])

    def test_full_chain_passes_when_helpers_succeed(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        run_card.with_suffix(".attestation.dsse.json").write_text("{}", encoding="utf-8")
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(release_assessment, "decode_run_card_statement_from_envelope", return_value={}),
            patch.object(release_assessment, "validate_run_card_statement_binding"),
            patch.object(release_assessment, "decode_dsse_signature_bytes", return_value=b"sig"),
            patch.object(release_assessment, "decode_dsse_payload", return_value=b"payload"),
            patch.object(release_assessment, "pre_auth_encode", return_value=b"pae"),
            patch.object(release_assessment, "gpg_verify_detached_signature_bytes") as mock_gpg,
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_dsse_attestation=True,
                verify_gpg=True,
            )
        assert mock_gpg.call_count == 1
        dsse = _check(report, "dsse_attestation")
        assert dsse["status"] == "PASS"
        assert dsse["details"]["errors"] == []

    def test_chain_passes_without_gpg_skips_detached_verify(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        run_card.with_suffix(".attestation.dsse.json").write_text("{}", encoding="utf-8")
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(release_assessment, "decode_run_card_statement_from_envelope", return_value={}),
            patch.object(release_assessment, "validate_run_card_statement_binding"),
            patch.object(release_assessment, "gpg_verify_detached_signature_bytes") as mock_gpg,
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_dsse_attestation=True,
                verify_gpg=False,
            )
        assert mock_gpg.call_count == 0
        assert _check(report, "dsse_attestation")["status"] == "PASS"


class TestSigstoreBundle:
    def test_bundle_required_without_dsse_yields_failure(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_sigstore_bundle=True,
            )
        bundle = _check(report, "sigstore_bundle")
        assert bundle["status"] == "FAIL"
        assert any("cannot verify Sigstore bundle without DSSE attestation" in e for e in bundle["details"]["errors"])

    def test_bundle_required_with_dsse_but_missing_bundle_yields_failure(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        run_card.with_suffix(".attestation.dsse.json").write_text("{}", encoding="utf-8")
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_sigstore_bundle=True,
            )
        bundle = _check(report, "sigstore_bundle")
        assert bundle["status"] == "FAIL"
        assert any("missing Sigstore bundle" in e for e in bundle["details"]["errors"])

    def test_bundle_with_rekor_inclusion_passes_cosign_mocked(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        run_card.with_suffix(".attestation.dsse.json").write_text("{}", encoding="utf-8")
        bundle_path = run_card.with_suffix(".attestation.dsse.sigstore.bundle.json")
        bundle_path.write_text(
            json.dumps(
                {
                    "verificationMaterial": {
                        "tlogEntries": [{"logIndex": "1"}],
                    }
                }
            ),
            encoding="utf-8",
        )
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(release_assessment, "cosign_verify_blob") as mock_cosign,
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_sigstore_bundle=True,
                require_rekor_inclusion=True,
            )
        assert mock_cosign.call_count == 1
        bundle = _check(report, "sigstore_bundle")
        assert bundle["status"] == "PASS"
        assert bundle["details"]["rekor_inclusion"] is True
        assert bundle["details"]["errors"] == []

    def test_bundle_without_rekor_inclusion_when_required_fails(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        run_card.with_suffix(".attestation.dsse.json").write_text("{}", encoding="utf-8")
        bundle_path = run_card.with_suffix(".attestation.dsse.sigstore.bundle.json")
        bundle_path.write_text(
            json.dumps({"verificationMaterial": {"tlogEntries": []}}),
            encoding="utf-8",
        )
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(release_assessment, "cosign_verify_blob"),
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_sigstore_bundle=True,
                require_rekor_inclusion=True,
            )
        bundle = _check(report, "sigstore_bundle")
        assert bundle["status"] == "FAIL"
        assert bundle["details"]["rekor_inclusion"] is False
        assert any("does not record Rekor inclusion evidence" in e for e in bundle["details"]["errors"])

    def test_bundle_without_verification_material_records_no_rekor(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        run_card.with_suffix(".attestation.dsse.json").write_text("{}", encoding="utf-8")
        bundle_path = run_card.with_suffix(".attestation.dsse.sigstore.bundle.json")
        bundle_path.write_text(json.dumps({}), encoding="utf-8")
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(release_assessment, "cosign_verify_blob"),
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_sigstore_bundle=True,
            )
        bundle = _check(report, "sigstore_bundle")
        assert bundle["details"]["rekor_inclusion"] is False
        # Without require_rekor_inclusion, missing Rekor evidence is not an error.
        assert bundle["status"] == "PASS"

    def test_cosign_failure_is_normalized_into_errors(self, tmp_path):
        run_card = _write_run_card(tmp_path, {"run_card_version": "v2"})
        run_card.with_suffix(".attestation.dsse.json").write_text("{}", encoding="utf-8")
        bundle_path = run_card.with_suffix(".attestation.dsse.sigstore.bundle.json")
        bundle_path.write_text(json.dumps({"verificationMaterial": {}}), encoding="utf-8")
        with (
            patch.object(release_assessment, "verify_run_card_integrity", return_value=[]),
            patch.object(
                release_assessment,
                "cosign_verify_blob",
                side_effect=RuntimeError("cosign signature mismatch"),
            ),
        ):
            report = assess_run_card_release(
                run_card_path=run_card,
                require_sigstore_bundle=True,
            )
        bundle = _check(report, "sigstore_bundle")
        assert bundle["status"] == "FAIL"
        assert any("cosign signature mismatch" in e for e in bundle["details"]["errors"])


class TestRunCardLoadingErrors:
    def test_invalid_json_in_run_card_raises_value_error(self, tmp_path):
        run_card = tmp_path / "run_card.json"
        run_card.write_text("{not valid json", encoding="utf-8")
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            with pytest.raises(ValueError, match="unable to load run card"):
                assess_run_card_release(run_card_path=run_card)

    def test_run_card_must_be_object(self, tmp_path):
        run_card = tmp_path / "run_card.json"
        run_card.write_text("[1, 2, 3]", encoding="utf-8")
        with patch.object(release_assessment, "verify_run_card_integrity", return_value=[]):
            with pytest.raises(ValueError, match="must be a JSON object"):
                assess_run_card_release(run_card_path=run_card)
