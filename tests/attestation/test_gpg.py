"""Security regressions for exact OpenPGP clear-sign verification."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from transformation_portal.attestation.gpg import gpg_clearsign_bytes, gpg_verify_clearsign

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAKE_GPG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "attestation" / "fake_gpg.py"
PRIMARY_FINGERPRINT = "A" * 40


def _install_fake_gpg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gpg_path = bin_dir / "gpg"
    gpg_path.write_bytes(FAKE_GPG_PATH.read_bytes())
    gpg_path.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]))


def test_gpg_clearsign_roundtrip_binds_exact_payload_and_primary_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gpg(tmp_path, monkeypatch)
    payload = b'{"schema":"tp.attestation.detached.v1.preimage"}'
    signature = gpg_clearsign_bytes(payload, key_id=PRIMARY_FINGERPRINT.lower())

    assert (
        gpg_verify_clearsign(
            signature,
            expected_payload=payload,
            key_id=PRIMARY_FINGERPRINT.lower(),
        )
        is None
    )


def test_gpg_verify_accepts_signing_subkey_when_reported_primary_matches_recorded_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gpg(tmp_path, monkeypatch)
    monkeypatch.setenv("TP_FAKE_GPG_SIGNING_FINGERPRINT", "B" * 40)
    payload = b'{"schema":"expected"}'
    signature = gpg_clearsign_bytes(payload, key_id=PRIMARY_FINGERPRINT)

    assert (
        gpg_verify_clearsign(
            signature,
            expected_payload=payload,
            key_id=PRIMARY_FINGERPRINT,
        )
        is None
    )


def test_gpg_verify_rejects_unrelated_valid_clearsign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gpg(tmp_path, monkeypatch)
    signature = gpg_clearsign_bytes(b'{"schema":"unrelated"}', key_id=PRIMARY_FINGERPRINT)

    with pytest.raises(ValueError, match="does not match the expected canonical preimage bytes"):
        gpg_verify_clearsign(
            signature,
            expected_payload=b'{"schema":"expected"}',
            key_id=PRIMARY_FINGERPRINT,
        )


def test_gpg_verify_rejects_altered_extracted_preimage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gpg(tmp_path, monkeypatch)
    payload = b'{"schema":"expected"}'
    signature = gpg_clearsign_bytes(payload, key_id=PRIMARY_FINGERPRINT)
    monkeypatch.setenv(
        "TP_FAKE_GPG_VERIFIED_PAYLOAD_B64",
        base64.b64encode(payload + b" ").decode("ascii"),
    )

    with pytest.raises(ValueError, match="does not match the expected canonical preimage bytes"):
        gpg_verify_clearsign(signature, expected_payload=payload, key_id=PRIMARY_FINGERPRINT)


def test_gpg_verify_rejects_recorded_key_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gpg(tmp_path, monkeypatch)
    payload = b'{"schema":"expected"}'
    signature = gpg_clearsign_bytes(payload, key_id=PRIMARY_FINGERPRINT)
    monkeypatch.setenv("TP_FAKE_GPG_RESOLVED_FINGERPRINT", "B" * 40)

    with pytest.raises(ValueError, match="primary fingerprint does not match recorded key_id"):
        gpg_verify_clearsign(signature, expected_payload=payload, key_id="recorded-key")


@pytest.mark.parametrize("status_mode, expected_count", [("missing", 0), ("ambiguous", 2)])
def test_gpg_verify_requires_exactly_one_validsig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_mode: str,
    expected_count: int,
) -> None:
    _install_fake_gpg(tmp_path, monkeypatch)
    payload = b'{"schema":"expected"}'
    signature = gpg_clearsign_bytes(payload, key_id=PRIMARY_FINGERPRINT)
    monkeypatch.setenv("TP_FAKE_GPG_STATUS_MODE", status_mode)

    with pytest.raises(ValueError, match=rf"exactly one VALIDSIG record; found {expected_count}"):
        gpg_verify_clearsign(signature, expected_payload=payload, key_id=PRIMARY_FINGERPRINT)


@pytest.mark.parametrize("resolve_mode", ["missing", "ambiguous", "error"])
def test_gpg_verify_rejects_non_resolvable_recorded_key_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolve_mode: str,
) -> None:
    _install_fake_gpg(tmp_path, monkeypatch)
    payload = b'{"schema":"expected"}'
    signature = gpg_clearsign_bytes(payload, key_id=PRIMARY_FINGERPRINT)
    monkeypatch.setenv("TP_FAKE_GPG_RESOLVE_MODE", resolve_mode)

    with pytest.raises(ValueError, match="recorded key_id"):
        gpg_verify_clearsign(signature, expected_payload=payload, key_id="logical-label")
