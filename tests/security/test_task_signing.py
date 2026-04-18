from __future__ import annotations

import time

import pytest

from transformation_portal.core.security import task_signing
from transformation_portal.core.security.signing import CertificateSigner, generate_ed25519_keypair
from transformation_portal.core.security.task_signing import SignedTask, TaskSigner, TaskSigningError, TaskVerifier

pytestmark = [pytest.mark.unit, pytest.mark.security]


def test_task_signing_round_trip_verifies() -> None:
    private_key, _ = generate_ed25519_keypair()
    signer = TaskSigner(CertificateSigner(private_key))

    signed = signer.sign(
        {
            "node_cls": "ProcessNode",
            "inputs": {"artifact": "sha256:abc123"},
        },
        metadata={"queue": "default"},
    )

    verifier = TaskVerifier(authorized_keys={signed.public_key})

    assert verifier.verify(signed) is True
    assert verifier.get_stats()["verified"] == 1


def test_task_signer_requires_crypto_backed_signer() -> None:
    class BrokenSigner:
        pass

    with pytest.raises(TaskSigningError, match="CertificateSigner"):
        TaskSigner(BrokenSigner()).sign({"node_cls": "Broken"})


def test_task_verifier_rejects_placeholder_signature() -> None:
    placeholder = SignedTask(
        payload={"node_cls": "Unsigned"},
        signature="no_crypto",
        public_key="no_crypto",
        timestamp=time.time(),
        nonce="abc123",
    )

    assert TaskVerifier().verify(placeholder) is False


def test_task_verifier_fails_closed_when_crypto_verification_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    private_key, _ = generate_ed25519_keypair()
    signer = TaskSigner(CertificateSigner(private_key))
    signed = signer.sign({"node_cls": "ProcessNode"})

    def _raise_missing_crypto():
        raise TaskSigningError("cryptography library required for task signature verification")

    monkeypatch.setattr(task_signing, "_load_ed25519_public_key_class", _raise_missing_crypto)

    verifier = TaskVerifier(authorized_keys={signed.public_key})

    assert verifier.verify(signed) is False
