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


@pytest.mark.parametrize("authorized_keys", [None, set()], ids=["default", "empty-set"])
def test_task_verifier_without_authorized_keys_rejects_any_signer(authorized_keys: set[str] | None) -> None:
    """A valid signature only proves the task matches the key it carries."""
    attacker_key, _ = generate_ed25519_keypair()
    forged = TaskSigner(CertificateSigner(attacker_key)).sign({"node_cls": "ShellNode", "inputs": {"cmd": "id"}})
    verifier = TaskVerifier(authorized_keys=authorized_keys)

    assert verifier.verify(forged) is False
    assert verifier.get_stats() == {"verified": 0, "rejected": 1, "authorized_keys": 0, "cached_nonces": 0}


def test_task_verifier_accepts_a_signer_only_after_its_key_is_authorized() -> None:
    private_key, _ = generate_ed25519_keypair()
    signed = TaskSigner(CertificateSigner(private_key)).sign({"node_cls": "ProcessNode"})
    verifier = TaskVerifier()

    assert verifier.verify(signed) is False

    verifier.add_authorized_key(signed.public_key)

    assert verifier.verify(signed) is True


@pytest.mark.parametrize("reserved_key", ["_ts", "_nonce", "_metadata"])
def test_task_signer_rejects_payload_using_a_reserved_envelope_key(reserved_key: str) -> None:
    """A payload key named like an envelope field would not be covered by the signature."""
    private_key, _ = generate_ed25519_keypair()
    signer = TaskSigner(CertificateSigner(private_key))

    with pytest.raises(TaskSigningError, match="reserved signing keys"):
        signer.sign({"node_cls": "ProcessNode", reserved_key: "caller-value"})

    assert signer.sign_count == 0


def test_task_verifier_rejects_metadata_moved_into_the_payload() -> None:
    """Envelope fields cannot change role without invalidating the task."""
    private_key, _ = generate_ed25519_keypair()
    signer = TaskSigner(CertificateSigner(private_key))
    signed = signer.sign({"node_cls": "ProcessNode"}, metadata={"tenant": "alice", "priority": "low"})
    tampered = SignedTask(
        payload={"node_cls": "ProcessNode", "_metadata": {"tenant": "alice", "priority": "low"}},
        signature=signed.signature,
        public_key=signed.public_key,
        timestamp=signed.timestamp,
        nonce=signed.nonce,
        metadata=None,
    )
    verifier = TaskVerifier(authorized_keys={signed.public_key})

    assert verifier.verify(tampered) is False
    assert verifier.verify(signed) is True


class _Clock:
    """Stand-in for the ``time`` module as task_signing uses it."""

    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def time(self) -> float:
        return self.now


def test_replay_is_rejected_after_more_tasks_than_the_nonce_cache_holds() -> None:
    """A full nonce cache refuses new tasks; it never forgets a nonce that can still be replayed."""
    private_key, _ = generate_ed25519_keypair()
    signer = TaskSigner(CertificateSigner(private_key))
    tasks = [signer.sign({"n": index}) for index in range(8)]
    verifier = TaskVerifier(authorized_keys={tasks[0].public_key}, max_nonce_cache=3)

    first_pass = [verifier.verify(task) for task in tasks]
    replay_pass = [verifier.verify(task) for task in tasks]

    assert first_pass == [True, True, True, False, False, False, False, False]
    assert replay_pass == [False] * 8
    assert verifier.get_stats()["cached_nonces"] == 3


def test_nonces_of_expired_tasks_make_room_for_new_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(task_signing, "time", clock)
    private_key, _ = generate_ed25519_keypair()
    signer = TaskSigner(CertificateSigner(private_key))
    old_tasks = [signer.sign({"n": index}) for index in range(3)]
    verifier = TaskVerifier(authorized_keys={old_tasks[0].public_key}, max_age_seconds=300.0, max_nonce_cache=2)

    assert [verifier.verify(task) for task in old_tasks] == [True, True, False]

    clock.now += 301.0
    fresh = signer.sign({"n": "fresh"})

    assert verifier.verify(fresh) is True
    assert verifier.verify(fresh) is False  # replay of the fresh task
    assert verifier.verify(old_tasks[0]) is False  # too old now, so its nonce is no longer needed
    assert verifier.get_stats()["cached_nonces"] == 1


def test_waived_nonce_check_is_not_blocked_by_a_full_nonce_cache() -> None:
    private_key, _ = generate_ed25519_keypair()
    signer = TaskSigner(CertificateSigner(private_key))
    tasks = [signer.sign({"n": index}) for index in range(3)]
    verifier = TaskVerifier(authorized_keys={tasks[0].public_key}, max_nonce_cache=2)
    assert [verifier.verify(task) for task in tasks[:2]] == [True, True]

    assert verifier.verify(tasks[2], check_nonce=False) is True
    assert verifier.verify(tasks[0], check_nonce=False) is True
    assert verifier.verify(tasks[0]) is False


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
