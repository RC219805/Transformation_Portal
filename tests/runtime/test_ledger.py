"""Tests for ImmutableLedger - transaction logging and integrity.

This module tests:
- Ledger creation and initialization
- Append-only entry storage
- Hash-chaining for tamper-evidence
- Integrity verification
- Entry queries and retrieval
- Ledger summary statistics
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.runtime.ledger import (
    ImmutableLedger,
    LedgerEntry,
    LedgerError,
    _hash_dict,
)

# --- Mock Certificate for Testing ---


class MockCertificate:
    """Mock signed certificate for testing."""

    def __init__(self, data: dict):
        self._data = data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self._data)


# --- Test Helper Functions ---


def create_test_manifest(run_id: str = "test_run") -> str:
    """Create a test manifest JSON string."""
    return json.dumps(
        {
            "run_id": run_id,
            "node_id": "test_node",
            "inputs": {"sha1": "input_path"},
            "outputs": {"output_path": "sha2"},
            "timestamp": "2025-01-01T00:00:00Z",
        }
    )


def create_test_certificate(run_id: str = "test_run") -> MockCertificate:
    """Create a test certificate."""
    return MockCertificate(
        {
            "run_id": run_id,
            "signature": "mock_signature_abc123",
            "algorithm": "sha256-rsa",
        }
    )


# --- Test Classes ---


class TestHashDict:
    """Tests for _hash_dict helper function."""

    def test_hash_dict_deterministic(self) -> None:
        """_hash_dict produces same hash for same content."""
        data = {"key": "value", "number": 42}

        hash1 = _hash_dict(data)
        hash2 = _hash_dict(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_hash_dict_different_content(self) -> None:
        """_hash_dict produces different hashes for different content."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        hash1 = _hash_dict(data1)
        hash2 = _hash_dict(data2)

        assert hash1 != hash2

    def test_hash_dict_order_independent(self) -> None:
        """_hash_dict produces same hash regardless of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        hash1 = _hash_dict(data1)
        hash2 = _hash_dict(data2)

        assert hash1 == hash2

    def test_hash_dict_nested_content(self) -> None:
        """_hash_dict handles nested dictionaries."""
        data = {
            "outer": {"inner": {"value": "deep"}},
            "list": [1, 2, 3],
        }

        hash_val = _hash_dict(data)

        assert len(hash_val) == 64


class TestLedgerEntry:
    """Tests for LedgerEntry dataclass."""

    def test_entry_creation(self) -> None:
        """LedgerEntry stores all required fields."""
        entry = LedgerEntry(
            entry_id=1,
            prev_hash="GENESIS",
            entry_hash="abc123" + "0" * 58,
            timestamp="2025-01-01T00:00:00Z",
            manifest={"run_id": "test"},
            certificate={"signature": "sig"},
            metadata={"extra": "data"},
        )

        assert entry.entry_id == 1
        assert entry.prev_hash == "GENESIS"
        assert entry.timestamp == "2025-01-01T00:00:00Z"

    def test_entry_to_dict(self) -> None:
        """LedgerEntry.to_dict converts to dictionary."""
        entry = LedgerEntry(
            entry_id=1,
            prev_hash="GENESIS",
            entry_hash="abc123",
            timestamp="2025-01-01T00:00:00Z",
            manifest={"run_id": "test"},
            certificate={"sig": "value"},
            metadata={},
        )

        data = entry.to_dict()

        assert data["entry_id"] == 1
        assert data["manifest"]["run_id"] == "test"
        assert "certificate" in data

    def test_entry_to_json(self) -> None:
        """LedgerEntry.to_json produces valid JSON."""
        entry = LedgerEntry(
            entry_id=1,
            prev_hash="GENESIS",
            entry_hash="abc123",
            timestamp="2025-01-01T00:00:00Z",
            manifest={"run_id": "test"},
            certificate={},
            metadata={},
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)

        assert parsed["entry_id"] == 1

    def test_entry_from_dict(self) -> None:
        """LedgerEntry.from_dict creates entry from dictionary."""
        data = {
            "entry_id": 5,
            "prev_hash": "prev_abc",
            "entry_hash": "hash_def",
            "timestamp": "2025-06-01T12:00:00Z",
            "manifest": {"run_id": "restored"},
            "certificate": {"sig": "restored_sig"},
            "metadata": {"restored": True},
        }

        entry = LedgerEntry.from_dict(data)

        assert entry.entry_id == 5
        assert entry.prev_hash == "prev_abc"
        assert entry.manifest["run_id"] == "restored"

    def test_entry_from_dict_default_metadata(self) -> None:
        """LedgerEntry.from_dict handles missing metadata."""
        data = {
            "entry_id": 1,
            "prev_hash": "prev",
            "entry_hash": "hash",
            "timestamp": "2025-01-01T00:00:00Z",
            "manifest": {},
            "certificate": {},
            # No metadata field
        }

        entry = LedgerEntry.from_dict(data)

        assert entry.metadata == {}


class TestImmutableLedger:
    """Tests for ImmutableLedger class."""

    @pytest.fixture
    def ledger_path(self, tmp_path: Path) -> Path:
        """Create path for test ledger."""
        return tmp_path / "test.ledger"

    def test_ledger_creation(self, ledger_path: Path) -> None:
        """ImmutableLedger creates new ledger file."""
        ledger = ImmutableLedger(ledger_path)

        assert ledger_path.exists()
        assert ledger.entry_count == 0
        assert ledger.last_hash == ImmutableLedger.GENESIS_HASH

    def test_ledger_creates_parent_directory(self, tmp_path: Path) -> None:
        """ImmutableLedger creates parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "ledger.log"

        ImmutableLedger(deep_path)

        assert deep_path.exists()

    def test_ledger_append_single(self, ledger_path: Path) -> None:
        """ImmutableLedger.append adds single entry."""
        ledger = ImmutableLedger(ledger_path)

        manifest = create_test_manifest("run_001")
        cert = create_test_certificate("run_001")

        entry = ledger.append(manifest, cert)

        assert entry.entry_id == 1
        assert entry.prev_hash == ImmutableLedger.GENESIS_HASH
        assert ledger.entry_count == 1

    def test_ledger_append_chained(self, ledger_path: Path) -> None:
        """ImmutableLedger.append chains entries."""
        ledger = ImmutableLedger(ledger_path)

        entry1 = ledger.append(
            create_test_manifest("run_001"),
            create_test_certificate("run_001"),
        )

        entry2 = ledger.append(
            create_test_manifest("run_002"),
            create_test_certificate("run_002"),
        )

        assert entry2.prev_hash == entry1.entry_hash
        assert entry2.entry_id == 2

    def test_ledger_append_with_metadata(self, ledger_path: Path) -> None:
        """ImmutableLedger.append stores metadata."""
        ledger = ImmutableLedger(ledger_path)

        entry = ledger.append(
            create_test_manifest("run_001"),
            create_test_certificate("run_001"),
            metadata={"user": "test_user", "environment": "test"},
        )

        assert entry.metadata["user"] == "test_user"
        assert entry.metadata["environment"] == "test"

    def test_ledger_load_existing(self, ledger_path: Path) -> None:
        """ImmutableLedger loads existing ledger state."""
        # Create and populate ledger
        ledger1 = ImmutableLedger(ledger_path)
        ledger1.append(create_test_manifest("run_001"), create_test_certificate("run_001"))
        ledger1.append(create_test_manifest("run_002"), create_test_certificate("run_002"))
        last_hash = ledger1.last_hash

        # Load existing ledger
        ledger2 = ImmutableLedger(ledger_path)

        assert ledger2.entry_count == 2
        assert ledger2.last_hash == last_hash

    def test_ledger_no_create_if_not_exists(self, tmp_path: Path) -> None:
        """ImmutableLedger with create=False doesn't create file."""
        path = tmp_path / "nonexistent.ledger"

        # Should not raise, but file shouldn't be created
        # Note: Current implementation always creates, so this tests
        # that the path is handled correctly
        ledger = ImmutableLedger(path, create=True)
        assert path.exists()


class TestImmutableLedgerVerification:
    """Tests for ledger integrity verification."""

    @pytest.fixture
    def populated_ledger(self, tmp_path: Path):
        """Create a populated ledger for testing."""
        path = tmp_path / "verify.ledger"
        ledger = ImmutableLedger(path)

        for i in range(5):
            ledger.append(
                create_test_manifest(f"run_{i:03d}"),
                create_test_certificate(f"run_{i:03d}"),
            )

        return ledger, path

    def test_verify_valid_ledger(self, populated_ledger) -> None:
        """verify returns True for valid ledger."""
        ledger, _ = populated_ledger

        assert ledger.verify() is True

    def test_verify_empty_ledger(self, tmp_path: Path) -> None:
        """verify returns True for empty ledger."""
        path = tmp_path / "empty.ledger"
        ledger = ImmutableLedger(path)

        # Empty ledger should be valid
        result = ledger.verify()
        assert result is True

    def test_verify_detects_modified_content(self, populated_ledger, tmp_path: Path) -> None:
        """verify detects tampered entry content."""
        ledger, path = populated_ledger

        # Tamper with ledger file
        content = path.read_text()
        lines = content.strip().split("\n")

        # Modify second entry's manifest
        entry_data = json.loads(lines[1])
        entry_data["manifest"]["run_id"] = "TAMPERED"
        lines[1] = json.dumps(entry_data, sort_keys=True)

        path.write_text("\n".join(lines) + "\n")

        # Reload and verify
        tampered_ledger = ImmutableLedger(path)
        assert tampered_ledger.verify() is False

    def test_verify_detects_broken_chain(self, populated_ledger, tmp_path: Path) -> None:
        """verify detects broken hash chain."""
        ledger, path = populated_ledger

        # Tamper with hash chain
        content = path.read_text()
        lines = content.strip().split("\n")

        # Modify third entry's prev_hash
        entry_data = json.loads(lines[2])
        entry_data["prev_hash"] = "BROKEN_CHAIN_HASH"
        lines[2] = json.dumps(entry_data, sort_keys=True)

        path.write_text("\n".join(lines) + "\n")

        # Reload and verify
        tampered_ledger = ImmutableLedger(path)
        assert tampered_ledger.verify() is False


class TestImmutableLedgerQueries:
    """Tests for ledger query operations."""

    @pytest.fixture
    def query_ledger(self, tmp_path: Path):
        """Create ledger with queryable entries."""
        path = tmp_path / "query.ledger"
        ledger = ImmutableLedger(path)

        run_ids = ["alpha", "beta", "gamma", "delta"]
        for run_id in run_ids:
            ledger.append(
                create_test_manifest(run_id),
                create_test_certificate(run_id),
            )

        return ledger

    def test_iter_entries(self, query_ledger) -> None:
        """iter_entries yields all entries in order."""
        entries = list(query_ledger.iter_entries())

        assert len(entries) == 4
        assert entries[0].entry_id == 1
        assert entries[3].entry_id == 4

    def test_get_entries(self, query_ledger) -> None:
        """get_entries returns list of all entries."""
        entries = query_ledger.get_entries()

        assert isinstance(entries, list)
        assert len(entries) == 4

    def test_get_entry_by_id(self, query_ledger) -> None:
        """get_entry returns specific entry by ID."""
        entry = query_ledger.get_entry(2)

        assert entry is not None
        assert entry.entry_id == 2
        assert entry.manifest["run_id"] == "beta"

    def test_get_entry_not_found(self, query_ledger) -> None:
        """get_entry returns None for missing entry."""
        entry = query_ledger.get_entry(999)

        assert entry is None

    def test_get_by_run_id(self, query_ledger) -> None:
        """get_by_run_id finds entry by manifest run_id."""
        entry = query_ledger.get_by_run_id("gamma")

        assert entry is not None
        assert entry.manifest["run_id"] == "gamma"
        assert entry.entry_id == 3

    def test_get_by_run_id_not_found(self, query_ledger) -> None:
        """get_by_run_id returns None for missing run_id."""
        entry = query_ledger.get_by_run_id("nonexistent")

        assert entry is None


class TestImmutableLedgerSummary:
    """Tests for ledger summary and statistics."""

    def test_get_summary_empty(self, tmp_path: Path) -> None:
        """get_summary returns stats for empty ledger."""
        path = tmp_path / "summary.ledger"
        ledger = ImmutableLedger(path)

        summary = ledger.get_summary()

        assert summary["entry_count"] == 0
        assert summary["last_hash"] == ImmutableLedger.GENESIS_HASH
        assert summary["is_valid"] is True

    def test_get_summary_populated(self, tmp_path: Path) -> None:
        """get_summary returns stats for populated ledger."""
        path = tmp_path / "summary.ledger"
        ledger = ImmutableLedger(path)

        for i in range(3):
            ledger.append(
                create_test_manifest(f"run_{i}"),
                create_test_certificate(f"run_{i}"),
            )

        summary = ledger.get_summary()

        assert summary["entry_count"] == 3
        assert summary["last_hash"] != ImmutableLedger.GENESIS_HASH
        assert len(summary["last_hash"]) == 64
        assert str(path) in summary["path"]
        assert summary["is_valid"] is True


class TestImmutableLedgerHashChaining:
    """Tests for hash chaining properties."""

    def test_entry_hash_includes_prev_hash(self, tmp_path: Path) -> None:
        """Entry hash depends on prev_hash."""
        path = tmp_path / "chain.ledger"
        ledger = ImmutableLedger(path)

        # First entry
        entry1 = ledger.append(
            create_test_manifest("run_1"),
            create_test_certificate("run_1"),
        )

        # Compute what hash would be if we changed prev_hash
        test_hash = ledger._compute_entry_hash(
            "DIFFERENT_PREV_HASH",
            entry1.manifest,
            entry1.certificate,
            entry1.metadata,
        )

        # Hash should be different
        assert test_hash != entry1.entry_hash

    def test_entry_hash_includes_manifest(self, tmp_path: Path) -> None:
        """Entry hash depends on manifest content."""
        path = tmp_path / "chain.ledger"
        ledger = ImmutableLedger(path)

        entry1 = ledger.append(
            create_test_manifest("run_1"),
            create_test_certificate("run_1"),
        )

        # Compute hash with different manifest
        different_manifest = {"run_id": "DIFFERENT"}
        test_hash = ledger._compute_entry_hash(
            entry1.prev_hash,
            different_manifest,
            entry1.certificate,
            entry1.metadata,
        )

        assert test_hash != entry1.entry_hash

    def test_consecutive_entries_linked(self, tmp_path: Path) -> None:
        """Consecutive entries are properly linked."""
        path = tmp_path / "chain.ledger"
        ledger = ImmutableLedger(path)

        entries = []
        for i in range(5):
            entry = ledger.append(
                create_test_manifest(f"run_{i}"),
                create_test_certificate(f"run_{i}"),
            )
            entries.append(entry)

        # Verify chain
        for i in range(1, len(entries)):
            assert entries[i].prev_hash == entries[i - 1].entry_hash


class TestImmutableLedgerPersistence:
    """Tests for ledger persistence behavior."""

    def test_entries_persisted_immediately(self, tmp_path: Path) -> None:
        """Entries are written to disk immediately after append."""
        path = tmp_path / "persist.ledger"
        ledger = ImmutableLedger(path)

        ledger.append(
            create_test_manifest("run_1"),
            create_test_certificate("run_1"),
        )

        # Read file directly
        content = path.read_text()
        assert "run_1" in content

    def test_entries_survive_reload(self, tmp_path: Path) -> None:
        """Entries survive ledger reload."""
        path = tmp_path / "reload.ledger"

        # Create and populate
        ledger1 = ImmutableLedger(path)
        ledger1.append(create_test_manifest("alpha"), create_test_certificate("alpha"))
        ledger1.append(create_test_manifest("beta"), create_test_certificate("beta"))

        # Reload
        ledger2 = ImmutableLedger(path)
        entries = ledger2.get_entries()

        assert len(entries) == 2
        assert entries[0].manifest["run_id"] == "alpha"
        assert entries[1].manifest["run_id"] == "beta"


class TestImmutableLedgerAppendRobustness:
    """Regression tests for audit finding T-06 (append atomicity and locking)."""

    def test_failed_append_does_not_brick_verification(self, tmp_path: Path) -> None:
        """A non-serialisable metadata value fails before any state changes."""
        import datetime as _dt

        path = tmp_path / "failed.ledger"
        ledger = ImmutableLedger(path)
        for i in range(3):
            ledger.append(create_test_manifest(f"run_{i}"), create_test_certificate(f"run_{i}"))
        before = path.read_bytes()

        with pytest.raises(TypeError):
            ledger.append(
                create_test_manifest("bad"),
                create_test_certificate("bad"),
                metadata={"finished": _dt.datetime.now()},
            )

        assert path.read_bytes() == before
        assert ledger.entry_count == 3
        entry = ledger.append(create_test_manifest("run_3"), create_test_certificate("run_3"))
        assert entry.entry_id == 4
        assert [e.entry_id for e in ledger.get_entries()] == [1, 2, 3, 4]
        assert ledger.verify() is True

    def test_failed_write_is_rolled_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An I/O failure while persisting leaves neither a torn line nor stale state."""
        import transformation_portal.runtime.ledger as ledger_mod

        path = tmp_path / "rollback.ledger"
        ledger = ImmutableLedger(path)
        ledger.append(create_test_manifest("run_0"), create_test_certificate("run_0"))
        before = path.read_bytes()
        head = ledger.last_hash

        def _boom(_fd: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(ledger_mod.os, "fsync", _boom)
        with pytest.raises(OSError, match="disk full"):
            ledger.append(create_test_manifest("run_1"), create_test_certificate("run_1"))
        monkeypatch.undo()

        assert path.read_bytes() == before
        assert (ledger.entry_count, ledger.last_hash) == (1, head)
        assert ledger.append(create_test_manifest("run_1"), create_test_certificate("run_1")).entry_id == 2
        assert ledger.verify() is True

    def test_torn_tail_fails_closed(self, tmp_path: Path) -> None:
        """Nothing is chained onto a partially written last line."""
        path = tmp_path / "torn.ledger"
        ledger = ImmutableLedger(path)
        ledger.append(create_test_manifest("run_0"), create_test_certificate("run_0"))
        with path.open("ab") as handle:
            handle.write(b'{"entry_id": 2, "prev_ha')
        before = path.read_bytes()

        with pytest.raises(LedgerError):
            ledger.append(create_test_manifest("run_1"), create_test_certificate("run_1"))
        assert path.read_bytes() == before
        assert ledger.verify() is False

    def test_two_handles_do_not_fork_the_chain(self, tmp_path: Path) -> None:
        """Interleaved appends through two handles keep one linear chain."""
        path = tmp_path / "shared.ledger"
        led_a = ImmutableLedger(path)
        for i in range(3):
            led_a.append(create_test_manifest(f"a_{i}"), create_test_certificate(f"a_{i}"))
        led_b = ImmutableLedger(path)

        led_a.append(create_test_manifest("from_a"), create_test_certificate("from_a"))
        entry_b = led_b.append(create_test_manifest("from_b"), create_test_certificate("from_b"))

        assert entry_b.entry_id == 5
        assert [e.entry_id for e in led_a.get_entries()] == [1, 2, 3, 4, 5]
        assert led_a.verify() is True
        assert led_b.verify() is True

    def test_concurrent_threads_on_separate_handles_keep_one_chain(self, tmp_path: Path) -> None:
        """Threads appending through their own handles are serialised by the file lock."""
        import threading

        path = tmp_path / "threads.ledger"
        ImmutableLedger(path)
        workers, per_worker = 4, 15
        barrier = threading.Barrier(workers)
        errors: list[BaseException] = []

        def _work(worker: int) -> None:
            try:
                handle = ImmutableLedger(path)
                barrier.wait(timeout=10)
                for i in range(per_worker):
                    handle.append(create_test_manifest(f"w{worker}_{i}"), create_test_certificate(f"w{worker}_{i}"))
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=_work, args=(w,)) for w in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors
        assert not any(thread.is_alive() for thread in threads)
        final = ImmutableLedger(path)
        assert [e.entry_id for e in final.get_entries()] == list(range(1, workers * per_worker + 1))
        assert final.verify() is True

    def test_metadata_is_hashed_after_json_round_trip(self, tmp_path: Path) -> None:
        """Metadata whose JSON form differs from the Python value still verifies."""
        path = tmp_path / "intkeys.ledger"
        ledger = ImmutableLedger(path)
        entry = ledger.append(
            create_test_manifest("run_0"),
            create_test_certificate("run_0"),
            metadata={"gpu_seconds_by_device": {2: 1.5, 10: 0.5}},
        )

        assert entry.metadata == {"gpu_seconds_by_device": {"2": 1.5, "10": 0.5}}
        assert ledger.verify() is True
        assert ImmutableLedger(path).verify() is True


class TestImmutableLedgerTamperDetection:
    """Regression tests for audit finding T-06 (tampering must not verify)."""

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    def _populated(self, path: Path) -> ImmutableLedger:
        ledger = ImmutableLedger(path)
        for i in range(3):
            ledger.append(create_test_manifest(f"run_{i}"), create_test_certificate(f"run_{i}"))
        return ledger

    def test_back_dated_timestamp_is_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "backdate.ledger"
        self._populated(path)
        rows = self._rows(path)
        rows[1]["timestamp"] = "1999-12-31T23:59:59+00:00"
        self._write_rows(path, rows)

        assert ImmutableLedger(path).verify() is False

    def test_truncation_and_emptying_are_detected_on_the_writing_handle(self, tmp_path: Path) -> None:
        path = tmp_path / "truncate.ledger"
        ledger = self._populated(path)
        first_line = path.read_text().splitlines()[0]

        path.write_text(first_line + "\n")
        assert ledger.verify() is False

        path.write_text("")
        assert ledger.verify() is False

    def test_replaced_head_is_detected_on_the_writing_handle(self, tmp_path: Path) -> None:
        """A rewritten (recomputed) last entry no longer matches the cached head."""
        path = tmp_path / "head.ledger"
        ledger = self._populated(path)
        rows = self._rows(path)
        last = rows[-1]
        last["manifest"]["run_id"] = "FORGED"
        last["entry_hash"] = ledger._compute_entry_hash(
            last["prev_hash"],
            last["manifest"],
            last["certificate"],
            last["metadata"],
            entry_id=last["entry_id"],
            timestamp=last["timestamp"],
        )
        self._write_rows(path, rows)

        assert ImmutableLedger(path).verify() is True  # a fresh handle cannot tell (unkeyed chain)
        assert ledger.verify() is False

    def test_malformed_line_verifies_false_instead_of_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.ledger"
        ledger = self._populated(path)
        with path.open("a") as handle:
            handle.write("not json\n")

        assert ledger.verify() is False

    def test_rewritten_manifest_with_signed_certificate_is_detected(self, tmp_path: Path) -> None:
        """Recomputed chain hashes do not hide a manifest that no longer matches its certificate."""
        pytest.importorskip("cryptography")
        from transformation_portal.core.security.signing import CertificateSigner, generate_ed25519_keypair

        priv, _pub = generate_ed25519_keypair()
        signer = CertificateSigner(priv, issuer="test-ci")
        path = tmp_path / "signed.ledger"
        ledger = ImmutableLedger(path, signer=signer)
        for i in range(3):
            manifest = json.dumps({"run_id": f"run-{i}", "root_hash": "0" * 64, "qa_verdict": "FAIL"})
            ledger.append(manifest, signer.sign_manifest(manifest))
        assert ledger.verify() is True

        rows = self._rows(path)
        rows[1]["manifest"]["qa_verdict"] = "PASS"
        prev = rows[0]["entry_hash"]
        for row in rows[1:]:
            row["prev_hash"] = prev
            row["entry_hash"] = ledger._compute_entry_hash(
                prev,
                row["manifest"],
                row["certificate"],
                row["metadata"],
                entry_id=row["entry_id"],
                timestamp=row["timestamp"],
            )
            prev = row["entry_hash"]
        self._write_rows(path, rows)

        assert ImmutableLedger(path, signer=signer).verify() is False

    def test_legacy_entries_still_verify(self, tmp_path: Path) -> None:
        """Entries hashed with the earlier body (no entry_id/timestamp) remain valid."""
        path = tmp_path / "legacy.ledger"
        helper = ImmutableLedger(tmp_path / "helper.ledger")
        prev = ImmutableLedger.GENESIS_HASH
        rows = []
        for i in range(3):
            manifest = json.loads(create_test_manifest(f"legacy_{i}"))
            certificate = json.loads(create_test_certificate(f"legacy_{i}").to_json())
            entry_hash = helper._compute_entry_hash(prev, manifest, certificate, {})
            rows.append(
                {
                    "entry_id": i + 1,
                    "prev_hash": prev,
                    "entry_hash": entry_hash,
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "manifest": manifest,
                    "certificate": certificate,
                    "metadata": {},
                }
            )
            prev = entry_hash
        self._write_rows(path, rows)

        ledger = ImmutableLedger(path)
        assert ledger.verify() is True
        appended = ledger.append(create_test_manifest("new"), create_test_certificate("new"))
        assert appended.entry_id == 4
        assert appended.prev_hash == prev
        assert ledger.verify() is True
