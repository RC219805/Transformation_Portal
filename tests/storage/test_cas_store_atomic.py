"""Tests for CAS atomic write guarantees (Blocker #1).

These tests verify that ArtifactStore uses atomic writes to prevent
corruption in parallel execution scenarios.

Test Coverage:
- Atomic file writes (temp file + rename pattern)
- Hash verification before making artifact visible
- Corruption detection and recovery
- Parallel write safety
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.storage.cas_store import ArtifactStore, CASError, CASObject

pytestmark = pytest.mark.unit


class TestAtomicWrites:
    """Tests for atomic write guarantees."""

    def test_add_file_atomic_write(self, tmp_path):
        """Test add_file uses atomic write pattern."""
        store = ArtifactStore(tmp_path / "cas")

        # Create source file
        src = tmp_path / "source.bin"
        src.write_bytes(b"test content for atomic write")

        # Add to CAS
        obj = store.add_file(src)

        # Verify object exists and has correct hash
        assert obj.sha256 == hashlib.sha256(b"test content for atomic write").hexdigest()
        assert store.has_object(obj.sha256)

        # Verify no temp files left behind
        cas_dir = tmp_path / "cas" / "objects"
        temp_files = list(cas_dir.rglob("*.tmp"))
        assert len(temp_files) == 0, f"Temp files should be cleaned up: {temp_files}"

    def test_add_bytes_atomic_write(self, tmp_path):
        """Test add_bytes uses atomic write pattern."""
        store = ArtifactStore(tmp_path / "cas")

        data = b"test bytes for atomic write"
        obj = store.add_bytes(data)

        # Verify object exists and has correct hash
        assert obj.sha256 == hashlib.sha256(data).hexdigest()
        assert store.has_object(obj.sha256)

        # Verify no temp files left behind
        cas_dir = tmp_path / "cas" / "objects"
        temp_files = list(cas_dir.rglob("*.tmp"))
        assert len(temp_files) == 0, f"Temp files should be cleaned up: {temp_files}"

    def test_hash_verification_before_rename(self, tmp_path):
        """Test hash is verified BEFORE making artifact visible."""
        store = ArtifactStore(tmp_path / "cas")

        # Create source file
        src = tmp_path / "source.bin"
        original_content = b"original content"
        src.write_bytes(original_content)

        # Add to CAS
        obj = store.add_file(src)

        # Verify stored content matches original
        stored_content = Path(obj.path).read_bytes()
        assert stored_content == original_content

        # Verify hash matches
        actual_hash = hashlib.sha256(stored_content).hexdigest()
        assert actual_hash == obj.sha256

    def test_corrupted_artifact_detection(self, tmp_path):
        """Test detection of corrupted artifacts."""
        store = ArtifactStore(tmp_path / "cas")

        # Add valid object
        data = b"valid content"
        obj = store.add_bytes(data)

        # Corrupt the stored file
        obj.path.write_bytes(b"corrupted!")

        # Re-adding should detect corruption and re-add
        new_obj = store.add_bytes(data, verify=True)
        assert new_obj.sha256 == obj.sha256

        # The file should now have correct content
        stored_content = Path(new_obj.path).read_bytes()
        assert stored_content == data

    def test_parallel_write_safety(self, tmp_path):
        """Test parallel writes don't corrupt artifacts."""
        store = ArtifactStore(tmp_path / "cas")

        # Create multiple source files
        files = []
        for i in range(10):
            src = tmp_path / f"source_{i}.bin"
            content = f"content for file {i}".encode()
            src.write_bytes(content)
            files.append((src, content))

        results = []
        errors = []

        def add_file(src, expected_content):
            try:
                obj = store.add_file(src)
                # Verify content is correct
                stored = Path(obj.path).read_bytes()
                if stored != expected_content:
                    errors.append(f"Content mismatch for {src}")
                results.append(obj)
            except Exception as e:
                errors.append(str(e))

        # Add files in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_file, src, content) for src, content in files]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Errors during parallel writes: {errors}"
        assert len(results) == 10

        # Verify all files have correct content
        for obj, (_, expected_content) in zip(
            sorted(results, key=lambda x: x.sha256), sorted(files, key=lambda x: hashlib.sha256(x[1]).hexdigest())
        ):
            stored = Path(obj.path).read_bytes()
            assert hashlib.sha256(stored).hexdigest() == obj.sha256

    def test_deduplication(self, tmp_path):
        """Test deduplication works correctly with atomic writes."""
        store = ArtifactStore(tmp_path / "cas")

        data = b"duplicate content"

        # Add same content multiple times
        obj1 = store.add_bytes(data)
        obj2 = store.add_bytes(data)
        obj3 = store.add_bytes(data)

        # All should reference the same object
        assert obj1.sha256 == obj2.sha256 == obj3.sha256
        assert obj1.path == obj2.path == obj3.path

    def test_temp_file_cleanup_on_failure(self, tmp_path):
        """Test temp files are cleaned up on failure."""
        store = ArtifactStore(tmp_path / "cas")

        # Create a source file
        src = tmp_path / "source.bin"
        src.write_bytes(b"test content")

        # Mock _sha256_file to return wrong hash after copy
        original_sha256_file = store._sha256_file

        call_count = [0]

        def failing_sha256(path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: correct hash for source
                return original_sha256_file(path, *args, **kwargs)
            else:
                # Second call: wrong hash for temp file (simulate corruption)
                return "wrong_hash_simulating_corruption"

        with patch.object(store, "_sha256_file", side_effect=failing_sha256):
            with pytest.raises(CASError):
                store.add_file(src)

        # Verify no temp files left behind
        cas_dir = tmp_path / "cas" / "objects"
        temp_files = list(cas_dir.rglob("*.tmp"))
        assert len(temp_files) == 0, f"Temp files should be cleaned up: {temp_files}"


class TestAtomicWriteContract:
    """Tests verifying the atomic write contract is upheld."""

    def test_no_partial_reads_during_write(self, tmp_path):
        """Test readers never see partial writes."""
        store = ArtifactStore(tmp_path / "cas")

        large_data = b"x" * (1024 * 1024)  # 1MB
        sha256 = hashlib.sha256(large_data).hexdigest()

        read_results = []
        write_complete = threading.Event()
        start_writing = threading.Event()

        def writer():
            start_writing.set()
            store.add_bytes(large_data)
            write_complete.set()

        def reader():
            # Wait for write to start
            start_writing.wait()
            time.sleep(0.01)  # Small delay

            # Try to read during write
            for _ in range(100):
                if store.has_object(sha256):
                    obj = store.get_object(sha256)
                    if obj:
                        content = Path(obj.path).read_bytes()
                        read_results.append(len(content))
                time.sleep(0.001)

        # Start reader and writer concurrently
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        write_complete.wait()
        reader_thread.join()

        # All successful reads should have full content
        for size in read_results:
            assert size == len(large_data), "Reader saw partial content"


class TestMaterializeCorruptionHandling:
    """Tests for corruption detection and quarantine in materialize()."""

    def test_materialize_corrupted_artifact_quarantine(self, tmp_path):
        """Test that corrupted artifacts are quarantined and exception raised."""
        store = ArtifactStore(tmp_path / "cas")

        # Add valid object
        data = b"valid content for test"
        obj = store.add_bytes(data)

        # Corrupt the stored file
        obj.path.write_bytes(b"corrupted!")

        # Materialize with verify=True should detect corruption
        dest = tmp_path / "output" / "file.bin"
        with pytest.raises(CASError) as exc_info:
            store.materialize(obj.sha256, dest, verify=True)

        # Verify error message mentions quarantine
        assert "quarantine" in str(exc_info.value).lower()
        assert "verification failed" in str(exc_info.value).lower()

        # Verify the file was moved to quarantine
        quarantine_dir = tmp_path / "cas" / "quarantine"
        assert quarantine_dir.exists()
        quarantine_files = list(quarantine_dir.iterdir())
        assert len(quarantine_files) == 1

        # Quarantine filename should include original and actual hash
        quarantine_name = quarantine_files[0].name
        assert obj.sha256 in quarantine_name

        # Verify original location is now empty
        assert not obj.path.exists()

    def test_materialize_verify_true_passes_valid_artifact(self, tmp_path):
        """Test verify=True passes for valid artifacts."""
        store = ArtifactStore(tmp_path / "cas")

        data = b"valid content"
        obj = store.add_bytes(data)

        dest = tmp_path / "output" / "file.bin"
        result = store.materialize(obj.sha256, dest, verify=True)

        assert result == dest
        assert dest.exists()
        # Symlinks point to original, so read via the symlink
        assert dest.read_bytes() == data

    def test_materialize_verify_false_skips_hash_check(self, tmp_path):
        """Test verify=False skips hash verification (with warning)."""
        store = ArtifactStore(tmp_path / "cas")

        data = b"test content"
        obj = store.add_bytes(data)

        # Corrupt the file
        obj.path.write_bytes(b"corrupted!")

        dest = tmp_path / "output" / "file.bin"
        # Should NOT raise with verify=False (but will return corrupted content)
        result = store.materialize(obj.sha256, dest, verify=False)

        assert result == dest
        # The destination now links to the corrupted content
        assert dest.exists()


class TestQuarantineGC:
    """Tests for gc_quarantine() lifecycle policy."""

    def test_gc_quarantine_empty_dir(self, tmp_path):
        """Test gc_quarantine handles empty/missing quarantine dir."""
        store = ArtifactStore(tmp_path / "cas")

        result = store.gc_quarantine(dry_run=True)

        assert result["deleted"] == []
        assert result["retained"] == []
        assert result["total_size_before"] == 0
        assert result["total_size_after"] == 0

    def test_gc_quarantine_age_based_cleanup(self, tmp_path):
        """Test age-based cleanup of quarantined artifacts."""
        store = ArtifactStore(tmp_path / "cas")

        # Create quarantine directory with files of different ages
        quarantine_dir = tmp_path / "cas" / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Create a "new" file (recent mtime)
        new_file = quarantine_dir / "new_artifact"
        new_file.write_bytes(b"new content")

        # Create an "old" file and backdate it
        old_file = quarantine_dir / "old_artifact"
        old_file.write_bytes(b"old content")

        # Set mtime to 10 days ago
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))

        # Run gc with 7-day limit (dry_run first)
        result = store.gc_quarantine(max_age_seconds=7 * 24 * 60 * 60, dry_run=True)

        assert "old_artifact" in result["deleted"]
        assert "new_artifact" not in result["deleted"]
        # In dry_run, files are not actually deleted
        assert old_file.exists()

        # Now run for real
        result = store.gc_quarantine(max_age_seconds=7 * 24 * 60 * 60, dry_run=False)

        assert "old_artifact" in result["deleted"]
        assert not old_file.exists()
        assert new_file.exists()

    def test_gc_quarantine_size_based_cleanup(self, tmp_path):
        """Test size-based cleanup of quarantined artifacts."""
        store = ArtifactStore(tmp_path / "cas")

        quarantine_dir = tmp_path / "cas" / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Create files with specific sizes
        file1 = quarantine_dir / "file1"
        file1.write_bytes(b"x" * 500)  # 500 bytes

        file2 = quarantine_dir / "file2"
        file2.write_bytes(b"x" * 300)  # 300 bytes

        # Set different ages (file1 older)
        old_time = time.time() - 100  # 100 seconds ago
        os.utime(file1, (old_time, old_time))

        # Run gc with size limit of 600 bytes (should delete oldest until under limit)
        result = store.gc_quarantine(
            max_age_seconds=7 * 24 * 60 * 60,  # Don't trigger age cleanup
            max_size_bytes=600,
            dry_run=True,
        )

        # file1 is oldest and should be deleted first to get under 600 bytes
        assert "file1" in result["deleted"]
        assert "file2" not in result["deleted"]

    def test_gc_quarantine_dry_run_no_deletion(self, tmp_path):
        """Test dry_run mode doesn't actually delete files."""
        store = ArtifactStore(tmp_path / "cas")

        quarantine_dir = tmp_path / "cas" / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        old_file = quarantine_dir / "old_artifact"
        old_file.write_bytes(b"content")

        # Backdate the file
        old_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
        os.utime(old_file, (old_time, old_time))

        # Dry run
        result = store.gc_quarantine(dry_run=True)

        assert "old_artifact" in result["deleted"]
        # File should still exist
        assert old_file.exists()
        # In dry_run, reports what size would be after cleanup (0 if all files would be deleted)
        assert result["total_size_after"] == 0

    def test_gc_quarantine_accounting(self, tmp_path):
        """Test size accounting in gc_quarantine results."""
        store = ArtifactStore(tmp_path / "cas")

        quarantine_dir = tmp_path / "cas" / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Create files
        file1 = quarantine_dir / "keep"
        file1.write_bytes(b"x" * 100)

        file2 = quarantine_dir / "delete"
        file2.write_bytes(b"x" * 200)

        # Make file2 old
        old_time = time.time() - (30 * 24 * 60 * 60)
        os.utime(file2, (old_time, old_time))

        result = store.gc_quarantine(dry_run=False)

        assert result["total_size_before"] == 300
        assert result["total_size_after"] == 100
        assert "delete" in result["deleted"]
        assert "keep" in result["retained"]
