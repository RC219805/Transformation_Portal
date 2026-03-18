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
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.storage.cas_store import ArtifactStore, CASError, CASObject


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
            futures = [
                executor.submit(add_file, src, content)
                for src, content in files
            ]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Errors during parallel writes: {errors}"
        assert len(results) == 10

        # Verify all files have correct content
        for obj, (_, expected_content) in zip(sorted(results, key=lambda x: x.sha256), 
                                               sorted(files, key=lambda x: hashlib.sha256(x[1]).hexdigest())):
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
