"""Tests for the Zero-Trust Filesystem Guard (FSGuard).

This module tests the centralized filesystem security layer that
enforces policy-based access control for all disk IO.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.security

from transformation_portal.core.security.fs_guard import (
    FSContext,
    FSGuard,
    FSPolicyError,
    get_fs_guard,
    set_fs_guard,
)
from transformation_portal.core.security.path_safety import PathSafetyError


@pytest.mark.security
class TestFSContext:
    """Tests for FSContext trust boundary."""

    def test_create_user_context(self, tmp_path: Path) -> None:
        """User context can be created."""
        ctx = FSContext(mode="user", base_dir=tmp_path)
        assert ctx.mode == "user"
        assert ctx.base_dir == tmp_path

    def test_create_internal_context(self, tmp_path: Path) -> None:
        """Internal context can be created."""
        ctx = FSContext(mode="internal", base_dir=tmp_path)
        assert ctx.mode == "internal"

    def test_create_cas_context(self, tmp_path: Path) -> None:
        """CAS context can be created."""
        ctx = FSContext(mode="cas", base_dir=tmp_path)
        assert ctx.mode == "cas"

    def test_context_is_immutable(self, tmp_path: Path) -> None:
        """FSContext is frozen (immutable)."""
        ctx = FSContext(mode="user", base_dir=tmp_path)
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.mode = "internal"  # type: ignore


@pytest.mark.security
class TestFSGuardUserFile:
    """Tests for FSGuard.user_file() method."""

    def test_user_file_basic(self, tmp_path: Path) -> None:
        """Basic user file path construction works."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "valid_name", suffix=".json")
        assert path == tmp_path / "valid_name.json"

    def test_user_file_rejects_traversal(self, tmp_path: Path) -> None:
        """Path traversal attempts are rejected."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        with pytest.raises(PathSafetyError):
            fs.user_file(ctx, "../evil", suffix=".json")

    def test_user_file_rejects_slash(self, tmp_path: Path) -> None:
        """Slashes in name are rejected."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        with pytest.raises(PathSafetyError):
            fs.user_file(ctx, "a/b", suffix=".json")

    def test_user_file_wrong_context(self, tmp_path: Path) -> None:
        """User file requires user context."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        with pytest.raises(FSPolicyError):
            fs.user_file(ctx, "name", suffix=".json")

    @pytest.mark.parametrize(
        "bad_name",
        ["", ".", "..", "a/b", "a\\b", "a..b", "🔥", "<script>"],
    )
    def test_user_file_rejects_bad_names(self, tmp_path: Path, bad_name: str) -> None:
        """Invalid names are rejected."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        with pytest.raises((PathSafetyError, FSPolicyError)):
            fs.user_file(ctx, bad_name, suffix=".json")


@pytest.mark.security
class TestFSGuardInternalPath:
    """Tests for FSGuard.internal_path() method."""

    def test_internal_path_basic(self, tmp_path: Path) -> None:
        """Basic internal path construction works."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        path = fs.internal_path(ctx, ["level1", "level2"])
        assert path == tmp_path / "level1" / "level2"

    def test_internal_path_single_segment(self, tmp_path: Path) -> None:
        """Single segment works."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        path = fs.internal_path(ctx, ["single"])
        assert path == tmp_path / "single"

    def test_internal_path_wrong_context(self, tmp_path: Path) -> None:
        """Internal path requires internal context."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        with pytest.raises(FSPolicyError):
            fs.internal_path(ctx, ["path"])

    def test_internal_path_validates_segments(self, tmp_path: Path) -> None:
        """Each segment is validated."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        with pytest.raises(PathSafetyError):
            fs.internal_path(ctx, ["valid", "../evil"])


@pytest.mark.security
class TestFSGuardInternalFile:
    """Tests for FSGuard.internal_file() method."""

    def test_internal_file_basic(self, tmp_path: Path) -> None:
        """Basic internal file construction works."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        path = fs.internal_file(ctx, ["subdir", "file"], suffix=".json")
        assert path == tmp_path / "subdir" / "file.json"

    def test_internal_file_single_part(self, tmp_path: Path) -> None:
        """Single part (just filename) works."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        path = fs.internal_file(ctx, ["filename"], suffix=".txt")
        assert path == tmp_path / "filename.txt"


@pytest.mark.security
class TestFSGuardCAS:
    """Tests for FSGuard.cas_object() method."""

    def test_cas_object_basic(self, tmp_path: Path) -> None:
        """Basic CAS path uses 2-char sharding."""
        fs = FSGuard()
        ctx = FSContext(mode="cas", base_dir=tmp_path)

        sha = "a" * 64
        path = fs.cas_object(ctx, sha)
        assert path == tmp_path / "aa" / sha

    def test_cas_object_wrong_context(self, tmp_path: Path) -> None:
        """CAS object requires CAS context."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        with pytest.raises(FSPolicyError):
            fs.cas_object(ctx, "a" * 64)

    def test_cas_object_invalid_sha(self, tmp_path: Path) -> None:
        """Invalid SHA256 is rejected."""
        fs = FSGuard()
        ctx = FSContext(mode="cas", base_dir=tmp_path)

        with pytest.raises(PathSafetyError):
            fs.cas_object(ctx, "invalid")


@pytest.mark.security
class TestFSGuardFileOperations:
    """Tests for FSGuard file operations."""

    def test_write_and_read_text(self, tmp_path: Path) -> None:
        """Write and read text works."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "test", suffix=".txt")
        fs.write_text(path, "hello world")

        assert fs.read_text(path) == "hello world"

    def test_write_and_read_bytes(self, tmp_path: Path) -> None:
        """Write and read bytes works."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "binary", suffix=".bin")
        fs.write_bytes(path, b"\x00\x01\x02\x03")

        assert fs.read_bytes(path) == b"\x00\x01\x02\x03"

    def test_exists(self, tmp_path: Path) -> None:
        """Exists check works."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "file", suffix=".txt")

        assert not fs.exists(path)
        fs.write_text(path, "content")
        assert fs.exists(path)

    def test_delete(self, tmp_path: Path) -> None:
        """Delete removes file."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "todelete", suffix=".txt")
        fs.write_text(path, "content")
        assert fs.exists(path)

        fs.delete(path)
        assert not fs.exists(path)

    def test_delete_missing_ok(self, tmp_path: Path) -> None:
        """Delete with missing_ok doesn't raise."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "nonexistent", suffix=".txt")
        result = fs.delete(path, missing_ok=True)
        assert result is False

    def test_mkdir(self, tmp_path: Path) -> None:
        """Mkdir creates directory."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        path = fs.internal_path(ctx, ["newdir", "subdir"])
        fs.mkdir(path)

        assert path.exists()
        assert path.is_dir()

    def test_list_dir(self, tmp_path: Path) -> None:
        """List dir returns contents."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        # Create some files
        for name in ["file1", "file2", "file3"]:
            path = fs.user_file(ctx, name, suffix=".txt")
            fs.write_text(path, "content")

        files = fs.list_dir(tmp_path)
        assert len(files) == 3

    def test_copy(self, tmp_path: Path) -> None:
        """Copy duplicates file."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        src = fs.user_file(ctx, "source", suffix=".txt")
        dst = fs.user_file(ctx, "dest", suffix=".txt")

        fs.write_text(src, "original content")
        fs.copy(src, dst)

        assert fs.read_text(dst) == "original content"

    def test_symlink(self, tmp_path: Path) -> None:
        """Symlink creates link."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        src = fs.user_file(ctx, "source", suffix=".txt")
        link = fs.user_file(ctx, "link", suffix=".txt")

        fs.write_text(src, "content")
        fs.symlink(src, link)

        assert link.is_symlink()
        assert fs.read_text(link) == "content"


@pytest.mark.security
class TestFSGuardAuditLog:
    """Tests for FSGuard audit logging."""

    def test_audit_log_created(self, tmp_path: Path) -> None:
        """Audit log is created when specified."""
        audit_path = tmp_path / "audit.jsonl"
        fs = FSGuard(audit_log=audit_path)
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "test", suffix=".txt")
        fs.write_text(path, "content")
        fs.read_text(path)
        fs.delete(path)

        assert audit_path.exists()

        # Parse audit entries
        entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert len(entries) == 3

        ops = [e["op"] for e in entries]
        assert ops == ["write", "read", "delete"]

    def test_operation_count(self, tmp_path: Path) -> None:
        """Operation count is tracked."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        assert fs.operation_count == 0

        path = fs.user_file(ctx, "test", suffix=".txt")
        fs.write_text(path, "content")
        assert fs.operation_count == 1

        fs.read_text(path)
        assert fs.operation_count == 2


@pytest.mark.security
class TestFSGuardGlobalInstance:
    """Tests for global FSGuard instance."""

    def test_get_fs_guard_returns_instance(self) -> None:
        """get_fs_guard returns an FSGuard instance."""
        fs = get_fs_guard()
        assert isinstance(fs, FSGuard)

    def test_set_fs_guard_changes_instance(self) -> None:
        """set_fs_guard allows replacing the global instance."""
        original = get_fs_guard()

        custom = FSGuard()
        set_fs_guard(custom)

        assert get_fs_guard() is custom

        # Restore original
        set_fs_guard(original)


@pytest.mark.security
class TestFSGuardAtomicWrite:
    """Tests for atomic write functionality."""

    def test_atomic_write(self, tmp_path: Path) -> None:
        """Atomic write uses temp file then rename."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "atomic", suffix=".json")
        fs.write_text(path, '{"key": "value"}', atomic=True)

        # Temp file should not exist
        tmp_file = path.with_suffix(".json.tmp")
        assert not tmp_file.exists()

        # Final file should have content
        assert fs.read_text(path) == '{"key": "value"}'

    def test_non_atomic_write(self, tmp_path: Path) -> None:
        """Non-atomic write writes directly."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "nonatomic", suffix=".txt")
        fs.write_text(path, "content", atomic=False)

        assert fs.read_text(path) == "content"
