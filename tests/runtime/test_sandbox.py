"""Tests for the execution sandbox and runtime components.

This module tests:
- Sandbox creation and CAS-only IO
- FSGuard integration
- Workspace isolation
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.core.security.fs_guard import FSContext, FSGuard
from transformation_portal.core.security.path_safety import PathSafetyError
from transformation_portal.runtime.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxError,
)
from transformation_portal.storage.cas_store import ArtifactStore


@pytest.mark.security
class TestSandbox:
    """Tests for execution sandbox."""

    @pytest.fixture
    def sandbox_env(self, tmp_path: Path):
        """Create sandbox environment."""
        fs = FSGuard()
        cas = ArtifactStore(tmp_path / "cas")

        config = SandboxConfig(
            workspace_root=tmp_path / "workspace",
            cas_root=tmp_path / "cas",
        )

        return fs, cas, config

    def test_sandbox_creation(self, sandbox_env, tmp_path: Path) -> None:
        """Sandbox creates workspace directory."""
        fs, cas, config = sandbox_env

        sandbox = Sandbox(
            node_id="test_node",
            config=config,
            fs=fs,
            cas=cas,
        )

        assert sandbox.workspace.exists()
        assert sandbox.workspace.name == "test_node"

    def test_sandbox_invalid_node_id(self, sandbox_env) -> None:
        """Invalid node ID is rejected."""
        fs, cas, config = sandbox_env

        with pytest.raises(SandboxError):
            Sandbox(
                node_id="../evil",
                config=config,
                fs=fs,
                cas=cas,
            )

    def test_sandbox_write_and_read(self, sandbox_env) -> None:
        """Write and read within sandbox works."""
        fs, cas, config = sandbox_env

        sandbox = Sandbox(
            node_id="write_test",
            config=config,
            fs=fs,
            cas=cas,
        )

        path = sandbox.write(["output"], "test content", suffix=".txt")
        assert path.exists()

        content = sandbox.read("output.txt")
        assert content == "test content"

    def test_sandbox_persist_output(self, sandbox_env, tmp_path: Path) -> None:
        """Persisting output to CAS works."""
        fs, cas, config = sandbox_env

        sandbox = Sandbox(
            node_id="persist_test",
            config=config,
            fs=fs,
            cas=cas,
        )

        # Write a file
        path = sandbox.write(["result"], "output data", suffix=".json")

        # Persist to CAS
        sha = sandbox.persist_output(path)

        assert len(sha) == 64
        assert cas.has_object(sha)

    def test_sandbox_materialize_input(self, sandbox_env, tmp_path: Path) -> None:
        """Materializing input from CAS works."""
        fs, cas, config = sandbox_env

        # Add an object to CAS first
        src_file = tmp_path / "source.txt"
        src_file.write_text("source content")
        obj = cas.add_file(src_file)

        sandbox = Sandbox(
            node_id="materialize_test",
            config=config,
            fs=fs,
            cas=cas,
        )

        # Materialize
        path = sandbox.materialize_input(obj.sha256, "input.txt")

        assert path.exists()
        # Should be a symlink
        assert path.is_symlink() or path.read_text() == "source content"

    def test_sandbox_metrics(self, sandbox_env) -> None:
        """Sandbox tracks metrics."""
        fs, cas, config = sandbox_env

        sandbox = Sandbox(
            node_id="metrics_test",
            config=config,
            fs=fs,
            cas=cas,
        )

        sandbox.start()
        sandbox.write(["data"], "test", suffix=".txt")
        sandbox.finish()

        assert sandbox.metrics.bytes_written > 0
        assert sandbox.metrics.duration_seconds is not None

    def test_sandbox_manifest(self, sandbox_env) -> None:
        """Sandbox generates execution manifest."""
        fs, cas, config = sandbox_env

        sandbox = Sandbox(
            node_id="manifest_test",
            config=config,
            fs=fs,
            cas=cas,
        )

        manifest = sandbox.get_manifest()

        assert manifest["node_id"] == "manifest_test"
        assert "inputs" in manifest
        assert "outputs" in manifest


@pytest.mark.security
class TestSandboxWorkspaceConfinement:
    """Workspace confinement must hold after resolution, not just lexically (T-03)."""

    @pytest.fixture
    def env(self, tmp_path: Path):
        cas = ArtifactStore(tmp_path / "cas")
        config = SandboxConfig(workspace_root=tmp_path / "ws", cas_root=tmp_path / "cas")
        victim = Sandbox("tenant_b_node", config, FSGuard(), cas)
        (victim.workspace / "secret.txt").write_text("tenant-b private data")
        outside = tmp_path / "outside_secret.txt"
        outside.write_text("host file")
        attacker = Sandbox("tenant_a_node", config, FSGuard(), cas)
        return attacker, cas, outside

    @pytest.mark.parametrize(
        "rel_path",
        [
            "../tenant_b_node/secret.txt",
            "../../outside_secret.txt",
            "./../../outside_secret.txt",
            "..\\..\\outside_secret.txt",
            "",
        ],
    )
    def test_read_rejects_dotdot_escape(self, env, rel_path: str) -> None:
        attacker, _cas, _outside = env
        with pytest.raises(SandboxError):
            attacker.read(rel_path)
        with pytest.raises(SandboxError):
            attacker.read_bytes(rel_path)

    def test_read_rejects_absolute_path(self, env) -> None:
        attacker, _cas, outside = env
        with pytest.raises(SandboxError):
            attacker.read_bytes(str(outside))

    def test_read_rejects_symlink_out_of_workspace(self, env) -> None:
        attacker, _cas, outside = env
        (attacker.workspace / "link.txt").symlink_to(outside)
        (attacker.workspace / "linkdir").symlink_to(outside.parent, target_is_directory=True)
        with pytest.raises(SandboxError):
            attacker.read("link.txt")
        with pytest.raises(SandboxError):
            attacker.read_bytes("linkdir/outside_secret.txt")

    @pytest.mark.parametrize("rel_path", ["loop", "loop/missing/data.bin"])
    def test_read_rejects_symlink_loop_as_unresolvable(self, env, rel_path: str) -> None:
        attacker, _cas, _outside = env
        (attacker.workspace / "loop").symlink_to("loop")
        with pytest.raises(SandboxError, match="Cannot resolve workspace path"):
            attacker.read(rel_path)
        with pytest.raises(SandboxError, match="Cannot resolve workspace path"):
            attacker.read_bytes(rel_path)

    def test_persist_output_rejects_dotdot_and_symlink_escape(self, env) -> None:
        attacker, cas, outside = env
        before = sorted(p.name for p in cas.objects_dir.rglob("*") if p.is_file())
        with pytest.raises(SandboxError):
            attacker.persist_output(attacker.workspace / ".." / ".." / "outside_secret.txt")
        (attacker.workspace / "exfil.txt").symlink_to(outside)
        with pytest.raises(SandboxError):
            attacker.persist_output(attacker.workspace / "exfil.txt")
        assert sorted(p.name for p in cas.objects_dir.rglob("*") if p.is_file()) == before

    def test_materialize_input_rejects_absolute_and_dotdot_target(self, env, tmp_path: Path) -> None:
        attacker, cas, _outside = env
        obj = cas.add_bytes(b"attacker controlled bytes")
        target = tmp_path / "important_host_file.cfg"
        target.write_text("original")
        for rel_path in (str(target), "../../important_host_file.cfg", "..", "sub/.."):
            with pytest.raises(SandboxError):
                attacker.materialize_input(obj.sha256, rel_path)
        assert not target.is_symlink()
        assert target.read_text() == "original"

    def test_materialize_input_rejects_symlinked_directory(self, env, tmp_path: Path) -> None:
        attacker, cas, _outside = env
        obj = cas.add_bytes(b"payload")
        escape_dir = tmp_path / "escape_dir"
        escape_dir.mkdir()
        (attacker.workspace / "inputs").symlink_to(escape_dir, target_is_directory=True)
        with pytest.raises(SandboxError):
            attacker.materialize_input(obj.sha256, "inputs/x.bin")
        assert list(escape_dir.iterdir()) == []

    def test_materialized_input_stays_readable_and_rematerializable(self, env) -> None:
        attacker, cas, _outside = env
        first = cas.add_bytes(b"first")
        second = cas.add_bytes(b"second")
        path = attacker.materialize_input(first.sha256, "inputs/image.png")
        assert path == attacker.workspace / "inputs" / "image.png"
        assert attacker.read_bytes("inputs/image.png") == b"first"
        attacker.materialize_input(second.sha256, "inputs/image.png")
        assert attacker.read_bytes("inputs/image.png") == b"second"
        assert attacker.persist_output(path) == second.sha256


@pytest.mark.security
class TestFSGuardIntegration:
    """Tests for FSGuard integration with sandbox."""

    def test_fsguard_user_file(self, tmp_path: Path) -> None:
        """FSGuard user file creation works."""
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=tmp_path)

        path = fs.user_file(ctx, "testfile", suffix=".json")
        assert path == tmp_path / "testfile.json"

    def test_fsguard_internal_path(self, tmp_path: Path) -> None:
        """FSGuard internal path creation works."""
        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        path = fs.internal_path(ctx, ["level1", "level2"])
        assert path == tmp_path / "level1" / "level2"

    def test_fsguard_cas_object(self, tmp_path: Path) -> None:
        """FSGuard CAS object path works."""
        fs = FSGuard()
        ctx = FSContext(mode="cas", base_dir=tmp_path)

        sha = "a" * 64
        path = fs.cas_object(ctx, sha)
        assert path == tmp_path / "aa" / sha

    def test_fsguard_wrong_context_rejected(self, tmp_path: Path) -> None:
        """Wrong context mode is rejected."""
        from transformation_portal.core.security.fs_guard import FSPolicyError

        fs = FSGuard()
        ctx = FSContext(mode="internal", base_dir=tmp_path)

        with pytest.raises(FSPolicyError):
            fs.user_file(ctx, "file", suffix=".txt")
