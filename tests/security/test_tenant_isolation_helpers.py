from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from transformation_portal.core.security.fs_guard import FSGuard
from transformation_portal.core.security.tenant import (
    TenantAwareFSGuard,
    TenantContext,
    TenantError,
    TenantManager,
    TenantPolicy,
    create_tenant_sandbox,
)

pytestmark = [pytest.mark.unit, pytest.mark.security]


def test_tenant_context_rejects_invalid_tenant_id(tmp_path: Path) -> None:
    with pytest.raises(TenantError, match="Invalid tenant_id"):
        TenantContext(tenant_id="../escape", workspace_root=tmp_path / "work", cas_root=tmp_path / "cas")


def test_tenant_context_properties_derive_isolated_paths(tmp_path: Path) -> None:
    tenant = TenantContext(tenant_id="tenant_a", workspace_root=tmp_path / "work", cas_root=tmp_path / "cas")

    assert tenant.tenant_workspace == tmp_path / "work" / "tenant_a"
    assert tenant.tenant_cas == tmp_path / "cas" / "tenant_a"


def test_tenant_manager_creates_tenant_directories_and_copies_default_policy(tmp_path: Path) -> None:
    default_policy = TenantPolicy(max_workspace_size_mb=128, allowed_node_types={"render"}, network_allowed=True)
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas", default_policy=default_policy)

    tenant = manager.create_tenant("tenant_a", metadata={"region": "west"})
    policy = manager.get_policy("tenant_a")

    assert tenant.metadata == {"region": "west"}
    assert tenant.tenant_workspace.is_dir()
    assert tenant.tenant_cas.is_dir()
    assert policy == default_policy
    assert policy is not default_policy
    policy.allowed_node_types.add("composite")
    assert policy.allowed_node_types == {"render", "composite"}
    assert default_policy.allowed_node_types == {"render"}


def test_tenant_manager_default_policy_copies_are_isolated_between_tenants(tmp_path: Path) -> None:
    default_policy = TenantPolicy(allowed_node_types={"render"})
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas", default_policy=default_policy)

    manager.create_tenant("tenant_a")
    manager.create_tenant("tenant_b")

    policy_a = manager.get_policy("tenant_a")
    policy_b = manager.get_policy("tenant_b")
    assert policy_a is not None
    assert policy_b is not None
    assert policy_a is not policy_b

    policy_a.allowed_node_types.add("composite")

    assert policy_a.allowed_node_types == {"render", "composite"}
    assert policy_b.allowed_node_types == {"render"}
    assert default_policy.allowed_node_types == {"render"}


def test_tenant_manager_rejects_duplicate_tenants(tmp_path: Path) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    manager.create_tenant("tenant_a")

    with pytest.raises(TenantError, match="Tenant already exists"):
        manager.create_tenant("tenant_a")


def test_tenant_manager_delete_tenant_optionally_removes_data(tmp_path: Path) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    (tenant.tenant_workspace / "payload.txt").write_text("payload", encoding="utf-8")
    (tenant.tenant_cas / "obj.txt").write_text("obj", encoding="utf-8")

    assert manager.delete_tenant("tenant_a", delete_data=True) is True
    assert not tenant.tenant_workspace.exists()
    assert not tenant.tenant_cas.exists()
    assert manager.delete_tenant("tenant_a") is False


def test_tenant_manager_access_enforcement_and_node_type_policy(tmp_path: Path) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a", policy=TenantPolicy(allowed_node_types={"render"}))
    allowed_workspace = tenant.tenant_workspace / "result.json"
    allowed_cas = tenant.tenant_cas / "sha256"
    outside = tmp_path / "outside.txt"

    manager.enforce_tenant_access(tenant, allowed_workspace)
    manager.enforce_tenant_access(tenant, allowed_cas)
    manager.enforce_node_type("tenant_a", "render")

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        manager.enforce_tenant_access(tenant, outside)

    with pytest.raises(TenantError, match="Node type not allowed"):
        manager.enforce_node_type("tenant_a", "other")


def test_tenant_aware_fs_guard_enforces_current_tenant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    inside = tenant.tenant_workspace / "payload.txt"
    outside = tmp_path / "outside.txt"

    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(FSGuard, "read_text", lambda self, path, encoding="utf-8": calls.append(("read", path)) or "payload")
    monkeypatch.setattr(
        FSGuard,
        "write_text",
        lambda self, path, data, encoding="utf-8", atomic=True: calls.append(("write", path)),
    )
    monkeypatch.setattr(FSGuard, "delete", lambda self, path, missing_ok=True: calls.append(("delete", path)) or True)

    guard.enforce_path(inside)
    assert guard.read_text(inside) == "payload"
    guard.write_text(inside, "data")
    assert guard.delete(inside) is True
    assert calls == [("read", inside), ("write", inside), ("delete", inside)]

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        guard.enforce_path(outside)

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        guard.read_text(outside)


def test_create_tenant_sandbox_uses_tenant_scoped_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tenant = TenantContext(tenant_id="tenant_a", workspace_root=tmp_path / "workspaces", cas_root=tmp_path / "cas")
    captured: dict[str, object] = {}

    class FakeSandbox:
        def __init__(self, *, node_id: str, config: Any, fs: Any, cas: Any) -> None:
            captured["node_id"] = node_id
            captured["config"] = config
            captured["fs"] = fs
            captured["cas"] = cas

    monkeypatch.setattr("transformation_portal.runtime.sandbox.Sandbox", FakeSandbox)

    fs = object()
    cas = object()
    sandbox = create_tenant_sandbox(tenant, "node_001", fs, cas)

    assert isinstance(sandbox, FakeSandbox)
    config = captured["config"]
    assert config.workspace_root == tenant.tenant_workspace
    assert config.cas_root == tenant.tenant_cas
    assert captured["node_id"] == "node_001"
    assert captured["fs"] is fs
    assert captured["cas"] is cas


@pytest.mark.parametrize("escape", ["parent", "symlink", "tenant_root_symlink"])
def test_tenant_guard_rejects_resolved_cross_tenant_paths(tmp_path: Path, escape: str) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    other = manager.create_tenant("tenant_b")
    private_file = other.tenant_workspace / "private.txt"
    private_file.write_text("other tenant", encoding="utf-8")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)

    if escape == "parent":
        candidate = tenant.tenant_workspace / ".." / other.tenant_id / private_file.name
    elif escape == "symlink":
        alias = tenant.tenant_workspace / "alias"
        alias.symlink_to(other.tenant_workspace, target_is_directory=True)
        candidate = alias / private_file.name
    else:
        tenant.tenant_workspace.rmdir()
        tenant.tenant_workspace.symlink_to(other.tenant_workspace, target_is_directory=True)
        candidate = tenant.tenant_workspace / private_file.name

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        guard.read_text(candidate)
    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        guard.write_text(candidate, "replaced")
    assert private_file.read_text(encoding="utf-8") == "other tenant"


@pytest.mark.parametrize(
    "operation",
    [
        "read_bytes",
        "write_bytes",
        "delete",
        "exists",
        "mkdir",
        "list_dir",
        "copy_source",
        "copy_destination",
        "symlink_source",
        "symlink_destination",
        "symlink_relative_source",
    ],
)
def test_tenant_guard_rejects_cross_tenant_io_before_side_effects(tmp_path: Path, operation: str) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    other = manager.create_tenant("tenant_b")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    private_file = other.tenant_workspace / "private.bin"
    private_file.write_bytes(b"other tenant")
    own_file = tenant.tenant_workspace / "own.bin"
    own_file.write_bytes(b"own tenant")
    destination = tenant.tenant_workspace / "new.bin"
    other_destination = other.tenant_workspace / "new.bin"

    operations = {
        "read_bytes": lambda: guard.read_bytes(private_file),
        "write_bytes": lambda: guard.write_bytes(private_file, b"replaced"),
        "delete": lambda: guard.delete(private_file),
        "exists": lambda: guard.exists(private_file),
        "mkdir": lambda: guard.mkdir(other_destination / "nested"),
        "list_dir": lambda: guard.list_dir(other.tenant_workspace),
        "copy_source": lambda: guard.copy(private_file, destination),
        "copy_destination": lambda: guard.copy(own_file, other_destination),
        "symlink_source": lambda: guard.symlink(private_file, destination),
        "symlink_destination": lambda: guard.symlink(own_file, other_destination),
        "symlink_relative_source": lambda: guard.symlink(Path("..") / other.tenant_id / private_file.name, destination),
    }
    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        operations[operation]()

    assert private_file.read_bytes() == b"other tenant"
    assert own_file.read_bytes() == b"own tenant"
    assert not destination.exists() and not destination.is_symlink()
    assert not other_destination.exists() and not other_destination.is_symlink()


@pytest.mark.parametrize("binary", [False, True])
def test_tenant_guard_rejects_atomic_staging_symlink(tmp_path: Path, binary: bool) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    other = manager.create_tenant("tenant_b")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    private_file = other.tenant_workspace / "private.bin"
    private_file.write_bytes(b"other tenant")
    destination = tenant.tenant_workspace / "result.bin"
    staging_path = destination.with_suffix(destination.suffix + ".tmp")
    staging_path.symlink_to(private_file)

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        if binary:
            guard.write_bytes(destination, b"replaced")
        else:
            guard.write_text(destination, "replaced")

    assert private_file.read_bytes() == b"other tenant"
    assert not destination.exists()
    assert staging_path.is_symlink()


def test_tenant_guard_allows_same_tenant_io_and_relative_symlinks(tmp_path: Path) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    guard.mkdir(tenant.tenant_workspace)
    assert guard.exists(tenant.tenant_workspace)
    assert guard.list_dir(tenant.tenant_cas) == []
    nested = tenant.tenant_workspace / "nested"
    guard.mkdir(nested)
    payload = nested / "payload.bin"
    guard.write_bytes(payload, b"first")
    guard.write_bytes(payload, b"payload", atomic=False)
    assert guard.read_bytes(payload) == b"payload"
    assert guard.exists(payload)
    assert guard.list_dir(nested) == [payload]

    link = nested / "relative.bin"
    guard.symlink(Path(payload.name), link)
    assert link.readlink() == Path(payload.name)
    assert guard.read_text(link) == "payload"
    copied = tenant.tenant_cas / "copied.bin"
    guard.copy(link, copied)
    assert guard.read_bytes(copied) == b"payload"
    guard.copy(payload, tenant.tenant_cas)
    assert guard.read_bytes(tenant.tenant_cas / payload.name) == b"payload"
    assert guard.delete(link)
    assert payload.exists()


def test_tenant_guard_rejects_copy_directory_child_symlink(tmp_path: Path) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    other = manager.create_tenant("tenant_b")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    source = tenant.tenant_workspace / "payload.bin"
    source.write_bytes(b"own tenant")
    private_file = other.tenant_workspace / "private.bin"
    private_file.write_bytes(b"other tenant")
    target = tenant.tenant_cas / source.name
    target.symlink_to(private_file)

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        guard.copy(source, tenant.tenant_cas)

    assert private_file.read_bytes() == b"other tenant"
    assert target.is_symlink()


@pytest.mark.parametrize("operation", ["write_text", "write_bytes", "copy", "symlink", "mkdir"])
def test_tenant_guard_rejects_parent_paths_before_creating_external_directories(tmp_path: Path, operation: str) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    source = tenant.tenant_workspace / "source.bin"
    source.write_bytes(b"own tenant")
    external_dir = manager.workspace_root / "unrelated"
    destination = tenant.tenant_workspace / ".." / external_dir.name / ".." / tenant.tenant_id / "new.bin"
    operations = {
        "write_text": lambda: guard.write_text(destination, "payload"),
        "write_bytes": lambda: guard.write_bytes(destination, b"payload"),
        "copy": lambda: guard.copy(source, destination),
        "symlink": lambda: guard.symlink(source, destination),
        "mkdir": lambda: guard.mkdir(destination),
    }

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        operations[operation]()

    assert not external_dir.exists()
    assert not (tenant.tenant_workspace / "new.bin").exists()
    assert not (tenant.tenant_workspace / "new.bin").is_symlink()
    assert source.read_bytes() == b"own tenant"


def test_tenant_guard_rejects_parent_relative_symlink_target(tmp_path: Path) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    nested = tenant.tenant_workspace / "nested"
    nested.mkdir()
    source = tenant.tenant_workspace / "payload.bin"
    source.write_bytes(b"own tenant")
    link = nested / "link.bin"

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        guard.symlink(Path("..") / source.name, link)

    assert not link.is_symlink()
    assert source.read_bytes() == b"own tenant"


@pytest.mark.parametrize("via_alias", [False, True])
@pytest.mark.parametrize("operation", ["delete", "symlink", "copy_destination", "write_text", "write_bytes"])
def test_tenant_guard_preserves_foreign_symlink_pointing_into_own_tenant(
    tmp_path: Path, operation: str, via_alias: bool
) -> None:
    manager = TenantManager(tmp_path / "workspaces", tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    other = manager.create_tenant("tenant_b")
    guard = TenantAwareFSGuard(manager)
    guard.set_tenant(tenant)
    own_file = tenant.tenant_workspace / "owned.bin"
    own_file.write_bytes(b"own tenant")
    source = tenant.tenant_workspace / "source.bin"
    source.write_bytes(b"source")
    foreign_link = other.tenant_workspace / "foreign.bin"
    foreign_link.symlink_to(own_file)
    original_inode = foreign_link.lstat().st_ino
    candidate = foreign_link
    if via_alias:
        alias = tenant.tenant_workspace / "alias"
        alias.symlink_to(other.tenant_workspace, target_is_directory=True)
        candidate = alias / foreign_link.name
    operations = {
        "delete": lambda: guard.delete(candidate),
        "symlink": lambda: guard.symlink(source, candidate),
        "copy_destination": lambda: guard.copy(source, candidate),
        "write_text": lambda: guard.write_text(candidate, "replaced"),
        "write_bytes": lambda: guard.write_bytes(candidate, b"replaced"),
    }

    with pytest.raises(TenantError, match="Cross-tenant access denied"):
        operations[operation]()

    assert foreign_link.is_symlink()
    assert foreign_link.readlink() == own_file
    assert foreign_link.lstat().st_ino == original_inode
    assert own_file.read_bytes() == b"own tenant"


def test_tenant_guard_preserves_configured_base_aliases_and_unset_tenant_behavior(tmp_path: Path) -> None:
    real_workspace = tmp_path / "real-workspaces"
    real_workspace.mkdir()
    workspace_alias = tmp_path / "workspaces"
    workspace_alias.symlink_to(real_workspace, target_is_directory=True)
    manager = TenantManager(workspace_alias, tmp_path / "cas")
    tenant = manager.create_tenant("tenant_a")
    guard = TenantAwareFSGuard(manager)
    outside = tmp_path / "unscoped.bin"
    guard.write_bytes(outside, b"legacy unscoped")
    assert guard.read_bytes(outside) == b"legacy unscoped"

    guard.set_tenant(tenant)
    payload = tenant.tenant_workspace / "payload.txt"
    guard.write_text(payload, "scoped")
    assert guard.read_text(real_workspace / tenant.tenant_id / payload.name) == "scoped"
