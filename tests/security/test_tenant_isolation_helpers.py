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
