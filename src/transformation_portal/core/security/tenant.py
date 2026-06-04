"""Multi-tenant isolation for secure resource partitioning.

This module provides tenant-based isolation ensuring:
- Separate CAS namespaces per tenant
- Separate workspace roots per tenant
- Policy enforcement preventing cross-tenant access
- Tenant-scoped credentials

Example:
    >>> # Create tenant context
    >>> tenant = TenantContext(
    ...     tenant_id="customer_001",
    ...     workspace_root=Path("/data/workspaces"),
    ...     cas_root=Path("/data/cas"),
    ...     public_key="...",
    ... )
    >>>
    >>> # Create tenant-isolated sandbox
    >>> sandbox = create_tenant_sandbox(tenant, node_id, fs, cas)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from transformation_portal.core.security.fs_guard import (
    FSContext,
    FSGuard,
    FSPolicyError,
)
from transformation_portal.core.security.path_safety import (
    PathSafetyError,
    validate_safe_name,
)

logger = logging.getLogger(__name__)


class TenantError(RuntimeError):
    """Raised for tenant isolation violations."""


@dataclass(frozen=True)
class TenantContext:
    """Context for a tenant's isolated environment.

    Attributes:
        tenant_id: Unique tenant identifier
        workspace_root: Base path for tenant workspaces
        cas_root: Base path for tenant CAS storage
        public_key: Tenant's public key (base64)
        metadata: Additional tenant metadata
    """

    tenant_id: str
    workspace_root: Path
    cas_root: Path
    public_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tenant ID."""
        try:
            validate_safe_name(self.tenant_id)
        except PathSafetyError as e:
            raise TenantError(f"Invalid tenant_id: {e}")

    @property
    def tenant_workspace(self) -> Path:
        """Get tenant-specific workspace root."""
        return self.workspace_root / self.tenant_id

    @property
    def tenant_cas(self) -> Path:
        """Get tenant-specific CAS root."""
        return self.cas_root / self.tenant_id


@dataclass
class TenantPolicy:
    """Security policy for a tenant.

    Attributes:
        max_workspace_size_mb: Maximum workspace size
        max_cas_objects: Maximum CAS objects
        allowed_node_types: Set of allowed node types
        gpu_quota: Maximum GPU slots
        network_allowed: Whether network access is allowed
    """

    max_workspace_size_mb: int = 0  # 0 = unlimited
    max_cas_objects: int = 0  # 0 = unlimited
    allowed_node_types: Set[str] = field(default_factory=set)
    gpu_quota: int = 0  # 0 = unlimited
    network_allowed: bool = False


class TenantManager:
    """Manages tenant contexts and policies.

    Provides:
    - Tenant registration and lookup
    - Policy enforcement
    - Cross-tenant isolation

    Example:
        >>> manager = TenantManager(
        ...     workspace_root=Path("/data/workspaces"),
        ...     cas_root=Path("/data/cas"),
        ... )
        >>>
        >>> tenant = manager.create_tenant("customer_001")
        >>> sandbox = manager.create_sandbox(tenant, "node_001", fs, cas)
    """

    def __init__(
        self,
        workspace_root: Path,
        cas_root: Path,
        *,
        default_policy: Optional[TenantPolicy] = None,
    ) -> None:
        """Initialize tenant manager.

        Args:
            workspace_root: Base workspace directory
            cas_root: Base CAS directory
            default_policy: Default policy for new tenants
        """
        self.workspace_root = workspace_root
        self.cas_root = cas_root
        self.default_policy = default_policy or TenantPolicy()

        self._tenants: Dict[str, TenantContext] = {}
        self._policies: Dict[str, TenantPolicy] = {}

        # Ensure base directories exist
        workspace_root.mkdir(parents=True, exist_ok=True)
        cas_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "TenantManager initialized: workspace=%s, cas=%s",
            workspace_root,
            cas_root,
        )

    def create_tenant(
        self,
        tenant_id: str,
        *,
        public_key: Optional[str] = None,
        policy: Optional[TenantPolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TenantContext:
        """Create a new tenant.

        Args:
            tenant_id: Unique tenant identifier
            public_key: Tenant's public key
            policy: Tenant policy (uses default if not provided)
            metadata: Additional metadata

        Returns:
            TenantContext

        Raises:
            TenantError: If tenant already exists or ID is invalid
        """
        if tenant_id in self._tenants:
            raise TenantError(f"Tenant already exists: {tenant_id}")

        tenant = TenantContext(
            tenant_id=tenant_id,
            workspace_root=self.workspace_root,
            cas_root=self.cas_root,
            public_key=public_key,
            metadata=metadata or {},
        )

        # Create tenant directories
        tenant.tenant_workspace.mkdir(parents=True, exist_ok=True)
        tenant.tenant_cas.mkdir(parents=True, exist_ok=True)

        self._tenants[tenant_id] = tenant
        self._policies[tenant_id] = policy or copy.deepcopy(self.default_policy)

        logger.info("Created tenant: %s", tenant_id)
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[TenantContext]:
        """Get a tenant by ID.

        Args:
            tenant_id: Tenant identifier

        Returns:
            TenantContext or None
        """
        return self._tenants.get(tenant_id)

    def get_policy(self, tenant_id: str) -> Optional[TenantPolicy]:
        """Get tenant policy.

        Args:
            tenant_id: Tenant identifier

        Returns:
            TenantPolicy or None
        """
        return self._policies.get(tenant_id)

    def delete_tenant(
        self,
        tenant_id: str,
        *,
        delete_data: bool = False,
    ) -> bool:
        """Delete a tenant.

        Args:
            tenant_id: Tenant identifier
            delete_data: If True, delete tenant data

        Returns:
            True if deleted
        """
        tenant = self._tenants.pop(tenant_id, None)
        self._policies.pop(tenant_id, None)

        if tenant and delete_data:
            import shutil

            if tenant.tenant_workspace.exists():
                shutil.rmtree(tenant.tenant_workspace)
            if tenant.tenant_cas.exists():
                shutil.rmtree(tenant.tenant_cas)

            logger.info("Deleted tenant and data: %s", tenant_id)
        elif tenant:
            logger.info("Deleted tenant (data preserved): %s", tenant_id)

        return tenant is not None

    def list_tenants(self) -> List[str]:
        """List all tenant IDs."""
        return list(self._tenants.keys())

    def enforce_tenant_access(
        self,
        tenant: TenantContext,
        path: Path,
    ) -> None:
        """Enforce tenant path access.

        Raises:
            TenantError: If path is outside tenant's scope

        Args:
            tenant: Tenant context
            path: Path to check
        """
        # Check workspace access
        try:
            path.relative_to(tenant.tenant_workspace)
            return
        except ValueError:
            pass

        # Check CAS access
        try:
            path.relative_to(tenant.tenant_cas)
            return
        except ValueError:
            pass

        raise TenantError(f"Cross-tenant access denied: {path} not in tenant {tenant.tenant_id}")

    def enforce_node_type(
        self,
        tenant_id: str,
        node_type: str,
    ) -> None:
        """Enforce allowed node types.

        Args:
            tenant_id: Tenant identifier
            node_type: Node type to check

        Raises:
            TenantError: If node type not allowed
        """
        policy = self._policies.get(tenant_id)
        if policy and policy.allowed_node_types:
            if node_type not in policy.allowed_node_types:
                raise TenantError(f"Node type not allowed for tenant {tenant_id}: {node_type}")


class TenantAwareFSGuard(FSGuard):
    """FSGuard with tenant isolation enforcement.

    Extends FSGuard to check tenant boundaries on all operations.

    Example:
        >>> fs = TenantAwareFSGuard(tenant_manager)
        >>> fs.set_tenant(tenant)
        >>>
        >>> # This will check tenant boundaries
        >>> path = fs.user_file(ctx, "myfile", suffix=".json")
    """

    def __init__(
        self,
        tenant_manager: TenantManager,
        **kwargs,
    ) -> None:
        """Initialize tenant-aware FSGuard.

        Args:
            tenant_manager: TenantManager instance
            **kwargs: Additional FSGuard arguments
        """
        super().__init__(**kwargs)
        self.tenant_manager = tenant_manager
        self._current_tenant: Optional[TenantContext] = None

    def set_tenant(self, tenant: TenantContext) -> None:
        """Set current tenant context.

        Args:
            tenant: Tenant context
        """
        self._current_tenant = tenant
        logger.debug("FSGuard tenant set: %s", tenant.tenant_id)

    def _enforce_tenant(self, path: Path) -> None:
        """Check path is within current tenant's scope."""
        if self._current_tenant:
            self.tenant_manager.enforce_tenant_access(
                self._current_tenant,
                path,
            )

    def enforce_path(self, path: Path) -> None:
        """Validate that a path stays within the current tenant's scope."""
        self._enforce_tenant(path)

    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        """Read with tenant check."""
        self._enforce_tenant(path)
        return super().read_text(path, encoding)

    def write_text(
        self,
        path: Path,
        data: str,
        encoding: str = "utf-8",
        atomic: bool = True,
    ) -> None:
        """Write with tenant check."""
        self._enforce_tenant(path)
        super().write_text(path, data, encoding, atomic)

    def delete(self, path: Path, missing_ok: bool = True) -> bool:
        """Delete with tenant check."""
        self._enforce_tenant(path)
        return super().delete(path, missing_ok)


def create_tenant_sandbox(
    tenant: TenantContext,
    node_id: str,
    fs: FSGuard,
    cas: "ArtifactStore",
) -> "Sandbox":
    """Create a sandbox isolated to a tenant.

    Args:
        tenant: Tenant context
        node_id: Node identifier
        fs: FSGuard instance
        cas: CAS store

    Returns:
        Tenant-isolated Sandbox
    """
    from transformation_portal.runtime.sandbox import Sandbox, SandboxConfig

    config = SandboxConfig(
        workspace_root=tenant.tenant_workspace,
        cas_root=tenant.tenant_cas,
    )

    return Sandbox(
        node_id=node_id,
        config=config,
        fs=fs,
        cas=cas,
    )


# Import for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformation_portal.runtime.sandbox import Sandbox
    from transformation_portal.storage.cas_store import ArtifactStore
