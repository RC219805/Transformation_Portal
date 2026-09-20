"""Worker-host path and runtime authorization, independent of the HTTP app."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from transformation_portal.core.security.tenant import TenantAwareFSGuard, TenantContext, TenantManager, TenantPolicy
from transformation_portal.orchestrator.dispatch import DispatchLocator
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost
from transformation_portal.portal import path_security
from transformation_portal.vlm_captioning.fastvlm_runtime import FASTVLM_CHECKPOINT_DIRS, default_fastvlm_runtime_root


@dataclass(frozen=True)
class ExecutionPolicy:
    repo_root: Path
    input_roots: tuple[Path, ...]
    output_roots: tuple[Path, ...]
    fastvlm_roots: tuple[Path, ...]
    sam2_roots: tuple[Path, ...]
    caption_roles: frozenset[str]
    fastvlm_runtime_root: Path
    pilot_enabled: bool = False
    allowed_tenants: frozenset[str] = frozenset()
    allowed_pipelines: frozenset[str] = frozenset({"lux-depth-v3"})
    manager: TenantManager | None = None

    @property
    def path_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys((*self.input_roots, *self.output_roots)))

    @property
    def archive_policy_root(self) -> Path:
        return self.repo_root / "policy" / "archive"

    def resolve_path(self, value: str, roots: Sequence[Path]) -> Path:
        return path_security._resolve_allowed_request_path(value, list(roots), repo_root=self.repo_root)

    def resolve_output_root(self, value: str) -> Path:
        return self.resolve_path(value, self.output_roots)

    def tenant_guard(self, tenant: TenantContext) -> TenantAwareFSGuard:
        if self.manager is None:
            raise DispatchAuthorityLost("tenant manager is unavailable")
        guard = TenantAwareFSGuard(self.manager)
        guard.set_tenant(tenant)
        return guard

    def validate_dispatch_paths(
        self, locator: DispatchLocator, plan_bytes: bytes, output_root: Path, *, execution_bindings: bytes | None = None
    ) -> None:
        if execution_bindings is not None:
            self._validate_photography_paths(locator, plan_bytes, output_root, execution_bindings)
            return
        from transformation_portal.core.archive_execution_plan import ARCHIVE_OPERATIONS
        from transformation_portal.core.execution_plan import parse_execution_plan_json
        from transformation_portal.stage_graph.registry import StageRegistryIdentifier

        plan = parse_execution_plan_json(plan_bytes)
        input_root = self.resolve_path(plan.input_root, self.input_roots)
        output_root = self.resolve_path(str(output_root), self.output_roots)
        paths: list[tuple[Path, Sequence[Path]]] = [(input_root, []), (output_root, [])]
        pipeline = "lux-depth-v3"
        if plan.planned_backend == "archive":
            configuration = plan.to_payload()["nodes"][0]["configuration"]
            operation = ARCHIVE_OPERATIONS[configuration["operation"]]
            pipeline = operation.pipeline
            for name in operation.files + operation.directories:
                value = configuration["parameters"].get(name)
                if value is not None:
                    is_rights_policy = configuration["operation"] == "rights-apply" and name == "policy_yaml"
                    shared_roots = [self.archive_policy_root] if is_rights_policy else []
                    allowed_roots = self.input_roots if is_rights_policy else self.path_roots
                    paths.append((self.resolve_path(value, allowed_roots), shared_roots))
        else:
            from transformation_portal.lux_depth_v3 import config_resolver
            from transformation_portal.lux_depth_v3.config import EnhanceConfig

            # Only the immutable plan supplies execution paths. Default config is
            # used solely to read this worker's currently selected server runtimes.
            defaults = EnhanceConfig()
            interpreter_resolvers = {
                "raw_python_executable": config_resolver.resolve_prepared_raw_python_executable,
                "da3_python_executable": config_resolver.resolve_prepared_da3_python_executable,
                "depth_pro_python_executable": config_resolver.resolve_prepared_depth_pro_python_executable,
            }
            for node in plan.nodes:
                configuration = node.configuration
                for field, resolver in interpreter_resolvers.items():
                    carried = configuration.get(field)
                    if carried is not None:
                        current = resolver(defaults, preparation_cwd=self.repo_root)
                        # Do not dereference Python symlinks: two virtualenvs may
                        # share a binary while authorizing different packages.
                        if carried != current:
                            raise DispatchAuthorityLost(f"worker runtime selection changed: {field}")
                if node.stage_registry_id == StageRegistryIdentifier.LUX_DEPTH:
                    checkpoint = configuration.get("depth_pro_checkpoint_path")
                    if checkpoint is not None and "depth_pro" in plan.candidate_fallback_chain:
                        current_checkpoint = config_resolver.resolve_prepared_depth_pro_checkpoint_path(
                            defaults, preparation_cwd=self.repo_root
                        )
                        if checkpoint != current_checkpoint:
                            paths.append((self.resolve_path(checkpoint, self.input_roots), [self.repo_root / "checkpoints"]))
                elif node.stage_registry_id == StageRegistryIdentifier.LUX_MATERIALS_V3:
                    for field in ("sam2_checkpoint_path", "sam_vit_h_checkpoint_path"):
                        checkpoint = configuration.get(field)
                        if checkpoint is not None:
                            roots = self.sam2_roots if field == "sam2_checkpoint_path" else [self.repo_root / "checkpoints"]
                            paths.append((self.resolve_path(checkpoint, self.input_roots), roots))
                elif node.stage_registry_id == StageRegistryIdentifier.LUX_RECONSTRUCTION:
                    sidecar = configuration.get("cameras_sidecar_path")
                    if sidecar is not None:
                        paths.append((self.resolve_path(sidecar, self.input_roots), []))
                elif node.stage_registry_id == StageRegistryIdentifier.LUX_OUTPUT:
                    captioning = configuration.get("captioning")
                    if isinstance(captioning, Mapping) and captioning.get("enabled"):
                        selectors = [
                            captioning.get(field)
                            for field in ("model_path", "review_model_path", "python_executable", "mlx_vlm_dir")
                        ]
                        selector = captioning.get("selector")
                        if isinstance(selector, str) and selector.lower() not in self.caption_roles:
                            selectors.append(selector)
                        for value in selectors:
                            if value is not None:
                                paths.append(
                                    (
                                        self.resolve_path(value, self.fastvlm_roots),
                                        [self.fastvlm_runtime_root],
                                    )
                                )
        if self.pilot_enabled:
            if (
                self.allowed_tenants and locator.tenant_id not in self.allowed_tenants
            ) or pipeline not in self.allowed_pipelines:
                raise DispatchAuthorityLost("dispatch tenant or pipeline is no longer allowed")
            manager = self.manager
            if manager is None:
                raise DispatchAuthorityLost("tenant manager is unavailable")
            tenant = manager.get_tenant(locator.tenant_id)
            if tenant is None:
                tenant = manager.create_tenant(
                    locator.tenant_id, policy=TenantPolicy(allowed_node_types=set(self.allowed_pipelines))
                )
            guard = self.tenant_guard(tenant)
            for path, path_shared_roots in paths:
                if not any(path.is_relative_to(root.resolve()) for root in path_shared_roots):
                    guard.enforce_path(path)

    def _validate_photography_paths(
        self, locator: DispatchLocator, plan_bytes: bytes, output_root: Path, bindings_bytes: bytes
    ) -> None:
        from transformation_portal.orchestrator.photography_adapter import (
            PhotographyBindings,
            server_runtime_bindings,
            validate_photography_dispatch,
        )

        if not managed_photography_enabled():
            raise DispatchAuthorityLost("managed photography is disabled on this worker")
        validate_photography_dispatch(plan_bytes, bindings_bytes)
        bound = PhotographyBindings(bindings_bytes).to_payload()
        data_paths = [Path(bound["input_root"]), output_root]
        data_paths.extend(Path(bound[name]) for name in ("companion_root", "materials_root") if bound[name] is not None)
        # Reject another tenant's lexical namespace before resolving any of it.
        if self.pilot_enabled:
            if (
                self.allowed_tenants and locator.tenant_id not in self.allowed_tenants
            ) or "lux-depth-v5" not in self.allowed_pipelines:
                raise DispatchAuthorityLost("dispatch tenant or pipeline is no longer allowed")
            if self.manager is None:
                raise DispatchAuthorityLost("tenant manager is unavailable")
            tenant = self.manager.get_tenant(locator.tenant_id)
            if tenant is None:
                tenant = self.manager.create_tenant(
                    locator.tenant_id, policy=TenantPolicy(allowed_node_types=set(self.allowed_pipelines))
                )
            roots = tuple(
                base / locator.tenant_id
                for configured in (tenant.workspace_root, tenant.cas_root)
                for base in (configured.absolute(), configured.resolve())
            )
            for path in data_paths:
                if ".." in path.parts or not any(path.absolute().is_relative_to(root) for root in roots):
                    raise DispatchAuthorityLost("photography data path is outside the dispatch tenant")
            guard = self.tenant_guard(tenant)
            for path in data_paths:
                guard.enforce_path(path)
        if self.resolve_output_root(str(output_root)) != output_root:
            raise DispatchAuthorityLost("photography output binding is no longer canonical")
        for path in data_paths[:1] + data_paths[2:]:
            if self.resolve_path(str(path), self.input_roots) != path:
                raise DispatchAuthorityLost("photography data binding is not canonical")
        if (bound["runtime_python"], bound["raw_python"]) != server_runtime_bindings():
            raise DispatchAuthorityLost("photography runtime selection changed")
        cache = server_photography_cache_root(locator.tenant_id)
        if bound["cache_root"] != (None if cache is None else str(cache)):
            raise DispatchAuthorityLost("photography cache selection changed")


def managed_photography_enabled() -> bool:
    return os.getenv("TP_LUX_V5_MANAGED_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def server_photography_cache_root(tenant_id: str) -> Path | None:
    """A configured service-owned cache is namespaced by admitted tenant."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tenant_id):
        raise ValueError("invalid cache tenant")
    raw = os.environ.get("TP_LUX_V5_CACHE_DIR", "").strip()
    if not raw:
        return None
    return Path(os.path.abspath(os.path.expanduser(raw))) / tenant_id


def load_execution_policy() -> ExecutionPolicy:
    """Load the same server-owned roots and tenant policy without importing app."""
    root = Path(__file__).resolve().parents[3]
    logger = logging.getLogger(__name__)
    defaults = path_security._default_allowed_path_roots(repo_root=root)
    inputs = tuple(path_security._env_path_roots("TP_ALLOWED_INPUT_ROOTS", defaults, repo_root=root, logger=logger))
    outputs = tuple(path_security._env_path_roots("TP_ALLOWED_OUTPUT_ROOTS", defaults, repo_root=root, logger=logger))
    enabled = os.getenv("TP_PILOT_CONTROL_PLANE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    tenants = frozenset(item.strip() for item in os.getenv("TP_PILOT_ALLOWED_TENANTS", "").split(",") if item.strip())
    pipelines = frozenset(
        item.strip() for item in os.getenv("TP_PILOT_ALLOWED_PIPELINES", "lux-depth-v3").split(",") if item.strip()
    )
    manager = None
    if enabled:
        base = Path(tempfile.gettempdir()) / "tp-pilot-tenants"
        manager = TenantManager(
            workspace_root=Path(os.getenv("TP_PILOT_TENANT_WORKSPACE_ROOT", str(base / "workspaces"))),
            cas_root=Path(os.getenv("TP_PILOT_TENANT_CAS_ROOT", str(base / "cas"))),
            default_policy=TenantPolicy(allowed_node_types=set(pipelines)),
        )
    return ExecutionPolicy(
        repo_root=root,
        input_roots=inputs,
        output_roots=outputs,
        fastvlm_roots=tuple(dict.fromkeys((*inputs, Path(os.path.realpath(default_fastvlm_runtime_root()))))),
        sam2_roots=(Path(os.path.realpath(root / "models" / "sam2")), Path(os.path.realpath(root / "checkpoints"))),
        caption_roles=frozenset(FASTVLM_CHECKPOINT_DIRS),
        fastvlm_runtime_root=default_fastvlm_runtime_root(),
        pilot_enabled=enabled,
        allowed_tenants=tenants,
        allowed_pipelines=pipelines,
        manager=manager,
    )
