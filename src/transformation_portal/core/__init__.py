"""
Platform Core Module

Unified infrastructure for all Transformation Portal pipelines.
Eliminates duplication across lux_depth_v2, luxury_video_master_grader,
and other pipelines while providing a clean, maintainable foundation.

This package exposes a broad surface area, but many exports pull in optional
runtime dependencies such as torch-backed stage-graph machinery. Keep the
package import lightweight so submodule imports like
`transformation_portal.core.security.path` do not eagerly load the full stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Config
    "ConfigSchema": (".config", "ConfigSchema"),
    "DeviceConfig": (".config", "DeviceConfig"),
    "PathsConfig": (".config", "PathsConfig"),
    "PerformanceConfig": (".config", "PerformanceConfig"),
    "PresetRegistry": (".config", "PresetRegistry"),
    "load_preset": (".config", "load_preset"),
    "validate_config": (".config", "validate_config"),
    # Device
    "DeviceDetector": (".device", "DeviceDetector"),
    "DeviceCapabilities": (".device", "DeviceCapabilities"),
    "DeviceType": (".device", "DeviceType"),
    "PerformanceProfiler": (".device", "PerformanceProfiler"),
    "MemoryManager": (".device", "MemoryManager"),
    # Security
    "InputValidator": (".security", "InputValidator"),
    "PathValidator": (".security", "PathValidator"),
    "SanitizationPolicy": (".security", "SanitizationPolicy"),
    "validate_input_file": (".security", "validate_input_file"),
    "safe_resolve_path": (".security", "safe_resolve_path"),
    # Platform Matrix (ADR-032)
    "CURRENT_PLATFORM": (".platform_matrix", "CURRENT_PLATFORM"),
    "PlatformAccel": (".platform_matrix", "PlatformAccel"),
    "PlatformISA": (".platform_matrix", "PlatformISA"),
    "PlatformMatrix": (".platform_matrix", "PlatformMatrix"),
    "PlatformOS": (".platform_matrix", "PlatformOS"),
    "compute_cas_identity": (".platform_matrix", "compute_cas_identity"),
    "compute_lockfile_hash": (".platform_matrix", "compute_lockfile_hash"),
    "determine_ml_core_lockfile_name": (".platform_matrix", "determine_ml_core_lockfile_name"),
    "get_env_fingerprint": (".platform_matrix", "get_env_fingerprint"),
    "get_pip_version": (".platform_matrix", "get_pip_version"),
    "get_platform_fingerprint": (".platform_matrix", "get_platform_fingerprint"),
    # Execution Identity (Phase 2)
    "ALLOW_CROSS_PLATFORM": (".execution_identity", "ALLOW_CROSS_PLATFORM"),
    "ArtifactMetadata": (".execution_identity", "ArtifactMetadata"),
    "CAS_IDENTITY_VERSION": (".execution_identity", "CAS_IDENTITY_VERSION"),
    "DeterminismViolationError": (".execution_identity", "DeterminismViolationError"),
    "ExecutionIdentity": (".execution_identity", "ExecutionIdentity"),
    "LOCKFILE_HASH_PLACEHOLDER": (".execution_identity", "LOCKFILE_HASH_PLACEHOLDER"),
    "compute_cas_id": (".execution_identity", "compute_cas_id"),
    "compute_code_hash": (".execution_identity", "compute_code_hash"),
    "compute_config_hash": (".execution_identity", "compute_config_hash"),
    "compute_stage_code_hash": (".execution_identity", "compute_stage_code_hash"),
    "create_artifact_metadata": (".execution_identity", "create_artifact_metadata"),
    "explain_cache_miss": (".execution_identity", "explain_cache_miss"),
    "is_ci_environment": (".execution_identity", "is_ci_environment"),
    "is_compatible": (".execution_identity", "is_compatible"),
    "log_cache_decision": (".execution_identity", "log_cache_decision"),
    "resolve_platform_lockfile": (".execution_identity", "resolve_platform_lockfile"),
    "should_execute": (".execution_identity", "should_execute"),
    "verify_determinism": (".execution_identity", "verify_determinism"),
    # Execution Identity v3 (ADR-051)
    "BackendRuntimeIdentity": (".execution_identity_v3", "BackendRuntimeIdentity"),
    "EXECUTION_IDENTITY_V3_INCOMPLETE": (
        ".execution_identity_v3",
        "EXECUTION_IDENTITY_V3_INCOMPLETE",
    ),
    "EXECUTION_IDENTITY_V3_SCHEMA": (".execution_identity_v3", "EXECUTION_IDENTITY_V3_SCHEMA"),
    "EXECUTION_IDENTITY_V3_MATERIALIZED": (
        ".execution_identity_v3",
        "EXECUTION_IDENTITY_V3_MATERIALIZED",
    ),
    "ExecutionIdentityV3": (".execution_identity_v3", "ExecutionIdentityV3"),
    "ExecutionIdentityV3SeedError": (".execution_identity_v3", "ExecutionIdentityV3SeedError"),
    "IncompleteExecutionIdentityV3Error": (
        ".execution_identity_v3",
        "IncompleteExecutionIdentityV3Error",
    ),
    "MaterializedExecutionIdentityV3": (
        ".execution_identity_v3",
        "MaterializedExecutionIdentityV3",
    ),
    "MaterializedExecutionIdentityV3Error": (
        ".execution_identity_v3",
        "MaterializedExecutionIdentityV3Error",
    ),
    # Canonical execution plan (ADR-051, non-activating contract)
    "BackendCandidateIntent": (".execution_plan", "BackendCandidateIntent"),
    "BackendModelIntent": (".execution_plan", "BackendModelIntent"),
    "CanonicalExecutionPlan": (".execution_plan", "CanonicalExecutionPlan"),
    "EXECUTION_COMPLETE": (".execution_plan", "EXECUTION_COMPLETE"),
    "EXECUTION_PLAN_SCHEMA": (".execution_plan", "EXECUTION_PLAN_SCHEMA"),
    "ExecutionPlanError": (".execution_plan", "ExecutionPlanError"),
    "load_execution_plan_schema": (".execution_plan", "load_execution_plan_schema"),
    "parse_execution_plan_json": (".execution_plan", "parse_execution_plan_json"),
    "STRUCTURAL_LEGACY": (".execution_plan", "STRUCTURAL_LEGACY"),
    "validate_execution_plan_payload": (".execution_plan", "validate_execution_plan_payload"),
    # Execution Wrapper (Phase 2)
    "CacheResult": (".execution_wrapper", "CacheResult"),
    "CASExecutor": (".execution_wrapper", "CASExecutor"),
    "ExecutorConfig": (".execution_wrapper", "ExecutorConfig"),
    "FileLock": (".execution_wrapper", "FileLock"),
    "execute_with_caching": (".execution_wrapper", "execute_with_caching"),
    # CAS DAG Executor (Phase 2)
    "CASDAGConfig": (".cas_dag_executor", "CASDAGConfig"),
    "CASDAGExecutor": (".cas_dag_executor", "CASDAGExecutor"),
    "CASExecutionResult": (".cas_dag_executor", "CASExecutionResult"),
    "verify_dag_determinism": (".cas_dag_executor", "verify_dag_determinism"),
}

__all__ = [
    # Config
    "ConfigSchema",
    "DeviceConfig",
    "PathsConfig",
    "PerformanceConfig",
    "PresetRegistry",
    "load_preset",
    "validate_config",
    # Device
    "DeviceDetector",
    "DeviceCapabilities",
    "DeviceType",
    "PerformanceProfiler",
    "MemoryManager",
    # Security
    "InputValidator",
    "PathValidator",
    "SanitizationPolicy",
    "validate_input_file",
    "safe_resolve_path",
    # Platform Matrix (ADR-032)
    "CURRENT_PLATFORM",
    "PlatformAccel",
    "PlatformISA",
    "PlatformMatrix",
    "PlatformOS",
    "compute_cas_identity",
    "compute_lockfile_hash",
    "determine_ml_core_lockfile_name",
    "get_env_fingerprint",
    "get_pip_version",
    "get_platform_fingerprint",
    # Execution Identity (Phase 2)
    "ALLOW_CROSS_PLATFORM",
    "ArtifactMetadata",
    "CAS_IDENTITY_VERSION",
    "DeterminismViolationError",
    "ExecutionIdentity",
    "LOCKFILE_HASH_PLACEHOLDER",
    "compute_cas_id",
    "compute_code_hash",
    "compute_config_hash",
    "compute_stage_code_hash",
    "create_artifact_metadata",
    "explain_cache_miss",
    "is_ci_environment",
    "is_compatible",
    "log_cache_decision",
    "resolve_platform_lockfile",
    "should_execute",
    "verify_determinism",
    # Execution Identity v3
    "BackendRuntimeIdentity",
    "EXECUTION_IDENTITY_V3_INCOMPLETE",
    "EXECUTION_IDENTITY_V3_MATERIALIZED",
    "EXECUTION_IDENTITY_V3_SCHEMA",
    "ExecutionIdentityV3",
    "ExecutionIdentityV3SeedError",
    "IncompleteExecutionIdentityV3Error",
    "MaterializedExecutionIdentityV3",
    "MaterializedExecutionIdentityV3Error",
    # Canonical execution plan (named explicitly to avoid collision with the
    # retained Lux flat pipeline_coordinator.ExecutionPlan compatibility type)
    "BackendCandidateIntent",
    "BackendModelIntent",
    "CanonicalExecutionPlan",
    "EXECUTION_COMPLETE",
    "EXECUTION_PLAN_SCHEMA",
    "ExecutionPlanError",
    "load_execution_plan_schema",
    "parse_execution_plan_json",
    "STRUCTURAL_LEGACY",
    "validate_execution_plan_payload",
    # Execution Wrapper (Phase 2)
    "CacheResult",
    "CASExecutor",
    "ExecutorConfig",
    "FileLock",
    "execute_with_caching",
    # CAS DAG Executor (Phase 2)
    "CASDAGConfig",
    "CASDAGExecutor",
    "CASExecutionResult",
    "verify_dag_determinism",
]

__version__ = "1.1.0"


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
