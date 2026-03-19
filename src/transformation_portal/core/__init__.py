"""
Platform Core Module

Unified infrastructure for all Transformation Portal pipelines.
Eliminates duplication across lux_depth_v2, luxury_video_master_grader,
and other pipelines while providing a clean, maintainable foundation.

This module provides:
- Config schemas and preset management (config/)
- Device detection and optimization (device/)
- Artifact and cache management (storage/)
- Security validation and sanitization (security/)
- Observability integration (observability/)
- Platform matrix and CAS identity (platform_matrix.py)
- Execution identity for deterministic caching (execution_identity.py)
- CAS-aware execution wrapper (execution_wrapper.py)
- CAS-aware DAG executor (cas_dag_executor.py)

Architecture Goals:
- Zero breaking changes during migration
- Performance neutral or improved
- Clean, intuitive APIs
- Comprehensive test coverage
- Foundation for future stage graph
- Deterministic execution with CAS-aware caching (Phase 2)

Version: 1.1.0 (Phase 2 Deterministic Execution Layer)
"""

from .cas_dag_executor import (
    CASDAGConfig,
    CASDAGExecutor,
    CASExecutionResult,
    verify_dag_determinism,
)
from .config import ConfigSchema, DeviceConfig, PathsConfig, PerformanceConfig, PresetRegistry, load_preset, validate_config
from .device import DeviceCapabilities, DeviceDetector, DeviceType, MemoryManager, PerformanceProfiler
from .execution_identity import (
    ALLOW_CROSS_PLATFORM,
    CAS_IDENTITY_VERSION,
    LOCKFILE_HASH_PLACEHOLDER,
    ArtifactMetadata,
    DeterminismViolationError,
    ExecutionIdentity,
    compute_cas_id,
    compute_code_hash,
    compute_config_hash,
    compute_stage_code_hash,
    create_artifact_metadata,
    explain_cache_miss,
    is_ci_environment,
    is_compatible,
    log_cache_decision,
    resolve_platform_lockfile,
    should_execute,
    verify_determinism,
)
from .execution_wrapper import (
    CacheResult,
    CASExecutor,
    ExecutorConfig,
    FileLock,
    execute_with_caching,
)
from .platform_matrix import (
    CURRENT_PLATFORM,
    PlatformAccel,
    PlatformISA,
    PlatformMatrix,
    PlatformOS,
    compute_cas_identity,
    compute_lockfile_hash,
    get_env_fingerprint,
    get_pip_version,
    get_platform_fingerprint,
)
from .security import InputValidator, PathValidator, SanitizationPolicy, safe_resolve_path, validate_input_file

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
