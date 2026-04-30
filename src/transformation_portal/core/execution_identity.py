"""Unified CAS Identity Model for deterministic execution.

This module implements Phase 2 of the Deterministic Execution Layer (ADR-030/032).
Every stage execution becomes:

    Stage Execution = f(inputs, code, config, environment, lockfile)

The CAS identity materializes this function into a single content-addressable hash.

CAS_ID = sha256(
    stage_name +
    input_artifact_ids +
    code_hash +
    config_hash +
    env_fingerprint +
    platform_id +
    lockfile_hash  # ADR-032: Required for dependency graph integrity
)

Design Principles:
    - Deterministic: Same inputs always produce same CAS ID
    - Platform-aware: Different platforms produce different IDs
    - Code-aware: Code changes invalidate cache (AST-normalized)
    - Config-aware: Config changes invalidate cache
    - Dependency-aware: Lockfile changes invalidate cache (ADR-032)

Example:
    >>> from transformation_portal.core.execution_identity import compute_cas_id
    >>> cas_id = compute_cas_id(
    ...     stage_name="depth_estimation",
    ...     input_ids=["sha256:abc123..."],
    ...     config={"model": "DA3-Large"},
    ...     lockfile_path="requirements/ml-core-darwin-arm64.txt",
    ... )
    >>> print(cas_id)
    'sha256:def456...'
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

from transformation_portal.core.platform_matrix import (
    CURRENT_PLATFORM,
    PlatformMatrix,
    compute_lockfile_hash,
    determine_ml_core_lockfile_name,
    get_env_fingerprint,
)
from transformation_portal.determinism.jcs import dumpb as jcs_dumpb

logger = logging.getLogger(__name__)

# Version tag for CAS identity schema evolution
CAS_IDENTITY_VERSION = "adr-032-v2"  # v2: Added lockfile_hash

# Placeholder lockfile hash when none is provided
# Uses 64 zeros to be visually distinct and indicate "no lockfile"
# This is used for debugging - when you see this hash, it means
# no lockfile was provided and caching may be inconsistent
LOCKFILE_HASH_PLACEHOLDER = "sha256:0000000000000000000000000000000000000000000000000000000000000000"


class DeterminismViolationError(Exception):
    """Raised when determinism requirements are violated in strict mode.

    This exception is raised in CI environments when critical determinism
    inputs (like lockfile_hash) are missing, to prevent cache poisoning
    and non-reproducible artifacts.
    """

    pass


def is_ci_environment() -> bool:
    """Detect if we're running in a CI environment.

    Returns:
        True if running in CI (GitHub Actions, Jenkins, etc.)
    """
    # Common CI environment variables
    ci_vars = [
        "CI",  # Generic CI indicator
        "GITHUB_ACTIONS",  # GitHub Actions
        "JENKINS_URL",  # Jenkins
        "GITLAB_CI",  # GitLab CI
        "CIRCLECI",  # CircleCI
        "TRAVIS",  # Travis CI
        "BUILDKITE",  # Buildkite
    ]
    return any(os.environ.get(var) for var in ci_vars)


@dataclass(frozen=True)
class ExecutionIdentity:
    """Immutable execution identity for CAS-aware caching.

    Captures all determinism-relevant factors for a stage execution.

    Attributes:
        stage_name: Name of the pipeline stage
        stage_version: Semantic version of the stage implementation
        input_ids: Sorted list of input artifact SHA-256 hashes
        code_hash: SHA-256 of relevant source code
        config_hash: SHA-256 of canonicalized configuration
        env_fingerprint: Environment fingerprint (pip freeze hash)
        lockfile_hash: SHA-256 of requirements lockfile (ADR-032)
        platform_id: Canonical platform target (e.g., darwin-arm64-mps)
        cas_id: Final CAS identity hash
        schema_version: Schema version for future compatibility
    """

    stage_name: str
    stage_version: str
    input_ids: tuple[str, ...]
    code_hash: str
    config_hash: str
    env_fingerprint: str
    lockfile_hash: str  # ADR-032: Required for dependency graph integrity
    platform_id: str
    cas_id: str
    schema_version: str = CAS_IDENTITY_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "stage_name": self.stage_name,
            "stage_version": self.stage_version,
            "input_ids": list(self.input_ids),
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "env_fingerprint": self.env_fingerprint,
            "lockfile_hash": self.lockfile_hash,
            "platform_id": self.platform_id,
            "cas_id": self.cas_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionIdentity":
        """Reconstruct from dictionary."""
        return cls(
            stage_name=data["stage_name"],
            stage_version=data["stage_version"],
            input_ids=tuple(data["input_ids"]),
            code_hash=data["code_hash"],
            config_hash=data["config_hash"],
            env_fingerprint=data["env_fingerprint"],
            lockfile_hash=data.get("lockfile_hash", "sha256:unknown"),
            platform_id=data["platform_id"],
            cas_id=data["cas_id"],
            schema_version=data.get("schema_version", CAS_IDENTITY_VERSION),
        )


@dataclass(frozen=True)
class ArtifactMetadata:
    """Extended artifact metadata for CAS-aware execution.

    This schema captures all information needed for:
    - Cache key validation
    - Platform compatibility checks
    - Provenance tracking
    - Determinism verification

    Attributes:
        artifact_id: SHA-256 hash of artifact content
        stage: Stage name that produced this artifact
        inputs: Input artifact IDs that were consumed
        code_hash: Code version at time of creation
        config_hash: Configuration hash at time of creation
        env_fingerprint: Environment fingerprint at creation
        lockfile_hash: Lockfile hash at creation (ADR-032)
        platform_id: Platform that created the artifact
        created_at: ISO timestamp of creation
        version: Schema version tag
        execution_identity: Full execution identity (optional)
    """

    artifact_id: str
    stage: str
    inputs: tuple[str, ...]
    code_hash: str
    config_hash: str
    env_fingerprint: str
    lockfile_hash: str  # ADR-032: Required for dependency graph integrity
    platform_id: str
    created_at: str
    version: str = CAS_IDENTITY_VERSION
    execution_identity: Optional[str] = None  # CAS ID that produced this artifact

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage,
            "inputs": list(self.inputs),
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "env_fingerprint": self.env_fingerprint,
            "lockfile_hash": self.lockfile_hash,
            "platform_id": self.platform_id,
            "created_at": self.created_at,
            "version": self.version,
            "execution_identity": self.execution_identity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactMetadata":
        """Reconstruct from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            stage=data["stage"],
            inputs=tuple(data.get("inputs", [])),
            code_hash=data["code_hash"],
            config_hash=data["config_hash"],
            env_fingerprint=data["env_fingerprint"],
            lockfile_hash=data.get("lockfile_hash", "sha256:unknown"),
            platform_id=data["platform_id"],
            created_at=data["created_at"],
            version=data.get("version", CAS_IDENTITY_VERSION),
            execution_identity=data.get("execution_identity"),
        )


def compute_code_hash(
    paths: Optional[list[Union[str, Path]]] = None,
    use_git: bool = True,
) -> str:
    """Compute deterministic hash of source code.

    Uses git tree hash when available for maximum reproducibility.
    Falls back to file content hashing when git is unavailable.

    Args:
        paths: Specific paths to hash (default: ['src/'] relative to repo root)
        use_git: If True, use git ls-tree for hashing (recommended)

    Returns:
        SHA-256 hash in format "sha256:..."

    Note:
        When use_git=True, this uses `git ls-tree -r HEAD <paths>` which
        provides a deterministic hash of all tracked files under the specified
        paths. This is preferred over `git rev-parse HEAD` because it:
        - Only includes relevant source files, not the entire commit
        - Is stable across rebases that don't change source content
        - Excludes untracked/ignored files
    """
    if paths is None:
        paths = ["src/"]

    if use_git:
        try:
            # Use git ls-tree for deterministic subset hash
            cmd = ["git", "ls-tree", "-r", "HEAD"]
            cmd.extend(str(p) for p in paths)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                # Hash the sorted tree output for determinism
                digest = hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()
                return f"sha256:{digest}"
            else:
                # Fallback to git rev-parse HEAD
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0:
                    commit_hash = result.stdout.strip()
                    return f"git:{commit_hash}"

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Fallback: hash file contents directly
    return _compute_file_hash(paths)


def _normalize_code_ast(source: str) -> str:
    """Normalize source code to AST representation for stable hashing.

    This eliminates formatting drift (whitespace, comments, decorators)
    that would cause unnecessary cache invalidation.

    CRITICAL: Also strips line numbers, column offsets, and other
    non-semantic attributes to ensure stability across Python versions.
    Python 3.10/3.11/3.12 can produce different AST structures for the
    same code if these attributes are included.

    Args:
        source: Python source code string

    Returns:
        Normalized AST dump string

    Note:
        Uses ast.dump() with annotate_fields=False and include_attributes=False
        for maximal stability across Python versions.
    """
    try:
        tree = ast.parse(source)

        # Strip non-semantic attributes that can vary across Python versions
        # These include: lineno, col_offset, end_lineno, end_col_offset
        for node in ast.walk(tree):
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                if hasattr(node, attr):
                    setattr(node, attr, None)

        # annotate_fields=False + include_attributes=False = maximal stability
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        # Fall back to raw source if parsing fails
        return source


def compute_stage_code_hash(
    stage: Any,
    *,
    include_dependencies: bool = True,
) -> str:
    """Compute deterministic hash of a stage's implementation code.

    Uses AST-based normalization to prevent cache invalidation from:
    - Formatting changes (whitespace, indentation)
    - Comment changes
    - Decorator ordering

    Args:
        stage: Stage object (class instance or callable)
        include_dependencies: If True, include imported module dependencies

    Returns:
        SHA-256 hash in format "sha256:..."

    Note:
        This uses AST normalization instead of raw source code to ensure
        that only semantic changes (actual logic) invalidate the cache.
        Formatting changes and comments are ignored.

    Example:
        >>> class MyStage:
        ...     def execute(self, inputs, config):
        ...         return inputs * 2
        >>>
        >>> hash1 = compute_stage_code_hash(MyStage())
        >>> # Modify the stage
        >>> MyStage.execute = lambda self, i, c: i * 3
        >>> hash2 = compute_stage_code_hash(MyStage())
        >>> assert hash1 != hash2  # Code changed, hash changed
    """
    digest = hashlib.sha256()

    # Include schema version for identity stability
    digest.update(f"schema:{CAS_IDENTITY_VERSION}".encode("utf-8"))

    # Get the class or function to inspect
    target = type(stage) if hasattr(stage, "__class__") and not callable(stage) else stage

    try:
        # Get source code of the main target
        source = inspect.getsource(target)
        # Normalize using AST to eliminate formatting drift
        normalized = _normalize_code_ast(source)
        digest.update(normalized.encode("utf-8"))

        # Include method sources for class-based stages
        if inspect.isclass(target):
            for name, method in inspect.getmembers(target, predicate=inspect.isfunction):
                if not name.startswith("_") or name in ("__init__", "__call__"):
                    try:
                        method_source = inspect.getsource(method)
                        method_normalized = _normalize_code_ast(method_source)
                        digest.update(f"{name}:{method_normalized}".encode("utf-8"))
                    except (OSError, TypeError):
                        pass

        # Optionally include imported dependencies
        if include_dependencies:
            module = inspect.getmodule(target)
            if module and hasattr(module, "__file__") and module.__file__:
                # Hash the module file path for dependency tracking
                digest.update(f"module:{module.__file__}".encode("utf-8"))

    except (OSError, TypeError) as e:
        # inspect.getsource fails for built-in/C extension types
        # Fall back to repr + type name
        logger.debug("Cannot get source for %s: %s, using repr fallback", target, e)
        fallback = f"{type(target).__module__}.{type(target).__qualname__}"
        digest.update(fallback.encode("utf-8"))

    return f"sha256:{digest.hexdigest()}"


def resolve_platform_lockfile() -> Optional[Path]:
    """Resolve the canonical lockfile path for the current platform.

    Returns:
        Absolute Path to platform-specific lockfile, or None if not found.

    Note:
        This ensures lockfile paths are canonical (absolute, resolved),
        eliminating relative path ambiguity that could cause identity drift.
    """
    # Get repository root (relative to this module)
    try:
        module_path = Path(__file__).resolve()
        # Navigate up from src/transformation_portal/core/ to repo root
        repo_root = module_path.parent.parent.parent.parent

        # Determine platform-specific lockfile.
        try:
            matrix = PlatformMatrix.detect()
            lockfile = repo_root / "requirements" / determine_ml_core_lockfile_name(matrix)
        except ValueError:
            lockfile = None

        if lockfile is not None and lockfile.exists():
            return lockfile.resolve()  # Canonical absolute path

        # Fall back to requirements.txt
        fallback = repo_root / "requirements.txt"
        if fallback.exists():
            return fallback.resolve()

        return None
    except Exception:
        return None


def _compute_file_hash(paths: list[Union[str, Path]]) -> str:
    """Compute hash of file contents (fallback when git unavailable)."""
    digest = hashlib.sha256()
    files_found = 0

    for path in paths:
        path = Path(path)
        if path.is_file():
            digest.update(path.read_bytes())
            files_found += 1
        elif path.is_dir():
            for file_path in sorted(path.rglob("*.py")):
                if file_path.is_file():
                    digest.update(file_path.read_bytes())
                    files_found += 1

    if files_found == 0:
        return "sha256:unknown-no-files"

    return f"sha256:{digest.hexdigest()}"


def compute_config_hash(config: Any) -> str:
    """Compute deterministic hash of configuration.

    Uses JCS (RFC 8785) canonical JSON for deterministic serialization.

    Args:
        config: Configuration object (dict, dataclass, or any JSON-serializable)

    Returns:
        SHA-256 hash in format "sha256:..."

    Note:
        - Dict keys are sorted per JCS specification
        - Floats are normalized to ECMAScript format
        - Unicode is preserved (not escaped to ASCII)
    """
    if config is None:
        config = {}

    # Convert to dict if needed
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    elif hasattr(config, "__dict__") and not isinstance(config, dict):
        config = {k: v for k, v in vars(config).items() if not k.startswith("_")}

    # Use JCS for canonical serialization
    canonical_bytes = jcs_dumpb(config)
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return f"sha256:{digest}"


def compute_cas_id(
    stage_name: str,
    input_ids: list[str],
    config: Any,
    *,
    stage_version: str = "1.0.0",
    code_hash: Optional[str] = None,
    env_fingerprint: Optional[str] = None,
    lockfile_hash: Optional[str] = None,
    lockfile_path: Optional[str] = None,
    platform: Optional[PlatformMatrix] = None,
) -> ExecutionIdentity:
    """Compute unified CAS identity for stage execution.

    This is the core function of the deterministic execution layer.
    It produces a unique, content-addressable identity for a stage execution
    based on all determinism-relevant factors.

    Args:
        stage_name: Name of the pipeline stage
        input_ids: List of input artifact SHA-256 hashes
        config: Stage configuration (dict or config object)
        stage_version: Semantic version of stage implementation
        code_hash: Pre-computed code hash (if None, computed automatically)
        env_fingerprint: Pre-computed env fingerprint (if None, computed automatically)
        lockfile_hash: Pre-computed lockfile hash (ADR-032)
        lockfile_path: Path to lockfile (used if lockfile_hash is None)
        platform: Platform matrix (if None, uses CURRENT_PLATFORM)

    Returns:
        ExecutionIdentity with all determinism factors and final CAS ID

    Note:
        ADR-032 requires lockfile_hash for dependency graph integrity.
        If neither lockfile_hash nor lockfile_path is provided, a placeholder
        hash is used. This should be avoided in production.

    Example:
        >>> identity = compute_cas_id(
        ...     stage_name="depth_estimation",
        ...     input_ids=["sha256:abc123"],
        ...     config={"model": "DA3-Large", "quantization": "none"},
        ...     lockfile_path="requirements/ml-core-darwin-arm64.txt",
        ... )
        >>> if artifact_store.exists(identity.cas_id):
        ...     return artifact_store.load(identity.cas_id)  # Cache hit
    """
    # Normalize inputs
    sorted_inputs = tuple(sorted(input_ids))

    # Compute or use provided hashes
    if code_hash is None:
        code_hash = compute_code_hash()

    if env_fingerprint is None:
        env_fingerprint = get_env_fingerprint()

    # Compute lockfile hash (ADR-032: Required for dependency graph integrity)
    if lockfile_hash is None:
        if lockfile_path is not None:
            lockfile_hash = compute_lockfile_hash(lockfile_path)
        else:
            # FAIL-CLOSED in CI: Missing lockfile hash is a determinism violation
            # that can cause cache poisoning and non-reproducible artifacts
            allow_nondet = os.environ.get("TP_ALLOW_NONDETERMINISTIC", "").lower() in ("1", "true", "yes")
            if is_ci_environment() and not allow_nondet:
                raise DeterminismViolationError(
                    f"Missing lockfile_hash for stage '{stage_name}' in CI environment. "
                    "Deterministic builds require explicit lockfile_path parameter. "
                    "To proceed in non-deterministic mode, set TP_ALLOW_NONDETERMINISTIC=1."
                )

            # In local dev or when explicitly allowed, use placeholder with warning
            lockfile_hash = LOCKFILE_HASH_PLACEHOLDER
            logger.warning(
                "No lockfile_hash or lockfile_path provided for %s. "
                "Using placeholder. This may cause cache invalidation issues. "
                "For deterministic builds, provide lockfile_path parameter.",
                stage_name,
            )

    if platform is None:
        platform = CURRENT_PLATFORM

    platform_id = platform.canonical_target if platform else "unknown-platform"

    # Compute config hash
    config_hash = compute_config_hash(config)

    # Build CAS identity payload (ADR-032 v2: includes lockfile_hash)
    identity_payload = {
        "stage": stage_name,
        "stage_version": stage_version,
        "inputs": list(sorted_inputs),
        "code": code_hash,
        "config": config_hash,
        "env": env_fingerprint,
        "lockfile": lockfile_hash,  # ADR-032: Required for dependency graph integrity
        "platform": platform_id,
        "schema_version": CAS_IDENTITY_VERSION,
    }

    # Compute final CAS ID using JCS
    canonical_bytes = jcs_dumpb(identity_payload)
    cas_id = f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"

    return ExecutionIdentity(
        stage_name=stage_name,
        stage_version=stage_version,
        input_ids=sorted_inputs,
        code_hash=code_hash,
        config_hash=config_hash,
        env_fingerprint=env_fingerprint,
        lockfile_hash=lockfile_hash,
        platform_id=platform_id,
        cas_id=cas_id,
        schema_version=CAS_IDENTITY_VERSION,
    )


def should_execute(
    identity: ExecutionIdentity,
    artifact_store: Any,  # ArtifactStore from storage.cas_store
) -> bool:
    """Check if an artifact exists in the CAS store.

    IMPORTANT: This function checks if an object exists in the CAS store
    keyed by the raw SHA-256 digest portion of the CAS ID. However, the
    CASExecutor stores stage results in a separate result cache directory,
    NOT in the ArtifactStore keyed by CAS ID.

    This function is primarily useful for checking if specific artifacts
    (like numpy arrays) have been stored in CAS, not for checking if a
    stage execution has been cached.

    For checking if a stage should execute, use CASExecutor directly,
    which handles result cache lookups properly.

    Args:
        identity: Execution identity for the stage
        artifact_store: ArtifactStore instance to check

    Returns:
        True if CAS object does NOT exist (should execute)
        False if CAS object exists (cached)

    Note:
        The CAS ID format is 'sha256:<hex_digest>'. This function
        extracts the hex digest portion before checking the store.
    """
    # Extract the hex digest from the CAS ID
    cas_id = identity.cas_id
    if cas_id.startswith("sha256:"):
        sha256_digest = cas_id[7:]
    else:
        sha256_digest = cas_id
    return not artifact_store.has_object(sha256_digest)


def explain_cache_miss(
    current_identity: ExecutionIdentity,
    cached_identity: Optional[ExecutionIdentity],
) -> dict[str, Any]:
    """Explain why a cache miss occurred (debug utility).

    Compares current and cached identities to pinpoint the exact
    dimension(s) that caused the cache miss.

    Args:
        current_identity: The identity computed for current execution
        cached_identity: The identity from cached artifact (None if no cache)

    Returns:
        Dict with:
        - "reason": Human-readable summary
        - "differences": List of fields that differ
        - "details": Detailed comparison for each differing field

    Example:
        >>> [CAS MISS]
        >>> stage: depth
        >>> reason:
        >>>   - lockfile_hash mismatch
        >>>   - code_hash mismatch
    """
    if cached_identity is None:
        return {
            "reason": "No cached artifact exists",
            "differences": [],
            "details": {},
        }

    differences = []
    details = {}

    # Compare all identity dimensions
    fields_to_compare = [
        ("stage_name", "Stage name"),
        ("stage_version", "Stage version"),
        ("input_ids", "Input artifact IDs"),
        ("code_hash", "Code hash"),
        ("config_hash", "Config hash"),
        ("env_fingerprint", "Environment fingerprint"),
        ("lockfile_hash", "Lockfile hash"),
        ("platform_id", "Platform ID"),
        ("schema_version", "Schema version"),
    ]

    for field, label in fields_to_compare:
        current_val = getattr(current_identity, field)
        cached_val = getattr(cached_identity, field)
        if current_val != cached_val:
            differences.append(field)
            details[field] = {
                "label": label,
                "current": current_val[:32] if isinstance(current_val, str) else current_val,
                "cached": cached_val[:32] if isinstance(cached_val, str) else cached_val,
            }

    if not differences:
        return {
            "reason": "Identities match but CAS miss (possible store issue)",
            "differences": [],
            "details": {},
        }

    reason_parts = [f"{details[d]['label']} mismatch" for d in differences]
    return {
        "reason": f"Identity mismatch: {', '.join(reason_parts)}",
        "differences": differences,
        "details": details,
    }


def log_cache_decision(
    identity: ExecutionIdentity,
    cache_hit: bool,
    cached_identity: Optional[ExecutionIdentity] = None,
) -> None:
    """Log cache hit/miss decision with identity details (debug utility).

    Args:
        identity: Current execution identity
        cache_hit: True if cache hit, False if miss
        cached_identity: If miss, the cached identity for comparison
    """
    if cache_hit:
        logger.debug(
            "[CAS HIT] stage=%s cas_id=%s",
            identity.stage_name,
            identity.cas_id[:16],
        )
    else:
        explanation = explain_cache_miss(identity, cached_identity)
        logger.debug(
            "[CAS MISS] stage=%s reason=%s differences=%s",
            identity.stage_name,
            explanation["reason"],
            explanation["differences"],
        )
        for field, detail in explanation.get("details", {}).items():
            logger.debug(
                "  %s: current=%s cached=%s",
                detail["label"],
                detail.get("current", "N/A"),
                detail.get("cached", "N/A"),
            )


# Environment variable for cross-platform artifact reuse (default: disabled)
ALLOW_CROSS_PLATFORM = os.environ.get("TP_ALLOW_CROSS_PLATFORM", "false").lower() in (
    "true",
    "1",
    "yes",
)


def is_compatible(
    artifact_metadata: ArtifactMetadata,
    current_platform: Optional[PlatformMatrix] = None,
    current_env_fingerprint: Optional[str] = None,
    current_lockfile_hash: Optional[str] = None,
    *,
    strict: bool = True,
    allow_cross_platform: Optional[bool] = None,
) -> bool:
    """Check if an artifact is compatible with current platform/environment.

    IMPORTANT: Platform compatibility is STRICT by default (Blocker #4).
    Cross-platform artifact reuse is DISABLED unless explicitly enabled.

    This guard prevents invalid reuse across incompatible platforms which
    can cause silent numerical output corruption.

    Args:
        artifact_metadata: Metadata of the artifact to validate
        current_platform: Current platform (default: CURRENT_PLATFORM)
        current_env_fingerprint: Current env fingerprint (default: computed)
        current_lockfile_hash: Current lockfile hash (default: None)
        strict: If True (default), require exact platform + env + lockfile match
        allow_cross_platform: If True, allow cross-platform reuse for CPU artifacts.
                              If None, uses TP_ALLOW_CROSS_PLATFORM env var (default: false)

    Returns:
        True if artifact is safe to reuse, False otherwise

    Security Note:
        Cross-platform artifact reuse (e.g., darwin-arm64-mps -> linux-x86_64-cpu)
        can produce INVALID NUMERICAL OUTPUT even for CPU-only operations.
        This is because:
        - Different floating-point implementations across architectures
        - Different library versions (BLAS, OpenMP, etc.)
        - Different memory layouts and alignment

        NEVER enable cross-platform reuse for numerical artifacts in production.

    Example:
        >>> # Strict mode (default) - exact match required
        >>> is_compatible(artifact, current_platform)
        False  # Different platform

        >>> # Explicit cross-platform opt-in (DANGEROUS)
        >>> is_compatible(artifact, allow_cross_platform=True)
        True   # CPU fallback allowed
    """
    # Resolve allow_cross_platform from env var if not specified
    if allow_cross_platform is None:
        allow_cross_platform = ALLOW_CROSS_PLATFORM

    if current_platform is None:
        current_platform = CURRENT_PLATFORM

    if current_env_fingerprint is None:
        current_env_fingerprint = get_env_fingerprint()

    current_platform_id = current_platform.canonical_target if current_platform else ""

    # Rule 1: Platform must match exactly (unless cross-platform explicitly allowed)
    platform_match = artifact_metadata.platform_id == current_platform_id

    if not platform_match:
        if allow_cross_platform:
            # Only allow CPU-to-CPU cross-platform reuse
            artifact_is_cpu = artifact_metadata.platform_id.endswith("-cpu")
            current_is_cpu = current_platform_id.endswith("-cpu")

            if artifact_is_cpu and current_is_cpu:
                logger.warning(
                    "Cross-platform artifact reuse enabled (DANGEROUS): %s -> %s. "
                    "This may produce invalid numerical output.",
                    artifact_metadata.platform_id,
                    current_platform_id,
                )
                # Continue to check other constraints
            else:
                logger.debug(
                    "Cross-platform reuse rejected: %s -> %s (GPU artifacts cannot cross platforms)",
                    artifact_metadata.platform_id,
                    current_platform_id,
                )
                return False
        else:
            logger.debug(
                "Artifact %s incompatible: platform %s != %s (strict mode)",
                artifact_metadata.artifact_id[:8],
                artifact_metadata.platform_id,
                current_platform_id,
            )
            return False

    # Rule 2: Environment fingerprint must match (in strict mode)
    if strict:
        if artifact_metadata.env_fingerprint != current_env_fingerprint:
            logger.debug(
                "Artifact %s incompatible: env fingerprint %s != %s",
                artifact_metadata.artifact_id[:8],
                artifact_metadata.env_fingerprint[:16],
                current_env_fingerprint[:16],
            )
            return False

    # Rule 3: Lockfile hash must match if provided (ADR-032)
    if strict and current_lockfile_hash is not None:
        if artifact_metadata.lockfile_hash != current_lockfile_hash:
            logger.debug(
                "Artifact %s incompatible: lockfile hash %s != %s",
                artifact_metadata.artifact_id[:8],
                artifact_metadata.lockfile_hash[:16] if artifact_metadata.lockfile_hash else "none",
                current_lockfile_hash[:16],
            )
            return False

    return True


def create_artifact_metadata(
    artifact_id: str,
    execution_identity: ExecutionIdentity,
) -> ArtifactMetadata:
    """Create artifact metadata from execution identity.

    Helper function to create complete artifact metadata after
    successful stage execution.

    Args:
        artifact_id: SHA-256 hash of the artifact content
        execution_identity: Identity of the execution that produced it

    Returns:
        ArtifactMetadata with full provenance information
    """
    return ArtifactMetadata(
        artifact_id=artifact_id,
        stage=execution_identity.stage_name,
        inputs=execution_identity.input_ids,
        code_hash=execution_identity.code_hash,
        config_hash=execution_identity.config_hash,
        env_fingerprint=execution_identity.env_fingerprint,
        lockfile_hash=execution_identity.lockfile_hash,
        platform_id=execution_identity.platform_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        version=execution_identity.schema_version,
        execution_identity=execution_identity.cas_id,
    )


def verify_determinism(
    stage_fn: Any,
    inputs: dict[str, Any],
    config: Any,
    *,
    runs: int = 2,
) -> tuple[bool, list[str]]:
    """Verify stage produces deterministic output.

    Runs the stage multiple times and compares output hashes.

    Args:
        stage_fn: Callable that takes (inputs, config) and returns output
        inputs: Input artifacts/data
        config: Stage configuration
        runs: Number of runs to compare (default: 2)

    Returns:
        Tuple of (is_deterministic, list of output hashes)

    Example:
        >>> is_deterministic, hashes = verify_determinism(
        ...     stage_fn=my_stage.run,
        ...     inputs={"image": img_array},
        ...     config={"model": "DA3-Large"},
        ... )
        >>> assert is_deterministic, f"Non-deterministic: {hashes}"
    """
    from transformation_portal.ingest.canonical_json import canonicalize_json

    hashes = []

    for i in range(runs):
        result = stage_fn(inputs, config)

        # Compute hash of result
        if hasattr(result, "tobytes"):
            # numpy array
            result_hash = hashlib.sha256(result.tobytes()).hexdigest()
        elif isinstance(result, (bytes, bytearray)):
            result_hash = hashlib.sha256(result).hexdigest()
        elif isinstance(result, dict):
            # Use JCS for deterministic dict hashing
            result_hash = hashlib.sha256(jcs_dumpb(result)).hexdigest()
        else:
            # Fallback: Canonical JSON serialization (policy-compliant)
            result_hash = hashlib.sha256(canonicalize_json(result)).hexdigest()

        hashes.append(result_hash)
        logger.debug("Run %d hash: %s", i + 1, result_hash[:16])

    # Check all hashes are identical
    is_deterministic = len(set(hashes)) == 1

    if not is_deterministic:
        logger.warning(
            "Non-deterministic stage output: %d unique hashes from %d runs",
            len(set(hashes)),
            runs,
        )

    return is_deterministic, hashes
