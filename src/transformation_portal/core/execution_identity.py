"""Unified CAS Identity Model for deterministic execution.

This module implements Phase 2 of the Deterministic Execution Layer (ADR-030/032).
Every stage execution becomes:

    Stage Execution = f(inputs, code, config, environment)

The CAS identity materializes this function into a single content-addressable hash.

CAS_ID = sha256(
    stage_name +
    input_artifact_ids +
    code_hash +
    config_hash +
    env_fingerprint +
    platform_id
)

Design Principles:
    - Deterministic: Same inputs always produce same CAS ID
    - Platform-aware: Different platforms produce different IDs
    - Code-aware: Code changes invalidate cache
    - Config-aware: Config changes invalidate cache

Example:
    >>> from transformation_portal.core.execution_identity import compute_cas_id
    >>> cas_id = compute_cas_id(
    ...     stage_name="depth_estimation",
    ...     input_ids=["sha256:abc123..."],
    ...     config={"model": "DA3-Large"},
    ... )
    >>> print(cas_id)
    'sha256:def456...'
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from transformation_portal.core.platform_matrix import (
    CURRENT_PLATFORM,
    PlatformMatrix,
    get_env_fingerprint,
)
from transformation_portal.determinism.jcs import dumpb as jcs_dumpb

logger = logging.getLogger(__name__)

# Version tag for CAS identity schema evolution
CAS_IDENTITY_VERSION = "adr-032-v1"


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
        platform: Platform matrix (if None, uses CURRENT_PLATFORM)

    Returns:
        ExecutionIdentity with all determinism factors and final CAS ID

    Example:
        >>> identity = compute_cas_id(
        ...     stage_name="depth_estimation",
        ...     input_ids=["sha256:abc123"],
        ...     config={"model": "DA3-Large", "quantization": "none"},
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

    if platform is None:
        platform = CURRENT_PLATFORM

    platform_id = platform.canonical_target if platform else "unknown-platform"

    # Compute config hash
    config_hash = compute_config_hash(config)

    # Build CAS identity payload
    identity_payload = {
        "stage": stage_name,
        "stage_version": stage_version,
        "inputs": list(sorted_inputs),
        "code": code_hash,
        "config": config_hash,
        "env": env_fingerprint,
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
        platform_id=platform_id,
        cas_id=cas_id,
        schema_version=CAS_IDENTITY_VERSION,
    )


def should_execute(
    identity: ExecutionIdentity,
    artifact_store: Any,  # ArtifactStore from storage.cas_store
) -> bool:
    """Determine if stage should execute or use cached result.

    This is the execution gate that enables CAS-aware skipping.

    Args:
        identity: Execution identity for the stage
        artifact_store: ArtifactStore instance to check for cached results

    Returns:
        True if stage should execute (cache miss), False if cached (cache hit)

    Example:
        >>> identity = compute_cas_id(...)
        >>> if should_execute(identity, artifact_store):
        ...     result = stage.run(inputs)
        ...     artifact_store.store(identity.cas_id, result)
        ... else:
        ...     result = artifact_store.load(identity.cas_id)
    """
    return not artifact_store.has_object(identity.cas_id)


def is_compatible(
    artifact_metadata: ArtifactMetadata,
    current_platform: Optional[PlatformMatrix] = None,
    current_env_fingerprint: Optional[str] = None,
    *,
    allow_cpu_fallback: bool = False,
) -> bool:
    """Check if an artifact is compatible with current platform/environment.

    This guard prevents invalid reuse across incompatible platforms.

    Args:
        artifact_metadata: Metadata of the artifact to validate
        current_platform: Current platform (default: CURRENT_PLATFORM)
        current_env_fingerprint: Current env fingerprint (default: computed)
        allow_cpu_fallback: If True, allow CPU artifacts on any platform

    Returns:
        True if artifact is safe to reuse, False otherwise

    Note:
        When allow_cpu_fallback=True, artifacts created on any CPU platform
        can be reused on any other CPU platform. This is useful for artifacts
        that are platform-independent (e.g., JSON configs, model weights).
    """
    if current_platform is None:
        current_platform = CURRENT_PLATFORM

    if current_env_fingerprint is None:
        current_env_fingerprint = get_env_fingerprint()

    current_platform_id = current_platform.canonical_target if current_platform else ""

    # Strict platform match
    if artifact_metadata.platform_id == current_platform_id:
        # Also check env fingerprint for full determinism
        if artifact_metadata.env_fingerprint == current_env_fingerprint:
            return True
        else:
            logger.debug(
                "Artifact %s has different env fingerprint: %s != %s",
                artifact_metadata.artifact_id[:8],
                artifact_metadata.env_fingerprint[:16],
                current_env_fingerprint[:16],
            )
            return False

    # Optional CPU fallback mode
    if allow_cpu_fallback:
        # Check if both are CPU platforms (no GPU acceleration)
        artifact_is_cpu = artifact_metadata.platform_id.endswith("-cpu")
        current_is_cpu = current_platform_id.endswith("-cpu")

        if artifact_is_cpu and current_is_cpu:
            logger.debug(
                "CPU fallback allowed: %s -> %s",
                artifact_metadata.platform_id,
                current_platform_id,
            )
            return True

    logger.debug(
        "Artifact %s incompatible: platform %s != %s",
        artifact_metadata.artifact_id[:8],
        artifact_metadata.platform_id,
        current_platform_id,
    )
    return False


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
    import json

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
            # Fallback: JSON serialization
            result_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True, default=str).encode()
            ).hexdigest()

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
