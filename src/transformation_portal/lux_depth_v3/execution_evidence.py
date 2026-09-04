"""Fail-closed runtime evidence for prepared Lux execution plans.

The legacy run-card artifact index is a compatibility catalog assembled by
best-effort discovery.  It is useful to older readers, but it cannot prove
that every output declared by an authoritative plan was produced.  This
module instead starts from the exact output declarations in an
``execution_complete`` plan, expands their scope/cardinality over the frozen
input selection, and records every requested output as produced, omitted, or
failed.

Evidence is written as a detached sidecar after the existing manifests and
run card.  Keeping it detached avoids an impossible self-hash cycle while
allowing the full bytes of every requested manifest artifact to be hashed.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import secrets
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from ..core.execution_plan import EXECUTION_COMPLETE, CanonicalExecutionPlan
from ..ingest.canonical_json import TP_CANONICAL_JSON_PROFILE, canonicalize_json
from ..stage_graph.registry import OutputCardinality, OutputScope, get_output_definition
from ._backend_contract import normalize_backend_id

EXECUTION_EVIDENCE_SCHEMA = "tp.lux.execution.evidence.v1"
EXECUTION_EVIDENCE_SCHEMA_RESOURCE = "evidence.v1.schema.json"
MANIFEST_OUTCOME_PROJECTION_SCHEMA = "tp.lux.execution.outcome_projection.v1"

_READ_CHUNK_BYTES = 1024 * 1024
_DEFAULT_NEW_FILE_MODE = 0o644
_MAX_EVIDENCE_BYTES = 256 * 1024 * 1024
_MAX_BOUND_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_CUMULATIVE_DECODED_MANIFEST_BYTES = 128 * 1024 * 1024
_MAX_RETAINED_MANIFEST_AUTHORITY_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_OBSERVATIONS = 524_288
_MAX_ARTIFACTS_PER_OUTCOME = 4_096
_MAX_PLAN_INPUTS = 4_096
_MAX_PLAN_OUTPUTS = 128
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024
_MAX_AGGREGATE_ARTIFACT_BYTES = 1024 * 1024 * 1024 * 1024
_MAX_RENDERED_ERROR_CHARS = 2_048
_MAX_RENDERED_SCHEMA_PATH_CHARS = 512
_MAX_BATCH_ID_CHARS = 512
_MAX_EVIDENCE_INTEGER_BITS = 4_096
_MAX_EVIDENCE_NESTING_DEPTH = 64
_MAX_EVIDENCE_VALIDATION_NODES = 8_000_000
_HAS_SECURE_DIR_FD_OPEN = (
    os.name == "posix"
    and os.open in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
    and hasattr(os, "fchmod")
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)
_HAS_SECURE_NO_REPLACE_LINK = os.name == "posix" and os.link in os.supports_dir_fd and os.link in os.supports_follow_symlinks
_INPUT_STATUSES = frozenset({"ok", "error", "skipped", "missing"})
_MANIFEST_ARTIFACT_KINDS = frozenset({"combined_manifest_json", "batch_manifest_json", "run_card"})
_EXPLICIT_FAILURE_CODES = frozenset(
    {
        "artifact_input_mismatch",
        "incomplete_reconstruction_bundle",
        "invalid_manifest_binding",
    }
)


class ExecutionEvidenceError(RuntimeError):
    """Raised when prepared-run evidence is incomplete or invalid."""


class ArtifactEvidenceError(ExecutionEvidenceError):
    """Raised when an observed artifact cannot be verified safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _ArtifactCardinalityError(ArtifactEvidenceError):
    """Internal bounded-discovery failure carrying the observed count."""

    def __init__(self, message: str, *, observed_count: int):
        super().__init__("artifact_cardinality_limit_exceeded", message)
        self.observed_count = observed_count


@dataclass(frozen=True)
class InputExecution:
    """Runtime outcome for one exact input carried by the prepared plan."""

    input_id: str
    status: str
    executed_backend: Optional[str]
    error_code: Optional[str] = None


@dataclass(frozen=True)
class ArtifactObservation:
    """One path claimed for a logical output declaration."""

    artifact_kind: str
    path: Optional[Path]
    input_id: Optional[str] = None
    failure_code: Optional[str] = None


@dataclass(frozen=True)
class ConfinedArtifactSnapshot:
    """Immutable bytes and identity claims from one confined artifact read."""

    data: bytes
    relative_path: str
    sha256: str
    size_bytes: int
    device: Optional[int] = None
    inode: Optional[int] = None

    def matches(self, record: Mapping[str, Any]) -> bool:
        """Return whether an evidence record names these exact captured bytes."""

        return (
            isinstance(record, Mapping)
            and record.get("path") == self.relative_path
            and record.get("sha256") == self.sha256
            and type(record.get("size_bytes")) is int
            and record.get("size_bytes") == self.size_bytes
        )


@dataclass(frozen=True)
class ConfinedArtifactCopy:
    """Identity and digest of one separately materialized confined copy."""

    source_relative_path: str
    relative_path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int

    def matches(self, record: Mapping[str, Any]) -> bool:
        """Return whether an evidence record names these exact copied bytes."""

        return (
            isinstance(record, Mapping)
            and record.get("path") == self.relative_path
            and record.get("sha256") == self.sha256
            and type(record.get("size_bytes")) is int
            and record.get("size_bytes") == self.size_bytes
        )


@dataclass
class ConfinedArtifactCopyBudget:
    """Cumulative byte budget shared by one prepared carrier activation."""

    max_bytes: int = _MAX_AGGREGATE_ARTIFACT_BYTES
    total_bytes: int = 0

    def check_known_size(self, size_bytes: int) -> None:
        """Reject one copy before I/O when it would exceed the batch limit."""

        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        if self.total_bytes + size_bytes > self.max_bytes:
            raise ArtifactEvidenceError(
                "aggregate_artifact_bytes_exceeded",
                f"Artifact carrier copies exceed the aggregate limit of {self.max_bytes} bytes",
            )

    def commit(self, size_bytes: int) -> None:
        """Debit one successfully published carrier."""

        self.check_known_size(size_bytes)
        self.total_bytes += size_bytes


@dataclass(frozen=True)
class _PinnedOutputRoot:
    """Descriptor-pinned authority for one canonical output directory."""

    canonical_path: Path
    lexical_path: Path
    descriptor: int
    device: int
    inode: int
    component_names: tuple[str, ...]
    component_descriptors: tuple[int, ...]
    component_identities: tuple[tuple[int, int], ...]

    def confined_relative_path(self, candidate: Path) -> str:
        """Return a canonical relative path without resolving in-root links."""

        raw_candidate = Path(candidate)
        if raw_candidate.is_absolute():
            absolute_candidate = Path(os.path.abspath(os.fspath(raw_candidate)))
            variants = (absolute_candidate, _canonicalize_top_level_alias(absolute_candidate))
            root_spellings = (self.lexical_path, self.canonical_path)
            relative: Optional[Path] = None
            for variant in variants:
                for root_spelling in root_spellings:
                    try:
                        relative = variant.relative_to(root_spelling)
                    except ValueError:
                        continue
                    break
                if relative is not None:
                    break
            if relative is None:
                raise ArtifactEvidenceError("path_escape", f"Artifact path escapes output root: {candidate}")
        else:
            if ".." in raw_candidate.parts:
                raise ArtifactEvidenceError(
                    "path_escape",
                    f"Artifact path is not canonical and confined: {candidate}",
                )
            # Runtime writers preserve a caller's relative output-root
            # spelling (``out/depth/x.png``), while manifest pointers may be
            # root-relative (``depth/x.png``). Resolve the former against the
            # current working directory first, then fall back to the latter;
            # both candidates must still be lexically confined.
            cwd_candidate = Path(os.path.abspath(os.fspath(raw_candidate)))
            cwd_variants = (cwd_candidate, _canonicalize_top_level_alias(cwd_candidate))
            root_spellings = (self.lexical_path, self.canonical_path)
            relative = None
            for variant in cwd_variants:
                for root_spelling in root_spellings:
                    try:
                        relative = variant.relative_to(root_spelling)
                    except ValueError:
                        continue
                    break
                if relative is not None:
                    break
            if relative is None:
                rooted_candidate = Path(os.path.abspath(os.fspath(self.canonical_path / raw_candidate)))
                try:
                    relative = rooted_candidate.relative_to(self.canonical_path)
                except ValueError as exc:
                    raise ArtifactEvidenceError("path_escape", f"Artifact path escapes output root: {candidate}") from exc

        relative_path = relative.as_posix()
        portable = PurePosixPath(relative_path)
        if (
            not relative.parts
            or not relative_path
            or relative_path == "."
            or "\\" in relative_path
            or PureWindowsPath(relative_path).drive
            or portable.as_posix() != relative_path
            or any(part in {"", ".", ".."} for part in portable.parts)
        ):
            raise ArtifactEvidenceError("path_escape", f"Artifact path is not canonical and confined: {candidate}")
        return relative_path


@dataclass(frozen=True)
class _ArtifactCapture:
    record: dict[str, Any]
    identity: tuple[int, int]


@dataclass
class _CaptureBudget:
    total_bytes: int = 0

    def check_known_size(self, size_bytes: int, *, context: str) -> None:
        if size_bytes > _MAX_ARTIFACT_BYTES:
            raise ArtifactEvidenceError(
                "artifact_too_large",
                f"{context} exceeds the per-file limit of {_MAX_ARTIFACT_BYTES} bytes",
            )
        if self.total_bytes + size_bytes > _MAX_AGGREGATE_ARTIFACT_BYTES:
            raise ArtifactEvidenceError(
                "aggregate_artifact_bytes_exceeded",
                f"Artifact capture exceeds the aggregate limit of {_MAX_AGGREGATE_ARTIFACT_BYTES} bytes",
            )

    def commit(self, size_bytes: int) -> None:
        self.total_bytes += size_bytes


@dataclass
class _ManifestBudget:
    """Bound cumulative manifest decoding and retained authority state."""

    decoded_bytes: int = 0
    retained_bytes: int = 0

    def reserve_decode(self, size_bytes: int, *, context: str) -> None:
        if size_bytes > _MAX_BOUND_MANIFEST_BYTES:
            raise ArtifactEvidenceError(
                "artifact_too_large",
                f"{context} exceeds the manifest limit of {_MAX_BOUND_MANIFEST_BYTES} bytes",
            )
        if self.decoded_bytes + size_bytes > _MAX_CUMULATIVE_DECODED_MANIFEST_BYTES:
            raise ArtifactEvidenceError(
                "aggregate_artifact_bytes_exceeded",
                "Manifest decoding exceeds the cumulative limit of " f"{_MAX_CUMULATIVE_DECODED_MANIFEST_BYTES} bytes",
            )
        self.decoded_bytes += size_bytes

    def retain(self, size_bytes: int, *, context: str) -> None:
        if self.retained_bytes + size_bytes > _MAX_RETAINED_MANIFEST_AUTHORITY_BYTES:
            raise ArtifactEvidenceError(
                "aggregate_artifact_bytes_exceeded",
                f"{context} exceeds the retained manifest-authority limit of "
                f"{_MAX_RETAINED_MANIFEST_AUTHORITY_BYTES} bytes",
            )
        self.retained_bytes += size_bytes


@dataclass
class _BatchRunCardBinding:
    """Cross-bind the two per-run manifests without depending on their order."""

    batch_id: Optional[str] = None

    def bind(self, artifact_kind: str, payload: Mapping[str, Any]) -> None:
        batch_id = payload.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id or len(batch_id) > _MAX_BATCH_ID_CHARS:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                f"{artifact_kind} has an invalid batch_id",
            )
        if self.batch_id is not None and batch_id != self.batch_id:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                "batch_manifest_json and run_card batch_id values do not match",
            )
        self.batch_id = batch_id


def load_execution_evidence_schema() -> dict[str, Any]:
    """Load the package-shipped evidence schema."""

    payload = (
        resources.files("transformation_portal.schemas.execution")
        .joinpath(EXECUTION_EVIDENCE_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):  # pragma: no cover - governed resource
        raise ExecutionEvidenceError("Packaged execution evidence schema is not a JSON object")
    return decoded


def _requested_declarations(plan: CanonicalExecutionPlan) -> list[dict[str, Any]]:
    declarations_by_kind: dict[str, dict[str, Any]] = {}
    for node in plan.nodes:
        for output in node.outputs:
            if output.disposition != "requested":
                continue
            declarations_by_kind[output.artifact_kind] = {
                "declaration_id": output.output_id,
                "stage_registry_id": node.stage_registry_id.value,
                "artifact_kind": output.artifact_kind,
                "scope": output.scope,
                "cardinality": output.cardinality,
                "required": output.required,
            }
    return [declarations_by_kind[artifact_kind] for artifact_kind in plan.requested_outputs]


def _validate_prepared_plan(plan: CanonicalExecutionPlan) -> None:
    if type(plan) is not CanonicalExecutionPlan:
        raise TypeError("plan must be a CanonicalExecutionPlan")
    if plan.configuration_completeness != EXECUTION_COMPLETE:
        raise ExecutionEvidenceError("Runtime evidence requires an execution-complete prepared plan")
    # Reparse the in-memory payload so drifted/mocked carriers cannot bypass
    # the plan schema, semantic, or fingerprint checks at this boundary.
    CanonicalExecutionPlan.from_payload(plan.to_payload())


def build_manifest_plan_projection(
    plan: CanonicalExecutionPlan,
    *,
    input_executions: Sequence[InputExecution] = (),
    evidence_path: str,
) -> dict[str, Any]:
    """Build the plan/runtime binding embedded in existing manifest formats."""

    return _ManifestPlanProjector(plan, evidence_path).build(input_executions)


def build_manifest_outcome_projection(
    payload: Mapping[str, Any],
    *,
    evidence_path: str,
    input_id: Optional[str] = None,
) -> dict[str, Any]:
    """Project outcome identity into a manifest without creating a hash cycle.

    The detached evidence sidecar remains authoritative for artifact paths,
    checksums, sizes, and media metadata.  Carriers repeat only the typed
    declaration/outcome identity and reason fields needed to make their
    requested/produced/omitted/failed state directly observable.  A combined
    manifest receives only outcomes for its own input; batch manifests and run
    cards receive the complete execution projection.
    """

    if payload.get("schema") != EXECUTION_EVIDENCE_SCHEMA:
        raise ExecutionEvidenceError("Manifest outcome projection requires execution evidence v1")
    _validate_evidence_path(evidence_path)

    requested = payload.get("requested_artifacts")
    if not isinstance(requested, list) or not all(isinstance(item, Mapping) for item in requested):
        raise ExecutionEvidenceError("Execution evidence requested_artifacts must be an array of objects")

    if input_id is None:
        projected_requested = [dict(item) for item in requested]
        scope: dict[str, Any] = {"kind": "execution"}
    else:
        projected_requested = [dict(item) for item in requested if item.get("scope") == OutputScope.PER_INPUT.value]
        scope = {"kind": "input", "input_id": input_id}

    def project_bucket(field_name: str) -> list[dict[str, Any]]:
        raw_bucket = payload.get(field_name)
        if not isinstance(raw_bucket, list) or not all(isinstance(item, Mapping) for item in raw_bucket):
            raise ExecutionEvidenceError(f"Execution evidence {field_name} must be an array of objects")
        projected: list[dict[str, Any]] = []
        for raw_item in raw_bucket:
            if input_id is not None and raw_item.get("input_id") != input_id:
                continue
            item = dict(raw_item)
            artifacts = item.pop("artifacts", None)
            if field_name == "produced_artifacts":
                if not isinstance(artifacts, list):
                    raise ExecutionEvidenceError("Produced execution evidence must carry artifact records")
                item["artifact_count"] = len(artifacts)
            projected.append(item)
        return projected

    return {
        "artifact_outcome_authority": {
            "schema": MANIFEST_OUTCOME_PROJECTION_SCHEMA,
            "source_schema": EXECUTION_EVIDENCE_SCHEMA,
            "execution_evidence_path": evidence_path,
            "record_authority": "detached_execution_evidence",
            "scope": scope,
        },
        "requested_artifacts": projected_requested,
        "produced_artifacts": project_bucket("produced_artifacts"),
        "omitted_artifacts": project_bucket("omitted_artifacts"),
        "failed_artifacts": project_bucket("failed_artifacts"),
    }


def _validate_evidence_path(evidence_path: str) -> None:
    """Validate one canonical output-root-relative evidence path."""

    if (
        not isinstance(evidence_path, str)
        or not evidence_path
        or len(evidence_path) > 4096
        or "\x00" in evidence_path
        or "\\" in evidence_path
        or bool(PureWindowsPath(evidence_path).drive)
        or evidence_path.startswith("/")
        or PurePosixPath(evidence_path).as_posix() != evidence_path
        or PurePosixPath(evidence_path).as_posix() == "."
        or ".." in PurePosixPath(evidence_path).parts
    ):
        raise ExecutionEvidenceError("Manifest execution_evidence_path must be a canonical confined relative path")


def _projection_from_validated_rows(
    plan: CanonicalExecutionPlan,
    execution_rows: Sequence[InputExecution],
    evidence_path: str,
) -> dict[str, Any]:
    """Project already-validated runtime rows without reparsing the plan."""

    distinct_backends = sorted(
        {row.executed_backend for row in execution_rows if isinstance(row.executed_backend, str) and row.executed_backend}
    )
    projection: dict[str, Any] = {
        "plan_schema": plan.schema,
        "plan_fingerprint": plan.plan_fingerprint_sha256,
        "config_fingerprint_sha256": plan.config_fingerprint_sha256,
        "planned_backend": plan.planned_backend,
        "candidate_fallback_chain": list(plan.candidate_fallback_chain),
        "executed_backend": distinct_backends[0] if len(distinct_backends) == 1 else None,
        "executed_backend_by_input": [
            {
                "input_id": row.input_id,
                "executed_backend": row.executed_backend,
                "status": row.status,
            }
            for row in execution_rows
        ],
        "requested_artifacts": list(plan.requested_outputs),
        "execution_evidence_path": evidence_path,
    }
    return projection


def _validated_input_executions(
    plan: CanonicalExecutionPlan,
    input_executions: Sequence[InputExecution],
    *,
    require_all: bool,
) -> tuple[InputExecution, ...]:
    plan_ids = tuple(item.input_id for item in plan.inputs)
    return _validated_input_executions_against(
        plan_ids,
        frozenset(plan.candidate_fallback_chain),
        input_executions,
        require_all=require_all,
    )


def _validated_input_executions_against(
    plan_ids: Sequence[str],
    allowed_backends: frozenset[str],
    input_executions: Sequence[InputExecution],
    *,
    require_all: bool,
) -> tuple[InputExecution, ...]:
    """Validate runtime rows against pre-indexed plan authority."""

    plan_id_set = frozenset(plan_ids)
    rows_by_id: dict[str, InputExecution] = {}

    for row in input_executions:
        if not isinstance(row, InputExecution):
            raise TypeError("input_executions must contain InputExecution values")
        if row.input_id not in plan_id_set:
            raise ExecutionEvidenceError(f"Runtime outcome references unknown input id {row.input_id!r}")
        if row.input_id in rows_by_id:
            raise ExecutionEvidenceError(f"Duplicate runtime outcome for input id {row.input_id!r}")
        if row.status not in _INPUT_STATUSES:
            raise ExecutionEvidenceError(f"Unsupported runtime input status {row.status!r}")
        if row.executed_backend is not None and row.executed_backend not in allowed_backends:
            raise ExecutionEvidenceError(
                f"Executed backend {row.executed_backend!r} is outside the prepared candidate authority"
            )
        if row.status == "ok" and row.executed_backend is None:
            raise ExecutionEvidenceError(f"Successful input {row.input_id!r} is missing its executed backend")
        rows_by_id[row.input_id] = row

    if require_all and set(rows_by_id) != plan_id_set:
        missing = sorted(plan_id_set - set(rows_by_id))
        raise ExecutionEvidenceError(f"Runtime outcomes do not cover the prepared input selection: missing={missing}")
    return tuple(rows_by_id[input_id] for input_id in plan_ids if input_id in rows_by_id)


class _ManifestPlanProjector:
    """Reusable, validated plan projection with O(1) input membership checks."""

    def __init__(self, plan: CanonicalExecutionPlan, evidence_path: str):
        _validate_prepared_plan(plan)
        _validate_evidence_path(evidence_path)
        self.plan = plan
        self.evidence_path = evidence_path
        self._plan_ids = tuple(item.input_id for item in plan.inputs)
        self._allowed_backends = frozenset(plan.candidate_fallback_chain)

    def validated_rows(
        self,
        input_executions: Sequence[InputExecution],
        *,
        require_all: bool,
    ) -> tuple[InputExecution, ...]:
        return _validated_input_executions_against(
            self._plan_ids,
            self._allowed_backends,
            input_executions,
            require_all=require_all,
        )

    def build(self, input_executions: Sequence[InputExecution] = ()) -> dict[str, Any]:
        rows = self.validated_rows(input_executions, require_all=False)
        return _projection_from_validated_rows(self.plan, rows, self.evidence_path)

    def build_validated(self, input_executions: Sequence[InputExecution]) -> dict[str, Any]:
        return _projection_from_validated_rows(self.plan, input_executions, self.evidence_path)


@dataclass(frozen=True)
class _ManifestValidationContext:
    """Precomputed projections and indexes reused across manifest checks."""

    projector: _ManifestPlanProjector
    execution_rows: tuple[InputExecution, ...]
    execution_by_id: Mapping[str, InputExecution]
    plan_input_by_id: Mapping[str, Any]
    full_projection: Mapping[str, Any]
    projection_by_input: Mapping[str, Mapping[str, Any]]
    authoritative_plan: Mapping[str, Any]
    manifest_budget: _ManifestBudget
    batch_binding: _BatchRunCardBinding
    carrier_outcome_projections: list[tuple[str, Optional[str], Optional[Mapping[str, Any]]]]
    expected_carrier_records: Mapping[str, tuple[str, int]]
    observed_expected_carrier_paths: set[str]


def _manifest_validation_context(
    projector: _ManifestPlanProjector,
    execution_rows: tuple[InputExecution, ...],
    *,
    expected_carrier_records: Optional[Mapping[str, tuple[str, int]]] = None,
) -> _ManifestValidationContext:
    execution_by_id = {row.input_id: row for row in execution_rows}
    return _ManifestValidationContext(
        projector=projector,
        execution_rows=execution_rows,
        execution_by_id=execution_by_id,
        plan_input_by_id={item.input_id: item for item in projector.plan.inputs},
        full_projection=projector.build_validated(execution_rows),
        projection_by_input={input_id: projector.build_validated((row,)) for input_id, row in execution_by_id.items()},
        authoritative_plan=projector.plan.to_payload(),
        manifest_budget=_ManifestBudget(),
        batch_binding=_BatchRunCardBinding(),
        carrier_outcome_projections=[],
        expected_carrier_records=expected_carrier_records or {},
        observed_expected_carrier_paths=set(),
    )


def _validate_expected_carrier_capture(
    capture: _ArtifactCapture,
    *,
    context: _ManifestValidationContext,
) -> None:
    """Bind an observed carrier to the exact bytes frozen before indexing."""

    path = capture.record["path"]
    expected = context.expected_carrier_records.get(path)
    if expected is None:
        return
    context.observed_expected_carrier_paths.add(path)
    if capture.record.get("sha256") != expected[0] or capture.record.get("size_bytes") != expected[1]:
        raise ArtifactEvidenceError("artifact_changed", f"Prepared artifact carrier changed before completion: {path}")


def _canonicalize_top_level_alias(path: Path) -> Path:
    """Canonicalize only standard macOS ``/tmp`` and ``/var`` aliases.

    Resolving the complete candidate would follow symlinks inside the output
    tree and erase the components descriptor traversal must reject. Treating
    arbitrary top-level symlinks as aliases would likewise broaden the output
    authority, so only the standard temporary-directory spelling is eligible.
    """

    if sys.platform != "darwin" or not path.is_absolute() or len(path.parts) < 2:
        return path
    alias_name = path.parts[1]
    expected_targets = {
        "tmp": Path("/private/tmp"),
        "var": Path("/private/var"),
    }
    expected_target = expected_targets.get(alias_name)
    if expected_target is None:
        return path
    top_level = Path(path.anchor) / alias_name
    try:
        canonical_top_level = top_level.resolve(strict=True)
    except (OSError, RuntimeError):
        return path
    if canonical_top_level != expected_target:
        return path
    return canonical_top_level.joinpath(*path.parts[2:])


def _map_safe_open_error(exc: OSError, *, missing_code: str = "artifact_missing") -> str:
    if exc.errno == errno.ENOENT:
        return missing_code
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return "symlink_forbidden"
    return "artifact_unreadable"


@contextmanager
def _pin_output_root(output_root: Path) -> Iterator[_PinnedOutputRoot]:
    """Open an output root from ``/`` without following canonical components."""

    if not _HAS_SECURE_DIR_FD_OPEN:
        raise ArtifactEvidenceError(
            "secure_traversal_unavailable",
            "Prepared execution evidence requires secure descriptor-relative filesystem traversal",
        )

    lexical_root = Path(os.path.abspath(os.fspath(output_root)))
    try:
        # Only the platform's root-level alias is normalized. Following an
        # arbitrary output-root ancestor here would let a pre-call symlink
        # swap redirect the supposedly confined authority before it is pinned.
        canonical_root = _canonicalize_top_level_alias(lexical_root)
        expected = os.lstat(canonical_root)
    except (OSError, RuntimeError) as exc:
        raise ArtifactEvidenceError("output_root_unavailable", "Execution evidence output root is unavailable") from exc
    if not stat.S_ISDIR(expected.st_mode):
        raise ArtifactEvidenceError("output_root_unavailable", "Execution evidence output root is not a directory")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    component_names = tuple(canonical_root.parts[1:])
    component_descriptors: list[int] = []
    try:
        component_descriptors.append(os.open(canonical_root.anchor, directory_flags))
        for component in component_names:
            component_descriptors.append(os.open(component, directory_flags, dir_fd=component_descriptors[-1]))
        actual = os.fstat(component_descriptors[-1])
        if not stat.S_ISDIR(actual.st_mode) or actual.st_dev != expected.st_dev or actual.st_ino != expected.st_ino:
            raise ArtifactEvidenceError(
                "output_root_changed",
                "Execution evidence output root changed while its authority was pinned",
            )
        pinned = _PinnedOutputRoot(
            canonical_path=canonical_root,
            lexical_path=lexical_root,
            descriptor=component_descriptors[-1],
            device=actual.st_dev,
            inode=actual.st_ino,
            component_names=component_names,
            component_descriptors=tuple(component_descriptors),
            component_identities=tuple(
                (component_stat.st_dev, component_stat.st_ino)
                for component_stat in (os.fstat(descriptor) for descriptor in component_descriptors)
            ),
        )
        _validate_pinned_root_namespace(pinned)
        yield pinned
    except ArtifactEvidenceError:
        raise
    except OSError as exc:
        code = _map_safe_open_error(exc, missing_code="output_root_unavailable")
        raise ArtifactEvidenceError(code, "Execution evidence output root cannot be pinned safely") from exc
    finally:
        for descriptor in reversed(component_descriptors):
            os.close(descriptor)


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


def _validate_pinned_root_namespace(root: _PinnedOutputRoot) -> None:
    """Prove every retained ancestor still names the next pinned directory."""

    try:
        for descriptor, expected_identity in zip(root.component_descriptors, root.component_identities):
            descriptor_stat = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(descriptor_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino) != expected_identity
            ):
                raise ArtifactEvidenceError(
                    "output_root_changed",
                    "A pinned output-root ancestor changed identity",
                )
        for index, component in enumerate(root.component_names):
            reopened = os.open(component, _directory_open_flags(), dir_fd=root.component_descriptors[index])
            try:
                reopened_stat = os.fstat(reopened)
            finally:
                os.close(reopened)
            if (reopened_stat.st_dev, reopened_stat.st_ino) != root.component_identities[index + 1]:
                raise ArtifactEvidenceError(
                    "output_root_changed",
                    "An output-root ancestor no longer names the pinned directory",
                )
    except ArtifactEvidenceError:
        raise
    except OSError as exc:
        raise ArtifactEvidenceError(
            "output_root_changed",
            "The pinned output-root namespace is no longer intact",
        ) from exc


def _validate_relative_parent_namespace(
    root: _PinnedOutputRoot,
    relative_path: str,
    parent_descriptor: int,
) -> None:
    """Prove a held publication parent remains reachable below ``root``."""

    _validate_pinned_root_namespace(root)
    expected_parent = os.fstat(parent_descriptor)
    current_descriptor = os.dup(root.descriptor)
    try:
        for component in PurePosixPath(relative_path).parts[:-1]:
            next_descriptor = os.open(component, _directory_open_flags(), dir_fd=current_descriptor)
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        actual_parent = os.fstat(current_descriptor)
        if (
            not stat.S_ISDIR(expected_parent.st_mode)
            or not stat.S_ISDIR(actual_parent.st_mode)
            or actual_parent.st_dev != expected_parent.st_dev
            or actual_parent.st_ino != expected_parent.st_ino
        ):
            raise ArtifactEvidenceError(
                "output_root_changed",
                "Execution evidence publication parent left the pinned output root",
            )
    except ArtifactEvidenceError:
        raise
    except OSError as exc:
        raise ArtifactEvidenceError(
            "output_root_changed",
            "Execution evidence publication parent is no longer securely reachable",
        ) from exc
    finally:
        os.close(current_descriptor)


def _artifact_media_metadata(relative_path: str) -> tuple[str, Optional[str]]:
    suffix = Path(relative_path).suffix.lower()
    # Keep evidence byte-stable across hosts. ``mimetypes.guess_type`` may
    # consult the operating system's MIME database, so even familiar suffixes
    # can acquire environment-dependent values.
    media_types = {
        ".json": "application/json",
        ".npy": "application/x-npy",
        ".npz": "application/x-npz",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
        ".exr": "image/x-exr",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "model/obj",
        ".mtl": "model/mtl",
        ".ply": "model/ply",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".log": "text/plain",
    }
    media_type = media_types.get(suffix)
    return media_type or "application/octet-stream", suffix[1:] if suffix else None


def _open_relative_parent(root: _PinnedOutputRoot, relative_path: str) -> tuple[int, str]:
    parts = PurePosixPath(relative_path).parts
    current_descriptor = os.dup(root.descriptor)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        for component in parts[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=current_descriptor)
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        return current_descriptor, parts[-1]
    except BaseException:
        os.close(current_descriptor)
        raise


def _open_or_create_relative_parent(root: _PinnedOutputRoot, relative_path: str) -> tuple[int, str]:
    """Open a confined parent, creating missing directories without following links."""

    if os.mkdir not in os.supports_dir_fd:
        raise ArtifactEvidenceError(
            "secure_traversal_unavailable",
            "Prepared artifact carriers require descriptor-relative directory creation",
        )
    parts = PurePosixPath(relative_path).parts
    current_descriptor = os.dup(root.descriptor)
    directory_flags = _directory_open_flags()
    try:
        for component in parts[:-1]:
            _validate_pinned_root_namespace(root)
            try:
                next_descriptor = os.open(component, directory_flags, dir_fd=current_descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o755, dir_fd=current_descriptor)
                    os.fsync(current_descriptor)
                except FileExistsError:
                    # A cooperating publisher may have created the same
                    # deterministic batch directory between open and mkdir.
                    pass
                next_descriptor = os.open(component, directory_flags, dir_fd=current_descriptor)
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        _validate_relative_parent_namespace(root, relative_path, current_descriptor)
        return current_descriptor, parts[-1]
    except ArtifactEvidenceError:
        os.close(current_descriptor)
        raise
    except OSError as exc:
        os.close(current_descriptor)
        code = _map_safe_open_error(exc)
        raise ArtifactEvidenceError(code, "Artifact carrier parent cannot be created safely") from exc


def _open_confined_artifact(root: _PinnedOutputRoot, candidate: Path) -> tuple[int, str]:
    relative_path = root.confined_relative_path(candidate)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    parent_descriptor: Optional[int] = None
    descriptor: Optional[int] = None
    try:
        _validate_pinned_root_namespace(root)
        parent_descriptor, final_component = _open_relative_parent(root, relative_path)
        descriptor = os.open(final_component, flags, dir_fd=parent_descriptor)
        _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
        return descriptor, relative_path
    except ArtifactEvidenceError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        code = _map_safe_open_error(exc)
        message = (
            f"Artifact path is unavailable: {candidate}"
            if code == "artifact_missing"
            else f"Artifact cannot be opened safely: {candidate}"
        )
        raise ArtifactEvidenceError(code, message) from exc
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _validate_confined_entry_identity(
    root: _PinnedOutputRoot,
    relative_path: str,
    expected: os.stat_result,
    *,
    context: str,
) -> None:
    """Prove a confined name still resolves to the exact captured inode."""

    descriptor, _ = _open_confined_artifact(root, Path(relative_path))
    try:
        actual = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(actual.st_mode)
        or actual.st_nlink != 1
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
        or actual.st_size != expected.st_size
        or actual.st_mtime_ns != expected.st_mtime_ns
        or actual.st_ctime_ns != expected.st_ctime_ns
    ):
        raise ArtifactEvidenceError(
            "artifact_changed",
            f"{context} no longer names the captured inode",
        )


def _capture_artifact(
    root: _PinnedOutputRoot,
    candidate: Path,
    *,
    budget: _CaptureBudget,
    max_bytes: Optional[int] = None,
) -> _ArtifactCapture:
    descriptor, relative_path = _open_confined_artifact(root, candidate)
    effective_max_bytes = _MAX_ARTIFACT_BYTES if max_bytes is None else min(max_bytes, _MAX_ARTIFACT_BYTES)

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactEvidenceError("not_regular_file", f"Artifact is not a regular file: {candidate}")
        if before.st_nlink != 1:
            raise ArtifactEvidenceError("hardlink_forbidden", f"Artifact has hardlink aliases: {candidate}")
        if before.st_size > effective_max_bytes:
            raise ArtifactEvidenceError(
                "artifact_too_large",
                f"Artifact {relative_path!r} exceeds its pre-read byte limit of {effective_max_bytes}",
            )
        budget.check_known_size(before.st_size, context=f"Artifact {relative_path!r}")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if byte_count > effective_max_bytes:
                raise ArtifactEvidenceError(
                    "artifact_too_large",
                    f"Artifact {relative_path!r} exceeds its byte limit",
                )
            if budget.total_bytes + byte_count > _MAX_AGGREGATE_ARTIFACT_BYTES:
                raise ArtifactEvidenceError(
                    "aggregate_artifact_bytes_exceeded",
                    "Artifact capture exceeds the aggregate byte limit",
                )
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or after.st_nlink != 1
            or byte_count != after.st_size
        ):
            raise ArtifactEvidenceError("artifact_changed", f"Artifact changed while evidence was captured: {candidate}")
        _validate_confined_entry_identity(
            root,
            relative_path,
            after,
            context=f"Artifact {relative_path!r}",
        )
    except ArtifactEvidenceError:
        raise
    except OSError as exc:
        raise ArtifactEvidenceError("artifact_unreadable", f"Artifact cannot be read safely: {candidate}") from exc
    finally:
        os.close(descriptor)

    budget.commit(byte_count)
    media_type, extension = _artifact_media_metadata(relative_path)
    return _ArtifactCapture(
        record={
            "path": relative_path,
            "sha256": digest.hexdigest(),
            "size_bytes": byte_count,
            "media_type": media_type,
            "file_extension": extension,
        },
        identity=(before.st_dev, before.st_ino),
    )


def _read_confined_artifact_bytes(
    root: _PinnedOutputRoot,
    artifact_path: Path,
    *,
    context: str,
    max_bytes: Optional[int] = None,
    manifest_budget: Optional[_ManifestBudget] = None,
) -> tuple[bytes, str, os.stat_result]:
    descriptor, relative_path = _open_confined_artifact(root, artifact_path)
    data = bytearray()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactEvidenceError("not_regular_file", f"{context} is not a regular file")
        if before.st_nlink != 1:
            raise ArtifactEvidenceError("hardlink_forbidden", f"{context} has hardlink aliases")
        effective_max_bytes = _MAX_EVIDENCE_BYTES if max_bytes is None else max_bytes
        if before.st_size > effective_max_bytes:
            raise ArtifactEvidenceError("artifact_too_large", f"{context} exceeds {effective_max_bytes} bytes")
        if manifest_budget is not None:
            manifest_budget.reserve_decode(before.st_size, context=context)
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            data.extend(chunk)
            byte_count += len(chunk)
            if byte_count > effective_max_bytes:
                raise ArtifactEvidenceError("artifact_too_large", f"{context} exceeds {effective_max_bytes} bytes")
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or after.st_nlink != 1
            or byte_count != after.st_size
        ):
            raise ArtifactEvidenceError("artifact_changed", f"{context} changed while it was read")
        _validate_confined_entry_identity(
            root,
            relative_path,
            after,
            context=context,
        )
    except ExecutionEvidenceError:
        raise
    except OSError as exc:
        raise ArtifactEvidenceError("artifact_unreadable", f"{context} cannot be read safely") from exc
    finally:
        os.close(descriptor)
    return bytes(data), relative_path, after


def read_confined_artifact_snapshot(
    output_root: Path,
    artifact_path: Path,
    *,
    context: str,
    max_bytes: int,
) -> ConfinedArtifactSnapshot:
    """Read one output artifact through bounded, no-follow root authority."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    effective_max_bytes = min(max_bytes, _MAX_EVIDENCE_BYTES)
    with _pin_output_root(Path(output_root)) as root:
        data, relative_path, identity = _read_confined_artifact_bytes(
            root,
            Path(artifact_path),
            context=context,
            max_bytes=effective_max_bytes,
        )
    return ConfinedArtifactSnapshot(
        data=data,
        relative_path=relative_path,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        device=identity.st_dev,
        inode=identity.st_ino,
    )


def copy_confined_artifact(
    output_root: Path,
    source_path: Path,
    destination_path: Path,
    *,
    budget: Optional[ConfinedArtifactCopyBudget] = None,
    transform_bytes: Optional[Callable[[bytes], bytes]] = None,
) -> ConfinedArtifactCopy:
    """Stream one artifact into a distinct, immutable batch carrier.

    Both names remain under one descriptor-pinned output root. The source is
    rebound before and after publication, while the destination is written to
    a unique temporary inode and linked into place with no-replace semantics
    only after a complete fsync. Carrier destinations are create-only. When
    ``transform_bytes`` is supplied, the bounded source bytes are transformed
    while the destination inode is still unpublished.
    """

    with _pin_output_root(Path(output_root)) as root:
        source_relative_path = root.confined_relative_path(Path(source_path))
        destination_relative_path = root.confined_relative_path(Path(destination_path))
        if source_relative_path == destination_relative_path:
            raise ArtifactEvidenceError("artifact_changed", "Artifact carrier source and destination are identical")
        source_descriptor, opened_source_relative_path = _open_confined_artifact(root, Path(source_path))
        if opened_source_relative_path != source_relative_path:  # pragma: no cover - defensive invariant
            os.close(source_descriptor)
            raise ArtifactEvidenceError("artifact_changed", "Artifact carrier source identity changed")

        parent_descriptor: Optional[int] = None
        temporary_descriptor: Optional[int] = None
        published_descriptor: Optional[int] = None
        temporary_name: Optional[str] = None
        final_component = ""
        temporary_identity: Optional[tuple[int, int]] = None
        published_identity: Optional[tuple[int, int]] = None
        renamed = False
        temporary_unlinked = False
        completed = False
        digest = hashlib.sha256()
        source_data = bytearray() if transform_bytes is not None else None
        source_byte_count = 0
        byte_count = 0
        try:
            source_before = os.fstat(source_descriptor)
            if not stat.S_ISREG(source_before.st_mode):
                raise ArtifactEvidenceError("not_regular_file", "Artifact carrier source is not a regular file")
            if source_before.st_nlink != 1:
                raise ArtifactEvidenceError("hardlink_forbidden", "Artifact carrier source has hardlink aliases")
            if source_before.st_size > _MAX_ARTIFACT_BYTES:
                raise ArtifactEvidenceError(
                    "artifact_too_large",
                    f"Artifact carrier source exceeds {_MAX_ARTIFACT_BYTES} bytes",
                )
            if budget is not None and transform_bytes is None:
                budget.check_known_size(source_before.st_size)

            parent_descriptor, final_component = _open_or_create_relative_parent(
                root,
                destination_relative_path,
            )
            try:
                destination_before = os.stat(final_component, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                destination_before = None
            if destination_before is not None:
                if source_before.st_dev == destination_before.st_dev and source_before.st_ino == destination_before.st_ino:
                    raise ArtifactEvidenceError(
                        "artifact_changed",
                        "Artifact carrier destination aliases its source inode",
                    )
                if not stat.S_ISREG(destination_before.st_mode) or destination_before.st_nlink != 1:
                    raise ArtifactEvidenceError(
                        "artifact_unreadable",
                        "Artifact carrier destination is not a unique regular file",
                    )
                raise ArtifactEvidenceError("artifact_changed", "Artifact carrier destination already exists")

            temporary_name = f".{final_component}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
            create_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            temporary_descriptor = os.open(
                temporary_name,
                create_flags,
                _DEFAULT_NEW_FILE_MODE,
                dir_fd=parent_descriptor,
            )
            temporary_before = os.fstat(temporary_descriptor)
            if not stat.S_ISREG(temporary_before.st_mode) or temporary_before.st_nlink != 1:
                raise ArtifactEvidenceError("artifact_unreadable", "Unsafe artifact carrier temporary file")
            temporary_identity = (temporary_before.st_dev, temporary_before.st_ino)
            if temporary_identity == (source_before.st_dev, source_before.st_ino):
                raise ArtifactEvidenceError("artifact_changed", "Artifact carrier copy reused its source inode")

            while True:
                chunk = os.read(source_descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                source_byte_count += len(chunk)
                if source_byte_count > _MAX_ARTIFACT_BYTES:
                    raise ArtifactEvidenceError("artifact_too_large", "Artifact carrier source exceeds its byte limit")
                if source_data is not None:
                    if source_byte_count > _MAX_BOUND_MANIFEST_BYTES:
                        raise ArtifactEvidenceError(
                            "artifact_too_large",
                            f"Transformable artifact carrier exceeds {_MAX_BOUND_MANIFEST_BYTES} bytes",
                        )
                    source_data.extend(chunk)
                else:
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if budget is not None:
                        budget.check_known_size(byte_count)
                    remaining = memoryview(chunk)
                    while remaining:
                        written = os.write(temporary_descriptor, remaining)
                        if written <= 0:
                            raise OSError(errno.EIO, "short write while copying artifact carrier")
                        remaining = remaining[written:]

            source_after = os.fstat(source_descriptor)
            if (
                not stat.S_ISREG(source_after.st_mode)
                or source_after.st_nlink != 1
                or source_before.st_dev != source_after.st_dev
                or source_before.st_ino != source_after.st_ino
                or source_before.st_size != source_after.st_size
                or source_before.st_mtime_ns != source_after.st_mtime_ns
                or source_before.st_ctime_ns != source_after.st_ctime_ns
                or source_byte_count != source_after.st_size
            ):
                raise ArtifactEvidenceError("artifact_changed", "Artifact carrier source changed while copied")
            _validate_confined_entry_identity(
                root,
                source_relative_path,
                source_after,
                context="Artifact carrier source",
            )

            if source_data is not None:
                assert transform_bytes is not None
                transformed = transform_bytes(bytes(source_data))
                if not isinstance(transformed, bytes):
                    raise TypeError("transform_bytes must return bytes")
                if len(transformed) > _MAX_BOUND_MANIFEST_BYTES:
                    raise ArtifactEvidenceError(
                        "artifact_too_large",
                        f"Transformed artifact carrier exceeds {_MAX_BOUND_MANIFEST_BYTES} bytes",
                    )
                if budget is not None:
                    budget.check_known_size(len(transformed))
                digest.update(transformed)
                byte_count = len(transformed)
                remaining = memoryview(transformed)
                while remaining:
                    written = os.write(temporary_descriptor, remaining)
                    if written <= 0:
                        raise OSError(errno.EIO, "short write while transforming artifact carrier")
                    remaining = remaining[written:]

            os.fsync(temporary_descriptor)
            os.fchmod(temporary_descriptor, _DEFAULT_NEW_FILE_MODE)
            os.fsync(temporary_descriptor)
            temporary_after = os.fstat(temporary_descriptor)
            if (
                temporary_after.st_dev != temporary_before.st_dev
                or temporary_after.st_ino != temporary_before.st_ino
                or temporary_after.st_nlink != 1
                or temporary_after.st_size != byte_count
                or stat.S_IMODE(temporary_after.st_mode) != _DEFAULT_NEW_FILE_MODE
            ):
                raise ArtifactEvidenceError("artifact_changed", "Artifact carrier temporary file changed")
            published_identity = temporary_identity
            os.close(temporary_descriptor)
            temporary_descriptor = None

            _validate_relative_parent_namespace(root, destination_relative_path, parent_descriptor)
            _validate_named_entry_identity(parent_descriptor, temporary_name, temporary_after)
            if not _HAS_SECURE_NO_REPLACE_LINK:
                raise ArtifactEvidenceError(
                    "secure_traversal_unavailable",
                    "Prepared artifact carriers require descriptor-relative no-follow hard linking",
                )
            try:
                os.link(
                    temporary_name,
                    final_component,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ArtifactEvidenceError(
                    "artifact_changed",
                    "Artifact carrier destination appeared before publication",
                ) from exc
            renamed = True
            linked_stat = os.stat(final_component, dir_fd=parent_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(linked_stat.st_mode)
                or (linked_stat.st_dev, linked_stat.st_ino) != temporary_identity
                or linked_stat.st_nlink != 2
            ):
                raise ArtifactEvidenceError("artifact_changed", "Artifact carrier no-replace link changed")
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_unlinked = True
            _validate_relative_parent_namespace(root, destination_relative_path, parent_descriptor)
            published_descriptor = os.open(
                final_component,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            published_before = os.fstat(published_descriptor)
            if (
                not stat.S_ISREG(published_before.st_mode)
                or published_before.st_nlink != 1
                or (published_before.st_dev, published_before.st_ino) != temporary_identity
                or published_before.st_size != byte_count
            ):
                raise ArtifactEvidenceError("artifact_changed", "Published artifact carrier identity changed")

            published_digest = hashlib.sha256()
            published_byte_count = 0
            while True:
                chunk = os.read(published_descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                published_digest.update(chunk)
                published_byte_count += len(chunk)
            published_after = os.fstat(published_descriptor)
            if (
                published_before.st_dev != published_after.st_dev
                or published_before.st_ino != published_after.st_ino
                or published_before.st_size != published_after.st_size
                or published_before.st_mtime_ns != published_after.st_mtime_ns
                or published_before.st_ctime_ns != published_after.st_ctime_ns
                or published_after.st_nlink != 1
                or published_byte_count != byte_count
                or published_digest.digest() != digest.digest()
            ):
                raise ArtifactEvidenceError("artifact_changed", "Published artifact carrier bytes changed")
            _validate_named_entry_identity(parent_descriptor, final_component, published_after)
            os.fsync(parent_descriptor)
            _validate_relative_parent_namespace(root, destination_relative_path, parent_descriptor)
            _validate_named_entry_identity(parent_descriptor, final_component, published_after)
            _validate_confined_entry_identity(
                root,
                source_relative_path,
                source_after,
                context="Artifact carrier source",
            )
            if budget is not None:
                budget.commit(byte_count)
            completed = True
        except ArtifactEvidenceError:
            raise
        except OSError as exc:
            code = _map_safe_open_error(exc)
            raise ArtifactEvidenceError(code, "Artifact carrier cannot be copied safely") from exc
        finally:
            if parent_descriptor is not None and renamed and not completed and published_identity is not None:
                if temporary_name is not None and not temporary_unlinked:
                    _unlink_carrier_link_pair_if_identity(
                        root,
                        destination_relative_path,
                        parent_descriptor,
                        temporary_name,
                        final_component,
                        published_identity,
                    )
                    _unlink_named_entry_if_identity(
                        root,
                        destination_relative_path,
                        parent_descriptor,
                        temporary_name,
                        published_identity,
                    )
                else:
                    _unlink_named_entry_if_identity(
                        root,
                        destination_relative_path,
                        parent_descriptor,
                        final_component,
                        published_identity,
                    )
            if parent_descriptor is not None and temporary_name is not None and not renamed and temporary_identity is not None:
                _unlink_named_entry_if_identity(
                    root,
                    destination_relative_path,
                    parent_descriptor,
                    temporary_name,
                    temporary_identity,
                )
            if published_descriptor is not None:
                os.close(published_descriptor)
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            if parent_descriptor is not None:
                os.close(parent_descriptor)
            os.close(source_descriptor)

    if published_identity is None:  # pragma: no cover - completed-copy invariant
        raise ArtifactEvidenceError("artifact_changed", "Artifact carrier lost its published identity")
    return ConfinedArtifactCopy(
        source_relative_path=source_relative_path,
        relative_path=destination_relative_path,
        sha256=digest.hexdigest(),
        size_bytes=byte_count,
        device=published_identity[0],
        inode=published_identity[1],
    )


def discard_confined_artifact_copy(
    output_root: Path,
    destination_path: Path,
    *,
    expected: ConfinedArtifactCopy,
) -> bool:
    """Remove one failed-batch carrier only while its exact inode is retained."""

    parent_descriptor: Optional[int] = None
    try:
        with _pin_output_root(Path(output_root)) as root:
            relative_path = root.confined_relative_path(Path(destination_path))
            if relative_path != expected.relative_path:
                return False
            try:
                parent_descriptor, final_component = _open_relative_parent(root, relative_path)
                _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
                current = os.stat(final_component, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return True
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or current.st_dev != expected.device
                or current.st_ino != expected.inode
            ):
                return False
            os.unlink(final_component, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
            try:
                os.stat(final_component, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return True
            return False
    except (ArtifactEvidenceError, OSError):
        return False
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def restore_confined_artifact_bytes_if_matches(
    output_root: Path,
    artifact_path: Path,
    replacement_bytes: bytes,
    *,
    expected: ConfinedArtifactSnapshot,
) -> bool:
    """Restore bytes only while the name retains an exact captured inode."""

    if (
        expected.device is None
        or expected.inode is None
        or not isinstance(replacement_bytes, bytes)
        or len(replacement_bytes) > _MAX_BOUND_MANIFEST_BYTES
    ):
        return False
    parent_descriptor: Optional[int] = None
    temporary_descriptor: Optional[int] = None
    temporary_name: Optional[str] = None
    temporary_identity: Optional[tuple[int, int]] = None
    renamed = False
    try:
        with _pin_output_root(Path(output_root)) as root:
            current_data, relative_path, current = _read_confined_artifact_bytes(
                root,
                Path(artifact_path),
                context="Prepared manifest rollback candidate",
                max_bytes=_MAX_BOUND_MANIFEST_BYTES,
            )
            if (
                relative_path != expected.relative_path
                or current.st_dev != expected.device
                or current.st_ino != expected.inode
                or current_data != expected.data
                or hashlib.sha256(current_data).hexdigest() != expected.sha256
            ):
                return False
            parent_descriptor, final_component = _open_relative_parent(root, relative_path)
            _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
            _validate_named_entry_identity(parent_descriptor, final_component, current)
            destination_mode = stat.S_IMODE(current.st_mode)

            temporary_name = f".{final_component}.{os.getpid()}.{secrets.token_hex(8)}.rollback.tmp"
            temporary_descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                destination_mode,
                dir_fd=parent_descriptor,
            )
            temporary_before = os.fstat(temporary_descriptor)
            if not stat.S_ISREG(temporary_before.st_mode) or temporary_before.st_nlink != 1:
                return False
            temporary_identity = (temporary_before.st_dev, temporary_before.st_ino)
            remaining = memoryview(replacement_bytes)
            while remaining:
                written = os.write(temporary_descriptor, remaining)
                if written <= 0:
                    return False
                remaining = remaining[written:]
            os.fchmod(temporary_descriptor, destination_mode)
            os.fsync(temporary_descriptor)
            temporary_after = os.fstat(temporary_descriptor)
            if (
                (temporary_after.st_dev, temporary_after.st_ino) != temporary_identity
                or temporary_after.st_nlink != 1
                or temporary_after.st_size != len(replacement_bytes)
                or stat.S_IMODE(temporary_after.st_mode) != destination_mode
            ):
                return False
            os.close(temporary_descriptor)
            temporary_descriptor = None

            _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
            _validate_named_entry_identity(parent_descriptor, final_component, current)
            _validate_named_entry_identity(parent_descriptor, temporary_name, temporary_after)
            os.rename(
                temporary_name,
                final_component,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            renamed = True
            restored = os.stat(final_component, dir_fd=parent_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(restored.st_mode)
                or restored.st_nlink != 1
                or (restored.st_dev, restored.st_ino) != temporary_identity
                or restored.st_size != len(replacement_bytes)
                or stat.S_IMODE(restored.st_mode) != destination_mode
            ):
                return False
            os.fsync(parent_descriptor)
            _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
            _validate_named_entry_identity(parent_descriptor, final_component, restored)
            restored_data, restored_relative_path, restored_identity = _read_confined_artifact_bytes(
                root,
                Path(relative_path),
                context="Restored prepared manifest",
                max_bytes=_MAX_BOUND_MANIFEST_BYTES,
            )
            if (
                restored_relative_path != relative_path
                or (restored_identity.st_dev, restored_identity.st_ino) != temporary_identity
                or restored_data != replacement_bytes
            ):
                return False
            return True
    except (ArtifactEvidenceError, OSError):
        return False
    finally:
        if parent_descriptor is not None and temporary_name is not None and not renamed and temporary_identity is not None:
            try:
                with _pin_output_root(Path(output_root)) as cleanup_root:
                    relative_path = cleanup_root.confined_relative_path(Path(artifact_path))
                    _unlink_named_entry_if_identity(
                        cleanup_root,
                        relative_path,
                        parent_descriptor,
                        temporary_name,
                        temporary_identity,
                    )
            except (ArtifactEvidenceError, OSError):
                pass
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _validate_named_entry_identity(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    """Prove a directory entry still names the exact published regular file."""

    try:
        entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ArtifactEvidenceError(
            "artifact_changed",
            "Published execution evidence entry is unavailable",
        ) from exc
    if (
        not stat.S_ISREG(entry_stat.st_mode)
        or entry_stat.st_nlink != 1
        or entry_stat.st_dev != expected.st_dev
        or entry_stat.st_ino != expected.st_ino
        or entry_stat.st_size != expected.st_size
        or entry_stat.st_mtime_ns != expected.st_mtime_ns
        or entry_stat.st_ctime_ns != expected.st_ctime_ns
    ):
        raise ArtifactEvidenceError(
            "artifact_changed",
            "Published execution evidence entry no longer names the verified inode",
        )


def _unlink_named_entry_if_identity(
    root: _PinnedOutputRoot,
    relative_path: str,
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Best-effort cleanup for one publisher-owned entry.

    Cleanup is allowed only while the retained parent is still reachable
    through the pinned output namespace and the name still identifies the
    unique regular inode created by this publisher.  Any uncertainty retains
    the entry for operator reconciliation instead of removing a replacement.
    """

    try:
        _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or (current.st_dev, current.st_ino) != expected_identity:
            return
        os.unlink(name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except (ArtifactEvidenceError, OSError):
        # Error cleanup must neither mask the publication failure nor act on
        # an entry whose namespace or identity can no longer be proven.
        return


def _unlink_carrier_link_pair_if_identity(
    root: _PinnedOutputRoot,
    relative_path: str,
    parent_descriptor: int,
    temporary_name: str,
    final_name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Best-effort cleanup of the two names created by a no-replace link."""

    try:
        _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
        temporary_stat = os.stat(temporary_name, dir_fd=parent_descriptor, follow_symlinks=False)
        final_stat = os.stat(final_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(temporary_stat.st_mode)
            or not stat.S_ISREG(final_stat.st_mode)
            or (temporary_stat.st_dev, temporary_stat.st_ino) != expected_identity
            or (final_stat.st_dev, final_stat.st_ino) != expected_identity
            or temporary_stat.st_nlink != 2
            or final_stat.st_nlink != 2
        ):
            return
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        remaining = os.stat(final_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            stat.S_ISREG(remaining.st_mode)
            and (remaining.st_dev, remaining.st_ino) == expected_identity
            and remaining.st_nlink == 1
        ):
            os.unlink(final_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except (ArtifactEvidenceError, OSError):
        return


def _destination_entry_mode(parent_descriptor: int, name: str) -> int:
    """Return an existing regular destination's mode or the fixed new mode."""

    try:
        destination_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return _DEFAULT_NEW_FILE_MODE
    except OSError as exc:
        raise ArtifactEvidenceError(
            "artifact_unreadable",
            "Execution evidence destination mode cannot be inspected safely",
        ) from exc
    if not stat.S_ISREG(destination_stat.st_mode) or destination_stat.st_nlink != 1:
        raise ArtifactEvidenceError(
            "artifact_unreadable",
            "Execution evidence destination is not a unique regular file",
        )
    destination_mode = stat.S_IMODE(destination_stat.st_mode)
    if destination_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH) == 0:
        raise ArtifactEvidenceError(
            "artifact_unreadable",
            "Execution evidence destination mode does not permit post-publication verification",
        )
    return destination_mode


def _secure_atomic_write_bytes(root: _PinnedOutputRoot, relative_path: str, data: bytes) -> None:
    """Atomically publish bytes through a parent descriptor pinned under ``root``."""

    if len(data) > _MAX_EVIDENCE_BYTES:
        raise ArtifactEvidenceError("artifact_too_large", "Execution evidence sidecar exceeds its byte limit")
    parent_descriptor: Optional[int] = None
    temporary_name: Optional[str] = None
    temporary_descriptor: Optional[int] = None
    published_descriptor: Optional[int] = None
    renamed = False
    completed = False
    temporary_identity: Optional[tuple[int, int]] = None
    published_identity: Optional[tuple[int, int]] = None
    try:
        parent_descriptor, final_component = _open_relative_parent(root, relative_path)
        _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
        destination_mode = _destination_entry_mode(parent_descriptor, final_component)
        temporary_name = f".{final_component}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        create_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        temporary_descriptor = os.open(temporary_name, create_flags, 0o644, dir_fd=parent_descriptor)
        temporary_stat = os.fstat(temporary_descriptor)
        if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_nlink != 1:
            raise ArtifactEvidenceError("artifact_unreadable", "Unsafe execution evidence temporary file")
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)

        remaining = memoryview(data)
        while remaining:
            written = os.write(temporary_descriptor, remaining)
            if written <= 0:
                raise OSError(errno.EIO, "short write while publishing execution evidence")
            remaining = remaining[written:]
        os.fsync(temporary_descriptor)
        os.fchmod(temporary_descriptor, destination_mode)
        os.fsync(temporary_descriptor)
        written_stat = os.fstat(temporary_descriptor)
        if (
            written_stat.st_dev != temporary_stat.st_dev
            or written_stat.st_ino != temporary_stat.st_ino
            or written_stat.st_nlink != 1
            or written_stat.st_size != len(data)
            or stat.S_IMODE(written_stat.st_mode) != destination_mode
        ):
            raise ArtifactEvidenceError("artifact_changed", "Execution evidence temporary file changed during write")
        published_identity = temporary_identity
        os.close(temporary_descriptor)
        temporary_descriptor = None

        _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
        _validate_named_entry_identity(
            parent_descriptor,
            temporary_name,
            written_stat,
        )
        os.rename(
            temporary_name,
            final_component,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        renamed = True
        _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
        published_descriptor = os.open(
            final_component,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        published_stat = os.fstat(published_descriptor)
        if (
            published_stat.st_dev != temporary_stat.st_dev
            or published_stat.st_ino != temporary_stat.st_ino
            or published_stat.st_nlink != 1
            or published_stat.st_size != len(data)
            or stat.S_IMODE(published_stat.st_mode) != destination_mode
        ):
            raise ArtifactEvidenceError("artifact_changed", "Published execution evidence identity changed")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(published_descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        final_published_stat = os.fstat(published_descriptor)
        if (
            not stat.S_ISREG(final_published_stat.st_mode)
            or final_published_stat.st_nlink != 1
            or final_published_stat.st_dev != published_stat.st_dev
            or final_published_stat.st_ino != published_stat.st_ino
            or final_published_stat.st_size != published_stat.st_size
            or final_published_stat.st_mtime_ns != published_stat.st_mtime_ns
            or final_published_stat.st_ctime_ns != published_stat.st_ctime_ns
        ):
            raise ArtifactEvidenceError("artifact_changed", "Published execution evidence changed during verification")
        if byte_count != len(data) or digest.digest() != hashlib.sha256(data).digest():
            raise ArtifactEvidenceError("artifact_changed", "Published execution evidence bytes changed")
        _validate_named_entry_identity(
            parent_descriptor,
            final_component,
            final_published_stat,
        )
        os.fsync(parent_descriptor)
        _validate_relative_parent_namespace(root, relative_path, parent_descriptor)
        _validate_named_entry_identity(
            parent_descriptor,
            final_component,
            final_published_stat,
        )
        completed = True
    except ArtifactEvidenceError:
        raise
    except OSError as exc:
        code = _map_safe_open_error(exc)
        raise ArtifactEvidenceError(code, "Execution evidence cannot be published safely") from exc
    finally:
        if parent_descriptor is not None and renamed and not completed and published_identity is not None:
            _unlink_named_entry_if_identity(
                root,
                relative_path,
                parent_descriptor,
                final_component,
                published_identity,
            )
        if parent_descriptor is not None and temporary_name is not None and not renamed and temporary_identity is not None:
            _unlink_named_entry_if_identity(
                root,
                relative_path,
                parent_descriptor,
                temporary_name,
                temporary_identity,
            )
        if published_descriptor is not None:
            os.close(published_descriptor)
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _bounded_rendered(value: Any, maximum: int = _MAX_RENDERED_ERROR_CHARS) -> str:
    """Render untrusted diagnostic text without reflecting unbounded payloads."""

    rendered = str(value)
    if len(rendered) <= maximum:
        return rendered
    return f"{rendered[:maximum]}...[truncated]"


def _decode_bound_manifest(data: bytes, *, artifact_kind: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ArtifactEvidenceError(
                    "invalid_manifest_binding",
                    f"{artifact_kind} contains duplicate member {_bounded_rendered(key)!r}",
                )
            decoded[key] = value
        return decoded

    def reject_non_finite(value: str) -> Any:
        raise ArtifactEvidenceError(
            "invalid_manifest_binding",
            f"{artifact_kind} contains forbidden non-finite number {value}",
        )

    try:
        payload = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except ArtifactEvidenceError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ArtifactEvidenceError(
            "invalid_manifest_binding",
            f"{artifact_kind} is not one valid UTF-8 JSON object",
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactEvidenceError(
            "invalid_manifest_binding",
            f"{artifact_kind} must contain one JSON object",
        )
    return payload


def _invalid_manifest_binding(message: str) -> ArtifactEvidenceError:
    return ArtifactEvidenceError("invalid_manifest_binding", message)


_EXECUTION_CONTRACT_CONTAINER = {
    "combined_manifest_json": "environment",
    "batch_manifest_json": "config",
    "run_card": "effective_config",
}
_BUILTIN_MODEL_IDS = {
    "ensemble": "ensemble/multi-backend",
    "synthetic": "synthetic/depth-analytic-v1",
}


def _validate_manifest_execution_contract(
    payload: Mapping[str, Any],
    *,
    artifact_kind: str,
    expected_projection: Mapping[str, Any],
    authoritative_plan: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    """Validate the nested compatibility carrier, with legacy fallback."""

    container_name = _EXECUTION_CONTRACT_CONTAINER[artifact_kind]
    container = payload.get(container_name)
    has_nested = isinstance(container, Mapping) and "execution_contract" in container
    nested = container.get("execution_contract") if isinstance(container, Mapping) else None
    legacy_projection = payload.get("execution_plan")
    legacy_path = payload.get("execution_evidence_path")

    if has_nested:
        if not isinstance(nested, Mapping):
            raise _invalid_manifest_binding(f"{artifact_kind} execution_contract must be an object")
        bound_path = nested.get("execution_evidence_path")
        bound_projection = nested.get("runtime")
        if nested.get("authoritative_plan") != authoritative_plan:
            raise _invalid_manifest_binding(f"{artifact_kind} authoritative_plan does not match the exact prepared plan")
        if bound_projection != expected_projection:
            raise _invalid_manifest_binding(f"{artifact_kind} runtime projection does not match the prepared runtime outcome")
        if legacy_path is not None and legacy_path != bound_path:
            raise _invalid_manifest_binding(f"{artifact_kind} has conflicting execution evidence pointers")
        if legacy_projection is not None and legacy_projection != expected_projection:
            raise _invalid_manifest_binding(f"{artifact_kind} has a conflicting legacy execution-plan projection")
    else:
        bound_projection = legacy_projection
        bound_path = legacy_path

    if bound_path != expected_projection["execution_evidence_path"]:
        raise _invalid_manifest_binding(f"{artifact_kind} does not point to the prepared execution evidence sidecar")
    if bound_projection != expected_projection:
        raise _invalid_manifest_binding(
            f"{artifact_kind} execution-plan authority does not match the prepared plan/runtime outcome"
        )

    if not has_nested:
        return None

    assert isinstance(nested, Mapping)
    outcome_fields = (
        "artifact_outcome_authority",
        "requested_artifacts",
        "produced_artifacts",
        "omitted_artifacts",
        "failed_artifacts",
    )
    present_fields = {field_name for field_name in outcome_fields if field_name in nested}
    if not present_fields:
        # A nested execution_contract is the canonical v1 carrier. Preserve
        # the genuinely legacy top-level projection fallback (which returns
        # ``None`` above), but distinguish a canonical carrier whose outcome
        # projection was stripped so final verification cannot downgrade all
        # carriers to the legacy contract by omission.
        return {}
    if present_fields != set(outcome_fields):
        missing = sorted(set(outcome_fields) - present_fields)
        raise _invalid_manifest_binding(f"{artifact_kind} execution outcome projection is incomplete: {missing}")

    outcome_authority = nested.get("artifact_outcome_authority")
    if not isinstance(outcome_authority, Mapping):
        raise _invalid_manifest_binding(f"{artifact_kind} artifact_outcome_authority must be an object")
    if (
        outcome_authority.get("schema") != MANIFEST_OUTCOME_PROJECTION_SCHEMA
        or outcome_authority.get("source_schema") != EXECUTION_EVIDENCE_SCHEMA
        or outcome_authority.get("execution_evidence_path") != bound_path
        or outcome_authority.get("record_authority") != "detached_execution_evidence"
    ):
        raise _invalid_manifest_binding(f"{artifact_kind} artifact outcome authority is invalid")
    for field_name in outcome_fields[1:]:
        value = nested.get(field_name)
        if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
            raise _invalid_manifest_binding(f"{artifact_kind} {field_name} must be an array of objects")
    return {field_name: nested[field_name] for field_name in outcome_fields}


def _candidate_for_backend(plan: CanonicalExecutionPlan, backend_id: Optional[str]) -> Any:
    matches = [candidate for candidate in plan.backend_candidates if candidate.backend_id == backend_id]
    if len(matches) != 1:
        raise _invalid_manifest_binding(f"Runtime backend {backend_id!r} is outside unique prepared candidate authority")
    return matches[0]


def _enabled_model_contracts(candidate: Any) -> tuple[Any, ...]:
    return tuple(contract for contract in candidate.model_contracts if contract.enabled)


def _expected_model_id(candidate: Any) -> Optional[str]:
    contracts = _enabled_model_contracts(candidate)
    if len(contracts) == 1:
        model = contracts[0].model
        return model.repo_id or model.canonical_key
    return _BUILTIN_MODEL_IDS.get(candidate.backend_id)


def _contract_for_claim(candidate: Any, model_id: Any) -> Optional[Any]:
    if not isinstance(model_id, str) or not model_id:
        return None
    matches = [
        contract
        for contract in _enabled_model_contracts(candidate)
        if model_id in {contract.model.repo_id, contract.model.canonical_key}
    ]
    return matches[0] if len(matches) == 1 else None


def _validate_model_artifact_claims(
    claim: Mapping[str, Any],
    contract: Any,
    *,
    context: str,
    require_artifact_identity: bool = False,
) -> None:
    expected_path = contract.artifact_path
    expected_filename = Path(expected_path).name if isinstance(expected_path, str) and expected_path else None
    expected_sha = contract.artifact_sha256
    if (
        require_artifact_identity
        and expected_filename is not None
        and not any(field_name in claim for field_name in ("model_artifact_filename", "artifact_filename"))
    ):
        raise _invalid_manifest_binding(f"{context} is missing prepared model artifact filename")
    if (
        require_artifact_identity
        and expected_sha is not None
        and not any(field_name in claim for field_name in ("model_artifact_sha256", "artifact_sha256"))
    ):
        raise _invalid_manifest_binding(f"{context} is missing prepared model artifact digest")
    for field_name in ("model_artifact_filename", "artifact_filename"):
        if field_name in claim and claim.get(field_name) != expected_filename:
            raise _invalid_manifest_binding(f"{context} {field_name} does not match prepared model authority")
    for field_name in ("model_artifact_sha256", "artifact_sha256"):
        if field_name in claim and claim.get(field_name) != expected_sha:
            raise _invalid_manifest_binding(f"{context} {field_name} does not match prepared model authority")
    for field_name in ("resolved_revision", "revision"):
        if field_name in claim and claim.get(field_name) != contract.model.revision:
            raise _invalid_manifest_binding(f"{context} {field_name} does not match prepared model authority")
    if "license_id" in claim and claim.get("license_id") != contract.model.license_id:
        raise _invalid_manifest_binding(f"{context} license_id does not match prepared model authority")


def _validate_runtime_licensing(
    payload: Mapping[str, Any],
    candidates: Sequence[Any],
    *,
    plan: CanonicalExecutionPlan,
    context: str,
) -> None:
    expected_contracts = tuple(contract for candidate in candidates for contract in _enabled_model_contracts(candidate))
    licensing = payload.get("licensing")
    if licensing is None:
        if expected_contracts:
            raise _invalid_manifest_binding(f"{context} licensing is required by prepared model authority")
        return
    if not isinstance(licensing, Mapping):
        raise _invalid_manifest_binding(f"{context} licensing must be an object")
    models = licensing.get("models")
    if not isinstance(models, list):
        raise _invalid_manifest_binding(f"{context} licensing.models must be an array")
    expected_by_identity = {
        (contract.model.repo_id or contract.model.canonical_key, contract.backend_id): contract
        for contract in expected_contracts
    }
    if len(models) != len(expected_by_identity):
        raise _invalid_manifest_binding(f"{context} licensing.models does not cover prepared model authority")
    seen: set[tuple[str, str]] = set()
    for model_claim in models:
        if not isinstance(model_claim, Mapping):
            raise _invalid_manifest_binding(f"{context} licensing model entries must be objects")
        model_id = model_claim.get("id")
        runtime_role = normalize_backend_id(model_claim.get("runtime_role"))
        identity = (model_id, runtime_role) if isinstance(model_id, str) and isinstance(runtime_role, str) else None
        contract = expected_by_identity.get(identity) if identity is not None else None
        if identity is None or contract is None or identity in seen:
            raise _invalid_manifest_binding(f"{context} licensing model identity is outside prepared authority")
        seen.add(identity)
        expected_fields = {
            "license": contract.model.license_id or "unknown",
            "runtime_role": contract.backend_id,
            "usage_class": contract.model.usage_class,
            "requires_non_commercial_ok": contract.model.requires_non_commercial_ok,
        }
        if any(model_claim.get(field) != value for field, value in expected_fields.items()):
            raise _invalid_manifest_binding(f"{context} licensing model metadata does not match prepared authority")

    acknowledgements = plan.license_acknowledgements
    model_requires_research = any(
        contract.model.requires_non_commercial_ok or "non_commercial" in str(contract.model.usage_class or "")
        for contract in expected_contracts
    )
    research_required = bool(
        model_requires_research or acknowledgements.apple_depth_pro_research or acknowledgements.research_tools
    )
    non_commercial_active = bool(
        acknowledgements.non_commercial_ok
        and (model_requires_research or acknowledgements.apple_depth_pro_research or acknowledgements.research_tools)
    )
    expected_aggregate = {
        "schema_version": "1.0",
        "software_license_tier": (
            "research_or_non_commercial" if research_required or non_commercial_active else "commercial"
        ),
        "non_commercial_active": non_commercial_active,
        "research_acknowledgement_required": research_required,
    }
    if any(licensing.get(field) != value for field, value in expected_aggregate.items()):
        raise _invalid_manifest_binding(f"{context} licensing aggregate does not match prepared plan acknowledgements")


def _validate_attempt_claims(attempts: Any, *, plan: CanonicalExecutionPlan, selected_backend: str) -> None:
    if attempts is None:
        candidate = _candidate_for_backend(plan, selected_backend)
        if _enabled_model_contracts(candidate):
            raise _invalid_manifest_binding("Model-backed execution is missing backend attempt authority")
        return
    if not isinstance(attempts, list):
        raise _invalid_manifest_binding("Backend attempts must be an array")
    successful_backends: list[str] = []
    observed_backends: list[str] = []
    observed_statuses: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise _invalid_manifest_binding("Backend attempt entries must be objects")
        attempt_backend = normalize_backend_id(attempt.get("backend"))
        candidate = _candidate_for_backend(plan, attempt_backend)
        status = attempt.get("status")
        if status not in {"failed", "success"}:
            raise _invalid_manifest_binding("Backend attempt status must be failed or success")
        observed_backends.append(candidate.backend_id)
        observed_statuses.append(status)
        expected_model_id = _expected_model_id(candidate)
        claimed_model_id = attempt.get("model_id")
        if claimed_model_id is not None and claimed_model_id != expected_model_id:
            raise _invalid_manifest_binding("Backend attempt model_id does not match prepared candidate authority")
        contract = _contract_for_claim(candidate, claimed_model_id)
        if contract is not None:
            if "device" in attempt and attempt.get("device") != contract.device:
                raise _invalid_manifest_binding("Backend attempt device does not match prepared candidate authority")
            _validate_model_artifact_claims(
                attempt,
                contract,
                context="Backend attempt",
                require_artifact_identity=attempt.get("status") == "success",
            )
        elif any(
            attempt.get(field) is not None
            for field in (
                "model_artifact_filename",
                "artifact_filename",
                "model_artifact_sha256",
                "artifact_sha256",
                "resolved_revision",
                "revision",
                "license_id",
            )
        ):
            expected_contracts = _enabled_model_contracts(candidate)
            if len(expected_contracts) != 1:
                raise _invalid_manifest_binding("Backend attempt artifact claim is ambiguous")
            _validate_model_artifact_claims(attempt, expected_contracts[0], context="Backend attempt")
        if status == "success":
            if claimed_model_id != expected_model_id:
                raise _invalid_manifest_binding("Successful backend attempt is missing exact model identity")
            successful_backends.append(candidate.backend_id)
    expected_prefix = list(plan.candidate_fallback_chain[: len(observed_backends)])
    if observed_backends != expected_prefix:
        raise _invalid_manifest_binding(
            "Backend attempts must be an ordered duplicate-free prefix of candidate_fallback_chain"
        )
    if not observed_statuses or observed_statuses[-1] != "success" or observed_backends[-1] != selected_backend:
        raise _invalid_manifest_binding("Backend attempts must terminate in the selected successful candidate")
    if any(status != "failed" for status in observed_statuses[:-1]):
        raise _invalid_manifest_binding("Only the terminal selected backend attempt may succeed")
    if successful_backends != [selected_backend]:
        raise _invalid_manifest_binding("Backend attempts do not identify exactly one selected successful candidate")


def _validate_combined_runtime_model_binding(
    payload: Mapping[str, Any],
    *,
    plan: CanonicalExecutionPlan,
    selected_backend: str,
) -> None:
    candidate = _candidate_for_backend(plan, selected_backend)
    expected_model_id = _expected_model_id(candidate)
    if expected_model_id is None:
        raise _invalid_manifest_binding("Prepared candidate model identity is ambiguous")
    backend_selection = payload.get("backend_selection")
    assert isinstance(backend_selection, Mapping)
    if backend_selection.get("model_id") != expected_model_id:
        raise _invalid_manifest_binding("combined_manifest_json model_id does not match prepared candidate authority")
    contracts = _enabled_model_contracts(candidate)
    if len(contracts) == 1:
        if backend_selection.get("device") != contracts[0].device:
            raise _invalid_manifest_binding("combined_manifest_json device does not match prepared candidate authority")
        _validate_model_artifact_claims(
            backend_selection,
            contracts[0],
            context="combined_manifest_json",
        )

    depth = payload.get("depth")
    if not isinstance(depth, Mapping) or normalize_backend_id(depth.get("model")) != selected_backend:
        raise _invalid_manifest_binding("combined_manifest_json depth.model does not match the selected candidate")
    attempts = backend_selection.get("attempts")
    depth_stats = depth.get("stats")
    if isinstance(depth_stats, Mapping):
        if normalize_backend_id(depth_stats.get("backend")) not in {None, selected_backend}:
            raise _invalid_manifest_binding("combined_manifest_json depth.stats backend does not match")
        if "attempts" in depth_stats and depth_stats.get("attempts") != attempts:
            raise _invalid_manifest_binding("combined_manifest_json depth/backend attempt claims conflict")
        for field_name in ("model_id", "resolved_model_id"):
            if field_name in depth_stats and depth_stats.get(field_name) != expected_model_id:
                raise _invalid_manifest_binding(
                    f"combined_manifest_json depth.stats {field_name} does not match prepared candidate authority"
                )
    _validate_attempt_claims(attempts, plan=plan, selected_backend=selected_backend)
    _validate_runtime_licensing(
        payload,
        (candidate,),
        plan=plan,
        context="combined_manifest_json",
    )


def _validate_run_card_runtime_model_binding(
    payload: Mapping[str, Any],
    *,
    plan: CanonicalExecutionPlan,
    runtime_projection: Optional[Mapping[str, Any]] = None,
) -> None:
    backend_selection = payload.get("backend_selection")
    if not isinstance(backend_selection, Mapping):
        raise _invalid_manifest_binding("run_card backend_selection must be an object")
    selected_backend = normalize_backend_id(backend_selection.get("resolved"))
    candidate = _candidate_for_backend(plan, selected_backend)
    expected_model_id = _expected_model_id(candidate)
    if expected_model_id is None:
        raise _invalid_manifest_binding("Prepared run-card candidate model identity is ambiguous")
    if backend_selection.get("model_id") != expected_model_id:
        raise _invalid_manifest_binding("run_card model_id does not match prepared candidate authority")

    contracts = _enabled_model_contracts(candidate)
    model_contract = payload.get("model_contract")
    if not contracts:
        if model_contract is not None:
            raise _invalid_manifest_binding("run_card claims an unprepared model contract")
    elif len(contracts) > 1 and model_contract is None:
        # Frozen run-card schemas expose one optional model_contract. The
        # exact constituent set remains bound by authoritative_plan, while
        # backend_selection identifies the logical aggregate candidate.
        pass
    else:
        if not isinstance(model_contract, Mapping):
            raise _invalid_manifest_binding("run_card is missing its prepared model contract")
        claimed_model_id = model_contract.get("resolved_repo_id") or model_contract.get("canonical_model_key")
        contract = _contract_for_claim(candidate, claimed_model_id)
        if contract is None:
            raise _invalid_manifest_binding("run_card model contract is outside prepared candidate authority")
        expected_fields = {
            "requested_model_selector": contract.model.requested_selector,
            "resolution_reason": contract.model.resolution_reason,
            "canonical_model_key": contract.model.canonical_key,
            "resolved_repo_id": contract.model.repo_id,
            "resolved_revision": contract.model.revision,
            "license_id": contract.model.license_id,
            "usage_class": contract.model.usage_class,
            "requires_non_commercial_ok": contract.model.requires_non_commercial_ok,
            "backend_kind": contract.backend_id,
            "accelerator_kind": contract.model.accelerator_kind,
            "non_commercial_ok": plan.license_acknowledgements.non_commercial_ok,
        }
        if any(model_contract.get(field) != value for field, value in expected_fields.items()):
            raise _invalid_manifest_binding("run_card model contract does not match prepared candidate authority")
        _validate_model_artifact_claims(
            model_contract,
            contract,
            context="run_card model contract",
            require_artifact_identity=True,
        )
        if backend_selection.get("device") != contract.device:
            raise _invalid_manifest_binding("run_card device does not match prepared candidate authority")
        _validate_model_artifact_claims(
            backend_selection,
            contract,
            context="run_card backend selection",
            require_artifact_identity=True,
        )
    backend_summary = payload.get("backend_summary")
    final_backends_used = backend_summary.get("final_backends_used") if isinstance(backend_summary, Mapping) else None
    if not isinstance(final_backends_used, list) or not final_backends_used:
        raise _invalid_manifest_binding("run_card backend summary lacks final backend authority")
    summary_candidates = tuple(
        _candidate_for_backend(plan, normalize_backend_id(backend_id)) for backend_id in final_backends_used
    )
    runtime_candidates = summary_candidates
    if runtime_projection is not None:
        execution_rows = runtime_projection.get("executed_backend_by_input")
        if not isinstance(execution_rows, list):
            raise _invalid_manifest_binding("run_card carried runtime projection lacks per-input backend authority")
        projected_backend_ids = frozenset(
            _candidate_for_backend(plan, normalize_backend_id(row.get("executed_backend"))).backend_id
            for row in execution_rows
            if isinstance(row, Mapping) and row.get("status") == "ok"
        )
        summary_backend_ids = tuple(candidate.backend_id for candidate in summary_candidates)
        if len(summary_backend_ids) != len(projected_backend_ids) or frozenset(summary_backend_ids) != projected_backend_ids:
            raise _invalid_manifest_binding(
                "run_card backend_summary.final_backends_used does not match carried runtime projection"
            )
        runtime_candidates = tuple(_candidate_for_backend(plan, backend_id) for backend_id in sorted(projected_backend_ids))
    _validate_runtime_licensing(
        payload,
        runtime_candidates,
        plan=plan,
        context="run_card",
    )


_BATCH_RESULT_PATH_BINDINGS = {
    "depth_path": "depth_u16_png",
    "depth_float_path": "depth_float_npy",
    "depth_metadata_path": "depth_metadata_json",
    "segmentation_mask_path": "materials_v3_masks",
    "v2_output": "v2_enhanced_image",
    "v2_output_path": "v2_enhanced_image",
}
_REQUIRED_BATCH_RESULT_FIELDS = {
    "depth_u16_png": "depth_path",
    "depth_float_npy": "depth_float_path",
    "v2_enhanced_image": "v2_output_path",
}
_BATCH_RECONSTRUCTION_PATH_FIELDS = (
    "reconstruction_preflight_path",
    "reconstruction_scene_manifest_path",
    "reconstruction_debug_manifest_path",
    "reconstruction_debug_cameras_path",
    "reconstruction_debug_preview_path",
    "reconstruction_manifest_path",
    "reconstruction_report_path",
    "reconstruction_diagnostics_path",
)
_NON_OK_FORBIDDEN_ARTIFACT_FIELDS = frozenset(
    {
        "manifest",
        "provenance_sidecar",
        "pbr_manifest_path",
        *_BATCH_RESULT_PATH_BINDINGS,
        *_BATCH_RECONSTRUCTION_PATH_FIELDS,
    }
)


def _validate_batch_result_bindings(
    payload: Mapping[str, Any],
    *,
    plan: CanonicalExecutionPlan,
    execution_rows: Sequence[InputExecution],
    root: _PinnedOutputRoot,
    combined_manifest_paths: Mapping[str, str],
    combined_manifest_authorities: Mapping[str, "_CombinedManifestAuthority"],
    expected_reconstruction_paths: Optional[Sequence[Path]],
) -> None:
    """Bind every batch result row to one exact prepared input and manifest."""

    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(plan.inputs):
        raise _invalid_manifest_binding("batch_manifest_json results do not cover the prepared input cardinality")
    if len(execution_rows) != len(plan.inputs):
        raise _invalid_manifest_binding("batch_manifest_json has incomplete prepared runtime outcomes")

    observed_reconstruction_paths: list[str] = []
    requested_kinds = frozenset(plan.requested_outputs)
    for plan_input, execution_row, result in zip(plan.inputs, execution_rows, results):
        if not isinstance(result, dict):
            raise _invalid_manifest_binding("batch_manifest_json result rows must be objects")
        if result.get("image") != plan_input.path:
            raise _invalid_manifest_binding(
                f"batch_manifest_json result does not match prepared input {plan_input.input_id!r}"
            )
        if result.get("status") != execution_row.status:
            raise _invalid_manifest_binding(
                f"batch_manifest_json status does not match prepared input {plan_input.input_id!r}"
            )
        result_backend = normalize_backend_id(result.get("backend"))
        if result_backend != execution_row.executed_backend:
            raise _invalid_manifest_binding(
                f"batch_manifest_json backend does not match prepared input {plan_input.input_id!r}"
            )
        if execution_row.status != "ok":
            stale_fields = sorted(
                field_name
                for field_name in _NON_OK_FORBIDDEN_ARTIFACT_FIELDS
                if field_name in result and result.get(field_name) is not None and result.get(field_name) != ""
            )
            if stale_fields:
                raise _invalid_manifest_binding(
                    f"batch_manifest_json non-ok input {plan_input.input_id!r} carries artifact paths: {stale_fields}"
                )
            continue
        manifest_path = result.get("manifest")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise _invalid_manifest_binding(
                f"batch_manifest_json successful input {plan_input.input_id!r} lacks its combined manifest"
            )
        try:
            relative_manifest_path = root.confined_relative_path(Path(manifest_path))
        except ArtifactEvidenceError as exc:
            raise _invalid_manifest_binding(
                f"batch_manifest_json combined-manifest path is invalid for {plan_input.input_id!r}"
            ) from exc
        if combined_manifest_paths.get(plan_input.input_id) != relative_manifest_path:
            raise _invalid_manifest_binding(
                f"batch_manifest_json combined-manifest association is incomplete for {plan_input.input_id!r}"
            )

        combined_authority = combined_manifest_authorities.get(plan_input.input_id)
        if combined_authority is None:
            raise _invalid_manifest_binding(
                f"batch_manifest_json has no combined-manifest output authority for {plan_input.input_id!r}"
            )
        declared_paths_by_kind: dict[str, set[str]] = {}
        for artifact_kind in frozenset(_BATCH_RESULT_PATH_BINDINGS.values()):
            try:
                declared_paths_by_kind[artifact_kind] = {
                    root.confined_relative_path(Path(declared_path))
                    for declared_path in combined_authority.declared_paths(artifact_kind)
                }
            except ArtifactEvidenceError as exc:
                raise _invalid_manifest_binding(
                    f"batch_manifest_json {artifact_kind} is outside combined-manifest authority "
                    f"for {plan_input.input_id!r}"
                ) from exc

        for artifact_kind, field_name in _REQUIRED_BATCH_RESULT_FIELDS.items():
            if artifact_kind not in requested_kinds or not declared_paths_by_kind[artifact_kind]:
                continue
            path_value = result.get(field_name)
            if not isinstance(path_value, str) or not path_value:
                raise _invalid_manifest_binding(
                    f"batch_manifest_json successful input {plan_input.input_id!r} lacks required {field_name}"
                )

        for field_name, artifact_kind in _BATCH_RESULT_PATH_BINDINGS.items():
            path_value = result.get(field_name)
            if path_value is None:
                continue
            if not isinstance(path_value, str) or not path_value:
                raise _invalid_manifest_binding(f"batch_manifest_json {field_name} is invalid for {plan_input.input_id!r}")
            try:
                result_relative_path = root.confined_relative_path(Path(path_value))
            except ArtifactEvidenceError as exc:
                raise _invalid_manifest_binding(
                    f"batch_manifest_json {field_name} is outside combined-manifest authority " f"for {plan_input.input_id!r}"
                ) from exc
            if result_relative_path not in declared_paths_by_kind[artifact_kind]:
                raise _invalid_manifest_binding(
                    f"batch_manifest_json {field_name} does not match combined-manifest authority "
                    f"for {plan_input.input_id!r}"
                )

        if "reconstruction_bundle" in requested_kinds:
            for field_name in _BATCH_RECONSTRUCTION_PATH_FIELDS:
                path_value = result.get(field_name)
                if path_value is None:
                    continue
                if not isinstance(path_value, str) or not path_value:
                    raise _invalid_manifest_binding(f"batch_manifest_json {field_name} is invalid for {plan_input.input_id!r}")
                try:
                    observed_reconstruction_paths.append(root.confined_relative_path(Path(path_value)))
                except ArtifactEvidenceError as exc:
                    raise _invalid_manifest_binding(
                        f"batch_manifest_json {field_name} is outside reconstruction-bundle authority"
                    ) from exc

    if expected_reconstruction_paths is not None:
        try:
            expected_paths = sorted(root.confined_relative_path(Path(path)) for path in expected_reconstruction_paths)
        except ArtifactEvidenceError as exc:
            raise _invalid_manifest_binding("Detached reconstruction-bundle authority is invalid") from exc
        if len(observed_reconstruction_paths) != len(set(observed_reconstruction_paths)):
            raise _invalid_manifest_binding("batch_manifest_json repeats reconstruction-bundle artifact paths")
        if sorted(observed_reconstruction_paths) != expected_paths:
            raise _invalid_manifest_binding(
                "batch_manifest_json reconstruction paths do not match detached reconstruction-bundle authority"
            )


def _validate_run_card_artifact_index(
    payload: Mapping[str, Any],
    *,
    root: _PinnedOutputRoot,
    context: _ManifestValidationContext,
) -> None:
    """Bind the compatibility index/tree to the current confined bytes."""

    artifact_index = payload.get("artifact_index")
    if not isinstance(artifact_index, list):
        raise _invalid_manifest_binding("run_card artifact_index must be an array")

    capture_budget = _CaptureBudget()
    observed_paths: set[str] = set()
    observed_identities: set[tuple[int, int]] = set()
    carrier_change: Optional[ArtifactEvidenceError] = None
    index_binding_error: Optional[ArtifactEvidenceError] = None
    for index, raw_entry in enumerate(artifact_index):
        if not isinstance(raw_entry, Mapping):
            if index_binding_error is None:
                index_binding_error = _invalid_manifest_binding(f"run_card artifact_index[{index}] must be an object")
            continue
        relative_path = raw_entry.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            if index_binding_error is None:
                index_binding_error = _invalid_manifest_binding(
                    f"run_card artifact_index[{index}].relative_path must be a non-empty string"
                )
            continue
        try:
            capture = _capture_artifact(root, Path(relative_path), budget=capture_budget)
        except ArtifactEvidenceError as exc:
            if exc.code == "artifact_changed" and carrier_change is None:
                carrier_change = exc
            elif index_binding_error is None:
                index_binding_error = exc
            continue
        if capture.record["path"] in observed_paths or capture.identity in observed_identities:
            if index_binding_error is None:
                index_binding_error = _invalid_manifest_binding(
                    "run_card artifact_index contains duplicate artifact authority"
                )
        observed_paths.add(capture.record["path"])
        observed_identities.add(capture.identity)
        # Carrier identity is stronger than the run-card compatibility index:
        # surface post-copy replacement as ``artifact_changed`` even when the
        # stale index would otherwise collapse it into a generic binding error.
        entry_carrier_changed = False
        try:
            _validate_expected_carrier_capture(capture, context=context)
        except ArtifactEvidenceError as exc:
            if exc.code != "artifact_changed":
                raise
            entry_carrier_changed = True
            if carrier_change is None:
                carrier_change = exc
        index_entry_changed = (
            raw_entry.get("path") != capture.record["path"]
            or relative_path != capture.record["path"]
            or raw_entry.get("sha256") != capture.record["sha256"]
            or raw_entry.get("size_bytes") != capture.record["size_bytes"]
        )
        if index_entry_changed and not entry_carrier_changed and index_binding_error is None:
            index_binding_error = _invalid_manifest_binding(
                f"run_card artifact_index[{index}] does not match current confined artifact bytes"
            )
    if carrier_change is not None:
        raise carrier_change
    if index_binding_error is not None:
        raise index_binding_error

    version = payload.get("run_card_version")
    if version == "v2":
        from .artifact_tree import verify_artifact_tree_payload

        artifact_tree = payload.get("artifact_tree")
        if not isinstance(artifact_tree, Mapping):
            raise _invalid_manifest_binding("run_card artifact_tree must be an object")
        tree_errors = verify_artifact_tree_payload(artifact_tree, artifact_index=artifact_index)
        if tree_errors:
            raise _invalid_manifest_binding(
                "run_card artifact_tree does not match artifact_index: " + _bounded_rendered("; ".join(tree_errors))
            )
    else:
        from .artifact_manager import compute_artifact_merkle_root

        try:
            expected_merkle_root = compute_artifact_merkle_root(artifact_index)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise _invalid_manifest_binding("run_card artifact_index cannot produce its Merkle root") from exc
        if payload.get("artifact_merkle_root") != expected_merkle_root:
            raise _invalid_manifest_binding("run_card artifact_merkle_root does not match artifact_index")


def _validate_run_card_self_integrity(
    payload: Mapping[str, Any],
    *,
    data: bytes,
    relative_path: str,
    root: _PinnedOutputRoot,
    manifest_budget: _ManifestBudget,
) -> None:
    """Apply the dedicated verifier's two-part run-card self-integrity contract."""

    integrity = payload.get("run_card_integrity")
    if not isinstance(integrity, dict):
        raise _invalid_manifest_binding("run_card is missing run_card_integrity")
    expected_run_card_path = Path(relative_path).name
    if integrity.get("self_indexing") != "excluded_self_hash_cycle":
        raise _invalid_manifest_binding("run_card_integrity.self_indexing is invalid")
    if integrity.get("path") != expected_run_card_path:
        raise _invalid_manifest_binding("run_card_integrity.path does not match the run card")
    expected_payload_sha = integrity.get("canonical_payload_sha256")
    if not isinstance(expected_payload_sha, str) or len(expected_payload_sha) != 64:
        raise _invalid_manifest_binding("run_card_integrity.canonical_payload_sha256 is invalid")
    integrity_without_hash = {key: value for key, value in integrity.items() if key != "canonical_payload_sha256"}
    payload_without_hash = {**payload, "run_card_integrity": integrity_without_hash}
    actual_payload_sha = hashlib.sha256(canonicalize_json(payload_without_hash)).hexdigest()
    if expected_payload_sha != actual_payload_sha:
        raise _invalid_manifest_binding("run_card_integrity.canonical_payload_sha256 does not match")

    self_sidecar_path = PurePosixPath(relative_path).with_suffix(".self.json").as_posix()
    sidecar_data, _, _identity = _read_confined_artifact_bytes(
        root,
        Path(self_sidecar_path),
        context="run_card self-integrity sidecar",
        max_bytes=_MAX_BOUND_MANIFEST_BYTES,
        manifest_budget=manifest_budget,
    )
    sidecar = _decode_bound_manifest(sidecar_data, artifact_kind="run_card self-integrity sidecar")
    if sidecar.get("run_card_path") != expected_run_card_path:
        raise _invalid_manifest_binding("run-card self-integrity sidecar path does not match")
    if sidecar.get("self_indexing") != "excluded_self_hash_cycle":
        raise _invalid_manifest_binding("run-card self-integrity sidecar self_indexing is invalid")
    if sidecar.get("hash_algorithm") != "sha256":
        raise _invalid_manifest_binding("run-card self-integrity sidecar hash_algorithm is invalid")
    final_sha = sidecar.get("final_run_card_sha256")
    if not isinstance(final_sha, str) or final_sha != hashlib.sha256(data).hexdigest():
        raise _invalid_manifest_binding("run-card self-integrity sidecar final_run_card_sha256 does not match")


def _validate_bound_manifest_artifact(
    *,
    artifact_kind: str,
    input_id: Optional[str],
    candidate: Path,
    record: Mapping[str, Any],
    root: _PinnedOutputRoot,
    context: _ManifestValidationContext,
    combined_manifest_paths: Optional[Mapping[str, str]] = None,
    combined_manifest_authorities: Optional[Mapping[str, "_CombinedManifestAuthority"]] = None,
    expected_reconstruction_paths: Optional[Sequence[Path]] = None,
) -> dict[str, Any]:
    """Prove a requested manifest points back to this exact plan/sidecar."""

    data, relative_path, _identity = _read_confined_artifact_bytes(
        root,
        candidate,
        context=artifact_kind,
        max_bytes=_MAX_BOUND_MANIFEST_BYTES,
        manifest_budget=context.manifest_budget,
    )
    if (
        record.get("path") != relative_path
        or record.get("size_bytes") != len(data)
        or record.get("sha256") != hashlib.sha256(data).hexdigest()
    ):
        raise ArtifactEvidenceError(
            "artifact_changed",
            f"{artifact_kind} changed while its binding was validated",
        )
    payload = _decode_bound_manifest(data, artifact_kind=artifact_kind)
    plan = context.projector.plan
    if artifact_kind == "combined_manifest_json":
        if input_id is None:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                "combined_manifest_json is missing its prepared input identity",
            )
        matching_row = context.execution_by_id.get(input_id)
        if matching_row is None:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                f"combined_manifest_json has no unique runtime row for {input_id!r}",
            )
        expected_projection = context.projection_by_input[input_id]
    else:
        if input_id is not None:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                f"{artifact_kind} unexpectedly carries an input identity",
            )
        expected_projection = context.full_projection
    carrier_outcome_projection = _validate_manifest_execution_contract(
        payload,
        artifact_kind=artifact_kind,
        expected_projection=expected_projection,
        authoritative_plan=context.authoritative_plan,
    )

    if artifact_kind == "combined_manifest_json":
        if input_id is None or matching_row is None:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                "combined_manifest_json lost its validated prepared input binding",
            )
        manifest_input = payload.get("input")
        manifest_input_path = manifest_input.get("image_path") if isinstance(manifest_input, dict) else None
        plan_input = context.plan_input_by_id[input_id]
        expected_input_path = os.path.normpath(os.path.join(plan.input_root, plan_input.path))
        if (
            not isinstance(manifest_input_path, str)
            or not Path(manifest_input_path).is_absolute()
            or os.path.normpath(manifest_input_path) != expected_input_path
        ):
            raise ArtifactEvidenceError(
                "artifact_input_mismatch",
                f"combined_manifest_json input does not match prepared input {input_id!r}",
            )
        backend_selection = payload.get("backend_selection")
        resolved_backend = (
            normalize_backend_id(backend_selection.get("resolved_backend")) if isinstance(backend_selection, dict) else None
        )
        expected_backend = matching_row.executed_backend
        if resolved_backend != expected_backend or resolved_backend != expected_projection["executed_backend"]:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                "combined_manifest_json backend_selection.resolved_backend does not match execution projection",
            )
        assert resolved_backend is not None
        _validate_combined_runtime_model_binding(
            payload,
            plan=plan,
            selected_backend=resolved_backend,
        )
    elif artifact_kind == "batch_manifest_json":
        required_fields = {
            "batch_id": str,
            "start_time": str,
            "end_time": str,
            "config": dict,
            "results": list,
            "stats": dict,
        }
        if any(not isinstance(payload.get(field), expected_type) for field, expected_type in required_fields.items()):
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                "batch_manifest_json is missing its governed batch fields",
            )
        _validate_batch_result_bindings(
            payload,
            plan=plan,
            execution_rows=context.execution_rows,
            root=root,
            combined_manifest_paths=combined_manifest_paths or {},
            combined_manifest_authorities=combined_manifest_authorities or {},
            expected_reconstruction_paths=expected_reconstruction_paths,
        )
        context.batch_binding.bind(artifact_kind, payload)
    elif artifact_kind == "run_card":
        version = payload.get("run_card_version")
        if version not in {"v1", "v2"}:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                "run_card has no supported run_card_version",
            )
        from .validators.run_card_integrity import _verify_config_fingerprint
        from .validators.run_card_validator import validate_run_card_backend_semantics, validate_run_card_payload

        try:
            validate_run_card_payload(payload, schema_version=version)
            validate_run_card_backend_semantics(payload)
        except RuntimeError as exc:
            raise ArtifactEvidenceError(
                "invalid_manifest_binding",
                f"run_card does not satisfy its packaged contract: {_bounded_rendered(exc)}",
            ) from exc
        fingerprint_errors: list[str] = []
        _verify_config_fingerprint(payload, fingerprint_errors)
        if fingerprint_errors:
            raise _invalid_manifest_binding(
                "run_card config fingerprint is invalid: " + _bounded_rendered("; ".join(fingerprint_errors))
            )
        _validate_run_card_runtime_model_binding(
            payload,
            plan=plan,
            runtime_projection=expected_projection,
        )
        _validate_run_card_self_integrity(
            payload,
            data=data,
            relative_path=relative_path,
            root=root,
            manifest_budget=context.manifest_budget,
        )
        _validate_run_card_artifact_index(payload, root=root, context=context)
        context.batch_binding.bind(artifact_kind, payload)
    context.carrier_outcome_projections.append(
        (artifact_kind, input_id, carrier_outcome_projection),
    )
    return payload


def _combined_manifest_declared_paths(payload: Mapping[str, Any], artifact_kind: str) -> tuple[str, ...]:
    depth = payload.get("depth")
    depth_path = depth.get("depth_path") if isinstance(depth, Mapping) else None
    if artifact_kind == "depth_u16_png":
        return (depth_path,) if isinstance(depth_path, str) and depth_path else ()
    if artifact_kind == "depth_metadata_json":
        if not isinstance(depth_path, str) or not depth_path:
            return ()
        path = Path(depth_path)
        return (str(path.with_name(f"{path.stem}_metadata.json")),)
    if artifact_kind == "depth_float_npy":
        return (str(Path(depth_path).with_suffix(".npy")),) if isinstance(depth_path, str) and depth_path else ()
    if artifact_kind == "materials_v3_masks":
        materials = payload.get("materials_v3")
        if not isinstance(materials, Mapping) or materials.get("enabled") is not True:
            return ()
        segmentation = materials.get("segmentation_metadata") if isinstance(materials, Mapping) else None
        if not isinstance(segmentation, Mapping):
            return ()
        segmentation_status = segmentation.get("status")
        if segmentation_status is not None:
            from .artifact_manager import normalize_v2_status

            if normalize_v2_status(segmentation_status) != "ok":
                return ()
        errors = segmentation.get("errors")
        if isinstance(errors, list) and errors:
            return ()
        mask_path = segmentation.get("mask_artifact_path") if isinstance(segmentation, Mapping) else None
        return (mask_path,) if isinstance(mask_path, str) and mask_path else ()
    if artifact_kind == "pbr_maps":
        pbr_assets = payload.get("pbr_assets")
        if not isinstance(pbr_assets, Mapping):
            return ()
        paths = tuple(pbr_assets.get(key) for key in ("normal_path", "roughness_path", "ao_path"))
        if not all(isinstance(value, str) and value for value in paths):
            return ()
        return paths  # type: ignore[return-value]
    if artifact_kind == "v2_enhanced_image":
        v2 = payload.get("v2")
        if not isinstance(v2, Mapping):
            return ()
        from .artifact_manager import normalize_v2_status

        if normalize_v2_status(v2.get("status")) != "ok":
            return ()
        output_paths = v2.get("output_paths") if isinstance(v2, Mapping) else None
        if not isinstance(output_paths, list):
            return ()
        declared_paths: list[str] = []
        for observed_count, value in enumerate(output_paths, start=1):
            if observed_count > _MAX_ARTIFACTS_PER_OUTCOME:
                raise _ArtifactCardinalityError(
                    "v2.output_paths exceeds the bounded artifact cardinality limit " f"of {_MAX_ARTIFACTS_PER_OUTCOME}",
                    observed_count=observed_count,
                )
            if isinstance(value, str) and value:
                declared_paths.append(value)
        return tuple(declared_paths)
    return ()


@dataclass(frozen=True)
class _CombinedManifestAuthority:
    """Small path-only authority retained after a combined manifest is decoded."""

    paths_by_kind: Mapping[str, tuple[str, ...]]
    cardinality_by_kind: Mapping[str, int]

    def declared_paths(self, artifact_kind: str) -> tuple[str, ...]:
        observed_count = self.cardinality_by_kind.get(artifact_kind)
        if observed_count is not None:
            raise _ArtifactCardinalityError(
                f"{artifact_kind} exceeds the bounded artifact cardinality limit of " f"{_MAX_ARTIFACTS_PER_OUTCOME}",
                observed_count=observed_count,
            )
        return self.paths_by_kind.get(artifact_kind, ())


def _compact_combined_manifest_authority(
    payload: Mapping[str, Any],
    *,
    manifest_budget: _ManifestBudget,
) -> _CombinedManifestAuthority:
    paths_by_kind: dict[str, tuple[str, ...]] = {}
    cardinality_by_kind: dict[str, int] = {}
    for artifact_kind in (
        "depth_u16_png",
        "depth_metadata_json",
        "depth_float_npy",
        "materials_v3_masks",
        "pbr_maps",
        "v2_enhanced_image",
    ):
        try:
            declared_paths = _combined_manifest_declared_paths(payload, artifact_kind)
        except _ArtifactCardinalityError as exc:
            cardinality_by_kind[artifact_kind] = exc.observed_count
            continue
        paths_by_kind[artifact_kind] = declared_paths
    retained_bytes = sys.getsizeof(paths_by_kind) + sys.getsizeof(cardinality_by_kind)
    retained_bytes += sum(
        sys.getsizeof(artifact_kind) + sys.getsizeof(declared_paths) + sum(sys.getsizeof(path) for path in declared_paths)
        for artifact_kind, declared_paths in paths_by_kind.items()
    )
    retained_bytes += sum(
        sys.getsizeof(artifact_kind) + sys.getsizeof(observed_count)
        for artifact_kind, observed_count in cardinality_by_kind.items()
    )
    manifest_budget.retain(retained_bytes, context="Combined-manifest path authority")
    return _CombinedManifestAuthority(
        paths_by_kind=paths_by_kind,
        cardinality_by_kind=cardinality_by_kind,
    )


def _outcome_base(declaration: Mapping[str, Any], input_id: Optional[str]) -> dict[str, Any]:
    return {
        **dict(declaration),
        "input_id": input_id,
    }


def _missing_outcome(
    declaration: Mapping[str, Any],
    *,
    input_id: Optional[str],
    input_status: Optional[str],
) -> tuple[str, dict[str, Any]]:
    required = bool(declaration["required"])
    if input_status == "error":
        reason_code = "upstream_failed"
    elif input_status in {"skipped", "missing"}:
        reason_code = "upstream_not_executed"
    else:
        reason_code = "required_output_missing" if required else "optional_stage_no_output"
    bucket = "failed" if required else "omitted"
    return bucket, {
        **_outcome_base(declaration, input_id),
        "reason_code": reason_code,
    }


def compute_execution_evidence_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash canonical evidence with its self-describing digest omitted."""

    body = dict(payload)
    body.pop("evidence_fingerprint_sha256", None)
    return hashlib.sha256(canonicalize_json(body)).hexdigest()


def _validate_carrier_outcome_projections(
    payload: Mapping[str, Any],
    *,
    context: _ManifestValidationContext,
    require_all: bool,
) -> None:
    """Bind non-self-referential carrier summaries to detached evidence."""

    carriers = context.carrier_outcome_projections
    if not require_all:
        return
    for artifact_kind, input_id, projection in carriers:
        if not projection:
            raise _invalid_manifest_binding(f"{artifact_kind} is missing its execution outcome projection")
        expected = build_manifest_outcome_projection(
            payload,
            evidence_path=context.projector.evidence_path,
            input_id=input_id if artifact_kind == "combined_manifest_json" else None,
        )
        if projection != expected:
            raise _invalid_manifest_binding(f"{artifact_kind} execution outcome projection does not match detached evidence")


def build_execution_evidence(
    plan: CanonicalExecutionPlan,
    *,
    output_root: Path,
    evidence_path: str,
    input_executions: Sequence[InputExecution],
    artifact_observations: Sequence[ArtifactObservation],
    derive_manifest_outputs: bool = False,
    expected_artifact_records: Optional[Mapping[tuple[str, Optional[str]], Sequence[Mapping[str, Any]]]] = None,
    expected_carrier_records: Optional[Sequence[Mapping[str, Any]]] = None,
    require_carrier_outcome_projections: bool = False,
) -> dict[str, Any]:
    """Account for every requested output in one prepared execution.

    Missing required outputs are recorded in ``failed_artifacts`` so the
    evidence remains inspectable.  Call :func:`require_required_artifacts`
    after durable evidence publication to fail the run.
    """

    if len(artifact_observations) > _MAX_ARTIFACT_OBSERVATIONS:
        raise ExecutionEvidenceError(f"Artifact observations exceed the bounded limit of {_MAX_ARTIFACT_OBSERVATIONS}")
    if expected_carrier_records is not None and len(expected_carrier_records) > _MAX_ARTIFACT_OBSERVATIONS:
        raise ExecutionEvidenceError(f"Expected carrier records exceed the bounded limit of {_MAX_ARTIFACT_OBSERVATIONS}")
    projector = _ManifestPlanProjector(plan, evidence_path)
    execution_rows = projector.validated_rows(input_executions, require_all=True)
    with _pin_output_root(Path(output_root)) as root:
        return _build_execution_evidence_under_root(
            plan,
            root=root,
            projector=projector,
            execution_rows=execution_rows,
            artifact_observations=artifact_observations,
            derive_manifest_outputs=derive_manifest_outputs,
            expected_artifact_records=expected_artifact_records or {},
            expected_carrier_records=expected_carrier_records or (),
            require_carrier_outcome_projections=require_carrier_outcome_projections,
        )


def _build_execution_evidence_under_root(
    plan: CanonicalExecutionPlan,
    *,
    root: _PinnedOutputRoot,
    projector: _ManifestPlanProjector,
    execution_rows: tuple[InputExecution, ...],
    artifact_observations: Sequence[ArtifactObservation],
    derive_manifest_outputs: bool,
    expected_artifact_records: Mapping[tuple[str, Optional[str]], Sequence[Mapping[str, Any]]],
    expected_carrier_records: Sequence[Mapping[str, Any]],
    require_carrier_outcome_projections: bool,
) -> dict[str, Any]:
    """Build evidence while retaining one pinned output-root authority."""

    evidence_path = projector.evidence_path
    execution_by_id = {row.input_id: row for row in execution_rows}
    distinct_backends = sorted(
        {row.executed_backend for row in execution_rows if isinstance(row.executed_backend, str) and row.executed_backend}
    )
    declarations = _requested_declarations(plan)
    declaration_kinds = {str(item["artifact_kind"]): item for item in declarations}
    if tuple(declaration_kinds) != plan.requested_outputs:
        raise ExecutionEvidenceError("Prepared output declarations do not match requested_outputs order")

    normalized_expected_carriers: dict[str, tuple[str, int]] = {}
    for record in expected_carrier_records:
        path = record.get("path") if isinstance(record, Mapping) else None
        sha256 = record.get("sha256") if isinstance(record, Mapping) else None
        size_bytes = record.get("size_bytes") if isinstance(record, Mapping) else None
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ExecutionEvidenceError("Expected carrier records must contain canonical path, sha256, and size_bytes")
        try:
            canonical_path = root.confined_relative_path(Path(path))
        except ArtifactEvidenceError as exc:
            raise ExecutionEvidenceError("Expected carrier record path is not confined") from exc
        if canonical_path != path or path in normalized_expected_carriers:
            raise ExecutionEvidenceError("Expected carrier record paths must be unique and canonical")
        normalized_expected_carriers[path] = (sha256, size_bytes)

    manifest_context = _manifest_validation_context(
        projector,
        execution_rows,
        expected_carrier_records=normalized_expected_carriers,
    )

    normalized_expected_records: dict[tuple[str, Optional[str]], tuple[tuple[str, str, int], ...]] = {}
    for raw_key, raw_records in expected_artifact_records.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise ExecutionEvidenceError("Expected artifact record keys must be (artifact_kind, input_id) pairs")
        artifact_kind, input_id = raw_key
        declaration = declaration_kinds.get(artifact_kind)
        if declaration is None:
            raise ExecutionEvidenceError(f"Expected artifact records are not requested by the plan: {artifact_kind!r}")
        if declaration["scope"] != OutputScope.PER_INPUT.value or input_id not in execution_by_id:
            raise ExecutionEvidenceError("Expected artifact records must identify one known per-input output")
        normalized: list[tuple[str, str, int]] = []
        for record in raw_records:
            path = record.get("path") if isinstance(record, Mapping) else None
            sha256 = record.get("sha256") if isinstance(record, Mapping) else None
            size_bytes = record.get("size_bytes") if isinstance(record, Mapping) else None
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                raise ExecutionEvidenceError("Expected artifact records must contain path, sha256, and size_bytes")
            normalized.append((path, sha256, size_bytes))
        if not normalized:
            raise ExecutionEvidenceError("Expected artifact records must not be empty")
        normalized_expected_records[(artifact_kind, input_id)] = tuple(sorted(normalized))

    observations: dict[tuple[str, Optional[str]], list[Path]] = {}
    observation_failures: dict[tuple[str, Optional[str]], str] = {}
    observation_failure_counts: dict[tuple[str, Optional[str]], int] = {}
    total_observation_count = len(artifact_observations)
    for observation in artifact_observations:
        if not isinstance(observation, ArtifactObservation):
            raise TypeError("artifact_observations must contain ArtifactObservation values")
        declaration = declaration_kinds.get(observation.artifact_kind)
        if declaration is None:
            raise ExecutionEvidenceError(f"Artifact observation is not requested by the plan: {observation.artifact_kind!r}")
        expected_scope = get_output_definition(observation.artifact_kind).scope.value
        if expected_scope == OutputScope.PER_INPUT.value:
            if observation.input_id not in execution_by_id:
                raise ExecutionEvidenceError(f"Per-input artifact {observation.artifact_kind!r} lacks a known input id")
        elif observation.input_id is not None:
            raise ExecutionEvidenceError(f"Per-run artifact {observation.artifact_kind!r} must not carry an input id")
        key = (observation.artifact_kind, observation.input_id)
        if (observation.path is None) == (observation.failure_code is None):
            raise ExecutionEvidenceError("Artifact observations must carry exactly one path or typed failure")
        if observation.failure_code is not None:
            if observation.failure_code not in _EXPLICIT_FAILURE_CODES:
                raise ExecutionEvidenceError(f"Unsupported explicit artifact failure {observation.failure_code!r}")
            if key in observation_failures:
                raise ExecutionEvidenceError(f"Duplicate explicit artifact failure for {key!r}")
            observation_failures[key] = observation.failure_code
        else:
            assert observation.path is not None
            candidate_paths = observations.setdefault(key, [])
            candidate_paths.append(Path(observation.path))
            if len(candidate_paths) > _MAX_ARTIFACTS_PER_OUTCOME:
                raise ExecutionEvidenceError(
                    f"Artifact observations for {key!r} exceed the bounded cardinality limit "
                    f"of {_MAX_ARTIFACTS_PER_OUTCOME}"
                )

    capture_budget = _CaptureBudget()
    combined_authority_by_input: dict[str, tuple[_ArtifactCapture, _CombinedManifestAuthority]] = {}
    for plan_input in plan.inputs:
        candidates = observations.get(("combined_manifest_json", plan_input.input_id), [])
        if len(candidates) != 1:
            continue
        try:
            capture = _capture_artifact(
                root,
                candidates[0],
                budget=capture_budget,
                max_bytes=_MAX_EVIDENCE_BYTES,
            )
            _validate_expected_carrier_capture(capture, context=manifest_context)
            manifest_payload = _validate_bound_manifest_artifact(
                artifact_kind="combined_manifest_json",
                input_id=plan_input.input_id,
                candidate=candidates[0],
                record=capture.record,
                root=root,
                context=manifest_context,
            )
            combined_authority_by_input[plan_input.input_id] = (
                capture,
                _compact_combined_manifest_authority(
                    manifest_payload,
                    manifest_budget=manifest_context.manifest_budget,
                ),
            )
        except ArtifactEvidenceError:
            # The normal outcome pass below records the precise combined-
            # manifest failure. Other claimed per-input outputs cannot use an
            # invalid manifest as their input/path authority.
            continue
    combined_manifest_paths = {
        input_id: capture.record["path"] for input_id, (capture, _payload) in combined_authority_by_input.items()
    }
    combined_manifest_authorities = {
        input_id: authority for input_id, (_capture, authority) in combined_authority_by_input.items()
    }
    expected_reconstruction_paths: Optional[tuple[Path, ...]] = None
    if "reconstruction_bundle" in declaration_kinds:
        expected_reconstruction_paths = tuple(observations.get(("reconstruction_bundle", None), ()))
    if derive_manifest_outputs:
        for plan_input in plan.inputs:
            combined_entry = combined_authority_by_input.get(plan_input.input_id)
            if combined_entry is None:
                continue
            combined_authority = combined_entry[1]
            for artifact_kind, declaration in declaration_kinds.items():
                if artifact_kind == "combined_manifest_json" or declaration["scope"] != OutputScope.PER_INPUT.value:
                    continue
                key = (artifact_kind, plan_input.input_id)
                if key in observations or key in observation_failures:
                    continue
                try:
                    declared_paths = combined_authority.declared_paths(artifact_kind)
                except _ArtifactCardinalityError as exc:
                    if total_observation_count + exc.observed_count > _MAX_ARTIFACT_OBSERVATIONS:
                        raise ExecutionEvidenceError(
                            f"Artifact observations exceed the bounded limit of {_MAX_ARTIFACT_OBSERVATIONS}"
                        ) from exc
                    total_observation_count += exc.observed_count
                    observation_failures[key] = exc.code
                    observation_failure_counts[key] = exc.observed_count
                    continue
                if declared_paths:
                    if total_observation_count + len(declared_paths) > _MAX_ARTIFACT_OBSERVATIONS:
                        raise ExecutionEvidenceError(
                            f"Artifact observations exceed the bounded limit of {_MAX_ARTIFACT_OBSERVATIONS}"
                        )
                    total_observation_count += len(declared_paths)
                    observations[key] = [Path(declared_path) for declared_path in declared_paths]

    produced: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_identities: set[tuple[int, int]] = set()

    def account(declaration: Mapping[str, Any], input_id: Optional[str]) -> None:
        outcome_key = (str(declaration["artifact_kind"]), input_id)
        explicit_failure = observation_failures.get(outcome_key)
        if explicit_failure is not None:
            failure = {
                **_outcome_base(declaration, input_id),
                "reason_code": explicit_failure,
            }
            if outcome_key in observation_failure_counts:
                failure["observed_count"] = observation_failure_counts[outcome_key]
            failed.append(failure)
            return
        candidates = observations.get(outcome_key, [])
        input_status = execution_by_id[input_id].status if input_id is not None else None
        if not candidates:
            bucket, outcome = _missing_outcome(declaration, input_id=input_id, input_status=input_status)
            (failed if bucket == "failed" else omitted).append(outcome)
            return

        cardinality = str(declaration["cardinality"])
        if len(candidates) > _MAX_ARTIFACTS_PER_OUTCOME:
            failed.append(
                {
                    **_outcome_base(declaration, input_id),
                    "reason_code": "artifact_cardinality_limit_exceeded",
                    "observed_count": len(candidates),
                }
            )
            return
        if cardinality == OutputCardinality.ONE.value and len(candidates) != 1:
            failed.append(
                {
                    **_outcome_base(declaration, input_id),
                    "reason_code": "cardinality_mismatch",
                    "observed_count": len(candidates),
                }
            )
            return

        artifact_kind = str(declaration["artifact_kind"])
        if input_id is not None and artifact_kind != "combined_manifest_json":
            combined_entry = combined_authority_by_input.get(input_id)
            combined_authority = combined_entry[1] if combined_entry is not None else None
            try:
                expected_paths = () if combined_authority is None else combined_authority.declared_paths(artifact_kind)
            except _ArtifactCardinalityError as exc:
                failed.append(
                    {
                        **_outcome_base(declaration, input_id),
                        "reason_code": exc.code,
                        "observed_count": exc.observed_count,
                    }
                )
                return
            try:
                candidate_relatives = sorted(root.confined_relative_path(candidate) for candidate in candidates)
            except ArtifactEvidenceError:
                # Preserve the more precise path-integrity reason emitted by
                # the normal capture path below.
                pass
            else:
                try:
                    expected_relatives = sorted(root.confined_relative_path(Path(expected)) for expected in expected_paths)
                except ArtifactEvidenceError:
                    expected_relatives = []
                if candidate_relatives != expected_relatives:
                    failed.append(
                        {
                            **_outcome_base(declaration, input_id),
                            "reason_code": "artifact_input_mismatch",
                        }
                    )
                    return

        artifact_records: list[dict[str, Any]] = []
        local_paths: set[str] = set()
        local_identities: set[tuple[int, int]] = set()
        try:
            for candidate in candidates:
                if artifact_kind == "combined_manifest_json" and input_id in combined_authority_by_input:
                    capture = combined_authority_by_input[input_id][0]
                else:
                    capture = _capture_artifact(
                        root,
                        candidate,
                        budget=capture_budget,
                        max_bytes=_MAX_EVIDENCE_BYTES if artifact_kind in _MANIFEST_ARTIFACT_KINDS else None,
                    )
                _validate_expected_carrier_capture(capture, context=manifest_context)
                record = capture.record
                if artifact_kind in _MANIFEST_ARTIFACT_KINDS and not (
                    artifact_kind == "combined_manifest_json" and input_id in combined_authority_by_input
                ):
                    _validate_bound_manifest_artifact(
                        artifact_kind=artifact_kind,
                        input_id=input_id,
                        candidate=candidate,
                        record=record,
                        root=root,
                        context=manifest_context,
                        combined_manifest_paths=combined_manifest_paths,
                        combined_manifest_authorities=combined_manifest_authorities,
                        expected_reconstruction_paths=expected_reconstruction_paths,
                    )
                if record["path"] in seen_paths or record["path"] in local_paths:
                    raise ArtifactEvidenceError(
                        "duplicate_artifact_path",
                        f"Artifact path is claimed by more than one output: {record['path']}",
                    )
                if capture.identity in seen_identities or capture.identity in local_identities:
                    raise ArtifactEvidenceError(
                        "duplicate_artifact_inode",
                        f"Artifact inode is claimed by more than one output: {record['path']}",
                    )
                local_paths.add(record["path"])
                local_identities.add(capture.identity)
                artifact_records.append(record)
        except ArtifactEvidenceError as exc:
            failed.append(
                {
                    **_outcome_base(declaration, input_id),
                    "reason_code": exc.code,
                }
            )
            return

        artifact_records.sort(key=lambda item: item["path"])
        expected_records = normalized_expected_records.get(outcome_key)
        if expected_records is not None:
            actual_records = tuple((record["path"], record["sha256"], record["size_bytes"]) for record in artifact_records)
            if actual_records != expected_records:
                failed.append(
                    {
                        **_outcome_base(declaration, input_id),
                        "reason_code": "artifact_changed",
                    }
                )
                return
        seen_paths.update(local_paths)
        seen_identities.update(local_identities)
        produced.append(
            {
                **_outcome_base(declaration, input_id),
                "artifacts": artifact_records,
            }
        )

    for declaration in declarations:
        if declaration["scope"] == OutputScope.PER_INPUT.value:
            for plan_input in plan.inputs:
                account(declaration, plan_input.input_id)
        else:
            account(declaration, None)

    missing_carrier_paths = set(normalized_expected_carriers) - manifest_context.observed_expected_carrier_paths
    if missing_carrier_paths:
        raise ExecutionEvidenceError(
            "Prepared artifact carriers are not bound by completion evidence: " f"count={len(missing_carrier_paths)}"
        )

    payload: dict[str, Any] = {
        "schema": EXECUTION_EVIDENCE_SCHEMA,
        "canonicalization": TP_CANONICAL_JSON_PROFILE,
        "plan_schema": plan.schema,
        "plan_fingerprint": plan.plan_fingerprint_sha256,
        "config_fingerprint_sha256": plan.config_fingerprint_sha256,
        "planned_backend": plan.planned_backend,
        "candidate_fallback_chain": list(plan.candidate_fallback_chain),
        "executed_backend": distinct_backends[0] if len(distinct_backends) == 1 else None,
        "requested_inputs": [
            {
                "input_id": plan_input.input_id,
                "path": plan_input.path,
                "status": execution_by_id[plan_input.input_id].status,
                "executed_backend": execution_by_id[plan_input.input_id].executed_backend,
                "error_code": execution_by_id[plan_input.input_id].error_code,
            }
            for plan_input in plan.inputs
        ],
        "requested_artifacts": declarations,
        "produced_artifacts": produced,
        "omitted_artifacts": omitted,
        "failed_artifacts": failed,
    }
    _validate_carrier_outcome_projections(
        payload,
        context=manifest_context,
        require_all=require_carrier_outcome_projections,
    )
    payload["evidence_fingerprint_sha256"] = compute_execution_evidence_fingerprint(payload)
    validate_execution_evidence_payload(payload, plan=plan)
    return payload


def _format_schema_path(parts: Sequence[Any]) -> str:
    if not parts:
        return "$"
    rendered = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in parts)
    return _bounded_rendered(rendered, _MAX_RENDERED_SCHEMA_PATH_CHARS)


def validate_execution_evidence_payload(
    payload: Mapping[str, Any],
    *,
    plan: CanonicalExecutionPlan,
) -> None:
    """Validate schema, plan binding, outcome coverage, and evidence digest."""

    _validate_prepared_plan(plan)
    if not isinstance(payload, Mapping):
        raise ExecutionEvidenceError("Execution evidence payload must be an object")
    collection_limits = {
        "requested_inputs": _MAX_PLAN_INPUTS,
        "requested_artifacts": _MAX_PLAN_OUTPUTS,
        "produced_artifacts": _MAX_ARTIFACT_OBSERVATIONS,
        "omitted_artifacts": _MAX_ARTIFACT_OBSERVATIONS,
        "failed_artifacts": _MAX_ARTIFACT_OBSERVATIONS,
    }
    for field_name, maximum in collection_limits.items():
        value = payload.get(field_name)
        if isinstance(value, list) and len(value) > maximum:
            raise ExecutionEvidenceError(f"Execution evidence {field_name} exceeds the bounded limit of {maximum}")

    visited_nodes = 0

    def validate_scalar_bounds(current: Any, *, depth: int, ancestors: set[int]) -> None:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > _MAX_EVIDENCE_VALIDATION_NODES:
            raise ExecutionEvidenceError(
                "Execution evidence exceeds the bounded validation node limit of " f"{_MAX_EVIDENCE_VALIDATION_NODES}"
            )
        if depth > _MAX_EVIDENCE_NESTING_DEPTH:
            raise ExecutionEvidenceError(
                "Execution evidence exceeds the bounded nesting limit of " f"{_MAX_EVIDENCE_NESTING_DEPTH}"
            )
        if type(current) is int and current.bit_length() > _MAX_EVIDENCE_INTEGER_BITS:
            raise ExecutionEvidenceError(
                "Execution evidence integer exceeds the bounded bit-length limit of " f"{_MAX_EVIDENCE_INTEGER_BITS}"
            )
        if isinstance(current, float) and not math.isfinite(current):
            raise ExecutionEvidenceError("Execution evidence contains a forbidden non-finite number")
        if not isinstance(current, (Mapping, list)):
            return
        identity = id(current)
        if identity in ancestors:
            raise ExecutionEvidenceError("Execution evidence contains a cyclic container")
        ancestors.add(identity)
        children = current.values() if isinstance(current, Mapping) else current
        for child in children:
            validate_scalar_bounds(child, depth=depth + 1, ancestors=ancestors)
        ancestors.remove(identity)

    validate_scalar_bounds(payload, depth=0, ancestors=set())
    import jsonschema

    validator = jsonschema.Draft202012Validator(load_execution_evidence_schema())
    first = next(validator.iter_errors(payload), None)
    if first is not None:
        raise ExecutionEvidenceError(
            f"Execution evidence schema validation failed at {_format_schema_path(list(first.absolute_path))}: "
            f"{_bounded_rendered(first.message)}"
        ) from first

    expected_plan_fields = {
        "plan_schema": plan.schema,
        "plan_fingerprint": plan.plan_fingerprint_sha256,
        "config_fingerprint_sha256": plan.config_fingerprint_sha256,
        "planned_backend": plan.planned_backend,
        "candidate_fallback_chain": list(plan.candidate_fallback_chain),
    }
    for field_name, expected in expected_plan_fields.items():
        if payload[field_name] != expected:
            raise ExecutionEvidenceError(f"Execution evidence {field_name} does not match the prepared plan")

    expected_inputs = [(item.input_id, item.path) for item in plan.inputs]
    observed_inputs = [(item["input_id"], item["path"]) for item in payload["requested_inputs"]]
    if observed_inputs != expected_inputs:
        raise ExecutionEvidenceError("Execution evidence requested_inputs do not match the prepared plan")
    validated_inputs = _validated_input_executions(
        plan,
        tuple(
            InputExecution(
                input_id=item["input_id"],
                status=item["status"],
                executed_backend=item["executed_backend"],
                error_code=item["error_code"],
            )
            for item in payload["requested_inputs"]
        ),
        require_all=True,
    )
    input_backends = sorted(
        {
            item.executed_backend
            for item in validated_inputs
            if isinstance(item.executed_backend, str) and item.executed_backend
        }
    )
    expected_executed_backend = input_backends[0] if len(input_backends) == 1 else None
    if payload["executed_backend"] != expected_executed_backend:
        raise ExecutionEvidenceError("Execution evidence executed_backend does not match requested_inputs")

    declarations = _requested_declarations(plan)
    if list(payload["requested_artifacts"]) != declarations:
        raise ExecutionEvidenceError("Execution evidence requested_artifacts do not match plan declarations")
    declarations_by_id = {item["declaration_id"]: item for item in declarations}

    expected_outcomes: set[tuple[str, Optional[str]]] = set()
    for declaration in declarations:
        if declaration["scope"] == OutputScope.PER_INPUT.value:
            expected_outcomes.update((declaration["declaration_id"], item.input_id) for item in plan.inputs)
        else:
            expected_outcomes.add((declaration["declaration_id"], None))
    observed_outcomes: set[tuple[str, Optional[str]]] = set()
    observed_paths: set[str] = set()
    recorded_artifact_bytes = 0
    for bucket_name in ("produced_artifacts", "omitted_artifacts", "failed_artifacts"):
        for outcome in payload[bucket_name]:
            key = (outcome["declaration_id"], outcome["input_id"])
            if key in observed_outcomes:
                raise ExecutionEvidenceError(f"Execution evidence repeats artifact outcome {key!r}")
            observed_outcomes.add(key)
            outcome_declaration = declarations_by_id.get(outcome["declaration_id"])
            if outcome_declaration is None or any(outcome[field] != value for field, value in outcome_declaration.items()):
                raise ExecutionEvidenceError(f"Execution evidence outcome {key!r} drifts from its plan declaration")
            if bucket_name == "omitted_artifacts" and outcome["required"]:
                raise ExecutionEvidenceError("Required artifact outputs may not be classified as omitted")
            if bucket_name == "produced_artifacts":
                artifact_count = len(outcome["artifacts"])
                if outcome["cardinality"] == OutputCardinality.ONE.value and artifact_count != 1:
                    raise ExecutionEvidenceError("Produced cardinality-one output must contain exactly one artifact")
                if artifact_count < 1:
                    raise ExecutionEvidenceError("Produced output must contain at least one artifact")
                for artifact in outcome["artifacts"]:
                    if artifact["path"] in observed_paths:
                        raise ExecutionEvidenceError(f"Execution evidence repeats artifact path {artifact['path']!r}")
                    observed_paths.add(artifact["path"])
                    if artifact["size_bytes"] > _MAX_ARTIFACT_BYTES:
                        raise ExecutionEvidenceError("Produced artifact exceeds the bounded per-file byte limit")
                    recorded_artifact_bytes += artifact["size_bytes"]
                    if recorded_artifact_bytes > _MAX_AGGREGATE_ARTIFACT_BYTES:
                        raise ExecutionEvidenceError("Produced artifacts exceed the bounded aggregate byte limit")
            elif bucket_name == "omitted_artifacts" and outcome["reason_code"] not in {
                "optional_stage_no_output",
                "upstream_failed",
                "upstream_not_executed",
            }:
                raise ExecutionEvidenceError("Omitted artifact uses a failure-only reason code")
            elif bucket_name == "failed_artifacts":
                if outcome["reason_code"] == "optional_stage_no_output" or (
                    outcome["reason_code"] == "required_output_missing" and not outcome["required"]
                ):
                    raise ExecutionEvidenceError("Failed artifact uses an outcome-incompatible reason code")
                has_observed_count = "observed_count" in outcome
                counted_failures = {"cardinality_mismatch", "artifact_cardinality_limit_exceeded"}
                if (outcome["reason_code"] in counted_failures) != has_observed_count:
                    raise ExecutionEvidenceError("Cardinality failures must exclusively carry observed_count")
    if observed_outcomes != expected_outcomes:
        raise ExecutionEvidenceError("Execution evidence outcomes do not exactly cover plan declarations")

    expected_fingerprint = compute_execution_evidence_fingerprint(payload)
    if payload["evidence_fingerprint_sha256"] != expected_fingerprint:
        raise ExecutionEvidenceError("evidence_fingerprint_sha256 does not match the canonical evidence body")


def require_required_artifacts(payload: Mapping[str, Any]) -> None:
    """Fail required omissions and any claimed artifact integrity failure."""

    failures = [
        (item["declaration_id"], item["input_id"], item["reason_code"])
        for item in payload.get("failed_artifacts", [])
        if item.get("required") is True or item.get("reason_code") == "artifact_changed"
    ]
    if failures:
        raise ExecutionEvidenceError(
            f"Prepared execution failed required artifact accounting or carrier integrity: {failures}"
        )


def write_execution_evidence(
    path: Path,
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    plan: CanonicalExecutionPlan,
) -> None:
    """Validate and durably publish exact bytes under a pinned output root."""

    validate_execution_evidence_payload(payload, plan=plan)
    data = canonicalize_json(payload)
    with _pin_output_root(Path(output_root)) as root:
        relative_path = root.confined_relative_path(Path(path))
        _secure_atomic_write_bytes(root, relative_path, data)


def verify_execution_evidence_file(
    path: Path,
    *,
    output_root: Path,
    plan: CanonicalExecutionPlan,
) -> dict[str, Any]:
    """Revalidate canonical evidence and every claimed produced file.

    The detached sidecar is the prepared run's completion record. Verification
    therefore reopens it through the same confined-path boundary used for
    artifacts, checks canonical bytes and semantic plan binding, and then
    compares every produced file's complete digest, byte size, and media
    metadata against the recorded claim.
    """

    with _pin_output_root(Path(output_root)) as root:
        return _verify_execution_evidence_under_root(
            Path(path),
            root=root,
            plan=plan,
        )


def _verify_execution_evidence_under_root(
    path: Path,
    *,
    root: _PinnedOutputRoot,
    plan: CanonicalExecutionPlan,
) -> dict[str, Any]:
    """Verify one evidence file while retaining its output-root authority."""

    data, evidence_relative_path, _identity = _read_confined_artifact_bytes(
        root,
        Path(path),
        context="Execution evidence",
    )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ExecutionEvidenceError(f"Duplicate execution evidence member: {_bounded_rendered(key)!r}")
            decoded[key] = value
        return decoded

    def reject_non_finite(value: str) -> Any:
        raise ExecutionEvidenceError(f"Non-finite execution evidence number is forbidden: {value}")

    try:
        text = data.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except ExecutionEvidenceError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ExecutionEvidenceError("Execution evidence is not one valid UTF-8 JSON object") from exc
    if not isinstance(payload, dict):
        raise ExecutionEvidenceError("Execution evidence must contain one JSON object")

    validate_execution_evidence_payload(payload, plan=plan)
    if canonicalize_json(payload) != data:
        raise ExecutionEvidenceError("Execution evidence bytes are not canonical tp.canonical.json.v1")
    execution_rows = tuple(
        InputExecution(
            input_id=item["input_id"],
            status=item["status"],
            executed_backend=item["executed_backend"],
            error_code=item["error_code"],
        )
        for item in payload["requested_inputs"]
    )
    manifest_context = _manifest_validation_context(
        _ManifestPlanProjector(plan, evidence_relative_path),
        execution_rows,
    )
    capture_budget = _CaptureBudget()
    observed_identities: set[tuple[int, int]] = set()
    for outcome in payload["produced_artifacts"]:
        artifact_kind = outcome["artifact_kind"]
        for recorded in outcome["artifacts"]:
            capture = _capture_artifact(
                root,
                Path(recorded["path"]),
                budget=capture_budget,
                max_bytes=_MAX_EVIDENCE_BYTES if artifact_kind in _MANIFEST_ARTIFACT_KINDS else None,
            )
            if capture.record != recorded:
                raise ExecutionEvidenceError(f"Produced artifact evidence does not match final bytes: {recorded['path']!r}")
            if capture.identity in observed_identities:
                raise ExecutionEvidenceError(f"Produced artifacts repeat inode identity at {recorded['path']!r}")
            observed_identities.add(capture.identity)

    combined_authority_by_input: dict[str, _CombinedManifestAuthority] = {}
    combined_manifest_paths: dict[str, str] = {}
    for outcome in payload["produced_artifacts"]:
        if outcome["artifact_kind"] != "combined_manifest_json":
            continue
        input_id = outcome["input_id"]
        if not isinstance(input_id, str):
            raise ExecutionEvidenceError("Produced combined manifest is missing its input identity")
        recorded = outcome["artifacts"][0]
        manifest_payload = _validate_bound_manifest_artifact(
            artifact_kind="combined_manifest_json",
            input_id=input_id,
            candidate=Path(recorded["path"]),
            record=recorded,
            root=root,
            context=manifest_context,
        )
        combined_authority_by_input[input_id] = _compact_combined_manifest_authority(
            manifest_payload,
            manifest_budget=manifest_context.manifest_budget,
        )
        combined_manifest_paths[input_id] = recorded["path"]

    expected_reconstruction_paths: Optional[tuple[Path, ...]] = None
    produced_reconstruction = next(
        (item for item in payload["produced_artifacts"] if item["artifact_kind"] == "reconstruction_bundle"),
        None,
    )
    if produced_reconstruction is not None:
        expected_reconstruction_paths = tuple(Path(record["path"]) for record in produced_reconstruction["artifacts"])
    elif any(item["artifact_kind"] == "reconstruction_bundle" for item in payload["omitted_artifacts"]):
        expected_reconstruction_paths = ()

    for outcome in payload["produced_artifacts"]:
        artifact_kind = outcome["artifact_kind"]
        if artifact_kind not in {"batch_manifest_json", "run_card"}:
            continue
        for recorded in outcome["artifacts"]:
            _validate_bound_manifest_artifact(
                artifact_kind=artifact_kind,
                input_id=outcome["input_id"],
                candidate=Path(recorded["path"]),
                record=recorded,
                root=root,
                context=manifest_context,
                combined_manifest_paths=combined_manifest_paths,
                combined_manifest_authorities=combined_authority_by_input,
                expected_reconstruction_paths=expected_reconstruction_paths,
            )

    for outcome in payload["produced_artifacts"]:
        input_id = outcome["input_id"]
        artifact_kind = outcome["artifact_kind"]
        if input_id is None or artifact_kind == "combined_manifest_json":
            continue
        combined_authority = combined_authority_by_input.get(input_id)
        if combined_authority is None:
            raise ExecutionEvidenceError(
                f"Produced per-input artifact {artifact_kind!r} has no valid combined-manifest authority"
            )
        expected_paths = combined_authority.declared_paths(artifact_kind)
        recorded_paths = sorted(item["path"] for item in outcome["artifacts"])
        try:
            expected_relative_paths = sorted(root.confined_relative_path(Path(expected)) for expected in expected_paths)
        except ArtifactEvidenceError as exc:
            raise ExecutionEvidenceError(
                f"Combined manifest declares an invalid {artifact_kind!r} path for {input_id!r}"
            ) from exc
        if recorded_paths != expected_relative_paths:
            raise ExecutionEvidenceError(
                f"Produced artifact paths for {artifact_kind!r} do not match combined-manifest input authority"
            )
    _validate_carrier_outcome_projections(
        payload,
        context=manifest_context,
        require_all=True,
    )
    return payload


__all__ = [
    "EXECUTION_EVIDENCE_SCHEMA",
    "ArtifactEvidenceError",
    "ArtifactObservation",
    "ConfinedArtifactSnapshot",
    "ExecutionEvidenceError",
    "InputExecution",
    "MANIFEST_OUTCOME_PROJECTION_SCHEMA",
    "build_execution_evidence",
    "build_manifest_outcome_projection",
    "build_manifest_plan_projection",
    "compute_execution_evidence_fingerprint",
    "load_execution_evidence_schema",
    "read_confined_artifact_snapshot",
    "require_required_artifacts",
    "validate_execution_evidence_payload",
    "verify_execution_evidence_file",
    "write_execution_evidence",
]
