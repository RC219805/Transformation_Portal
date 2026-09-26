"""One admission/run/verification surface over the established LuxDepth engines.

No schema conversion, automatic model fallback, or second execution loop lives
here. Exact typed carriers select fixed in-repository engines; managed callers
retain their publisher, process ownership, and frozen dispatch authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v4.lifecycle import LuxDepthV4Request, PreparedLuxExecutionV4
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, PreparedLuxExecutionV5
    from transformation_portal.lux_depth_v6.depth_pro import NativeDepthProRequest, PreparedNativeDepthPro
    from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request, PreparedManagedLuxExecutionV6
    from transformation_portal.lux_depth_v6.plan import LuxDepthV6Request, PreparedLuxExecutionV6
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits, GenerationPublisher

    LuxRequest = LuxDepthV4Request | LuxDepthV5Request | ManagedLuxDepthV6Request | LuxDepthV6Request | NativeDepthProRequest
    PreparedLuxDepth = (
        PreparedLuxExecutionV4
        | PreparedLuxExecutionV5
        | PreparedManagedLuxExecutionV6
        | PreparedLuxExecutionV6
        | PreparedNativeDepthPro
    )


@dataclass(frozen=True)
class _Engine:
    admission: str
    request: str
    prepared: str
    execution: str
    managed: bool = False


# Only these source-owned classes can select execution. Never import a module
# named by serialized data or by a request supplied by a caller.
_ENGINES = (
    _Engine("lux_depth_v4.lifecycle", "LuxDepthV4Request", "PreparedLuxExecutionV4", "lux_depth_v4.pipeline"),
    _Engine("lux_depth_v5.lifecycle", "LuxDepthV5Request", "PreparedLuxExecutionV5", "lux_depth_v5.pipeline", True),
    _Engine("lux_depth_v6.managed", "ManagedLuxDepthV6Request", "PreparedManagedLuxExecutionV6", "lux_depth_v6.managed", True),
    _Engine("lux_depth_v6.plan", "LuxDepthV6Request", "PreparedLuxExecutionV6", "lux_depth_v6.pipeline"),
    _Engine("lux_depth_v6.depth_pro", "NativeDepthProRequest", "PreparedNativeDepthPro", "lux_depth_v6.depth_pro"),
)


def _module(name: str) -> Any:
    return import_module(f"transformation_portal.{name}")


def _engine(value: object, *, prepared: bool = False) -> _Engine:
    for engine in _ENGINES:
        name = engine.prepared if prepared else engine.request
        if (type(value).__module__, type(value).__name__) == (f"transformation_portal.{engine.admission}", name):
            if type(value) is getattr(_module(engine.admission), name):
                return engine
    raise TypeError("LuxDepth requires an exact supported request or prepared execution carrier")


def prepare(request: LuxRequest, *, publisher: GenerationPublisher | None = None) -> PreparedLuxDepth:
    """Freeze native plan bytes before inference or output creation."""
    engine = _engine(request)
    if publisher is not None and not engine.managed:
        raise ValueError("Only V5 inference and composite photography support managed publication")
    kwargs = {"publisher": publisher} if engine.managed else {}
    return _module(engine.admission).prepare(request, **kwargs)


def run(
    prepared: PreparedLuxDepth,
    *,
    cancellation: Callable[[], bool] | None = None,
    publication_limits: GenerationPublicationLimits | None = None,
    managed_process_group: bool = False,
) -> Any:
    """Consume the exact prepared carrier without rediscovery or replanning."""
    engine = _engine(prepared, prepared=True)
    if type(managed_process_group) is not bool:
        raise TypeError("managed_process_group must be an exact boolean")
    if cancellation is not None and cancellation():
        raise RuntimeError("LuxDepth execution cancelled")
    supports_dispatch = engine.admission in {"lux_depth_v5.lifecycle", "lux_depth_v6.managed"}
    if not supports_dispatch and (publication_limits is not None or managed_process_group):
        raise ValueError("This LuxDepth engine does not support managed execution options")
    kwargs: dict[str, Any] = {"cancellation": cancellation}
    if supports_dispatch:
        kwargs.update(publication_limits=publication_limits, managed_process_group=managed_process_group)
    return _module(engine.execution).run(prepared, **kwargs)


@dataclass(frozen=True)
class VerifiedLuxDepth:
    """Normalized verification receipt; native evidence bytes remain unchanged."""

    output_root: Path
    plan_sha256: str
    canonical_bytes: bytes

    def to_payload(self) -> dict[str, Any]:
        from transformation_portal.core.execution_plan import decode_bounded_json_object

        return decode_bounded_json_object(self.canonical_bytes)


def _plan_snapshot(root: Path) -> tuple[bytes, dict[str, Any]]:
    from transformation_portal.lux_depth_v4.io import snapshot

    candidates = [root / name for name in ("execution-plan.json", "plan.json")]
    found = [path for path in candidates if path.exists() or path.is_symlink()]
    if len(found) != 1:
        raise ValueError("LuxDepth verification requires exactly one native plan at the output root")
    return snapshot(root, found[0], maximum_bytes=16 * 1024**2)


def verify(
    output_root: Path,
    *,
    source_root: Path | None = None,
    expected_plan_sha256: str | None = None,
    cancellation: Callable[[], bool] | None = None,
) -> VerifiedLuxDepth:
    """Select a native verifier from bounded plan bytes, then replay and rehash.

    The optional digest always identifies the exact canonical plan bytes, never
    the older unsigned-plan fingerprint. V4/V5 fingerprints are passed only to
    their native validators. Retained-V5 finishing and Depth Pro require their
    original source root; self-contained inference/composite outputs do not.
    """
    from transformation_portal.core.execution_plan import decode_bounded_json_object
    from transformation_portal.core.execution_plan_v2 import require_digest
    from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory

    def check() -> None:
        if cancellation is not None and cancellation():
            raise RuntimeError("LuxDepth verification cancelled")

    check()
    if expected_plan_sha256 is not None:
        require_digest(expected_plan_sha256)
    root = directory_path(output_root)
    with pinned_directory(root):
        raw, record = _plan_snapshot(root)
        digest = hashlib.sha256(raw).hexdigest()
        if expected_plan_sha256 is not None and expected_plan_sha256 != digest:
            raise ValueError("LuxDepth plan differs from the expected exact bytes")
        schema = decode_bounded_json_object(raw).get("schema")
        check()
        if schema == "tp.execution.plan.v5":
            if source_root is not None:
                raise ValueError("Composite verification uses its retained source-v5 namespace; omit source_root")
            from transformation_portal.lux_depth_v6.managed import verify_managed_evidence

            managed_evidence = verify_managed_evidence(root, expected_plan_bytes=raw, cancellation=cancellation)
            evidence_bytes = managed_evidence.canonical_bytes
        elif schema in {"tp.execution.plan.v2", "tp.execution.plan.v3", "tp.execution.plan.v4"}:
            if source_root is not None:
                raise ValueError("Inference verification uses retained sources; omit source_root")
            from transformation_portal.core.execution_plan_v3 import parse_photography_plan
            from transformation_portal.lux_depth_v4.evidence import verify_execution_evidence_v2

            verifier = verify_execution_evidence_v2
            if schema == "tp.execution.plan.v4":
                from transformation_portal.core.execution_plan_v4 import parse_plan_v4
                from transformation_portal.lux_depth_v5.evidence import verify_execution_evidence_v3

                fingerprint = parse_plan_v4(raw).plan_fingerprint_sha256
                verifier = verify_execution_evidence_v3
            else:
                fingerprint = parse_photography_plan(raw).plan_fingerprint_sha256
            inference_evidence = verifier(root, expected_plan_sha256=fingerprint)
            evidence_bytes = inference_evidence.canonical_bytes
        elif schema in {"tp.lux.grade.plan.v1", "tp.lux.grade.plan.v2", "tp.lux.depth_pro.plan.v1"}:
            if source_root is None:
                raise ValueError("Finishing and Depth Pro verification require source_root")
            if schema == "tp.lux.depth_pro.plan.v1":
                from transformation_portal.lux_depth_v6.depth_pro import verify as verifier_with_source
            else:
                from transformation_portal.lux_depth_v6.evidence import verify_execution_evidence as verifier_with_source
            grade_evidence = verifier_with_source(
                root, source_root=source_root, expected_plan_sha256=digest, cancellation=cancellation
            )
            evidence_bytes = grade_evidence.canonical_bytes
        else:
            raise ValueError(f"Unsupported LuxDepth verification schema: {schema!r}")
        check()
        _, after = _plan_snapshot(root)
        if after != record:
            raise ValueError("LuxDepth plan changed during verification")
    return VerifiedLuxDepth(root, digest, evidence_bytes)


def result_summary(result: Any) -> dict[str, Any]:
    """Report a native completion without confusing fingerprints with byte hashes."""
    from transformation_portal.lux_depth_v4.io import directory_path

    root = directory_path(result.output_root)
    raw, _ = _plan_snapshot(root)
    count = result.to_payload()["input_count"] if hasattr(result, "to_payload") else result.input_count
    return {
        "output_root": str(root),
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "input_count": count,
        "production_acceptance": "not_established",
    }
