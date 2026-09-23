"""Bounded V6 completion verification with exact parent and semantic replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.evidence import _inventory
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot

from .color import GradeRecipe, RenderRecipe
from .plan import MAX_PLAN_BYTES, GradePlan, depth_recipe, digest, processing_identity, source_binding, validate_resources
from .products import image_products
from .source import SourceLimits, VerifiedV5Source, prepare_source, validate_source

EVIDENCE_SCHEMA = "tp.lux.grade.evidence.v1"


@dataclass(frozen=True)
class VerifiedGradeEvidence:
    output_root: Path
    canonical_bytes: bytes
    plan_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)


def verify_artifacts(
    root: Path,
    plan: GradePlan,
    source: VerifiedV5Source,
    records: list[dict[str, Any]],
    *,
    checkpoint: Callable[[], None],
    completed: bool,
) -> None:
    """Replay every expected product; an updated inventory alone proves nothing."""
    payload = plan.to_payload()
    depth_maps = depth_recipe(payload)
    if payload["source"] != source_binding(source, depth_maps=depth_maps is not None) or payload[
        "processing"
    ] != processing_identity(depth_maps=depth_maps is not None):
        raise ValueError("V6 source or processing identity differs from the frozen plan")
    maximum_records = (21 if depth_maps is not None else 8) * len(source.images) + 1
    if not isinstance(records, list) or len(records) > maximum_records:
        raise ValueError("V6 inventory exceeds its bound")
    declared: dict[str, dict[str, Any]] = {}
    total = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
            raise ValueError("Malformed V6 inventory record")
        path, size = record["path"], record["size_bytes"]
        if not isinstance(path, str) or path in declared or type(size) is not int or size <= 0:
            raise ValueError("Duplicate or invalid V6 inventory record")
        declared[path] = record
        total += size
    if total > payload["limits"]["max_output_bytes"]:
        raise ValueError("V6 output exceeds its byte budget")
    seen: set[str] = set()

    def check_product(relative: str, expected: bytes) -> None:
        checkpoint()
        record = declared.get(relative)
        if record != {"path": relative, "size_bytes": len(expected), "sha256": digest(expected)}:
            raise ValueError(f"V6 product differs from semantic replay: {relative}")
        _, observed = snapshot(root, root / relative, maximum_bytes=len(expected), retain_bytes=False)
        if observed != record:
            raise ValueError(f"V6 output bytes changed: {relative}")
        seen.add(relative)

    check_product("plan.json", plan.canonical_bytes)
    grade, render = GradeRecipe.from_payload(payload["grade"]), RenderRecipe.from_payload(payload["render"])
    for image in source.images:
        for relative, data in image_products(
            source, image.input_id, grade, render, checkpoint=checkpoint, depth_maps=depth_maps
        ):
            check_product(relative, data)
    if set(declared) != seen:
        raise ValueError("V6 inventory contains unexpected products")
    expected_files = seen | ({"evidence.json"} if completed else set())
    if _inventory(root) != expected_files:
        raise ValueError("V6 output namespace differs from the exact product inventory")
    checkpoint()
    validate_source(source, cancellation=lambda: _cancelled(checkpoint))
    if payload["processing"] != processing_identity(depth_maps=depth_maps is not None):
        raise ValueError("V6 processing source changed during verification")


def _cancelled(checkpoint: Callable[[], None]) -> bool:
    checkpoint()
    return False


def verify_execution_evidence(
    output_root: Path,
    *,
    source_root: Path,
    expected_plan_sha256: str | None = None,
    source_limits: SourceLimits | None = None,
    cancellation: Callable[[], bool] | None = None,
) -> VerifiedGradeEvidence:
    """Verify the retained parent, canonical plan, complete inventory, and pixels."""
    import time

    started = time.monotonic()
    deadline = 3600

    def checkpoint() -> None:
        if cancellation is not None and cancellation():
            raise RuntimeError("V6 verification cancelled")
        if time.monotonic() - started > deadline:
            raise RuntimeError("V6 verification exceeded its wall-time budget")

    root = directory_path(output_root)
    with pinned_directory(root):
        raw, _ = snapshot(root, root / "plan.json", maximum_bytes=MAX_PLAN_BYTES)
        plan = GradePlan(raw)
        if expected_plan_sha256 is not None and plan.sha256 != expected_plan_sha256:
            raise ValueError("V6 plan differs from the expected exact bytes")
        payload = plan.to_payload()
        deadline = payload["limits"]["wall_time_seconds"]
        raw, completion_snapshot = snapshot(root, root / "evidence.json", maximum_bytes=MAX_PLAN_BYTES)
        evidence = decode_bounded_json_object(raw)
        keys = {"schema", "plan_sha256", "source_digest", "artifacts", "input_count", "production_acceptance"}
        if (
            set(evidence) != keys
            or evidence["schema"] != EVIDENCE_SCHEMA
            or evidence["plan_sha256"] != plan.sha256
            or evidence["source_digest"] != payload["source"]["digest"]
            or evidence["production_acceptance"] != "not_established"
            or type(evidence["input_count"]) is not int
            or evidence["input_count"] != len(payload["source"]["images"])
            or canonicalize_json(evidence) != raw
        ):
            raise ValueError("Invalid V6 completion evidence")
        if not isinstance(evidence["artifacts"], list):
            raise ValueError("Invalid V6 artifact inventory")
        admitted_limits = SourceLimits(**payload["source"]["limits"])
        # A caller may constrain verification further, never silently raise the
        # frozen input/memory admission limits. The source binding retains the
        # admitted limits after that additional independent preflight.
        if source_limits is not None:
            constrained = SourceLimits(
                **{
                    name: min(getattr(admitted_limits, name), getattr(source_limits, name))
                    for name in ("max_input_bytes", "max_pixels", "memory_mib")
                }
            )
            # The caller's tighter budget applies to reconstruction and product
            # replay as well as retained-parent admission. Keep exact frozen
            # plan bytes/source identity unchanged after this extra preflight.
            validate_resources({**payload, "source": {**payload["source"], "limits": constrained.to_payload()}})
            prepare_source(source_root, limits=constrained, cancellation=lambda: _cancelled(checkpoint))
        source = prepare_source(source_root, limits=admitted_limits, cancellation=lambda: _cancelled(checkpoint))
        verify_artifacts(root, plan, source, evidence["artifacts"], checkpoint=checkpoint, completed=True)
        if sum(record["size_bytes"] for record in evidence["artifacts"]) + len(raw) > payload["limits"]["max_output_bytes"]:
            raise ValueError("V6 completion exceeds its total byte budget")
        checkpoint()
        # Parent verification and numeric replay can take significant time. Bind
        # the completion file again before accepting the bytes read at entry;
        # root pinning alone does not detect a replaced or edited child file.
        _, observed_completion = snapshot(
            root,
            root / "evidence.json",
            maximum_bytes=completion_snapshot["size_bytes"],
            retain_bytes=False,
        )
        if observed_completion != completion_snapshot:
            raise ValueError("V6 completion evidence changed during verification")
    return VerifiedGradeEvidence(root, raw, plan.sha256)
