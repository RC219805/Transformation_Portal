"""Standalone V6 execution; successful completion is written only after replay."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import (
    _pin_output_root,
    _secure_atomic_write_bytes,
    _validate_pinned_root_namespace,
)
from transformation_portal.lux_depth_v4.io import directory_path

from .color import GradeRecipe, RenderRecipe
from .evidence import EVIDENCE_SCHEMA, verify_artifacts
from .plan import GradePlan, PreparedLuxExecutionV6, depth_recipe, digest, processing_identity, source_binding
from .products import image_products
from .source import validate_source


@dataclass(frozen=True)
class LuxDepthV6Result:
    output_root: Path
    evidence_path: Path
    plan_sha256: str
    input_count: int
    canonical_evidence_bytes: bytes = b""


def run(prepared: PreparedLuxExecutionV6, *, cancellation: Callable[[], bool] | None = None) -> LuxDepthV6Result:
    if type(prepared) is not PreparedLuxExecutionV6 or type(prepared.plan) is not GradePlan:
        raise TypeError("V6 execution requires the exact prepared grading carrier")
    plan = GradePlan(prepared.canonical_plan_bytes)
    payload = plan.to_payload()
    depth_maps = depth_recipe(payload)
    started = time.monotonic()

    def check() -> None:
        if cancellation is not None and cancellation():
            raise RuntimeError("V6 execution cancelled")
        if time.monotonic() - started > payload["limits"]["wall_time_seconds"]:
            raise RuntimeError("V6 execution exceeded its wall-time budget")

    check()
    if payload["source"] != source_binding(prepared.source, depth_maps=depth_maps is not None) or payload[
        "processing"
    ] != processing_identity(depth_maps=depth_maps is not None):
        raise ValueError("Prepared V6 source or processing identity changed")
    validate_source(prepared.source, cancellation=cancellation)
    root = directory_path(prepared.output_root, allow_missing=True)
    if root != prepared.output_root or root.exists() or not root.parent.is_dir():
        raise ValueError("V6 output must still be new with its admitted parent")
    if root.is_relative_to(prepared.source.root) or prepared.source.root.is_relative_to(root):
        raise ValueError("V6 input/output roots must remain disjoint")
    # No success artifact exists until all products have passed semantic replay.
    with _pin_output_root(root.parent) as parent:
        os.mkdir(root.name, mode=0o700, dir_fd=parent.descriptor)
        _validate_pinned_root_namespace(parent)
        with _pin_output_root(root) as pinned:
            records: list[dict[str, Any]] = []
            written = 0

            def write(relative: str, data: bytes, *, record: bool = True) -> None:
                nonlocal written
                check()
                _validate_pinned_root_namespace(pinned)
                if len(data) + written > payload["limits"]["max_output_bytes"] or len(data) > shutil.disk_usage(root).free:
                    raise RuntimeError("V6 product exceeds admitted disk/output budget")
                _secure_atomic_write_bytes(pinned, relative, data, maximum_bytes=max(1, len(data)))
                written += len(data)
                if record:
                    records.append({"path": relative, "size_bytes": len(data), "sha256": digest(data)})

            try:
                write("plan.json", plan.canonical_bytes)
                grade = GradeRecipe.from_payload(payload["grade"])
                render = RenderRecipe.from_payload(payload["render"])
                for image in prepared.source.images:
                    check()
                    _validate_pinned_root_namespace(pinned)
                    os.mkdir(image.input_id, mode=0o700, dir_fd=pinned.descriptor)
                    for relative, data in image_products(
                        prepared.source, image.input_id, grade, render, checkpoint=check, depth_maps=depth_maps
                    ):
                        write(relative, data)
                records.sort(key=lambda item: item["path"])
                verify_artifacts(root, plan, prepared.source, records, checkpoint=check, completed=False)
                evidence = canonicalize_json(
                    {
                        "schema": EVIDENCE_SCHEMA,
                        "plan_sha256": plan.sha256,
                        "source_digest": prepared.source.source_digest,
                        "artifacts": records,
                        "input_count": len(prepared.source.images),
                        "production_acceptance": "not_established",
                    }
                )
                write("evidence.json", evidence, record=False)
                _validate_pinned_root_namespace(pinned)
            except Exception:
                # Keep the exact failed attempt for inspection; never overwrite or
                # remove unrelated paths, and never label partial products complete.
                raise
    return LuxDepthV6Result(root, root / "evidence.json", plan.sha256, len(prepared.source.images), evidence)
