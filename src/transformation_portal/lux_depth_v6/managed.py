"""Frozen raw-photograph inference followed by complete, replayable V6 finishing.

Each stage keeps an independently verifiable namespace. The outer generation
binds their exact plans and complete inventories before fenced publication.
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import digest_payload
from transformation_portal.core.execution_plan_v5 import (
    MAX_COMPLETION_BYTES,
    MAX_PLAN_BYTES,
    ExecutionPlanV5,
    stage_output_budget,
)
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import _pin_output_root, _validate_pinned_root_namespace
from transformation_portal.lux_depth_v4.evidence import VerifiedArtifact, _inventory
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot, write_evidence
from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, PreparedLuxExecutionV5, validate_prepared_bindings
from transformation_portal.orchestrator.artifact_store.generation import (
    MAX_GENERATION_BYTES,
    MAX_GENERATION_FILE_BYTES,
    MAX_GENERATION_FILES,
    MAX_MANIFEST_BYTES,
    GenerationPublicationLimits,
    GenerationPublisher,
)

from .color import GradeRecipe, RenderRecipe
from .depth_maps import DepthMapRecipe
from .evidence import VerifiedGradeEvidence, verify_execution_evidence
from .plan import GradePlan, LuxDepthV6Request, OutputLimits, digest, processing_identity
from .source import SourceLimits

SOURCE_DIRECTORY = "source-v5"
DELIVERY_DIRECTORY = "v6"
COMPLETION_SCHEMA = "tp.lux.managed_v6.evidence.v1"
_MANAGED_MODULES = (
    "transformation_portal.core.execution_plan_v5",
    "transformation_portal.lux_depth_v6.managed",
    "transformation_portal.lux_depth_v6.publication",
)


def managed_processing_identity() -> dict[str, Any]:
    identity = processing_identity(depth_maps=True)
    for name in _MANAGED_MODULES:
        filename = importlib.import_module(name).__file__
        if filename is None or Path(filename).suffix != ".py":
            raise ValueError("Managed V6 requires auditable processing source")
        identity["modules"][name] = digest(Path(filename).read_bytes())
    return identity


@dataclass(frozen=True)
class ManagedLuxDepthV6Request:
    inference: LuxDepthV5Request
    grade: GradeRecipe = field(default_factory=GradeRecipe)
    render: RenderRecipe = field(default_factory=RenderRecipe)
    depth_maps: DepthMapRecipe = field(default_factory=DepthMapRecipe)


@dataclass(frozen=True)
class PreparedManagedLuxExecutionV6:
    plan: ExecutionPlanV5
    inference: PreparedLuxExecutionV5

    @property
    def canonical_plan_bytes(self) -> bytes:
        return self.plan.canonical_bytes

    @property
    def output_root(self) -> Path:
        return self.inference.output_root.parent


@dataclass(frozen=True)
class VerifiedManagedV6Evidence:
    output_root: Path
    canonical_bytes: bytes
    artifacts: tuple[VerifiedArtifact, ...]

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)


def prepare(
    request: ManagedLuxDepthV6Request,
    *,
    publisher: GenerationPublisher | None = None,
    publication_limits: GenerationPublicationLimits | None = None,
) -> PreparedManagedLuxExecutionV6:
    """Freeze choices before model/output creation, with bounded local or managed admission.

    Without a publisher, the same composite runs locally under the generation
    limits. This admits products only; it does not authorize managed publication.
    """
    from transformation_portal.lux_depth_v5.lifecycle import prepare as prepare_inference

    from .publication import validate_publication_plan

    if type(request) is not ManagedLuxDepthV6Request:
        raise TypeError("Managed V6 preparation requires its exact request carrier")
    if publisher is not None and publication_limits is not None:
        raise ValueError("Specify a publisher or explicit publication limits, never both")
    if publisher is not None:
        publication_limits = publisher.limits
    elif publication_limits is None:
        publication_limits = GenerationPublicationLimits(
            max_files=MAX_GENERATION_FILES,
            max_file_bytes=MAX_GENERATION_FILE_BYTES,
            max_total_bytes=MAX_GENERATION_BYTES,
            max_manifest_bytes=MAX_MANIFEST_BYTES,
        )
    if type(publication_limits) is not GenerationPublicationLimits:
        raise TypeError("Publication admission requires exact GenerationPublicationLimits")
    for value, expected in (
        (request.inference, LuxDepthV5Request),
        (request.grade, GradeRecipe),
        (request.render, RenderRecipe),
        (request.depth_maps, DepthMapRecipe),
    ):
        if type(value) is not expected:
            raise TypeError("Managed V6 requires exact typed recipes and inference request")
    if request.inference.materials_manifest is not None or request.inference.materials_policy is not None:
        raise ValueError("Managed V6 cannot replay applied Materials responses")
    if request.inference.max_pixels > 100_000_000:
        raise ValueError("Managed V6 supports at most 100 million pixels per image")
    total = min(request.inference.max_output_bytes, publication_limits.max_total_bytes)
    budget = stage_output_budget(total)
    output = directory_path(request.inference.output_dir, allow_missing=True)
    inference = prepare_inference(
        replace(request.inference, output_dir=output / SOURCE_DIRECTORY, max_output_bytes=budget),
        publication_limits=publication_limits,
    )
    payload = {
        "schema": "tp.execution.plan.v5",
        "canonicalization": "tp.canonical.json.v1",
        "pipeline": "lux_depth_v6",
        "inference": inference.plan.to_payload(),
        "finishing": {
            "grade": request.grade.to_payload(),
            "render": request.render.to_payload(),
            "depth_maps": request.depth_maps.to_payload(),
        },
        "processing": managed_processing_identity(),
        "resources": {**inference.plan.to_payload()["resources"], "max_output_bytes": total},
        "publication": publication_limits.to_payload(),
    }
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    plan = ExecutionPlanV5.from_payload(payload)
    validate_publication_plan(plan.to_payload(), publication_limits)
    return PreparedManagedLuxExecutionV6(plan, inference)


def _finishing_request(payload: dict[str, Any], root: Path) -> LuxDepthV6Request:
    resources = payload["resources"]
    budget = stage_output_budget(resources["max_output_bytes"])
    recipes = payload["finishing"]
    return LuxDepthV6Request(
        root / SOURCE_DIRECTORY,
        root / DELIVERY_DIRECTORY,
        GradeRecipe.from_payload(recipes["grade"]),
        RenderRecipe.from_payload(recipes["render"]),
        SourceLimits(max_input_bytes=budget, max_pixels=resources["max_pixels"], memory_mib=resources["memory_mib"]),
        OutputLimits(max_output_bytes=budget, wall_time_seconds=resources["wall_time_seconds"]),
        DepthMapRecipe.from_payload(recipes["depth_maps"]),
    )


def _stage_records(
    root: Path,
    plan: ExecutionPlanV5,
    *,
    checkpoint: Callable[[], None],
    verified_finishing: VerifiedGradeEvidence | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Bind both completed stage inventories to the admitted composite recipes."""
    payload = plan.to_payload()
    request = _finishing_request(payload, root)
    source, delivery = root / SOURCE_DIRECTORY, root / DELIVERY_DIRECTORY
    observed_plan, plan_record = snapshot(root, root / "execution-plan.json", maximum_bytes=MAX_PLAN_BYTES)
    if observed_plan != plan.canonical_bytes or payload["processing"] != managed_processing_identity():
        raise ValueError("Managed V6 plan or processing identity changed")
    upstream_bytes, _ = snapshot(source, source / "execution-plan.json", maximum_bytes=MAX_PLAN_BYTES)
    if upstream_bytes != plan.inference_plan.canonical_bytes:
        raise ValueError("Managed V6 inference changed from its admitted plan")
    grade_bytes, _ = snapshot(delivery, delivery / "plan.json", maximum_bytes=16 * 1024**2)
    grade_plan = GradePlan(grade_bytes)
    grade_payload = grade_plan.to_payload()
    if (
        any(grade_payload.get(key) != value for key, value in payload["finishing"].items())
        or grade_payload["source"]["limits"] != request.source_limits.to_payload()
        or grade_payload["limits"] != request.output_limits.to_payload()
        or grade_payload["source"]["plan_sha256"] != digest(upstream_bytes)
    ):
        raise ValueError("Managed V6 finishing differs from the frozen composite policy")
    checkpoint()
    if verified_finishing is None:
        # Exact frozen source limits were checked above. Passing them again
        # would cause an unnecessary second full retained-parent verification.
        verified_finishing = verify_execution_evidence(
            delivery,
            source_root=source,
            expected_plan_sha256=grade_plan.sha256,
            cancellation=lambda: _cancelled(checkpoint),
        )
    source_bytes, source_completion = snapshot(source, source / "execution-evidence.json", maximum_bytes=MAX_COMPLETION_BYTES)
    if digest(source_bytes) != grade_payload["source"]["evidence_sha256"]:
        raise ValueError("Managed V6 retained parent completion changed")
    delivery_bytes, delivery_completion = snapshot(delivery, delivery / "evidence.json", maximum_bytes=MAX_COMPLETION_BYTES)
    if (
        verified_finishing.output_root != delivery
        or verified_finishing.plan_sha256 != grade_plan.sha256
        or delivery_bytes != verified_finishing.canonical_bytes
    ):
        raise ValueError("Managed V6 finishing completion changed after verified replay")
    source_evidence = decode_bounded_json_object(source_bytes)
    grade_evidence = decode_bounded_json_object(delivery_bytes)
    if grade_evidence["plan_sha256"] != grade_plan.sha256:
        raise ValueError("Managed V6 finishing completion changed")
    records = [plan_record]
    for prefix, stage_root, stage_records, completion in (
        (SOURCE_DIRECTORY, source, source_evidence["artifacts"], source_completion),
        (DELIVERY_DIRECTORY, delivery, grade_evidence["artifacts"], delivery_completion),
    ):
        for record in [*stage_records, completion]:
            checkpoint()
            # The parent V5 inventory includes additional role/input metadata.
            expected = {key: record[key] for key in ("path", "size_bytes", "sha256")}
            _, observed = snapshot(
                stage_root, stage_root / expected["path"], maximum_bytes=expected["size_bytes"], retain_bytes=False
            )
            if observed != expected:
                raise ValueError("Managed V6 stage changed after completion verification")
            records.append({**expected, "path": f"{prefix}/{expected['path']}"})
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("Managed V6 stage inventory contains duplicates")
    if payload["processing"] != managed_processing_identity():
        raise ValueError("Managed V6 processing source changed during verification")
    return sorted(records, key=lambda record: record["path"]), grade_plan.sha256


def _cancelled(checkpoint: Callable[[], None]) -> bool:
    checkpoint()
    return False


def run(
    prepared: PreparedManagedLuxExecutionV6,
    *,
    cancellation: Callable[[], bool] | None = None,
    publication_limits: GenerationPublicationLimits | None = None,
    managed_process_group: bool = False,
) -> VerifiedManagedV6Evidence:
    """Run one inference, then all V6 finishing; never publish partial success."""
    from transformation_portal.lux_depth_v5.pipeline import run as run_inference

    from .pipeline import run as run_finishing
    from .plan import prepare as prepare_finishing
    from .publication import validate_publication_plan

    if type(prepared) is not PreparedManagedLuxExecutionV6 or type(prepared.plan) is not ExecutionPlanV5:
        raise TypeError("Managed V6 execution requires exact prepared authority")
    plan = ExecutionPlanV5(prepared.canonical_plan_bytes)
    payload = plan.to_payload()
    limits = GenerationPublicationLimits.from_payload(payload["publication"])
    if publication_limits is not None and limits != publication_limits:
        raise ValueError("Managed V6 publisher policy changed")
    validate_publication_plan(payload, limits)
    if prepared.inference.canonical_plan_bytes != plan.inference_plan.canonical_bytes:
        raise ValueError("Managed V6 inference differs from frozen authority")
    if payload["processing"] != managed_processing_identity():
        raise ValueError("Managed V6 processing identity changed before inference")
    validate_prepared_bindings(prepared.inference)
    root = directory_path(prepared.output_root, allow_missing=True)
    if prepared.inference.output_root != root / SOURCE_DIRECTORY or root.exists() or not root.parent.is_dir():
        raise ValueError("Managed V6 requires a new canonical composite output")
    if root.is_relative_to(prepared.inference.input_root) or prepared.inference.input_root.is_relative_to(root):
        raise ValueError("Managed V6 source/output roots must be disjoint")
    started = time.monotonic()

    def check() -> None:
        if cancellation is not None and cancellation():
            raise RuntimeError("Managed V6 execution cancelled")
        if time.monotonic() - started > payload["resources"]["wall_time_seconds"]:
            raise RuntimeError("Managed V6 execution exceeded its total wall-time budget")

    check()
    with _pin_output_root(root.parent) as parent:
        os.mkdir(root.name, mode=0o700, dir_fd=parent.descriptor)
        _validate_pinned_root_namespace(parent)
    with pinned_directory(root):
        write_evidence(root, "execution-plan.json", plan.canonical_bytes, maximum_bytes=MAX_PLAN_BYTES)
        print("[INFO] LuxDepthV6: governed depth inference", flush=True)
        run_inference(
            prepared.inference,
            cancellation=lambda: _cancelled(check),
            publication_limits=limits,
            managed_process_group=managed_process_group,
        )
        check()
        print("[INFO] LuxDepthV6: reconstructing depth, grading, and rendering photographic TIFF/PNG", flush=True)
        finishing = prepare_finishing(_finishing_request(payload, root), cancellation=lambda: _cancelled(check))
        finished = run_finishing(finishing, cancellation=lambda: _cancelled(check))
        # Both stage runners already verified their products. Snapshot their
        # exact inventories here; publication performs independent semantic replay.
        verified_finishing = VerifiedGradeEvidence(
            finished.output_root, finished.canonical_evidence_bytes, finished.plan_sha256
        )
        records, grade_digest = _stage_records(root, plan, checkpoint=check, verified_finishing=verified_finishing)
        if _inventory(root) != {record["path"] for record in records}:
            raise ValueError("Managed V6 output contains undeclared artifacts")
        evidence = canonicalize_json(
            {
                "schema": COMPLETION_SCHEMA,
                "pipeline": "lux_depth_v6",
                "plan_sha256": digest(plan.canonical_bytes),
                "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
                "grade_plan_sha256": grade_digest,
                "input_count": len(payload["inference"]["inputs"]),
                "artifacts": records,
                "production_acceptance": "not_established",
            }
        )
        if sum(record["size_bytes"] for record in records) + len(evidence) > payload["resources"]["max_output_bytes"]:
            raise ValueError("Managed V6 composite output exceeds its total byte budget")
        check()
        write_evidence(root, "execution-evidence.json", evidence, maximum_bytes=MAX_COMPLETION_BYTES)
    completion = {"path": "execution-evidence.json", "size_bytes": len(evidence), "sha256": digest(evidence)}
    return VerifiedManagedV6Evidence(root, evidence, tuple(VerifiedArtifact(**record) for record in [*records, completion]))


def verify_managed_evidence(
    root: Path,
    *,
    expected_plan_bytes: bytes,
    cancellation: Callable[[], bool] | None = None,
) -> VerifiedManagedV6Evidence:
    """Replay source, depth, grade and render before accepting complete generation."""
    plan = ExecutionPlanV5(expected_plan_bytes)
    payload = plan.to_payload()
    root = directory_path(root)
    started = time.monotonic()

    def check() -> None:
        if cancellation is not None and cancellation():
            raise RuntimeError("Managed V6 verification cancelled")
        if time.monotonic() - started > payload["resources"]["wall_time_seconds"]:
            raise RuntimeError("Managed V6 verification exceeded its wall-time budget")

    with pinned_directory(root):
        raw, completion = snapshot(root, root / "execution-evidence.json", maximum_bytes=MAX_COMPLETION_BYTES)
        observed = decode_bounded_json_object(raw)
        records, grade_digest = _stage_records(root, plan, checkpoint=check)
        expected = {
            "schema": COMPLETION_SCHEMA,
            "pipeline": "lux_depth_v6",
            "plan_sha256": digest(expected_plan_bytes),
            "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
            "grade_plan_sha256": grade_digest,
            "input_count": len(payload["inference"]["inputs"]),
            "artifacts": records,
            "production_acceptance": "not_established",
        }
        if observed != expected or raw != canonicalize_json(expected):
            raise ValueError("Managed V6 completion differs from verified replay")
        if _inventory(root) != {record["path"] for record in [*records, completion]}:
            raise ValueError("Managed V6 complete namespace contains undeclared artifacts")
        if sum(record["size_bytes"] for record in records) + len(raw) > payload["resources"]["max_output_bytes"]:
            raise ValueError("Managed V6 composite exceeds its total output budget")
        _, after = snapshot(root, root / "execution-evidence.json", maximum_bytes=len(raw), retain_bytes=False)
        if after != completion:
            raise ValueError("Managed V6 completion changed during replay")
    return VerifiedManagedV6Evidence(root, raw, tuple(VerifiedArtifact(**record) for record in [*records, completion]))
