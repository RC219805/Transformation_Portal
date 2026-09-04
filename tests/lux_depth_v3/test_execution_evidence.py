"""Prepared-plan runtime evidence and artifact-accounting contracts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from transformation_portal.ingest.canonical_json import canonicalize_json, dumps_json
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_evidence import (
    EXECUTION_EVIDENCE_SCHEMA,
    ArtifactEvidenceError,
    ArtifactObservation,
    ConfinedArtifactCopyBudget,
    ExecutionEvidenceError,
    InputExecution,
    build_execution_evidence,
    build_manifest_outcome_projection,
    build_manifest_plan_projection,
    compute_execution_evidence_fingerprint,
    copy_confined_artifact,
    load_execution_evidence_schema,
    require_required_artifacts,
    validate_execution_evidence_payload,
    verify_execution_evidence_file,
    write_execution_evidence,
)
from transformation_portal.lux_depth_v3.execution_lifecycle import PreparedLuxExecution, prepare_lux_execution
from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from transformation_portal.lux_depth_v3.manifest import (
    BackendSelectionMetadata,
    BatchManifest,
    CombinedManifest,
    DepthMetadata,
    InputMetadata,
    MaterialsV3Metadata,
    V2Metadata,
)
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

pytestmark = [pytest.mark.unit]


def _prepared(tmp_path: Path, **overrides: Any) -> tuple[PreparedLuxExecution, Path]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    image_path = input_dir / "sample.png"
    Image.new("RGB", (4, 4), color=(127, 127, 127)).save(image_path)
    config_values: dict[str, Any] = {
        "depth_backend": "synthetic",
        "enable_v2": False,
        "emit_run_card": False,
    }
    config_values.update(overrides)
    prepared = prepare_lux_execution(
        EnhanceConfig(**config_values),
        input_root=input_dir,
        input_files=[image_path],
    )
    return prepared, image_path


def _prepared_many(tmp_path: Path, count: int = 2, **overrides: Any) -> PreparedLuxExecution:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    image_paths: list[Path] = []
    for index in range(count):
        image_path = input_dir / f"sample_{index}.png"
        Image.new("RGB", (4, 4), color=(index, index, index)).save(image_path)
        image_paths.append(image_path)
    config_values: dict[str, Any] = {
        "depth_backend": "synthetic",
        "enable_v2": False,
        "emit_run_card": False,
    }
    config_values.update(overrides)
    return prepare_lux_execution(
        EnhanceConfig(**config_values),
        input_root=input_dir,
        input_files=image_paths,
    )


def _runtime_licensing(
    prepared: PreparedLuxExecution,
    *backend_ids: str,
) -> dict[str, Any]:
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.config = prepared.runtime_config
    return orchestrator._build_runtime_licensing_evidence(
        model_contract=None,
        backend_selection=None,
        backend_ids=backend_ids,
    )


def _required_observations(
    prepared: PreparedLuxExecution,
    output_root: Path,
) -> list[ArtifactObservation]:
    evidence_path = "manifests/execution_evidence_test.json"
    execution_rows = [
        InputExecution(
            input_id=plan_input.input_id,
            status="ok",
            executed_backend=prepared.plan.planned_backend,
        )
        for plan_input in prepared.plan.inputs
    ]
    observations: list[ArtifactObservation] = []
    batch_results: list[dict[str, Any]] = []
    for index, (plan_input, execution_row) in enumerate(zip(prepared.plan.inputs, execution_rows)):
        candidate = next(
            item for item in prepared.plan.backend_candidates if item.backend_id == execution_row.executed_backend
        )
        model_contracts = tuple(item for item in candidate.model_contracts if item.enabled)
        model_id = {
            "ensemble": "ensemble/multi-backend",
            "synthetic": "synthetic/depth-analytic-v1",
        }.get(candidate.backend_id)
        if len(model_contracts) == 1:
            model_id = model_contracts[0].model.repo_id or model_contracts[0].model.canonical_key
        assert model_id is not None
        attempt: dict[str, Any] = {
            "backend": candidate.backend_id,
            "status": "success",
            "model_id": model_id,
            "device": model_contracts[0].device if len(model_contracts) == 1 else "cpu",
        }
        if len(model_contracts) == 1 and model_contracts[0].artifact_path is not None:
            attempt["model_artifact_filename"] = Path(model_contracts[0].artifact_path).name
            attempt["model_artifact_sha256"] = model_contracts[0].artifact_sha256
        licensing = _runtime_licensing(prepared, candidate.backend_id)
        stem = Path(plan_input.path).stem
        depth_path = output_root / f"depth/{stem}_depth.png"
        depth_metadata_path = output_root / f"depth/{stem}_depth_metadata.json"
        combined_path = output_root / f"manifests/{stem}_combined.json"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.write_bytes(f"artifact-{index}-depth".encode("ascii"))
        depth_metadata_path.write_bytes(f"artifact-{index}-metadata".encode("ascii"))
        combined_projection = build_manifest_plan_projection(
            prepared.plan,
            input_executions=[execution_row],
            evidence_path=evidence_path,
        )
        combined_contract = {
            "authoritative_plan": prepared.plan.to_payload(),
            "runtime": combined_projection,
            "execution_evidence_path": evidence_path,
        }
        CombinedManifest(
            input=InputMetadata(
                image_path=str(Path(prepared.plan.input_root) / plan_input.path),
            ),
            depth=DepthMetadata(
                model=candidate.backend_id,
                depth_path=str(depth_path),
                runtime_seconds=0.01,
                scaling={},
            ),
            backend_selection=BackendSelectionMetadata(
                requested_backend=prepared.plan.planned_backend,
                resolved_backend=candidate.backend_id,
                resolution_status="success",
                resolution_reason=None,
                model_id=model_id,
                device=model_contracts[0].device if len(model_contracts) == 1 else "cpu",
                attempts=[attempt],
            ),
            environment={"execution_contract": combined_contract},
            licensing=licensing,
        ).save(combined_path)
        observations.extend(
            (
                ArtifactObservation("depth_u16_png", depth_path, plan_input.input_id),
                ArtifactObservation("depth_metadata_json", depth_metadata_path, plan_input.input_id),
                ArtifactObservation("combined_manifest_json", combined_path, plan_input.input_id),
            )
        )
        batch_results.append(
            {
                "status": "ok",
                "image": plan_input.path,
                "backend": candidate.backend_id,
                "depth_path": depth_path.relative_to(output_root).as_posix(),
                "manifest": combined_path.relative_to(output_root).as_posix(),
            }
        )

    batch_path = output_root / "manifests/batch_test.json"
    batch_projection = build_manifest_plan_projection(
        prepared.plan,
        input_executions=execution_rows,
        evidence_path=evidence_path,
    )
    BatchManifest(
        batch_id="test",
        start_time="2026-09-02T00:00:00Z",
        end_time="2026-09-02T00:00:01Z",
        config={
            "execution_contract": {
                "authoritative_plan": prepared.plan.to_payload(),
                "runtime": batch_projection,
                "execution_evidence_path": evidence_path,
            }
        },
        results=batch_results,
        stats={},
    ).write(batch_path)
    observations.append(ArtifactObservation("batch_manifest_json", batch_path, None))
    return observations


def _successful_input(prepared: PreparedLuxExecution) -> list[InputExecution]:
    return [
        InputExecution(
            input_id=plan_input.input_id,
            status="ok",
            executed_backend=prepared.plan.planned_backend,
        )
        for plan_input in prepared.plan.inputs
    ]


def _rewrite_observation_carrier_outcomes(
    prepared: PreparedLuxExecution,
    observations: list[ArtifactObservation],
    evidence: dict[str, Any],
    *,
    evidence_path: str = "manifests/execution_evidence_test.json",
) -> None:
    """Apply a preliminary detached outcome projection to fixture carriers."""

    combined_observations = [item for item in observations if item.artifact_kind == "combined_manifest_json"]
    assert len(combined_observations) == len(prepared.plan.inputs)
    for plan_input, observation in zip(prepared.plan.inputs, combined_observations):
        assert observation.path is not None
        combined = json.loads(observation.path.read_bytes())
        combined["environment"]["execution_contract"].update(
            build_manifest_outcome_projection(
                evidence,
                evidence_path=evidence_path,
                input_id=plan_input.input_id,
            )
        )
        observation.path.write_text(
            dumps_json(combined, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    batch_observation = next(item for item in observations if item.artifact_kind == "batch_manifest_json")
    assert batch_observation.path is not None
    batch = json.loads(batch_observation.path.read_bytes())
    batch["config"]["execution_contract"].update(build_manifest_outcome_projection(evidence, evidence_path=evidence_path))
    batch_observation.path.write_text(
        dumps_json(batch, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _observation_paths(observations: list[ArtifactObservation]) -> dict[str, Path]:
    return {item.artifact_kind: item.path for item in observations if item.path is not None}


def _completed_prepared_run_with_card(
    tmp_path: Path,
    **overrides: Any,
) -> tuple[PreparedLuxExecution, EnhanceOrchestrator, list[dict[str, Any]], Path, Path, Path, Path]:
    prepared, image_path = _prepared(tmp_path, emit_run_card=True, **overrides)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    batch_path = next((output_root / "manifests").glob("batch_*.json"))
    run_card_path = next(path for path in output_root.glob("run_card_*.json") if not path.name.endswith(".self.json"))
    return prepared, orchestrator, results, output_root, evidence_path, batch_path, run_card_path


def test_prepared_batch_result_carries_snapshot_identity_and_dimensions(tmp_path: Path) -> None:
    prepared, image_path = _prepared(tmp_path)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")

    result = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])[0]

    assert result["input_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert result["original_shape"] == [4, 4]
    assert result["enforced_shape"] == [14, 14]
    combined = CombinedManifest.load(Path(result["manifest"]))
    assert combined.timing is not None
    assert combined.timing.total_seconds == pytest.approx(result["runtime_s"])


def test_prepared_rerun_preserves_prior_completion_carriers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later latest-manifest projection cannot invalidate an earlier run."""

    from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity

    prepared, image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)

    first_results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    stable_manifest_path = Path(first_results[0]["manifest"])
    first_evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    first_evidence = verify_execution_evidence_file(
        first_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    require_required_artifacts(first_evidence)
    first_combined_record = next(
        outcome["artifacts"][0]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "combined_manifest_json"
    )
    first_carrier_path = output_root / first_combined_record["path"]
    first_carrier_bytes = first_carrier_path.read_bytes()
    first_batch_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "batch_manifest_json"
    )
    first_run_card_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "run_card"
    )

    first_carrier_relative = first_carrier_path.relative_to(output_root).as_posix()
    assert first_carrier_relative.startswith("manifests/execution/")
    assert stable_manifest_path != first_carrier_path
    assert stable_manifest_path.read_bytes() == first_carrier_bytes
    assert json.loads(first_batch_path.read_bytes())["results"][0]["manifest"] == first_carrier_relative
    first_run_card = json.loads(first_run_card_path.read_bytes())
    assert first_run_card["result_summary"][0]["manifest_path"] == first_carrier_relative
    assert first_carrier_relative in {entry["relative_path"] for entry in first_run_card["artifact_index"]}
    assert verify_run_card_integrity(first_run_card_path) == []

    depth_backend = orchestrator.depth_backend
    assert depth_backend is not None

    def reject_recomputed_depth(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("prepared rerun recomputed depth instead of using verified evidence")

    monkeypatch.setattr(depth_backend, "compute", reject_recomputed_depth)
    second_results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    evidence_paths = set((output_root / "manifests").glob("execution_evidence_*.json"))
    second_evidence_path = next(path for path in evidence_paths if path != first_evidence_path)
    second_evidence = verify_execution_evidence_file(
        second_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    require_required_artifacts(second_evidence)
    second_combined_record = next(
        outcome["artifacts"][0]
        for outcome in second_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "combined_manifest_json"
    )
    second_carrier_path = output_root / second_combined_record["path"]
    second_run_card_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in second_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "run_card"
    )

    assert Path(second_results[0]["manifest"]) == stable_manifest_path
    assert second_carrier_path != first_carrier_path
    assert first_carrier_path.read_bytes() == first_carrier_bytes
    assert stable_manifest_path.read_bytes() == second_carrier_path.read_bytes()
    verify_execution_evidence_file(first_evidence_path, output_root=output_root, plan=prepared.plan)
    assert verify_run_card_integrity(first_run_card_path) == []
    assert verify_run_card_integrity(second_run_card_path) == []


def test_prepared_forced_depth_rerun_preserves_prior_artifact_carriers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced mutable outputs receive new carriers without rewriting history."""

    from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity

    prepared, image_path = _prepared(tmp_path, emit_run_card=True, force_depth=True)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    depth_backend = orchestrator.depth_backend
    assert depth_backend is not None
    real_compute = depth_backend.compute
    compute_calls = 0

    def counted_compute(*args: Any, **kwargs: Any) -> Any:
        nonlocal compute_calls
        compute_calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(depth_backend, "compute", counted_compute)
    orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    first_evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    first_evidence = verify_execution_evidence_file(
        first_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    first_run_card_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "run_card"
    )
    first_records = {
        outcome["artifact_kind"]: tuple(record["path"] for record in outcome["artifacts"])
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] in {"depth_u16_png", "depth_metadata_json"}
    }
    assert set(first_records) == {"depth_u16_png", "depth_metadata_json"}
    assert all("/execution/" in path for paths in first_records.values() for path in paths)

    orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    second_evidence_path = next(
        path for path in (output_root / "manifests").glob("execution_evidence_*.json") if path != first_evidence_path
    )
    second_evidence = verify_execution_evidence_file(
        second_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    second_records = {
        outcome["artifact_kind"]: tuple(record["path"] for record in outcome["artifacts"])
        for outcome in second_evidence["produced_artifacts"]
        if outcome["artifact_kind"] in {"depth_u16_png", "depth_metadata_json"}
    }

    assert compute_calls == 2
    assert set(second_records) == set(first_records)
    assert all(set(second_records[kind]).isdisjoint(first_records[kind]) for kind in first_records)
    verify_execution_evidence_file(first_evidence_path, output_root=output_root, plan=prepared.plan)
    assert verify_run_card_integrity(first_run_card_path) == []


def test_prepared_forced_v2_rerun_preserves_prior_output_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced V2 writes cannot invalidate the prior completed batch."""

    from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity

    prepared, image_path = _prepared(
        tmp_path,
        emit_run_card=True,
        enable_v2=True,
        force_v2=True,
    )
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    v2_calls = 0

    def fake_v2_stage(**kwargs: Any) -> tuple[dict[str, Any], float, Path]:
        nonlocal v2_calls
        v2_calls += 1
        output_key = Path(kwargs["output_key"])
        output_path = output_root / "v2" / output_key.parent / f"{output_key.name}_v2_enhanced.png"
        report_path = output_root / "v2" / output_key.parent / f"{output_key.name}_v2_report.json"
        log_path = Path(kwargs["v2_log_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(v2_calls, v2_calls, v2_calls)).save(output_path)
        log_path.write_text(f"v2 run {v2_calls}\n", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(output_path),
                    "depth_map": str(kwargs["depth_path"]),
                    "depth_consumed": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return (
            {
                "status": "ok",
                "output": str(output_path),
                "output_paths": [str(output_path)],
                "report_path": str(report_path),
                "depth_map": str(kwargs["depth_path"]),
                "depth_consumed": True,
                "runtime_s": float(v2_calls),
            },
            float(v2_calls),
            report_path,
        )

    monkeypatch.setattr(orchestrator, "_run_v2_stage", fake_v2_stage)
    orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    first_evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    first_evidence = verify_execution_evidence_file(
        first_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    first_record = next(
        outcome["artifacts"][0]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "v2_enhanced_image"
    )
    first_output_path = output_root / first_record["path"]
    first_output_bytes = first_output_path.read_bytes()
    first_run_card_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "run_card"
    )
    first_run_card = json.loads(first_run_card_path.read_bytes())
    first_log_entries = [entry for entry in first_run_card["artifact_index"] if entry["artifact_type"] == "v2_log"]
    assert "/execution/" in first_record["path"]
    assert first_log_entries
    assert all("/execution/" in entry["relative_path"] for entry in first_log_entries)

    orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    second_evidence_path = next(
        path for path in (output_root / "manifests").glob("execution_evidence_*.json") if path != first_evidence_path
    )
    second_evidence = verify_execution_evidence_file(
        second_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    second_record = next(
        outcome["artifacts"][0]
        for outcome in second_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "v2_enhanced_image"
    )

    assert v2_calls == 2
    assert second_record["path"] != first_record["path"]
    assert first_output_path.read_bytes() == first_output_bytes
    verify_execution_evidence_file(first_evidence_path, output_root=output_root, plan=prepared.plan)
    assert verify_run_card_integrity(first_run_card_path) == []


def test_prepared_failed_rerun_restores_latest_manifest_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed final evidence write cannot publish an uncompleted latest view."""

    from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity

    prepared, image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    first_results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    stable_manifest_path = Path(first_results[0]["manifest"])
    stable_provenance_path = stable_manifest_path.with_name(f"{stable_manifest_path.stem}_provenance.json")
    stable_manifest_bytes = stable_manifest_path.read_bytes()
    stable_provenance_bytes = stable_provenance_path.read_bytes()
    first_evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    first_evidence = verify_execution_evidence_file(
        first_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    first_run_card_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "run_card"
    )
    latest_manifest_paths = dict(orchestrator._latest_prepared_combined_manifest_paths)
    latest_artifact_paths = dict(orchestrator._latest_prepared_volatile_artifact_paths)

    def reject_final_evidence(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected final evidence failure")

    monkeypatch.setattr(orchestrator, "_emit_prepared_execution_evidence", reject_final_evidence)
    with pytest.raises(RuntimeError, match="injected final evidence failure"):
        orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert stable_manifest_path.read_bytes() == stable_manifest_bytes
    assert stable_provenance_path.read_bytes() == stable_provenance_bytes
    assert orchestrator._latest_prepared_combined_manifest_paths == latest_manifest_paths
    assert orchestrator._latest_prepared_volatile_artifact_paths == latest_artifact_paths
    verify_execution_evidence_file(first_evidence_path, output_root=output_root, plan=prepared.plan)
    assert verify_run_card_integrity(first_run_card_path) == []


def test_prepared_completion_rejects_carrier_changed_after_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final evidence stays bound to the exact bytes copied from volatile output."""

    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    real_activate = orchestrator._activate_volatile_artifact_carriers
    changed_path: Path | None = None

    def activate_then_change(results: Any, *, batch_id: str) -> None:
        nonlocal changed_path
        real_activate(results, batch_id=batch_id)
        changed_path = next(
            carrier
            for source, carrier in orchestrator._active_prepared_volatile_artifact_paths.items()
            if source.endswith("_depth.png")
        )
        changed_path.write_bytes(b"changed-after-secure-copy")

    monkeypatch.setattr(orchestrator, "_activate_volatile_artifact_carriers", activate_then_change)
    with pytest.raises(ExecutionEvidenceError, match="artifact_changed"):
        orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert changed_path is not None
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    evidence = json.loads(evidence_path.read_bytes())
    depth_failure = next(outcome for outcome in evidence["failed_artifacts"] if outcome["artifact_kind"] == "depth_u16_png")
    assert depth_failure["reason_code"] == "artifact_changed"


def test_prepared_completion_rejects_changed_run_card_only_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed carrier is fatal even when only the optional run card indexes it."""

    prepared, image_path = _prepared(
        tmp_path,
        emit_run_card=True,
        enable_v2=True,
        force_v2=True,
    )
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)

    def fake_v2_stage(**kwargs: Any) -> tuple[dict[str, Any], float, Path]:
        output_key = Path(kwargs["output_key"])
        output_path = output_root / "v2" / output_key.parent / f"{output_key.name}_v2_enhanced.png"
        report_path = output_root / "v2" / output_key.parent / f"{output_key.name}_v2_report.json"
        log_path = Path(kwargs["v2_log_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(1, 2, 3)).save(output_path)
        log_path.write_text("original V2 log\n", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(output_path),
                    "depth_map": str(kwargs["depth_path"]),
                    "depth_consumed": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return (
            {
                "status": "ok",
                "output": str(output_path),
                "output_paths": [str(output_path)],
                "report_path": str(report_path),
                "depth_map": str(kwargs["depth_path"]),
                "depth_consumed": True,
                "runtime_s": 0.1,
            },
            0.1,
            report_path,
        )

    monkeypatch.setattr(orchestrator, "_run_v2_stage", fake_v2_stage)
    real_activate = orchestrator._activate_volatile_artifact_carriers
    changed_path: Path | None = None

    def activate_then_change(results: Any, *, batch_id: str) -> None:
        nonlocal changed_path
        real_activate(results, batch_id=batch_id)
        changed_path = next(
            carrier
            for source, carrier in orchestrator._active_prepared_volatile_artifact_paths.items()
            if source.endswith(".log")
        )
        changed_path.write_text("changed after secure copy\n", encoding="utf-8")

    monkeypatch.setattr(orchestrator, "_activate_volatile_artifact_carriers", activate_then_change)
    with pytest.raises(ExecutionEvidenceError, match="artifact_changed"):
        orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert changed_path is not None
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    evidence = json.loads(evidence_path.read_bytes())
    run_card_failure = next(outcome for outcome in evidence["failed_artifacts"] if outcome["artifact_kind"] == "run_card")
    assert run_card_failure["required"] is False
    assert run_card_failure["reason_code"] == "artifact_changed"


def test_prepared_latest_projection_rolls_back_post_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visible replacement bytes are rolled back when durability fails."""

    from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module

    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    first_results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    stable_manifest_path = Path(first_results[0]["manifest"])
    stable_manifest_bytes = stable_manifest_path.read_bytes()
    first_evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    real_atomic_write_bytes = orchestrator_module.atomic_write_bytes
    injected = False

    def fail_after_visible_replace(path: Path, data: bytes) -> None:
        nonlocal injected
        real_atomic_write_bytes(path, data)
        if not injected and Path(path) == stable_manifest_path and data != stable_manifest_bytes:
            injected = True
            raise OSError(errno.EIO, "injected post-rename durability failure")

    monkeypatch.setattr(orchestrator_module, "atomic_write_bytes", fail_after_visible_replace)
    with pytest.raises(OSError, match="injected post-rename durability failure"):
        orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert injected is True
    assert stable_manifest_path.read_bytes() == stable_manifest_bytes
    verify_execution_evidence_file(first_evidence_path, output_root=output_root, plan=prepared.plan)


def test_prepared_rerun_preserves_prior_vlm_advisory_carriers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Volatile captioning metadata must not corrupt an earlier completion."""

    from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module
    from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity
    from transformation_portal.vlm_captioning.fastvlm_runtime import FastVLMRuntimeResult
    from transformation_portal.vlm_captioning.parser import parse_fastvlm_caption

    prepared, image_path = _prepared(
        tmp_path,
        emit_run_card=True,
        vlm_captioning_enabled=True,
    )
    output_root = tmp_path / "output"
    call_count = 0

    def caption(*_args: Any, **_kwargs: Any) -> FastVLMRuntimeResult:
        nonlocal call_count
        call_count += 1
        raw_text = (
            f"SCENE=Pool {call_count}; MATERIALS=stone; FEATURES=steps; NATURAL=sky; "
            "LIGHTING=daylight; ISSUES=none; UNCERTAIN=none."
        )
        return FastVLMRuntimeResult(
            success=True,
            status="ok",
            caption_parse=parse_fastvlm_caption(raw_text),
            raw_stdout=raw_text,
            raw_stderr="",
            returncode=0,
            command=["fake-fastvlm", str(call_count)],
            runtime_seconds=float(call_count),
        )

    monkeypatch.setattr(orchestrator_module, "run_fastvlm_caption", caption)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    first_results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    first_evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    first_evidence = verify_execution_evidence_file(
        first_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    first_run_card_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "run_card"
    )
    first_run_card = json.loads(first_run_card_path.read_bytes())
    first_vlm_entries = [
        entry for entry in first_run_card["artifact_index"] if str(entry.get("artifact_type", "")).startswith("vlm_caption")
    ]
    assert {entry["artifact_type"] for entry in first_vlm_entries} == {
        "vlm_caption_proxy",
        "vlm_caption_raw",
        "vlm_caption_sidecar",
    }
    assert all("/execution/" in entry["relative_path"] for entry in first_vlm_entries)
    first_carrier_bytes = {
        entry["relative_path"]: (output_root / entry["relative_path"]).read_bytes() for entry in first_vlm_entries
    }
    sidecar_entry = next(entry for entry in first_vlm_entries if entry["artifact_type"] == "vlm_caption_sidecar")
    sidecar_payload = json.loads((output_root / sidecar_entry["relative_path"]).read_bytes())
    assert sidecar_payload["vlm_captioning"]["image_proxy"]["source_path"] == str(image_path)

    orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert call_count == 2
    for relative_path, expected_bytes in first_carrier_bytes.items():
        assert (output_root / relative_path).read_bytes() == expected_bytes
    verify_execution_evidence_file(first_evidence_path, output_root=output_root, plan=prepared.plan)
    assert verify_run_card_integrity(first_run_card_path) == []
    assert Path(first_results[0]["vlm_caption_sidecar_path"]).exists()


def test_prepared_vlm_without_run_card_completes_without_unbound_auxiliary_carriers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module
    from transformation_portal.vlm_captioning.fastvlm_runtime import FastVLMRuntimeResult
    from transformation_portal.vlm_captioning.parser import parse_fastvlm_caption

    prepared, image_path = _prepared(
        tmp_path,
        emit_run_card=False,
        vlm_captioning_enabled=True,
    )
    output_root = tmp_path / "output"
    raw_text = "; ".join(
        (
            "SCENE=Pool",
            "MATERIALS=stone",
            "FEATURES=steps",
            "NATURAL=sky",
            "LIGHTING=daylight",
            "ISSUES=none",
            "UNCERTAIN=none.",
        )
    )

    def caption(*_args: Any, **_kwargs: Any) -> FastVLMRuntimeResult:
        return FastVLMRuntimeResult(
            success=True,
            status="ok",
            caption_parse=parse_fastvlm_caption(raw_text),
            raw_stdout=raw_text,
            raw_stderr="",
            returncode=0,
            command=["fake-fastvlm"],
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(orchestrator_module, "run_fastvlm_caption", caption)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert results[0]["status"] == "ok"
    for key in ("vlm_caption_proxy_path", "vlm_caption_sidecar_path", "vlm_caption_raw_path"):
        auxiliary_path = Path(results[0][key])
        assert auxiliary_path.exists()
        assert "execution" not in auxiliary_path.parts
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    evidence = verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)
    require_required_artifacts(evidence)
    assert not any(outcome["artifact_kind"] == "run_card" for outcome in evidence["produced_artifacts"])
    assert not list((output_root / "captioning").rglob("execution/*"))


def test_prepared_completion_prioritizes_changed_carrier_over_later_index_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later compatibility-index race cannot mask changed carrier bytes."""

    from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module
    from transformation_portal.vlm_captioning.fastvlm_runtime import FastVLMRuntimeResult
    from transformation_portal.vlm_captioning.parser import parse_fastvlm_caption

    prepared, image_path = _prepared(
        tmp_path,
        emit_run_card=True,
        vlm_captioning_enabled=True,
    )
    output_root = tmp_path / "output"
    raw_text = "; ".join(
        (
            "SCENE=Pool",
            "MATERIALS=stone",
            "FEATURES=steps",
            "NATURAL=sky",
            "LIGHTING=daylight",
            "ISSUES=none",
            "UNCERTAIN=none.",
        )
    )

    def caption(*_args: Any, **_kwargs: Any) -> FastVLMRuntimeResult:
        return FastVLMRuntimeResult(
            success=True,
            status="ok",
            caption_parse=parse_fastvlm_caption(raw_text),
            raw_stdout=raw_text,
            raw_stderr="",
            returncode=0,
            command=["fake-fastvlm"],
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(orchestrator_module, "run_fastvlm_caption", caption)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    real_emit = orchestrator._emit_run_card
    emit_count = 0

    def emit_then_race(*args: Any, **kwargs: Any) -> Path | None:
        nonlocal emit_count
        run_card_path = real_emit(*args, **kwargs)
        emit_count += 1
        combined_path = next(iter(orchestrator._active_prepared_combined_manifest_paths.values()))
        provenance_path = combined_path.with_name(f"{combined_path.stem}_provenance.json")
        if emit_count == 1:
            raw_carrier = next(
                carrier
                for source, carrier in orchestrator._active_prepared_volatile_artifact_paths.items()
                if source.endswith(".vlm_captioning.raw.txt")
            )
            raw_carrier.write_bytes(raw_carrier.read_bytes() + b"changed carrier\n")
        if emit_count <= 2:
            provenance_path.write_bytes(provenance_path.read_bytes() + b"later index race\n")
        return run_card_path

    monkeypatch.setattr(orchestrator, "_emit_run_card", emit_then_race)
    with pytest.raises(ExecutionEvidenceError, match="artifact_changed"):
        orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    evidence = json.loads(evidence_path.read_bytes())
    run_card_failure = next(outcome for outcome in evidence["failed_artifacts"] if outcome["artifact_kind"] == "run_card")
    assert run_card_failure["required"] is False
    assert run_card_failure["reason_code"] == "artifact_changed"
    with pytest.raises(ExecutionEvidenceError, match="artifact_changed"):
        require_required_artifacts(evidence)


@pytest.mark.parametrize(
    ("source_kind", "expected_code"),
    [
        pytest.param("symlink", "symlink_forbidden", id="symlink"),
        pytest.param("hardlink", "hardlink_forbidden", id="hardlink"),
        pytest.param("oversized", "artifact_too_large", id="bounded-read"),
    ],
)
def test_prepared_volatile_carrier_reads_are_confined_and_bounded(
    tmp_path: Path,
    source_kind: str,
    expected_code: str,
) -> None:
    prepared, _image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = tmp_path / "output"
    caption_dir = output_root / "captioning"
    caption_dir.mkdir(parents=True)
    source_path = caption_dir / "sample.vlm_captioning.raw.txt"
    outside_path = tmp_path / "outside.txt"
    outside_path.write_bytes(b"outside")
    if source_kind == "symlink":
        source_path.symlink_to(outside_path)
    elif source_kind == "hardlink":
        os.link(outside_path, source_path)
    else:
        with source_path.open("wb") as handle:
            handle.truncate(64 * 1024 * 1024 * 1024 + 1)

    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        orchestrator._activate_volatile_artifact_carriers(
            ({"vlm_caption_raw_path": str(source_path)},),
            batch_id="bounded-carrier-test",
        )

    assert exc_info.value.code == expected_code
    assert not (caption_dir / "execution" / "bounded-carrier-test" / source_path.name).exists()


def test_prepared_carrier_mapping_closes_alias_equivalent_json_paths(tmp_path: Path) -> None:
    prepared, _image_path = _prepared(tmp_path)
    output_root = tmp_path / "actual-output"
    output_root.mkdir()
    alias_root = tmp_path / "alias-output"
    alias_root.symlink_to(output_root, target_is_directory=True)
    aliased_source = alias_root / "masks" / "mask.npz"
    canonical_source = output_root / "masks" / "mask.npz"
    carrier_path = output_root / "masks" / "execution" / "batch" / "mask.npz"
    canonical_source.parent.mkdir()
    canonical_source.write_bytes(b"mask")
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    orchestrator._active_prepared_batch_token = object()
    source_key = orchestrator._prepared_volatile_artifact_key(aliased_source)
    assert source_key == "masks/mask.npz"
    orchestrator._active_prepared_volatile_artifact_paths[source_key] = carrier_path

    rewritten = orchestrator._rewrite_json_carrier_bytes(
        json.dumps({"segmentation_artifacts": [{"path": str(canonical_source)}]}).encode(),
        nested_path_entries=("segmentation_artifacts",),
    )

    assert json.loads(rewritten)["segmentation_artifacts"][0]["path"] == str(carrier_path)


def test_prepared_json_carrier_rejects_unmapped_configured_output(tmp_path: Path) -> None:
    prepared, _image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    orchestrator._active_prepared_batch_token = object()

    with pytest.raises(LuxExecutionPlanAuthorityError, match="without a frozen carrier"):
        orchestrator._rewrite_json_carrier_bytes(
            json.dumps({"output": str(output_root / "v2" / "mutable.png")}).encode(),
            scalar_keys=("output",),
        )


def test_prepared_carrier_activation_rolls_back_earlier_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    prepared, _image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = tmp_path / "output"
    caption_dir = output_root / "captioning"
    caption_dir.mkdir(parents=True)
    proxy_path = caption_dir / "sample.proxy.png"
    raw_path = caption_dir / "sample.raw.txt"
    proxy_path.write_bytes(b"proxy")
    raw_path.write_bytes(b"raw")
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    real_copy = orchestrator_module.copy_confined_artifact
    copy_calls = 0

    def fail_second_copy(*args: Any, **kwargs: Any) -> Any:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2:
            raise RuntimeError("injected second copy failure")
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "copy_confined_artifact", fail_second_copy)
    with pytest.raises(RuntimeError, match="second copy failure"):
        orchestrator._activate_volatile_artifact_carriers(
            (
                {
                    "vlm_caption_proxy_path": str(proxy_path),
                    "vlm_caption_raw_path": str(raw_path),
                },
            ),
            batch_id="rollback-copy-test",
        )

    assert copy_calls == 2
    assert not orchestrator._active_prepared_volatile_artifact_paths
    assert not orchestrator._active_prepared_volatile_artifact_records
    assert not [
        path
        for path in output_root.rglob("*")
        if path.is_file() and "execution" in path.parts and "rollback-copy-test" in path.parts
    ]


def test_prepared_carrier_activation_preserves_replacement_when_save_ownership_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    caption_dir = output_root / "captioning"
    caption_dir.mkdir(parents=True)
    raw_path = caption_dir / "sample.raw.txt"
    raw_path.write_bytes(b"raw")
    public_manifest_path = output_root / "manifests" / "sample_combined.json"
    carrier_manifest_path = output_root / "manifests" / "execution" / "rollback-save-test" / "sample_combined.json"
    carrier_manifest_path.parent.mkdir(parents=True)
    CombinedManifest(input=InputMetadata(image_path=str(image_path))).save(carrier_manifest_path)
    original_manifest_bytes = carrier_manifest_path.read_bytes()
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    orchestrator._active_prepared_batch_token = object()
    orchestrator._active_prepared_combined_manifest_paths[str(public_manifest_path)] = carrier_manifest_path
    real_save = CombinedManifest.save
    replacement_bytes = b'{"replacement":true}\n'

    def save_then_fail(manifest: CombinedManifest, path: Path) -> None:
        real_save(manifest, path)
        path.write_bytes(replacement_bytes)
        raise RuntimeError("injected manifest save failure")

    monkeypatch.setattr(CombinedManifest, "save", save_then_fail)
    with pytest.raises(LuxExecutionPlanAuthorityError, match="rollback was incomplete"):
        orchestrator._activate_volatile_artifact_carriers(
            (
                {
                    "manifest": str(public_manifest_path),
                    "vlm_caption_raw_path": str(raw_path),
                },
            ),
            batch_id="rollback-save-test",
        )

    assert carrier_manifest_path.read_bytes() == replacement_bytes
    assert carrier_manifest_path.read_bytes() != original_manifest_bytes
    assert not orchestrator._active_prepared_volatile_artifact_paths
    assert not orchestrator._active_prepared_volatile_artifact_records
    assert not (caption_dir / "execution" / "rollback-save-test" / raw_path.name).exists()


def test_prepared_carrier_activation_restores_owned_manifest_after_later_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    caption_dir = output_root / "captioning"
    caption_dir.mkdir(parents=True)
    raw_paths = (caption_dir / "first.raw.txt", caption_dir / "second.raw.txt")
    for raw_path in raw_paths:
        raw_path.write_bytes(raw_path.name.encode())
    public_manifest_paths = (
        output_root / "manifests" / "first_combined.json",
        output_root / "manifests" / "second_combined.json",
    )
    carrier_manifest_paths = tuple(
        output_root / "manifests" / "execution" / "rollback-owned-test" / path.name for path in public_manifest_paths
    )
    for carrier_manifest_path in carrier_manifest_paths:
        carrier_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        CombinedManifest(input=InputMetadata(image_path=str(image_path))).save(carrier_manifest_path)
        carrier_manifest_path.chmod(0o600)
    original_manifest_bytes = tuple(path.read_bytes() for path in carrier_manifest_paths)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    orchestrator._active_prepared_batch_token = object()
    for public_path, carrier_path in zip(public_manifest_paths, carrier_manifest_paths):
        orchestrator._active_prepared_combined_manifest_paths[str(public_path)] = carrier_path
    monkeypatch.setattr(
        orchestrator,
        "_rewrite_manifest_carrier_paths",
        lambda manifest: setattr(manifest, "environment", {"rewritten": True}),
    )
    real_save = CombinedManifest.save
    save_calls = 0

    def fail_second_save(manifest: CombinedManifest, path: Path) -> None:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise RuntimeError("injected later manifest save failure")
        real_save(manifest, path)

    monkeypatch.setattr(CombinedManifest, "save", fail_second_save)
    with pytest.raises(RuntimeError, match="later manifest save failure"):
        orchestrator._activate_volatile_artifact_carriers(
            tuple(
                {
                    "manifest": str(public_path),
                    "vlm_caption_raw_path": str(raw_path),
                }
                for public_path, raw_path in zip(public_manifest_paths, raw_paths)
            ),
            batch_id="rollback-owned-test",
        )

    assert save_calls == 2
    assert tuple(path.read_bytes() for path in carrier_manifest_paths) == original_manifest_bytes
    assert tuple(stat.S_IMODE(path.stat().st_mode) for path in carrier_manifest_paths) == (0o600, 0o600)
    assert not orchestrator._active_prepared_volatile_artifact_paths
    assert not orchestrator._active_prepared_volatile_artifact_records
    assert not [
        path
        for path in output_root.rglob("*")
        if path.is_file() and "captioning" in path.parts and "rollback-owned-test" in path.parts
    ]


def test_copy_confined_artifact_streams_to_distinct_nested_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    source_path = output_root / "depth" / "source.bin"
    destination_path = output_root / "depth" / "execution" / "batch" / "source.bin"
    source_path.parent.mkdir(parents=True)
    source_bytes = b"carrier-stream-is-not-a-bounded-snapshot"
    source_path.write_bytes(source_bytes)
    monkeypatch.setattr(evidence_module, "_MAX_EVIDENCE_BYTES", 1)

    copied = copy_confined_artifact(output_root, source_path, destination_path)

    assert destination_path.read_bytes() == source_bytes
    assert destination_path.stat().st_ino != source_path.stat().st_ino
    assert copied.source_relative_path == "depth/source.bin"
    assert copied.relative_path == "depth/execution/batch/source.bin"
    assert copied.sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert copied.size_bytes == len(source_bytes)
    assert copied.matches(
        {
            "path": copied.relative_path,
            "sha256": copied.sha256,
            "size_bytes": copied.size_bytes,
        }
    )


def test_copy_confined_artifact_rejects_existing_destination_and_same_path(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    source_path = output_root / "source.bin"
    destination_path = output_root / "destination.bin"
    output_root.mkdir()
    source_path.write_bytes(b"source")
    destination_path.write_bytes(b"existing")

    with pytest.raises(ArtifactEvidenceError) as existing_exc:
        copy_confined_artifact(output_root, source_path, destination_path)
    with pytest.raises(ArtifactEvidenceError) as same_exc:
        copy_confined_artifact(output_root, source_path, source_path)

    assert existing_exc.value.code == "artifact_changed"
    assert same_exc.value.code == "artifact_changed"
    assert destination_path.read_bytes() == b"existing"


def test_copy_confined_artifact_no_replace_race_preserves_competing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    source_path = output_root / "source.bin"
    destination_path = output_root / "execution" / "batch" / "source.bin"
    output_root.mkdir()
    source_path.write_bytes(b"source")
    competing_bytes = b"competing-publisher"
    real_link = evidence_module.os.link
    injected = False

    def raced_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        nonlocal injected
        assert follow_symlinks is False
        if not injected:
            injected = True
            destination_path.write_bytes(competing_bytes)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(evidence_module.os, "link", raced_link)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        copy_confined_artifact(output_root, source_path, destination_path)

    assert injected is True
    assert exc_info.value.code == "artifact_changed"
    assert destination_path.read_bytes() == competing_bytes
    assert not list(destination_path.parent.glob(".*.tmp"))


def test_copy_confined_artifact_enforces_shared_aggregate_budget(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    first_source = output_root / "first.bin"
    second_source = output_root / "second.bin"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    first_destination = output_root / "execution" / "batch" / "first.bin"
    second_destination = output_root / "execution" / "batch" / "second.bin"
    budget = ConfinedArtifactCopyBudget(max_bytes=10)

    copy_confined_artifact(output_root, first_source, first_destination, budget=budget)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        copy_confined_artifact(output_root, second_source, second_destination, budget=budget)

    assert exc_info.value.code == "aggregate_artifact_bytes_exceeded"
    assert budget.total_bytes == len(b"first")
    assert first_destination.read_bytes() == b"first"
    assert not second_destination.exists()


def test_copy_confined_artifact_budgets_transformed_destination_bytes(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    source_path = output_root / "source.json"
    destination_path = output_root / "execution" / "batch" / "source.json"
    source_path.write_bytes(b'{"larger":true}\n')
    transformed = b"{}\n"
    budget = ConfinedArtifactCopyBudget(max_bytes=len(transformed))

    copied = copy_confined_artifact(
        output_root,
        source_path,
        destination_path,
        budget=budget,
        transform_bytes=lambda _source: transformed,
    )

    assert destination_path.read_bytes() == transformed
    assert copied.size_bytes == len(transformed)
    assert budget.total_bytes == len(transformed)


def test_copy_confined_artifact_enforces_budget_when_source_grows_during_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    output_root.mkdir()
    source_path = output_root / "source.bin"
    destination_path = output_root / "execution" / "batch" / "source.bin"
    source_path.write_bytes(b"12345678")
    source_identity = (source_path.stat().st_dev, source_path.stat().st_ino)
    budget = ConfinedArtifactCopyBudget(max_bytes=9)
    real_read = evidence_module.os.read
    grew = False

    def growing_read(descriptor: int, size: int) -> bytes:
        nonlocal grew
        chunk = real_read(descriptor, size)
        descriptor_stat = os.fstat(descriptor)
        if not grew and (descriptor_stat.st_dev, descriptor_stat.st_ino) == source_identity and chunk:
            with source_path.open("ab") as handle:
                handle.write(b"90")
                handle.flush()
                os.fsync(handle.fileno())
            grew = True
        return chunk

    monkeypatch.setattr(evidence_module.os, "read", growing_read)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        copy_confined_artifact(
            output_root,
            source_path,
            destination_path,
            budget=budget,
        )

    assert grew is True
    assert exc_info.value.code == "aggregate_artifact_bytes_exceeded"
    assert budget.total_bytes == 0
    assert not destination_path.exists()


def test_copy_confined_artifact_transforms_before_destination_publication(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    source_path = output_root / "source.json"
    destination_path = output_root / "execution" / "batch" / "source.json"
    outside_alias = tmp_path / "outside-alias.json"
    output_root.mkdir()
    source_bytes = b'{"source":true}\n'
    transformed_bytes = b'{"final":true}\n'
    source_path.write_bytes(source_bytes)
    transform_called = False

    def transform(data: bytes) -> bytes:
        nonlocal transform_called
        transform_called = True
        assert data == source_bytes
        assert not destination_path.exists()
        with pytest.raises(FileNotFoundError):
            os.link(destination_path, outside_alias)
        return transformed_bytes

    copied = copy_confined_artifact(
        output_root,
        source_path,
        destination_path,
        transform_bytes=transform,
    )

    assert transform_called is True
    assert source_path.read_bytes() == source_bytes
    assert destination_path.read_bytes() == transformed_bytes
    assert not outside_alias.exists()
    assert destination_path.stat().st_ino != source_path.stat().st_ino
    assert copied.sha256 == hashlib.sha256(transformed_bytes).hexdigest()
    assert copied.size_bytes == len(transformed_bytes)


def test_copy_confined_artifact_detects_source_mutation_and_cleans_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    source_path = output_root / "source.bin"
    destination_path = output_root / "execution" / "batch" / "source.bin"
    output_root.mkdir()
    source_path.write_bytes(b"a" * (2 * 1024 * 1024))
    source_identity = (source_path.stat().st_dev, source_path.stat().st_ino)
    real_read = evidence_module.os.read
    mutated = False

    def mutating_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, size)
        descriptor_stat = os.fstat(descriptor)
        if not mutated and (descriptor_stat.st_dev, descriptor_stat.st_ino) == source_identity and chunk:
            with source_path.open("r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                handle.write(b"b")
                handle.flush()
                os.fsync(handle.fileno())
            mutated = True
        return chunk

    monkeypatch.setattr(evidence_module.os, "read", mutating_read)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        copy_confined_artifact(output_root, source_path, destination_path)

    assert mutated is True
    assert exc_info.value.code == "artifact_changed"
    assert not destination_path.exists()
    assert not list(destination_path.parent.glob(".*.tmp"))


def _rewrite_run_card_with_valid_self_integrity(run_card_path: Path, run_card: dict[str, Any]) -> None:
    """Rewrite a test run card while preserving its two-part self-integrity contract."""

    integrity = run_card["run_card_integrity"]
    integrity_without_hash = {key: value for key, value in integrity.items() if key != "canonical_payload_sha256"}
    run_card["run_card_integrity"] = {
        **integrity_without_hash,
        "canonical_payload_sha256": hashlib.sha256(
            canonicalize_json({**run_card, "run_card_integrity": integrity_without_hash})
        ).hexdigest(),
    }
    run_card_bytes = dumps_json(
        run_card,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    run_card_path.write_bytes(run_card_bytes)
    sidecar_path = run_card_path.with_suffix(".self.json")
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["final_run_card_sha256"] = hashlib.sha256(run_card_bytes).hexdigest()
    sidecar_path.write_bytes(
        dumps_json(
            sidecar,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _refresh_evidence_artifact_record(
    evidence: dict[str, Any],
    *,
    artifact_kind: str,
    artifact_path: Path,
) -> None:
    artifact_bytes = artifact_path.read_bytes()
    outcome = next(item for item in evidence["produced_artifacts"] if item["artifact_kind"] == artifact_kind)
    record = outcome["artifacts"][0]
    record["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    record["size_bytes"] = len(artifact_bytes)
    evidence["evidence_fingerprint_sha256"] = compute_execution_evidence_fingerprint(evidence)


def _refresh_run_card_index_entries(run_card_path: Path, artifact_paths: list[Path]) -> dict[str, Any]:
    """Refresh selected index leaves and the corresponding v1/v2 commitment."""

    from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
    from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree

    run_card = json.loads(run_card_path.read_bytes())
    entries_by_path = {
        entry["relative_path"]: entry
        for entry in run_card["artifact_index"]
        if isinstance(entry, dict) and isinstance(entry.get("relative_path"), str)
    }
    for artifact_path in artifact_paths:
        relative_path = artifact_path.relative_to(run_card_path.parent).as_posix()
        entry = entries_by_path[relative_path]
        artifact_bytes = artifact_path.read_bytes()
        entry["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
        entry["size_bytes"] = len(artifact_bytes)
    if run_card["run_card_version"] == "v2":
        include_proofs = isinstance(run_card.get("artifact_tree"), dict) and "proofs" in run_card["artifact_tree"]
        run_card["artifact_tree"] = build_artifact_tree(run_card["artifact_index"], include_proofs=include_proofs)
    else:
        run_card["artifact_merkle_root"] = compute_artifact_merkle_root(run_card["artifact_index"])
    _rewrite_run_card_with_valid_self_integrity(run_card_path, run_card)
    return run_card


def test_execution_evidence_schema_ships_with_package() -> None:
    schema = load_execution_evidence_schema()
    assert schema["$id"] == EXECUTION_EVIDENCE_SCHEMA


def test_execution_evidence_hashes_full_confined_artifacts_and_binds_plan(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )
    _rewrite_observation_carrier_outcomes(prepared, observations, payload)
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
        require_carrier_outcome_projections=True,
    )

    assert payload["plan_schema"] == "tp.execution.plan.v1"
    assert payload["plan_fingerprint"] == prepared.plan.plan_fingerprint_sha256
    assert payload["planned_backend"] == "synthetic"
    assert payload["candidate_fallback_chain"] == ["synthetic"]
    assert payload["executed_backend"] == "synthetic"
    assert not payload["omitted_artifacts"]
    assert not payload["failed_artifacts"]
    assert len(payload["produced_artifacts"]) == 4
    depth = next(item for item in payload["produced_artifacts"] if item["artifact_kind"] == "depth_u16_png")
    depth_file = output_root / depth["artifacts"][0]["path"]
    assert depth["artifacts"][0] == {
        "path": "depth/sample_depth.png",
        "sha256": hashlib.sha256(depth_file.read_bytes()).hexdigest(),
        "size_bytes": depth_file.stat().st_size,
        "media_type": "image/png",
        "file_extension": "png",
    }
    require_required_artifacts(payload)

    evidence_path = output_root / "manifests/execution_evidence_test.json"
    write_execution_evidence(evidence_path, payload, output_root=output_root, plan=prepared.plan)
    assert evidence_path.read_bytes() == canonicalize_json(payload)
    assert (
        verify_execution_evidence_file(
            evidence_path,
            output_root=output_root,
            plan=prepared.plan,
        )
        == payload
    )

    combined = next(item for item in payload["produced_artifacts"] if item["artifact_kind"] == "combined_manifest_json")
    combined_path = output_root / combined["artifacts"][0]["path"]
    assert combined["artifacts"][0]["sha256"] == hashlib.sha256(combined_path.read_bytes()).hexdigest()
    batch = next(item for item in payload["produced_artifacts"] if item["artifact_kind"] == "batch_manifest_json")
    batch_path = output_root / batch["artifacts"][0]["path"]
    assert batch["artifacts"][0]["sha256"] == hashlib.sha256(batch_path.read_bytes()).hexdigest()

    depth_file.write_bytes(b"tampered-after-publication")
    with pytest.raises(ExecutionEvidenceError, match="does not match final bytes"):
        verify_execution_evidence_file(
            evidence_path,
            output_root=output_root,
            plan=prepared.plan,
        )


def test_execution_evidence_records_then_rejects_missing_required_output(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    observations = [item for item in observations if item.artifact_kind != "depth_metadata_json"]

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    assert payload["failed_artifacts"] == [
        {
            "declaration_id": "lux.depth.output.depth_metadata_json",
            "stage_registry_id": "tp.stage.lux.depth.v1",
            "artifact_kind": "depth_metadata_json",
            "scope": "per_input",
            "cardinality": "one",
            "required": True,
            "input_id": prepared.plan.inputs[0].input_id,
            "reason_code": "required_output_missing",
        }
    ]
    with pytest.raises(ExecutionEvidenceError, match="required artifact accounting"):
        require_required_artifacts(payload)


@pytest.mark.parametrize("artifact_kind", ["combined_manifest_json", "batch_manifest_json"])
def test_execution_evidence_rejects_manifest_without_exact_plan_sidecar_binding(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    manifest_observation = next(item for item in observations if item.artifact_kind == artifact_kind)
    assert manifest_observation.path is not None
    manifest_observation.path.write_text("{}\n", encoding="utf-8")

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == artifact_kind)
    assert failure["reason_code"] == "invalid_manifest_binding"
    with pytest.raises(ExecutionEvidenceError, match=artifact_kind):
        require_required_artifacts(payload)


def test_evidence_verifier_rejects_sidecar_path_not_discoverable_from_manifests(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    wrong_path = output_root / "manifests/execution_evidence_other.json"
    write_execution_evidence(wrong_path, payload, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ExecutionEvidenceError, match="does not point|runtime projection"):
        verify_execution_evidence_file(
            wrong_path,
            output_root=output_root,
            plan=prepared.plan,
        )


def test_execution_evidence_rejects_cross_input_artifact_path_swap(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_paths = [input_dir / name for name in ("a.png", "b.png")]
    for index, path in enumerate(input_paths):
        Image.new("RGB", (4, 4), color=(index * 64, 127, 127)).save(path)
    prepared = prepare_lux_execution(
        EnhanceConfig(depth_backend="synthetic", enable_v2=False, emit_run_card=False),
        input_root=input_dir,
        input_files=input_paths,
    )
    output_root = tmp_path / "output"
    evidence_path = "manifests/execution_evidence_test.json"
    execution_rows = [
        InputExecution(input_id=item.input_id, status="ok", executed_backend="synthetic") for item in prepared.plan.inputs
    ]
    artifacts_by_input: dict[str, dict[str, Path]] = {}
    for plan_input, input_path, execution_row in zip(prepared.plan.inputs, input_paths, execution_rows):
        depth_path = output_root / "depth" / f"{input_path.stem}_depth.png"
        metadata_path = depth_path.with_name(f"{depth_path.stem}_metadata.json")
        combined_path = output_root / "manifests" / f"{input_path.stem}_combined.json"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.write_bytes(f"depth-{input_path.stem}".encode("ascii"))
        metadata_path.write_text("{}\n", encoding="utf-8")
        projection = build_manifest_plan_projection(
            prepared.plan,
            input_executions=[execution_row],
            evidence_path=evidence_path,
        )
        CombinedManifest(
            input=InputMetadata(image_path=str(input_path)),
            depth=DepthMetadata(
                model="synthetic",
                depth_path=str(depth_path),
                runtime_seconds=0.01,
                scaling={},
            ),
            backend_selection=BackendSelectionMetadata(
                requested_backend="synthetic",
                resolved_backend="synthetic",
                resolution_status="success",
                resolution_reason=None,
                model_id="synthetic/depth-analytic-v1",
                device="cpu",
            ),
            environment={
                "execution_contract": {
                    "authoritative_plan": prepared.plan.to_payload(),
                    "runtime": projection,
                    "execution_evidence_path": evidence_path,
                }
            },
        ).save(combined_path)
        artifacts_by_input[plan_input.input_id] = {
            "depth_u16_png": depth_path,
            "depth_metadata_json": metadata_path,
            "combined_manifest_json": combined_path,
        }
    batch_path = output_root / "manifests/batch_test.json"
    batch_projection = build_manifest_plan_projection(
        prepared.plan,
        input_executions=execution_rows,
        evidence_path=evidence_path,
    )
    BatchManifest(
        batch_id="test",
        start_time="2026-09-02T00:00:00Z",
        end_time="2026-09-02T00:00:01Z",
        config={
            "execution_contract": {
                "authoritative_plan": prepared.plan.to_payload(),
                "runtime": batch_projection,
                "execution_evidence_path": evidence_path,
            }
        },
        results=[],
        stats={},
    ).write(batch_path)

    first_id, second_id = (item.input_id for item in prepared.plan.inputs)
    observations = [ArtifactObservation(kind, path, first_id) for kind, path in artifacts_by_input[second_id].items()] + [
        ArtifactObservation(kind, path, second_id) for kind, path in artifacts_by_input[first_id].items()
    ]
    observations.append(ArtifactObservation("batch_manifest_json", batch_path, None))
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path=evidence_path,
        input_executions=execution_rows,
        artifact_observations=observations,
    )

    failed_kinds = {item["artifact_kind"] for item in payload["failed_artifacts"]}
    assert {"depth_u16_png", "depth_metadata_json", "combined_manifest_json"} <= failed_kinds
    with pytest.raises(ExecutionEvidenceError, match="artifact_input_mismatch"):
        require_required_artifacts(payload)


def test_execution_evidence_permits_typed_omission_only_for_optional_output(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path, generate_pbr=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    assert len(payload["omitted_artifacts"]) == 1
    omission = payload["omitted_artifacts"][0]
    assert omission["artifact_kind"] == "pbr_maps"
    assert omission["required"] is False
    assert omission["reason_code"] == "optional_stage_no_output"
    require_required_artifacts(payload)


def test_execution_evidence_enforces_registry_cardinality(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path, generate_pbr=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    input_id = prepared.plan.inputs[0].input_id
    pbr_paths = [output_root / "pbr" / name for name in ("normal.png", "roughness.png", "ao.png")]
    for index, path in enumerate(pbr_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"pbr-{index}".encode("ascii"))
        observations.append(ArtifactObservation("pbr_maps", path, input_id))
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    combined = CombinedManifest.load(combined_observation.path)
    combined.pbr_assets = {
        "normal_path": str(pbr_paths[0]),
        "roughness_path": str(pbr_paths[1]),
        "ao_path": str(pbr_paths[2]),
    }
    combined.save(combined_observation.path)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )
    pbr = next(item for item in payload["produced_artifacts"] if item["artifact_kind"] == "pbr_maps")
    assert len(pbr["artifacts"]) == 3

    duplicated_one = [*observations, observations[0]]
    mismatched = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=duplicated_one,
    )
    failure = next(item for item in mismatched["failed_artifacts"] if item["artifact_kind"] == "depth_u16_png")
    assert failure["reason_code"] == "cardinality_mismatch"
    assert failure["observed_count"] == 2
    with pytest.raises(ExecutionEvidenceError, match="cardinality_mismatch"):
        require_required_artifacts(mismatched)


def test_execution_evidence_fails_closed_on_unconfined_artifact(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    observations = [
        ArtifactObservation(item.artifact_kind, outside, item.input_id) if item.artifact_kind == "depth_u16_png" else item
        for item in observations
    ]

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "depth_u16_png")
    assert failure["reason_code"] == "path_escape"
    with pytest.raises(ExecutionEvidenceError, match="depth_u16_png"):
        require_required_artifacts(payload)


def test_relative_output_spelling_with_parent_component_is_rejected_before_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "out"
    output_root.mkdir()
    (tmp_path / "outside.bin").write_bytes(b"outside")
    monkeypatch.chdir(tmp_path)

    with evidence_module._pin_output_root(output_root) as root:
        with pytest.raises(ArtifactEvidenceError) as exc_info:
            root.confined_relative_path(Path("out/../outside.bin"))

    assert exc_info.value.code == "path_escape"


def test_execution_evidence_rejects_symlinked_artifact_path(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    depth_observation = next(item for item in observations if item.artifact_kind == "depth_u16_png")
    assert depth_observation.path is not None
    target = depth_observation.path.with_name("target.png")
    depth_observation.path.replace(target)
    depth_observation.path.symlink_to(target.name)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "depth_u16_png")
    assert failure["reason_code"] == "symlink_forbidden"


@pytest.mark.skipif(os.name == "nt" or os.open not in os.supports_dir_fd, reason="requires descriptor-relative open")
def test_execution_evidence_blocks_parent_symlink_swap_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    depth_dir = output_root / "depth"
    held_dir = output_root / "held-depth"
    outside_dir = tmp_path / "outside-depth"
    outside_dir.mkdir()
    (outside_dir / "sample_depth.png").write_bytes(b"outside-depth")
    (outside_dir / "sample_depth_metadata.json").write_text("{}\n", encoding="utf-8")
    real_open = evidence_module.os.open
    swapped = False

    def swapping_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        if not swapped and path == "depth" and dir_fd is not None:
            depth_dir.rename(held_dir)
            depth_dir.symlink_to(outside_dir, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evidence_module.os, "open", swapping_open)
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    assert swapped is True
    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "depth_u16_png")
    assert failure["reason_code"] == "symlink_forbidden"


@pytest.mark.skipif(os.name == "nt" or os.open not in os.supports_dir_fd, reason="requires descriptor-relative open")
@pytest.mark.parametrize(
    ("replacement_kind", "expected_error"),
    [("symlink", "symlink_forbidden"), ("directory", "output_root_changed")],
)
def test_execution_evidence_pins_output_root_identity_during_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
    expected_error: str,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    held_root = tmp_path / "held-output"
    outside_root = tmp_path / "outside-output"
    outside_root.mkdir()
    real_open = evidence_module.os.open
    swapped = False

    def swapping_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        if not swapped and path == output_root.name and dir_fd is not None:
            output_root.rename(held_root)
            if replacement_kind == "symlink":
                output_root.symlink_to(outside_root, target_is_directory=True)
            else:
                output_root.mkdir()
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evidence_module.os, "open", swapping_open)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        build_execution_evidence(
            prepared.plan,
            output_root=output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=observations,
        )

    assert swapped is True
    assert exc_info.value.code == expected_error


@pytest.mark.skipif(os.name == "nt" or os.open not in os.supports_dir_fd, reason="requires descriptor-relative open")
def test_execution_evidence_rejects_output_root_ancestor_redirect_during_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    job_root = tmp_path / "job-root"
    job_root.mkdir()
    prepared, _ = _prepared(job_root)
    output_root = job_root / "output"
    observations = _required_observations(prepared, output_root)
    held_job_root = tmp_path / "held-job-root"
    outside_job_root = tmp_path / "outside-job-root"
    (outside_job_root / "output").mkdir(parents=True)
    real_open = evidence_module.os.open
    swapped = False

    def swapping_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        if not swapped and path == job_root.name and dir_fd is not None:
            job_root.rename(held_job_root)
            job_root.symlink_to(outside_job_root, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evidence_module.os, "open", swapping_open)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        build_execution_evidence(
            prepared.plan,
            output_root=output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=observations,
        )

    assert swapped is True
    assert exc_info.value.code == "symlink_forbidden"


def test_artifact_capture_revalidates_pinned_root_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    output_root.mkdir()
    artifact_path = output_root / "artifact.bin"
    artifact_path.write_bytes(b"artifact")
    outside_held_root = tmp_path / "outside-held-output"
    real_read = evidence_module.os.read
    reads = 0

    def tracking_read(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        return real_read(descriptor, size)

    monkeypatch.setattr(evidence_module.os, "read", tracking_read)
    with evidence_module._pin_output_root(output_root) as root:
        output_root.rename(outside_held_root)
        output_root.mkdir()
        with pytest.raises(ArtifactEvidenceError) as exc_info:
            evidence_module._capture_artifact(
                root,
                artifact_path,
                budget=evidence_module._CaptureBudget(),
            )

    assert exc_info.value.code == "output_root_changed"
    assert reads == 0


def test_execution_evidence_fails_closed_without_secure_descriptor_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    monkeypatch.setattr(evidence_module, "_HAS_SECURE_DIR_FD_OPEN", False)

    with pytest.raises(ArtifactEvidenceError) as exc_info:
        build_execution_evidence(
            prepared.plan,
            output_root=output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=observations,
        )

    assert exc_info.value.code == "secure_traversal_unavailable"


@pytest.mark.skipif(os.name == "nt" or os.rename not in os.supports_dir_fd, reason="requires fd-relative rename")
def test_execution_evidence_publication_cannot_follow_swapped_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    manifests_dir = output_root / "manifests"
    outside_held_manifests = tmp_path / "outside-held-manifests"
    real_rename = evidence_module.os.rename
    swapped = False

    def swapping_rename(
        source: Any,
        destination: Any,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if not swapped and src_dir_fd is not None and dst_dir_fd is not None:
            manifests_dir.rename(outside_held_manifests)
            swapped = True
        real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(evidence_module.os, "rename", swapping_rename)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert swapped is True
    assert exc_info.value.code == "output_root_changed"
    assert not evidence_path.exists()
    # POSIX has no atomic compare-and-unlink operation. The failed publisher
    # retains its fully written orphan in the descriptor-pinned directory
    # rather than risk deleting a replacement installed at the same name.
    assert (outside_held_manifests / evidence_path.name).read_bytes() == canonicalize_json(payload)


@pytest.mark.skipif(os.name == "nt" or os.unlink not in os.supports_dir_fd, reason="requires fd-relative unlink")
def test_execution_evidence_temp_cleanup_removes_owned_inode_after_partial_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    real_write = evidence_module.os.write
    targeted_writes = 0

    def failing_write(descriptor: int, data: Any) -> int:
        nonlocal targeted_writes
        candidates = list(evidence_path.parent.glob(f".{evidence_path.name}.*.tmp"))
        if candidates:
            descriptor_stat = os.fstat(descriptor)
            candidate_stat = candidates[0].stat(follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (candidate_stat.st_dev, candidate_stat.st_ino):
                targeted_writes += 1
                if targeted_writes == 1:
                    return real_write(descriptor, data[:17])
                raise OSError(errno.EIO, "injected partial write failure")
        return real_write(descriptor, data)

    monkeypatch.setattr(evidence_module.os, "write", failing_write)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert targeted_writes == 2
    assert exc_info.value.code == "artifact_unreadable"
    assert not evidence_path.exists()
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.*.tmp"))


@pytest.mark.skipif(os.name == "nt" or os.unlink not in os.supports_dir_fd, reason="requires fd-relative unlink")
def test_execution_evidence_temp_cleanup_removes_owned_inode_after_pre_rename_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    real_fsync = evidence_module.os.fsync
    injected = False

    def failing_fsync(descriptor: int) -> None:
        nonlocal injected
        candidates = list(evidence_path.parent.glob(f".{evidence_path.name}.*.tmp"))
        if not injected and candidates:
            descriptor_stat = os.fstat(descriptor)
            candidate_stat = candidates[0].stat(follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (candidate_stat.st_dev, candidate_stat.st_ino):
                injected = True
                raise OSError(errno.EIO, "injected pre-rename fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", failing_fsync)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert injected is True
    assert exc_info.value.code == "artifact_unreadable"
    assert not evidence_path.exists()
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.*.tmp"))


@pytest.mark.skipif(os.name == "nt" or os.unlink not in os.supports_dir_fd, reason="requires fd-relative unlink")
def test_execution_evidence_final_cleanup_removes_owned_inode_after_parent_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    real_fsync = evidence_module.os.fsync
    injected = False

    def failing_fsync(descriptor: int) -> None:
        nonlocal injected
        if not injected and evidence_path.exists() and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            injected = True
            raise OSError(errno.EIO, "injected post-rename parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", failing_fsync)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert injected is True
    assert exc_info.value.code == "artifact_unreadable"
    assert not evidence_path.exists()
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.*.tmp"))


@pytest.mark.skipif(os.name == "nt" or os.stat not in os.supports_dir_fd, reason="requires fd-relative stat")
def test_execution_evidence_publication_rejects_final_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    replaced_path = evidence_path.with_name("verified-old.json")
    real_read = evidence_module.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced and evidence_path.exists():
            evidence_path.rename(replaced_path)
            evidence_path.write_bytes(b"evil")
            replaced = True
        return real_read(descriptor, size)

    monkeypatch.setattr(evidence_module.os, "read", replacing_read)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert replaced is True
    assert exc_info.value.code == "artifact_changed"
    assert evidence_path.read_bytes() == b"evil"
    assert replaced_path.read_bytes() == canonicalize_json(payload)


@pytest.mark.skipif(os.name == "nt" or os.stat not in os.supports_dir_fd, reason="requires fd-relative stat")
def test_execution_evidence_temp_cleanup_preserves_replacement_inode_after_pre_rename_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    held_original = output_root / "manifests/held-original.tmp"
    replacement_path: Path | None = None
    real_fsync = evidence_module.os.fsync

    def replacing_fsync(descriptor: int) -> None:
        nonlocal replacement_path
        candidates = list(evidence_path.parent.glob(f".{evidence_path.name}.*.tmp"))
        if replacement_path is None and candidates:
            candidate = candidates[0]
            descriptor_stat = os.fstat(descriptor)
            candidate_stat = candidate.stat(follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (candidate_stat.st_dev, candidate_stat.st_ino):
                candidate.rename(held_original)
                candidate.write_bytes(b"replacement-temp-inode")
                replacement_path = candidate
                raise OSError(errno.EIO, "injected pre-rename fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", replacing_fsync)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert exc_info.value.code == "artifact_unreadable"
    assert replacement_path is not None
    assert replacement_path.read_bytes() == b"replacement-temp-inode"
    assert held_original.read_bytes() == canonicalize_json(payload)
    assert not evidence_path.exists()


def test_execution_evidence_publication_uses_fixed_new_mode_and_preserves_existing_mode(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"

    previous_umask = os.umask(0o077)
    try:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o644

    evidence_path.chmod(0o640)
    write_execution_evidence(
        evidence_path,
        payload,
        output_root=output_root,
        plan=prepared.plan,
    )
    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o640


def test_execution_evidence_publication_preserves_unreadable_existing_destination(
    tmp_path: Path,
) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    evidence_path.write_bytes(b"existing-evidence")
    evidence_path.chmod(0o200)

    with pytest.raises(ArtifactEvidenceError) as exc_info:
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert exc_info.value.code == "artifact_unreadable"
    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o200
    assert not list(evidence_path.parent.glob(f".{evidence_path.name}.*.tmp"))
    evidence_path.chmod(0o600)
    assert evidence_path.read_bytes() == b"existing-evidence"


@pytest.mark.skipif(os.name == "nt" or os.open not in os.supports_dir_fd, reason="requires descriptor-relative open")
def test_execution_evidence_publication_rejects_output_root_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    job_root = tmp_path / "job-root"
    job_root.mkdir()
    prepared, _ = _prepared(job_root)
    output_root = job_root / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    evidence_path = output_root / "manifests/execution_evidence_test.json"
    held_root = tmp_path / "held-job-root"
    outside_root = tmp_path / "outside-job-root"
    (outside_root / "output/manifests").mkdir(parents=True)
    real_open = evidence_module.os.open
    swapped = False

    def swapping_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        if not swapped and path == job_root.name and dir_fd is not None:
            job_root.rename(held_root)
            job_root.symlink_to(outside_root, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evidence_module.os, "open", swapping_open)
    with pytest.raises(ArtifactEvidenceError):
        write_execution_evidence(
            evidence_path,
            payload,
            output_root=output_root,
            plan=prepared.plan,
        )

    assert swapped is True
    assert not (outside_root / "output" / "manifests" / evidence_path.name).exists()


def test_execution_evidence_rejects_outside_inode_hardlink_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    depth_observation = next(item for item in observations if item.artifact_kind == "depth_u16_png")
    assert depth_observation.path is not None
    outside = tmp_path / "outside-depth.png"
    outside.write_bytes(b"outside-inode")
    depth_observation.path.unlink()
    os.link(outside, depth_observation.path)
    real_read = evidence_module.os.read
    hardlink_descriptor_reads = 0

    def tracking_read(descriptor: int, size: int) -> bytes:
        nonlocal hardlink_descriptor_reads
        descriptor_stat = os.fstat(descriptor)
        if descriptor_stat.st_ino == outside.stat().st_ino:
            hardlink_descriptor_reads += 1
        return real_read(descriptor, size)

    monkeypatch.setattr(evidence_module.os, "read", tracking_read)
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "depth_u16_png")
    assert failure["reason_code"] == "hardlink_forbidden"
    assert hardlink_descriptor_reads == 0


def test_orchestrator_combined_manifest_discovery_rejects_outside_symlink_without_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    fixture_observations = _required_observations(prepared, output_root)
    fixture_paths = _observation_paths(fixture_observations)
    combined_path = fixture_paths["combined_manifest_json"]
    batch_path = fixture_paths["batch_manifest_json"]
    outside_manifest = tmp_path / "outside-combined.json"
    outside_manifest.write_bytes(combined_path.read_bytes())
    combined_path.unlink()
    combined_path.symlink_to(outside_manifest)
    outside_identity = (outside_manifest.stat().st_dev, outside_manifest.stat().st_ino)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    results = [
        {
            "status": "ok",
            "image": str(image_path),
            "backend": "synthetic",
            "manifest": str(combined_path),
        }
    ]
    observations = orchestrator._execution_artifact_observations(
        results,
        batch_manifest_path=batch_path,
        run_card_path=None,
    )
    real_read = evidence_module.os.read
    outside_reads = 0

    def tracking_read(descriptor: int, size: int) -> bytes:
        nonlocal outside_reads
        descriptor_stat = os.fstat(descriptor)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) == outside_identity:
            outside_reads += 1
        return real_read(descriptor, size)

    monkeypatch.setattr(evidence_module.os, "read", tracking_read)
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
        derive_manifest_outputs=True,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "combined_manifest_json")
    assert failure["reason_code"] == "symlink_forbidden"
    assert outside_reads == 0


def test_combined_manifest_discovery_enforces_max_and_max_plus_one_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    fixture_observations = _required_observations(prepared, output_root)
    fixture_paths = _observation_paths(fixture_observations)
    combined_path = fixture_paths["combined_manifest_json"]
    batch_path = fixture_paths["batch_manifest_json"]
    combined_size = combined_path.stat().st_size
    combined_identity = (combined_path.stat().st_dev, combined_path.stat().st_ino)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    results = [
        {
            "status": "ok",
            "image": str(image_path),
            "backend": "synthetic",
            "manifest": str(combined_path),
        }
    ]
    observations = orchestrator._execution_artifact_observations(
        results,
        batch_manifest_path=batch_path,
        run_card_path=None,
    )

    monkeypatch.setattr(evidence_module, "_MAX_EVIDENCE_BYTES", combined_size)
    exact_payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
        derive_manifest_outputs=True,
    )
    assert any(item["artifact_kind"] == "combined_manifest_json" for item in exact_payload["produced_artifacts"])

    real_read = evidence_module.os.read
    combined_reads = 0

    def tracking_read(descriptor: int, size: int) -> bytes:
        nonlocal combined_reads
        descriptor_stat = os.fstat(descriptor)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) == combined_identity:
            combined_reads += 1
        return real_read(descriptor, size)

    monkeypatch.setattr(evidence_module.os, "read", tracking_read)
    monkeypatch.setattr(evidence_module, "_MAX_EVIDENCE_BYTES", combined_size - 1)
    oversized_payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
        derive_manifest_outputs=True,
    )

    failure = next(item for item in oversized_payload["failed_artifacts"] if item["artifact_kind"] == "combined_manifest_json")
    assert failure["reason_code"] == "artifact_too_large"
    assert combined_reads == 0


def test_execution_evidence_rejects_duplicate_inode_aliases_for_many_output(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path, generate_pbr=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    input_id = prepared.plan.inputs[0].input_id
    source = output_root / "pbr/source.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"one-inode")
    pbr_paths = [output_root / f"pbr/map_{index}.png" for index in range(3)]
    for path in pbr_paths:
        os.link(source, path)
        observations.append(ArtifactObservation("pbr_maps", path, input_id))
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    combined = CombinedManifest.load(combined_observation.path)
    combined.pbr_assets = {
        "normal_path": str(pbr_paths[0]),
        "roughness_path": str(pbr_paths[1]),
        "ao_path": str(pbr_paths[2]),
    }
    combined.save(combined_observation.path)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "pbr_maps")
    assert failure["reason_code"] == "hardlink_forbidden"


def test_execution_evidence_observation_limit_accepts_max_and_rejects_max_plus_one_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    monkeypatch.setattr(evidence_module, "_MAX_ARTIFACT_OBSERVATIONS", len(observations))
    build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    monkeypatch.setattr(
        evidence_module,
        "_capture_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("capture must not start")),
    )
    with pytest.raises(ExecutionEvidenceError, match="observations exceed"):
        build_execution_evidence(
            prepared.plan,
            output_root=output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=[*observations, observations[0]],
        )


def test_execution_evidence_payload_rejects_oversized_input_collection_before_schema_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    payload["requested_inputs"].append(dict(payload["requested_inputs"][0]))
    monkeypatch.setattr(evidence_module, "_MAX_PLAN_INPUTS", 1)

    with pytest.raises(ExecutionEvidenceError, match="requested_inputs exceeds the bounded limit of 1"):
        validate_execution_evidence_payload(payload, plan=prepared.plan)


def test_execution_evidence_rejects_extreme_integer_before_schema_walk(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    payload["produced_artifacts"][0]["artifacts"][0]["size_bytes"] = -(10**5_000)

    with pytest.raises(ExecutionEvidenceError, match="integer exceeds the bounded bit-length limit"):
        validate_execution_evidence_payload(payload, plan=prepared.plan)


def test_execution_evidence_many_cardinality_limit_accepts_max_and_rejects_max_plus_one_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path, generate_pbr=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    input_id = prepared.plan.inputs[0].input_id
    pbr_paths = [output_root / f"pbr/map_{index}.png" for index in range(4)]
    for index, path in enumerate(pbr_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"map-{index}".encode())
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    combined = CombinedManifest.load(combined_observation.path)
    combined.pbr_assets = {
        "normal_path": str(pbr_paths[0]),
        "roughness_path": str(pbr_paths[1]),
        "ao_path": str(pbr_paths[2]),
    }
    combined.save(combined_observation.path)
    max_observations = [*observations]
    max_observations.extend(ArtifactObservation("pbr_maps", path, input_id) for path in pbr_paths[:3])
    monkeypatch.setattr(evidence_module, "_MAX_ARTIFACTS_PER_OUTCOME", 3)
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=max_observations,
    )
    assert any(item["artifact_kind"] == "pbr_maps" for item in payload["produced_artifacts"])

    monkeypatch.setattr(
        evidence_module,
        "_capture_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("capture must not start")),
    )
    with pytest.raises(ExecutionEvidenceError, match="bounded cardinality limit"):
        build_execution_evidence(
            prepared.plan,
            output_root=output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=[*max_observations, ArtifactObservation("pbr_maps", pbr_paths[3], input_id)],
        )


def test_derived_v2_observations_report_typed_max_plus_one_cardinality_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path, enable_v2=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    combined = CombinedManifest.load(combined_observation.path)
    v2_paths = [output_root / f"v2/output_{index}.png" for index in range(4)]
    for index, path in enumerate(v2_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"v2-{index}".encode("ascii"))

    monkeypatch.setattr(evidence_module, "_MAX_ARTIFACTS_PER_OUTCOME", 3)
    combined.v2 = V2Metadata(
        preset="standard",
        status="ok",
        output_paths=[str(path) for path in v2_paths[:3]],
    )
    combined.save(combined_observation.path)
    exact_payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
        derive_manifest_outputs=True,
    )
    exact_v2 = next(item for item in exact_payload["failed_artifacts"] if item["artifact_kind"] == "v2_enhanced_image")
    assert exact_v2["reason_code"] == "cardinality_mismatch"
    assert exact_v2["observed_count"] == 3

    combined.v2.output_paths = [str(path) for path in v2_paths]
    combined.save(combined_observation.path)
    overflow_payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
        derive_manifest_outputs=True,
    )

    failure = next(item for item in overflow_payload["failed_artifacts"] if item["artifact_kind"] == "v2_enhanced_image")
    assert failure["reason_code"] == "artifact_cardinality_limit_exceeded"
    assert failure["observed_count"] == 4


def test_global_observation_limit_counts_supplied_and_manifest_derived_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path, enable_v2=True)
    output_root = tmp_path / "output"
    all_observations = _required_observations(prepared, output_root)
    supplied_observations = [
        item for item in all_observations if item.artifact_kind in {"combined_manifest_json", "batch_manifest_json"}
    ]
    combined_observation = next(item for item in supplied_observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    combined = CombinedManifest.load(combined_observation.path)
    v2_paths = [output_root / "v2/global_0.png"]
    for index, path in enumerate(v2_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"global-v2-{index}".encode("ascii"))
    combined.v2 = V2Metadata(
        preset="standard",
        status="ok",
        output_paths=[str(path) for path in v2_paths],
    )
    combined.save(combined_observation.path)
    batch_observation = next(item for item in supplied_observations if item.artifact_kind == "batch_manifest_json")
    assert batch_observation.path is not None
    batch = json.loads(batch_observation.path.read_bytes())
    batch["results"][0]["v2_output_path"] = v2_paths[0].relative_to(output_root).as_posix()
    batch_observation.path.write_text(
        dumps_json(batch, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    # Two supplied manifest observations plus three paths derived from the
    # combined manifest (depth, metadata, and one V2 output).
    monkeypatch.setattr(evidence_module, "_MAX_ARTIFACT_OBSERVATIONS", 5)
    exact_payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=supplied_observations,
        derive_manifest_outputs=True,
    )
    assert not exact_payload["failed_artifacts"]

    monkeypatch.setattr(evidence_module, "_MAX_ARTIFACT_OBSERVATIONS", 4)
    with pytest.raises(ExecutionEvidenceError, match="observations exceed"):
        build_execution_evidence(
            prepared.plan,
            output_root=output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=supplied_observations,
            derive_manifest_outputs=True,
        )


def test_artifact_per_file_limit_accepts_max_and_rejects_max_plus_one_without_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    output_root.mkdir()
    exact = output_root / "exact.bin"
    oversized = output_root / "oversized.bin"
    exact.write_bytes(b"1234")
    oversized.write_bytes(b"12345")
    monkeypatch.setattr(evidence_module, "_MAX_ARTIFACT_BYTES", 4)
    with evidence_module._pin_output_root(output_root) as root:
        exact_capture = evidence_module._capture_artifact(root, exact, budget=evidence_module._CaptureBudget())
        assert exact_capture.record["size_bytes"] == 4
        real_read = evidence_module.os.read
        reads = 0

        def tracking_read(descriptor: int, size: int) -> bytes:
            nonlocal reads
            reads += 1
            return real_read(descriptor, size)

        monkeypatch.setattr(evidence_module.os, "read", tracking_read)
        with pytest.raises(ArtifactEvidenceError) as exc_info:
            evidence_module._capture_artifact(root, oversized, budget=evidence_module._CaptureBudget())
    assert exc_info.value.code == "artifact_too_large"
    assert reads == 0


def test_artifact_aggregate_limit_accepts_max_and_rejects_max_plus_one_without_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    output_root.mkdir()
    exact = output_root / "exact.bin"
    extra = output_root / "extra.bin"
    exact.write_bytes(b"12345678")
    extra.write_bytes(b"x")
    monkeypatch.setattr(evidence_module, "_MAX_ARTIFACT_BYTES", 8)
    monkeypatch.setattr(evidence_module, "_MAX_AGGREGATE_ARTIFACT_BYTES", 8)
    budget = evidence_module._CaptureBudget()
    with evidence_module._pin_output_root(output_root) as root:
        exact_capture = evidence_module._capture_artifact(root, exact, budget=budget)
        assert exact_capture.record["size_bytes"] == 8
        assert budget.total_bytes == 8
        real_read = evidence_module.os.read
        reads = 0

        def tracking_read(descriptor: int, size: int) -> bytes:
            nonlocal reads
            reads += 1
            return real_read(descriptor, size)

        monkeypatch.setattr(evidence_module.os, "read", tracking_read)
        with pytest.raises(ArtifactEvidenceError) as exc_info:
            evidence_module._capture_artifact(root, extra, budget=budget)
    assert exc_info.value.code == "aggregate_artifact_bytes_exceeded"
    assert reads == 0


def test_combined_manifest_backend_selection_must_match_execution_projection(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    combined = CombinedManifest.load(combined_observation.path)
    assert combined.backend_selection is not None
    combined.backend_selection.resolved_backend = "da3"
    combined.save(combined_observation.path)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "combined_manifest_json")
    assert failure["reason_code"] == "invalid_manifest_binding"


@pytest.mark.parametrize("mutation", ["missing_result", "swapped_manifest"])
def test_batch_manifest_success_rows_must_cover_and_associate_all_prepared_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    prepared = _prepared_many(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    batch_observation = next(item for item in observations if item.artifact_kind == "batch_manifest_json")
    assert batch_observation.path is not None
    batch = BatchManifest.load(batch_observation.path)
    assert len(batch.results) == 2
    if mutation == "missing_result":
        batch.results.pop()
    else:
        batch.results[0]["manifest"] = batch.results[1]["manifest"]
    batch.write(batch_observation.path)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "batch_manifest_json")
    assert failure["reason_code"] == "invalid_manifest_binding"


def test_execution_evidence_validates_fingerprint_and_outcome_partition(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )

    drifted = json.loads(json.dumps(payload))
    drifted["planned_backend"] = "da3"
    with pytest.raises(ExecutionEvidenceError, match="planned_backend"):
        validate_execution_evidence_payload(drifted, plan=prepared.plan)

    repeated = json.loads(json.dumps(payload))
    repeated["failed_artifacts"].append(repeated["produced_artifacts"][0] | {"reason_code": "duplicate_artifact_path"})
    repeated["failed_artifacts"][-1].pop("artifacts")
    repeated["evidence_fingerprint_sha256"] = hashlib.sha256(
        canonicalize_json({key: value for key, value in repeated.items() if key != "evidence_fingerprint_sha256"})
    ).hexdigest()
    with pytest.raises(ExecutionEvidenceError, match="repeats artifact outcome"):
        validate_execution_evidence_payload(repeated, plan=prepared.plan)

    missing = json.loads(json.dumps(payload))
    missing["produced_artifacts"].pop()
    missing["evidence_fingerprint_sha256"] = hashlib.sha256(
        canonicalize_json({key: value for key, value in missing.items() if key != "evidence_fingerprint_sha256"})
    ).hexdigest()
    with pytest.raises(ExecutionEvidenceError, match="do not exactly cover"):
        validate_execution_evidence_payload(missing, plan=prepared.plan)

    evidence_path = output_root / "manifests/execution_evidence_test.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(canonicalize_json(missing))
    with pytest.raises(ExecutionEvidenceError, match="do not exactly cover"):
        verify_execution_evidence_file(
            evidence_path,
            output_root=output_root,
            plan=prepared.plan,
        )

    noncanonical_path = json.loads(json.dumps(payload))
    noncanonical_path["produced_artifacts"][0]["artifacts"][0]["path"] = "depth//sample_depth.png"
    noncanonical_path["evidence_fingerprint_sha256"] = hashlib.sha256(
        canonicalize_json({key: value for key, value in noncanonical_path.items() if key != "evidence_fingerprint_sha256"})
    ).hexdigest()
    with pytest.raises(ExecutionEvidenceError, match="schema validation failed"):
        validate_execution_evidence_payload(noncanonical_path, plan=prepared.plan)

    falsified_hash = json.loads(json.dumps(payload))
    falsified_hash["produced_artifacts"][0]["artifacts"][0]["sha256"] = "0" * 64
    falsified_hash["evidence_fingerprint_sha256"] = hashlib.sha256(
        canonicalize_json({key: value for key, value in falsified_hash.items() if key != "evidence_fingerprint_sha256"})
    ).hexdigest()
    evidence_path.write_bytes(canonicalize_json(falsified_hash))
    with pytest.raises(ExecutionEvidenceError, match="does not match final bytes"):
        verify_execution_evidence_file(
            evidence_path,
            output_root=output_root,
            plan=prepared.plan,
        )


@pytest.mark.parametrize("drive_path", ["C:/outside.bin", "C:outside.bin"])
def test_execution_evidence_schema_rejects_windows_drive_qualified_artifact_path(
    tmp_path: Path,
    drive_path: str,
) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    payload["produced_artifacts"][0]["artifacts"][0]["path"] = drive_path
    payload["evidence_fingerprint_sha256"] = compute_execution_evidence_fingerprint(payload)

    with pytest.raises(ExecutionEvidenceError, match="schema validation failed"):
        validate_execution_evidence_payload(payload, plan=prepared.plan)


def test_manifest_projection_never_puts_executed_backend_in_plan(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    projection = build_manifest_plan_projection(
        prepared.plan,
        input_executions=_successful_input(prepared),
        evidence_path="manifests/execution_evidence_test.json",
    )

    assert projection["executed_backend"] == "synthetic"
    assert projection["executed_backend_by_input"] == [
        {
            "input_id": prepared.plan.inputs[0].input_id,
            "executed_backend": "synthetic",
            "status": "ok",
        }
    ]
    assert "executed_backend" not in prepared.plan.to_payload()


def test_orchestrator_reuses_validated_manifest_projector_for_one_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.output_root = tmp_path / "output"
    orchestrator.manifests_dir = orchestrator.output_root / "manifests"
    orchestrator._active_manifest_plan_projector = None
    validation_calls = 0
    real_validate = evidence_module._validate_prepared_plan

    def tracking_validate(plan: Any) -> None:
        nonlocal validation_calls
        validation_calls += 1
        real_validate(plan)

    monkeypatch.setattr(evidence_module, "_validate_prepared_plan", tracking_validate)
    rows = _successful_input(prepared)

    first = orchestrator._execution_plan_projection(input_executions=rows, batch_id="one")
    second = orchestrator._execution_plan_projection(input_executions=rows, batch_id="one")
    assert first == second
    assert validation_calls == 1

    orchestrator._execution_plan_projection(input_executions=rows, batch_id="two")
    assert validation_calls == 2


def test_prepared_input_id_uses_constructor_index_without_revalidating_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    prepared, image_path = _prepared(tmp_path)
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator._prepared_input_ids_by_path = {
        prepared.input_files[0]: prepared.plan.inputs[0].input_id,
    }
    monkeypatch.setattr(
        orchestrator_module,
        "validate_prepared_lux_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prepared plan must not be reparsed")),
    )

    assert {orchestrator._prepared_input_id(image_path) for _ in range(8)} == {prepared.plan.inputs[0].input_id}


def test_batch_ids_remain_unique_with_identical_wall_clock_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    class FrozenDateTime(orchestrator_module.datetime.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Any:
            return cls(2026, 9, 3, 12, 34, 56, 123456, tzinfo=tz)

    tokens = iter(("0000000000000001", "0000000000000002"))
    monkeypatch.setattr(orchestrator_module.datetime, "datetime", FrozenDateTime)
    monkeypatch.setattr(orchestrator_module.secrets, "token_hex", lambda _size: next(tokens))

    first = orchestrator_module._new_batch_id()
    second = orchestrator_module._new_batch_id()

    assert first == "2026-09-03_123456_123456Z_0000000000000001"
    assert second == "2026-09-03_123456_123456Z_0000000000000002"
    assert first != second


def test_execution_evidence_rejects_ensemble_constituent_as_top_level_executed_backend(tmp_path: Path) -> None:
    prepared, _ = _prepared(
        tmp_path,
        depth_backend="ensemble",
        model_key="da3-metric",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        accept_research_tools_license=True,
    )
    assert prepared.plan.candidate_fallback_chain == ("ensemble",)

    with pytest.raises(ExecutionEvidenceError, match="outside the prepared candidate authority"):
        build_execution_evidence(
            prepared.plan,
            output_root=tmp_path / "output",
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=[
                InputExecution(
                    input_id=prepared.plan.inputs[0].input_id,
                    status="ok",
                    executed_backend="da3",
                )
            ],
            artifact_observations=[],
        )


def test_execution_evidence_fails_reused_artifact_that_changed_after_authorization(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    initial = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )
    input_id = prepared.plan.inputs[0].input_id
    depth_outcome = next(
        outcome
        for outcome in initial["produced_artifacts"]
        if outcome["artifact_kind"] == "depth_u16_png" and outcome["input_id"] == input_id
    )
    depth_observation = next(
        observation
        for observation in observations
        if observation.artifact_kind == "depth_u16_png" and observation.input_id == input_id
    )
    assert depth_observation.path is not None
    depth_observation.path.write_bytes(b"changed-after-reuse-authorization")

    changed = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
        expected_artifact_records={
            ("depth_u16_png", input_id): depth_outcome["artifacts"],
        },
    )

    failure = next(
        outcome
        for outcome in changed["failed_artifacts"]
        if outcome["artifact_kind"] == "depth_u16_png" and outcome["input_id"] == input_id
    )
    assert failure["reason_code"] == "artifact_changed"
    with pytest.raises(ExecutionEvidenceError, match="artifact_changed"):
        require_required_artifacts(changed)


def test_existing_manifest_formats_round_trip_nested_execution_contract_without_top_level_fields(
    tmp_path: Path,
) -> None:
    prepared, _ = _prepared(tmp_path)
    evidence_path = "manifests/execution_evidence_test.json"
    projection = build_manifest_plan_projection(
        prepared.plan,
        input_executions=_successful_input(prepared),
        evidence_path=evidence_path,
    )
    contract = {
        "authoritative_plan": prepared.plan.to_payload(),
        "runtime": projection,
        "execution_evidence_path": evidence_path,
    }

    combined_path = tmp_path / "combined.json"
    combined = CombinedManifest(environment={"execution_contract": contract})
    combined.save(combined_path)
    loaded_combined = CombinedManifest.load(combined_path)
    assert loaded_combined.environment == {"execution_contract": contract}
    combined_payload = json.loads(combined_path.read_bytes())
    assert "execution_plan" not in combined_payload
    assert "execution_evidence_path" not in combined_payload

    batch_path = tmp_path / "batch.json"
    batch = BatchManifest(
        batch_id="test",
        start_time="2026-09-02T00:00:00Z",
        end_time="2026-09-02T00:00:01Z",
        config={"execution_contract": contract},
        results=[],
        stats={},
    )
    batch.write(batch_path)
    loaded_batch = BatchManifest.load(batch_path)
    assert loaded_batch.config == {"execution_contract": contract}
    batch_payload = json.loads(batch_path.read_bytes())
    assert "execution_plan" not in batch_payload
    assert "execution_evidence_path" not in batch_payload


@pytest.mark.parametrize("invalid_path", ["C:/outside.json", "C:outside.json"])
def test_manifest_projection_rejects_windows_drive_qualified_pointer(tmp_path: Path, invalid_path: str) -> None:
    prepared, _ = _prepared(tmp_path)
    with pytest.raises(ExecutionEvidenceError, match="canonical confined relative"):
        build_manifest_plan_projection(prepared.plan, evidence_path=invalid_path)


def test_orchestrator_publishes_evidence_before_failing_required_output(tmp_path: Path) -> None:
    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    paths = _observation_paths(observations)

    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.output_root = output_root
    orchestrator.manifests_dir = output_root / "manifests"
    paths["depth_metadata_json"].unlink()
    results = [
        {
            "status": "ok",
            "image": str(image_path),
            "backend": "synthetic",
            "depth_path": str(paths["depth_u16_png"]),
            "manifest": str(paths["combined_manifest_json"]),
        }
    ]
    preliminary = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=orchestrator._execution_input_rows(results),
        artifact_observations=observations,
    )
    _rewrite_observation_carrier_outcomes(prepared, observations, preliminary)
    orchestrator._active_execution_outcome_payload = preliminary

    with pytest.raises(ExecutionEvidenceError, match="depth_metadata_json"):
        orchestrator._emit_prepared_execution_evidence(
            results,
            batch_id="test",
            batch_manifest_path=paths["batch_manifest_json"],
            run_card_path=None,
        )

    evidence_path = output_root / "manifests/execution_evidence_test.json"
    payload = json.loads(evidence_path.read_bytes())
    assert payload["plan_fingerprint"] == prepared.plan.plan_fingerprint_sha256
    assert payload["executed_backend"] == "synthetic"
    assert [item["artifact_kind"] for item in payload["failed_artifacts"]] == ["depth_metadata_json"]


def test_reconstruction_preflight_alone_cannot_satisfy_required_bundle(tmp_path: Path) -> None:
    prepared, image_path = _prepared(
        tmp_path,
        enable_reconstruction=True,
        grouping_mode="parent_dir",
        non_commercial_ok=True,
        accept_research_tools_license=True,
    )
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    paths = _observation_paths(observations)
    preflight_path = output_root / "reconstruction/scene_preflight.json"
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text('{"valid":false}\n', encoding="utf-8")

    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.output_root = output_root
    orchestrator.manifests_dir = output_root / "manifests"
    orchestrator._active_reconstruction_expected_scene_ids = ("scene",)
    orchestrator._active_reconstruction_completed_scene_ids = set()
    results = [
        {
            "status": "ok",
            "image": str(image_path),
            "backend": "synthetic",
            "manifest": str(paths["combined_manifest_json"]),
            "reconstruction_preflight_path": str(preflight_path),
        }
    ]
    artifact_observations = orchestrator._execution_artifact_observations(
        results,
        batch_manifest_path=paths["batch_manifest_json"],
        run_card_path=None,
    )
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=artifact_observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "reconstruction_bundle")
    assert failure["reason_code"] == "incomplete_reconstruction_bundle"
    with pytest.raises(ExecutionEvidenceError, match="reconstruction_bundle"):
        require_required_artifacts(payload)


def test_orchestrator_cannot_complete_without_verifiable_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    prepared, image_path = _prepared(tmp_path)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    paths = _observation_paths(observations)
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.output_root = output_root
    orchestrator.manifests_dir = output_root / "manifests"
    results = [
        {
            "status": "ok",
            "image": str(image_path),
            "backend": "synthetic",
            "depth_path": str(paths["depth_u16_png"]),
            "manifest": str(paths["combined_manifest_json"]),
        }
    ]
    monkeypatch.setattr(orchestrator_module, "write_execution_evidence", lambda *args, **kwargs: None)

    with pytest.raises(ExecutionEvidenceError, match="Artifact path is unavailable"):
        orchestrator._emit_prepared_execution_evidence(
            results,
            batch_id="missing",
            batch_manifest_path=paths["batch_manifest_json"],
            run_card_path=None,
        )


@pytest.mark.regression
def test_valid_synthetic_prepared_batch_uses_existing_metadata_and_emits_verified_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    existing_metadata = orchestrator._backend_metadata
    real_default_model_id = orchestrator._default_model_id_for_backend

    def forbid_unplanned_da3_lookup(backend_id: str) -> str:
        if backend_id == "da3":
            raise AssertionError("synthetic prepared execution must not resolve absent DA3 authority")
        return real_default_model_id(backend_id)

    monkeypatch.setattr(orchestrator, "_default_model_id_for_backend", forbid_unplanned_da3_lookup)
    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert orchestrator._capture_backend_metadata() is existing_metadata
    assert results[0]["status"] == "ok"
    evidence_paths = list((output_root / "manifests").glob("execution_evidence_*.json"))
    assert len(evidence_paths) == 1
    verified = verify_execution_evidence_file(
        evidence_paths[0],
        output_root=output_root,
        plan=prepared.plan,
    )
    require_required_artifacts(verified)
    assert verified["executed_backend"] == "synthetic"
    assert verified["failed_artifacts"] == []
    assert verified["omitted_artifacts"] == []
    assert {item["artifact_kind"] for item in verified["produced_artifacts"]} == {
        "depth_u16_png",
        "depth_metadata_json",
        "combined_manifest_json",
        "batch_manifest_json",
        "run_card",
    }
    evidence_relative_path = evidence_paths[0].relative_to(output_root).as_posix()
    manifest_outcomes = {
        item["artifact_kind"]: item
        for item in verified["produced_artifacts"]
        if item["artifact_kind"] in {"combined_manifest_json", "batch_manifest_json", "run_card"}
    }
    contract_container_by_kind = {
        "combined_manifest_json": "environment",
        "batch_manifest_json": "config",
        "run_card": "effective_config",
    }
    for artifact_kind, outcome in manifest_outcomes.items():
        manifest_path = output_root / outcome["artifacts"][0]["path"]
        manifest_payload = json.loads(manifest_path.read_bytes())
        assert "execution_evidence_path" not in manifest_payload
        assert "execution_plan" not in manifest_payload
        contract = manifest_payload[contract_container_by_kind[artifact_kind]]["execution_contract"]
        assert contract["execution_evidence_path"] == evidence_relative_path
        assert contract["runtime"]["execution_evidence_path"] == evidence_relative_path
        assert contract["authoritative_plan"] == prepared.plan.to_payload()
        expected_outcomes = build_manifest_outcome_projection(
            verified,
            evidence_path=evidence_relative_path,
            input_id=outcome["input_id"] if artifact_kind == "combined_manifest_json" else None,
        )
        for field_name, expected_value in expected_outcomes.items():
            assert contract[field_name] == expected_value
        for produced in contract["produced_artifacts"]:
            assert "artifacts" not in produced
            assert produced["artifact_count"] > 0


@pytest.mark.regression
def test_prepared_startup_synthetic_fallback_records_full_candidate_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module
    from transformation_portal.depth.backends.registry import DepthBackendRegistry

    monkeypatch.setenv("TP_ALLOW_SYNTHETIC_FALLBACK", "1")
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    image_path = input_dir / "minimal.png"
    Image.new("RGB", (2, 2), color=(128, 128, 128)).save(image_path)
    prepared = prepare_lux_execution(
        EnhanceConfig(
            model_key="da3-metric",
            depth_device="cpu",
            enable_v2=False,
            emit_run_card=True,
            run_card_version="v2",
            enable_depth_cache=False,
            generate_pbr=False,
        ),
        input_dir,
        [image_path],
    )
    assert prepared.plan.candidate_fallback_chain == ("da3", "da2", "synthetic")

    real_registry = DepthBackendRegistry()

    fallback_help_url = "https://huggingface.co/docs/transformers"

    class UnavailableBackend:
        def __init__(self, name: str) -> None:
            self.name = name

        def ensure_available(self) -> None:
            raise ImportError(f"{self.name} unavailable; see {fallback_help_url}")

    class StartupFallbackRegistry:
        @staticmethod
        def get_backend(
            backend_id: str,
            config: EnhanceConfig,
            **kwargs: Any,
        ) -> Any:
            if backend_id in {"da3", "da2"}:
                return UnavailableBackend(backend_id)
            return real_registry.get_backend(backend_id, config, **kwargs)

    monkeypatch.setattr(orchestrator_module, "DepthBackendRegistry", StartupFallbackRegistry)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)

    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert results[0]["status"] == "ok"
    assert results[0]["backend"] == "synthetic"
    assert [attempt["backend"] for attempt in results[0]["attempts"]] == ["da3", "da2", "synthetic"]
    assert [attempt["status"] for attempt in results[0]["attempts"]] == ["failed", "failed", "success"]
    assert [attempt["attempt"] for attempt in results[0]["attempts"]] == [0, 1, 2]

    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    verified = verify_execution_evidence_file(
        evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    require_required_artifacts(verified)
    assert verified["executed_backend"] == "synthetic"
    assert verified["failed_artifacts"] == []
    assert verified["omitted_artifacts"] == []
    combined_outcome = next(
        outcome for outcome in verified["produced_artifacts"] if outcome["artifact_kind"] == "combined_manifest_json"
    )
    combined_payload = json.loads((output_root / combined_outcome["artifacts"][0]["path"]).read_bytes())
    assert combined_payload["depth"]["stats"]["attempts"] == combined_payload["backend_selection"]["attempts"]
    assert all(fallback_help_url in attempt["error_message"] for attempt in combined_payload["depth"]["stats"]["attempts"][:2])


@pytest.mark.regression
def test_prepared_batch_with_relative_output_root_emits_verifiable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    prepared, image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = Path("relative-output")
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)

    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert results[0]["status"] == "ok"
    evidence_path = next((tmp_path / output_root / "manifests").glob("execution_evidence_*.json"))
    verified = verify_execution_evidence_file(
        evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    require_required_artifacts(verified)
    assert verified["executed_backend"] == "synthetic"
    assert not verified["failed_artifacts"]


@pytest.mark.regression
def test_prepared_mixed_backend_fallback_round_trips_all_execution_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TP_ALLOW_SYNTHETIC_FALLBACK", raising=False)
    prepared = _prepared_many(
        tmp_path,
        count=2,
        depth_backend=None,
        model_key="da3-metric",
        allow_synthetic_fallback=False,
        emit_run_card=True,
    )
    assert prepared.plan.planned_backend == "da3"
    assert prepared.plan.candidate_fallback_chain == ("da3", "da2")
    output_root = tmp_path / "output"

    def fake_initialize_depth_backend(instance: EnhanceOrchestrator) -> None:
        instance._depth_registry = None
        instance.depth_backend = None
        instance._backend_init_errors = {}
        instance._depth_backend_cache = {}
        instance._backend_metadata = BackendSelectionMetadata(
            requested_backend="da3",
            resolved_backend="da3",
            resolution_status="success",
            resolution_reason="test runtime authority",
            model_id="depth-anything/DA3METRIC-LARGE",
            device="cpu",
            attempts=[],
        )
        instance._active_backend_metadata = instance._backend_metadata
        instance._active_depth_attempts = []
        instance._active_selected_attempt_index = None

    monkeypatch.setattr(EnhanceOrchestrator, "_initialize_depth_backend", fake_initialize_depth_backend)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    backend_by_name = {
        prepared.input_files[0].name: "da3",
        prepared.input_files[1].name: "da2",
    }

    def fake_enhance_image(image_input: Any, *, input_root: Path | None = None) -> dict[str, Any]:
        del input_root
        image_path = Path(image_input.path)
        backend = backend_by_name[image_path.name]
        input_id = orchestrator._prepared_input_id(image_path)
        depth_path = output_root / "depth" / f"{image_path.stem}_depth.png"
        depth_metadata_path = depth_path.with_name(f"{depth_path.stem}_metadata.json")
        combined_path = output_root / "manifests" / f"{image_path.stem}_combined.json"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.write_bytes(f"{backend}-depth".encode("ascii"))
        depth_metadata_path.write_text(dumps_json({"backend": backend}), encoding="utf-8")
        runtime_row = InputExecution(
            input_id=input_id,
            status="ok",
            executed_backend=backend,
        )
        evidence_relative_path = orchestrator._execution_evidence_relative_path(orchestrator._active_batch_id)
        if backend == "da3":
            attempts = [
                {
                    "attempt": 0,
                    "backend": "da3",
                    "status": "success",
                    "model_id": "depth-anything/DA3METRIC-LARGE",
                    "device": "cpu",
                }
            ]
            selected_attempt_index = 0
        else:
            attempts = [
                {
                    "attempt": 0,
                    "backend": "da3",
                    "status": "failed",
                    "failure_kind": "operational",
                    "error_code": "runtime_unavailable",
                    "error_message": "injected DA3 runtime failure",
                },
                {
                    "attempt": 1,
                    "backend": "da2",
                    "status": "success",
                    "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                    "device": "cpu",
                },
            ]
            selected_attempt_index = 1
        prepared_combined_path = orchestrator._prepared_manifest_write_path(combined_path)
        CombinedManifest(
            input=InputMetadata(image_path=str(image_path.resolve(strict=True))),
            depth=DepthMetadata(
                model=backend,
                depth_path=str(depth_path),
                runtime_seconds=0.01,
                scaling={},
            ),
            backend_selection=BackendSelectionMetadata(
                requested_backend="da3",
                resolved_backend=backend,
                resolution_status="success" if backend == "da3" else "fallback",
                resolution_reason=None if backend == "da3" else "DA3 runtime unavailable",
                model_id=(
                    "depth-anything/DA3METRIC-LARGE" if backend == "da3" else "depth-anything/Depth-Anything-V2-Small-hf"
                ),
                device="cpu",
                attempts=attempts,
            ),
            environment={
                "execution_contract": {
                    "authoritative_plan": prepared.plan.to_payload(),
                    "runtime": build_manifest_plan_projection(
                        prepared.plan,
                        input_executions=[runtime_row],
                        evidence_path=evidence_relative_path,
                    ),
                    "execution_evidence_path": evidence_relative_path,
                }
            },
            licensing=orchestrator._build_runtime_licensing_evidence(
                model_contract=None,
                backend_selection={"resolved": backend},
            ),
        ).save(prepared_combined_path)
        return {
            "status": "ok",
            "image": str(image_path),
            "backend": backend,
            "runtime_s": 0.01,
            "depth_path": str(depth_path),
            "depth_metadata_path": str(depth_metadata_path),
            "manifest": str(combined_path),
            "attempts": attempts,
            "selected_attempt_index": selected_attempt_index,
            "fallback_used": backend == "da2",
        }

    monkeypatch.setattr(orchestrator, "_enhance_image_from_active_batch", fake_enhance_image)
    results = orchestrator.enhance_batch(
        prepared.input_root,
        input_files=list(prepared.input_files),
    )

    assert [result["backend"] for result in results] == ["da3", "da2"]
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    batch_path = next((output_root / "manifests").glob("batch_*.json"))
    run_card_path = next(path for path in output_root.glob("run_card_*.json") if not path.name.endswith(".self.json"))
    verified = verify_execution_evidence_file(
        evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    require_required_artifacts(verified)
    assert verified["executed_backend"] is None
    assert [row["executed_backend"] for row in verified["requested_inputs"]] == ["da3", "da2"]
    assert not verified["failed_artifacts"]

    evidence_relative_path = evidence_path.relative_to(output_root).as_posix()
    batch = json.loads(batch_path.read_bytes())
    run_card = json.loads(run_card_path.read_bytes())
    assert "execution_evidence_path" not in batch
    assert "execution_plan" not in batch
    batch_contract = batch["config"]["execution_contract"]
    assert batch_contract["execution_evidence_path"] == evidence_relative_path
    assert batch_contract["runtime"]["executed_backend"] is None
    assert [row["executed_backend"] for row in batch_contract["runtime"]["executed_backend_by_input"]] == [
        "da3",
        "da2",
    ]
    assert "execution_evidence_path" not in run_card
    assert "execution_plan" not in run_card
    run_card_contract = run_card["effective_config"]["execution_contract"]
    assert run_card_contract["execution_evidence_path"] == evidence_relative_path
    assert run_card_contract["runtime"] == batch_contract["runtime"]
    assert run_card["backend_summary"]["final_backends_used"] == ["da3", "da2"]
    assert run_card["backend_summary"]["fallback_images"] == 1

    for result, expected_backend in zip(results, ("da3", "da2")):
        combined = json.loads(Path(result["manifest"]).read_bytes())
        assert "execution_evidence_path" not in combined
        assert "execution_plan" not in combined
        combined_contract = combined["environment"]["execution_contract"]
        assert combined_contract["execution_evidence_path"] == evidence_relative_path
        assert combined_contract["runtime"]["executed_backend"] == expected_backend

    run_card["backend_summary"]["final_backends_used"] = ["da3"]
    run_card["licensing"] = orchestrator._build_runtime_licensing_evidence(
        model_contract=run_card.get("model_contract"),
        backend_selection=run_card["backend_selection"],
        backend_ids=("da3",),
    )
    _rewrite_run_card_with_valid_self_integrity(run_card_path, run_card)
    forged_payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path=evidence_relative_path,
        input_executions=orchestrator._execution_input_rows(results),
        artifact_observations=orchestrator._execution_artifact_observations(
            results,
            batch_manifest_path=batch_path,
            run_card_path=run_card_path,
        ),
    )

    run_card_failure = next(item for item in forged_payload["failed_artifacts"] if item["artifact_kind"] == "run_card")
    assert run_card_failure["reason_code"] == "invalid_manifest_binding"


def test_execution_evidence_rejects_bound_run_card_with_invalid_datetime_format(
    tmp_path: Path,
) -> None:
    prepared, orchestrator, results, output_root, evidence_path, batch_path, run_card_path = _completed_prepared_run_with_card(
        tmp_path
    )
    run_card = json.loads(run_card_path.read_bytes())
    run_card["start_time"] = "not-a-date-time"
    _rewrite_run_card_with_valid_self_integrity(run_card_path, run_card)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path=evidence_path.relative_to(output_root).as_posix(),
        input_executions=orchestrator._execution_input_rows(results),
        artifact_observations=orchestrator._execution_artifact_observations(
            results,
            batch_manifest_path=batch_path,
            run_card_path=run_card_path,
        ),
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "run_card")
    assert failure["reason_code"] == "invalid_manifest_binding"


@pytest.mark.parametrize("tamper", ["canonical_payload", "final_sidecar"])
def test_execution_evidence_construction_rejects_run_card_self_integrity_drift(
    tmp_path: Path,
    tamper: str,
) -> None:
    prepared, orchestrator, results, output_root, evidence_path, batch_path, run_card_path = _completed_prepared_run_with_card(
        tmp_path
    )
    run_card = json.loads(run_card_path.read_bytes())
    sidecar_path = run_card_path.with_suffix(".self.json")
    sidecar = json.loads(sidecar_path.read_bytes())
    if tamper == "canonical_payload":
        run_card["run_card_integrity"]["canonical_payload_sha256"] = "0" * 64
        run_card_bytes = dumps_json(
            run_card,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        run_card_path.write_bytes(run_card_bytes)
        sidecar["final_run_card_sha256"] = hashlib.sha256(run_card_bytes).hexdigest()
    else:
        sidecar["final_run_card_sha256"] = "0" * 64
    sidecar_path.write_bytes(
        dumps_json(
            sidecar,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path=evidence_path.relative_to(output_root).as_posix(),
        input_executions=orchestrator._execution_input_rows(results),
        artifact_observations=orchestrator._execution_artifact_observations(
            results,
            batch_manifest_path=batch_path,
            run_card_path=run_card_path,
        ),
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "run_card")
    assert failure["reason_code"] == "invalid_manifest_binding"


@pytest.mark.parametrize("tamper", ["canonical_payload", "final_sidecar"])
def test_execution_evidence_verifier_rejects_run_card_self_integrity_drift(
    tmp_path: Path,
    tamper: str,
) -> None:
    prepared, _orchestrator, _results, output_root, evidence_path, _batch_path, run_card_path = (
        _completed_prepared_run_with_card(tmp_path)
    )
    evidence = json.loads(evidence_path.read_bytes())
    run_card = json.loads(run_card_path.read_bytes())
    sidecar_path = run_card_path.with_suffix(".self.json")
    sidecar = json.loads(sidecar_path.read_bytes())
    if tamper == "canonical_payload":
        run_card["run_card_integrity"]["canonical_payload_sha256"] = "0" * 64
        run_card_bytes = dumps_json(
            run_card,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        run_card_path.write_bytes(run_card_bytes)
        sidecar["final_run_card_sha256"] = hashlib.sha256(run_card_bytes).hexdigest()
        run_card_outcome = next(item for item in evidence["produced_artifacts"] if item["artifact_kind"] == "run_card")
        run_card_record = run_card_outcome["artifacts"][0]
        run_card_record["sha256"] = hashlib.sha256(run_card_bytes).hexdigest()
        run_card_record["size_bytes"] = len(run_card_bytes)
        evidence["evidence_fingerprint_sha256"] = compute_execution_evidence_fingerprint(evidence)
        write_execution_evidence(
            evidence_path,
            evidence,
            output_root=output_root,
            plan=prepared.plan,
        )
    else:
        sidecar["final_run_card_sha256"] = "0" * 64
    sidecar_path.write_bytes(
        dumps_json(
            sidecar,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )

    with pytest.raises(ArtifactEvidenceError, match="self-integrity|canonical_payload"):
        verify_execution_evidence_file(
            evidence_path,
            output_root=output_root,
            plan=prepared.plan,
        )


@pytest.mark.skipif(
    sys.platform != "darwin" or Path("/tmp").resolve() == Path("/tmp"),
    reason="requires the macOS /tmp to /private/tmp alias",
)
def test_execution_evidence_canonicalizes_macos_tmp_alias_without_weakening_confinement() -> None:
    with tempfile.TemporaryDirectory(prefix="tp-evidence-alias-", dir="/tmp") as temporary:
        base = Path(temporary)
        prepared, _ = _prepared(base)
        lexical_output_root = base / "output"
        observations = _required_observations(prepared, lexical_output_root)
        canonical_output_root = lexical_output_root.resolve(strict=True)

        payload = build_execution_evidence(
            prepared.plan,
            output_root=canonical_output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=observations,
        )

        depth = next(item for item in payload["produced_artifacts"] if item["artifact_kind"] == "depth_u16_png")
        assert depth["artifacts"][0]["path"] == "depth/sample_depth.png"


@pytest.mark.skipif(
    sys.platform != "darwin" or Path("/var").resolve() == Path("/var"),
    reason="requires the macOS /var to /private/var alias",
)
def test_execution_evidence_canonicalizes_macos_var_alias_without_weakening_confinement() -> None:
    canonical_temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    try:
        temp_suffix = canonical_temp_root.relative_to("/private/var")
    except ValueError:
        pytest.skip("the configured temporary directory is not under /private/var")
    lexical_temp_root = Path("/var") / temp_suffix
    with tempfile.TemporaryDirectory(prefix="tp-evidence-var-alias-", dir=lexical_temp_root) as temporary:
        base = Path(temporary)
        prepared, _ = _prepared(base)
        lexical_output_root = base / "output"
        observations = _required_observations(prepared, lexical_output_root)
        canonical_output_root = lexical_output_root.resolve(strict=True)

        payload = build_execution_evidence(
            prepared.plan,
            output_root=canonical_output_root,
            evidence_path="manifests/execution_evidence_test.json",
            input_executions=_successful_input(prepared),
            artifact_observations=observations,
        )

        depth = next(item for item in payload["produced_artifacts"] if item["artifact_kind"] == "depth_u16_png")
        assert depth["artifacts"][0]["path"] == "depth/sample_depth.png"


def test_prepared_backend_metadata_fallback_uses_only_authorized_planned_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    prepared, _ = _prepared(tmp_path)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")
    del orchestrator._backend_metadata

    metadata = orchestrator._capture_backend_metadata()

    assert metadata.requested_backend == "synthetic"
    assert metadata.resolved_backend == "synthetic"
    assert metadata.model_id == "synthetic/depth-analytic-v1"

    def reject_candidate(*args: Any, **kwargs: Any) -> None:
        raise LuxExecutionPlanAuthorityError("planned candidate cannot be authorized")

    monkeypatch.setattr(orchestrator_module, "backend_candidate_authority", reject_candidate)
    with pytest.raises(LuxExecutionPlanAuthorityError, match="cannot be authorized"):
        orchestrator._capture_backend_metadata()


@pytest.mark.parametrize("drift", ["revision", "license", "artifact"])
def test_nested_execution_contract_rejects_immutable_plan_authority_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    overrides: dict[str, Any]
    if drift == "artifact":
        overrides = {
            "depth_backend": "depth_pro",
            "non_commercial_ok": True,
            "accept_apple_depth_pro_research_license": True,
        }
    else:
        overrides = {"depth_backend": None, "model_key": "da3-metric"}
    prepared, _ = _prepared(tmp_path, **overrides)
    evidence_path = "manifests/execution_evidence_test.json"
    rows = _successful_input(prepared)
    runtime = build_manifest_plan_projection(
        prepared.plan,
        input_executions=rows,
        evidence_path=evidence_path,
    )
    authority = prepared.plan.to_payload()
    model_contract = authority["backend_candidates"][0]["model_contracts"][0]
    if drift == "revision":
        model_contract["model"]["revision"] = "0" * 40
    elif drift == "license":
        model_contract["model"]["license_id"] = "forged-license"
    else:
        model_contract["artifact_sha256"] = "0" * 64
    payload = {
        "environment": {
            "execution_contract": {
                "authoritative_plan": authority,
                "runtime": runtime,
                "execution_evidence_path": evidence_path,
            }
        }
    }

    with pytest.raises(ArtifactEvidenceError, match="authoritative_plan"):
        evidence_module._validate_manifest_execution_contract(
            payload,
            artifact_kind="combined_manifest_json",
            expected_projection=runtime,
            authoritative_plan=prepared.plan.to_payload(),
        )


def test_public_evidence_build_rejects_nested_candidate_revision_drift(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path, depth_backend=None, model_key="da3-metric")
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    combined_payload = json.loads(combined_observation.path.read_bytes())
    combined_payload["environment"]["execution_contract"]["authoritative_plan"]["backend_candidates"][0]["model_contracts"][0][
        "model"
    ]["revision"] = ("0" * 40)
    combined_observation.path.write_text(
        dumps_json(combined_payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "combined_manifest_json")
    assert failure["reason_code"] == "invalid_manifest_binding"


@pytest.mark.parametrize("drift", ["runtime", "path", "legacy"])
def test_nested_execution_contract_rejects_conflicting_runtime_or_pointer(
    tmp_path: Path,
    drift: str,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path)
    evidence_path = "manifests/execution_evidence_test.json"
    runtime = build_manifest_plan_projection(
        prepared.plan,
        input_executions=_successful_input(prepared),
        evidence_path=evidence_path,
    )
    contract = {
        "authoritative_plan": prepared.plan.to_payload(),
        "runtime": json.loads(json.dumps(runtime)),
        "execution_evidence_path": evidence_path,
    }
    payload: dict[str, Any] = {"config": {"execution_contract": contract}}
    if drift == "runtime":
        contract["runtime"]["executed_backend"] = "da3"
    elif drift == "path":
        contract["execution_evidence_path"] = "manifests/other.json"
    else:
        payload["execution_plan"] = runtime
        payload["execution_evidence_path"] = "manifests/other.json"

    with pytest.raises(ArtifactEvidenceError, match="runtime|sidecar|conflicting"):
        evidence_module._validate_manifest_execution_contract(
            payload,
            artifact_kind="batch_manifest_json",
            expected_projection=runtime,
            authoritative_plan=prepared.plan.to_payload(),
        )


def _exact_candidate_runtime_payloads(prepared: PreparedLuxExecution) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = prepared.plan.backend_candidates[0]
    contract = next(item for item in candidate.model_contracts if item.enabled)
    model = contract.model
    model_id = model.repo_id or model.canonical_key
    attempt: dict[str, Any] = {
        "backend": candidate.backend_id,
        "status": "success",
        "model_id": model_id,
        "device": contract.device,
        "revision": model.revision,
        "license_id": model.license_id,
    }
    if contract.artifact_path is not None:
        attempt["model_artifact_filename"] = Path(contract.artifact_path).name
        attempt["model_artifact_sha256"] = contract.artifact_sha256
    licensing = _runtime_licensing(prepared, candidate.backend_id)
    combined = {
        "backend_selection": {
            "resolved_backend": candidate.backend_id,
            "model_id": model_id,
            "device": contract.device,
            "attempts": [attempt],
        },
        "depth": {
            "model": candidate.backend_id,
            "stats": {"backend": candidate.backend_id, "attempts": [attempt]},
        },
        "licensing": json.loads(json.dumps(licensing)),
    }
    model_contract: dict[str, Any] = {
        "requested_model_selector": model.requested_selector,
        "resolution_reason": model.resolution_reason,
        "canonical_model_key": model.canonical_key,
        "resolved_repo_id": model.repo_id,
        "resolved_revision": model.revision,
        "license_id": model.license_id,
        "usage_class": model.usage_class,
        "requires_non_commercial_ok": model.requires_non_commercial_ok,
        "backend_kind": contract.backend_id,
        "accelerator_kind": model.accelerator_kind,
        "non_commercial_ok": prepared.plan.license_acknowledgements.non_commercial_ok,
    }
    if contract.artifact_path is not None:
        model_contract["model_artifact_filename"] = Path(contract.artifact_path).name
        model_contract["model_artifact_sha256"] = contract.artifact_sha256
    run_card = {
        "backend_selection": {
            "resolved": candidate.backend_id,
            "model_id": model_id,
            "device": contract.device,
            "model_artifact_filename": (Path(contract.artifact_path).name if contract.artifact_path is not None else None),
            "model_artifact_sha256": contract.artifact_sha256,
        },
        "model_contract": model_contract,
        "backend_summary": {"final_backends_used": [candidate.backend_id]},
        "licensing": json.loads(json.dumps(licensing)),
    }
    return combined, run_card


def test_backend_attempts_must_follow_the_exact_ordered_candidate_prefix(tmp_path: Path) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(
        tmp_path,
        depth_backend=None,
        model_key="da3-metric",
        allow_synthetic_fallback=False,
    )
    assert prepared.plan.candidate_fallback_chain[:2] == ("da3", "da2")

    def attempt(backend_id: str, status: str) -> dict[str, Any]:
        candidate = next(item for item in prepared.plan.backend_candidates if item.backend_id == backend_id)
        contract = next(item for item in candidate.model_contracts if item.enabled)
        row: dict[str, Any] = {
            "backend": backend_id,
            "status": status,
            "model_id": contract.model.repo_id or contract.model.canonical_key,
            "device": contract.device,
        }
        if contract.artifact_path is not None:
            row["model_artifact_filename"] = Path(contract.artifact_path).name
            row["model_artifact_sha256"] = contract.artifact_sha256
        return row

    valid = [attempt("da3", "failed"), attempt("da2", "success")]
    evidence_module._validate_attempt_claims(
        valid,
        plan=prepared.plan,
        selected_backend="da2",
    )

    invalid_histories = (
        [valid[1], valid[0]],
        [valid[0], attempt("da3", "success")],
        [valid[1]],
        [attempt("da3", "started")],
        [attempt("da3", "success"), attempt("da2", "failed")],
    )
    for history in invalid_histories:
        with pytest.raises(ArtifactEvidenceError, match="attempt|candidate|terminal|status|succeed"):
            evidence_module._validate_attempt_claims(
                history,
                plan=prepared.plan,
                selected_backend="da2",
            )


@pytest.mark.parametrize(
    "surface,field",
    [
        ("combined", "model_id"),
        ("depth", "model"),
        ("attempt", "revision"),
        ("attempt", "license_id"),
        ("run_card", "resolved_revision"),
        ("run_card", "license_id"),
    ],
)
def test_runtime_model_claims_are_bound_directly_to_prepared_candidate(
    tmp_path: Path,
    surface: str,
    field: str,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(tmp_path, depth_backend=None, model_key="da3-metric")
    combined, run_card = _exact_candidate_runtime_payloads(prepared)
    if surface == "combined":
        combined["backend_selection"][field] = "forged/model"
    elif surface == "depth":
        combined["depth"][field] = "da2"
    elif surface == "attempt":
        combined["backend_selection"]["attempts"][0][field] = "forged"
        combined["depth"]["stats"]["attempts"] = combined["backend_selection"]["attempts"]
    else:
        run_card["model_contract"][field] = "forged"

    with pytest.raises(ArtifactEvidenceError, match="model|candidate|license|revision"):
        if surface == "run_card":
            evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)
        else:
            evidence_module._validate_combined_runtime_model_binding(
                combined,
                plan=prepared.plan,
                selected_backend="da3",
            )


def test_runtime_model_binding_uses_no_model_reresolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module
    import transformation_portal.lux_depth_v3.execution_lifecycle as lifecycle_module

    prepared, _ = _prepared(tmp_path, depth_backend=None, model_key="da3-metric")
    combined, run_card = _exact_candidate_runtime_payloads(prepared)
    monkeypatch.setattr(
        lifecycle_module,
        "backend_candidate_authority",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not re-resolve")),
    )

    evidence_module._validate_combined_runtime_model_binding(
        combined,
        plan=prepared.plan,
        selected_backend="da3",
    )
    evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)


def test_ensemble_run_card_uses_nested_plan_for_frozen_aggregate_model_shape(tmp_path: Path) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(
        tmp_path,
        depth_backend="ensemble",
        model_key="da3-metric",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        accept_research_tools_license=True,
    )
    candidate = prepared.plan.backend_candidates[0]
    contracts = tuple(item for item in candidate.model_contracts if item.enabled)
    assert len(contracts) == 2
    run_card = {
        "backend_selection": {
            "resolved": "ensemble",
            "model_id": "ensemble/multi-backend",
            "device": "cpu",
        },
        "backend_summary": {"final_backends_used": ["ensemble"]},
        "licensing": _runtime_licensing(prepared, "ensemble"),
    }
    expected_models = list(run_card["licensing"]["models"])
    run_card["licensing"]["models"] = []

    with pytest.raises(ArtifactEvidenceError, match="does not cover prepared model authority"):
        evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)

    run_card["licensing"]["models"] = expected_models
    evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)

    run_card["licensing"]["models"] = run_card["licensing"]["models"][:1]
    with pytest.raises(ArtifactEvidenceError, match="does not cover prepared model authority"):
        evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)


def test_runtime_licensing_aggregate_is_bound_to_prepared_acknowledgements(tmp_path: Path) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(
        tmp_path,
        depth_backend="depth_pro",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
    )
    combined, run_card = _exact_candidate_runtime_payloads(prepared)
    combined["licensing"]["non_commercial_active"] = False
    run_card["licensing"]["research_acknowledgement_required"] = False

    with pytest.raises(ArtifactEvidenceError, match="aggregate"):
        evidence_module._validate_combined_runtime_model_binding(
            combined,
            plan=prepared.plan,
            selected_backend="depth_pro",
        )
    with pytest.raises(ArtifactEvidenceError, match="aggregate"):
        evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)


def test_mixed_backend_run_card_licensing_covers_every_executed_candidate(tmp_path: Path) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(
        tmp_path,
        depth_backend=None,
        model_key="da3-metric",
        allow_synthetic_fallback=False,
    )
    assert prepared.plan.candidate_fallback_chain[:2] == ("da3", "da2")
    _combined, run_card = _exact_candidate_runtime_payloads(prepared)
    run_card["backend_summary"]["final_backends_used"] = ["da3", "da2"]
    run_card["licensing"] = _runtime_licensing(prepared, "da3", "da2")

    evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)
    run_card["licensing"]["models"] = run_card["licensing"]["models"][:1]
    with pytest.raises(ArtifactEvidenceError, match="does not cover prepared model authority"):
        evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)


def test_prepared_ensemble_licensing_producer_emits_every_enabled_constituent(tmp_path: Path) -> None:
    prepared, _ = _prepared(
        tmp_path,
        depth_backend="ensemble",
        model_key="da3-metric",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        accept_research_tools_license=True,
    )
    candidate = prepared.plan.backend_candidates[0]
    contracts = tuple(item for item in candidate.model_contracts if item.enabled)
    assert len(contracts) == 2
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.config = prepared.runtime_config

    licensing = orchestrator._build_runtime_licensing_evidence(
        model_contract=None,
        backend_selection={"resolved": "ensemble"},
    )

    assert [model["id"] for model in licensing["models"]] == [
        contract.model.repo_id or contract.model.canonical_key for contract in contracts
    ]
    assert [model["runtime_role"] for model in licensing["models"]] == [contract.backend_id for contract in contracts]


def test_ensemble_combined_manifest_binds_through_public_evidence_build(tmp_path: Path) -> None:
    prepared, _ = _prepared(
        tmp_path,
        depth_backend="ensemble",
        model_key="da3-metric",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        accept_research_tools_license=True,
    )
    output_root = tmp_path / "output"

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )

    assert not payload["failed_artifacts"]
    assert payload["requested_inputs"][0]["executed_backend"] == "ensemble"


@pytest.mark.parametrize("surface", ["combined", "attempt", "run_card_contract", "run_card_selection"])
def test_runtime_checkpoint_identity_is_bound_to_prepared_candidate(
    tmp_path: Path,
    surface: str,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    prepared, _ = _prepared(
        tmp_path,
        depth_backend="depth_pro",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
    )
    combined, run_card = _exact_candidate_runtime_payloads(prepared)
    if surface == "combined":
        combined["backend_selection"]["model_artifact_sha256"] = "0" * 64
    elif surface == "attempt":
        combined["backend_selection"]["attempts"][0]["model_artifact_sha256"] = "0" * 64
        combined["depth"]["stats"]["attempts"] = combined["backend_selection"]["attempts"]
    elif surface == "run_card_contract":
        run_card["model_contract"]["model_artifact_sha256"] = "0" * 64
    else:
        run_card["backend_selection"]["model_artifact_filename"] = "other.pt"

    with pytest.raises(ArtifactEvidenceError, match="artifact"):
        if surface.startswith("run_card"):
            evidence_module._validate_run_card_runtime_model_binding(run_card, plan=prepared.plan)
        else:
            evidence_module._validate_combined_runtime_model_binding(
                combined,
                plan=prepared.plan,
                selected_backend="depth_pro",
            )


@pytest.mark.parametrize("status", ["error", "failed", "skipped", ""])
def test_stale_v2_paths_cannot_satisfy_current_execution_output(
    tmp_path: Path,
    status: str,
) -> None:
    prepared, _ = _prepared(tmp_path, enable_v2=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    input_id = prepared.plan.inputs[0].input_id
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    stale_path = output_root / "v2/stale.png"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_bytes(b"stale-v2")
    combined = CombinedManifest.load(combined_observation.path)
    combined.v2 = V2Metadata(preset="standard", status=status, output_paths=[str(stale_path)])
    combined.save(combined_observation.path)
    observations.append(ArtifactObservation("v2_enhanced_image", stale_path, input_id))

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "v2_enhanced_image")
    assert failure["reason_code"] == "artifact_input_mismatch"


def test_normalized_v2_success_can_satisfy_current_execution_output(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path, enable_v2=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    input_id = prepared.plan.inputs[0].input_id
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    current_path = output_root / "v2/current.png"
    current_path.parent.mkdir(parents=True)
    current_path.write_bytes(b"current-v2")
    combined = CombinedManifest.load(combined_observation.path)
    combined.v2 = V2Metadata(preset="standard", status="SUCCESS", output_paths=[str(current_path)])
    combined.save(combined_observation.path)
    observations.append(ArtifactObservation("v2_enhanced_image", current_path, input_id))

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    outcome = next(item for item in payload["produced_artifacts"] if item["artifact_kind"] == "v2_enhanced_image")
    assert outcome["artifacts"][0]["path"] == "v2/current.png"


@pytest.mark.parametrize(
    "enabled,status,errors",
    [
        (False, "ok", []),
        (True, "failed", ["segmentation failed"]),
        (True, None, ["stale failure"]),
    ],
)
def test_stale_material_mask_paths_cannot_satisfy_current_execution_output(
    tmp_path: Path,
    enabled: bool,
    status: str | None,
    errors: list[str],
) -> None:
    prepared, _ = _prepared(
        tmp_path,
        enable_materials_v3=True,
        enable_material_segmentation=True,
    )
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    input_id = prepared.plan.inputs[0].input_id
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    stale_path = output_root / "segmentation/stale.npz"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_bytes(b"stale-mask")
    segmentation_metadata: dict[str, Any] = {
        "mask_artifact_path": str(stale_path),
        "errors": errors,
    }
    if status is not None:
        segmentation_metadata["status"] = status
    combined = CombinedManifest.load(combined_observation.path)
    combined.materials_v3 = MaterialsV3Metadata(
        enabled=enabled,
        segmentation_metadata=segmentation_metadata,
    )
    combined.save(combined_observation.path)
    observations.append(ArtifactObservation("materials_v3_masks", stale_path, input_id))

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "materials_v3_masks")
    assert failure["reason_code"] == "artifact_input_mismatch"


def test_partial_pbr_path_set_cannot_satisfy_bundle_output(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path, generate_pbr=True)
    output_root = tmp_path / "output"
    observations = _required_observations(prepared, output_root)
    input_id = prepared.plan.inputs[0].input_id
    combined_observation = next(item for item in observations if item.artifact_kind == "combined_manifest_json")
    assert combined_observation.path is not None
    normal_path = output_root / "pbr/normal.png"
    normal_path.parent.mkdir(parents=True)
    normal_path.write_bytes(b"partial-pbr")
    combined = CombinedManifest.load(combined_observation.path)
    combined.pbr_assets = {"normal_path": str(normal_path)}
    combined.save(combined_observation.path)
    observations.append(ArtifactObservation("pbr_maps", normal_path, input_id))

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=observations,
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "pbr_maps")
    assert failure["reason_code"] == "artifact_input_mismatch"


def test_execution_evidence_schema_error_rendering_is_bounded(tmp_path: Path) -> None:
    prepared, _ = _prepared(tmp_path)
    output_root = tmp_path / "output"
    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path="manifests/execution_evidence_test.json",
        input_executions=_successful_input(prepared),
        artifact_observations=_required_observations(prepared, output_root),
    )
    payload["x" * 100_000] = "untrusted additional property"

    with pytest.raises(ExecutionEvidenceError) as exc_info:
        validate_execution_evidence_payload(payload, plan=prepared.plan)

    assert "[truncated]" in str(exc_info.value)
    assert len(str(exc_info.value)) < 2_300


def test_cumulative_manifest_decode_budget_rejects_before_second_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    output_root = tmp_path / "output"
    output_root.mkdir()
    first = output_root / "first.json"
    second = output_root / "second.json"
    first.write_bytes(b"{}")
    second.write_bytes(b"{}")
    monkeypatch.setattr(evidence_module, "_MAX_CUMULATIVE_DECODED_MANIFEST_BYTES", 2)
    budget = evidence_module._ManifestBudget()
    with evidence_module._pin_output_root(output_root) as root:
        evidence_module._read_confined_artifact_bytes(
            root,
            first,
            context="first manifest",
            max_bytes=2,
            manifest_budget=budget,
        )
        real_read = evidence_module.os.read
        second_reads = 0

        def tracking_read(descriptor: int, size: int) -> bytes:
            nonlocal second_reads
            second_reads += 1
            return real_read(descriptor, size)

        monkeypatch.setattr(evidence_module.os, "read", tracking_read)
        with pytest.raises(ArtifactEvidenceError) as exc_info:
            evidence_module._read_confined_artifact_bytes(
                root,
                second,
                context="second manifest",
                max_bytes=2,
                manifest_budget=budget,
            )

    assert exc_info.value.code == "aggregate_artifact_bytes_exceeded"
    assert second_reads == 0


def test_combined_manifest_retains_only_bounded_path_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.execution_evidence as evidence_module

    giant_unrelated_value = "x" * 1_000_000
    payload = {
        "depth": {"depth_path": "/output/depth/example.png"},
        "environment": {"unrelated": giant_unrelated_value},
    }
    budget = evidence_module._ManifestBudget()
    authority = evidence_module._compact_combined_manifest_authority(payload, manifest_budget=budget)

    assert authority.declared_paths("depth_u16_png") == ("/output/depth/example.png",)
    assert budget.retained_bytes < 2_000
    assert not hasattr(authority, "environment")

    monkeypatch.setattr(evidence_module, "_MAX_RETAINED_MANIFEST_AUTHORITY_BYTES", budget.retained_bytes - 1)
    with pytest.raises(ArtifactEvidenceError) as exc_info:
        evidence_module._compact_combined_manifest_authority(
            payload,
            manifest_budget=evidence_module._ManifestBudget(),
        )
    assert exc_info.value.code == "aggregate_artifact_bytes_exceeded"


@pytest.mark.parametrize("run_card_version", ["v1", "v2"])
def test_nested_prepared_contract_preserves_frozen_manifest_and_run_card_readers(
    tmp_path: Path,
    run_card_version: str,
) -> None:
    from transformation_portal.lux_depth_v3.validators.run_card_validator import validate_run_card_payload

    prepared, image_path = _prepared(
        tmp_path,
        emit_run_card=True,
        run_card_version=run_card_version,
    )
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    batch_path = next((output_root / "manifests").glob("batch_*.json"))
    combined_path = Path(results[0]["manifest"])
    run_card_path = next(path for path in output_root.glob("run_card_*.json") if not path.name.endswith(".self.json"))

    batch_payload = json.loads(batch_path.read_bytes())
    combined_payload = json.loads(combined_path.read_bytes())
    run_card_payload = json.loads(run_card_path.read_bytes())
    validate_run_card_payload(run_card_payload, schema_version=run_card_version)

    assert BatchManifest.load(batch_path).config["execution_contract"]["authoritative_plan"] == prepared.plan.to_payload()
    assert (
        CombinedManifest.load(combined_path).environment["execution_contract"]["authoritative_plan"]
        == prepared.plan.to_payload()
    )
    for payload in (batch_payload, combined_payload, run_card_payload):
        assert "execution_plan" not in payload
        assert "execution_evidence_path" not in payload
    expected_contract_fields = {
        "authoritative_plan",
        "runtime",
        "execution_evidence_path",
        "artifact_outcome_authority",
        "requested_artifacts",
        "produced_artifacts",
        "omitted_artifacts",
        "failed_artifacts",
    }
    assert set(batch_payload["config"]["execution_contract"]) == expected_contract_fields
    assert set(combined_payload["environment"]["execution_contract"]) == expected_contract_fields
    assert set(run_card_payload["effective_config"]["execution_contract"]) == expected_contract_fields


@pytest.mark.parametrize("tamper", ["backend_summary", "config_fingerprint"])
def test_bound_run_card_reuses_non_filesystem_semantic_validation(
    tmp_path: Path,
    tamper: str,
) -> None:
    prepared, orchestrator, results, output_root, evidence_path, batch_path, run_card_path = _completed_prepared_run_with_card(
        tmp_path
    )
    run_card = json.loads(run_card_path.read_bytes())
    if tamper == "backend_summary":
        run_card["backend_summary"]["primary_backend"] = "da3"
    else:
        canonical = run_card["config_fingerprint"]["canonical_json"] + " "
        run_card["config_fingerprint"]["canonical_json"] = canonical
        run_card["config_fingerprint"]["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _rewrite_run_card_with_valid_self_integrity(run_card_path, run_card)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path=evidence_path.relative_to(output_root).as_posix(),
        input_executions=orchestrator._execution_input_rows(results),
        artifact_observations=orchestrator._execution_artifact_observations(
            results,
            batch_manifest_path=batch_path,
            run_card_path=run_card_path,
        ),
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "run_card")
    assert failure["reason_code"] == "invalid_manifest_binding"


def test_build_cross_binds_batch_and_run_card_ids(tmp_path: Path) -> None:
    prepared, orchestrator, results, output_root, evidence_path, batch_path, run_card_path = _completed_prepared_run_with_card(
        tmp_path
    )
    run_card = json.loads(run_card_path.read_bytes())
    run_card["batch_id"] = "different-batch"
    _rewrite_run_card_with_valid_self_integrity(run_card_path, run_card)

    payload = build_execution_evidence(
        prepared.plan,
        output_root=output_root,
        evidence_path=evidence_path.relative_to(output_root).as_posix(),
        input_executions=orchestrator._execution_input_rows(results),
        artifact_observations=orchestrator._execution_artifact_observations(
            results,
            batch_manifest_path=batch_path,
            run_card_path=run_card_path,
        ),
    )

    failure = next(item for item in payload["failed_artifacts"] if item["artifact_kind"] == "run_card")
    assert failure["reason_code"] == "invalid_manifest_binding"


def test_verifier_cross_binds_batch_and_run_card_ids(tmp_path: Path) -> None:
    prepared, _orchestrator, _results, output_root, evidence_path, _batch_path, run_card_path = (
        _completed_prepared_run_with_card(tmp_path)
    )
    evidence = json.loads(evidence_path.read_bytes())
    run_card = json.loads(run_card_path.read_bytes())
    run_card["batch_id"] = "different-batch"
    _rewrite_run_card_with_valid_self_integrity(run_card_path, run_card)
    _refresh_evidence_artifact_record(
        evidence,
        artifact_kind="run_card",
        artifact_path=run_card_path,
    )
    write_execution_evidence(
        evidence_path,
        evidence,
        output_root=output_root,
        plan=prepared.plan,
    )

    with pytest.raises(ArtifactEvidenceError, match="batch_id values do not match"):
        verify_execution_evidence_file(
            evidence_path,
            output_root=output_root,
            plan=prepared.plan,
        )


def test_verifier_rejects_all_canonical_carriers_with_stripped_outcome_projections(
    tmp_path: Path,
) -> None:
    prepared, _orchestrator, _results, output_root, evidence_path, batch_path, run_card_path = (
        _completed_prepared_run_with_card(tmp_path)
    )
    evidence = json.loads(evidence_path.read_bytes())
    combined_outcome = next(
        item for item in evidence["produced_artifacts"] if item["artifact_kind"] == "combined_manifest_json"
    )
    combined_path = output_root / combined_outcome["artifacts"][0]["path"]
    carriers = (
        (combined_path, "environment"),
        (batch_path, "config"),
    )
    outcome_fields = (
        "artifact_outcome_authority",
        "requested_artifacts",
        "produced_artifacts",
        "omitted_artifacts",
        "failed_artifacts",
    )
    for carrier_path, container_name in carriers:
        carrier = json.loads(carrier_path.read_bytes())
        contract = carrier[container_name]["execution_contract"]
        for field_name in outcome_fields:
            contract.pop(field_name)
        carrier_path.write_text(
            dumps_json(carrier, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    run_card = json.loads(run_card_path.read_bytes())
    run_card_contract = run_card["effective_config"]["execution_contract"]
    for field_name in outcome_fields:
        run_card_contract.pop(field_name)
    run_card_path.write_text(
        dumps_json(run_card, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _refresh_run_card_index_entries(run_card_path, [combined_path, batch_path])
    for artifact_kind, artifact_path in (
        ("combined_manifest_json", combined_path),
        ("batch_manifest_json", batch_path),
        ("run_card", run_card_path),
    ):
        _refresh_evidence_artifact_record(
            evidence,
            artifact_kind=artifact_kind,
            artifact_path=artifact_path,
        )
    write_execution_evidence(evidence_path, evidence, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ArtifactEvidenceError, match="missing its execution outcome projection"):
        verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)


def test_verifier_rejects_no_card_carriers_downgraded_to_legacy_projections(tmp_path: Path) -> None:
    prepared, image_path = _prepared(tmp_path, emit_run_card=False)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    batch_path = next((output_root / "manifests").glob("batch_*.json"))
    evidence = json.loads(evidence_path.read_bytes())
    combined_outcome = next(
        item for item in evidence["produced_artifacts"] if item["artifact_kind"] == "combined_manifest_json"
    )
    combined_path = output_root / combined_outcome["artifacts"][0]["path"]

    for carrier_path, container_name in ((combined_path, "environment"), (batch_path, "config")):
        carrier = json.loads(carrier_path.read_bytes())
        contract = carrier[container_name].pop("execution_contract")
        carrier["execution_plan"] = contract["runtime"]
        carrier["execution_evidence_path"] = contract["execution_evidence_path"]
        carrier_path.write_text(
            dumps_json(carrier, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        _refresh_evidence_artifact_record(
            evidence,
            artifact_kind=("combined_manifest_json" if carrier_path == combined_path else "batch_manifest_json"),
            artifact_path=carrier_path,
        )
    write_execution_evidence(evidence_path, evidence, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ArtifactEvidenceError, match="missing its execution outcome projection"):
        verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)


@pytest.mark.parametrize("run_card_version", ["v1", "v2"])
def test_verifier_rejects_self_consistent_stale_run_card_artifact_commitment(
    tmp_path: Path,
    run_card_version: str,
) -> None:
    from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
    from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree

    prepared, _orchestrator, _results, output_root, evidence_path, batch_path, run_card_path = (
        _completed_prepared_run_with_card(tmp_path, run_card_version=run_card_version)
    )
    evidence = json.loads(evidence_path.read_bytes())
    run_card = json.loads(run_card_path.read_bytes())
    batch_relative_path = batch_path.relative_to(output_root).as_posix()
    batch_entry = next(entry for entry in run_card["artifact_index"] if entry["relative_path"] == batch_relative_path)
    batch_entry["sha256"] = "0" * 64
    if run_card_version == "v2":
        include_proofs = "proofs" in run_card["artifact_tree"]
        run_card["artifact_tree"] = build_artifact_tree(
            run_card["artifact_index"],
            include_proofs=include_proofs,
        )
    else:
        run_card["artifact_merkle_root"] = compute_artifact_merkle_root(run_card["artifact_index"])
    _rewrite_run_card_with_valid_self_integrity(run_card_path, run_card)
    _refresh_evidence_artifact_record(evidence, artifact_kind="run_card", artifact_path=run_card_path)
    write_execution_evidence(evidence_path, evidence, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ArtifactEvidenceError, match="does not match current confined artifact bytes"):
        verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)


def test_verifier_rejects_stale_batch_native_output_path_after_index_refresh(
    tmp_path: Path,
) -> None:
    prepared, _orchestrator, _results, output_root, evidence_path, batch_path, run_card_path = (
        _completed_prepared_run_with_card(tmp_path)
    )
    evidence = json.loads(evidence_path.read_bytes())
    batch = json.loads(batch_path.read_bytes())
    batch["results"][0]["depth_path"] = "depth/nonexistent-stale-depth.png"
    batch_path.write_text(
        dumps_json(batch, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _refresh_run_card_index_entries(run_card_path, [batch_path])
    _refresh_evidence_artifact_record(
        evidence,
        artifact_kind="batch_manifest_json",
        artifact_path=batch_path,
    )
    _refresh_evidence_artifact_record(evidence, artifact_kind="run_card", artifact_path=run_card_path)
    write_execution_evidence(evidence_path, evidence, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ArtifactEvidenceError, match="depth_path does not match combined-manifest authority"):
        verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)


@pytest.mark.parametrize(
    ("field_name", "overrides"),
    [
        pytest.param("depth_path", {}, id="depth"),
        pytest.param("depth_float_path", {"save_float_depth": True}, id="float-depth"),
        pytest.param("v2_output_path", {"enable_v2": True}, id="v2-output"),
    ],
)
def test_verifier_rejects_deleted_produced_batch_native_output(
    tmp_path: Path,
    field_name: str,
    overrides: dict[str, Any],
) -> None:
    prepared, _orchestrator, _results, output_root, evidence_path, batch_path, run_card_path = (
        _completed_prepared_run_with_card(tmp_path, **overrides)
    )
    evidence = json.loads(evidence_path.read_bytes())
    batch = json.loads(batch_path.read_bytes())
    assert batch["results"][0].pop(field_name)
    batch_path.write_text(
        dumps_json(batch, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _refresh_run_card_index_entries(run_card_path, [batch_path])
    _refresh_evidence_artifact_record(evidence, artifact_kind="batch_manifest_json", artifact_path=batch_path)
    _refresh_evidence_artifact_record(evidence, artifact_kind="run_card", artifact_path=run_card_path)
    write_execution_evidence(evidence_path, evidence, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ArtifactEvidenceError, match=f"lacks required {field_name}"):
        verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)


def test_verifier_rejects_artifact_path_on_failed_batch_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, image_path = _prepared(tmp_path, emit_run_card=False)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    monkeypatch.setattr(
        orchestrator,
        "_enhance_image_from_active_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected failure")),
    )

    with pytest.raises(ExecutionEvidenceError, match="failed required artifact accounting"):
        orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    batch_path = next((output_root / "manifests").glob("batch_*.json"))
    evidence = json.loads(evidence_path.read_bytes())
    batch = json.loads(batch_path.read_bytes())
    assert batch["results"][0]["status"] == "error"
    batch["results"][0]["manifest"] = "manifests/stale_combined.json"
    batch_path.write_text(
        dumps_json(batch, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _refresh_evidence_artifact_record(evidence, artifact_kind="batch_manifest_json", artifact_path=batch_path)
    write_execution_evidence(evidence_path, evidence, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ArtifactEvidenceError, match="non-ok input .* carries artifact paths"):
        verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)


def test_run_card_retry_rebuilds_outcome_projection_after_initial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    real_emit = orchestrator._emit_run_card
    call_count = 0

    def fail_once_then_emit(*args: Any, **kwargs: Any) -> Path | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None
        return real_emit(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_emit_run_card", fail_once_then_emit)
    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert results[0]["status"] == "ok"
    assert call_count == 3
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    verified = verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)
    require_required_artifacts(verified)
    run_card_outcome = next(item for item in verified["produced_artifacts"] if item["artifact_kind"] == "run_card")
    assert len(run_card_outcome["artifacts"]) == 1


def test_run_card_retry_final_failure_reconciles_carriers_to_missing_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module
    from transformation_portal.vlm_captioning.fastvlm_runtime import FastVLMRuntimeResult
    from transformation_portal.vlm_captioning.parser import parse_fastvlm_caption

    prepared, image_path = _prepared(
        tmp_path,
        emit_run_card=True,
        vlm_captioning_enabled=True,
    )
    output_root = tmp_path / "output"
    raw_text = "SCENE=Pool; MATERIALS=stone; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none."

    def caption(*_args: Any, **_kwargs: Any) -> FastVLMRuntimeResult:
        return FastVLMRuntimeResult(
            success=True,
            status="ok",
            caption_parse=parse_fastvlm_caption(raw_text),
            raw_stdout=raw_text,
            raw_stderr="",
            returncode=0,
            command=["fake-fastvlm"],
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(orchestrator_module, "run_fastvlm_caption", caption)
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    real_emit = orchestrator._emit_run_card
    call_count = 0

    def fail_emit_succeed_fail(*args: Any, **kwargs: Any) -> Path | None:
        nonlocal call_count
        call_count += 1
        if call_count in {1, 3}:
            return None
        return real_emit(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_emit_run_card", fail_emit_succeed_fail)
    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert results[0]["status"] == "ok"
    assert call_count == 3
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    verified = verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)
    run_card_omission = next(item for item in verified["omitted_artifacts"] if item["artifact_kind"] == "run_card")
    assert run_card_omission["reason_code"] == "optional_stage_no_output"


def test_run_card_retry_reconciles_initial_success_then_index_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, image_path = _prepared(tmp_path, emit_run_card=True)
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    real_emit = orchestrator._emit_run_card
    call_count = 0

    def emit_once_then_fail(*args: Any, **kwargs: Any) -> Path | None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return None
        return real_emit(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_emit_run_card", emit_once_then_fail)
    results = orchestrator.enhance_batch(prepared.input_root, input_files=[image_path])

    assert results[0]["status"] == "ok"
    assert call_count == 2
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    verified = verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)
    run_card_omission = next(item for item in verified["omitted_artifacts"] if item["artifact_kind"] == "run_card")
    assert run_card_omission["reason_code"] == "optional_stage_no_output"


def test_reconstruction_bundle_paths_are_portable_and_bound_to_evidence(
    tmp_path: Path,
) -> None:
    from transformation_portal.lux_depth_v3.reconstruction_runner import (
        diagnostics_artifact_path,
        manifest_artifact_path,
    )
    from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups

    input_root = tmp_path / "inputs"
    scene_root = input_root / "scene"
    scene_root.mkdir(parents=True)
    image_paths = (scene_root / "view_1.png", scene_root / "view_2.png")
    for index, image_path in enumerate(image_paths):
        Image.new("RGB", (64, 64), color=(index + 1, index + 2, index + 3)).save(image_path)
    scene = build_scene_groups(image_paths, dataset_root=input_root, grouping_mode="parent_dir")[0]
    camera = {
        "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 64,
        "height": 64,
    }
    second_camera = json.loads(json.dumps(camera))
    second_camera["extrinsics"][0][3] = 0.1
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema": "tp.scene_cameras.v1",
                "scenes": {
                    scene.scene_id: {
                        "images": [path.relative_to(input_root).as_posix() for path in scene.images],
                        "cameras": [camera, second_camera],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="synthetic",
            allow_synthetic_fallback=True,
            enable_v2=False,
            emit_run_card=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path=str(sidecar_path),
            non_commercial_ok=True,
            accept_research_tools_license=True,
        ),
        input_root,
        image_paths,
    )
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)

    def run_reconstruction(**kwargs: Any) -> Path:
        output_dir = Path(kwargs["output_dir"])
        scene_id = kwargs["context"].scene_id
        reconstruction_manifest = manifest_artifact_path(scene_id=scene_id, output_dir=output_dir)
        diagnostics = diagnostics_artifact_path(scene_id=scene_id, output_dir=output_dir)
        report = output_dir / f"{scene_id}_reconstruction_report.json"
        for path in (reconstruction_manifest, diagnostics, report):
            path.write_text("{}", encoding="utf-8")
        return report

    orchestrator.run_scene_reconstruction_fn = run_reconstruction
    orchestrator.enhance_batch(prepared.input_root, input_files=list(prepared.input_files))
    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    batch_path = next((output_root / "manifests").glob("batch_*.json"))
    evidence = json.loads(evidence_path.read_bytes())
    batch = json.loads(batch_path.read_bytes())
    reconstruction_fields = (
        "reconstruction_preflight_path",
        "reconstruction_scene_manifest_path",
        "reconstruction_manifest_path",
        "reconstruction_report_path",
        "reconstruction_diagnostics_path",
    )
    for field_name in reconstruction_fields:
        path_value = batch["results"][0][field_name]
        assert isinstance(path_value, str) and path_value
        assert not Path(path_value).is_absolute()

    del batch["results"][0]["reconstruction_manifest_path"]
    batch_path.write_text(
        dumps_json(batch, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _refresh_evidence_artifact_record(evidence, artifact_kind="batch_manifest_json", artifact_path=batch_path)
    write_execution_evidence(evidence_path, evidence, output_root=output_root, plan=prepared.plan)

    with pytest.raises(ArtifactEvidenceError, match="reconstruction paths do not match"):
        verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)


def test_prepared_reconstruction_manifests_survive_snapshot_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist durable input paths while the model consumes frozen snapshots."""

    import numpy as np

    from transformation_portal.lux_depth_v3 import reconstruction_runner
    from transformation_portal.lux_depth_v3.reconstruction_manifest import (
        load_reconstruction_manifest,
        manifest_image_paths,
    )
    from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups
    from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity
    from transformation_portal.spatial_ai.reconstruction.contracts import GaussianSplat, Scene3D

    input_root = tmp_path / "inputs"
    scene_root = input_root / "scene"
    scene_root.mkdir(parents=True)
    image_paths = (scene_root / "view_1.png", scene_root / "view_2.png")
    for index, image_path in enumerate(image_paths):
        Image.new("RGB", (64, 64), color=(index + 1, index + 2, index + 3)).save(image_path)
    scene = build_scene_groups(image_paths, dataset_root=input_root, grouping_mode="parent_dir")[0]
    camera = {
        "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 64,
        "height": 64,
    }
    second_camera = json.loads(json.dumps(camera))
    second_camera["extrinsics"][0][3] = 0.1
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema": "tp.scene_cameras.v1",
                "scenes": {
                    scene.scene_id: {
                        "images": [path.relative_to(input_root).as_posix() for path in scene.images],
                        "cameras": [camera, second_camera],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="synthetic",
            allow_synthetic_fallback=True,
            enable_v2=False,
            emit_run_card=True,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path=str(sidecar_path),
            non_commercial_ok=True,
            accept_research_tools_license=True,
        ),
        input_root,
        image_paths,
    )
    output_root = tmp_path / "output"
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_root)
    observed_model_paths: list[Path] = []
    builder_calls = 0

    class FakeSceneBuilder:
        def build_from_images(self, **kwargs: Any) -> Scene3D:
            nonlocal builder_calls
            builder_calls += 1
            model_paths = [Path(value) for value in kwargs["image_paths"]]
            observed_model_paths.extend(model_paths)
            assert all(path.exists() for path in model_paths)
            splats = GaussianSplat(
                positions=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
                colors=np.full((2, 3), 0.5, dtype=np.float32),
                scales=np.ones((2, 3), dtype=np.float32),
                rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
                opacities=np.full((2, 1), 0.5, dtype=np.float32),
                metadata={},
            )
            return Scene3D(
                splats=splats,
                cameras=list(kwargs["cameras"]),
                rmse=0.01,
                iteration=1,
                convergence="converged",
                metadata={},
            )

    monkeypatch.setattr(reconstruction_runner, "_build_scene_builder", lambda **_kwargs: FakeSceneBuilder())
    orchestrator.run_scene_reconstruction_fn = reconstruction_runner.run_scene_reconstruction
    results = orchestrator.enhance_batch(prepared.input_root, input_files=list(prepared.input_files))

    assert observed_model_paths
    assert all("tp-prepared-batch-inputs-" in path.as_posix() for path in observed_model_paths)
    assert all(not path.exists() for path in observed_model_paths)

    scene_manifest_path = Path(results[0]["reconstruction_scene_manifest_path"])
    reconstruction_manifest_path = Path(results[0]["reconstruction_manifest_path"])
    scene_manifest = json.loads(scene_manifest_path.read_bytes())
    reconstruction_manifest = load_reconstruction_manifest(manifest_path=reconstruction_manifest_path)
    assert {Path(image["path"]) for image in scene_manifest["images"]} == set(image_paths)
    assert tuple(path.resolve() for path in manifest_image_paths(reconstruction_manifest)) == tuple(
        path.resolve() for path in image_paths
    )
    assert "tp-prepared-batch-inputs-" not in json.dumps(scene_manifest, sort_keys=True)
    assert "tp-prepared-batch-inputs-" not in reconstruction_manifest_path.read_text(encoding="utf-8")

    evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    verified = verify_execution_evidence_file(evidence_path, output_root=output_root, plan=prepared.plan)
    require_required_artifacts(verified)
    run_card_path = next(path for path in output_root.glob("run_card_*.json") if not path.name.endswith(".self.json"))
    assert verify_run_card_integrity(run_card_path) == []

    stale_manifest = json.loads(reconstruction_manifest_path.read_bytes())
    stale_manifest["dataset_root"] = str(tmp_path / "tp-prepared-input-expired")
    reconstruction_manifest_path.write_text(
        dumps_json(stale_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    prior_evidence_paths = set((output_root / "manifests").glob("execution_evidence_*.json"))

    second_results = orchestrator.enhance_batch(prepared.input_root, input_files=list(prepared.input_files))

    assert builder_calls == 1
    refreshed_manifest_path = Path(second_results[0]["reconstruction_manifest_path"])
    refreshed_manifest = load_reconstruction_manifest(manifest_path=refreshed_manifest_path)
    assert refreshed_manifest.dataset_root == str(input_root.resolve())
    assert "tp-prepared-input-" not in refreshed_manifest_path.read_text(encoding="utf-8")
    second_evidence_path = next(
        path for path in (output_root / "manifests").glob("execution_evidence_*.json") if path not in prior_evidence_paths
    )
    second_evidence = verify_execution_evidence_file(
        second_evidence_path,
        output_root=output_root,
        plan=prepared.plan,
    )
    require_required_artifacts(second_evidence)


def test_prepared_reconstruction_cache_relocation_preserves_historical_carriers(
    tmp_path: Path,
) -> None:
    """Root relocation may refresh latest files but cannot rewrite run history."""

    from transformation_portal.lux_depth_v3.reconstruction_runner import (
        diagnostics_artifact_path,
        manifest_artifact_path,
    )
    from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups
    from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity

    roots = (tmp_path / "root-a", tmp_path / "root-b")
    image_sets: list[tuple[Path, Path]] = []
    for root in roots:
        scene_root = root / "scene"
        scene_root.mkdir(parents=True)
        images = (scene_root / "view_1.png", scene_root / "view_2.png")
        for index, image_path in enumerate(images):
            Image.new("RGB", (64, 64), color=(index + 1, index + 2, index + 3)).save(image_path)
        image_sets.append(images)

    scenes = [
        build_scene_groups(images, dataset_root=root, grouping_mode="parent_dir")[0] for root, images in zip(roots, image_sets)
    ]
    assert scenes[0].scene_id == scenes[1].scene_id
    camera = {
        "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 64,
        "height": 64,
    }
    second_camera = json.loads(json.dumps(camera))
    second_camera["extrinsics"][0][3] = 0.1
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema": "tp.scene_cameras.v1",
                "scenes": {
                    scenes[0].scene_id: {
                        "images": ["scene/view_1.png", "scene/view_2.png"],
                        "cameras": [camera, second_camera],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    prepared_runs = [
        prepare_lux_execution(
            EnhanceConfig(
                depth_backend="synthetic",
                allow_synthetic_fallback=True,
                enable_v2=False,
                emit_run_card=True,
                enable_reconstruction=True,
                grouping_mode="parent_dir",
                cameras_sidecar_path=str(sidecar_path),
                non_commercial_ok=True,
                accept_research_tools_license=True,
            ),
            root,
            images,
        )
        for root, images in zip(roots, image_sets)
    ]
    output_root = tmp_path / "output"
    builder_calls = 0

    def fake_reconstruction(**kwargs: Any) -> Path:
        nonlocal builder_calls
        builder_calls += 1
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_id = kwargs["context"].scene_id
        manifest_path = manifest_artifact_path(scene_id=scene_id, output_dir=output_dir)
        diagnostics_path = diagnostics_artifact_path(scene_id=scene_id, output_dir=output_dir)
        report_path = output_dir / f"{scene_id}_reconstruction_report.json"
        manifest_path.write_text(
            json.dumps({"dataset_root": str(kwargs["manifest_context"].dataset_root)}, sort_keys=True),
            encoding="utf-8",
        )
        diagnostics_path.write_text(
            json.dumps(
                {
                    "schema": "tp.reconstruction_diagnostics.v1",
                    "camera_count": 0,
                    "cameras": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                {
                    "scene_fingerprint": kwargs["scene_fingerprint"],
                    "run_card_merkle_root": kwargs["run_card_merkle_root"],
                    "manifest_path": str(manifest_path),
                    "diagnostics_path": str(diagnostics_path),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return report_path

    first_orchestrator = EnhanceOrchestrator.from_prepared(prepared_runs[0], output_root)
    first_orchestrator.run_scene_reconstruction_fn = fake_reconstruction
    first_results = first_orchestrator.enhance_batch(roots[0], input_files=list(image_sets[0]))
    first_evidence_path = next((output_root / "manifests").glob("execution_evidence_*.json"))
    first_evidence = verify_execution_evidence_file(
        first_evidence_path,
        output_root=output_root,
        plan=prepared_runs[0].plan,
    )
    first_run_card_path = output_root / next(
        outcome["artifacts"][0]["path"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "run_card"
    )
    first_reconstruction_records = next(
        outcome["artifacts"]
        for outcome in first_evidence["produced_artifacts"]
        if outcome["artifact_kind"] == "reconstruction_bundle"
    )
    first_reconstruction_paths = {record["path"]: output_root / record["path"] for record in first_reconstruction_records}
    first_report_path = next(
        path for relative, path in first_reconstruction_paths.items() if relative.endswith("_reconstruction_report.json")
    )
    first_report = json.loads(first_report_path.read_bytes())
    for key in ("manifest_path", "diagnostics_path"):
        referenced_path = Path(first_report[key])
        assert "/execution/" in referenced_path.as_posix()
        assert referenced_path.exists()
        assert referenced_path in first_reconstruction_paths.values()

    second_orchestrator = EnhanceOrchestrator.from_prepared(prepared_runs[1], output_root)
    second_orchestrator.run_scene_reconstruction_fn = fake_reconstruction
    prior_evidence_paths = set((output_root / "manifests").glob("execution_evidence_*.json"))
    second_results = second_orchestrator.enhance_batch(roots[1], input_files=list(image_sets[1]))

    assert builder_calls == 1
    second_evidence_path = next(
        path for path in (output_root / "manifests").glob("execution_evidence_*.json") if path not in prior_evidence_paths
    )
    verify_execution_evidence_file(
        second_evidence_path,
        output_root=output_root,
        plan=prepared_runs[1].plan,
    )
    verify_execution_evidence_file(
        first_evidence_path,
        output_root=output_root,
        plan=prepared_runs[0].plan,
    )
    assert verify_run_card_integrity(first_run_card_path) == []
    assert Path(first_results[0]["reconstruction_manifest_path"]) == Path(second_results[0]["reconstruction_manifest_path"])
    latest_manifest = json.loads(Path(second_results[0]["reconstruction_manifest_path"]).read_bytes())
    assert latest_manifest["dataset_root"] == str(roots[1])
