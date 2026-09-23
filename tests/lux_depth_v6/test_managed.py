"""Composite V6 admission, complete photographic products, and replay authority."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import digest_payload, parse_execution_plan
from transformation_portal.core.execution_plan_v5 import ENVELOPE_RESERVE, ExecutionPlanV5, stage_output_budget
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6.color import GradeRecipe, RenderRecipe
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request, prepare, run, verify_managed_evidence
from transformation_portal.lux_depth_v6.plan import digest
from transformation_portal.lux_depth_v6.publication import publication_paths
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher

pytestmark = pytest.mark.unit


@pytest.fixture(name="prepared")
def fixture_prepared(request_case):
    return prepare(
        ManagedLuxDepthV6Request(request_case, grade=GradeRecipe(exposure_stops=0.5), render=RenderRecipe("soft_srgb")),
        publisher=GenerationPublisher(artifact_store=None, record_store=None),
    )


def refingerprint(payload):
    payload.pop("plan_fingerprint_sha256", None)
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    return canonicalize_json(payload)


def test_composite_preparation_freezes_recipes_without_output_or_inference(prepared):
    payload = prepared.plan.to_payload()
    assert payload["finishing"]["grade"]["exposure_stops"] == 0.5
    assert payload["finishing"]["render"]["mode"] == "soft_srgb"
    assert payload["finishing"]["depth_maps"]["refinement"] == "guided_bilinear_v4"
    assert not prepared.output_root.exists()
    source_budget = payload["inference"]["resources"]["max_output_bytes"]
    assert source_budget == stage_output_budget(payload["resources"]["max_output_bytes"])
    assert 2 * source_budget + ENVELOPE_RESERVE <= payload["resources"]["max_output_bytes"]
    assert str(prepared.inference.input_root).encode() not in prepared.canonical_plan_bytes


@pytest.mark.parametrize("encoding", ["bytes", "text"])
def test_common_plan_parser_round_trips_managed_v6_authority(prepared, encoding):
    data = prepared.canonical_plan_bytes
    parsed = parse_execution_plan(data if encoding == "bytes" else data.decode("utf-8"))
    assert type(parsed) is ExecutionPlanV5
    assert parsed == prepared.plan
    assert parsed.canonical_bytes == data
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("mutation", ["unknown", "grade", "depth", "bool", "budget", "publisher"])
def test_refingerprinted_plan_cannot_authorize_unknown_or_unbounded_policy(prepared, mutation):
    payload = prepared.plan.to_payload()
    if mutation == "unknown":
        payload["argv"] = ["arbitrary"]
    elif mutation == "grade":
        payload["finishing"]["grade"]["exposure_stops"] = 9
    elif mutation == "depth":
        payload["finishing"]["depth_maps"] = None
    elif mutation == "bool":
        payload["finishing"]["grade"]["saturation"] = True
    elif mutation == "budget":
        payload["resources"]["max_output_bytes"] -= 20
    else:
        payload["publication"]["max_files"] += 1
    with pytest.raises(ExecutionPlanError):
        ExecutionPlanV5(refingerprint(payload))


@pytest.mark.parametrize(
    "section,field",
    [
        ("resources", "max_pixels"),
        ("resources", "max_input_bytes"),
        ("resources", "max_output_bytes"),
        ("resources", "wall_time_seconds"),
        ("resources", "memory_mib"),
        ("resources", "inference_slots"),
        ("publication", "max_files"),
        ("publication", "max_file_bytes"),
        ("publication", "max_total_bytes"),
        ("publication", "max_manifest_bytes"),
    ],
)
@pytest.mark.parametrize("invalid_type", ["integral_float", "boolean"])
def test_outer_budgets_require_exact_integers_before_inference(prepared, section, field, invalid_type):
    payload = prepared.plan.to_payload()
    payload[section][field] = float(payload[section][field]) if invalid_type == "integral_float" else True
    # Recompute a valid fingerprint: numeric equality with the exact child
    # budget must never authorize a differently typed outer resource value.
    with pytest.raises(ExecutionPlanError, match="exact integer" if invalid_type == "integral_float" else None):
        ExecutionPlanV5(refingerprint(payload))
    assert not prepared.output_root.exists()


def test_processing_drift_fails_before_model_or_output(prepared):
    payload = prepared.plan.to_payload()
    payload["processing"]["modules"]["transformation_portal.lux_depth_v6.managed"] = "0" * 64
    with pytest.raises(ValueError, match="processing identity"):
        run(replace(prepared, plan=ExecutionPlanV5(refingerprint(payload))))
    assert not prepared.output_root.exists()


def test_managed_execution_produces_photographic_tiff_png_and_depth(prepared):
    result = run(prepared)
    verified = verify_managed_evidence(result.output_root, expected_plan_bytes=prepared.canonical_plan_bytes)
    photo = result.output_root / "v6/input-0000"
    with tifffile.TiffFile(photo / "delivery.tif") as tiff:
        assert tiff.asarray().dtype == np.uint16
        assert tiff.asarray().shape == (28, 42, 3)
        assert tiff.pages[0].tags[34675].value
    with Image.open(photo / "preview.png") as draft:
        assert draft.mode == "RGB"
        assert draft.size == (42, 28)
        assert draft.info["icc_profile"]
    assert tifffile.imread(photo / "depth-relative.tif").dtype == np.float32
    assert np.load(photo / "depth-valid.npy", allow_pickle=False).dtype == bool
    names = {record.path for record in verified.artifacts}
    assert names.issubset(publication_paths(prepared.plan.to_payload()))
    assert "source-v5/execution-evidence.json" in names
    assert "v6/evidence.json" in names
    assert "execution-evidence.json" in names
    assert verified.to_payload()["input_count"] == 1


def test_rewritten_outer_inventory_cannot_authorize_changed_delivery(prepared):
    result = run(prepared)
    target = result.output_root / "v6/input-0000/delivery.tif"
    target.write_bytes(target.read_bytes() + b"changed")
    root_completion = result.output_root / "execution-evidence.json"
    payload = decode_bounded_json_object(root_completion.read_bytes())
    record = next(item for item in payload["artifacts"] if item["path"] == "v6/input-0000/delivery.tif")
    record.update(size_bytes=target.stat().st_size, sha256=digest(target.read_bytes()))
    root_completion.write_bytes(canonicalize_json(payload))
    with pytest.raises(ValueError):
        verify_managed_evidence(result.output_root, expected_plan_bytes=prepared.canonical_plan_bytes)


def test_cancelled_composition_never_creates_success(prepared):
    with pytest.raises(RuntimeError, match="cancelled"):
        run(prepared, cancellation=lambda: True)
    assert not prepared.output_root.exists()


def test_finish_failure_preserves_source_without_outer_success(prepared, monkeypatch):
    from transformation_portal.lux_depth_v6 import pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("finishing failed")

    monkeypatch.setattr(pipeline, "run", fail)
    with pytest.raises(RuntimeError, match="finishing failed"):
        run(prepared)
    assert (prepared.output_root / "source-v5/execution-evidence.json").exists()
    assert not (prepared.output_root / "execution-evidence.json").exists()


def test_materials_and_unsupported_pixel_ceiling_rejected_before_output(request_case):
    publisher = GenerationPublisher(artifact_store=None, record_store=None)
    for inference in (
        replace(request_case, materials_manifest=request_case.input_dir / "materials.json"),
        replace(request_case, max_pixels=100_000_001),
    ):
        with pytest.raises(ValueError, match="Materials|100 million"):
            prepare(ManagedLuxDepthV6Request(inference), publisher=publisher)
        assert not request_case.output_dir.exists()


def _forged_delivery(root):
    """Construct a self-consistent inventory for TIFF bytes never semantically replayed."""
    target = root / "v6/input-0000/delivery.tif"
    changed = target.read_bytes() + b"unverified delivery replacement"
    completion = root / "v6/evidence.json"
    inner = decode_bounded_json_object(completion.read_bytes())
    record = next(row for row in inner["artifacts"] if row["path"] == "input-0000/delivery.tif")
    record.update(size_bytes=len(changed), sha256=digest(changed))
    return target, changed, completion, canonicalize_json(inner)


def test_publication_rejects_completion_rewritten_after_semantic_replay(prepared, monkeypatch):
    from transformation_portal.lux_depth_v6 import managed

    result = run(prepared)
    target, changed, inner_path, inner_raw = _forged_delivery(result.output_root)
    outer_path = result.output_root / "execution-evidence.json"
    outer = decode_bounded_json_object(outer_path.read_bytes())
    for path, contents in (("v6/input-0000/delivery.tif", changed), ("v6/evidence.json", inner_raw)):
        record = next(row for row in outer["artifacts"] if row["path"] == path)
        record.update(size_bytes=len(contents), sha256=digest(contents))
    # Outer verification reads these claimed hashes before the legitimate inner
    # verifier runs. Changing both inventories must not authorize a later swap.
    outer_path.write_bytes(canonicalize_json(outer))
    original_verify = managed.verify_execution_evidence

    def swap_after_verified(*args, **kwargs):
        verified = original_verify(*args, **kwargs)
        target.write_bytes(changed)
        inner_path.write_bytes(inner_raw)
        return verified

    monkeypatch.setattr(managed, "verify_execution_evidence", swap_after_verified)
    with pytest.raises(ValueError, match="completion|verified"):
        verify_managed_evidence(result.output_root, expected_plan_bytes=prepared.canonical_plan_bytes)


def test_execution_rejects_completion_rewritten_after_finishing_returns(prepared, monkeypatch):
    from transformation_portal.lux_depth_v6 import pipeline

    original_run = pipeline.run

    def swap_after_finished(*args, **kwargs):
        result = original_run(*args, **kwargs)
        target, changed, completion, inner_raw = _forged_delivery(result.output_root.parent)
        target.write_bytes(changed)
        completion.write_bytes(inner_raw)
        return result

    monkeypatch.setattr(pipeline, "run", swap_after_finished)
    with pytest.raises(ValueError, match="completion|verified"):
        run(prepared)
    assert not (prepared.output_root / "execution-evidence.json").exists()


def test_publication_does_not_repeat_source_replay_for_identical_limits(prepared, monkeypatch):
    from unittest.mock import Mock

    from transformation_portal.lux_depth_v6 import evidence

    result = run(prepared)
    prepare_source = Mock(wraps=evidence.prepare_source)
    monkeypatch.setattr(evidence, "prepare_source", prepare_source)
    verify_managed_evidence(result.output_root, expected_plan_bytes=prepared.canonical_plan_bytes)
    # The composite already requires exact equality with the frozen finishing
    # limits. This path must not perform a duplicate full V5 verification.
    assert prepare_source.call_count == 1
