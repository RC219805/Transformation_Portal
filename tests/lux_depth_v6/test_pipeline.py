"""Complete retained V5 -> V6 grade -> semantic replay with controlled inference.

Only the upstream V5 neural worker and native runtime are fixtures. All source
admission, transforms, encoders, inventory, and V6 replay execute normally.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import tifffile

from tests.lux_depth_v5.test_pipeline import execute
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v6.__main__ import main
from transformation_portal.lux_depth_v6.color import GradeRecipe
from transformation_portal.lux_depth_v6.evidence import verify_execution_evidence
from transformation_portal.lux_depth_v6.pipeline import run
from transformation_portal.lux_depth_v6.plan import GradePlan, LuxDepthV6Request, OutputLimits, digest, prepare

pytestmark = pytest.mark.unit


@pytest.fixture(name="v5_parent")
def fixture_v5_parent(request_case):
    return execute(request_case).output_root


def test_end_to_end_grade_plan_delivery_and_replay(v5_parent, tmp_path):
    target = tmp_path / "v6"
    prepared = prepare(LuxDepthV6Request(v5_parent, target, grade=GradeRecipe(exposure_stops=0.5, saturation=1.1)))
    assert not target.exists()
    assert prepared.plan.to_payload()["pipeline"] == "lux_depth_v6"
    result = run(prepared)
    verified = verify_execution_evidence(target, source_root=v5_parent, expected_plan_sha256=result.plan_sha256)
    assert verified.to_payload()["input_count"] == 1
    descriptor = json.loads((target / "input-0000/photograph.json").read_bytes())
    assert descriptor["reconstruction"]["refinement"] == "guided_bilinear_v3"
    assert descriptor["grade"]["changed_pixels"] > 0
    assert descriptor["display"]["metadata"]["color_domain"] == "display_linear_srgb"
    assert descriptor["production_acceptance"] == "not_established"
    assert tifffile.imread(target / "input-0000/delivery.tif").dtype == np.uint16
    display = np.load(target / "input-0000/display.npy", allow_pickle=False)
    assert np.all((display >= 0) & (display <= 1))


def test_identity_grade_preserves_reconstructed_master_pixels(v5_parent, tmp_path):
    result = run(prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6")))
    base = np.load(result.output_root / "input-0000/baseline.npy", allow_pickle=False)
    master = np.load(result.output_root / "input-0000/master.npy", allow_pickle=False)
    assert np.array_equal(base.view(np.uint8), master.view(np.uint8))


def test_repeated_execution_is_byte_identical(v5_parent, tmp_path):
    first = prepare(LuxDepthV6Request(v5_parent, tmp_path / "one"))
    second = prepare(LuxDepthV6Request(v5_parent, tmp_path / "two"))
    assert first.canonical_plan_bytes == second.canonical_plan_bytes
    one, two = run(first), run(second)
    for path in one.output_root.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (two.output_root / path.relative_to(one.output_root)).read_bytes()


def test_changed_parent_rejected_before_output_creation(v5_parent, tmp_path):
    prepared = prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6"))
    master = v5_parent / "input-0000/master.npy"
    master.write_bytes(master.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        run(prepared)
    assert not prepared.output_root.exists()


def test_recomputed_inventory_does_not_authorize_changed_grade(v5_parent, tmp_path):
    result = run(prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6")))
    path = result.output_root / "input-0000/master.npy"
    values = np.load(path, allow_pickle=False)
    values[0, 0, 0] += 0.1
    np.save(path, values, allow_pickle=False)
    evidence = json.loads(result.evidence_path.read_bytes())
    record = next(record for record in evidence["artifacts"] if record["path"] == "input-0000/master.npy")
    record.update(size_bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
    result.evidence_path.write_bytes(canonicalize_json(evidence))
    with pytest.raises(ValueError, match="semantic replay"):
        verify_execution_evidence(result.output_root, source_root=v5_parent)


def test_extra_file_and_wrong_plan_digest_are_rejected(v5_parent, tmp_path):
    result = run(prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6")))
    with pytest.raises(ValueError, match="expected exact"):
        verify_execution_evidence(result.output_root, source_root=v5_parent, expected_plan_sha256="0" * 64)
    (result.output_root / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="namespace"):
        verify_execution_evidence(result.output_root, source_root=v5_parent)


@pytest.mark.parametrize("mutation", ["changed", "removed", "symlink", "hardlink"])
def test_completion_must_remain_bound_after_semantic_replay(v5_parent, tmp_path, monkeypatch, mutation):
    from transformation_portal.lux_depth_v6 import evidence as evidence_module

    result = run(prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6")))
    replay = evidence_module.verify_artifacts

    def replay_then_mutate_completion(*args, **kwargs):
        replay(*args, **kwargs)
        if mutation == "changed":
            result.evidence_path.write_bytes(b"{}")
        elif mutation == "removed":
            result.evidence_path.unlink()
        elif mutation == "symlink":
            retained = tmp_path / "retained-completion.json"
            result.evidence_path.rename(retained)
            result.evidence_path.symlink_to(retained)
        else:
            (tmp_path / "completion-alias.json").hardlink_to(result.evidence_path)

    monkeypatch.setattr(evidence_module, "verify_artifacts", replay_then_mutate_completion)
    with pytest.raises((ValueError, OSError, ArtifactEvidenceError)):
        verify_execution_evidence(result.output_root, source_root=v5_parent)


def test_cancellation_never_creates_completion(v5_parent, tmp_path):
    prepared = prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6"))
    with pytest.raises(RuntimeError, match="cancelled"):
        run(prepared, cancellation=lambda: True)
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("replace_root", [True, False])
def test_directory_replacement_cannot_redirect_product_write(v5_parent, tmp_path, monkeypatch, replace_root):
    from transformation_portal.lux_depth_v6 import pipeline

    target, outside, displaced = tmp_path / "v6", tmp_path / "outside", tmp_path / "displaced"
    outside.mkdir()
    prepared = prepare(LuxDepthV6Request(v5_parent, target))
    secure_write = pipeline._secure_atomic_write_bytes

    def replace_before_secure_write(root, relative, data, *, maximum_bytes):
        if relative == "input-0000/baseline.npy":
            directory = target if replace_root else target / "input-0000"
            directory.rename(displaced)
            directory.symlink_to(outside, target_is_directory=True)
        return secure_write(root, relative, data, maximum_bytes=maximum_bytes)

    monkeypatch.setattr(pipeline, "_secure_atomic_write_bytes", replace_before_secure_write)
    with pytest.raises(ArtifactEvidenceError):
        run(prepared)
    assert displaced.is_dir()
    assert not list(outside.iterdir())
    assert not (target / "evidence.json").exists()
    assert not (displaced / "evidence.json").exists()


def test_new_output_and_output_reservation_are_mandatory(v5_parent, tmp_path):
    with pytest.raises(ValueError, match="new"):
        prepare(LuxDepthV6Request(v5_parent, v5_parent))
    with pytest.raises(ValueError, match="disjoint"):
        prepare(LuxDepthV6Request(v5_parent, v5_parent / "nested"))
    with pytest.raises(ValueError, match="reservation"):
        prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6", output_limits=OutputLimits(max_output_bytes=1024)))


def test_closed_plan_and_processing_drift(v5_parent, tmp_path):
    prepared = prepare(LuxDepthV6Request(v5_parent, tmp_path / "v6"))
    payload = prepared.plan.to_payload()
    payload["extra"] = True
    with pytest.raises(ValueError, match="closed"):
        GradePlan(canonicalize_json(payload))
    payload.pop("extra")
    payload["processing"]["dependencies"]["numpy"] = "0.0.0"
    with pytest.raises(ValueError, match="processing"):
        run(replace(prepared, plan=GradePlan(canonicalize_json(payload))))
    assert not prepared.output_root.exists()


def test_cli_plan_and_execution_and_verify(v5_parent, tmp_path, capsys):
    arguments = ["--input-dir", str(v5_parent), "--output-dir", str(tmp_path / "cli")]
    assert main([*arguments, "--plan"]) == 0
    planned = capsys.readouterr().out.strip().encode()
    assert GradePlan(planned).to_payload()["pipeline"] == "lux_depth_v6"
    assert not (tmp_path / "cli").exists()
    assert main(arguments) == 0
    capsys.readouterr()
    assert main([*arguments, "--verify", "--expected-plan-sha256", digest(planned)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert main([*arguments, "--saturation", "nan"]) == 1
