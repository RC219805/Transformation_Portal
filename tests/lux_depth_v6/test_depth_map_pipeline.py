"""Versioned V6 depth products must survive complete retained-parent replay."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import tifffile

from tests.lux_depth_v5.test_pipeline import execute
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6.__main__ import main
from transformation_portal.lux_depth_v6.depth_maps import DepthMapRecipe
from transformation_portal.lux_depth_v6.evidence import verify_execution_evidence
from transformation_portal.lux_depth_v6.pipeline import run
from transformation_portal.lux_depth_v6.plan import (
    DEPTH_PLAN_SCHEMA,
    GradePlan,
    LuxDepthV6Request,
    OutputLimits,
    digest,
    prepare,
)
from transformation_portal.lux_depth_v6.source import SourceLimits

pytestmark = pytest.mark.unit


@pytest.fixture(name="parent")
def fixture_parent(request_case):
    return execute(request_case).output_root


def request(parent, output, **kwargs):
    return LuxDepthV6Request(parent, output, depth_maps=DepthMapRecipe(), **kwargs)


def test_depth_plan_products_and_shared_photographic_alignment(parent, tmp_path):
    prepared = prepare(request(parent, tmp_path / "v6"))
    plan = prepared.plan.to_payload()
    assert plan["schema"] == DEPTH_PLAN_SCHEMA
    assert plan["depth_maps"] == DepthMapRecipe().to_payload()
    assert "transformation_portal.lux_depth_v6.depth_maps" in plan["processing"]["modules"]
    result = run(prepared)
    verified = verify_execution_evidence(result.output_root, source_root=parent, expected_plan_sha256=result.plan_sha256)
    root = result.output_root / "input-0000"
    photo = json.loads((root / "photograph.json").read_bytes())
    depth = json.loads((root / "depth.json").read_bytes())
    assert photo["schema"] == "tp.lux.graded_photograph.v2"
    assert photo["reconstruction"]["refinement"] == "guided_bilinear_v4"
    assert photo["depth_maps"]["path"] == "input-0000/depth.json"
    assert photo["reconstruction"]["aligned_content_sha256"] == depth["reconstruction"]["aligned_content_sha256"]
    native = np.load(root / "native-depth.npy", allow_pickle=False)
    upstream = np.load(parent / "input-0000/native-depth.npy", allow_pickle=False)
    assert native.tobytes() == upstream.tobytes()
    assert list(native.shape) == plan["source"]["images"][0]["native_shape"]
    relative = np.load(root / "relative-depth.npy", allow_pickle=False)
    assert list(relative.shape) == plan["source"]["images"][0]["shape"]
    assert tifffile.imread(root / "depth-relative.tif").tobytes() == relative.tobytes()
    assert not (root / "metric-depth-m.npy").exists()
    assert verified.to_payload()["production_acceptance"] == "not_established"


@pytest.mark.parametrize("filename", ["relative-depth.npy", "depth-valid.npy", "native-depth.npy", "depth.json"])
def test_rehashed_depth_products_still_require_semantic_replay(parent, tmp_path, filename):
    result = run(prepare(request(parent, tmp_path / "v6")))
    path = result.output_root / "input-0000" / filename
    path.write_bytes(path.read_bytes() + b"altered")
    completion = json.loads(result.evidence_path.read_bytes())
    record = next(item for item in completion["artifacts"] if item["path"] == f"input-0000/{filename}")
    record.update(size_bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
    result.evidence_path.write_bytes(canonicalize_json(completion))
    with pytest.raises(ValueError, match="semantic replay"):
        verify_execution_evidence(result.output_root, source_root=parent)


def test_depth_recipe_geometry_and_code_are_frozen_before_output(parent, tmp_path):
    prepared = prepare(request(parent, tmp_path / "v6"))
    for mutation in ("geometry", "implementation"):
        payload = prepared.plan.to_payload()
        if mutation == "geometry":
            payload["source"]["images"][0]["native_shape"] = [14, 14]
        else:
            payload["processing"]["modules"]["transformation_portal.lux_depth_v6.depth_maps"] = "0" * 64
        with pytest.raises(ValueError, match="identity changed"):
            run(replace(prepared, plan=GradePlan(canonicalize_json(payload))))
        assert not prepared.output_root.exists()
    payload = prepared.plan.to_payload()
    del payload["depth_maps"]
    with pytest.raises(ValueError, match="closed"):
        GradePlan(canonicalize_json(payload))


def test_opt_in_depth_has_separate_output_reservation(parent, tmp_path):
    # Enough for the legacy fixture (plan/completion + preview), but not the
    # additional depth PNG pair, native maps, TIFF and descriptor reservation.
    base = prepare(LuxDepthV6Request(parent, tmp_path / "legacy"))
    from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES
    from transformation_portal.lux_depth_v6.plan import MAX_PLAN_BYTES

    pixels = np.prod(base.plan.to_payload()["source"]["images"][0]["shape"])
    legacy_bytes = int(2 * MAX_PLAN_BYTES + pixels * 48 + MAX_PREVIEW_BYTES + 2 * 1024**2)
    prepare(LuxDepthV6Request(parent, tmp_path / "legacy", output_limits=OutputLimits(max_output_bytes=legacy_bytes)))
    with pytest.raises(ValueError, match="reservation"):
        prepare(request(parent, tmp_path / "depth", output_limits=OutputLimits(max_output_bytes=legacy_bytes)))


def test_default_plan_and_product_inventory_stay_legacy(parent, tmp_path):
    prepared = prepare(LuxDepthV6Request(parent, tmp_path / "legacy"))
    assert prepared.plan.to_payload()["schema"] == "tp.lux.grade.plan.v1"
    assert "depth_maps" not in prepared.plan.to_payload()
    result = run(prepared)
    assert not (result.output_root / "input-0000/depth.json").exists()
    assert (
        json.loads((result.output_root / "input-0000/photograph.json").read_bytes())["schema"] == "tp.lux.graded_photograph.v1"
    )


@pytest.mark.parametrize("budget", ["output", "memory"])
def test_native_grid_reservation_is_independent_of_master_size(parent, tmp_path, budget):
    payload = prepare(request(parent, tmp_path / "v6")).plan.to_payload()
    # Adversarial plan admission: a tiny master must not conceal a large native
    # allocation. Such geometry would also fail the later retained-source bind.
    payload["source"]["images"][0]["native_shape"] = [4096, 4096]
    if budget == "output":
        payload["limits"]["max_output_bytes"] = 100 * 1024**2
    else:
        payload["source"]["limits"]["memory_mib"] = 512
    with pytest.raises(ValueError, match="reservation|memory admission"):
        GradePlan(canonicalize_json(payload))


def test_cli_depth_plan_execution_verify_and_recipe_guard(parent, tmp_path, capsys):
    arguments = ["--input-dir", str(parent), "--output-dir", str(tmp_path / "cli")]
    assert main([*arguments, "--depth-maps", "--plan"]) == 0
    raw = capsys.readouterr().out.strip().encode()
    assert GradePlan(raw).to_payload()["depth_maps"]["refinement"] == "guided_bilinear_v4"
    assert main([*arguments, "--depth-maps"]) == 0
    capsys.readouterr()
    assert main([*arguments, "--verify", "--expected-plan-sha256", digest(raw)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert main([*arguments, "--depth-refinement", "bilinear", "--plan"]) == 1
    assert "requires --depth-maps" in capsys.readouterr().err
    assert main([*arguments, "--verify", "--depth-maps"]) == 1
    assert "recorded depth recipe" in capsys.readouterr().err


def test_depth_execution_is_byte_identical(parent, tmp_path):
    one = run(prepare(request(parent, tmp_path / "one")))
    two = run(prepare(request(parent, tmp_path / "two")))
    for path in one.output_root.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (two.output_root / path.relative_to(one.output_root)).read_bytes()


@pytest.mark.parametrize("depth_maps", [None, DepthMapRecipe()])
def test_verifier_applies_tighter_caller_memory_to_replay_before_loading_parent(parent, tmp_path, monkeypatch, depth_maps):
    from transformation_portal.lux_depth_v6 import evidence

    result = run(prepare(LuxDepthV6Request(parent, tmp_path / "v6", depth_maps=depth_maps)))

    def unexpected_parent_load(*args, **kwargs):
        pytest.fail("Caller replay limits must fail before parent loading or reconstruction")

    monkeypatch.setattr(evidence, "prepare_source", unexpected_parent_load)
    with pytest.raises(ValueError, match="memory admission"):
        verify_execution_evidence(result.output_root, source_root=parent, source_limits=SourceLimits(memory_mib=128))
