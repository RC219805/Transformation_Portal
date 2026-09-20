"""Operator workflow contracts for supplied material evidence and evaluation."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import tifffile

from transformation_portal.lux_depth_v4 import LuxDepthV4Request, prepare
from transformation_portal.materials_v4.__main__ import main
from transformation_portal.materials_v4.engine import ResponsePolicy

pytestmark = pytest.mark.unit


@pytest.fixture
def supplied(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "photo.tif"
    tifffile.imwrite(source, np.full((28, 28, 3), 32768, dtype=np.uint16), photometric="rgb")
    masks = tmp_path / "masks"
    masks.mkdir()
    mask = masks / "glass.npy"
    np.save(mask, np.ones((28, 28), dtype=np.float32), allow_pickle=False)
    payload = {
        "schema": "tp.materials.supplied.v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape": [28, 28],
        "regions": [
            {
                "region_id": "glass-1",
                "label": "glass",
                "mask_path": "glass.npy",
                "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                "semantic_confidence": 0.99,
            }
        ],
    }
    manifest = masks / "regions.json"
    manifest.write_text(json.dumps(payload))
    return source, manifest, tmp_path / "evidence"


def _args(supplied):
    source, manifest, output = supplied
    return [
        "import-supplied",
        "--source",
        str(source),
        "--input-color",
        "srgb",
        "--regions",
        str(manifest),
        "--output-dir",
        str(output),
    ]


def test_import_supplied_produces_admissible_v3_plan(supplied, capsys):
    assert main(_args(supplied)) == 0
    result = json.loads(capsys.readouterr().out)
    source, _, output = supplied
    request = LuxDepthV4Request(source.parent, output.parent / "lux-output", materials_manifest=output / "lux-materials.json")
    prepared = prepare(request)
    assert prepared.plan.schema == "tp.execution.plan.v3"
    assert prepared.plan.to_payload()["inputs"][0]["materials_v4"]["content_sha256"] == result["content_sha256"]
    assert (
        main(["inspect", "--source", str(source), "--input-color", "srgb", "--evidence", str(output / "evidence.json")]) == 0
    )
    assert json.loads(capsys.readouterr().out)["regions"][0]["provenance"] == "supplied"


@pytest.mark.parametrize("bad", ["source", "mask", "confidence", "shape", "traversal"])
def test_bad_supplied_evidence_fails_before_output(supplied, capsys, bad):
    _, manifest, output = supplied
    payload = json.loads(manifest.read_bytes())
    if bad == "source":
        payload["source_sha256"] = "a" * 64
    elif bad == "mask":
        payload["regions"][0]["mask_sha256"] = "b" * 64
    elif bad == "confidence":
        payload["regions"][0]["semantic_confidence"] = True
    elif bad == "shape":
        payload["shape"] = [29, 28]
    else:
        payload["regions"][0]["mask_path"] = "../glass.npy"
    manifest.write_text(json.dumps(payload))
    assert main(_args(supplied)) == 1
    assert not output.exists()
    assert "materials-v4:" in capsys.readouterr().err


def test_output_is_never_overwritten(supplied, capsys):
    assert main(_args(supplied)) == 0
    capsys.readouterr()
    evidence = supplied[2] / "evidence.json"
    before = evidence.read_bytes()
    assert main(_args(supplied)) == 1
    assert evidence.read_bytes() == before


def test_policy_cli_is_complete_and_parseable(capsys):
    assert main(["policy"]) == 0
    assert ResponsePolicy.from_payload(json.loads(capsys.readouterr().out)) == ResponsePolicy()


def test_missing_evaluation_evidence_is_nonzero_and_explicit(tmp_path, capsys):
    source = tmp_path / "empty.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps({"schema": "tp.materials.evaluation_input.v1", "records": [], "selection_sources": []}))
    assert main(["evaluate", "--input", str(source), "--output", str(output)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unavailable" and result["acceptance"] == "not_assessed"
    assert json.loads(output.read_bytes()) == result


@pytest.mark.parametrize("kind", ["oversized_file", "oversized_header", "wrong_geometry"])
def test_mask_resource_rejection_precedes_immutable_copy(supplied, monkeypatch, capsys, kind):
    from transformation_portal.materials_v4 import __main__ as cli

    _, manifest, output = supplied
    mask = manifest.parent / "glass.npy"
    if kind == "oversized_file":
        np.save(mask, np.ones((100, 100), dtype=np.float32), allow_pickle=False)
    elif kind == "oversized_header":
        mask.write_bytes(b"\x93NUMPY\x02\x00" + (2**30).to_bytes(4, "little"))
    else:
        np.save(mask, np.ones((27, 28), dtype=np.float32), allow_pickle=False)
    payload = json.loads(manifest.read_bytes())
    payload["regions"][0]["mask_sha256"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))

    def forbidden_copy(*args, **kwargs):
        pytest.fail("Malformed or over-budget masks must fail before immutable pixel copies")

    monkeypatch.setattr(cli, "RegionEvidence", forbidden_copy)
    assert main(_args(supplied)) == 1
    assert not output.exists()
