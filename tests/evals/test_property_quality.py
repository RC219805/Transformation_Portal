"""Native quality protocol regression fixtures, entirely synthetic/non-promotional.

Tiny arrays stand in for sky/plaster gradients and architectural edge regions;
they prove calculations and registration behavior, never real quality acceptance.
Generated TIFFs and reports live only in pytest temporary directories. Real-case
declarations below exercise parser/review binding, not actual human acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from transformation_portal.evals.property_quality import EVALSET_SCHEMA, evaluate_property_evalset, main

pytestmark = pytest.mark.unit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evalset(root: Path) -> dict:
    # Wall/sky tone ramp plus a window/material edge, not a photograph.
    reference = np.repeat(np.tile(np.linspace(8000, 42000, 128, dtype=np.uint16), (64, 1))[..., None], 3, axis=2)
    reference[:32, 64:, 0] = 6000
    candidate = reference.copy()
    candidate[8:24, 8:24, 0] = 65535
    for name, array in (("reference.tif", reference), ("candidate.tif", candidate)):
        tifffile.imwrite(root / name, array, photometric="rgb")
    return {
        "schema": EVALSET_SCHEMA,
        "evalset_id": "synthetic-protocol-only",
        "fixtures": [
            {
                "fixture_id": "synthetic-wall-window",
                "synthetic_data": True,
                "reference": {"path": "reference.tif", "sha256": _sha(root / "reference.tif")},
                "candidate": {"path": "candidate.tif", "sha256": _sha(root / "candidate.tif")},
                "scene_tags": ["interior"],
                "lighting_group": "synthetic-daylight",
                "working_color_space": "sRGB",
                "working_transfer_function": "srgb",
                "crops": [
                    {"crop_id": "clipped-wall", "box": [8, 8, 24, 24], "concerns": ["plaster"]},
                    {"crop_id": "unchanged-window", "box": [64, 8, 80, 24], "concerns": ["window_edges"]},
                ],
            }
        ],
    }


def _review(fixture: dict) -> None:
    fixture["synthetic_data"] = False
    fixture["rights_record"] = "parser-test mock permission; no real property images"
    fixture["reference_review"] = {
        "decision": "approved_reference",
        "actor": "mock-reviewer",
        "reviewed_at": "2026-10-24T20:00:00Z",
        "sha256": fixture["reference"]["sha256"],
    }


def test_native_crops_reveal_local_defect_without_resizing_or_cross_crop_contamination(tmp_path):
    spec = _evalset(tmp_path)
    before = {name: _sha(tmp_path / name) for name in ("reference.tif", "candidate.tif")}

    report = evaluate_property_evalset(spec, asset_root=tmp_path)

    fixture = report["fixtures"][0]
    assert fixture["reference"]["dimensions"] == [128, 64]
    assert fixture["reference"]["dtype"] == "uint16"
    assert fixture["reference"]["sha256"] == before["reference.tif"]
    assert fixture["candidate"]["sha256"] == before["candidate.tif"]
    full = fixture["full_frame"]["metrics"]["highlight_clipping"]["value"]
    crop = fixture["native_crops"][0]
    assert crop["scale"] == "native_1_to_1"
    assert crop["box"] == [8, 8, 24, 24]
    assert crop["metrics"]["visible_delta"]["comparison"]["reference_dimensions"] == [16, 16]
    assert crop["metrics"]["highlight_clipping"]["value"] > full
    assert fixture["native_crops"][1]["metrics"]["visible_delta"]["value"] == 0.0
    assert report["photographic_production_acceptance"] == "not_assessed"
    assert report["real_reference_coverage"]["registered_scene_tags"] == []
    assert "plaster" in report["real_reference_coverage"]["missing_crop_concerns"]
    assert {name: _sha(tmp_path / name) for name in before} == before
    assert evaluate_property_evalset(spec, asset_root=tmp_path) == report


def test_declared_real_reference_requires_hash_bound_human_review(tmp_path):
    spec = _evalset(tmp_path)
    fixture = spec["fixtures"][0]
    fixture["synthetic_data"] = False
    fixture["rights_record"] = "mock permission"
    with pytest.raises(ValueError, match="saved approved_reference"):
        evaluate_property_evalset(spec, asset_root=tmp_path)
    _review(fixture)
    fixture["reference_review"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different artifact hash"):
        evaluate_property_evalset(spec, asset_root=tmp_path)


def test_coverage_of_registered_metadata_is_never_production_acceptance(tmp_path):
    spec = _evalset(tmp_path)
    fixture = spec["fixtures"][0]
    _review(fixture)
    fixture["scene_tags"] = ["interior", "exterior", "mixed_light", "twilight"]
    fixture["crops"][0]["concerns"] = ["sky", "ceiling_cove", "fixtures", "window_edges", "railings", "foliage", "plaster"]

    report = evaluate_property_evalset(spec, asset_root=tmp_path)

    assert report["real_reference_coverage"]["missing_scene_tags"] == []
    assert report["real_reference_coverage"]["missing_crop_concerns"] == []
    assert report["real_reference_coverage"]["representativeness"] == "requires_independent_human_assessment"
    assert report["photographic_production_acceptance"] == "not_assessed"
    assert report["fixtures"][0]["human_candidate_approval"] == "not_assessed"


def test_pending_real_references_are_measurable_without_fabricating_approval(tmp_path):
    spec = _evalset(tmp_path)
    fixture = spec["fixtures"][0]
    fixture["synthetic_data"] = False
    fixture["rights_record"] = "mock local-use permission record"
    fixture["reference_review"] = {
        "decision": "pending_reference_review",
        "sha256": fixture["reference"]["sha256"],
        "reason": "Prepared for a reviewer; no reference approval has been recorded",
    }
    report = evaluate_property_evalset(spec, asset_root=tmp_path)
    assert report["fixtures"][0]["reference_status"] == "pending_reference_review"
    assert report["real_reference_coverage"]["registered_scene_tags"] == ["interior"]
    assert report["approved_reference_coverage"]["registered_scene_tags"] == []
    assert report["photographic_production_acceptance"] == "not_assessed"
    fixture["reference_review"]["actor"] = "invented reviewer"
    with pytest.raises(ValueError, match="cannot claim reviewer"):
        evaluate_property_evalset(spec, asset_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    ["bad-hash", "wrong-size", "out-of-grid", "float-coordinate", "duplicate-crop", "unsafe-path", "missing-encoding"],
)
def test_malformed_or_stale_registration_fails_closed(tmp_path, change):
    spec = _evalset(tmp_path)
    fixture = spec["fixtures"][0]
    if change == "bad-hash":
        fixture["candidate"]["sha256"] = "0" * 64
    elif change == "wrong-size":
        tifffile.imwrite(tmp_path / "candidate.tif", np.zeros((32, 32, 3), dtype=np.uint16), photometric="rgb")
        fixture["candidate"]["sha256"] = _sha(tmp_path / "candidate.tif")
    elif change == "out-of-grid":
        fixture["crops"][0]["box"][2] = 129
    elif change == "float-coordinate":
        fixture["crops"][0]["box"][0] = 8.0
    elif change == "duplicate-crop":
        fixture["crops"].append(copy.deepcopy(fixture["crops"][0]))
    elif change == "unsafe-path":
        fixture["reference"]["path"] = "../reference.tif"
    else:
        del fixture["working_transfer_function"]
    with pytest.raises(ValueError):
        evaluate_property_evalset(spec, asset_root=tmp_path)


def test_symlinked_asset_is_rejected(tmp_path):
    spec = _evalset(tmp_path)
    (tmp_path / "link.tif").symlink_to(tmp_path / "reference.tif")
    spec["fixtures"][0]["reference"]["path"] = "link.tif"
    with pytest.raises(ValueError, match="symlink"):
        evaluate_property_evalset(spec, asset_root=tmp_path)


def test_cli_emits_deterministic_report_and_refuses_overwrite(tmp_path):
    spec = _evalset(tmp_path)
    spec_path, output = tmp_path / "evalset.json", tmp_path / "quality.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    args = ["--evalset", str(spec_path), "--asset-root", str(tmp_path), "--output", str(output)]

    assert main(args) == 0
    report_bytes = output.read_bytes()
    assert json.loads(report_bytes)["schema"] == "tp.property.quality_report.v1"
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert output.read_bytes() == report_bytes


@pytest.mark.parametrize("extension", ["png", "jpg"])
def test_registered_8bit_delivery_comparison_keeps_native_grid(tmp_path, extension):
    from PIL import Image

    spec = _evalset(tmp_path)
    image = np.full((32, 64, 3), 100, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / f"image.{extension}")
    identity = {"path": f"image.{extension}", "sha256": _sha(tmp_path / f"image.{extension}")}
    fixture = spec["fixtures"][0]
    fixture["reference"] = fixture["candidate"] = identity
    fixture["crops"] = [{"crop_id": "native", "box": [0, 0, 16, 16], "concerns": ["plaster"]}]

    report = evaluate_property_evalset(spec, asset_root=tmp_path)

    assert report["fixtures"][0]["reference"]["dtype"] == "uint8"
    assert report["fixtures"][0]["reference"]["dimensions"] == [64, 32]
    assert report["fixtures"][0]["full_frame"]["metrics"]["visible_delta"]["value"] == 0.0


def test_conflicting_embedded_working_profiles_fail_closed(tmp_path):
    spec = _evalset(tmp_path)
    candidate = tifffile.imread(tmp_path / "candidate.tif")
    # Synthetic profile bytes exercise identity comparison; no ICC assets committed.
    tifffile.imwrite(tmp_path / "candidate.tif", candidate, photometric="rgb", extratags=[(34675, "B", 4, b"test", False)])
    spec["fixtures"][0]["candidate"]["sha256"] = _sha(tmp_path / "candidate.tif")

    with pytest.raises(ValueError, match="matching embedded working profiles"):
        evaluate_property_evalset(spec, asset_root=tmp_path)


def test_non_normalized_tiff_orientation_cannot_invalidate_native_crop_coordinates(tmp_path):
    spec = _evalset(tmp_path)
    candidate = tifffile.imread(tmp_path / "candidate.tif")
    tifffile.imwrite(tmp_path / "candidate.tif", candidate, photometric="rgb", extratags=[(274, "H", 1, 6, False)])
    spec["fixtures"][0]["candidate"]["sha256"] = _sha(tmp_path / "candidate.tif")

    with pytest.raises(ValueError, match="normalize image orientation"):
        evaluate_property_evalset(spec, asset_root=tmp_path)


def test_oversized_tiff_header_is_rejected_before_pixel_allocation(tmp_path):
    import struct

    spec = _evalset(tmp_path)
    path = tmp_path / "candidate.tif"
    with tifffile.TiffFile(path) as source:
        offsets = [source.pages[0].tags[tag].valueoffset for tag in (256, 257)]
    with path.open("r+b") as target:
        for offset in offsets:
            target.seek(offset)
            target.write(struct.pack("<I", 100000))
    spec["fixtures"][0]["candidate"]["sha256"] = _sha(path)
    with pytest.raises(ValueError, match="bounded native image geometry"):
        evaluate_property_evalset(spec, asset_root=tmp_path)
