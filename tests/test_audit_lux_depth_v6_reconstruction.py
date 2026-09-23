"""Forensic reconstruction measurements preserve evidence and label their limits."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import audit_lux_depth_v6_reconstruction as audit

pytestmark = pytest.mark.unit


def test_measurements_match_analytical_errors_and_exact_adjacent_edges():
    reference = np.array([[0, 0, 1, 1]], np.float32)
    prediction = np.array([[0, 0.25, 0.75, 1]], np.float32)
    report = audit.measure(reference, prediction, np.ones(reference.shape, bool))
    assert report["mae"] == 0.125
    assert report["rmse"] == pytest.approx(np.sqrt(0.03125))
    assert report["adjacent_pair_gradient_mae"] == pytest.approx(1 / 3)
    assert report["edges"] == {
        "threshold": 0.1,
        "matching": "exact_adjacent_pair_no_tolerance",
        "reference_pairs": 1,
        "prediction_pairs": 3,
        "matched_pairs": 1,
        "false_pairs": 2,
        "missed_pairs": 0,
    }


def test_invalid_endpoints_do_not_contribute_pixels_or_edges():
    reference = np.array([[0, 0, 1, 1]], np.float32)
    prediction = np.array([[0, 0.25, 0.75, 1]], np.float32)
    report = audit.measure(reference, prediction, np.array([[True, False, False, True]]))
    assert report["selected_pixels"] == 2
    assert report["mae"] == report["rmse"] == 0
    assert report["adjacent_pair_gradient_mae"] is None
    assert report["edges"]["prediction_pairs"] == 0


@pytest.mark.parametrize("malformed", ["dtype", "shape", "mask_dtype", "empty_mask", "nonfinite", "range"])
def test_malformed_evidence_cannot_generate_successful_metrics(malformed):
    reference, prediction = np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32)
    valid = np.ones((2, 3), bool)
    if malformed == "dtype":
        reference = reference.astype(np.float64)
    elif malformed == "shape":
        prediction = prediction[:1]
    elif malformed == "mask_dtype":
        valid = valid.astype(np.uint8)
    elif malformed == "empty_mask":
        valid[:] = False
    elif malformed == "nonfinite":
        prediction[0, 0] = np.nan
        valid[0, 0] = False
    else:
        prediction[0, 0] = 2
    with pytest.raises(ValueError, match="bounded finite float32"):
        audit.measure(reference, prediction, valid)


def test_report_is_deterministic_and_native_evidence_is_unchanged():
    report = audit.build_report()
    assert json.dumps(report, sort_keys=True, allow_nan=False) == json.dumps(
        audit.build_report(), sort_keys=True, allow_nan=False
    )
    assert report["native_inference_executed"] is False
    assert report["new_inferred_detail"] is False
    assert report["model_accuracy"] == "unmeasured"
    assert report["photographic_acceptance"] == report["production_acceptance"] == "unestablished"
    assert [scene["name"] for scene in report["scenes"]] == list(audit.SCENES)
    for scene in report["scenes"]:
        assert scene["reference_authority"] == "analytical_synthetic_fixture"
        assert scene["geometry"]["original_shape"] == scene["source_shape"]
        assert scene["native_dtype"] == "float32"
        assert scene["native_sample_counts"]["usable_surface"] > 0
        for recipe in audit.RECIPES:
            result = scene["measurements"][recipe]
            assert result["native_sha256_after"] == scene["native_sha256_before"]
            assert result["units"] == "relative_depth_not_meters"
            assert result["selected_pixels"] <= result["valid_master_pixels"]


def test_plane_padding_and_checkerboard_probes_have_distinct_numeric_authority():
    plane = audit.audit_scene("plane")
    assert plane["geometry"]["resized_shape"] == [25, 28]
    assert plane["geometry"]["padded_shape"] == [28, 28]
    for measurement in plane["measurements"].values():
        assert measurement["mae"] == measurement["rmse"] == 0
        assert measurement["edges"]["prediction_pairs"] == 0
    noisy = audit.audit_scene("checkerboard_noise")["measurements"]
    assert noisy["guided_bilinear_v3"]["refined_master_pixels"] > 0
    assert noisy["guided_bilinear_v3"]["rmse"] > noisy["bilinear"]["rmse"]
    assert noisy["guided_bilinear_v4"]["refined_master_pixels"] == 0
    assert noisy["guided_bilinear_v4"]["aligned_sha256"] == noisy["bilinear"]["aligned_sha256"]


def test_cli_bootstraps_source_and_writes_parseable_report_outside_repository(tmp_path):
    script = Path(audit.__file__).resolve()
    destination = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(destination)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(destination.read_text())["schema"] == "tp.lux.depth_reconstruction_audit.v1"
    assert "Wrote synthetic reconstruction evidence" in result.stdout
