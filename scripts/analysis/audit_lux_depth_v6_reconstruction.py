#!/usr/bin/env python3
"""Measure synthetic depth reconstruction without loading a model or camera data.

The fixtures have analytical references, not surveyed scene geometry. Their
errors diagnose interpolation and RGB-guidance behavior only; no result here
establishes photographic accuracy or additional native model resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformation_portal.core.depth_evidence import build_depth_evidence  # noqa: E402
from transformation_portal.core.image_artifact import ImageMaster  # noqa: E402
from transformation_portal.lux_depth_v4.photography import create_proxy  # noqa: E402
from transformation_portal.lux_depth_v5.photography import align_depth  # noqa: E402

RECIPES = ("bilinear", "guided_bilinear_v3", "guided_bilinear_v4")
SCENES = ("plane", "affine_ramp", "step", "texture_mismatch", "checkerboard_noise", "isolated_outlier")
EDGE_THRESHOLD = 0.1
MASTER_SHAPE = (112, 112)
TARGET_SIZE = 28


def array_sha256(values: np.ndarray) -> str:
    """Hash exact raster bytes; callers separately carry shape and dtype."""
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def measure(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Measure finite shared-grid relative-depth samples and adjacent-pair edges.

    Edges mean absolute differences of at least 0.1 in relative-depth units.
    Counts use exact horizontal/vertical pair locations, with no tolerance or
    claim that an RGB edge is a depth edge. Invalid endpoints exclude a pair.
    """
    reference, prediction, valid = np.asarray(reference), np.asarray(prediction), np.asarray(valid)
    if (
        reference.dtype != np.float32
        or reference.ndim != 2
        or min(reference.shape) <= 0
        or prediction.dtype != np.float32
        or prediction.shape != reference.shape
        or valid.dtype != np.bool_
        or valid.shape != reference.shape
        or not valid.any()
        or not np.isfinite(reference).all()
        or not np.isfinite(prediction).all()
        or np.any((reference < 0) | (reference > 1))
        or np.any((prediction < 0) | (prediction > 1))
    ):
        raise ValueError("Measurements require bounded finite float32 HW rasters and a nonempty matching boolean mask")
    difference = prediction[valid].astype(np.float64) - reference[valid].astype(np.float64)
    edges = {"reference_pairs": 0, "prediction_pairs": 0, "matched_pairs": 0, "false_pairs": 0, "missed_pairs": 0}
    gradient_sum, gradient_count = 0.0, 0
    for axis in (0, 1):
        pairs = (valid[1:] & valid[:-1]) if axis == 0 else (valid[:, 1:] & valid[:, :-1])
        actual = np.diff(prediction.astype(np.float64), axis=axis)[pairs]
        target = np.diff(reference.astype(np.float64), axis=axis)[pairs]
        observed, expected = np.abs(actual) >= EDGE_THRESHOLD, np.abs(target) >= EDGE_THRESHOLD
        edges["reference_pairs"] += int(expected.sum())
        edges["prediction_pairs"] += int(observed.sum())
        edges["matched_pairs"] += int((observed & expected).sum())
        edges["false_pairs"] += int((observed & ~expected).sum())
        edges["missed_pairs"] += int((~observed & expected).sum())
        gradient_sum += float(np.abs(actual - target).sum())
        gradient_count += len(actual)
    return {
        "selected_pixels": int(valid.sum()),
        "units": "relative_depth_not_meters",
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "max_absolute_error": float(np.abs(difference).max()),
        "adjacent_pair_gradient_mae": gradient_sum / gradient_count if gradient_count else None,
        "edges": {"threshold": EDGE_THRESHOLD, "matching": "exact_adjacent_pair_no_tolerance", **edges},
    }


def _scene(name: str) -> tuple[ImageMaster, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact native samples, analytical master reference, and score mask."""
    if name not in SCENES:
        raise ValueError(f"Unknown synthetic scene: {name}")
    shape = (100, 112) if name == "plane" else MASTER_SHAPE
    pixels = np.full((*shape, 3), 0.1, np.float32)
    native = np.ones((TARGET_SIZE, TARGET_SIZE), np.float32)
    reference = np.zeros(shape, np.float32)
    score_mask = np.ones(shape, bool)
    if name == "plane":
        # These three padded rows must not stretch or contaminate the plane.
        native[25:] = 1000
    elif name == "affine_ramp":
        native[:] = 1 + np.arange(TARGET_SIZE, dtype=np.float32)[None, :] / (TARGET_SIZE - 1)
        native_limits = np.percentile(native, [1, 99])
        coordinate = np.clip((np.arange(MASTER_SHAPE[1]) + 0.5) * TARGET_SIZE / MASTER_SHAPE[1] - 0.5, 0, TARGET_SIZE - 1)
        analytic_depth = 1 + coordinate / (TARGET_SIZE - 1)
        reference[:] = np.clip((analytic_depth - native_limits[0]) / (native_limits[1] - native_limits[0]), 0, 1)
        # A high-contrast RGB boundary must not convert a native slope to a step.
        pixels[:, 56:] = 0.8
    elif name in {"step", "texture_mismatch"}:
        native[:, 14:] = 2
        reference[:, 56:] = 1
        if name == "step":
            pixels[:, 56:] = 0.8
        else:
            # Texture boundary is perpendicular to the true depth discontinuity.
            pixels[56:] = 0.8
    elif name == "checkerboard_noise":
        classes = (np.indices(native.shape).sum(axis=0) % 2).astype(np.float32)
        native += classes
        colors = np.repeat(np.repeat(classes, 4, axis=0), 4, axis=1)
        pixels[:] = (0.15 + 0.45 * colors)[..., None]
        reference[:] = 0.5  # Known plane; alternating predictions are injected errors.
    elif name == "isolated_outlier":
        native[:, 21:] = 2
        native[10, 10] = 2
        pixels[:, 84:] = 0.8
        pixels[38:46, 38:46] = 0.8
        reference[:, 84:] = 1
        # Isolate the false near-plane surface; the actual far plane remains in
        # the fixture to fix percentile normalization independently of outliers.
        score_mask[:] = False
        score_mask[32:52, 32:52] = True
    identity = hashlib.sha256(("synthetic_depth_fixture_v1:" + name).encode("ascii")).hexdigest()
    master = ImageMaster(pixels, identity, 32, metadata={"provenance": "analytical_synthetic_fixture"})
    return master, native, reference, score_mask


def audit_scene(name: str) -> dict[str, Any]:
    """Compare versioned reconstruction recipes on one frozen synthetic scene."""
    master, native, reference, score_mask = _scene(name)
    proxy = create_proxy(master, TARGET_SIZE)
    evidence = build_depth_evidence(native, np.zeros(native.shape, bool), proxy, master.source_sha256)
    original_hash = array_sha256(evidence.native_depth)
    outputs: dict[str, np.ndarray] = {}
    measurements: dict[str, Any] = {}
    for recipe in RECIPES:
        aligned = align_depth(evidence, master, proxy, refinement=recipe)
        scored = aligned.valid_mask & score_mask
        outputs[recipe] = aligned.relative_depth
        after_hash = array_sha256(evidence.native_depth)
        if after_hash != original_hash:
            raise RuntimeError("Reconstruction changed native inference evidence")
        measurements[recipe] = {
            **measure(reference, aligned.relative_depth, scored),
            "refined_master_pixels": aligned.metadata["refined_pixels"],
            "native_sha256_after": after_hash,
            "aligned_sha256": array_sha256(aligned.relative_depth),
            "valid_mask_sha256": array_sha256(aligned.valid_mask),
            "valid_master_pixels": int(aligned.valid_mask.sum()),
            "differs_from_bilinear_in_score_region": int(
                np.count_nonzero((aligned.relative_depth != outputs["bilinear"]) & scored)
            ),
        }
    return {
        "name": name,
        "source_sha256": master.source_sha256,
        "source_provenance": "synthetic_identifier_not_camera_file_digest",
        "source_shape": list(master.shape),
        "geometry": proxy.transform.to_payload(),
        "native_dtype": evidence.native_depth.dtype.name,
        "native_sample_counts": evidence.to_payload()["sample_counts"],
        "native_sha256_before": original_hash,
        "reference_sha256": array_sha256(reference),
        "score_mask_sha256": array_sha256(score_mask),
        "reference_authority": "analytical_synthetic_fixture",
        "normalization": "usable_native_percentile_1_99_relative_depth",
        "score_region": "rows32:52_cols32:52" if name == "isolated_outlier" else "whole_master",
        "measurements": measurements,
    }


def build_report() -> dict[str, Any]:
    """Return deterministic numeric evidence, with explicit acceptance limits."""
    return {
        "schema": "tp.lux.depth_reconstruction_audit.v1",
        "evidence_kind": "synthetic_analytical_reconstruction_only",
        "recipes": list(RECIPES),
        "native_inference_executed": False,
        "model_accuracy": "unmeasured",
        "photographic_acceptance": "unestablished",
        "production_acceptance": "unestablished",
        "new_inferred_detail": False,
        "scenes": [audit_scene(name) for name in SCENES],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Disposable JSON report path; existing report is replaced")
    args = parser.parse_args(argv)
    report = build_report()
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(f"Wrote synthetic reconstruction evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
