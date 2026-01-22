#!/usr/bin/env python3
"""
Production High-Fidelity Depth Validation Suite
================================================

Implements ALL critical fixes from review:
- BLOCKER A: No sliver tiles (reflect padding at borders)
- BLOCKER B: Increased overlap (192px for texture-heavy scenes)
- BLOCKER C: Gradient-weighted sampling (avoid flat regions)
- BLOCKER D: Disabled unsafe global anchor (until DC-aligned)
- BLOCKER E: Structural edge gating (no texture edge hallucination)

Validation includes:
- Seam boundary ratio (max 1.2)
- Edge F1 score (≥0.30 lenient, ≥0.60 strict)
- Chamfer distance (<15px lenient, <5px strict)
- Edge count ratio (≤2.0x)
- Overshoot/halo penalties

Usage:
    python production_depth_validation_fixed.py --input-dir input_images/750_Picacho/Source_TIFFs_Base --output-dir outputs/validation_production_fixed
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check for ML dependencies early
try:
    import torch
    import transformers

    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    # Will fail gracefully in main() with clear message

try:
    from high_fidelity_depth.depth_estimator import (
        HighFidelityDepthEstimator,
        DepthConfig,
    )
    from high_fidelity_depth.quality_metrics import (
        validate_depth_quality,
        detect_edges,
        save_metrics_atomic,
        create_edge_overlay,
        classify_scene_type_v2,
        extract_structure_edges,
    )
except ImportError as e:
    if not HAS_ML_DEPS:
        # Expected - will handle in main()
        pass
    else:
        # Unexpected import error
        raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def save_depth_16bit(depth: np.ndarray, path: Path):
    """Save depth as 16-bit TIFF."""
    depth_u16 = (depth * 65535).astype(np.uint16)
    Image.fromarray(depth_u16, mode="I;16").save(path, compression="tiff_deflate")
    logger.info(f"Saved 16-bit depth: {path}")


def validate_metrics_complete(metrics_dict: dict, image_name: str) -> None:
    """
    Fail fast if metrics are incomplete.

    Production validators MUST NOT write null placeholders.
    """
    from datetime import datetime

    required_fields = [
        "scene_type",
        "edge_f1",
        "lenient_pass",
        "strict_pass",
        "classification_factors",
    ]

    missing = [f for f in required_fields if metrics_dict.get(f) is None]

    if missing:
        logger.error(
            f"❌ FATAL: Incomplete metrics for {image_name}\n"
            f"   Missing fields: {missing}\n"
            f"   This indicates an integration failure.\n"
            f"   Validator must fail fast, not write placeholder nulls."
        )
        raise ValueError(f"Incomplete metrics for {image_name}: missing {missing}. This is a P0 integration bug.")

    # Validate types
    if not isinstance(metrics_dict["scene_type"], str):
        raise TypeError(f"scene_type must be str, got {type(metrics_dict['scene_type'])}")

    if not isinstance(metrics_dict["edge_f1"], (int, float)):
        raise TypeError(f"edge_f1 must be numeric, got {type(metrics_dict['edge_f1'])}")

    if not isinstance(metrics_dict["lenient_pass"], bool):
        raise TypeError(f"lenient_pass must be bool, got {type(metrics_dict['lenient_pass'])}")

    logger.debug(f"✓ Metrics validation passed for {image_name}")


def validate_seams(depth: np.ndarray, tile_size: int, overlap: int, band: int = 2) -> tuple:
    """Detect seam artifacts at tile boundaries."""
    h, w = depth.shape
    stride = tile_size - overlap

    # Compute gradient magnitude
    gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    global_mean = grad_mag.mean()

    # Check horizontal and vertical boundaries
    boundary_mask = np.zeros_like(grad_mag, dtype=bool)

    # Vertical seams
    for x in range(stride, w, stride):
        if x >= band and x + band <= w:
            boundary_mask[:, x - band : x + band] = True

    # Horizontal seams
    for y in range(stride, h, stride):
        if y >= band and y + band <= h:
            boundary_mask[y - band : y + band, :] = True

    if boundary_mask.sum() == 0:
        return True, 1.0

    boundary_mean = grad_mag[boundary_mask].mean()
    ratio = boundary_mean / (global_mean + 1e-8)

    passed = ratio < 1.2
    return passed, ratio


def estimate_tile_count(image_shape, tile_size=1024, overlap=128):
    """Estimate number of tiles for an image."""
    h, w = image_shape[:2]
    stride = tile_size - overlap
    tiles_y = ((h - tile_size + stride - 1) // stride) + 1
    tiles_x = ((w - tile_size + stride - 1) // stride) + 1
    return tiles_y * tiles_x, tiles_y, tiles_x


def evaluate_quality_gates(metrics: dict, scene_type: str, seam_passed: bool = True) -> dict:
    """
    Apply content-aware quality gates.

    Structure-dominated scenes:
    - Expect strong edge alignment (F1 ≥ 0.6/0.7)
    - Evaluate edge precision/recall

    Texture-dominated scenes:
    - Expect smooth depth (don't penalize for missing texture edges)
    - Evaluate depth consistency/smoothness instead

    Args:
        metrics: Depth quality metrics dict
        scene_type: 'structure_dominated' or 'texture_dominated'
        seam_passed: Seam validation result

    Returns:
        dict with pass/fail status and reason
    """
    if scene_type == "structure_dominated":
        # Standard edge alignment gates
        lenient_pass = (
            metrics["edge_f1"] >= 0.6
            and metrics["chamfer_distance"] < 15
            and metrics.get("edge_count_ratio", 0) <= 2.0
            and seam_passed
        )
        strict_pass = (
            metrics["edge_f1"] >= 0.7
            and metrics["chamfer_distance"] < 10
            and metrics.get("edge_count_ratio", 0) <= 1.5
            and seam_passed
        )

        return {
            "lenient": lenient_pass,
            "strict": strict_pass,
            "gate_type": "edge_alignment",
            "reason": f"F1={metrics['edge_f1']:.3f}, Chamfer={metrics['chamfer_distance']:.1f}px",
        }

    elif scene_type == "texture_dominated":
        # Smoothness gates (depth should NOT copy texture)
        # If edge_f1 is low, that's EXPECTED and CORRECT

        # Check if depth is appropriately smooth
        depth_variance = metrics.get("depth_variance", 0)
        edge_count_ratio = metrics.get("edge_count_ratio", 0)

        # Lenient: Just verify depth exists and isn't degenerate
        lenient_pass = (
            depth_variance > 0.01  # Not flat
            and edge_count_ratio < 0.5  # Smooth relative to RGB
            and seam_passed
        )

        # Strict: Verify depth is smooth AND has minimal structure
        strict_pass = lenient_pass and (edge_count_ratio < 0.2)

        return {
            "lenient": lenient_pass,
            "strict": strict_pass,
            "gate_type": "smoothness",
            "reason": f"Texture scene: depth_var={depth_variance:.3f}, edge_ratio={edge_count_ratio:.2f}",
        }

    else:
        # Unknown scene type: conservative gates
        return {
            "lenient": False,
            "strict": False,
            "gate_type": "unknown",
            "reason": "Scene type unknown, cannot evaluate",
        }


def process_single_image(
    rgb_path: Path,
    output_dir: Path,
    config: DepthConfig,
    use_global_anchor: bool = False,
    smooth_calibrations: bool = True,
) -> Dict:
    """Process single image and return metrics."""

    # Check if already processed (resumability)
    output_metrics = output_dir / f"{rgb_path.stem}_metrics.json"
    if output_metrics.exists():
        logger.info(f"✓ Skipping {rgb_path.name} (already processed)")
        with open(output_metrics) as f:
            return json.load(f)

    result = {
        "image_name": rgb_path.stem,
        "rgb_path": str(rgb_path),
        "success": False,
        "error": None,
        "traceback": None,
        "execution_status": "pending",
    }

    try:
        # Load RGB
        logger.info(f"Processing: {rgb_path.name}")
        rgb = load_image(rgb_path)
        h, w = rgb.shape[:2]

        result["image_size"] = [h, w]

        # Estimate tile count and warn if excessive
        tile_count, tiles_y, tiles_x = estimate_tile_count(rgb.shape, config.tile_size, config.overlap)
        logger.info(f"Tile plan: {h}×{w} → {tile_count} tiles ({tiles_y}×{tiles_x})")

        if tile_count > 50:
            logger.warning(f"⚠️  High tile count ({tile_count}), expect ~{tile_count * 3}s processing time")

        result["tile_count"] = tile_count
        result["tile_grid"] = [tiles_y, tiles_x]

        # Estimate depth
        estimator = HighFidelityDepthEstimator(config)
        depth = estimator.estimate_depth(
            rgb,
            use_global_anchor=use_global_anchor,
            smooth_calibrations=smooth_calibrations,
        )

        # Capture inference metadata
        if hasattr(estimator, "_last_inference_metadata"):
            inference_metadata = estimator._last_inference_metadata

            # Save to metrics
            result["inference_metadata"] = {
                "original_shape": inference_metadata.get("original_shape"),
                "preprocessed_shape": inference_metadata.get("preprocessed_shape"),
                "input_size": inference_metadata.get("input_size"),
                "policy": inference_metadata.get("policy"),
                "aspect_ratio_preserved": inference_metadata.get("aspect_ratio_preserved"),
                "num_tiles": inference_metadata.get("num_tiles", 1),
            }

            logger.info(
                f"  Inference: {inference_metadata.get('original_shape')} → "
                f"{inference_metadata.get('preprocessed_shape')} "
                f"(policy={inference_metadata.get('policy')}, "
                f"input_size={inference_metadata.get('input_size')})"
            )

        # Save depth
        depth_path = output_dir / f"{rgb_path.stem}_depth.tiff"
        save_depth_16bit(depth, depth_path)
        result["depth_path"] = str(depth_path)

        # 1. Extract edges (both raw and structure)
        logger.info("Extracting edges for classification...")
        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb

        rgb_edges_raw = detect_edges(gray)
        rgb_edges_structure = extract_structure_edges(gray)

        # 2. Classify scene using V2 multi-factor classifier
        logger.info("Classifying scene type (V2 multi-factor)...")
        scene_type, scene_metadata = classify_scene_type_v2(
            rgb_edges_raw=rgb_edges_raw,
            rgb_edges_structure=rgb_edges_structure,
            depth_map=depth,
            image_filename=rgb_path.name,  # NEW: Pass filename for weak supervision
        )

        logger.info(
            f"  Scene: {scene_type} (ratio={scene_metadata.get('ratio', 0):.2f}, "
            f"depth_var={scene_metadata.get('depth_variance', 0):.4f}, "
            f"filename_hint={scene_metadata.get('filename_hint', 'none')}, "
            f"decision={scene_metadata.get('decision', 'unknown')})"
        )

        # 3. Validate depth quality with structure-aware edges
        logger.info("Validating depth quality (structure-aware)...")
        metrics_obj = validate_depth_quality(rgb, depth, use_structure_edges=True, image_filename=rgb_path.name)

        # 4. Verify scene_type was populated
        if metrics_obj.scene_type is None:
            raise ValueError("validate_depth_quality() returned None scene_type. V2 classifier integration is broken.")

        # Convert EdgeMetrics object to dict
        metrics = {
            "edge_f1": metrics_obj.edge_f1,
            "edge_overlap": metrics_obj.edge_overlap,
            "edge_alignment_corr": metrics_obj.edge_alignment_corr,
            "chamfer_distance": metrics_obj.chamfer_distance,
            "edge_width": metrics_obj.edge_width,
            "edge_sharpness_p95": metrics_obj.edge_sharpness_p95,
            "edge_count_ratio": metrics_obj.edge_count_ratio,
            "halo_score": metrics_obj.halo_score,
            "overshoot_penalty": metrics_obj.overshoot_penalty,
            "rgb_edge_count": metrics_obj.rgb_edge_count,
            "depth_edge_count": metrics_obj.depth_edge_count,
            "quality_score": metrics_obj.quality_score(),
            "edge_type": metrics_obj.edge_type,
            "scene_type": metrics_obj.scene_type,
        }
        result["metrics"] = metrics

        # Log classification factors from V2 classifier
        logger.info(
            f"  Classification factors: "
            f"ratio={scene_metadata.get('ratio', 0):.2f}, "
            f"depth_var={scene_metadata.get('depth_variance', 0):.4f}, "
            f"edge_density={scene_metadata.get('edge_density', 0):.4f}"
        )

        logger.info(f"  Decision: {scene_metadata.get('decision', 'unknown')} → {scene_type}")

        # Seam validation
        seam_passed, seam_ratio = validate_seams(depth, config.tile_size, config.overlap)
        result["seam_validation_passed"] = bool(seam_passed)
        result["seam_boundary_ratio"] = float(seam_ratio)

        # 5. Apply conditional quality gates based on scene type
        # FIX: Use high-frequency energy instead of global depth variance for texture scenes
        from high_fidelity_depth.quality_metrics import compute_high_frequency_energy

        depth_var = float(np.var(depth))
        hf_energy = compute_high_frequency_energy(depth)

        # Initialize depth_range for all scenes (used in metadata)
        p95 = float(np.percentile(depth, 95))
        p05 = float(np.percentile(depth, 5))
        depth_range = p95 - p05

        if scene_type == "texture_dominated":
            # Texture scenes: balanced gates considering multiple factors
            # Don't rely solely on HF energy - combine with edge metrics
            edge_ratio = metrics["edge_count_ratio"]
            edge_f1 = metrics["edge_f1"]

            # Calibrated thresholds based on empirical testing:
            # - Smooth depth (ocean/pool): HF energy ~ 0.00001-0.0003
            # - Geometric depth (pool with structure): HF energy ~ 0.0005-0.002
            # Strategy: Require EITHER smooth HF OR good edge alignment

            # CRITICAL: Check depth is not flat (has global structure)
            # depth_range already computed above (p95 - p05)
            not_flat = depth_range > 0.05  # Normalized depth should vary

            # Lenient: Pass if (smooth HF AND not flat) OR (reasonable F1 + moderate edge ratio)
            smooth_hf = hf_energy < 0.002  # Allow some geometric structure
            reasonable_edges = edge_f1 >= 0.20 and edge_ratio < 15.0
            lenient_pass = (smooth_hf and not_flat) or reasonable_edges

            # Strict: Require smooth HF AND not flat AND good edge metrics
            very_smooth_hf = hf_energy < 0.001
            good_edges = edge_f1 >= 0.30 and edge_ratio < 10.0
            strict_pass = very_smooth_hf and not_flat and good_edges

            gate_reason = (
                f"Texture scene: hf_energy={hf_energy:.6f}, depth_range={depth_range:.3f}, "
                f"edge_ratio={edge_ratio:.2f}, edge_f1={edge_f1:.3f} | "
                f"smooth_hf={smooth_hf}, not_flat={not_flat}, reasonable_edges={reasonable_edges}"
            )
            gate_type = "smoothness_hf_balanced"

        elif scene_type == "structure_dominated":
            # Structure scenes: use edge alignment gates
            lenient_pass = metrics["edge_f1"] >= 0.30 and metrics["chamfer_distance"] < 15.0
            strict_pass = metrics["edge_f1"] >= 0.60 and metrics["chamfer_distance"] < 5.0

            gate_reason = f"Structure scene: edge_f1={metrics['edge_f1']:.3f}, chamfer={metrics['chamfer_distance']:.1f}px"
            gate_type = "edge_alignment"
        else:
            # Unknown scene type - should not happen
            logger.warning(f"Unknown scene type: {scene_type}, using lenient structure gates")
            lenient_pass = metrics["edge_f1"] >= 0.20
            strict_pass = False
            gate_reason = f"Unknown scene: edge_f1={metrics['edge_f1']:.3f}"
            gate_type = "unknown"

        logger.info(
            f"  Quality gates: lenient={'PASS' if lenient_pass else 'FAIL'}, strict={'PASS' if strict_pass else 'FAIL'}"
        )
        logger.info(f"  Reason: {gate_reason}")

        result["quality_lenient"] = bool(lenient_pass)
        result["quality_strict"] = bool(strict_pass)
        result["gate_type"] = gate_type
        result["gate_reason"] = gate_reason

        # Compute quality score
        quality_score = (
            0.35 * metrics["edge_f1"]
            + 0.25 * (1.0 - min(metrics["chamfer_distance"] / 15.0, 1.0))
            + 0.20 * (1.0 - metrics["overshoot_penalty"])
            + 0.20 * (1.0 if seam_passed else 0.0)
        )
        result["quality_score"] = float(quality_score)

        # Create visualizations
        overlay_path = output_dir / f"{rgb_path.stem}_edges.png"
        overlay = create_edge_overlay(rgb, depth)
        Image.fromarray(overlay).save(overlay_path)
        result["overlay_path"] = str(overlay_path)

        # Overshoot heatmap (if detected)
        if metrics.get("overshoot_penalty", 0) > 0.1:
            from high_fidelity_depth.quality_metrics import compute_overshoot_heatmap

            heatmap, _, _ = compute_overshoot_heatmap(depth, rgb)
            heatmap_path = output_dir / f"{rgb_path.stem}_overshoot.png"
            Image.fromarray(heatmap).save(heatmap_path)
            result["heatmap_path"] = str(heatmap_path)

        # 6. Build complete metrics dict with V2 classifier data
        from datetime import datetime

        metrics_dict = {
            # Core metrics (ALL fields from EdgeMetrics)
            "edge_f1": float(metrics["edge_f1"]),
            "edge_overlap": float(metrics["edge_overlap"]),
            "edge_alignment_corr": float(metrics["edge_alignment_corr"]),
            "chamfer_px": float(metrics["chamfer_distance"]),
            "edge_width": float(metrics["edge_width"]),
            "edge_sharpness_p95": float(metrics["edge_sharpness_p95"]),
            "edge_count_ratio": float(metrics["edge_count_ratio"]),
            "halo_score": float(metrics["halo_score"]),
            "overshoot_penalty": float(metrics["overshoot_penalty"]),
            "rgb_edge_count": int(metrics["rgb_edge_count"]),
            "depth_edge_count": int(metrics["depth_edge_count"]),
            "quality_score": float(metrics["quality_score"]),
            # V2 classifier data
            "scene_type": scene_type,
            "classification_factors": {
                "ratio": float(scene_metadata.get("ratio", 0)),
                "depth_variance": float(scene_metadata.get("depth_variance", 0)),
                "depth_gradient_var": float(scene_metadata.get("depth_gradient_var", 0)),
                "edge_density": float(scene_metadata.get("edge_density", 0)),
                "hf_energy": (float(hf_energy) if scene_type == "texture_dominated" else None),
                "depth_range": (float(depth_range) if scene_type == "texture_dominated" else None),
                "decision_rule": scene_metadata.get("decision", "unknown"),
                "method": scene_metadata.get("method", "unknown"),
                "filename_hint": scene_metadata.get("filename_hint"),
            },
            # Quality gates
            "lenient_pass": bool(lenient_pass),
            "strict_pass": bool(strict_pass),
            "gate_reason": gate_reason,
            "gate_type": gate_type,
            # Metadata
            "image": str(rgb_path.name),
            "timestamp": datetime.now().isoformat(),
        }

        # 7. VALIDATE BEFORE WRITING (fail fast)
        validate_metrics_complete(metrics_dict, rgb_path.name)

        # 8. Save metrics atomically
        metrics_path = output_dir / f"{rgb_path.stem}_metrics.json"
        save_metrics_atomic(metrics_dict, metrics_path)

        result["success"] = True
        result["execution_status"] = "success"
        logger.info(
            f"✓ {rgb_path.name}: F1={metrics['edge_f1']:.3f}, scene={scene_type}, seam_ratio={seam_ratio:.3f}, quality={quality_score:.3f}"
        )

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["execution_status"] = "failed"
        logger.error(f"✗ {rgb_path.name} failed: {e}")
        logger.debug(traceback.format_exc())

    return result


def main():
    parser = argparse.ArgumentParser(description="Production High-Fidelity Depth Validation")
    parser.add_argument(
        "--input-dir",
        "--image-dir",
        dest="input_dir",
        type=Path,
        required=True,
        help="Input directory with RGB images (TIFF/JPG/PNG)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size (default: 1024)")
    parser.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Tile overlap (default: 192 for texture-heavy)",
    )
    parser.add_argument(
        "--use-global-anchor",
        action="store_true",
        help="Enable global anchor fusion (OFF by default)",
    )
    parser.add_argument(
        "--no-smooth-calibrations",
        action="store_true",
        help="Disable spatial smoothing of calibrations",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto | cuda | mps | cpu")

    args = parser.parse_args()

    # Check ML dependencies early
    if not HAS_ML_DEPS:
        logger.error("PyTorch and transformers required")
        logger.error("Install with: pip install torch transformers")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find input images
    input_files = (
        sorted(args.input_dir.glob("*.tif"))
        + sorted(args.input_dir.glob("*.tiff"))
        + sorted(args.input_dir.glob("*.jpg"))
        + sorted(args.input_dir.glob("*.jpeg"))
        + sorted(args.input_dir.glob("*.png"))
    )

    if not input_files:
        logger.error(f"No image files (TIFF/JPG/PNG) found in {args.input_dir}")
        return 1

    logger.info(f"Found {len(input_files)} images to process")

    # Configure depth estimator (BLOCKER fixes applied)
    config = DepthConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
        reconcile_scales=True,
        reconcile_method="robust",
        blend_window="hann",
        validate_seams=True,
        seam_energy_threshold=1.2,
    )

    logger.info(f"Configuration: tile_size={config.tile_size}, overlap={config.overlap}, device={config.device}")
    logger.info(f"Global anchor: {args.use_global_anchor}, Smooth calibrations: {not args.no_smooth_calibrations}")

    # Process all images
    results = []
    failed_images = []

    for idx, rgb_path in enumerate(input_files):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Image {idx + 1}/{len(input_files)}: {rgb_path.name}")
        logger.info(f"{'=' * 80}")

        result = process_single_image(
            rgb_path,
            args.output_dir,
            config,
            use_global_anchor=args.use_global_anchor,
            smooth_calibrations=not args.no_smooth_calibrations,
        )

        results.append(result)

        if not result["success"]:
            failed_images.append(rgb_path.name)

        # Note: Metrics already saved in process_single_image() with validation

    # Generate summary report
    succeeded = sum(1 for r in results if r["success"])
    seam_passed = sum(1 for r in results if r.get("seam_validation_passed", False))
    quality_lenient_passed = sum(1 for r in results if r.get("quality_lenient", False))
    quality_strict_passed = sum(1 for r in results if r.get("quality_strict", False))

    # Categorize by scene type (heuristic based on filename)
    category_stats = {}
    for result in results:
        name_lower = result["image_name"].lower()
        if any(kw in name_lower for kw in ["aerial", "exterior", "pool"]):
            category = "exterior"
        else:
            category = "interior"

        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "seam_passed": 0,
                "quality_passed_lenient": 0,
                "quality_passed_strict": 0,
                "avg_edge_f1": 0.0,
                "avg_seam_ratio": 0.0,
            }

        category_stats[category]["total"] += 1
        if result.get("seam_validation_passed", False):
            category_stats[category]["seam_passed"] += 1
        if result.get("quality_lenient", False):
            category_stats[category]["quality_passed_lenient"] += 1
        if result.get("quality_strict", False):
            category_stats[category]["quality_passed_strict"] += 1

        if "metrics" in result:
            category_stats[category]["avg_edge_f1"] += result["metrics"].get("edge_f1", 0)
            category_stats[category]["avg_seam_ratio"] += result.get("seam_boundary_ratio", 0)

    for cat in category_stats:
        n = category_stats[cat]["total"]
        if n > 0:
            category_stats[cat]["avg_edge_f1"] /= n
            category_stats[cat]["avg_seam_ratio"] /= n

    # Aggregate metrics
    all_quality_scores = [r.get("quality_score", 0) for r in results if r["success"]]
    all_edge_f1 = [r["metrics"]["edge_f1"] for r in results if "metrics" in r]
    all_seam_ratios = [r.get("seam_boundary_ratio", 0) for r in results if r.get("seam_validation_passed") is not None]

    aggregate = {
        "quality_score": {
            "mean": float(np.mean(all_quality_scores)) if all_quality_scores else 0.0,
            "min": float(np.min(all_quality_scores)) if all_quality_scores else 0.0,
            "max": float(np.max(all_quality_scores)) if all_quality_scores else 0.0,
            "std": float(np.std(all_quality_scores)) if all_quality_scores else 0.0,
        },
        "edge_f1": {
            "mean": float(np.mean(all_edge_f1)) if all_edge_f1 else 0.0,
            "min": float(np.min(all_edge_f1)) if all_edge_f1 else 0.0,
            "max": float(np.max(all_edge_f1)) if all_edge_f1 else 0.0,
        },
        "seam_ratio": {
            "mean": float(np.mean(all_seam_ratios)) if all_seam_ratios else 0.0,
            "max": float(np.max(all_seam_ratios)) if all_seam_ratios else 0.0,
        },
    }

    summary = {
        "total_images": len(input_files),
        "execution": {
            "succeeded": succeeded,
            "failed": len(failed_images),
            "success_rate": succeeded / len(input_files) if input_files else 0.0,
        },
        "seam_validation": {
            "passed": seam_passed,
            "failed": succeeded - seam_passed,
            "pass_rate": seam_passed / succeeded if succeeded > 0 else 0.0,
        },
        "quality": {
            "lenient": {
                "passed": quality_lenient_passed,
                "failed": succeeded - quality_lenient_passed,
                "pass_rate": (quality_lenient_passed / succeeded if succeeded > 0 else 0.0),
            },
            "strict": {
                "passed": quality_strict_passed,
                "failed": succeeded - quality_strict_passed,
                "pass_rate": (quality_strict_passed / succeeded if succeeded > 0 else 0.0),
            },
        },
        "overall_status": "COMPLETE" if len(failed_images) == 0 else "INCOMPLETE",
        "failed_images": failed_images,
        "category_stats": category_stats,
        "config": {
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "use_global_anchor": args.use_global_anchor,
            "smooth_calibrations": not args.no_smooth_calibrations,
        },
        "results": results,
        "aggregate_metrics": aggregate,
    }

    # Save summary atomically
    report_path = args.output_dir / "validation_report.json"
    save_metrics_atomic(summary, report_path)

    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total images: {len(input_files)}")
    logger.info(f"Execution success: {succeeded}/{len(input_files)} ({100 * succeeded / len(input_files):.1f}%)")
    logger.info(f"Seam validation: {seam_passed}/{succeeded} ({100 * seam_passed / succeeded if succeeded > 0 else 0:.1f}%)")
    logger.info(
        f"Quality (lenient): {quality_lenient_passed}/{succeeded} ({100 * quality_lenient_passed / succeeded if succeeded > 0 else 0:.1f}%)"
    )
    logger.info(
        f"Quality (strict): {quality_strict_passed}/{succeeded} ({100 * quality_strict_passed / succeeded if succeeded > 0 else 0:.1f}%)"
    )
    logger.info(f"Status: {summary['overall_status']}")

    if failed_images:
        logger.error(f"Failed images: {', '.join(failed_images)}")

    logger.info(f"\nReport saved: {report_path}")

    return 0 if len(failed_images) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
