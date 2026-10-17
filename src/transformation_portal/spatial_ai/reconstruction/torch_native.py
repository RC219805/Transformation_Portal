"""Validation, bounded metric initialization, and provenance for native 3DGS."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .contracts import CameraParams, GaussianSplat, ReconstructionInput
from .protocol import NATIVE_DEFAULT_RECIPE, NATIVE_RECIPES

MAX_NATIVE_TRAINING_GAUSSIAN_PIXELS = 64_000_000
MAX_NATIVE_SPACING_POINTS = 10_000
NATIVE_POSITION_PRIOR_WEIGHT = 0.01
NATIVE_LOG_SCALE_PRIOR_WEIGHT = 0.001
MAX_NATIVE_SHUFFLED_UPDATES = 30_000
NATIVE_DEPTH_WEIGHT = 0.1
NATIVE_DEPTH_NORMALIZATION_METRES = 1.0
NATIVE_COVERAGE_ALPHA_THRESHOLD = 0.5
NATIVE_MAX_COVERAGE_DROP = 0.02


def metric_neighbor_scales(positions: np.ndarray) -> np.ndarray:
    """Estimate global point spacing with bounded NumPy nearest-neighbor batches.

    Per-view sampling density overestimates scale when calibrated views overlap.
    This needs no undeclared spatial-tree dependency and never allocates an
    N-by-N matrix. The explicit native point limit bounds quadratic work.
    """
    count = len(positions)
    if count < 2 or count > MAX_NATIVE_SPACING_POINTS:
        raise ValueError(f"torch_native metric spacing requires 2..{MAX_NATIVE_SPACING_POINTS} points")
    neighbors = min(3, count - 1)
    sigma = np.empty(count, dtype=np.float32)
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        difference = positions[start:stop, None, :] - positions[None, :, :]
        distance_squared = np.einsum("bnc,bnc->bn", difference, difference)
        distance_squared[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.partition(distance_squared, neighbors - 1, axis=1)[:, :neighbors]
        sigma[start:stop] = np.sqrt(nearest.mean(axis=1))
    return np.repeat(np.maximum(sigma, 1e-4)[:, None], 3, axis=1)


def validate_native_camera(camera: CameraParams, context: str) -> None:
    """Revalidate mutable camera arrays for the rigid, undistorted pinhole path."""
    camera.__post_init__()
    intrinsics, extrinsics = camera.intrinsics, camera.extrinsics
    if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError(f"{context}: finite positive pinhole focal lengths required")
    if not np.allclose(intrinsics[[0, 1], [1, 0]], 0, atol=1e-6):
        raise ValueError(f"{context}: skewed camera intrinsics are unsupported")
    if not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-6):
        raise ValueError(f"{context}: pinhole intrinsics bottom row must be [0, 0, 1]")
    rotation = extrinsics[:3, :3]
    if (
        not np.isfinite(extrinsics).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
    ):
        raise ValueError(f"{context}: rigid world-to-camera extrinsics required")
    if camera.distortion is not None and np.any(camera.distortion != 0):
        raise ValueError(f"{context}: images must be undistorted")


def validate_native_input(
    reconstruction_input: ReconstructionInput,
    iterations: int,
    use_depth_prior: bool,
    use_segmentation: bool,
    use_pbr_textures: bool,
) -> None:
    """Reject inputs the native metric, undistorted pinhole path cannot honor."""
    reconstruction_input.__post_init__()
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("torch_native requires a positive integer iteration count")
    if not use_depth_prior or reconstruction_input.depth_maps is None:
        raise ValueError("torch_native requires supplied metric depth priors; fixed-depth initialization is unsupported")
    if use_pbr_textures or reconstruction_input.material_maps is not None:
        raise ValueError("torch_native does not support PBR material maps")
    if any(image.shape[:2] != reconstruction_input.images[0].shape[:2] for image in reconstruction_input.images):
        raise ValueError("torch_native requires matching training image spatial dimensions")
    for index, (image, depth, camera) in enumerate(
        zip(reconstruction_input.images, reconstruction_input.depth_maps, reconstruction_input.cameras)
    ):
        validate_native_camera(camera, f"torch_native view {index}")
        if image.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"torch_native view {index}: image and camera dimensions differ")
        if not np.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError(f"torch_native view {index}: linear RGB must be finite and within [0, 1]")
        mask = (
            reconstruction_input.masks[index]
            if use_segmentation and reconstruction_input.masks is not None
            else np.ones(depth.shape, dtype=bool)
        )
        selected_depth = depth[mask]
        if selected_depth.size == 0 or not np.isfinite(selected_depth).all() or (selected_depth <= 0).any():
            raise ValueError(f"torch_native view {index}: selected metric depth must be finite, positive, and nonempty")


def initialize_metric_splats(
    reconstruction_input: ReconstructionInput,
    use_segmentation: bool,
    max_gaussians: int,
    seed: int | None,
) -> GaussianSplat:
    """Unproject a deterministic, view-balanced sample within the native cap.

    Depth values are camera-axis distances in metres, supplied by the caller.
    A local RNG covers initialization without changing ambient RNG state.
    """
    num_views = reconstruction_input.num_views
    if max_gaussians < num_views:
        raise ValueError("torch_native Gaussian cap must allow at least one point per view")
    rng = np.random.default_rng(seed)
    positions, colors, support_counts = [], [], []
    for index, (image, depth, camera) in enumerate(
        zip(reconstruction_input.images, reconstruction_input.depth_maps, reconstruction_input.cameras)
    ):
        mask = (
            reconstruction_input.masks[index]
            if use_segmentation and reconstruction_input.masks is not None
            else np.ones(depth.shape, dtype=bool)
        )
        valid = np.flatnonzero(mask)
        support_counts.append(int(valid.size))
        count = min(valid.size, max_gaussians // num_views + int(index < max_gaussians % num_views))
        if count < valid.size:
            valid = np.sort(rng.choice(valid, count, replace=False))
        v, u = np.unravel_index(valid, depth.shape)
        z = depth[v, u]
        intrinsics = camera.intrinsics
        camera_points = np.column_stack(
            [
                (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
                np.ones(count),
            ]
        )
        positions.append((np.linalg.inv(camera.extrinsics) @ camera_points.T).T[:, :3])
        colors.append(image[v, u])
    count = sum(len(points) for points in positions)
    # Native backward accumulation frees each view's graph before the next.
    gaussian_pixels = count * max(image.shape[0] * image.shape[1] for image in reconstruction_input.images)
    if gaussian_pixels > MAX_NATIVE_TRAINING_GAUSSIAN_PIXELS:
        raise ValueError(
            "torch_native full-image training exceeds the bounded Gaussian-pixel budget "
            f"({gaussian_pixels} > {MAX_NATIVE_TRAINING_GAUSSIAN_PIXELS}); reduce image resolution or Gaussian cap"
        )
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    world_positions = np.concatenate(positions).astype(np.float32)
    return GaussianSplat(
        positions=world_positions,
        colors=np.concatenate(colors).astype(np.float32),
        scales=metric_neighbor_scales(world_positions),
        rotations=rotations,
        opacities=np.full((count, 1), 0.5, dtype=np.float32),
        metadata={
            "initialization": "metric_depth",
            "input_depth_support_per_view": support_counts,
            "initialization_gaussian_cap": max_gaussians,
            "training_gaussian_pixels": gaussian_pixels,
            "training_gaussian_pixel_budget": MAX_NATIVE_TRAINING_GAUSSIAN_PIXELS,
            "scale_initialization": "global_metric_nearest_neighbors",
        },
    )


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def native_provenance(
    reconstruction_input: ReconstructionInput,
    device: str,
    seed: int | None,
    recipe: str = NATIVE_DEFAULT_RECIPE,
    optimizer_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the actual local implementation, inputs, and runtime without a model claim."""
    sources = {}
    for name in ("gaussian_backend.py", "gaussian_rasterizer.py", "torch_native.py", "protocol.py"):
        sources[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    views = []
    for index, (image, camera, depth) in enumerate(
        zip(reconstruction_input.images, reconstruction_input.cameras, reconstruction_input.depth_maps)
    ):
        view = {
            "linear_rgb": _array_identity(image),
            "metric_depth": _array_identity(depth),
            "intrinsics": _array_identity(camera.intrinsics),
            "world_to_camera": _array_identity(camera.extrinsics),
        }
        if reconstruction_input.masks is not None:
            view["mask"] = _array_identity(reconstruction_input.masks[index])
        views.append(view)
    return {
        "implementation": "torch_native",
        "initialization": "metric_depth_from_scratch",
        "pretrained_checkpoint": None,
        "source_sha256": sources,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": device,
        },
        "optimization_seed": seed,
        "native_recipe": recipe,
        "optimizer": {
            "scales": "log_parameterization",
            "position_prior_weight": NATIVE_POSITION_PRIOR_WEIGHT,
            "log_scale_prior_weight": NATIVE_LOG_SCALE_PRIOR_WEIGHT,
            "gradient_accumulation": "one_view_graph_at_a_time",
            **(optimizer_details or {}),
        },
        "views": views,
        "limitations": [
            "isotropic_gaussians" if recipe == NATIVE_DEFAULT_RECIPE else "bounded_anisotropic_gaussians",
            "no_spherical_harmonics_or_densification",
            (
                "alpha_weighted_depth_supervision_is_not_independent_surface_certification"
                if recipe == "anisotropic_shuffled_rgbd_v1"
                else "metric_surface_anchor_is_not_a_rendered_depth_loss"
            ),
            "bounded_full_image_torch_compositor_not_gsplat_cuda",
            "training_rmse_does_not_establish_held_out_quality",
        ],
    }


def metric_depth_supervision_loss(
    expected_depth: torch.Tensor, target_depth: torch.Tensor, fixed_valid_mask: torch.Tensor
) -> torch.Tensor:
    """MSE with a fixed supplied-depth mask and an explicit one-metre scale.

    Prediction-dependent masking would let unsupported geometry hide errors.
    Zero-alpha pixels therefore retain their zero-depth error in the objective.
    """
    if not fixed_valid_mask.any():
        raise ValueError("Native metric depth supervision requires nonempty input support")
    target = target_depth[fixed_valid_mask]
    if not torch.isfinite(target).all() or not (target > 0).all():
        raise ValueError("Native metric depth support must be finite and positive")
    return ((expected_depth[fixed_valid_mask] - target) / NATIVE_DEPTH_NORMALIZATION_METRES).square().mean()


def optimize_shuffled_metric_splats(
    splats: GaussianSplat,
    reconstruction_input: ReconstructionInput,
    iterations: int,
    device: str,
    seed: int,
    recipe: str,
    use_segmentation: bool,
    rmse_threshold: float,
) -> tuple[GaussianSplat, float, Literal["converged", "max_iterations"], int]:
    """Run an explicit, bounded native recipe with training-only selection.

    An iteration is one camera update, unlike the unchanged full-batch default.
    RGBD optimizes a measured geometry/photometry tradeoff; it is not proof of
    photographic quality or independently measured physical surface accuracy.
    """
    from .gaussian_rasterizer import render_gaussians_anisotropic_native

    if recipe not in NATIVE_RECIPES[1:]:
        raise ValueError("Shuffled native optimization requires an anisotropic recipe")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Shuffled native recipes require an explicit nonnegative integer optimization_seed")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= MAX_NATIVE_SHUFFLED_UPDATES:
        raise ValueError(f"Shuffled native recipes require 1..{MAX_NATIVE_SHUFFLED_UPDATES} optimizer updates")

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=device)

    params = {
        name: tensor(getattr(splats, name)).requires_grad_() for name in ("positions", "colors", "rotations", "opacities")
    }
    log_scales = tensor(np.log(splats.scales)).requires_grad_()
    initial_positions = params["positions"].detach().clone()
    initial_scales = tensor(splats.scales)
    initial_log_scales = log_scales.detach().clone()
    images = [tensor(image) for image in reconstruction_input.images]
    depths = [tensor(depth) for depth in reconstruction_input.depth_maps]
    masks = []
    for index, depth in enumerate(depths):
        mask = torch.isfinite(depth) & (depth > 0)
        if use_segmentation and reconstruction_input.masks is not None:
            mask = mask & torch.tensor(reconstruction_input.masks[index], device=device, dtype=torch.bool)
        masks.append(mask)
    cameras = [
        (tensor(camera.intrinsics), tensor(camera.extrinsics), (camera.height, camera.width))
        for camera in reconstruction_input.cameras
    ]
    learning_rates = {"positions": 0.00016, "colors": 0.0025, "log_scales": 0.005, "rotations": 0.001, "opacities": 0.05}
    trainable = {**params, "log_scales": log_scales}
    optimizer = torch.optim.Adam(
        [{"params": [parameter], "lr": learning_rates[name]} for name, parameter in trainable.items()]
    )
    depth_weight = NATIVE_DEPTH_WEIGHT if recipe == "anisotropic_shuffled_rgbd_v1" else 0.0
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < iterations:
        schedule.extend(rng.permutation(len(images)).tolist())
    schedule = schedule[:iterations]

    def render(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsics, extrinsics, size = cameras[index]
        return render_gaussians_anisotropic_native(
            params["positions"],
            params["colors"],
            log_scales.exp(),
            params["rotations"],
            params["opacities"],
            intrinsics,
            extrinsics,
            size,
            device=device,
        )

    def priors() -> torch.Tensor:
        position = ((params["positions"] - initial_positions) / initial_scales).square().mean()
        scale = (log_scales - initial_log_scales).square().mean()
        return NATIVE_POSITION_PRIOR_WEIGHT * position + NATIVE_LOG_SCALE_PRIOR_WEIGHT * scale

    def evaluate() -> dict[str, Any]:
        rgb_losses, depth_losses, coverage = [], [], []
        with torch.no_grad():
            for index, target in enumerate(images):
                rgb, expected_depth, alpha = render(index)
                rgb_losses.append(float((rgb - target).square().mean()))
                depth_losses.append(float(metric_depth_supervision_loss(expected_depth, depths[index], masks[index])))
                coverage.append(float((alpha[masks[index]] >= NATIVE_COVERAGE_ALPHA_THRESHOLD).float().mean()))
            objective = float(np.mean(rgb_losses) + depth_weight * np.mean(depth_losses) + priors())
        if not np.isfinite([objective, *rgb_losses, *depth_losses, *coverage]).all():
            raise RuntimeError("Native recipe produced nonfinite training metrics")
        return {
            "rgb_mse": float(np.mean(rgb_losses)),
            "depth_mse_metres_squared": float(np.mean(depth_losses)),
            "objective": objective,
            "coverage_per_view": coverage,
        }

    def snapshot() -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in trainable.items()}

    baseline = evaluate()
    best_metrics = baseline
    best = snapshot()
    selected_step = 0
    checkpoints = [{"step": 0, "objective": baseline["objective"], "coverage_guard_passed": True}]
    for step, index in enumerate(schedule, 1):
        optimizer.zero_grad()
        rgb, expected_depth, _ = render(index)
        rgb_loss = (rgb - images[index]).square().mean()
        depth_loss = metric_depth_supervision_loss(expected_depth, depths[index], masks[index])
        loss = rgb_loss + depth_weight * depth_loss + priors()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Native recipe produced nonfinite loss at update {step}")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable.values()):
            raise RuntimeError(f"Native recipe produced missing or nonfinite gradients at update {step}")
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            log_scales.clamp_(float(np.log(1e-6)), float(np.log(10.0)))
            params["colors"].clamp_(0, 1)
            params["opacities"].clamp_(0.01, 1)
            rotation_norm = params["rotations"].norm(dim=1, keepdim=True)
            if not torch.isfinite(rotation_norm).all() or not (rotation_norm > 1e-8).all():
                raise RuntimeError("Native recipe produced an invalid quaternion")
            params["rotations"].div_(rotation_norm)
        if not all(torch.isfinite(parameter).all() for parameter in trainable.values()):
            raise RuntimeError(f"Native recipe produced nonfinite parameters at update {step}")
        if step % len(images) == 0 or step == iterations:
            measured = evaluate()
            eligible = not depth_weight or all(
                actual >= original - NATIVE_MAX_COVERAGE_DROP
                for actual, original in zip(measured["coverage_per_view"], baseline["coverage_per_view"])
            )
            checkpoints.append({"step": step, "objective": measured["objective"], "coverage_guard_passed": bool(eligible)})
            if eligible and measured["objective"] < best_metrics["objective"]:
                best_metrics, best, selected_step = measured, snapshot(), step
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(best[name])
    final = evaluate()
    if depth_weight and any(
        actual < original - NATIVE_MAX_COVERAGE_DROP
        for actual, original in zip(final["coverage_per_view"], baseline["coverage_per_view"])
    ):
        raise RuntimeError("Native RGBD saved checkpoint lost supplied-depth alpha coverage")
    scales = log_scales.detach().exp().cpu().numpy()
    arrays = {name: parameter.detach().cpu().numpy() for name, parameter in params.items()}
    for _, extrinsics, _ in cameras:
        homogeneous = torch.cat([params["positions"], torch.ones(len(scales), 1, device=device)], dim=1)
        depth = (extrinsics @ homogeneous.T).T[:, 2]
        if not torch.isfinite(depth).all() or not (depth > 0.01).any():
            raise RuntimeError("Native recipe saved geometry has no finite forward depth support")
    details = {
        "native_recipe": recipe,
        "iterations_meaning": "single_camera_optimizer_updates",
        "performed_updates": iterations,
        "selected_checkpoint_step": selected_step,
        "training_convergence_rmse_threshold": rmse_threshold,
        "checkpoint_selection": "minimum_full_training_rgb_plus_weighted_depth_plus_original_metric_priors",
        "checkpoint_interval_updates": len(images),
        "camera_schedule": schedule,
        "learning_rates": learning_rates,
        "gradient_accumulation": "single_shuffled_camera_per_update",
        "depth_weight": depth_weight,
        "depth_normalization_metres": NATIVE_DEPTH_NORMALIZATION_METRES,
        "depth_mask": "fixed_supplied_valid_depth_and_enabled_input_mask",
        "coverage_alpha_threshold": NATIVE_COVERAGE_ALPHA_THRESHOLD,
        "maximum_coverage_drop": NATIVE_MAX_COVERAGE_DROP if depth_weight else None,
        "initial_coverage_per_view": baseline["coverage_per_view"],
        "selected_coverage_per_view": final["coverage_per_view"],
        "checkpoints": checkpoints,
        "final_training_objective": final["objective"],
        "expected_depth_rmse_metres": float(np.sqrt(final["depth_mse_metres_squared"])),
        "maximum_metric_position_displacement_metres": float(
            np.linalg.norm(arrays["positions"] - splats.positions, axis=1).max()
        ),
        "scale_ratio_to_original_metric_min_median_max": np.percentile(scales / splats.scales, [0, 50, 100]).tolist(),
    }
    optimized = GaussianSplat(
        **arrays,
        scales=scales,
        metadata={
            **splats.metadata,
            "native_recipe": recipe,
            "native_optimizer": details,
            "rmse_scope": "all_training_views_selected_checkpoint",
            "optimized": True,
        },
    )
    rmse = float(np.sqrt(final["rgb_mse"]))
    return optimized, rmse, "converged" if rmse < rmse_threshold else "max_iterations", iterations
