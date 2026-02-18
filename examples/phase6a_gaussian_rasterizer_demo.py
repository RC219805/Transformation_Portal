#!/usr/bin/env python3
"""Phase 6A Example: Simplified Gaussian Splatting Rasterizer.

Demonstrates end-to-end 3D reconstruction with the new differentiable rasterizer.

Usage:
    python examples/phase6a_gaussian_rasterizer_demo.py
"""

import logging

import numpy as np

from transformation_portal.spatial_ai.reconstruction import CameraParams, GaussianBackend, ReconstructionInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_scene():
    """Create a simple synthetic multi-view scene for testing."""
    # Create 2 views of a simple scene
    H, W = 240, 320

    # Simple gradient image (easier to reconstruct)
    image1 = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        image1[i, :, 0] = i / H  # Red gradient vertically

    image2 = np.zeros((H, W, 3), dtype=np.float32)
    for j in range(W):
        image2[:, j, 1] = j / W  # Green gradient horizontally

    images = [image1, image2]

    # Camera parameters (simple pinhole)
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = intrinsics[1, 1] = 300.0  # Focal length
    intrinsics[0, 2] = W / 2  # Principal point x
    intrinsics[1, 2] = H / 2  # Principal point y

    # Camera 1: at origin looking down +Z
    extrinsics1 = np.eye(4, dtype=np.float32)

    # Camera 2: slightly offset
    extrinsics2 = np.eye(4, dtype=np.float32)
    extrinsics2[0, 3] = 0.5  # Shift 0.5m along X-axis

    cameras = [
        CameraParams(intrinsics, extrinsics1, W, H),
        CameraParams(intrinsics, extrinsics2, W, H),
    ]

    # Optional: add depth maps for better initialization
    depth_maps = [
        np.ones((H, W), dtype=np.float32) * 5.0,  # 5 meters constant depth
        np.ones((H, W), dtype=np.float32) * 5.0,
    ]

    return images, cameras, depth_maps


def main():
    """Run Phase 6A Gaussian Splatting demo."""
    logger.info("=" * 60)
    logger.info("Phase 6A: Simplified Gaussian Splatting Rasterizer Demo")
    logger.info("=" * 60)

    # Create synthetic scene
    logger.info("\n1. Creating synthetic multi-view scene...")
    images, cameras, depth_maps = create_synthetic_scene()
    logger.info(f"   ✓ Created {len(images)} views ({images[0].shape[0]}x{images[0].shape[1]})")

    # Initialize backend
    logger.info("\n2. Initializing Gaussian Splatting backend...")
    backend = GaussianBackend(tier="apex_research")
    logger.info(f"   ✓ Backend ready (device: {backend.device})")

    # Prepare reconstruction input
    reconstruction_input = ReconstructionInput(
        images=images,
        gamma=1.0,  # Linear RGB
        cameras=cameras,
        depth_maps=depth_maps,  # Use depth-guided initialization
        tier="apex_research",
    )

    # Run reconstruction with reduced iterations for demo
    logger.info("\n3. Running 3D reconstruction (100 iterations)...")
    scene = backend.reconstruct(
        reconstruction_input,
        iterations=100,  # Reduced from 1000 for fast demo
        use_depth_prior=True,
    )

    logger.info(f"   ✓ Reconstruction complete!")
    logger.info(f"     - Gaussians: {scene.splats.num_gaussians}")
    logger.info(f"     - RMSE: {scene.rmse:.6f}")
    logger.info(f"     - Convergence: {scene.convergence}")
    logger.info(f"     - Time: {scene.metadata['elapsed_seconds']:.1f}s")

    # Render novel view
    logger.info("\n4. Rendering novel view...")
    novel_camera = CameraParams(
        cameras[0].intrinsics,
        np.eye(4, dtype=np.float32),  # Camera at origin
        cameras[0].width,
        cameras[0].height,
    )

    rendered = backend.render_view(scene, novel_camera)
    logger.info(f"   ✓ Rendered {rendered.shape[0]}x{rendered.shape[1]} image")

    # Validate output
    logger.info("\n5. Validation...")
    assert rendered.shape == (images[0].shape[0], images[0].shape[1], 3)
    assert rendered.dtype == np.float32
    assert np.all(rendered >= 0) and np.all(rendered <= 1)
    assert not np.isnan(rendered).any()
    logger.info("   ✓ All validations passed!")

    # Print optimization metrics
    if "loss_history" in scene.splats.metadata:
        loss_history = scene.splats.metadata["loss_history"]
        logger.info(f"\n6. Optimization metrics:")
        logger.info(f"   - Initial loss: {loss_history[0]:.6f}")
        logger.info(f"   - Final loss: {loss_history[-1]:.6f}")
        logger.info(f"   - Reduction: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Phase 6A Demo Complete!")
    logger.info("=" * 60)

    logger.info("\nKey Features Demonstrated:")
    logger.info("  • Depth-guided Gaussian initialization")
    logger.info("  • Differentiable rasterizer (PyTorch)")
    logger.info("  • Adam optimization with gradient descent")
    logger.info("  • Novel view synthesis")
    logger.info("  • MPS/CUDA/CPU device support")

    logger.info("\nNext Steps (Phase 6B):")
    logger.info("  • Add full rotation support (anisotropic Gaussians)")
    logger.info("  • Implement densification/pruning")
    logger.info("  • Add spherical harmonics for view-dependent effects")
    logger.info("  • Optimize tile-based rendering for speed")


if __name__ == "__main__":
    main()
