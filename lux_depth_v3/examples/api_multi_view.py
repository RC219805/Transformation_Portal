"""Multi-view depth estimation examples.

Demonstrates multi-view depth estimation with pose estimation using the DA3 API.
"""

from pathlib import Path
import numpy as np

from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper


def example_multiview_auto_pose():
    """Multi-view depth with automatic pose estimation."""
    print("Example 1: Multi-View with Auto Pose Estimation")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    # Multiple views of the same scene
    images = [f"views/view_{i:03d}.jpg" for i in range(10)]

    prediction = wrapper.inference(
        image=images,
        export_dir="output/multiview/auto_pose",
        export_format="full_npz-glb",
    )

    print(f"Processed {len(images)} views")
    print(f"Depth shape: {prediction.depth.shape}")

    if prediction.extrinsics is not None:
        print(f"Estimated camera poses: {prediction.extrinsics.shape}")
        print("First camera pose:")
        print(prediction.extrinsics[0])

    print("✅ Results saved with estimated poses")


def example_multiview_known_poses():
    """Multi-view depth with known camera poses."""
    print("\nExample 2: Multi-View with Known Poses")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    # Load calibrated camera data
    images = [f"calibrated/img_{i:04d}.jpg" for i in range(15)]

    # Camera extrinsics (world-to-camera) - shape (N, 4, 4)
    extrinsics = np.load("calibrated/extrinsics.npy")

    # Camera intrinsics - shape (N, 3, 3)
    intrinsics = np.load("calibrated/intrinsics.npy")

    print(f"Using {len(images)} calibrated views")
    print(f"Extrinsics shape: {extrinsics.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")

    prediction = wrapper.inference(
        image=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,  # Align predictions to input scale
        export_dir="output/multiview/known_poses",
        export_format="full_npz-glb",
        show_cameras=True,  # Visualize cameras in GLB
    )

    # Compare input vs refined poses
    if prediction.extrinsics is not None:
        pose_diff = np.abs(prediction.extrinsics - extrinsics).mean()
        print(f"Average pose refinement: {pose_diff:.6f}")

    print("✅ Multi-view reconstruction complete")


def example_reference_view_strategies():
    """Compare different reference view selection strategies."""
    print("\nExample 3: Reference View Strategies")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    images = [f"sequence/frame_{i:04d}.jpg" for i in range(20)]

    strategies = ["first", "middle", "saddle_balanced", "saddle_sim_range"]

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")

        prediction = wrapper.inference(
            image=images,
            ref_view_strategy=strategy,
            export_dir=f"output/ref_strategies/{strategy}",
            export_format="mini_npz",
        )

        # Analyze depth consistency
        depth_std = prediction.depth.std(axis=0).mean()
        print(f"  Depth std dev: {depth_std:.4f}")

    print("\n✅ Strategy comparison complete")


def example_ray_pose_estimation():
    """Use ray-based pose estimation."""
    print("\nExample 4: Ray-Based Pose Estimation")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    images = [f"scene/img_{i:03d}.jpg" for i in range(12)]

    # Compare standard vs ray-based pose estimation
    print("Standard pose estimation...")
    pred_standard = wrapper.inference(
        image=images,
        use_ray_pose=False,
        export_dir="output/pose_methods/standard",
        export_format="full_npz",
    )

    print("Ray-based pose estimation...")
    pred_ray = wrapper.inference(
        image=images,
        use_ray_pose=True,
        export_dir="output/pose_methods/ray",
        export_format="full_npz",
    )

    # Compare results
    if pred_standard.extrinsics is not None and pred_ray.extrinsics is not None:
        pose_diff = np.abs(pred_standard.extrinsics - pred_ray.extrinsics).mean()
        print(f"Pose difference: {pose_diff:.6f}")

    print("✅ Pose estimation comparison complete")


def example_high_quality_reconstruction():
    """High-quality multi-view reconstruction with optimized settings."""
    print("\nExample 5: High-Quality Reconstruction")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    # Collect comprehensive view coverage
    images = sorted(Path("scene_capture").glob("*.jpg"))
    print(f"Found {len(images)} images")

    # High-quality settings
    prediction = wrapper.inference(
        image=images,
        process_res=672,  # Higher resolution
        process_res_method="upper_bound_resize",
        ref_view_strategy="saddle_balanced",
        align_to_input_ext_scale=True,
        export_dir="output/high_quality",
        export_format="full_npz-glb",
        # GLB export settings
        conf_thresh_percentile=60.0,  # Higher quality threshold
        num_max_points=2_000_000,  # More points
        show_cameras=True,
    )

    print(f"Depth maps generated: {prediction.depth.shape[0]}")
    print(f"Camera poses estimated: {prediction.extrinsics.shape[0]}")

    # Quality metrics
    depth_range = (prediction.depth.min(), prediction.depth.max())
    print(f"Depth range: {depth_range[0]:.3f} - {depth_range[1]:.3f}")

    if prediction.conf is not None:
        avg_confidence = prediction.conf.mean()
        print(f"Average confidence: {avg_confidence:.3f}")

    print("✅ High-quality reconstruction saved")


def example_sequential_capture():
    """Process sequential camera captures (e.g., video frames)."""
    print("\nExample 6: Sequential Capture Processing")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    # Extract frames from video or use sequential captures
    images = [f"sequence/frame_{i:06d}.png" for i in range(0, 100, 5)]  # Every 5th frame

    print(f"Processing {len(images)} sequential frames")

    prediction = wrapper.inference(
        image=images,
        ref_view_strategy="middle",  # Middle frame as reference
        process_res=504,
        export_dir="output/sequential",
        export_format="full_npz-depth_vis",  # Include depth video
    )

    print(f"Generated depth sequence: {prediction.depth.shape}")

    # Temporal consistency check
    if prediction.depth.shape[0] > 1:
        temporal_diff = np.abs(np.diff(prediction.depth, axis=0)).mean()
        print(f"Temporal consistency (avg diff): {temporal_diff:.4f}")

    print("✅ Sequential processing complete")


if __name__ == "__main__":
    print("DA3 API Multi-View Examples")
    print("=" * 50)
    print()

    try:
        example_multiview_auto_pose()
        example_multiview_known_poses()
        example_reference_view_strategies()
        example_ray_pose_estimation()
        example_high_quality_reconstruction()
        example_sequential_capture()

        print("\n" + "=" * 50)
        print("All multi-view examples completed!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
