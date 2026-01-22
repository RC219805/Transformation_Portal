"""Gaussian Splatting workflow examples.

Demonstrates 3D Gaussian Splatting (3DGS) reconstruction with DA3 API.
"""

from pathlib import Path
import numpy as np

from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper


def create_circular_trajectory(center=(0, 0, 0), radius=5.0, height=2.0, num_frames=120):
    """Create circular camera trajectory for rendering.

    Args:
        center: Center point to orbit around
        radius: Orbit radius
        height: Camera height above center
        num_frames: Number of frames in trajectory

    Returns:
        Camera extrinsics (num_frames, 4, 4)
    """
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    extrinsics = []
    for angle in angles:
        # Camera position
        cam_x = center[0] + radius * np.cos(angle)
        cam_y = center[1] + height
        cam_z = center[2] + radius * np.sin(angle)

        # Look at center
        forward = np.array(center) - np.array([cam_x, cam_y, cam_z])
        forward = forward / np.linalg.norm(forward)

        # Compute camera axes
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Build extrinsic matrix (world-to-camera)
        R = np.column_stack([right, up, -forward])
        t = -R.T @ np.array([cam_x, cam_y, cam_z])

        ext = np.eye(4)
        ext[:3, :3] = R
        ext[:3, 3] = t
        extrinsics.append(ext)

    return np.array(extrinsics)


def example_basic_gs():
    """Basic Gaussian Splatting reconstruction."""
    print("Example 1: Basic GS Reconstruction")
    print("=" * 50)

    # Initialize with GS-capable model
    wrapper = DepthAnything3Wrapper(
        model_name="da3-giant",  # Required for GS
        device="cuda",
    )

    # Prepare training images
    scene_dir = Path("scene_capture")
    images = sorted(scene_dir.glob("*.jpg"))[:30]  # Use 30 views

    print(f"Training on {len(images)} views...")

    # Run GS reconstruction
    prediction = wrapper.inference(
        image=images,
        infer_gs=True,  # Enable Gaussian Splatting
        export_dir="output/gs/basic",
        export_format="gs_ply",  # Export Gaussian splats
    )

    print(f"Depth maps: {prediction.depth.shape}")

    # Access GS data
    if prediction.aux and "gaussian_splatting" in prediction.aux:
        gs_data = prediction.aux["gaussian_splatting"]
        print(f"Generated Gaussian splats: {len(gs_data.get('splats', []))}")

    print("✅ GS reconstruction saved to output/gs/basic/")


def example_gs_with_video():
    """GS reconstruction with novel view synthesis video."""
    print("\nExample 2: GS with Novel View Video")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-giant")

    # Training images
    images = [f"room_scan/img_{i:04d}.jpg" for i in range(40)]

    # Create rendering trajectory
    print("Creating camera trajectory...")
    render_exts = create_circular_trajectory(
        center=(0, 0, 0),
        radius=8.0,
        height=2.0,
        num_frames=240,  # 8 seconds at 30fps
    )

    print(f"Rendering {len(render_exts)} frames...")

    # Run inference with GS video export
    prediction = wrapper.inference(
        image=images,
        infer_gs=True,
        export_format="gs_ply-gs_video",
        render_exts=render_exts,
        render_hw=(1080, 1920),  # Full HD
        export_dir="output/gs/with_video",
    )

    print("✅ GS video rendered to output/gs/with_video/")


def example_high_quality_gs():
    """High-quality GS reconstruction with optimized settings."""
    print("\nExample 3: High-Quality GS")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-giant")

    # Use more images for better quality
    images = sorted(Path("high_res_capture").glob("*.png"))
    print(f"Using {len(images)} high-resolution views")

    # High-quality settings
    prediction = wrapper.inference(
        image=images,
        infer_gs=True,
        process_res=672,  # Higher processing resolution
        ref_view_strategy="saddle_balanced",
        export_dir="output/gs/high_quality",
        export_format="gs_ply",
    )

    print(f"Generated high-quality GS: {prediction.depth.shape}")
    print("✅ High-quality GS saved")


def example_gs_nested_model():
    """Use nested-giant-large model for maximum capability."""
    print("\nExample 4: GS with Nested-Giant-Large")
    print("=" * 50)

    # Nested model includes metric depth + GS
    wrapper = DepthAnything3Wrapper(model_name="da3nested-giant-large", device="cuda")

    images = [f"scene/view_{i:03d}.jpg" for i in range(25)]

    prediction = wrapper.inference(
        image=images,
        infer_gs=True,
        export_dir="output/gs/nested",
        export_format="full_npz-gs_ply-glb",  # All formats
    )

    print("Generated outputs:")
    print("  - Metric depth maps (absolute scale)")
    print("  - Gaussian splats (gs_ply)")
    print("  - 3D mesh (glb)")

    print("✅ Nested model outputs complete")


def example_gs_custom_rendering():
    """Custom rendering with multiple camera paths."""
    print("\nExample 5: GS Custom Rendering")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-giant")

    images = sorted(Path("object_scan").glob("*.jpg"))

    # First pass: Generate GS representation
    print("Generating GS representation...")
    gs_prediction = wrapper.inference(
        image=images,
        infer_gs=True,
        export_format="gs_ply",
        export_dir="output/gs/custom/model",
    )

    # Second pass: Render custom views
    print("Rendering custom views...")

    # Create multiple rendering trajectories

    # 1. Orbital view
    orbital_exts = create_circular_trajectory(radius=10.0, height=3.0, num_frames=120)

    wrapper.inference(
        image=images,
        infer_gs=True,
        export_format="gs_video",
        render_exts=orbital_exts,
        render_hw=(720, 1280),
        export_dir="output/gs/custom/orbital",
    )

    # 2. Top-down view
    top_down_exts = create_circular_trajectory(
        radius=5.0,
        height=15.0,  # High above
        num_frames=90,
    )

    wrapper.inference(
        image=images,
        infer_gs=True,
        export_format="gs_video",
        render_exts=top_down_exts,
        render_hw=(720, 1280),
        export_dir="output/gs/custom/top_down",
    )

    print("✅ Custom renderings complete")


def example_gs_quality_comparison():
    """Compare GS quality with different settings."""
    print("\nExample 6: GS Quality Comparison")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-giant")

    images = [f"comparison/img_{i:03d}.jpg" for i in range(20)]

    # Test different process_res values
    resolutions = [378, 504, 672]

    for res in resolutions:
        print(f"\nTesting process_res={res}...")

        prediction = wrapper.inference(
            image=images,
            infer_gs=True,
            process_res=res,
            export_dir=f"output/gs/quality/res_{res}",
            export_format="gs_ply",
        )

        # Quality metrics
        depth_std = prediction.depth.std()
        print(f"  Depth std: {depth_std:.4f}")

        if prediction.conf is not None:
            avg_conf = prediction.conf.mean()
            print(f"  Avg confidence: {avg_conf:.3f}")

    print("\n✅ Quality comparison complete")


def example_gs_with_poses():
    """GS reconstruction with known camera poses."""
    print("\nExample 7: GS with Known Poses")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-giant")

    # Load calibrated data
    images = sorted(Path("calibrated_scan").glob("*.jpg"))
    extrinsics = np.load("calibrated_scan/poses.npy")
    intrinsics = np.load("calibrated_scan/intrinsics.npy")

    print(f"Using {len(images)} calibrated views")

    prediction = wrapper.inference(
        image=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        infer_gs=True,
        align_to_input_ext_scale=True,
        export_dir="output/gs/with_poses",
        export_format="gs_ply-glb",
    )

    print("✅ GS with known poses complete")


if __name__ == "__main__":
    print("DA3 API Gaussian Splatting Examples")
    print("=" * 50)
    print()

    # Check GS model availability
    from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper

    wrapper = DepthAnything3Wrapper(model_name="da3-giant")
    if not wrapper.available:
        print("❌ DA3 API not available. Install with: pip install depth-anything-3")
        exit(1)

    if "da3-giant" not in wrapper.AVAILABLE_MODELS:
        print("⚠️  GS requires da3-giant or da3nested-giant-large model")

    try:
        example_basic_gs()
        example_gs_with_video()
        example_high_quality_gs()
        example_gs_nested_model()
        example_gs_custom_rendering()
        example_gs_quality_comparison()
        example_gs_with_poses()

        print("\n" + "=" * 50)
        print("All GS examples completed!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
