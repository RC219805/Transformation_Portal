"""Basic DA3 API usage examples.

Demonstrates simple monocular depth estimation with the official DA3 Python API.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper


def example_single_image():
    """Basic single-image depth estimation."""
    print("Example 1: Single Image Depth Estimation")
    print("=" * 50)

    # Initialize wrapper
    wrapper = DepthAnything3Wrapper(
        model_name="da3-large",
        device="cuda",  # Use "cpu" if no GPU available
    )

    # Run inference
    prediction = wrapper.inference(image=["path/to/image.jpg"], export_dir="output/basic", export_format="mini_npz")

    # Access results
    depth = prediction.depth[0]  # (H, W)
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: {depth.min():.3f} - {depth.max():.3f}")

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Depth Map")
    plt.imshow(depth, cmap="turbo")
    plt.colorbar(label="Depth")

    if prediction.conf is not None:
        plt.subplot(1, 2, 2)
        plt.title("Confidence")
        plt.imshow(prediction.conf[0], cmap="gray")
        plt.colorbar(label="Confidence")

    plt.tight_layout()
    plt.savefig("output/basic/visualization.png")
    print("✅ Saved visualization to output/basic/visualization.png")


def example_batch_processing():
    """Process multiple images."""
    print("\nExample 2: Batch Processing")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    # Process multiple images
    images = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
    ]

    prediction = wrapper.inference(image=images, export_dir="output/batch", export_format="mini_npz-depth_vis")

    print(f"Processed {prediction.depth.shape[0]} images")
    for i in range(prediction.depth.shape[0]):
        depth_range = (prediction.depth[i].min(), prediction.depth[i].max())
        print(f"  Image {i}: depth range {depth_range[0]:.3f} - {depth_range[1]:.3f}")

    print("✅ Results saved to output/batch/")


def example_different_models():
    """Compare different model variants."""
    print("\nExample 3: Model Comparison")
    print("=" * 50)

    models = ["da3-small", "da3-base", "da3-large"]
    image_path = "test_image.jpg"

    for model_name in models:
        print(f"\nTesting {model_name}...")

        wrapper = DepthAnything3Wrapper(model_name=model_name)

        prediction = wrapper.inference(
            image=[image_path],
            export_dir=f"output/models/{model_name}",
            export_format="mini_npz",
        )

        depth = prediction.depth[0]
        print(f"  Depth range: {depth.min():.3f} - {depth.max():.3f}")
        print(f"  Depth std: {depth.std():.3f}")


def example_metric_depth():
    """Metric depth estimation with da3metric-large."""
    print("\nExample 4: Metric Depth Estimation")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3metric-large", device="cuda")

    prediction = wrapper.inference(
        image=["outdoor_scene.jpg"],
        export_dir="output/metric",
        export_format="full_npz",
    )

    # Metric depth is in absolute units (meters)
    depth_meters = prediction.depth[0]

    print(f"Depth range: {depth_meters.min():.2f}m - {depth_meters.max():.2f}m")
    print(f"Mean depth: {depth_meters.mean():.2f}m")

    # Sky segmentation may be available
    if prediction.aux and "sky_mask" in prediction.aux:
        sky_mask = prediction.aux["sky_mask"]
        sky_percentage = sky_mask.sum() / sky_mask.size * 100
        print(f"Sky coverage: {sky_percentage:.1f}%")

    print("✅ Metric depth saved to output/metric/")


def example_export_formats():
    """Demonstrate different export formats."""
    print("\nExample 5: Export Formats")
    print("=" * 50)

    wrapper = DepthAnything3Wrapper(model_name="da3-large")

    # 1. Minimal NPZ (depth + confidence)
    print("Exporting mini_npz...")
    wrapper.inference(image=["image.jpg"], export_dir="output/exports/mini", export_format="mini_npz")

    # 2. Full NPZ (all data)
    print("Exporting full_npz...")
    wrapper.inference(image=["image.jpg"], export_dir="output/exports/full", export_format="full_npz")

    # 3. 3D Mesh (GLB)
    print("Exporting GLB...")
    wrapper.inference(
        image=["image.jpg"],
        export_dir="output/exports/glb",
        export_format="glb",
        conf_thresh_percentile=50.0,
        num_max_points=1_000_000,
    )

    # 4. Combined exports
    print("Exporting combined formats...")
    wrapper.inference(
        image=["image.jpg"],
        export_dir="output/exports/combined",
        export_format="mini_npz-glb-depth_vis",
    )

    print("✅ All exports complete")


if __name__ == "__main__":
    # Run all examples
    # Note: Update image paths before running

    print("DA3 API Basic Usage Examples")
    print("=" * 50)
    print()

    try:
        example_single_image()
        example_batch_processing()
        example_different_models()
        example_metric_depth()
        example_export_formats()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
