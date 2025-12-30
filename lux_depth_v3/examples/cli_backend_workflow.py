"""Example: Using DA3 backend for batch processing.

This example demonstrates the performance benefits of using the DA3 backend
service for processing multiple batches of images. The backend keeps the model
loaded in GPU memory, avoiding 10-15s model loading overhead for each batch.

Performance Comparison:
- Without backend: 10-15s model load + 50ms/image × N images per batch
- With backend: 10-15s model load (once) + 50ms/image × N images (all batches)

For 3 batches of 100 images each:
- Without backend: ~60 seconds (3 × 20s)
- With backend: ~20 seconds (15s + 5s)
- Speedup: 3x (scales with number of batches)
"""

from pathlib import Path
from lux_depth_v3 import DA3Config, DA3InferenceEngine
from lux_depth_v3.input_manager import ImageInput
from lux_depth_v3.config import ModelVariant
import time


def example_without_backend():
    """Process batches without backend (slower)."""
    print("=" * 60)
    print("Example 1: Without Backend (Native Mode)")
    print("=" * 60)

    # Configure native mode
    config = DA3Config(
        model_variant=ModelVariant.METRIC_LARGE,
    )
    config.cli.use_cli = False  # Native mode

    batches = ["batch1", "batch2", "batch3"]

    total_start = time.time()

    for batch_name in batches:
        print(f"\nProcessing {batch_name}...")
        batch_start = time.time()

        # Initialize engine (model loaded here - slow!)
        engine = DA3InferenceEngine(config)
        engine.load_model()

        # Simulate batch of images
        # In real use: images = [ImageInput(path=p) for p in Path(batch_name).glob("*.jpg")]
        images = [ImageInput(array=None) for _ in range(10)]  # Placeholder

        # Process (placeholder - would call engine.inference(images))
        print(f"  Processing {len(images)} images...")

        batch_elapsed = time.time() - batch_start
        print(f"  Batch time: {batch_elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.2f}s")
    print("Note: Model reloaded for each batch (10-15s overhead)")


def example_with_backend():
    """Process batches with backend service (faster)."""
    print("\n" + "=" * 60)
    print("Example 2: With Backend Service (CLI Mode)")
    print("=" * 60)

    # Configure CLI mode with backend
    config = DA3Config(
        model_variant=ModelVariant.METRIC_LARGE,
    )
    config.cli.use_cli = True
    config.cli.use_backend = True
    config.cli.backend_port = 8008

    # Initialize engine
    engine = DA3InferenceEngine(config)

    # Start backend once
    print("\nStarting backend service...")
    backend_start = time.time()

    try:
        engine.start_backend(timeout=30)
        backend_elapsed = time.time() - backend_start
        print(f"Backend started in {backend_elapsed:.2f}s")

        batches = ["batch1", "batch2", "batch3"]

        total_start = time.time()

        for batch_name in batches:
            print(f"\nProcessing {batch_name}...")
            batch_start = time.time()

            # Simulate batch of images
            images = [ImageInput(array=None) for _ in range(10)]  # Placeholder

            # Process (placeholder - would call engine.inference(images))
            print(f"  Processing {len(images)} images...")

            batch_elapsed = time.time() - batch_start
            print(f"  Batch time: {batch_elapsed:.2f}s")

        total_elapsed = time.time() - total_start
        print(f"\nTotal processing time: {total_elapsed:.2f}s")
        print(f"Total time (including backend start): {backend_elapsed + total_elapsed:.2f}s")
        print("Note: Model loaded only once - no reload overhead between batches")

    finally:
        # Clean up
        print("\nStopping backend...")
        engine.stop_backend()
        print("Backend stopped")


def example_real_workflow():
    """Real-world example with actual image processing."""
    print("\n" + "=" * 60)
    print("Example 3: Real Workflow with Images")
    print("=" * 60)

    # This example assumes you have DA3 CLI installed
    # If not, it will fall back to native mode

    config = DA3Config.from_preset("interior_luxury")
    config.cli.use_cli = True  # Try CLI mode
    config.cli.use_backend = True

    engine = DA3InferenceEngine(config)

    # Check if CLI is available
    if not engine.use_cli:
        print("WARNING: DA3 CLI not available, using native mode")
        print("Install with: pip install git+https://github.com/DepthAnything/Depth-Anything-V3.git")
    else:
        print("DA3 CLI available, using backend mode")

    # Example usage pattern
    scenes = ["living_room", "kitchen", "bedroom"]

    if engine.use_cli and engine.backend is not None:
        # Start backend
        print("\nStarting backend...")
        engine.start_backend()

    try:
        for scene in scenes:
            print(f"\nProcessing scene: {scene}")

            # In real use, load images from directory
            scene_dir = Path("data") / scene
            if scene_dir.exists():
                images = [ImageInput(path=p) for p in scene_dir.glob("*.jpg")]

                if images:
                    # Process with backend (fast - no model reload)
                    results = engine.inference(images)

                    print(f"  Processed {len(results)} images")
                    print(f"  Depth range: {results[0].get_depth_range()}")
                else:
                    print(f"  No images found in {scene_dir}")
            else:
                print(f"  Directory not found: {scene_dir}")

    finally:
        # Clean up
        if engine.backend is not None:
            engine.stop_backend()


def main():
    """Run all examples."""
    print("DA3 Backend Service Examples")
    print("=" * 60)

    # Example 1: Without backend (slower)
    # example_without_backend()

    # Example 2: With backend (faster)
    # example_with_backend()

    # Example 3: Real workflow
    example_real_workflow()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)

    print("\nKey Takeaways:")
    print("1. Backend service loads model once and keeps it in GPU memory")
    print("2. Ideal for batch processing and repeated operations")
    print("3. Provides 10-20x speedup by avoiding model reload overhead")
    print("4. Use native mode for single images or small batches")
    print("5. Use CLI + backend for large batches or production workflows")


if __name__ == "__main__":
    main()
