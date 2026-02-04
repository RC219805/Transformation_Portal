#!/usr/bin/env python3
"""Simple demonstration of DA3InferenceEngine.

Shows basic usage matching the task requirements.
"""

import numpy as np

from transformation_portal.lux_depth_v3 import DA3Config, DA3InferenceEngine


def main():
    """Demonstrate DA3InferenceEngine usage."""
    print("=" * 60)
    print("DA3InferenceEngine Integration Test")
    print("=" * 60)

    # Create configuration
    config = DA3Config()
    print(f"✅ Created DA3Config")
    print(f"   - Model variant: {config.model_variant.name}")
    print(f"   - Device: {config.device.device}")

    # Initialize engine
    engine = DA3InferenceEngine(config)
    print(f"✅ Initialized DA3InferenceEngine")
    print(f"   - Backend: {engine.backend.name}")
    print(f"   - Device: {engine.device}")

    # Create test image
    image = np.random.rand(512, 512, 3).astype(np.float32)
    print(f"✅ Created test image: {image.shape}")

    # Run inference
    print(f"🔄 Running depth inference...")
    result = engine.predict(image)

    # Validate results
    print(f"✅ Inference completed!")
    print(f"   - Depth map shape: {result.depth_map.shape}")
    print(f"   - Depth range: [{result.depth_map.min():.3f}, {result.depth_map.max():.3f}]")
    print(f"   - Inference time: {result.metadata['inference_time_ms']:.1f}ms")
    print(f"   - Backend: {result.metadata['backend']}")
    print(f"   - Device: {result.metadata['device']}")

    if result.metadata.get("using_fallback"):
        print(f"   - Using fallback model: {result.metadata['fallback_model']}")

    # Verify assertions from task
    assert result.depth_map.shape[:2] == image.shape[:2], "Shape mismatch"
    assert result.depth is result.depth_map, "Depth alias not working"
    assert result.metadata["inference_time_ms"] > 0, "Inference time not recorded"

    print(f"\n✅ All assertions passed!")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
