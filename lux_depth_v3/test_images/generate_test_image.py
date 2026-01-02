#!/usr/bin/env python3
"""
Generate a simple test image for lux_depth_v3 validation.
Can be run without heavy dependencies (only needs Python's built-in modules).
"""


def create_test_image_ppm():
    """Create a simple PPM test image (no PIL required)."""
    width, height = 1920, 1080

    # PPM header (P6 = binary RGB)
    with open("test_gradient.ppm", "wb") as f:
        # Header
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))

        # Pixel data (RGB gradient)
        for y in range(height):
            for x in range(width):
                r = int((y / height) * 255)  # Red gradient top to bottom
                g = 128  # Constant green
                b = int((1 - y / height) * 255)  # Blue gradient bottom to top
                f.write(bytes([r, g, b]))

    print(f"✓ Created test_gradient.ppm ({width}x{height})")
    print("  Note: Convert to PNG/JPEG using ImageMagick or Pillow if needed")


def create_instructions():
    """Create README for test images."""
    with open("README_TEST_IMAGES.md", "w") as f:
        f.write("""# Test Images

This directory contains test images for lux_depth_v3 integration testing.

## Generated Test Image

- `test_gradient.ppm` - Simple RGB gradient (1920x1080)
  - Generated without dependencies
  - PPM format (can be converted to PNG/JPEG)

## Adding Real Test Images

Place your own test images here for validation:

```bash
# Copy images
cp ~/path/to/image.jpg test_images/
cp ~/path/to/*.png test_images/

# Or use sample images from parent repo
cp ../input_images/*.jpg test_images/
```

## Running Tests

Once dependencies are installed:

```bash
# From lux_depth_v3/ directory
lux-depth-v3 enhance \\
  --input-dir test_images/ \\
  --output-dir test_output/ \\
  --model metric-large \\
  --v2-preset production_ultra \\
  --verbose
```

## Expected Output Structure

```
test_output/
├── depth/               # V3 depth maps
│   ├── test_gradient_depth.png
│   └── ...
├── v2/                  # V2 enhanced images
│   ├── test_gradient_enhanced.png
│   └── ...
└── manifests/           # Processing metadata
    ├── test_gradient.json
    └── ...
```
""")

    print("✓ Created README_TEST_IMAGES.md")


if __name__ == "__main__":
    print("=== Test Image Generation ===\n")
    create_test_image_ppm()
    create_instructions()
    print("\n✓ Test images setup complete")
    print("\nNext steps:")
    print("1. (Optional) Convert PPM to PNG: convert test_gradient.ppm test_gradient.png")
    print("2. (Optional) Add your own test images to this directory")
    print("3. Install dependencies: ../INSTALL_DEPENDENCIES.sh")
    print("4. Run integration tests: see README_TEST_IMAGES.md")
