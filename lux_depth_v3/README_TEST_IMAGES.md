# Test Images

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
lux-depth-v3 enhance \
  --input-dir test_images/ \
  --output-dir test_output/ \
  --model metric-large \
  --v2-preset production_ultra \
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
