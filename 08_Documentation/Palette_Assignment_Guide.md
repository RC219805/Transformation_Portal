# Material Palette Assignment Guide

## Overview

The board material aerial enhancer now supports **palette assignment files** - JSON configurations that define explicit mappings between cluster IDs and material names. This feature enables deterministic material assignments, manual overrides of automatic detection, and configuration sharing across projects.

## Palette File Format

A palette file is a simple JSON document with the following structure:

```json
{
  "version": "1.0",
  "assignments": {
    "0": "plaster",
    "1": "stone",
    "2": "roof",
    "3": "equitone",
    "7": "bronze"
  }
}
```

### Schema

- **`version`** (string): Palette format version. Currently "1.0".
- **`assignments`** (object): Maps cluster IDs (as strings) to material names.
  - Keys: Cluster IDs from k-means clustering (0, 1, 2, ..., k-1)
  - Values: Material names from the approved MBAR palette

### Available Materials

The following material names are recognized:

| Material Name | Description |
|---------------|-------------|
| `plaster` | Marmorino Palladino Plaster - Westwood Beige |
| `stone` | Eco Outdoor Bokara Stone - Coastal |
| `cladding` | Sculptform Click-On Systems - Warm Timber |
| `screens` | Grey Gum Screens - Natural |
| `equitone` | Equitone LT85 Panels - Anthracite |
| `roof` | Bison Weathered Ipe Pavers |
| `bronze` | Dark Bronze Anodized Metal |
| `shade` | Louvretec Powder White Shading |

## Usage

### 1. Generate a Baseline Palette

First, run the enhancer with automatic material detection and save the computed assignments:

```bash
python board_material_aerial_enhancer.py \
  input.jpg output.jpg \
  --k 8 \
  --save-palette baseline_palette.json
```

This creates `baseline_palette.json` with the automatically computed cluster-to-material mappings.

### 2. Customize the Palette

Edit the generated JSON file to adjust material assignments:

```json
{
  "version": "1.0",
  "assignments": {
    "0": "plaster",      # Keep automatic assignment
    "1": "equitone",     # Override: was "stone", now "equitone"
    "2": "roof",         # Keep automatic assignment
    "3": "bronze"        # Override: force this cluster to bronze
  }
}
```

### 3. Apply the Custom Palette

Use the customized palette for consistent, deterministic assignments:

```bash
python board_material_aerial_enhancer.py \
  input2.jpg output2.jpg \
  --palette baseline_palette.json
```

### 4. Palette-Based Workflow

For production pipelines, maintain palette files in version control:

```bash
# Initial generation with palette save
python board_material_aerial_enhancer.py \
  aerial_v1.jpg enhanced_v1.jpg \
  --save-palette project_palette.json

# Review and manually adjust project_palette.json

# Apply to all subsequent images
for img in aerial_*.jpg; do
  python board_material_aerial_enhancer.py \
    "$img" "enhanced_$img" \
    --palette project_palette.json
done
```

## Command-Line Options

### `board_material_aerial_enhancer.py`

```bash
--palette PATH          Load cluster-to-material assignments from JSON palette file.
                        When provided, uses these mappings instead of heuristic scoring.

--save-palette PATH     Save computed/loaded assignments to a JSON palette file.
                        Useful for creating reusable configurations.
```

### Example: Complete Workflow

```bash
# Step 1: Run with auto-detection and save palette
python board_material_aerial_enhancer.py \
  input.jpg test_output.jpg \
  --k 8 \
  --seed 42 \
  --save-palette auto_palette.json

# Step 2: Review auto_palette.json, adjust as needed

# Step 3: Apply refined palette to production images
python board_material_aerial_enhancer.py \
  final_input.jpg final_output.jpg \
  --palette auto_palette.json \
  --target-width 4096
```

## Visualization Script

The `visualize_material_assignments.py` script also supports palette files:

```bash
# Visualize with automatic assignments and save palette
python visualize_material_assignments.py \
  --save-palette viz_palette.json

# Visualize with custom palette
python visualize_material_assignments.py \
  --palette custom_palette.json
```

## Benefits

### Deterministic Assignments
Run the enhancer multiple times with the same palette to get identical material assignments, regardless of slight variations in automatic detection.

### Manual Override
Override automatic material detection when the heuristics misidentify a cluster. For example:
- Cluster 5 is detected as "stone" but should be "plaster"
- Edit the palette: `"5": "plaster"`
- Rerun with `--palette` to apply the correction

### Configuration Sharing
Share palette configurations across team members or between related projects:
```bash
# Team member A creates baseline
python board_material_aerial_enhancer.py img.jpg out.jpg --save-palette team_palette.json

# Team member B uses the same assignments
python board_material_aerial_enhancer.py img2.jpg out2.jpg --palette team_palette.json
```

### Version Control
Track palette files in Git alongside project assets:
```bash
git add palettes/750_picacho_palette.json
git commit -m "Add calibrated palette for 750 Picacho aerial"
```

## Error Handling

### Unknown Material Names

If a palette references a material name that doesn't exist, you'll see:

```
ValueError: Unknown material 'granite' in palette. 
Available materials: ['plaster', 'stone', 'cladding', ...]
```

**Solution**: Check the material name spelling and refer to the [Available Materials](#available-materials) table.

### Missing Palette File

If you specify `--palette path/to/missing.json`:

```
FileNotFoundError: Palette file not found: path/to/missing.json
```

**Solution**: Verify the file path and ensure the palette file exists.

### Cluster ID Mismatch

If your palette uses cluster IDs that don't exist in the current clustering:
- Palette references cluster `"10"` but `k=8` (clusters 0-7)
- The extra assignment is silently ignored
- Only valid cluster IDs are applied

**Best Practice**: Match `--k` value when using a saved palette.

## Advanced Usage

### Partial Palette

You don't need to assign all clusters. Unassigned clusters will have no material texture applied:

```json
{
  "version": "1.0",
  "assignments": {
    "0": "plaster",
    "2": "stone"
    // Clusters 1, 3, 4, 5, 6, 7 remain untextured
  }
}
```

### Palette Merging

To combine palettes or apply multiple configurations:

1. Load base palette
2. Apply overrides programmatically
3. Save merged result

```python
from board_material_aerial_enhancer import load_palette_assignments, save_palette_assignments
import json

# Load base
with open('base_palette.json', 'r') as f:
    base = json.load(f)

# Load overrides
with open('overrides.json', 'r') as f:
    overrides = json.load(f)

# Merge (overrides take precedence)
merged = {**base['assignments'], **overrides['assignments']}
save_palette_assignments(merged, 'merged_palette.json')
```

### Palette Inspection

View palette contents:

```bash
cat palette.json | jq '.assignments'
```

Count assigned clusters:

```bash
cat palette.json | jq '.assignments | length'
```

List materials used:

```bash
cat palette.json | jq '.assignments | to_entries[] | .value' | sort -u
```

## Integration with Existing Workflows

### MBAR Board Submissions

For MBAR board package assemblies:
1. Generate initial palette from architect-provided aerial
2. Review and calibrate palette with material specifications
3. Apply palette consistently across all aerial views
4. Include palette in board submission as configuration artifact

### Batch Processing

Process multiple aerials with consistent material assignments:

```bash
#!/bin/bash
PALETTE="mbar_approved_palette.json"
for aerial in aerials/*.jpg; do
  base=$(basename "$aerial" .jpg)
  python board_material_aerial_enhancer.py \
    "$aerial" \
    "enhanced/${base}_enhanced.jpg" \
    --palette "$PALETTE" \
    --target-width 4096
done
```

### Quality Control

Compare automatic vs. manual assignments:

```bash
# Generate automatic palette
python board_material_aerial_enhancer.py img.jpg out_auto.jpg --save-palette auto.json

# Apply manual palette
python board_material_aerial_enhancer.py img.jpg out_manual.jpg --palette manual.json

# Visual comparison
diff auto.json manual.json
```

## Troubleshooting

**Problem**: Palette assignments don't match expected results

**Solutions**:
1. Verify cluster IDs match by checking `--k` parameter
2. Ensure `--seed` is consistent for reproducible clustering
3. Check that material names exactly match available materials (case-sensitive)

**Problem**: Some clusters don't receive textures

**Solutions**:
1. Verify those cluster IDs are present in the palette JSON
2. Check that texture files exist for the assigned materials
3. Review `--blend` values if textures seem too subtle

**Problem**: Results vary between runs despite using palette

**Solutions**:
1. Use consistent `--seed` value for reproducible clustering
2. Ensure image dimensions are consistent (clustering is resolution-dependent)
3. Check that `--k` matches the palette's cluster count

## Technical Notes

### Implementation Details

- Palette files are validated on load
- Unknown material names trigger `ValueError` with available options
- Cluster IDs are converted from strings (JSON keys) to integers
- Empty or missing `assignments` object defaults to no assignments
- Version field is currently informational (no migration logic)

### Performance

- Loading palette adds negligible overhead (~1ms)
- Saving palette is instant (~1ms)
- Using a palette bypasses heuristic scoring (slight speed improvement)

### Backward Compatibility

All palette features are optional. Existing code continues to work:

```python
# Still works - uses automatic detection
enhance_aerial(input_path, output_path)

# New feature - uses palette
enhance_aerial(input_path, output_path, palette_path="my_palette.json")
```

## See Also

- [board_material_aerial_enhancer.py](../board_material_aerial_enhancer.py) - Main implementation
- [MBAR Enhancement Report](../processed_images/MBAR_Enhancement_Report.md) - Example output
- [Version History](./Version_History/changelog.md) - Feature changelog
