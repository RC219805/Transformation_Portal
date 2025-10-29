"""Generate a visualization showing MBAR material assignments for the aerial."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

from board_material_aerial_enhancer import (
    _downsample_image,
    _kmeans,
    _assign_full_image,
    _cluster_stats,
    build_material_rules,
    assign_materials,
    load_palette_assignments,
    save_palette_assignments,
    DEFAULT_TEXTURES,
)

# Load the input image
input_path = Path("/workspaces/800-Picacho-Lane-LUTs/input_images/RC-office750Picacho_Aerial.tiff")
image = Image.open(input_path).convert("RGB")
base_array = np.asarray(image, dtype=np.float32) / 255.0

# Perform clustering (same as enhancement)
# pylint: disable=duplicate-code  # Similar clustering logic used in enhance_pool_aerial.py
analysis_image = _downsample_image(image, 1280)
analysis_array = np.asarray(analysis_image, dtype=np.float32) / 255.0
pixels = analysis_array.reshape(-1, 3)

rng = np.random.default_rng(22)
sample_size = min(len(pixels), 200_000)
if sample_size < len(pixels):
    indices = rng.choice(len(pixels), size=sample_size, replace=False)
    sample = pixels[indices]
else:
    sample = pixels

centroids = _kmeans(sample, 8, rng)
labels_small = _assign_full_image(analysis_array, centroids)
# pylint: enable=duplicate-code
labels_small_img = Image.fromarray(labels_small.astype("uint8"))
labels_small_img = labels_small_img.convert("L")
labels_full = labels_small_img.resize(image.size, Image.Resampling.NEAREST)
labels = np.asarray(labels_full, dtype=np.uint8)

# Get material assignments
stats = _cluster_stats(base_array, labels)
rules = build_material_rules(DEFAULT_TEXTURES)

# Support optional palette file via command line argument
# Usage: python visualize_material_assignments.py [--palette path/to/palette.json] [--save-palette path/to/save.json]
palette_path = None
save_palette_path = None
if "--palette" in sys.argv:
    idx = sys.argv.index("--palette")
    if idx + 1 < len(sys.argv):
        palette_path = Path(sys.argv[idx + 1])
if "--save-palette" in sys.argv:
    idx = sys.argv.index("--save-palette")
    if idx + 1 < len(sys.argv):
        save_palette_path = Path(sys.argv[idx + 1])

# Load or compute assignments
if palette_path and palette_path.exists():
    print(f"Loading palette from: {palette_path}")
    assignments = load_palette_assignments(palette_path, rules)
else:
    assignments = assign_materials(stats, rules)

# Optionally save assignments
if save_palette_path:
    print(f"Saving palette to: {save_palette_path}")
    save_palette_assignments(assignments, save_palette_path)

# Create color-coded visualization
colors = [
    (255, 100, 100),  # Red
    (100, 255, 100),  # Green
    (100, 100, 255),  # Blue
    (255, 255, 100),  # Yellow
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Cyan
    (255, 200, 100),  # Orange
    (200, 100, 255),  # Purple
]

# Create visualization
viz_array = np.zeros((*labels.shape, 3), dtype=np.uint8)
for label in range(8):
    mask = labels == label
    viz_array[mask] = colors[label]

viz_img = Image.fromarray(viz_array)

# Add legend
legend_height = 400
legend_img = Image.new("RGB", (viz_img.width, viz_img.height + legend_height), (255, 255, 255))
legend_img.paste(viz_img, (0, 0))

draw = ImageDraw.Draw(legend_img)

# Draw legend
y_offset = viz_img.height + 20
x_offset = 40

draw.text((x_offset, y_offset), "MBAR MATERIAL ASSIGNMENTS:", fill=(0, 0, 0))
y_offset += 40

for label, rule in assignments.items():
    # Draw color box
    box_size = 30
    draw.rectangle(
        [x_offset, y_offset, x_offset + box_size, y_offset + box_size],
        fill=colors[label],
        outline=(0, 0, 0),
    )

    # Draw material name
    text = f"{rule.name.upper()} (Cluster {label})"
    draw.text((x_offset + box_size + 15, y_offset + 5), text, fill=(0, 0, 0))
    y_offset += 45

# Add unassigned clusters
unassigned = [i for i in range(8) if i not in assignments]
if unassigned:
    draw.text((x_offset, y_offset), "UNASSIGNED CLUSTERS:", fill=(128, 128, 128))
    y_offset += 35
    for label in unassigned:
        draw.rectangle(
            [x_offset, y_offset, x_offset + box_size, y_offset + box_size],
            fill=colors[label],
            outline=(0, 0, 0),
        )
        text = f"Cluster {label} (no material match)"
        draw.text((x_offset + box_size + 15, y_offset + 5), text, fill=(128, 128, 128))
        y_offset += 45

# Save visualization
output_path = Path("/workspaces/800-Picacho-Lane-LUTs/processed_images/750_Picacho_Material_Assignment_Map.jpg")
legend_img.save(output_path, quality=95)

print(f"✅ Material assignment map saved to: {output_path}")
print("\nMaterial Assignments:")
for label, rule in sorted(assignments.items(), key=lambda x: str(x[0])):
    cluster_pixels = (labels == label).sum()
    percentage = (cluster_pixels / labels.size) * 100
    print(f"  • {rule.name.upper()}: Cluster {label} ({percentage:.1f}% of image)")

if unassigned:
    print(f"\nUnassigned Clusters: {len(unassigned)}")
    for label in unassigned:
        cluster_pixels = (labels == label).sum()
        percentage = (cluster_pixels / labels.size) * 100
        print(f"  • Cluster {label}: {percentage:.1f}% of image (below threshold)")
