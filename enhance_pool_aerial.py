"""Enhance the 750 Picacho pool aerial with MBAR board materials."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from board_material_aerial_enhancer import (
    enhance_aerial,
    _downsample_image,
    _kmeans,
    _assign_full_image,
    _cluster_stats,
    build_material_rules,
    assign_materials,
    DEFAULT_TEXTURES,
)

# Input and output paths
input_path = Path("input_images/RC_002RC-office750Picacho_Pool 2.tiff")
output_dir = Path("processed_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate output filename
output_path = output_dir / "750_Picacho_Pool_MBAR_Enhanced.jpg"
material_map_path = output_dir / "750_Picacho_Pool_Material_Assignment_Map.jpg"

print(f"Processing: {input_path}")
print(f"Output: {output_path}")
print()

# pylint: disable=duplicate-code  # Similar enhance_aerial call in run_aerial_enhancement.py
# Run enhancement
result = enhance_aerial(
    input_path,
    output_path,
    analysis_max_dim=1280,  # Clustering resolution
    k=8,                     # Number of clusters
    seed=22,                 # Reproducibility
    target_width=4096,       # 4K output
)
# pylint: enable=duplicate-code

print(f"\n✓ Enhancement complete: {output_path}")
print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
print("  Resolution: 4096px width")

# Create material assignment visualization
image = Image.open(input_path).convert("RGB")
# pylint: disable=duplicate-code  # Similar clustering logic used in visualize_material_assignments.py
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
labels_small_img = Image.fromarray(labels_small.astype("uint8")).convert("L")
labels_full = labels_small_img.resize(image.size, Image.Resampling.NEAREST)
labels = np.asarray(labels_full, dtype=np.uint8)

base_array = np.asarray(image, dtype=np.float32) / 255.0
stats = _cluster_stats(base_array, labels)
rules = build_material_rules(DEFAULT_TEXTURES)
assignments = assign_materials(stats, rules)

# Create color-coded visualization
colors = [
    (220, 200, 180),  # Plaster - warm beige
    (180, 160, 140),  # Stone - tan
    (160, 140, 120),  # Cladding - warm brown
    (140, 135, 130),  # Screens - grey
    (90, 90, 95),     # Equitone - dark grey
    (150, 150, 155),  # Roof - silvered wood
    (70, 55, 45),     # Bronze - dark metallic
    (235, 235, 235),  # Shade - white
]

colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
for idx in range(labels.max() + 1):
    colored[labels == idx] = colors[idx % len(colors)]

vis = Image.fromarray(colored)
draw = ImageDraw.Draw(vis)

# Add legend
legend_y = 30
legend_x = 30
for idx, (label, rule) in enumerate(assignments.items()):
    stat = next(s for s in stats if s.label == label)
    coverage = (stat.count / labels.size) * 100
    color = colors[label % len(colors)]

    # Draw color swatch
    draw.rectangle(
        [legend_x, legend_y + idx * 40, legend_x + 30, legend_y + idx * 40 + 25],
        fill=color,
        outline=(255, 255, 255),
        width=2,
    )

    # Draw text
    text = f"{rule.name.upper()}: {coverage:.1f}%"
    draw.text(
        (legend_x + 40, legend_y + idx * 40 + 5),
        text,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

vis.save(material_map_path, quality=95)
print(f"\n✓ Material map saved: {material_map_path}")
print(f"  Size: {material_map_path.stat().st_size / 1024 / 1024:.1f} MB")

# Generate detailed report
report_path = output_dir / "Pool_MBAR_Enhancement_Report.md"
with open(report_path, "w") as f:
    f.write("# 750 Picacho Lane - Pool Aerial Enhancement Report\n\n")
    f.write("## Processing Parameters\n\n")
    f.write(f"- **Input**: `{input_path.name}`\n")
    f.write(f"- **Output**: `{output_path.name}`\n")
    f.write("- **Analysis Resolution**: 1280px max dimension\n")
    f.write("- **Clusters**: 8\n")
    f.write("- **Target Width**: 4096px (4K)\n")
    f.write("- **Random Seed**: 22\n\n")

    f.write("## Material Assignments\n\n")
    f.write("| Material | Coverage | Mean RGB | Mean HSV |\n")
    f.write("|----------|----------|----------|----------|\n")

    total_assigned = 0
    for label, rule in sorted(assignments.items(), key=lambda x: str(x[0])):
        stat = next(s for s in stats if s.label == label)
        coverage = (stat.count / labels.size) * 100
        total_assigned += coverage
        rgb = f"({stat.mean_rgb[0]:.2f}, {stat.mean_rgb[1]:.2f}, {stat.mean_rgb[2]:.2f})"
        hsv = f"({stat.mean_hsv[0]:.2f}, {stat.mean_hsv[1]:.2f}, {stat.mean_hsv[2]:.2f})"
        f.write(f"| **{rule.name.title()}** | {coverage:.1f}% | {rgb} | {hsv} |\n")

    unassigned = 100 - total_assigned
    if unassigned > 0.1:
        f.write(f"| *Unassigned* | {unassigned:.1f}% | - | - |\n")

    f.write("\n## MBAR Material Specifications\n\n")
    f.write("### Applied Materials\n\n")

    materials_specs = {
        "plaster": "Marmorino Palladino Plaster - Westwood Beige",
        "stone": "Eco Outdoor Bokara Stone - Coastal",
        "cladding": "Sculptform Click-On Systems - Warm Timber",
        "screens": "Grey Gum Screens - Natural",
        "equitone": "Equitone LT85 Panels - Anthracite",
        "roof": "Bison Weathered Ipe Pavers",
        "bronze": "Dark Bronze Anodized Metal",
        "shade": "Louvretec Powder Coated White",
    }

    for label, rule in assignments.items():
        if rule.name in materials_specs:
            f.write(f"- **{rule.name.title()}**: {materials_specs[rule.name]}\n")
            f.write(f"  - Blend: {rule.blend * 100:.0f}%\n")
            if rule.tint:
                f.write(f"  - Tint: RGB{rule.tint} @ {rule.tint_strength * 100:.0f}%\n")

    f.write("\n## Pool Area Specific Notes\n\n")
    f.write("This enhancement focuses on the pool and surrounding hardscape:\n\n")
    f.write("- **Pool Deck**: Likely identified as stone or roof material (Bokara/Ipe pavers)\n")
    f.write("- **Pool Water**: May be assigned to equitone or screens (blue-grey tones)\n")
    f.write("- **Landscaping**: Vegetation typically unassigned or low-confidence clusters\n")
    f.write("- **Structures**: Plaster walls, bronze details, shade elements\n\n")

    f.write("## Recommendations\n\n")
    f.write("- For more pool-specific material detection, consider increasing `k` to 10-12 clusters\n")
    f.write("- Water reflections may benefit from custom water material rule\n")
    f.write("- Deck materials could use higher blend strength (0.7-0.8) for stronger effect\n")
    f.write("- Consider masking vegetation areas before processing for cleaner results\n")

print(f"\n✓ Report saved: {report_path}")
print(f"\n{'=' * 60}")
print("Pool aerial enhancement complete!")
print(f"{'=' * 60}")
