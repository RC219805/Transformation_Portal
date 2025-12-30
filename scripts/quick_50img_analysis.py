#!/usr/bin/env python3
"""Quick analysis of partial 50-image run."""

import json
import glob
from pathlib import Path
from collections import Counter

metrics_dir = Path("outputs/validation_full_50img_20251218_214935_2a2b25c")
metrics_files = sorted(metrics_dir.glob("*_metrics.json"))

print(f"=== Quick Analysis: Partial 50-Image Run ===\n")
print(f"Metrics files found: {len(metrics_files)}/50\n")

scene_types = []
lenient_passes = 0
strict_passes = 0
failures = []

for mf in metrics_files:
    with open(mf) as f:
        data = json.load(f)

    scene_type = data.get("scene_type", "unknown")
    scene_types.append(scene_type)

    lenient = data.get("lenient_pass", False)
    strict = data.get("strict_pass", False)

    if lenient:
        lenient_passes += 1
    if strict:
        strict_passes += 1

    if not lenient:
        failures.append(
            {
                "name": mf.stem.replace("_metrics", ""),
                "scene": scene_type,
                "edge_f1": data.get("edge_f1", "N/A"),
                "reason": data.get("gate_reason", "N/A"),
            }
        )

# Scene distribution
scene_counter = Counter(scene_types)
print(f"Scene Distribution:")
for scene, count in scene_counter.most_common():
    print(f"  {scene}: {count}")

print(f"\nPass Rates (on {len(metrics_files)} completed):")
print(f"  Lenient: {lenient_passes}/{len(metrics_files)} = {lenient_passes / len(metrics_files) * 100:.1f}%")
print(f"  Strict:  {strict_passes}/{len(metrics_files)} = {strict_passes / len(metrics_files) * 100:.1f}%")

print(f"\nLenient Failures ({len(failures)}):")
for f in failures[:5]:
    print(f"  - {f['name']}: {f['scene']}, edge_f1={f['edge_f1']}")

if len(failures) > 5:
    print(f"  ... and {len(failures) - 5} more")

# Check which images are missing
labels_file = Path("data/validation_full/labels.csv")
if labels_file.exists():
    with open(labels_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("filename")]

    expected = {l.split(",")[0] for l in lines}
    completed = {mf.stem.replace("_metrics", "").replace("V2_", "") for mf in metrics_files}
    missing = expected - completed

    print(f"\nMissing Images ({len(missing)}/50):")
    for img in sorted(missing)[:10]:
        print(f"  - {img}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
