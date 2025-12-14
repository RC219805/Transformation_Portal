# Water Detection: Ground Truth Schema (Final)

## Schema Design

**Two labels only**: `pool` and `ocean` (both are water)

**No "non_water" class**: Detector outputs `present: true/false`, not pool vs ocean classification

**Metrics**: Per-class recall, not cross-class confusion or false positive rate

---

## Ground Truth JSON Format

```json
{
  "version": "v0",
  "created": "2025-12-14",
  "root": "data/water_v0/images",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {
      "label": "pool",
      "difficulty": "easy",
      "notes": "clear water, direct sun",
      "tags": []
    },
    "pool/pool_0012.jpg": {
      "label": "pool",
      "difficulty": "hard",
      "notes": "dark water, evening lighting",
      "tags": ["low-light", "reflection"]
    },
    "ocean/ocean_0001.jpg": {
      "label": "ocean",
      "difficulty": "medium",
      "notes": "calm water, overcast",
      "tags": []
    }
  }
}
```

**Path format**: Relative to `data/water_v0/images/` (cross-platform safe)

---

## Folder Layout

```
data/water_v0/
├── images/
│   ├── pool/           # 20 images
│   └── ocean/          # 20 images
├── ground_truth.json
├── LABELING_GUIDE.md
└── README.md
```

---

## Detector Output (Current)

```python
WaterCandidateDetector.detect() returns:
{
  "present": bool,        # Water detected (true/false)
  "coverage": float,      # 0.0-1.0 (fraction of image)
  "coverage_px": int,     # Pixel count
  "confidence": float,    # 0.0-1.0
  "mask": ndarray         # (H, W) float32
}
```

**Note**: Detector does NOT distinguish pool vs ocean. Both labels map to `water`.

---

## Validation Metrics

### Per-Class Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Pool recall | detected pool / total pool | ≥85% |
| Ocean recall | detected ocean / total ocean | ≥85% |
| Pool avg coverage | mean(coverage) for detected pools | Report only |
| Ocean avg coverage | mean(coverage) for detected oceans | Report only |
| Pool avg edge alignment | mean(edge_score) for pools | ≥0.6 |
| Ocean avg edge alignment | mean(edge_score) for oceans | ≥0.6 |
| Pool avg stability | mean(stability) for pools | ≥0.8 |
| Ocean avg stability | mean(stability) for oceans | ≥0.8 |

**True Positive**: `label in ["pool", "ocean"]` AND `detector.present == true`

**False Negative**: `label in ["pool", "ocean"]` AND `detector.present == false`

**No false positives** (no non-water class in dataset)

### Report Schema

```json
{
  "summary": {
    "dataset_version": "v0",
    "total_images": 40,
    "pool_images": 20,
    "ocean_images": 20,
    
    "pool_recall": 0.90,
    "pool_avg_coverage": 0.78,
    "pool_avg_confidence": 0.68,
    "pool_avg_edge_alignment": 0.62,
    "pool_avg_stability": 0.80,
    
    "ocean_recall": 0.85,
    "ocean_avg_coverage": 0.65,
    "ocean_avg_confidence": 0.58,
    "ocean_avg_edge_alignment": 0.55,
    "ocean_avg_stability": 0.75,
    
    "overall_avg_processing_time_ms": 32.1
  },
  "results": [...]
}
```

---

## Hard Negative Tags

Track tricky cases **without adding labels**:

**Pool tags**:
- `low-light` - Dark/evening pools
- `reflection` - Strong surface reflections
- `tile` - Visible tile pattern
- `teal` - Green-tinted water
- `infinity-edge` - Infinity pool blending with horizon
- `indoor` - Indoor pool with artificial lighting

**Ocean tags**:
- `waves` - Choppy/wavy water
- `gray` - Desaturated/gray water
- `distant` - Ocean far from camera
- `overcast` - Overcast lighting
- `foam` - Whitecaps/foam visible

**Usage**:
```python
# Analyze failures by tag
pool_failures = [r for r in results if r["ground_truth_label"] == "pool" and not r["is_detected"]]

by_tag = {}
for result in pool_failures:
    for tag in result["tags"]:
        by_tag.setdefault(tag, []).append(result["image_path"])

for tag, paths in sorted(by_tag.items(), key=lambda x: -len(x[1])):
    print(f"{tag}: {len(paths)} failures")
```

---

## CI Regression Gates (Warning Mode)

**Baseline**: `data/water_v0/baseline_v0.json` (store initial run)

**Regression rules** (warn, don't fail):
```python
# scripts/check_regression.py

baseline = json.load(open("data/water_v0/baseline_v0.json"))
current = json.load(open("water_validation_ci.json"))

warnings = []

# Recall drops >10%
if current["summary"]["pool_recall"] < baseline["summary"]["pool_recall"] - 0.10:
    warnings.append(f"Pool recall dropped")

if current["summary"]["ocean_recall"] < baseline["summary"]["ocean_recall"] - 0.10:
    warnings.append(f"Ocean recall dropped")

# Edge alignment drops >0.1
if current["summary"]["pool_avg_edge_alignment"] < baseline["summary"]["pool_avg_edge_alignment"] - 0.1:
    warnings.append(f"Pool edge alignment dropped")

if current["summary"]["ocean_avg_edge_alignment"] < baseline["summary"]["ocean_avg_edge_alignment"] - 0.1:
    warnings.append(f"Ocean edge alignment dropped")

if warnings:
    print("⚠️ WARNING: Regression detected")
    for w in warnings:
        print(f"  - {w}")
```

**CI workflow**: Upload JSON as artifact, print warnings, **don't fail build** (warning mode)

---

## CI Subset (12 images, <10s validation)

```json
{
  "version": "ci_subset",
  "root": "data/water_v0/images",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {"label": "pool", "difficulty": "easy", "tags": []},
    "pool/pool_0003.jpg": {"label": "pool", "difficulty": "hard", "tags": ["low-light"]},
    "pool/pool_0005.jpg": {"label": "pool", "difficulty": "hard", "tags": ["teal"]},
    "pool/pool_0007.jpg": {"label": "pool", "difficulty": "hard", "tags": ["infinity-edge"]},
    "pool/pool_0008.jpg": {"label": "pool", "difficulty": "easy", "tags": []},
    "pool/pool_0009.jpg": {"label": "pool", "difficulty": "medium", "tags": ["indoor"]},
    
    "ocean/ocean_0001.jpg": {"label": "ocean", "difficulty": "easy", "tags": []},
    "ocean/ocean_0003.jpg": {"label": "ocean", "difficulty": "hard", "tags": ["gray"]},
    "ocean/ocean_0004.jpg": {"label": "ocean", "difficulty": "hard", "tags": ["waves"]},
    "ocean/ocean_0005.jpg": {"label": "ocean", "difficulty": "hard", "tags": ["distant"]},
    "ocean/ocean_0007.jpg": {"label": "ocean", "difficulty": "easy", "tags": []},
    "ocean/ocean_0009.jpg": {"label": "ocean", "difficulty": "hard", "tags": ["low-light"]}
  }
}
```

**Balanced**: 6 pool, 6 ocean; 4 easy, 4 medium, 4 hard

---

## Validation Script Mapping

**Minimal changes to existing `scripts/prw_water_validation.py`**:

```python
@dataclass
class ValidationResult:
    """Single validation test result."""
    image_path: str
    ground_truth_label: str  # "pool" or "ocean"
    difficulty: str
    tags: list
    
    # Detector output
    detector_present: bool
    detector_coverage: float
    detector_confidence: float
    
    # Computed metrics
    is_detected: bool  # True if detector.present == True
    edge_alignment_score: float
    boundary_px: int
    stability_score: float
    processing_time_ms: float
```

**Report generation**:
```python
def generate_report(results, output_path):
    pool_results = [r for r in results if r.ground_truth_label == "pool"]
    ocean_results = [r for r in results if r.ground_truth_label == "ocean"]
    
    summary = {
        "pool_recall": sum(r.is_detected for r in pool_results) / len(pool_results),
        "pool_avg_coverage": np.mean([r.detector_coverage for r in pool_results if r.is_detected]),
        "pool_avg_edge_alignment": np.mean([r.edge_alignment_score for r in pool_results if r.is_detected]),
        
        "ocean_recall": sum(r.is_detected for r in ocean_results) / len(ocean_results),
        "ocean_avg_coverage": np.mean([r.detector_coverage for r in ocean_results if r.is_detected]),
        "ocean_avg_edge_alignment": np.mean([r.edge_alignment_score for r in ocean_results if r.is_detected]),
    }
    
    # ... rest of report ...
```

---

## Complete Example Dataset (40 images)

```json
{
  "version": "v0",
  "created": "2025-12-14",
  "root": "data/water_v0/images",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {"label": "pool", "difficulty": "easy", "notes": "Clear blue, sun", "tags": []},
    "pool/pool_0002.jpg": {"label": "pool", "difficulty": "easy", "notes": "Resort pool", "tags": []},
    "pool/pool_0003.jpg": {"label": "pool", "difficulty": "hard", "notes": "Dark evening", "tags": ["low-light"]},
    "pool/pool_0004.jpg": {"label": "pool", "difficulty": "hard", "notes": "Reflections", "tags": ["reflection"]},
    "pool/pool_0005.jpg": {"label": "pool", "difficulty": "hard", "notes": "Teal water", "tags": ["teal"]},
    "pool/pool_0006.jpg": {"label": "pool", "difficulty": "medium", "notes": "Tile visible", "tags": ["tile"]},
    "pool/pool_0007.jpg": {"label": "pool", "difficulty": "hard", "notes": "Infinity edge", "tags": ["infinity-edge"]},
    "pool/pool_0008.jpg": {"label": "pool", "difficulty": "easy", "notes": "Lap pool", "tags": []},
    "pool/pool_0009.jpg": {"label": "pool", "difficulty": "medium", "notes": "Indoor pool", "tags": ["indoor"]},
    "pool/pool_0010.jpg": {"label": "pool", "difficulty": "easy", "notes": "Overhead view", "tags": []},
    "pool/pool_0011.jpg": {"label": "pool", "difficulty": "medium", "notes": "Overcast", "tags": []},
    "pool/pool_0012.jpg": {"label": "pool", "difficulty": "hard", "notes": "Twilight", "tags": ["low-light"]},
    "pool/pool_0013.jpg": {"label": "pool", "difficulty": "easy", "notes": "Clear, direct sun", "tags": []},
    "pool/pool_0014.jpg": {"label": "pool", "difficulty": "medium", "notes": "Partial shade", "tags": []},
    "pool/pool_0015.jpg": {"label": "pool", "difficulty": "hard", "notes": "Green tint", "tags": ["teal"]},
    "pool/pool_0016.jpg": {"label": "pool", "difficulty": "easy", "notes": "Bright blue", "tags": []},
    "pool/pool_0017.jpg": {"label": "pool", "difficulty": "medium", "notes": "Stone surround", "tags": []},
    "pool/pool_0018.jpg": {"label": "pool", "difficulty": "hard", "notes": "Mosaic tile", "tags": ["tile"]},
    "pool/pool_0019.jpg": {"label": "pool", "difficulty": "easy", "notes": "Resort, clear", "tags": []},
    "pool/pool_0020.jpg": {"label": "pool", "difficulty": "medium", "notes": "Curved edge", "tags": []},
    
    "ocean/ocean_0001.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Calm, sunny", "tags": []},
    "ocean/ocean_0002.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Overcast", "tags": ["overcast"]},
    "ocean/ocean_0003.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Gray water", "tags": ["gray"]},
    "ocean/ocean_0004.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Choppy", "tags": ["waves"]},
    "ocean/ocean_0005.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Distant", "tags": ["distant"]},
    "ocean/ocean_0006.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Foam visible", "tags": ["foam"]},
    "ocean/ocean_0007.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Clear blue", "tags": []},
    "ocean/ocean_0008.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Sunset", "tags": []},
    "ocean/ocean_0009.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Dusk", "tags": ["low-light"]},
    "ocean/ocean_0010.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Beach view", "tags": []},
    "ocean/ocean_0011.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Horizon shot", "tags": []},
    "ocean/ocean_0012.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Storm clouds", "tags": ["gray", "overcast"]},
    "ocean/ocean_0013.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Calm, clear", "tags": []},
    "ocean/ocean_0014.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Small waves", "tags": []},
    "ocean/ocean_0015.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Rough sea", "tags": ["waves"]},
    "ocean/ocean_0016.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Tropical blue", "tags": []},
    "ocean/ocean_0017.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Rocky shore", "tags": []},
    "ocean/ocean_0018.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Far horizon", "tags": ["distant"]},
    "ocean/ocean_0019.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Clear water", "tags": []},
    "ocean/ocean_0020.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Mid-distance", "tags": []}
  }
}
```

**Distribution**:
- Pool: 20 images (8 easy, 6 medium, 6 hard)
- Ocean: 20 images (8 easy, 6 medium, 6 hard)
- Total: 40 images

---

## Summary

**Two labels only**: Pool and ocean (both water)

**No FP rate**: No non-water class

**Metrics**: Per-class recall, coverage, edge alignment, stability

**Tags**: Track hard cases without label explosion

**CI**: Warn on regression (>10% recall drop, >0.1 edge drop), don't fail

**Subset**: 12 images for fast CI (<10s)
