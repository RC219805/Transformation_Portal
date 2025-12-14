# Water Detection Ground Truth Schema

## Simplified Two-Label Schema (Photo-First, CI-Safe)

### Ground Truth JSON Format

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
    },
    "ocean/ocean_0015.jpg": {
      "label": "ocean",
      "difficulty": "hard",
      "notes": "choppy water, strong waves",
      "tags": ["texture", "waves"]
    }
  }
}
```

**Key Design Decisions**:
1. **Two labels only**: `pool` and `ocean` (both are water)
2. **Relative paths**: From `data/water_v0/images/` for cross-platform compatibility
3. **Difficulty tags**: Track hard cases without changing labels
4. **Optional tags**: Track tricky characteristics (reflection, tile, low-light, waves, etc.)

### Folder Layout

```
data/water_v0/
├── images/
│   ├── pool/
│   │   ├── pool_0001.jpg
│   │   ├── pool_0002.jpg
│   │   └── ...
│   └── ocean/
│       ├── ocean_0001.jpg
│       ├── ocean_0002.jpg
│       └── ...
├── ground_truth.json
├── LABELING_GUIDE.md
└── README.md
```

### Detector Output → Validation Metrics Mapping

**Current Detector Output** (`WaterCandidateDetector.detect()`):
```python
{
  "present": bool,        # Water detected
  "coverage": float,      # 0.0-1.0 (fraction of image)
  "coverage_px": int,     # Pixel count
  "confidence": float,    # 0.0-1.0
  "mask": ndarray         # (H, W) float32, values 0.0-1.0
}
```

**Validation Metrics** (computed in PR-W4 harness):

#### Per-Class Metrics (Pool / Ocean)

| Metric | Computation | Target |
|--------|-------------|--------|
| **Recall** | TP / (TP + FN) | ≥85% per class |
| **Coverage** | Average coverage for detected images | Report only |
| **Confidence** | Average confidence for detected images | Report only |
| **Edge Alignment** | Boundary-gradient overlap | ≥0.6 per class |
| **Stability** | Coverage variance across perturbations | ≥0.8 per class |

**True Positive (TP)**: Image labeled `pool` or `ocean`, detector returns `present=True`

**False Negative (FN)**: Image labeled `pool` or `ocean`, detector returns `present=False`

#### Cross-Class Confusion (Optional)

If detector can distinguish pool vs ocean (future enhancement):
- **Pool→Ocean**: Pool image classified as ocean
- **Ocean→Pool**: Ocean image classified as pool

**Current detector** returns only `water/not-water`, so cross-class confusion is N/A for v0.

### CI Regression Gates (Warning Mode)

**Baseline**: Store initial validation report as `data/water_v0/baseline_v0.json`

**Regression Rules** (warning, not error):
```python
# scripts/check_regression.py

baseline_pool_recall = baseline["summary"]["pool_recall"]
baseline_ocean_recall = baseline["summary"]["ocean_recall"]
baseline_edge_pool = baseline["summary"]["pool_avg_edge_alignment"]
baseline_edge_ocean = baseline["summary"]["ocean_avg_edge_alignment"]

current_pool_recall = current["summary"]["pool_recall"]
current_ocean_recall = current["summary"]["ocean_recall"]
current_edge_pool = current["summary"]["pool_avg_edge_alignment"]
current_edge_ocean = current["summary"]["ocean_avg_edge_alignment"]

warnings = []

# Recall drops
if current_pool_recall < baseline_pool_recall - 0.10:
    warnings.append(f"Pool recall dropped {current_pool_recall:.1%} vs baseline {baseline_pool_recall:.1%}")

if current_ocean_recall < baseline_ocean_recall - 0.10:
    warnings.append(f"Ocean recall dropped {current_ocean_recall:.1%} vs baseline {baseline_ocean_recall:.1%}")

# Edge alignment drops
if current_edge_pool < baseline_edge_pool - 0.1:
    warnings.append(f"Pool edge alignment dropped {current_edge_pool:.2f} vs baseline {baseline_edge_pool:.2f}")

if current_edge_ocean < baseline_edge_ocean - 0.1:
    warnings.append(f"Ocean edge alignment dropped {current_edge_ocean:.2f} vs baseline {baseline_edge_ocean:.2f}")

if warnings:
    print("⚠️ WARNING: Quality regression detected")
    for w in warnings:
        print(f"  - {w}")
else:
    print("✅ No regression detected")
```

### Updated Validation Report Schema

**Per-Image Results**:
```json
{
  "image_path": "pool/pool_0001.jpg",
  "ground_truth_label": "pool",
  "difficulty": "easy",
  "tags": [],
  
  "detector_present": true,
  "detector_coverage": 0.85,
  "detector_coverage_px": 55296,
  "detector_confidence": 0.72,
  
  "is_detected": true,
  "edge_alignment_score": 0.68,
  "boundary_px": 1024,
  "stability_score": 0.82,
  "processing_time_ms": 28.5
}
```

**Summary Statistics**:
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

### Hard Negative Tags

**Common tags for tracking tricky cases**:

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

**Usage in failure analysis**:
```python
# Analyze failures by tag
pool_failures_by_tag = {}
for result in results:
    if result["ground_truth_label"] == "pool" and not result["is_detected"]:
        for tag in result["tags"]:
            pool_failures_by_tag.setdefault(tag, []).append(result["image_path"])

print("Pool failures by tag:")
for tag, paths in sorted(pool_failures_by_tag.items(), key=lambda x: -len(x[1])):
    print(f"  {tag}: {len(paths)} failures")
    for path in paths[:3]:  # Show top 3
        print(f"    - {path}")
```

### Validation Script Updates

**Minimal changes to `scripts/prw_water_validation.py`**:

```python
def validate_single(self, img_path: Path, expected_label: str, difficulty: str, tags: list) -> ValidationResult:
    """Validate single image."""
    # ... existing code ...
    
    # Compute is_detected (both pool and ocean are "water")
    is_detected = water_dict.get('present', False)
    
    return ValidationResult(
        image_path=str(img_path),
        ground_truth_label=expected_label,
        difficulty=difficulty,
        tags=tags,
        
        detector_present=is_detected,
        detector_coverage=water_dict.get('coverage', 0.0),
        detector_coverage_px=water_dict.get('coverage_px', 0),
        detector_confidence=water_dict.get('confidence', 0.0),
        
        is_detected=is_detected,  # TP if present, FN if not
        edge_alignment_score=edge_score,
        boundary_px=boundary_px,
        stability_score=stability,
        processing_time_ms=elapsed_ms
    )

def generate_report(self, results: List[ValidationResult], output_path: Path):
    """Generate JSON validation report."""
    pool_results = [r for r in results if r.ground_truth_label == "pool"]
    ocean_results = [r for r in results if r.ground_truth_label == "ocean"]
    
    summary = {
        "dataset_version": "v0",
        "total_images": len(results),
        "pool_images": len(pool_results),
        "ocean_images": len(ocean_results),
        
        # Pool metrics
        "pool_recall": sum(r.is_detected for r in pool_results) / len(pool_results) if pool_results else 0.0,
        "pool_avg_coverage": np.mean([r.detector_coverage for r in pool_results if r.is_detected]) if any(r.is_detected for r in pool_results) else 0.0,
        "pool_avg_confidence": np.mean([r.detector_confidence for r in pool_results if r.is_detected]) if any(r.is_detected for r in pool_results) else 0.0,
        "pool_avg_edge_alignment": np.mean([r.edge_alignment_score for r in pool_results if r.is_detected]) if any(r.is_detected for r in pool_results) else 0.0,
        "pool_avg_stability": np.mean([r.stability_score for r in pool_results]) if pool_results else 0.0,
        
        # Ocean metrics
        "ocean_recall": sum(r.is_detected for r in ocean_results) / len(ocean_results) if ocean_results else 0.0,
        "ocean_avg_coverage": np.mean([r.detector_coverage for r in ocean_results if r.is_detected]) if any(r.is_detected for r in ocean_results) else 0.0,
        "ocean_avg_confidence": np.mean([r.detector_confidence for r in ocean_results if r.is_detected]) if any(r.is_detected for r in ocean_results) else 0.0,
        "ocean_avg_edge_alignment": np.mean([r.edge_alignment_score for r in ocean_results if r.is_detected]) if any(r.is_detected for r in ocean_results) else 0.0,
        "ocean_avg_stability": np.mean([r.stability_score for r in ocean_results]) if ocean_results else 0.0,
        
        # Overall
        "overall_avg_processing_time_ms": np.mean([r.processing_time_ms for r in results]),
    }
    
    report = {
        "summary": summary,
        "results": [vars(r) for r in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
```

### Example Ground Truth (20 pool, 20 ocean)

```json
{
  "version": "v0",
  "created": "2025-12-14",
  "root": "data/water_v0/images",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {"label": "pool", "difficulty": "easy", "notes": "Clear blue, direct sun", "tags": []},
    "pool/pool_0002.jpg": {"label": "pool", "difficulty": "medium", "notes": "Overcast lighting", "tags": []},
    "pool/pool_0003.jpg": {"label": "pool", "difficulty": "hard", "notes": "Dark evening pool", "tags": ["low-light"]},
    "pool/pool_0004.jpg": {"label": "pool", "difficulty": "hard", "notes": "Strong reflections", "tags": ["reflection"]},
    "pool/pool_0005.jpg": {"label": "pool", "difficulty": "hard", "notes": "Teal-tinted water", "tags": ["teal"]},
    "pool/pool_0006.jpg": {"label": "pool", "difficulty": "medium", "notes": "Tile pattern visible", "tags": ["tile"]},
    "pool/pool_0007.jpg": {"label": "pool", "difficulty": "hard", "notes": "Infinity edge", "tags": ["infinity-edge"]},
    "pool/pool_0008.jpg": {"label": "pool", "difficulty": "easy", "notes": "Resort pool, clear", "tags": []},
    "pool/pool_0009.jpg": {"label": "pool", "difficulty": "medium", "notes": "Indoor pool", "tags": ["indoor"]},
    "pool/pool_0010.jpg": {"label": "pool", "difficulty": "easy", "notes": "Lap pool, overhead view", "tags": []},
    
    "ocean/ocean_0001.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Calm ocean, sunny", "tags": []},
    "ocean/ocean_0002.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Overcast, desaturated", "tags": ["overcast"]},
    "ocean/ocean_0003.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Gray water, low saturation", "tags": ["gray"]},
    "ocean/ocean_0004.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Choppy waves", "tags": ["waves"]},
    "ocean/ocean_0005.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Distant ocean, horizon", "tags": ["distant"]},
    "ocean/ocean_0006.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Foam visible", "tags": ["foam"]},
    "ocean/ocean_0007.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Clear blue, calm", "tags": []},
    "ocean/ocean_0008.jpg": {"label": "ocean", "difficulty": "medium", "notes": "Sunset lighting", "tags": []},
    "ocean/ocean_0009.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Dark water, dusk", "tags": ["low-light"]},
    "ocean/ocean_0010.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Beach view", "tags": []}
  }
}
```

### CI Subset (12 images for fast validation)

```json
{
  "version": "ci_subset",
  "root": "data/water_v0/images",
  "parent": "data/water_v0/ground_truth.json",
  "labels": ["pool", "ocean"],
  "images": {
    "pool/pool_0001.jpg": {"label": "pool", "difficulty": "easy", "notes": "Clear blue", "tags": []},
    "pool/pool_0003.jpg": {"label": "pool", "difficulty": "hard", "notes": "Dark evening", "tags": ["low-light"]},
    "pool/pool_0005.jpg": {"label": "pool", "difficulty": "hard", "notes": "Teal water", "tags": ["teal"]},
    "pool/pool_0007.jpg": {"label": "pool", "difficulty": "hard", "notes": "Infinity edge", "tags": ["infinity-edge"]},
    "pool/pool_0008.jpg": {"label": "pool", "difficulty": "easy", "notes": "Resort pool", "tags": []},
    "pool/pool_0009.jpg": {"label": "pool", "difficulty": "medium", "notes": "Indoor", "tags": ["indoor"]},
    
    "ocean/ocean_0001.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Calm ocean", "tags": []},
    "ocean/ocean_0003.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Gray water", "tags": ["gray"]},
    "ocean/ocean_0004.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Choppy waves", "tags": ["waves"]},
    "ocean/ocean_0005.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Distant", "tags": ["distant"]},
    "ocean/ocean_0007.jpg": {"label": "ocean", "difficulty": "easy", "notes": "Clear blue", "tags": []},
    "ocean/ocean_0009.jpg": {"label": "ocean", "difficulty": "hard", "notes": "Dusk", "tags": ["low-light"]}
  }
}
```

**Balanced across**:
- Difficulty: 4 easy, 4 medium, 4 hard
- Class: 6 pool, 6 ocean
- Tags: Covers main failure modes

This keeps CI fast (<10 seconds) while catching regressions.
