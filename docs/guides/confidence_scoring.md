# Confidence Scoring for Materials V3 Segmentation

## Overview

Materials V3 segmentation now includes **confidence scoring** for all material classifications. This provides transparency and enables users to validate, filter, and debug segmentation results.

## What Changed

### Before (v2)
```python
# Backend returned only masks
results = backend.segment(image)
# {'glass': array([[0, 0, 1, ...]], dtype=float32)}
```

### After (v3 with Confidence)
```python
# Backend returns (mask, confidence) tuples
results = backend.segment(image)
# {'glass': (array([[0, 0, 1, ...]], dtype=float32), 0.87)}

for material, (mask, confidence) in results.items():
    print(f"{material}: {confidence:.0%} confidence")
# Output: glass: 87% confidence
```

## API Changes

### Backend Protocol
```python
from transformation_portal.lux_depth_v3.protocols import SegmentationBackend

class SegmentationBackend(Protocol):
    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Returns:
            Dict mapping material names to (mask, confidence) tuples:
            - mask: Binary mask (H, W) float32 [0.0-1.0]
            - confidence: Classification confidence [0.0-1.0]
        """
        ...
```

### Public API (Backward Compatible)
```python
from transformation_portal.lux_depth_v3.segmentation_backend import segment_materials

# Public API still returns Dict[str, np.ndarray] for backward compatibility
masks = segment_materials(image, config)

# For direct backend access with confidence:
backend = EfficientSAMBackend()
backend.load(device="cpu")
results = backend.segment(image)  # Returns tuples
```

## Confidence Score Interpretation

| Range | Meaning | Source | Action |
|-------|---------|--------|--------|
| 0.8-1.0 | Very High | Strong CLIP similarity | Trust and use |
| 0.6-0.8 | High | Good CLIP match | Use with confidence |
| 0.4-0.6 | Medium | Moderate match or heuristic | Review for critical tasks |
| 0.2-0.4 | Low | Weak CLIP match | Consider filtering |
| **0.5** | Heuristic | Color-based fallback | Not ML-classified |

### Special Values
- **0.5**: Returned by heuristic fallback (color-based segmentation, not CLIP)
- **< 0.2**: Automatically filtered (not returned by CLIP classifier)

## How Confidence is Computed

### CLIP Classification (ML-based)
```python
# For each material, confidence is area-weighted average of segment scores
segments = [
    {'material': 'glass', 'area': 1000, 'clip_score': 0.85},
    {'material': 'glass', 'area': 500,  'clip_score': 0.90},
]

# Weighted average:
confidence = (0.85 * 1000 + 0.90 * 500) / (1000 + 500)
           = 0.867 (87%)
```

### Heuristic Fallback (Color-based)
```python
# All heuristic results return fixed 0.5 confidence
# This signals "not ML-classified, use with caution"
heuristic_results = {
    'glass': (mask, 0.5),
    'water': (mask, 0.5),
    'foliage': (mask, 0.5),
}
```

## Usage Examples

### Basic Usage
```python
from transformation_portal.lux_depth_v3.segmentation_backend import EfficientSAMBackend

backend = EfficientSAMBackend()
backend.load(device="cpu")

results = backend.segment(image)

for material, (mask, confidence) in results.items():
    print(f"{material}: {confidence:.0%} confidence, {mask.sum():.0f}px coverage")

# Output:
# glass: 87% confidence, 45230px coverage
# water: 34% confidence, 12450px coverage (low - maybe filter?)
# foliage: 76% confidence, 28900px coverage
```

### Confidence-Based Filtering
```python
# Filter low-confidence detections
MIN_CONFIDENCE = 0.6

high_confidence_only = {
    material: (mask, conf)
    for material, (mask, conf) in results.items()
    if conf >= MIN_CONFIDENCE
}

print(f"Filtered {len(results)} → {len(high_confidence_only)} materials")
# Output: Filtered 3 → 2 materials (removed low-confidence water)
```

### Production Quality Control
```python
# Example: Warn on low confidence materials
for material, (mask, confidence) in results.items():
    if confidence < 0.5:
        logger.warning(
            f"Low confidence detection: {material} ({confidence:.1%}). "
            f"Consider manual review for this region."
        )
    elif confidence == 0.5:
        logger.info(f"{material} detected via heuristics (not ML)")
```

### Debugging False Positives
```python
# When debugging incorrect detections, check confidence
results = backend.segment(problematic_image)

for material, (mask, confidence) in results.items():
    coverage = (mask.sum() / mask.size) * 100

    print(f"{material:12s} | {confidence:5.1%} | {coverage:5.1f}% coverage")

    # Low confidence + high coverage = likely false positive
    if confidence < 0.4 and coverage > 10:
        print(f"  ⚠️  Suspicious: low confidence but high coverage")
```

## Logging Output

Confidence scores now appear in logs:

```
INFO: CLIP classified 12 segments into 3 materials: glass (87%), water (34%), foliage (76%)
```

## Testing

### Test Confidence Scores
```python
import pytest
from transformation_portal.lux_depth_v3.segmentation_backend import EfficientSAMBackend

@pytest.mark.ml
def test_confidence_scores_valid():
    backend = EfficientSAMBackend()
    backend.load(device="cpu")

    results = backend.segment(test_image)

    for material, (mask, confidence) in results.items():
        assert 0.0 <= confidence <= 1.0, \
            f"{material} confidence {confidence} not in [0.0-1.0]"
```

### Test Heuristic Confidence
```python
@pytest.mark.ml
def test_heuristic_returns_medium_confidence():
    backend = EfficientSAMBackend()
    backend.load(device="cpu")
    backend._use_real_model = False  # Force heuristic mode

    results = backend.segment(test_image)

    for material, (mask, confidence) in results.items():
        assert confidence == 0.5, \
            f"Heuristic should return 0.5, got {confidence}"
```

## Migration Guide

### For Direct Backend Users

**Old Code:**
```python
backend = EfficientSAMBackend()
backend.load()
masks = backend.segment(image)

for material, mask in masks.items():
    apply_enhancement(mask, material)
```

**New Code:**
```python
backend = EfficientSAMBackend()
backend.load()
results = backend.segment(image)

for material, (mask, confidence) in results.items():
    if confidence >= 0.5:  # Optional filtering
        apply_enhancement(mask, material)
    else:
        logger.warning(f"Skipping {material} (low confidence: {confidence:.1%})")
```

### For Public API Users

**No changes required!** The `segment_materials()` function still returns `Dict[str, np.ndarray]`:

```python
from transformation_portal.lux_depth_v3.segmentation_backend import segment_materials

# Still works exactly as before
masks = segment_materials(image, config)

for material, mask in masks.items():
    apply_enhancement(mask, material)
```

## Performance Impact

- **Zero performance impact**: Confidence scores are already computed by CLIP but were previously discarded
- **Memory**: Minimal (~8 bytes per material for float64 confidence)
- **Computation**: No additional inference, just simple averaging

## Future Enhancements

Potential additions for confidence scoring:

1. **Per-pixel confidence maps**: Store spatial confidence distribution
2. **Confidence thresholds in config**: User-configurable minimum confidence
3. **Confidence-based mask refinement**: Use confidence to weight mask fusion
4. **Calibrated confidence**: Calibrate CLIP scores to true accuracy

## Implementation Details

### Files Changed
- `src/transformation_portal/lux_depth_v3/protocols/segmentation_backend.py` - Protocol definition
- `src/transformation_portal/lux_depth_v3/segmentation_backend.py` - Backend implementations
- `src/transformation_portal/stage_graph/stages/materials.py` - Stage integration
- `tests/materials/test_segmentation_backend.py` - Test updates

### Test Coverage
- 6 new confidence-specific tests
- All existing tests updated for tuple unpacking
- 24 tests passing (100% success rate)

## References

- CLIP Paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- EfficientSAM: [Segment Anything in High Quality](https://arxiv.org/abs/2312.00863)
- Materials V3 Docs: `docs/reference/materials_v3_quick_reference_old.md` (Legacy Materials V3 quick reference)

## Support

For questions or issues:
1. Check logs for confidence values
2. Review `examples/confidence_scoring_demo.py`
3. Run tests: `pytest tests/materials/test_segmentation_backend.py -v`
