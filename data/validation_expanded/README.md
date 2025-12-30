# Validation Dataset (Expanded - 18 Images)

## Overview

This stratified validation dataset expands from 7 images (85.7% accuracy, statistically fragile) to **18 images** for robust classifier generalization testing. Images are selected to represent diverse scene types, sizes, and aspect ratios across three luxury real estate properties.

## Stratification Strategy

### Interiors (Structure-Dominated) - 6 images

Expected classification: `structure_dominated` (edge_to_texture_ratio < 5, edge_density > 0.05)

1. **750Picacho_Kitchen.jpg** (3998×2249, Medium)
   - Rationale: Clean architectural edges (cabinets, counters, appliances)
   - Challenges: Reflective surfaces, high-contrast lighting

2. **750Picacho_PrimaryBathroom.jpg** (3975×2981, Medium)
   - Rationale: Geometric tile patterns, vanity edges, mirror frames
   - Challenges: Reflections, chrome fixtures

3. **750Picacho_GreatRoom.jpg** (3974×2981, Medium)
   - Rationale: Windows, beams, furniture edges, artwork frames
   - Challenges: Mixed depth planes, soft furnishings

4. **Montecito-Shores-12.jpg** (6708×4472, Large)
   - Rationale: Interior with clear structural elements
   - Challenges: Coastal property aesthetic (potentially softer edges)

5. **800-picacho-11.jpg** (6608×4356, Large)
   - Rationale: High-resolution interior with architectural detail
   - Challenges: Complex lighting, multiple depth planes

6. **800-picacho-6.jpg** (6708×4472, Large)
   - Rationale: Interior scene with strong geometric features
   - Challenges: Large size tests processing efficiency

### Water/Pool (Texture-Dominated) - 4 images

Expected classification: `texture_dominated` (edge_to_texture_ratio > 10, depth_variance < 0.02)

1. **750Picacho_Pool.jpg** (3998×2249, Medium)
   - Rationale: Water ripples, reflections, smooth depth gradient
   - Challenges: High-frequency RGB texture with low depth structure

2. **Montecito-Shores-3.jpg** (6708×4472, Large)
   - Rationale: Coastal water feature (pool or ocean view)
   - Challenges: Complex reflections, foam patterns

3. **Montecito-Shores-16.jpg** (6708×4472, Large)
   - Rationale: Water-dominant scene with minimal architectural edges
   - Challenges: Dynamic texture patterns, varying depth

4. **800-picacho-38.jpg** (6324×4743, Large)
   - Rationale: Pool or water feature with texture-rich surface
   - Challenges: Large file size, high-frequency detail

### Glass/Facades (Texture-Dominated) - 3 images

Expected classification: `texture_dominated` (edge_to_texture_ratio 5-10, depth_variance < 0.03)

1. **Montecito-Shores-7.jpg** (6708×4472, Large)
   - Rationale: Glass facades, reflective surfaces with minimal depth structure
   - Challenges: Reflections create false edges

2. **Montecito-Shores-10.jpg** (6708×4472, Large)
   - Rationale: Modern architecture with extensive glazing
   - Challenges: Sky reflections, transparency effects

3. **800-picacho-1.jpg** (6708×4472, Large)
   - Rationale: Glass and metal exterior with smooth surfaces
   - Challenges: Specular highlights, environmental reflections

### Aerials (Mixed Classification) - 3 images

Expected classification: Mixed (depends on foliage density, terrain complexity)

1. **750Picacho_Aerial.jpg** (3993×2396, Medium)
   - Rationale: Estate overview with mixed structural and organic elements
   - Expected: Likely `structure_dominated` (roof edges, property boundaries)

2. **Montecito-shores-aerial-2.jpg** (6324×4743, Large)
   - Rationale: Coastal aerial with ocean, terrain, and development
   - Expected: Possibly `texture_dominated` (ocean smoothness, foliage)

3. **Montecito-shores-aerial-4.jpg** (6324×4743, Large)
   - Rationale: Mixed terrain with both structural and natural features
   - Expected: Edge case - may reveal threshold sensitivity

### Challenging Cases - 2 images

Expected: Tests edge cases and classifier robustness

1. **Montecito-Shores-18.jpg** (4472×6708, Large, Portrait)
   - Rationale: Portrait orientation (rare in architectural photography)
   - Challenges: Vertical composition, potential edge detection bias

2. **800-picacho-28.jpg** (4472×6708, Large, Portrait)
   - Rationale: Portrait interior/exterior with complex depth
   - Challenges: Vertical surfaces, unusual framing

## Size Distribution

- **Small (<2000px shortest side)**: 0 images
  - Note: Available images all exceed 2000px; smallest is 2249px
- **Medium (2000-4000px)**: 7 images
  - 750 Picacho Source JPEGS (6 images): ~2249-2981px
  - 750 Picacho Aerial (1 image): 2396px
- **Large (>4000px)**: 11 images
  - 16 Seaview Source JPEGS (7 images): 4472-6708px
  - 800 Picacho High-Res (4 images): 4356-6708px

## Aspect Ratio Distribution

- **Landscape (>1.3:1)**: 16 images (89%)
  - Standard 3:2 ratio: Most images
  - Wide format: Aerials
- **Portrait (<0.77:1)**: 2 images (11%)
  - Montecito-Shores-18, 800-picacho-28
- **Panorama (>2:1)**: 0 images
  - Note: Excluded Montecito-shores-aerial-Pano.jpg (8705×5515) to avoid processing overhead

## Property Distribution

- **750 Picacho**: 7 images (39%)
  - Interiors: Kitchen, Bathroom, GreatRoom
  - Pool: 1 image
  - Aerial: 1 image
- **16 Seaview (Montecito Shores)**: 7 images (39%)
  - Interiors/exteriors: 4 images
  - Aerials: 2 images
  - Water features: 1 image
- **800 Picacho**: 4 images (22%)
  - Interiors/exteriors: 4 images

## Expected Classification Outcomes

### Confusion Matrix Target

|                      | Predicted: Texture | Predicted: Structure |
|----------------------|-------------------|---------------------|
| **Actual: Texture**      | 7 (TP)           | 0-1 (FN)           |
| **Actual: Structure**    | 0-1 (FP)         | 6 (TP)             |
| **Actual: Mixed**        | 2-3              | 0-1                |

**Total Expected Accuracy**: ≥90% (16-17 out of 18 correct)

### Classification Ground Truth Assumptions

**Structure-Dominated (6 images)**:
- All 6 interiors should classify as `structure_dominated`
- Threshold: edge_to_texture_ratio < 5 AND edge_density > 0.05

**Texture-Dominated (10 images)**:
- 4 water/pool images: High-frequency texture, smooth depth
- 3 glass/facade images: Reflections dominate, minimal true edges
- 3 aerials: Expected to vary (2-3 likely texture_dominated due to foliage/ocean)

**Lenient Pass Criteria**: ≥75% accuracy (14/18) if structure and texture strata are both validated correctly within their groups.

### Known Edge Cases

1. **Aerials (750Picacho_Aerial.jpg)**:
   - May classify as `structure_dominated` if roof edges dominate
   - May classify as `texture_dominated` if foliage/terrain dominates

2. **Glass facades (Montecito-Shores-7.jpg, -10.jpg)**:
   - Reflections may create false edges
   - Classifier should suppress via depth variance check

3. **Portrait orientation (Montecito-Shores-18.jpg, 800-picacho-28.jpg)**:
   - Tests isotropy of edge detection (no orientation bias)

## Validation Workflow

### Step 1: Run Validation Script

```bash
python scripts/automation/production_depth_validation_fixed.py \
  --image-dir /Users/rc/Transformation_Portal/data/validation_expanded
```

### Step 2: Analyze Results

Expected outputs:
- Classification distribution (structure vs texture)
- Confusion matrix (if ground truth labels provided)
- Misclassification report with images and scores

### Step 3: Threshold Decision

**If accuracy ≥90%**:
- Lock thresholds: `edge_to_texture_ratio=5`, `edge_density=0.05`, `depth_variance=0.02`
- Document in `BASELINE_THRESHOLD_ANALYSIS.md`
- Proceed to production validation

**If accuracy <90%**:
- Analyze misclassified images (visual inspection)
- Tune thresholds iteratively:
  - Increase `edge_to_texture_ratio` if too many structure → texture FPs
  - Decrease `edge_density` if too many texture → structure FNs
- Revalidate on this dataset

### Step 4: Generate Report

```bash
# Extract metrics
python extract_validation_metrics.py \
  --results validation_expanded_results.json \
  --output validation_expanded_report.md

# Generate confusion matrix
python generate_validation_report.py \
  --dataset validation_expanded \
  --output validation_expanded_confusion_matrix.md
```

## Selection Rationale

### Inclusion Criteria

✅ **Clear exemplars**: Each scene type has 2-4 representative images
✅ **Property diversity**: 3 properties, varied architectural styles
✅ **Size diversity**: 7 medium + 11 large (no small due to source material)
✅ **Aspect diversity**: 16 landscape + 2 portrait
✅ **Known challenges**: Reflections, foliage, portrait orientation, high-res processing

### Exclusion Criteria

❌ **Duplicates**: Avoided DJI_*_D 2.JPG duplicates
❌ **Near-duplicates**: Selected best representatives (e.g., 1 pool instead of 3)
❌ **Panoramas**: Excluded Montecito-shores-aerial-Pano.jpg (8705×5515) for speed
❌ **Overly similar scenes**: One kitchen, one bathroom, etc. instead of multiples
❌ **Processed variants**: Used Source_JPEGS, not signature/enhanced versions

## Dataset Statistics

- **Total images**: 18
- **Total pixels**: ~643 megapixels (avg 35.7 MP per image)
- **File size**: ~320 MB combined
- **Processing time**: ~18-45 seconds (M4 Max, batch mode)
- **Storage**: Lightweight validation set (fits in memory)

## Expansion Opportunities

If validation fails or additional strata needed:

1. **Add small images** (<2000px):
   - Downscale existing images OR
   - Use Kitchen_2K_test.png (already available)

2. **Add more challenging cases**:
   - Low-light interiors
   - Dense foliage (gardens, landscaping)
   - Patterned surfaces (tile mosaics, wood grain)

3. **Add temporal variation**:
   - Same scene at different times of day (if available)
   - Tests lighting robustness

## Reproducibility

All source images are committed to the repository:
- `/Users/rc/Transformation_Portal/input_images/750_Picacho/`
- `/Users/rc/Transformation_Portal/input_images/16_Seaview/`
- `/Users/rc/Transformation_Portal/input_images/800_Picacho/`

Images are not modified during validation (read-only access).

## Version History

- **v1.0** (2025-12-19): Initial stratified dataset (18 images)
  - Baseline for threshold validation
  - Target: ≥90% classification accuracy

---

**Next Steps**: Run validation script and compare results against expected confusion matrix.
