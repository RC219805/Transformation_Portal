# Validation Dataset Architecture

**Document Version**: 1.0  
**Date**: 2025-12-19  
**Authored By**: Transformation Portal Architect  
**Purpose**: Stratified validation dataset for depth classifier threshold tuning

---

## Executive Summary

Successfully curated an **18-image stratified validation dataset** to replace the statistically fragile 7-image baseline (85.7% accuracy). The dataset tests classifier generalization across:

- **5 scene types**: Interiors, water/pool, glass/facades, aerials, challenging cases
- **3 properties**: 750 Picacho, 16 Seaview, 800 Picacho
- **2 size categories**: Medium (5), Large (13)
- **2 aspect ratios**: Landscape (16), Portrait (2)

**Expected accuracy**: ≥90% (16-17/18 correct)

---

## Dataset Structure

```
/data/validation_expanded/
├── images.txt                      # 18 image paths with stratification metadata
├── README.md                       # Comprehensive documentation (10 KB)
├── STRATIFICATION_SUMMARY.md       # Quick reference (4 KB)
├── 750Picacho_*.jpg               # 7 images from 750 Picacho estate
├── Montecito-Shores-*.jpg         # 7 images from 16 Seaview property
└── 800-picacho-*.jpg              # 4 images from 800 Picacho estate
```

**Total Size**: 330 MB (18 images, avg 35.7 MP)

---

## Stratification Design

### Scene Type Distribution

| Scene Type | Count | Expected Class | Rationale |
|------------|-------|----------------|-----------|
| **Interiors** | 6 (33%) | `structure_dominated` | Cabinets, tiles, frames, architectural edges |
| **Water/Pool** | 4 (22%) | `texture_dominated` | Ripples, reflections, smooth depth |
| **Glass/Facades** | 3 (17%) | `texture_dominated` | Reflections, minimal structural edges |
| **Aerials** | 3 (17%) | Mixed | Foliage vs. structure ratio determines class |
| **Challenging** | 2 (11%) | Variable | Portrait orientation, complex depth |

### Size & Aspect Ratio

- **Medium (2000-4000px)**: 5 images (28%)
- **Large (>4000px)**: 13 images (72%)
- **Landscape**: 16 images (89%)
- **Portrait**: 2 images (11%)

### Property Diversity

- **750 Picacho**: 7 images (39%) - Luxury estate with pool, interiors, aerial
- **16 Seaview**: 7 images (39%) - Coastal property with water features, glass facades
- **800 Picacho**: 4 images (22%) - High-resolution interiors and exteriors

---

## Architectural Decisions

### 1. No Small Images (<2000px)

**Decision**: Exclude small images from validation dataset.

**Rationale**:
- Source material does not contain images <2000px (smallest is 2249px)
- Adding downscaled versions would introduce artificial artifacts
- Production use case targets medium-to-large luxury real estate images

**Consequence**: Dataset does not test classifier on thumbnails or previews.

### 2. No Panoramas (>2:1 aspect ratio)

**Decision**: Exclude panoramas (e.g., Montecito-shores-aerial-Pano.jpg at 8705×5515).

**Rationale**:
- Processing overhead (45+ seconds per image on M4 Max)
- Rare in production workflows (panoramas typically stitched post-processing)
- 18 images already provide statistical power for generalization testing

**Consequence**: Classifier not validated on ultra-wide aspect ratios.

### 3. Portrait Orientation Inclusion (2 images)

**Decision**: Include 2 portrait images (11% of dataset).

**Rationale**:
- Tests isotropy of edge detection (no orientation bias)
- Rare but present in production (tall building facades, vertical interiors)
- Edge case for classifier robustness

**Consequence**: Low sample size may not reveal subtle portrait-specific issues.

### 4. Mixed Aerial Classification

**Decision**: Accept "mixed" classification for aerials (not forced into structure or texture).

**Rationale**:
- Aerials naturally vary based on foliage density, terrain complexity, ocean coverage
- Forcing binary classification would create false negatives
- Real-world use case: aerials may legitimately fall into either category

**Consequence**: Confusion matrix excludes aerials from strict accuracy calculation.

---

## Security Considerations

### Input Validation

- **Path Traversal**: `create_expanded_dataset.sh` uses `basename` to prevent directory traversal
- **File Type**: Validated `.jpg` and `.png` extensions only
- **Metadata Preservation**: `-p` flag preserves EXIF/IPTC without modification

### Privacy & Data Handling

- **GPS Coordinates**: Source images contain GPS metadata (luxury properties)
  - **Mitigation**: Validation dataset is local-only, not exposed to external APIs
  - **Recommendation**: Strip GPS metadata before production deployment
- **Client Data**: Images from real luxury properties (750 Picacho, 16 Seaview, 800 Picacho)
  - **Mitigation**: Used in internal validation only, not published

### Reproducibility

- **Deterministic Selection**: Image paths hardcoded in `images.txt`
- **Version Control**: All source images committed to repository (read-only)
- **Checksums**: Future enhancement - add SHA256 hashes to verify integrity

---

## Expected Validation Outcomes

### Confusion Matrix Target

|                      | Predicted: Texture | Predicted: Structure |
|----------------------|-------------------|---------------------|
| **Actual: Texture**      | 7 (TP)           | 0-1 (FN)           |
| **Actual: Structure**    | 0-1 (FP)         | 6 (TP)             |
| **Actual: Mixed (Aerials)** | 2-3              | 0-1                |

### Accuracy Targets

- **Strict**: ≥90% (16-17/18 correct)
  - Requires correct classification of all structure and texture images
  - Allows 1-2 misclassifications
- **Lenient**: ≥75% (14/18 correct)
  - Requires both structure and texture strata validated correctly
  - Allows 3-4 misclassifications (likely aerials)

### Key Metrics

- **Precision (Texture)**: 7/8 = 87.5%
- **Recall (Texture)**: 7/7 = 100%
- **Precision (Structure)**: 6/7 = 85.7%
- **Recall (Structure)**: 6/6 = 100%
- **Overall Accuracy**: 13/15 = 86.7% (excluding 3 mixed aerials)

---

## Integration with CI/CD

### Validation Workflow

```bash
# Step 1: Create dataset (idempotent, can be re-run)
bash scripts/validation/create_expanded_dataset.sh

# Step 2: Run classifier validation
python scripts/automation/production_depth_validation_fixed.py \
  --image-dir /Users/rc/Transformation_Portal/data/validation_expanded

# Step 3: Extract metrics
python extract_validation_metrics.py \
  --results validation_expanded_results.json \
  --output validation_expanded_report.md

# Step 4: Generate confusion matrix
python generate_validation_report.py \
  --dataset validation_expanded \
  --output validation_expanded_confusion_matrix.md
```

### Threshold Lock Criteria

**IF accuracy ≥90%**:
1. Lock thresholds in `BASELINE_THRESHOLD_ANALYSIS.md`:
   - `edge_to_texture_ratio = 5.0`
   - `edge_density = 0.05`
   - `depth_variance = 0.02`
2. Commit locked thresholds to version control
3. Proceed to production validation

**ELSE (accuracy <90%)**:
1. Analyze misclassified images (visual inspection)
2. Tune thresholds iteratively:
   - Increase `edge_to_texture_ratio` if too many structure → texture false positives
   - Decrease `edge_density` if too many texture → structure false negatives
3. Revalidate on this dataset (max 3 iterations)
4. If accuracy still <90%: Expand dataset or revise classification criteria

---

## Future Enhancements

### Phase 2: Expanded Stratification

- **Small Images**: Add downscaled versions or source thumbnails
- **Panoramas**: Include 1-2 wide-format aerials (if processing optimized)
- **Temporal Variation**: Same scene at different times of day (lighting robustness)
- **Material Focus**: More patterned surfaces (wood grain, tile mosaics, carpet)

### Phase 3: Ground Truth Annotation

- **Manual Labeling**: Expert review of "ground truth" structure vs. texture
- **Inter-Rater Reliability**: Multiple annotators for edge cases
- **Threshold Sensitivity**: Generate receiver operating characteristic (ROC) curves

### Phase 4: Production Monitoring

- **Drift Detection**: Monitor classification distribution in production
- **A/B Testing**: Compare locked thresholds vs. adaptive thresholds
- **Performance Tracking**: Log processing time, memory usage, accuracy over time

---

## Maintenance & Governance

### Dataset Ownership

- **Primary Contact**: Transformation Portal Architect
- **Approval Required**: Changes to stratification strategy or image selection
- **Version Control**: All changes tracked in `VALIDATION_DATASET_ARCHITECTURE.md`

### Update Cadence

- **Quarterly Review**: Assess dataset relevance as new properties added
- **Annual Refresh**: Re-stratify based on production classification distribution
- **On-Demand**: Add images for newly discovered edge cases

### Deprecation Policy

- **Retain Historical Versions**: Archive v1.0 dataset for reproducibility
- **Migration Path**: Provide scripts to convert old validation results to new format
- **Breaking Changes**: Require ADR approval and stakeholder notification

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-19 | Initial stratified dataset (18 images) | Transformation Portal Architect |

---

**Status**: ✓ Dataset Ready for Validation  
**Next Action**: Run `production_depth_validation_fixed.py` on expanded dataset
