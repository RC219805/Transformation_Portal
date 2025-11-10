# Corrective Action Plan: v1.2.0 Quality Recovery

**Plan Date**: November 10, 2025  
**Objective**: Restore v1.0.0 quality while addressing genuine issues  
**Target Version**: 1.2.0  
**Status**: DRAFT - Awaiting approval

---

## Strategic Approach

### Core Principle: First, Do No Harm

```
v1.0.0: ★★★★★ quality (expert validated)
v1.1.0: ★★★☆☆ quality (failed)
v1.2.0: ★★★★★ quality + targeted improvements ONLY
```

**Philosophy**: 
1. Start with v1.0.0 as gold standard
2. Add ONLY improvements that enhance, not alter
3. Validate EVERY change with visual comparison
4. Smaller, incremental changes > big rewrites

---

## Phase 1: Immediate Remediation (Week 1)

### Action 1.1: Revert to v1.0.0 Baseline ✅ CRITICAL

**Objective**: Restore known-good processing pipeline

**Implementation**:
```bash
# Git workflow
git checkout v1.0.0-tag  # Or commit hash before v1.1.0 changes
git checkout -b v1.2.0-development

# Alternatively: Manual revert of specific changes
git revert <v1.1.0-commit-hash>
```

**Files to Restore**:
1. `luxury_estate_master_pipeline.py` - Core pipeline
2. `config/750_picacho_master_preset.yaml` - Preset configuration

**Verification**:
```bash
# Re-process test image
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Pool_HDR_32-bit.tif \
  --preset 750_picacho \
  --output-dir verification_v1.0.0_restored

# Visual comparison with original v1.0.0 output
# MUST match exactly
```

**Success Criteria**:
- [ ] Output visually identical to original v1.0.0
- [ ] File sizes match v1.0.0 baseline (115MB TIFF, 8.4MB JPEG)
- [ ] Processing time matches v1.0.0 (~13-14s per image)
- [ ] Expert review confirms ★★★★★ quality restored

**Timeline**: Immediate (Day 1)

---

### Action 1.2: Document v1.1.0 Lessons Learned ✅

**Objective**: Create institutional knowledge to prevent recurrence

**Deliverables**:
1. ✅ `Root_Cause_Analysis.md` - Technical autopsy
2. ✅ `Corrective_Action_Plan.md` - This document
3. ⏳ Update `DEPRECATION_POLICY.md` - Mark v1.1.0 as deprecated
4. ⏳ Create `LESSONS_LEARNED.md` - Design patterns to avoid

**Key Documentation Points**:
- Pipeline order matters (shadow boost AFTER color grading)
- Metrics ≠ perceptual quality
- Visual validation is mandatory
- Feature additions need isolation testing

**Timeline**: Day 1-2

---

### Action 1.3: Create Visual QA Framework ✅ CRITICAL

**Objective**: Prevent future metric-only validation

**Implementation**: New QA checkpoint system

```python
# tools/visual_qa_checklist.py (NEW)

class VisualQAChecklist:
    """Mandatory visual quality checkpoints."""
    
    checkpoints = {
        "white_balance": {
            "check": "Whites are neutral (not yellow/blue tinted)",
            "method": "Sample white pixels, check RGB ratio",
            "threshold": "R:G:B within 1.05:1.00:1.05"
        },
        "color_fidelity": {
            "check": "Pool blue is vivid and saturated",
            "method": "Sample pool tiles, check HSV saturation",
            "threshold": "Saturation > 0.6, Hue in 180-220° range"
        },
        "dynamic_range": {
            "check": "Highlights preserved, not blown/muted",
            "method": "Histogram analysis of 95th-99th percentile",
            "threshold": "Highlight range > 0.15 (15% of tonal range)"
        },
        "microcontrast": {
            "check": "Textures crisp, not mushy",
            "method": "Local standard deviation in texture regions",
            "threshold": "Local std dev > 0.08"
        },
        "tonal_separation": {
            "check": "Clear warm interior / cool exterior separation",
            "method": "Color temperature difference",
            "threshold": "Interior-Exterior ΔT > 500K"
        }
    }
    
    def validate(self, image_path: Path, reference_path: Path = None):
        """Run all checkpoints, generate report."""
        results = {}
        for name, checkpoint in self.checkpoints.items():
            results[name] = self._run_checkpoint(image_path, checkpoint)
            
            # Compare with reference if provided
            if reference_path:
                ref_result = self._run_checkpoint(reference_path, checkpoint)
                results[name]["delta"] = results[name]["value"] - ref_result["value"]
                results[name]["regression"] = results[name]["delta"] < -0.1  # 10% worse
        
        return VisualQAReport(results)
```

**Integration**:
```python
# luxury_estate_master_pipeline.py - Add to process_image()

if self.config.quality_control.visual_qa_enabled:
    qa = VisualQAChecklist()
    report = qa.validate(
        output_path, 
        reference_path=self.config.quality_control.reference_image
    )
    
    if report.has_regressions():
        logger.warning(f"⚠️  Quality regression detected: {report.failures}")
        if self.config.quality_control.fail_on_regression:
            raise QualityRegressionError(report)
```

**Timeline**: Week 1

---

## Phase 2: Targeted Improvements (Week 2-3)

### Principle: ONE Change at a Time

**Critical Rule**: Each improvement must be:
1. Isolated (no other changes in same commit)
2. Visually validated (A/B comparison with expert review)
3. Metrics + perception verified
4. Reversible (feature flag to disable)

---

### Improvement 2.1: Gentle Shadow Recovery (Optional)

**Objective**: Reduce shadow clipping in Aerial ONLY, without affecting other images

**Problem Analysis**:
```
Current v1.0.0 shadow clipping:
  Aerial:      12.73%  (could be improved)
  Pool:         8.64%  (ACCEPTABLE - reviewer gave ★★★★★)
  Other:        3-6%   (excellent)
```

**Key Insight**: Pool doesn't need fixing! Reviewer loved it. Only Aerial could benefit.

**Proposed Implementation**:

```python
# Stage 5: Selective Shadow Recovery (NEW - AFTER color grading)

def _stage_5_selective_shadow_recovery(
    self, 
    image: np.ndarray, 
    depth_map: Optional[np.ndarray],
    scene_type: str
) -> np.ndarray:
    """
    Gentle shadow recovery for high-DR outdoor scenes.
    Applied AFTER color grading to preserve LUT calibration.
    """
    
    # Only apply to Aerial views (or other high-clipping scenes)
    if scene_type != "aerial_outdoor":
        logger.info("Scene type not aerial_outdoor, skipping shadow recovery")
        return image
    
    # Check if shadows actually clipped
    shadow_clipping = self._measure_shadow_clipping(image)
    if shadow_clipping < 0.10:  # 10% threshold
        logger.info(f"Shadow clipping {shadow_clipping:.1%} acceptable, skipping recovery")
        return image
    
    # Gentle lift: ONLY pixels below 0.15 luminance (deep shadows)
    luminance = rgb_to_luminance(image)
    shadow_mask = np.clip((0.15 - luminance) / 0.15, 0, 1)  # 0.0-0.15 range
    
    # Power curve for smooth transition
    shadow_mask = shadow_mask ** 1.5
    
    # Lift by 0.2 stops max (subtle)
    lift_amount = 2 ** (0.2 / 2.2)  # ~1.07 multiplier
    recovered = image * (1 + shadow_mask * (lift_amount - 1))
    
    # Preserve highlights (above 0.7 luminance untouched)
    highlight_mask = np.clip((luminance - 0.7) / 0.3, 0, 1)
    result = np.where(highlight_mask[..., None] > 0, image, recovered)
    
    logger.info(f"Shadow recovery: {shadow_clipping:.1%} → {self._measure_shadow_clipping(result):.1%}")
    return result
```

**Pipeline Order**:
```python
# CORRECT ORDER (v1.2.0):
Stage 1: Load HDR
Stage 2: Material Response
Stage 3: Tone Mapping (preserve v1.0.0 exactly)
Stage 4: Color Grading (preserve v1.0.0 exactly)
Stage 5: Selective Shadow Recovery (NEW - gentle, targeted)
Stage 6: AI Enhancement
Stage 7: Upscaling
```

**Configuration**:
```yaml
# config/750_picacho_master_preset.yaml

shadow_recovery:
  enabled: true
  apply_to_scenes: ["aerial_outdoor"]  # Whitelist only
  luminance_threshold: 0.15  # Only deep shadows
  max_lift_stops: 0.2  # Subtle (not aggressive 0.3-0.4)
  clipping_trigger: 0.10  # Only if >10% clipping

room_overrides:
  aerial:
    shadow_recovery:
      enabled: true  # Aerial only
  pool:
    shadow_recovery:
      enabled: false  # Pool is ★★★★★ already!
```

**Validation**:
1. Process Aerial with/without shadow recovery
2. Visual comparison: Does it improve depth WITHOUT yellow cast?
3. Expert review: Rate 1-5 stars
4. If not ★★★★★ or better: DISABLE feature

**Timeline**: Week 2  
**Risk**: Medium  
**Reversibility**: ✅ Feature flag `shadow_recovery.enabled: false`

---

### Improvement 2.2: Perceptual Quality Metrics (High Priority)

**Objective**: Replace PSNR/SSIM with perceptual metrics

**New Metrics Suite**:

```python
# tools/perceptual_quality_metrics.py (NEW)

class PerceptualQualityMetrics:
    """Perceptual quality metrics for artistic rendering."""
    
    def calculate_all(self, image: np.ndarray, reference: np.ndarray = None):
        return {
            # Color fidelity
            "white_balance_error": self._white_balance_error(image),
            "color_cast_severity": self._detect_color_cast(image),
            "saturation_score": self._saturation_score(image),
            
            # Tonal quality
            "dynamic_range_score": self._dynamic_range(image),
            "highlight_preservation": self._highlight_preservation(image),
            "shadow_detail_score": self._shadow_detail(image),
            
            # Microcontrast
            "local_contrast_score": self._local_contrast(image),
            "texture_clarity_score": self._texture_clarity(image),
            
            # Artistic quality
            "visual_impact_score": self._visual_impact(image),
            "tonal_separation_score": self._tonal_separation(image),
            
            # Comparison (if reference provided)
            "perceptual_delta": self._perceptual_delta(image, reference) if reference else None
        }
    
    def _white_balance_error(self, image: np.ndarray) -> float:
        """
        Detect color cast in neutral regions.
        Returns 0.0-1.0 (0 = perfect neutral, 1 = severe cast)
        """
        # Sample pixels that should be neutral (high luminance, low saturation)
        luminance = rgb_to_luminance(image)
        saturation = rgb_to_saturation(image)
        
        neutral_mask = (luminance > 0.6) & (saturation < 0.2)
        neutral_pixels = image[neutral_mask]
        
        if len(neutral_pixels) < 100:
            return 0.0  # Not enough neutral areas
        
        # Check RGB ratio deviation from 1:1:1
        mean_rgb = neutral_pixels.mean(axis=0)
        ratio = mean_rgb / mean_rgb.mean()
        
        # Perfect white: [1.0, 1.0, 1.0]
        # Yellow cast:   [1.1, 1.05, 0.95] (more red/green, less blue)
        error = np.abs(ratio - 1.0).max()
        
        return float(error)
    
    def _visual_impact_score(self, image: np.ndarray) -> float:
        """
        Composite score: dynamic range + color purity + microcontrast.
        Returns 0.0-1.0 (higher is better)
        """
        dr_score = self._dynamic_range(image)
        color_score = 1.0 - self._detect_color_cast(image)
        contrast_score = self._local_contrast(image)
        
        # Weighted combination
        return 0.4 * dr_score + 0.3 * color_score + 0.3 * contrast_score
```

**Integration**:
```python
# After processing, compare with reference
metrics = PerceptualQualityMetrics()
result_metrics = metrics.calculate_all(output_image, reference=v1_0_0_image)

logger.info("Perceptual Quality Report:")
logger.info(f"  White Balance Error:    {result_metrics['white_balance_error']:.3f} (target <0.05)")
logger.info(f"  Dynamic Range Score:    {result_metrics['dynamic_range_score']:.3f} (target >0.8)")
logger.info(f"  Visual Impact Score:    {result_metrics['visual_impact_score']:.3f} (target >0.85)")

if reference:
    logger.info(f"  Perceptual Delta vs Ref: {result_metrics['perceptual_delta']:.3f} (target <0.1)")
```

**Success Criteria**:
```
Target Scores (based on v1.0.0 baseline):
  White Balance Error:     <0.05  (v1.1.0 failed: ~0.15 yellow cast)
  Dynamic Range Score:     >0.80  (v1.1.0 failed: ~0.65 compression)
  Local Contrast Score:    >0.75  (v1.1.0 failed: ~0.60 mushiness)
  Visual Impact Score:     >0.85  (v1.1.0 failed: ~0.65 flatness)
```

**Timeline**: Week 2-3  
**Priority**: High (prevents future v1.1.0 disasters)

---

### Improvement 2.3: A/B Visual Comparison Workflow

**Objective**: Mandatory human validation before approval

**Implementation**:

```python
# tools/ab_comparison.py (NEW)

class ABComparisonWorkflow:
    """Generate side-by-side comparisons for human review."""
    
    def create_comparison(
        self, 
        version_a: Path, 
        version_b: Path,
        output_path: Path,
        labels: Tuple[str, str] = ("v1.0.0", "v1.2.0")
    ):
        """Create side-by-side comparison image."""
        
        img_a = load_image(version_a)
        img_b = load_image(version_b)
        
        # Resize to reasonable viewing size
        max_width = 2048
        img_a = resize_to_width(img_a, max_width)
        img_b = resize_to_width(img_b, max_width)
        
        # Create side-by-side canvas
        canvas = create_side_by_side(img_a, img_b, labels)
        
        # Add difference heatmap below
        diff_heatmap = self._create_difference_heatmap(img_a, img_b)
        canvas_with_diff = stack_vertical([canvas, diff_heatmap])
        
        save_image(canvas_with_diff, output_path)
        
        # Also create interactive HTML viewer
        self._create_html_viewer(version_a, version_b, output_path.with_suffix('.html'))
    
    def create_batch_comparison(self, versions: Dict[str, Path]) -> Path:
        """Create comprehensive comparison report."""
        # Generate all pairwise comparisons
        # Create HTML gallery with metrics
        # Include reviewer checklist
        pass
```

**Workflow**:

```bash
# After any pipeline change:
python tools/ab_comparison.py \
  --reference output_750_picacho_v1.0.0/ \
  --candidate output_750_picacho_v1.2.0_test/ \
  --output comparison_reports/v1.2.0_validation/

# Generates:
# - comparison_reports/v1.2.0_validation/Pool_comparison.jpg
# - comparison_reports/v1.2.0_validation/Aerial_comparison.jpg
# - comparison_reports/v1.2.0_validation/index.html (interactive gallery)
# - comparison_reports/v1.2.0_validation/metrics_report.json
```

**HTML Viewer Features**:
- Side-by-side images with synchronized zoom/pan
- Slider to blend between versions
- Difference heatmap overlay
- Metrics table with green/red indicators
- Expert review checklist (5-star ratings per category)

**Approval Process**:
1. Generate comparison report
2. Expert reviewer fills out checklist
3. MUST achieve ≥4 stars in ALL categories
4. If any category <4 stars: REJECT change, investigate
5. Only approved changes merge to main

**Timeline**: Week 3

---

## Phase 3: Enhanced Testing (Week 4)

### Test 3.1: Regression Test Suite

**Objective**: Automated detection of perceptual regressions

```python
# tests/test_perceptual_regression.py (NEW)

class TestPerceptualRegression:
    """Prevent v1.1.0-style quality degradation."""
    
    REFERENCE_IMAGES = {
        "pool": "test_artifacts/750_picacho_v1.0.0_gold/Pool_master.tif",
        "aerial": "test_artifacts/750_picacho_v1.0.0_gold/Aerial_master.tif",
    }
    
    QUALITY_THRESHOLDS = {
        "white_balance_error": 0.05,  # Max acceptable cast
        "dynamic_range_score": 0.80,  # Min dynamic range
        "local_contrast_score": 0.75, # Min microcontrast
        "visual_impact_score": 0.85,  # Min overall quality
    }
    
    def test_pool_no_yellow_cast(self):
        """Prevent yellow tint regression (v1.1.0 issue)."""
        output = self.process_with_current_pipeline("pool")
        metrics = PerceptualQualityMetrics().calculate_all(output)
        
        assert metrics["white_balance_error"] < 0.05, \
            f"Yellow cast detected: {metrics['white_balance_error']:.3f}"
        
        assert metrics["color_cast_severity"] < 0.10, \
            f"Color cast too severe: {metrics['color_cast_severity']:.3f}"
    
    def test_pool_dynamic_range_maintained(self):
        """Prevent tone compression regression (v1.1.0 issue)."""
        reference = load_image(self.REFERENCE_IMAGES["pool"])
        output = self.process_with_current_pipeline("pool")
        
        ref_metrics = PerceptualQualityMetrics().calculate_all(reference)
        out_metrics = PerceptualQualityMetrics().calculate_all(output)
        
        # Dynamic range must not decrease by >10%
        delta = out_metrics["dynamic_range_score"] - ref_metrics["dynamic_range_score"]
        assert delta > -0.10, \
            f"Dynamic range degraded: {ref_metrics['dynamic_range_score']:.3f} → {out_metrics['dynamic_range_score']:.3f}"
    
    def test_pool_highlight_preservation(self):
        """Prevent highlight compression regression (v1.1.0 issue)."""
        output = self.process_with_current_pipeline("pool")
        metrics = PerceptualQualityMetrics().calculate_all(output)
        
        assert metrics["highlight_preservation"] > 0.85, \
            f"Highlights too muted: {metrics['highlight_preservation']:.3f}"
    
    def test_pool_microcontrast_maintained(self):
        """Prevent texture softening regression (v1.1.0 issue)."""
        reference = load_image(self.REFERENCE_IMAGES["pool"])
        output = self.process_with_current_pipeline("pool")
        
        ref_metrics = PerceptualQualityMetrics().calculate_all(reference)
        out_metrics = PerceptualQualityMetrics().calculate_all(output)
        
        delta = out_metrics["local_contrast_score"] - ref_metrics["local_contrast_score"]
        assert delta > -0.10, \
            f"Microcontrast degraded: {ref_metrics['local_contrast_score']:.3f} → {out_metrics['local_contrast_score']:.3f}"
```

**CI Integration**:
```yaml
# .github/workflows/quality_regression_tests.yml

name: Perceptual Quality Regression Tests

on: [push, pull_request]

jobs:
  regression-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Download reference images
        run: |
          # Download v1.0.0 gold standard outputs
          aws s3 cp s3://transformation-portal/test-artifacts/v1.0.0-gold/ test_artifacts/ --recursive
      
      - name: Run regression tests
        run: pytest tests/test_perceptual_regression.py -v
      
      - name: Upload comparison reports
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: regression-comparison
          path: comparison_reports/
```

**Timeline**: Week 4

---

### Test 3.2: Expert Review Integration

**Objective**: Formalize expert review as part of release process

**Process**:

1. **Automated Pre-Review**:
   ```bash
   python tools/prepare_expert_review.py \
     --candidate output_750_picacho_v1.2.0/ \
     --reference output_750_picacho_v1.0.0/ \
     --output expert_review_package/
   ```
   
   Generates:
   - Side-by-side comparison images
   - Metrics comparison table
   - Reviewer checklist (web form)

2. **Expert Review Checklist**:
   ```
   Image: 750 Picacho Pool
   
   Rate 1-5 stars (5 = excellent, 3 = acceptable, <3 = reject):
   
   [ ] Lighting & Tonal Balance:    ★★★★★
       - Balanced warm/cool tones
       - Highlight preservation
       - Shadow detail
   
   [ ] Color Accuracy & Depth:      ★★★★★
       - Neutral whites
       - Vivid pool blues
       - No color cast
   
   [ ] Microcontrast & Sharpness:   ★★★★★
       - Crisp textures
       - Material definition
       - Tactile realism
   
   [ ] Atmosphere & Visual Impact:  ★★★★★
       - Cinematic quality
       - Architectural depth
       - Professional polish
   
   Overall Recommendation:
   [ ] Approve for production (all ≥4 stars)
   [ ] Minor revisions needed (any 3 stars)
   [ ] Reject and rework (any <3 stars)
   
   Comments: ________________________________
   ```

3. **Approval Workflow**:
   ```python
   # Only merge to main if:
   if expert_review.all_categories >= 4:
       approve_release("v1.2.0")
   elif expert_review.any_category < 3:
       reject_release("v1.2.0", reason=expert_review.comments)
   else:
       request_revisions("v1.2.0", issues=expert_review.get_3_star_categories())
   ```

**Timeline**: Ongoing (part of every release)

---

## Phase 4: Documentation & Training (Week 4-5)

### Doc 4.1: Pipeline Design Principles

**Objective**: Codify lessons learned into design guidelines

```markdown
# Pipeline Design Principles (NEW)

## 1. Pipeline Order is Sacred

CORRECT ORDER:
1. HDR Precision Load
2. Material Response (surface-aware enhancement)
3. Base Tone Mapping (set tonal foundation)
4. Color Grading (LUTs calibrated for tone-mapped input)
5. Selective Adjustments (gentle, targeted only)
6. AI Enhancement (refinement, not correction)
7. Upscaling

NEVER:
- Apply shadow boost BEFORE color grading (breaks LUT calibration)
- Apply aggressive tone mapping (compresses dynamic range)
- Change multiple stages simultaneously (impossible to debug)

## 2. Metrics Are Guides, Not Goals

WRONG:
- "Reduce shadow clipping from 8% to <5%" → Optimize metric
- Result: Degraded perceptual quality

RIGHT:
- "Improve shadow detail while maintaining overall quality" → Optimize perception
- Validate: Expert review must confirm improvement

## 3. If It Ain't Broke, Don't Fix It

v1.0.0 Pool: ★★★★★ with 8.64% shadow clipping
→ ACCEPTABLE, no fix needed

v1.0.0 Aerial: ★★★★★ with 12.73% shadow clipping
→ Could improve, but gently and separately

## 4. Visual Validation is Mandatory

NEVER approve based on:
- Automated metrics alone (PSNR, SSIM)
- "No change" in metrics
- Internal team review only

ALWAYS require:
- Side-by-side comparison with reference
- Expert review with 5-star checklist
- All categories ≥4 stars for approval

## 5. One Change at a Time

WRONG (v1.1.0):
- Added 5 features simultaneously
- Impossible to isolate cause of degradation

RIGHT (v1.2.0):
- Feature branches for each improvement
- Individual validation before merge
- Reversible via feature flags

## 6. Perceptual Quality First

Metrics to prioritize:
1. White balance (neutral whites)
2. Dynamic range (highlight preservation)
3. Microcontrast (texture clarity)
4. Visual impact (overall "wow" factor)

NOT:
1. PSNR (compression artifacts, not perception)
2. SSIM (structure, not color/tone)
3. Clipping percentage (some clipping is artistic)
```

**Timeline**: Week 4

---

### Doc 4.2: Quality Assurance Playbook

**Objective**: Step-by-step QA process for all future changes

```markdown
# Quality Assurance Playbook (NEW)

## Pre-Development Checklist

[ ] Define success criteria (perceptual, not just metrics)
[ ] Identify reference images (v1.0.0 gold standard)
[ ] Plan A/B validation workflow
[ ] Ensure expert reviewer availability

## Development Checklist

[ ] Feature implemented on isolated branch
[ ] Unit tests passing
[ ] Perceptual metrics evaluated
[ ] Comparison images generated

## Pre-Release Checklist

[ ] Regression tests passing (automated)
[ ] A/B comparison created (visual)
[ ] Expert review completed (5-star checklist)
[ ] All categories ≥4 stars
[ ] Documentation updated
[ ] Feature flag available (if major change)

## Release Checklist

[ ] Final visual validation on production data
[ ] Client notification (if applicable)
[ ] Rollback plan documented
[ ] Monitoring enabled for quality metrics

## Post-Release Monitoring

[ ] Track perceptual metrics over time
[ ] Collect client feedback
[ ] Review after 30 days
[ ] Document any issues for next version
```

**Timeline**: Week 4

---

## Phase 5: v1.2.0 Release (Week 5-6)

### Release Criteria

**Must Have**:
- ✅ v1.0.0 quality fully restored (expert verified)
- ✅ Perceptual quality metrics implemented
- ✅ Visual QA framework active
- ✅ Regression tests passing
- ✅ Documentation complete

**Nice to Have** (defer to v1.3.0 if not ready):
- ⏳ Gentle shadow recovery (Aerial only)
- ⏳ A/B comparison workflow
- ⏳ Expert review automation

### Release Validation

**Test Suite**:
```bash
# 1. Automated tests
pytest tests/test_perceptual_regression.py -v

# 2. Process all 6 test images
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/*.tif \
  --preset 750_picacho \
  --output-dir output_750_picacho_v1.2.0_final

# 3. Generate comparison report
python tools/ab_comparison.py \
  --reference output_750_picacho_v1.0.0/ \
  --candidate output_750_picacho_v1.2.0_final/ \
  --output comparison_reports/v1.2.0_release/

# 4. Expert review
open comparison_reports/v1.2.0_release/index.html
# → Fill out 5-star checklist for all images
```

**Approval Requirements**:
- [ ] All perceptual regression tests pass
- [ ] Expert review: ALL images ≥4 stars in ALL categories
- [ ] No quality regressions vs v1.0.0
- [ ] Documentation complete and accurate
- [ ] Rollback plan tested

### Release Communication

```markdown
# v1.2.0 Release Notes

## Quality Restored

v1.2.0 restores the excellent ★★★★★ quality of v1.0.0 while adding 
robust quality assurance measures to prevent future regressions.

## What Changed

### Removed (v1.1.0 issues fixed):
- ❌ Aggressive shadow boost (caused yellow cast)
- ❌ Zone-based tone mapping (compressed dynamic range)
- ❌ Pre-color-grading adjustments (broke LUT calibration)

### Added:
- ✅ Perceptual quality metrics (white balance, dynamic range, microcontrast)
- ✅ Visual QA framework (automated detection of color casts, tone compression)
- ✅ Regression test suite (prevents v1.1.0-style degradation)
- ✅ A/B comparison workflow (mandatory expert review)

### Maintained:
- ✅ v1.0.0 processing pipeline (★★★★★ quality)
- ✅ v1.0.0 performance (13-14s per image)
- ✅ All existing features and presets

## For Users

No action required. v1.2.0 produces identical output to v1.0.0 
(which you've been using successfully).

## For Developers

New quality assurance measures are now mandatory:
1. Run perceptual regression tests before merge
2. Generate A/B comparison for all pipeline changes
3. Obtain expert review approval (≥4 stars all categories)

See QUALITY_ASSURANCE_PLAYBOOK.md for details.
```

**Timeline**: Week 6

---

## Long-Term Improvements (v1.3.0+)

### Consideration List (NOT for v1.2.0)

1. **Selective Shadow Recovery**:
   - Target: Aerial only
   - Method: Gentle lift after color grading
   - Requirement: Expert review confirms improvement

2. **Material-Specific Adjustments**:
   - Per-material sharpness tuning
   - Surface-aware saturation control
   - Requirement: Isolated testing per material

3. **Scene-Adaptive Processing**:
   - Smarter scene detection (ML-based)
   - Custom profiles per room type
   - Requirement: Training data from expert-rated images

4. **HDR10+ Metadata**:
   - Preserve HDR10+ dynamic metadata
   - Optional HDR output format
   - Requirement: Client demand verification

**Principle**: Each improvement must demonstrate clear value without risk to existing quality.

---

## Risk Management

### Risk 1: v1.2.0 Fails Expert Review

**Mitigation**:
- Revert to exact v1.0.0 code (git revert)
- Use v1.0.0 for all production work
- Defer quality improvements to v1.3.0

**Impact**: Low (v1.0.0 is production-ready)

### Risk 2: Perceptual Metrics Unreliable

**Mitigation**:
- Calibrate metrics against expert ratings
- Adjust thresholds based on validation data
- Keep expert review as final authority

**Impact**: Medium (requires recalibration)

### Risk 3: Team Resistance to QA Process

**Mitigation**:
- Document v1.1.0 failure clearly (lessons learned)
- Show cost of quality issues (client relationships)
- Automate as much as possible (reduce manual burden)

**Impact**: Medium (process adoption)

---

## Success Metrics

### Technical Success
- [ ] All perceptual regression tests pass
- [ ] White balance error <0.05
- [ ] Dynamic range score >0.80
- [ ] Visual impact score >0.85
- [ ] Expert review ★★★★★ all categories

### Process Success
- [ ] QA playbook adopted by team
- [ ] Expert review integrated into release process
- [ ] No quality regressions in next 3 releases
- [ ] Client satisfaction maintained/improved

### Business Success
- [ ] v1.0.0 quality maintained
- [ ] Team confidence restored
- [ ] Pipeline improvements safely deployed
- [ ] Competitive advantage from quality assurance

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Immediate Remediation | v1.0.0 restored, visual QA framework |
| 2-3 | Targeted Improvements | Perceptual metrics, A/B workflow |
| 4 | Enhanced Testing | Regression tests, expert review process |
| 4-5 | Documentation | Design principles, QA playbook |
| 5-6 | Release | v1.2.0 validated and deployed |

**Total Duration**: 6 weeks  
**Critical Path**: Expert review availability

---

## Conclusion

v1.2.0 will succeed by:

1. **Restoring v1.0.0 quality** (proven ★★★★★)
2. **Adding QA guardrails** (prevent v1.1.0 disasters)
3. **Validating perception** (expert review mandatory)
4. **Documenting principles** (institutional knowledge)
5. **Enabling safe improvements** (feature flags, regression tests)

**Philosophy**: We don't need v1.1.0's "improvements." We need v1.0.0's quality with better validation to enable FUTURE improvements safely.

---

**Plan Status**: DRAFT - Awaiting approval  
**Next Steps**: 
1. Team review of this plan
2. Approve remediation approach
3. Begin Phase 1 (immediate revert to v1.0.0)

**Plan Author**: Transformation Portal Quality Team  
**Date**: November 10, 2025
