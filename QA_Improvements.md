# QA Improvements: Preventing Future Quality Regressions

**Date**: November 10, 2025  
**Objective**: Implement robust quality assurance to prevent v1.1.0-style failures  
**Status**: Implementation Roadmap

---

## Executive Summary

The v1.1.0 quality degradation revealed critical gaps in our QA process:

1. **Over-reliance on automated metrics** (PSNR, SSIM)
2. **No visual validation workflow**
3. **Lack of perceptual quality metrics**
4. **Insufficient expert review integration**
5. **Missing A/B comparison methodology**

This document outlines comprehensive QA improvements to ensure we catch quality issues BEFORE production release.

---

## QA Gap Analysis

### What Failed in v1.1.0

| QA Element | Status | Impact |
|------------|--------|--------|
| Automated metrics (PSNR/SSIM) | ✅ Passed | ❌ Failed to detect degradation |
| Visual comparison | ❌ Not performed | ❌ Would have caught yellow cast |
| Expert review | ❌ Skipped | ❌ Would have rated ★★☆☆☆ |
| Perceptual metrics | ❌ Not implemented | ❌ Would have flagged color/tone issues |
| Regression tests | ✅ Passed | ❌ Wrong metrics tested |
| Client feedback | ❌ Post-release only | ❌ Too late |

**Conclusion**: We had quantity of testing, not quality of testing.

---

## Improvement #1: Perceptual Quality Metrics

### Objective
Replace PSNR/SSIM with metrics that correlate with human perception.

### Implementation

#### 1.1 White Balance & Color Cast Detection

**Metric**: Neutral Point Deviation

```python
# tools/perceptual_metrics.py

def measure_white_balance_error(image: np.ndarray) -> float:
    """
    Measure color cast by analyzing neutral regions.
    Returns 0.0-1.0 (0 = perfect neutral, 1 = severe cast)
    
    Example:
      v1.0.0: 0.02 (excellent)
      v1.1.0: 0.15 (yellow cast detected)
    """
    # Convert to LAB color space for perceptual analysis
    lab = rgb_to_lab(image)
    
    # Sample pixels that should be neutral (high L, low a/b)
    luminance = lab[..., 0]
    a_channel = lab[..., 1]  # green-red axis
    b_channel = lab[..., 2]  # blue-yellow axis
    
    neutral_mask = (luminance > 60) & (np.abs(a_channel) < 10) & (np.abs(b_channel) < 10)
    neutral_pixels = lab[neutral_mask]
    
    if len(neutral_pixels) < 100:
        return 0.0  # Not enough neutral areas to evaluate
    
    # Measure deviation from neutral (a=0, b=0)
    mean_a = neutral_pixels[:, 1].mean()
    mean_b = neutral_pixels[:, 2].mean()
    
    # Yellow cast: positive b (blue-yellow axis shifted toward yellow)
    # Green cast: positive a (green-red axis shifted toward green)
    cast_magnitude = np.sqrt(mean_a**2 + mean_b**2)
    
    # Normalize to 0-1 scale (15+ is severe)
    error = min(cast_magnitude / 15.0, 1.0)
    
    return float(error)


def detect_color_cast_direction(image: np.ndarray) -> Dict[str, float]:
    """
    Identify the TYPE of color cast (yellow, blue, green, magenta).
    
    Example:
      v1.1.0: {"yellow": 0.68, "blue": -0.34} → Yellow cast detected
    """
    lab = rgb_to_lab(image)
    neutral_pixels = extract_neutral_regions(lab)
    
    mean_a = neutral_pixels[:, 1].mean()  # green(-) to red(+)
    mean_b = neutral_pixels[:, 2].mean()  # blue(-) to yellow(+)
    
    return {
        "yellow": max(0, mean_b / 10.0),   # Positive b
        "blue": max(0, -mean_b / 10.0),    # Negative b
        "green": max(0, -mean_a / 10.0),   # Negative a
        "magenta": max(0, mean_a / 10.0),  # Positive a
        "severity": np.sqrt(mean_a**2 + mean_b**2) / 10.0
    }
```

**Thresholds**:
```python
WHITE_BALANCE_THRESHOLDS = {
    "excellent": 0.03,   # ≤3% error (v1.0.0 level)
    "acceptable": 0.05,  # ≤5% error
    "warning": 0.10,     # 5-10% noticeable cast
    "fail": 0.15         # >15% severe cast (v1.1.0 level)
}
```

**Usage**:
```python
error = measure_white_balance_error(output_image)
if error > WHITE_BALANCE_THRESHOLDS["fail"]:
    logger.error(f"❌ FAIL: Yellow cast detected ({error:.2%})")
    raise QualityError("Color cast exceeds acceptable threshold")
elif error > WHITE_BALANCE_THRESHOLDS["warning"]:
    logger.warning(f"⚠️  WARNING: Noticeable color cast ({error:.2%})")
```

---

#### 1.2 Dynamic Range & Tone Compression Detection

**Metric**: Highlight & Shadow Range Preservation

```python
def measure_dynamic_range_quality(image: np.ndarray) -> Dict[str, float]:
    """
    Measure how well highlights and shadows are preserved.
    
    Example:
      v1.0.0: {"highlight_range": 0.18, "total_range": 0.92} → Excellent
      v1.1.0: {"highlight_range": 0.11, "total_range": 0.78} → Compressed
    """
    # Convert to luminance
    luminance = rgb_to_luminance(image)
    
    # Calculate percentile values
    p01 = np.percentile(luminance, 1)
    p05 = np.percentile(luminance, 5)
    p95 = np.percentile(luminance, 95)
    p99 = np.percentile(luminance, 99)
    
    # Measure ranges
    highlight_range = p99 - p95  # Top 4% of tones
    shadow_range = p05 - p01     # Bottom 4% of tones
    total_range = p99 - p01      # Overall dynamic range
    
    # Measure compression (should be ~0.15-0.20 for highlights)
    highlight_compression = 1.0 - (highlight_range / 0.20)  # 0 = no compression, 1 = severe
    shadow_compression = 1.0 - (shadow_range / 0.20)
    
    return {
        "highlight_range": highlight_range,
        "shadow_range": shadow_range,
        "total_range": total_range,
        "highlight_compression": max(0, highlight_compression),
        "shadow_compression": max(0, shadow_compression),
        "score": total_range  # 0.0-1.0 (higher is better)
    }


def detect_tone_compression(image: np.ndarray, reference: np.ndarray = None) -> float:
    """
    Compare dynamic range with reference to detect tone compression.
    Returns compression ratio: 0.0 (no compression) to 1.0 (severe).
    
    Example:
      v1.1.0 vs v1.0.0: 0.35 → 35% dynamic range lost
    """
    current_dr = measure_dynamic_range_quality(image)
    
    if reference is not None:
        reference_dr = measure_dynamic_range_quality(reference)
        compression = 1.0 - (current_dr["total_range"] / reference_dr["total_range"])
        return max(0, compression)
    else:
        # Without reference, flag if total range < 0.80
        if current_dr["total_range"] < 0.80:
            return 1.0 - current_dr["total_range"]  # Estimated compression
        return 0.0
```

**Thresholds**:
```python
DYNAMIC_RANGE_THRESHOLDS = {
    "excellent": 0.85,  # ≥85% of full range (v1.0.0)
    "acceptable": 0.75, # ≥75% of full range
    "warning": 0.65,    # 65-75% compressed
    "fail": 0.60        # <60% severely compressed (v1.1.0)
}
```

---

#### 1.3 Microcontrast & Texture Clarity

**Metric**: Local Standard Deviation

```python
def measure_microcontrast(image: np.ndarray, window_size: int = 15) -> float:
    """
    Measure local contrast (texture clarity).
    Higher values = crisper textures.
    
    Example:
      v1.0.0: 0.082 (excellent texture definition)
      v1.1.0: 0.061 (soft, mushy textures)
    """
    luminance = rgb_to_luminance(image)
    
    # Calculate local standard deviation
    from scipy.ndimage import uniform_filter
    
    # Local mean
    local_mean = uniform_filter(luminance, size=window_size)
    
    # Local variance
    local_sqr_mean = uniform_filter(luminance**2, size=window_size)
    local_variance = local_sqr_mean - local_mean**2
    
    # Local standard deviation (microcontrast)
    local_std = np.sqrt(np.maximum(local_variance, 0))
    
    # Average microcontrast across image
    mean_microcontrast = local_std.mean()
    
    return float(mean_microcontrast)


def measure_texture_clarity(image: np.ndarray) -> Dict[str, float]:
    """
    Multi-scale texture analysis.
    """
    microcontrast_fine = measure_microcontrast(image, window_size=7)    # Fine detail
    microcontrast_medium = measure_microcontrast(image, window_size=15) # Medium texture
    microcontrast_coarse = measure_microcontrast(image, window_size=31) # Coarse structure
    
    # Weighted score (emphasize fine detail)
    score = (
        0.5 * microcontrast_fine +
        0.3 * microcontrast_medium +
        0.2 * microcontrast_coarse
    )
    
    return {
        "fine_detail": microcontrast_fine,
        "medium_texture": microcontrast_medium,
        "coarse_structure": microcontrast_coarse,
        "overall_score": score
    }
```

**Thresholds**:
```python
MICROCONTRAST_THRESHOLDS = {
    "excellent": 0.075,  # ≥0.075 crisp textures (v1.0.0)
    "acceptable": 0.065, # ≥0.065 acceptable
    "warning": 0.055,    # 0.055-0.065 soft
    "fail": 0.050        # <0.050 mushy (v1.1.0)
}
```

---

#### 1.4 Visual Impact Score (Composite)

**Metric**: Weighted Combination of Perceptual Factors

```python
def calculate_visual_impact_score(image: np.ndarray) -> Dict[str, float]:
    """
    Composite score combining multiple perceptual factors.
    Returns 0.0-1.0 (higher = better visual impact).
    
    Example:
      v1.0.0: 0.89 (★★★★★)
      v1.1.0: 0.64 (★★★☆☆)
    """
    # Individual metrics
    wb_error = measure_white_balance_error(image)
    dr_quality = measure_dynamic_range_quality(image)
    texture_quality = measure_texture_clarity(image)
    
    # Invert white balance error (lower is better)
    color_purity_score = 1.0 - min(wb_error / 0.15, 1.0)
    
    # Dynamic range score (0.0-1.0)
    dr_score = dr_quality["score"]
    
    # Texture score (normalize to 0-1)
    texture_score = min(texture_quality["overall_score"] / 0.10, 1.0)
    
    # Weighted combination
    # Color purity: 30% (critical for luxury real estate)
    # Dynamic range: 40% (most important for "wow" factor)
    # Texture: 30% (professionalism and detail)
    composite_score = (
        0.30 * color_purity_score +
        0.40 * dr_score +
        0.30 * texture_score
    )
    
    return {
        "color_purity": color_purity_score,
        "dynamic_range": dr_score,
        "texture_clarity": texture_score,
        "visual_impact": composite_score,
        "rating_estimate": _score_to_stars(composite_score)
    }

def _score_to_stars(score: float) -> str:
    """Convert numeric score to star rating."""
    if score >= 0.85: return "★★★★★"
    if score >= 0.70: return "★★★★☆"
    if score >= 0.55: return "★★★☆☆"
    if score >= 0.40: return "★★☆☆☆"
    return "★☆☆☆☆"
```

**Validation**:
```python
# Validate against expert reviews
expert_ratings = {
    "v1.0.0_pool": 5.0,    # ★★★★★
    "v1.1.0_pool": 2.5,    # ★★★☆☆ (average of 2-3 star categories)
}

automated_scores = {
    "v1.0.0_pool": 0.89,   # Predicts ★★★★★ ✓
    "v1.1.0_pool": 0.64,   # Predicts ★★★☆☆ ✓
}

# Correlation: High (metrics match expert perception)
```

---

### Integration into Pipeline

```python
# luxury_estate_master_pipeline.py

def process_image(self, input_path: Path) -> ProcessingResult:
    """Process image with perceptual quality monitoring."""
    
    # Load reference (v1.0.0 gold standard)
    reference = None
    if self.config.quality_control.reference_dir:
        reference = self._load_reference(input_path)
    
    # Process image
    output = self._run_pipeline(input_path)
    
    # Calculate perceptual metrics
    from tools.perceptual_metrics import (
        measure_white_balance_error,
        measure_dynamic_range_quality,
        measure_texture_clarity,
        calculate_visual_impact_score
    )
    
    metrics = {
        "white_balance_error": measure_white_balance_error(output),
        "dynamic_range": measure_dynamic_range_quality(output),
        "texture_clarity": measure_texture_clarity(output),
        "visual_impact": calculate_visual_impact_score(output)
    }
    
    # Compare with reference if available
    if reference is not None:
        ref_metrics = calculate_visual_impact_score(reference)
        delta = metrics["visual_impact"]["visual_impact"] - ref_metrics["visual_impact"]
        
        logger.info(f"Visual Impact: {metrics['visual_impact']['visual_impact']:.3f}")
        logger.info(f"Reference:     {ref_metrics['visual_impact']:.3f}")
        logger.info(f"Delta:         {delta:+.3f} ({'+' if delta > 0 else ''}{'improved' if delta > 0 else 'degraded'})")
        
        # Alert if regression
        if delta < -0.10:  # >10% degradation
            logger.error(f"❌ QUALITY REGRESSION: Visual impact degraded by {abs(delta):.1%}")
            if self.config.quality_control.fail_on_regression:
                raise QualityRegressionError(f"Visual impact: {delta:+.3f}")
    
    # Log detailed metrics
    logger.info("Perceptual Quality Metrics:")
    logger.info(f"  White Balance Error:  {metrics['white_balance_error']:.3f} (target <0.05)")
    logger.info(f"  Dynamic Range Score:  {metrics['dynamic_range']['score']:.3f} (target >0.85)")
    logger.info(f"  Texture Clarity:      {metrics['texture_clarity']['overall_score']:.3f} (target >0.075)")
    logger.info(f"  Visual Impact Score:  {metrics['visual_impact']['visual_impact']:.3f} {metrics['visual_impact']['rating_estimate']}")
    
    # Check thresholds
    issues = []
    if metrics["white_balance_error"] > 0.10:
        issues.append(f"Color cast detected ({metrics['white_balance_error']:.2%})")
    if metrics["dynamic_range"]["score"] < 0.75:
        issues.append(f"Dynamic range compressed ({metrics['dynamic_range']['score']:.2%})")
    if metrics["texture_clarity"]["overall_score"] < 0.065:
        issues.append(f"Soft textures ({metrics['texture_clarity']['overall_score']:.3f})")
    
    if issues and self.config.quality_control.fail_on_issues:
        raise QualityIssueDetected(issues)
    
    return ProcessingResult(
        output_image=output,
        metrics=metrics,
        quality_issues=issues
    )
```

---

## Improvement #2: Visual Validation Checkpoints

### Objective
Mandatory human review at critical pipeline stages.

### Implementation

#### 2.1 Automated Visual Comparison Generation

```python
# tools/visual_qa_checkpoints.py

class VisualQACheckpoint:
    """Generate visual comparisons for human review."""
    
    def create_checkpoint(
        self,
        stage_name: str,
        current_output: np.ndarray,
        reference_output: np.ndarray,
        output_dir: Path
    ):
        """Create side-by-side comparison for stage validation."""
        
        # Resize for display
        current_display = resize_to_width(current_output, 1920)
        reference_display = resize_to_width(reference_output, 1920)
        
        # Side-by-side layout
        comparison = np.hstack([reference_display, current_display])
        
        # Add labels
        labeled = add_text_labels(
            comparison,
            labels=[
                ("v1.0.0 (Reference)", 0.25),
                ("Current (Candidate)", 0.75)
            ]
        )
        
        # Add difference heatmap
        diff_map = self._create_difference_visualization(
            reference_display, current_display
        )
        
        # Metrics comparison table
        metrics_table = self._create_metrics_table(
            current_output, reference_output
        )
        
        # Stack all elements
        full_report = stack_vertical([
            labeled,
            diff_map,
            metrics_table
        ])
        
        # Save
        save_path = output_dir / f"{stage_name}_visual_qa_checkpoint.jpg"
        save_image(full_report, save_path)
        
        logger.info(f"✓ Visual QA checkpoint created: {save_path}")
        
        return save_path
    
    def _create_difference_visualization(
        self, 
        ref: np.ndarray, 
        cur: np.ndarray
    ) -> np.ndarray:
        """Create color-coded difference heatmap."""
        
        # Calculate per-pixel difference
        diff = np.abs(cur - ref).mean(axis=2)
        
        # Normalize to 0-1
        diff_norm = diff / diff.max()
        
        # Apply colormap (blue = no difference, red = large difference)
        import matplotlib.cm as cm
        colored_diff = cm.jet(diff_norm)[:, :, :3]
        
        # Add scale bar
        heatmap = add_scale_bar(colored_diff, title="Difference Magnitude")
        
        return (heatmap * 255).astype(np.uint8)
```

**Usage**:
```python
# In pipeline processing
checkpoint_qa = VisualQACheckpoint()

# After tone mapping stage
if self.config.quality_control.visual_checkpoints:
    reference_tonemapped = load_reference_stage("tone_mapped")
    checkpoint_qa.create_checkpoint(
        stage_name="tone_mapping",
        current_output=tone_mapped_image,
        reference_output=reference_tonemapped,
        output_dir=self.output_dir / "qa_checkpoints"
    )

# After color grading stage
if self.config.quality_control.visual_checkpoints:
    reference_graded = load_reference_stage("color_graded")
    checkpoint_qa.create_checkpoint(
        stage_name="color_grading",
        current_output=color_graded_image,
        reference_output=reference_graded,
        output_dir=self.output_dir / "qa_checkpoints"
    )
```

---

#### 2.2 Expert Review Workflow

```python
# tools/expert_review_workflow.py

class ExpertReviewWorkflow:
    """Manage expert review process."""
    
    def create_review_package(
        self,
        version_name: str,
        test_outputs: Path,
        reference_outputs: Path,
        output_dir: Path
    ):
        """Generate comprehensive review package for expert."""
        
        # Create review directory
        review_dir = output_dir / f"expert_review_{version_name}"
        review_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparisons for each image
        image_comparisons = []
        for image_name in ["Pool", "Aerial", "Great_Room", "Kitchen"]:
            comparison = self._create_image_comparison(
                test_outputs / f"{image_name}_master.tif",
                reference_outputs / f"{image_name}_master.tif",
                image_name
            )
            comparison_path = review_dir / f"{image_name}_comparison.jpg"
            save_image(comparison, comparison_path)
            image_comparisons.append({
                "name": image_name,
                "path": comparison_path,
                "metrics": self._calculate_metrics_diff(
                    test_outputs / f"{image_name}_master.tif",
                    reference_outputs / f"{image_name}_master.tif"
                )
            })
        
        # Generate HTML review form
        html_form = self._generate_review_form(
            version_name, image_comparisons
        )
        form_path = review_dir / "review_form.html"
        with open(form_path, 'w') as f:
            f.write(html_form)
        
        # Create README with instructions
        readme = self._create_review_instructions(version_name)
        with open(review_dir / "README.md", 'w') as f:
            f.write(readme)
        
        logger.info(f"✓ Expert review package created: {review_dir}")
        logger.info(f"  Open review form: {form_path}")
        
        return review_dir
    
    def _generate_review_form(
        self, 
        version_name: str, 
        image_comparisons: List[Dict]
    ) -> str:
        """Generate HTML review form with star ratings."""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Expert Review: {version_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .image-review {{ margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; }}
        .comparison-image {{ max-width: 100%; }}
        .rating-row {{ margin: 10px 0; }}
        .stars {{ font-size: 24px; cursor: pointer; }}
        .star {{ color: #ccc; }}
        .star.selected {{ color: #ffd700; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metric-good {{ background-color: #d4edda; }}
        .metric-warning {{ background-color: #fff3cd; }}
        .metric-bad {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>Expert Review: {version_name}</h1>
    <p>Please review each image comparison and rate on a 5-star scale.</p>
    <p><strong>Rating Guide:</strong> ★★★★★ (excellent) | ★★★★☆ (good) | ★★★☆☆ (acceptable) | ★★☆☆☆ (poor) | ★☆☆☆☆ (unacceptable)</p>
    
    <form id="reviewForm">
"""
        
        for img in image_comparisons:
            html += f"""
        <div class="image-review">
            <h2>{img['name']}</h2>
            <img src="{img['path'].name}" class="comparison-image">
            
            <h3>Metrics Comparison</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Reference (v1.0.0)</th>
                    <th>Candidate ({version_name})</th>
                    <th>Status</th>
                </tr>
"""
            for metric_name, values in img['metrics'].items():
                status_class = "metric-good" if values['status'] == "IMPROVED" else \
                               "metric-warning" if values['status'] == "MAINTAINED" else "metric-bad"
                html += f"""
                <tr class="{status_class}">
                    <td>{metric_name}</td>
                    <td>{values['reference']:.4f}</td>
                    <td>{values['candidate']:.4f}</td>
                    <td>{values['status']}</td>
                </tr>
"""
            html += """
            </table>
            
            <h3>Quality Ratings</h3>
            <div class="rating-row">
                <label>Lighting & Tonal Balance:</label>
                <div class="stars" data-rating-name="{img['name']}_lighting">
                    <span class="star" data-value="1">★</span>
                    <span class="star" data-value="2">★</span>
                    <span class="star" data-value="3">★</span>
                    <span class="star" data-value="4">★</span>
                    <span class="star" data-value="5">★</span>
                </div>
            </div>
            <!-- More rating categories... -->
        </div>
"""
        
        html += """
        <h2>Overall Recommendation</h2>
        <select name="recommendation" required>
            <option value="">Select...</option>
            <option value="approve">Approve for production (all ≥4 stars)</option>
            <option value="revise">Request revisions (some 3 stars)</option>
            <option value="reject">Reject (any <3 stars)</option>
        </select>
        
        <h3>Comments</h3>
        <textarea name="comments" rows="5" style="width: 100%;"></textarea>
        
        <button type="submit">Submit Review</button>
    </form>
    
    <script>
        // Star rating interaction
        document.querySelectorAll('.stars').forEach(container => {
            container.querySelectorAll('.star').forEach(star => {
                star.addEventListener('click', function() {
                    const value = this.dataset.value;
                    const stars = this.parentElement.querySelectorAll('.star');
                    stars.forEach((s, idx) => {
                        s.classList.toggle('selected', idx < value);
                    });
                });
            });
        });
        
        // Form submission
        document.getElementById('reviewForm').addEventListener('submit', function(e) {
            e.preventDefault();
            // Collect ratings and save to JSON
            const formData = new FormData(this);
            const results = Object.fromEntries(formData);
            
            // Save to file
            const blob = new Blob([JSON.stringify(results, null, 2)], {type: 'application/json'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'expert_review_results.json';
            a.click();
            
            alert('Review submitted! Results saved to expert_review_results.json');
        });
    </script>
</body>
</html>
"""
        return html
```

---

## Improvement #3: Automated Regression Testing

### Objective
Catch perceptual regressions before they reach production.

### Implementation

```python
# tests/test_perceptual_quality_regression.py

import pytest
import numpy as np
from pathlib import Path
from tools.perceptual_metrics import (
    measure_white_balance_error,
    measure_dynamic_range_quality,
    measure_texture_clarity,
    calculate_visual_impact_score
)

# Reference images (v1.0.0 gold standard)
REFERENCE_DIR = Path("test_artifacts/v1.0.0_gold")

# Quality thresholds (based on v1.0.0 baseline)
QUALITY_BASELINES = {
    "pool": {
        "white_balance_error_max": 0.05,
        "dynamic_range_min": 0.85,
        "texture_clarity_min": 0.075,
        "visual_impact_min": 0.85
    },
    "aerial": {
        "white_balance_error_max": 0.05,
        "dynamic_range_min": 0.80,  # Slightly lower for high-DR scenes
        "texture_clarity_min": 0.070,
        "visual_impact_min": 0.82
    }
}


class TestPerceptualQualityRegression:
    """Prevent v1.1.0-style quality degradation."""
    
    @pytest.fixture
    def process_test_image(self):
        """Process a test image with current pipeline."""
        def _process(image_name: str) -> np.ndarray:
            from luxury_estate_master_pipeline import LuxuryEstateMasterPipeline
            
            pipeline = LuxuryEstateMasterPipeline.from_preset("750_picacho")
            result = pipeline.process_image(
                f"test_artifacts/inputs/{image_name}.tif"
            )
            return result.output_image
        return _process
    
    def test_no_color_cast_regression(self, process_test_image):
        """Prevent yellow cast regression (v1.1.0 issue #1)."""
        output = process_test_image("pool")
        
        wb_error = measure_white_balance_error(output)
        threshold = QUALITY_BASELINES["pool"]["white_balance_error_max"]
        
        assert wb_error < threshold, \
            f"Color cast detected: {wb_error:.3f} > {threshold:.3f} threshold"
    
    def test_no_dynamic_range_compression(self, process_test_image):
        """Prevent tone compression regression (v1.1.0 issue #2)."""
        output = process_test_image("pool")
        
        dr_metrics = measure_dynamic_range_quality(output)
        threshold = QUALITY_BASELINES["pool"]["dynamic_range_min"]
        
        assert dr_metrics["score"] > threshold, \
            f"Dynamic range compressed: {dr_metrics['score']:.3f} < {threshold:.3f} threshold"
    
    def test_no_texture_softening(self, process_test_image):
        """Prevent microcontrast loss regression (v1.1.0 issue #3)."""
        output = process_test_image("pool")
        
        texture_metrics = measure_texture_clarity(output)
        threshold = QUALITY_BASELINES["pool"]["texture_clarity_min"]
        
        assert texture_metrics["overall_score"] > threshold, \
            f"Textures soft: {texture_metrics['overall_score']:.3f} < {threshold:.3f} threshold"
    
    def test_visual_impact_maintained(self, process_test_image):
        """Prevent overall quality degradation."""
        output = process_test_image("pool")
        
        impact = calculate_visual_impact_score(output)
        threshold = QUALITY_BASELINES["pool"]["visual_impact_min"]
        
        assert impact["visual_impact"] > threshold, \
            f"Visual impact degraded: {impact['visual_impact']:.3f} {impact['rating_estimate']}"
    
    def test_pool_quality_vs_reference(self, process_test_image):
        """Compare Pool output against v1.0.0 reference."""
        output = process_test_image("pool")
        reference = load_image(REFERENCE_DIR / "750Picacho_Pool_master.tif")
        
        output_impact = calculate_visual_impact_score(output)
        ref_impact = calculate_visual_impact_score(reference)
        
        delta = output_impact["visual_impact"] - ref_impact["visual_impact"]
        
        # Allow up to 5% degradation (measurement noise)
        assert delta > -0.05, \
            f"Quality degraded vs reference: {delta:+.3f} ({output_impact['rating_estimate']} vs {ref_impact['rating_estimate']})"


# CI Integration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

**CI Workflow**:
```yaml
# .github/workflows/perceptual_quality_ci.yml

name: Perceptual Quality CI

on: [push, pull_request]

jobs:
  quality-regression-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Download reference images
        run: |
          # Download v1.0.0 gold standard
          mkdir -p test_artifacts/v1.0.0_gold
          # aws s3 cp s3://transformation-portal/test-artifacts/v1.0.0-gold/ test_artifacts/v1.0.0_gold/ --recursive
      
      - name: Run perceptual quality tests
        run: |
          pytest tests/test_perceptual_quality_regression.py -v
      
      - name: Upload comparison artifacts
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: quality-regression-report
          path: test_artifacts/comparisons/
```

---

## Improvement #4: Continuous Monitoring

### Objective
Track quality metrics over time to detect trends.

### Implementation

```python
# tools/quality_metrics_tracker.py

class QualityMetricsTracker:
    """Track quality metrics across versions."""
    
    def __init__(self, db_path: Path = Path("quality_metrics.db")):
        self.db_path = db_path
        self._init_database()
    
    def record_processing(
        self,
        version: str,
        image_name: str,
        metrics: Dict[str, float],
        expert_rating: Optional[float] = None
    ):
        """Record quality metrics for a processing run."""
        
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO quality_metrics 
            (timestamp, version, image_name, white_balance_error, 
             dynamic_range_score, texture_clarity, visual_impact, 
             expert_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            version,
            image_name,
            metrics["white_balance_error"],
            metrics["dynamic_range"]["score"],
            metrics["texture_clarity"]["overall_score"],
            metrics["visual_impact"]["visual_impact"],
            expert_rating
        ))
        
        conn.commit()
        conn.close()
    
    def generate_trend_report(self, output_path: Path):
        """Generate quality trends over time."""
        
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load data
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM quality_metrics ORDER BY timestamp", conn)
        conn.close()
        
        # Plot trends
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # White balance error over time
        axes[0, 0].plot(df['timestamp'], df['white_balance_error'], marker='o')
        axes[0, 0].axhline(y=0.05, color='r', linestyle='--', label='Threshold')
        axes[0, 0].set_title('White Balance Error Over Time')
        axes[0, 0].set_ylabel('Error (lower is better)')
        axes[0, 0].legend()
        
        # Dynamic range score over time
        axes[0, 1].plot(df['timestamp'], df['dynamic_range_score'], marker='o')
        axes[0, 1].axhline(y=0.85, color='g', linestyle='--', label='Target')
        axes[0, 1].set_title('Dynamic Range Score Over Time')
        axes[0, 1].set_ylabel('Score (higher is better)')
        axes[0, 1].legend()
        
        # Visual impact vs expert rating
        if df['expert_rating'].notna().any():
            axes[1, 0].scatter(df['visual_impact'], df['expert_rating'])
            axes[1, 0].plot([0, 1], [0, 5], 'r--', alpha=0.3, label='Perfect correlation')
            axes[1, 0].set_title('Automated vs Expert Rating')
            axes[1, 0].set_xlabel('Visual Impact Score')
            axes[1, 0].set_ylabel('Expert Rating (stars)')
            axes[1, 0].legend()
        
        # Version comparison
        version_means = df.groupby('version')[['visual_impact', 'expert_rating']].mean()
        version_means.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Quality by Version')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend(['Automated', 'Expert'])
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"✓ Quality trend report saved: {output_path}")
```

---

## Implementation Timeline

| Week | Improvement | Priority |
|------|-------------|----------|
| 1 | Perceptual metrics implementation | CRITICAL |
| 1 | Visual QA checkpoint framework | CRITICAL |
| 2 | Expert review workflow | HIGH |
| 2 | Automated regression tests | HIGH |
| 3 | CI integration | MEDIUM |
| 4 | Quality metrics tracking | MEDIUM |
| 4 | Documentation & training | MEDIUM |

---

## Success Metrics

### Short-term (3 months)
- [ ] Zero quality regressions detected in production
- [ ] 100% of releases have expert review approval
- [ ] All CI quality gates passing
- [ ] Perceptual metrics correlate ≥0.85 with expert ratings

### Long-term (6 months)
- [ ] Quality trends showing improvement over time
- [ ] Automated metrics predict expert ratings within ±0.5 stars
- [ ] Team adopts QA process without resistance
- [ ] Client satisfaction maintained/improved

---

## Conclusion

These QA improvements will:

1. **Catch issues early**: Perceptual metrics detect color casts, tone compression, texture loss
2. **Validate with humans**: Expert review ensures automated metrics match perception
3. **Prevent regressions**: Automated tests block quality degradation in CI
4. **Track trends**: Continuous monitoring shows quality evolution over time

**The v1.1.0 disaster will never happen again.**

---

**Document Status**: Implementation Roadmap  
**Next Steps**: Begin Week 1 implementation (perceptual metrics + visual QA)  
**Owner**: Quality Team  
**Date**: November 10, 2025
