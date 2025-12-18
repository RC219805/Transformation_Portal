# Advanced Depth Map Enhancement Strategies

**Current State**: 65,536 unique levels, 16-bit precision, 4K DCI resolution  
**Goal**: Further elevate quality, precision, and usability for luxury architectural rendering

---

## 1. Model Ensemble (Highest Quality Gain)

### Strategy: Multi-Model Fusion
Combine multiple depth estimation models to cancel out individual model errors and increase precision.

**Implementation**:
```python
models = [
    "depth-anything/Depth-Anything-V2-Large-hf",      # Current (335M params)
    "depth-anything/Depth-Anything-V2-Base-hf",       # 97M params
    "Intel/dpt-large",                                 # 344M params
    "LiheYoung/depth-anything-large-hf",              # Original DA v1
]

# Generate depth from each model
depths = [generate_depth(img, model) for model in models]

# Weighted average (or median for robustness)
depth_ensemble = np.average(depths, axis=0, weights=[0.4, 0.2, 0.3, 0.1])
```

**Expected Improvements**:
- ✅ Reduced noise (averaging cancels random errors)
- ✅ Better edge accuracy (consensus-based boundaries)
- ✅ Smoother gradients (model-specific artifacts reduced)
- ✅ More unique levels (sub-pixel precision from averaging)

**Cost**: 3-4x processing time (can parallelize on multiple GPUs)

---

## 2. Depth Refinement with Image Guidance

### Strategy: Edge-Aware Filtering
Use the original RGB image to guide depth map refinement, ensuring depth edges align with image edges.

**A. Guided Filter (Fast, Excellent Results)**
```python
from scipy.ndimage import generic_filter
import cv2

# Guided filter preserves edges from RGB while smoothing depth
depth_refined = cv2.ximgproc.guidedFilter(
    guide=rgb_image,       # RGB image as guide
    src=depth_map,         # Depth to refine
    radius=8,              # Larger = smoother
    eps=0.01              # Edge preservation strength
)
```

**B. Joint Bilateral Filter (Edge-Preserving)**
```python
depth_refined = cv2.bilateralFilter(
    depth_map,
    d=9,           # Diameter
    sigmaColor=75, # Color space sigma
    sigmaSpace=75  # Coordinate space sigma
)
```

**C. Domain Transform Filter (Fast Alternative)**
```python
depth_refined = cv2.ximgproc.dtFilter(
    guide=rgb_image,
    src=depth_map,
    sigmaSpatial=10,
    sigmaColor=25
)
```

**Expected Improvements**:
- ✅ Sharper depth boundaries aligned with image edges
- ✅ Reduced noise in flat regions
- ✅ Preserved fine details (windows, fixtures, molding)
- ✅ Better depth discontinuities at object boundaries

---

## 3. Super-Resolution Depth Enhancement

### Strategy: Generate at Higher Resolution, Then Downsample
Process image at 2x or 4x resolution, then downsample for better effective resolution.

**Implementation**:
```python
# Upscale input to 8K
image_8k = image.resize((8192, 4320), Image.Resampling.LANCZOS)

# Generate depth at 8K
depth_8k = model(image_8k)

# Downsample to 4K with antialiasing
depth_4k_supersampled = cv2.resize(
    depth_8k,
    (4096, 2160),
    interpolation=cv2.INTER_AREA  # Best for downsampling
)
```

**Expected Improvements**:
- ✅ Smoother gradients (antialiasing effect)
- ✅ Better fine detail capture
- ✅ Reduced stairstepping on edges
- ✅ More natural depth transitions

**Cost**: 4x memory, 2-3x processing time

---

## 4. Metric Depth Calibration

### Strategy: Convert Relative Depth to Absolute Metric Depth
Use architectural constraints to convert relative depth to real-world distances.

**A. MiDaS with Metric Depth (ZoeDepth)**
```python
# ZoeDepth provides metric depth (meters)
model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
depth_metric = model.infer(image)  # Output in meters
```

**B. Manual Calibration (If room dimensions known)**
```python
# Known: Kitchen is 15 feet deep
known_depth_ft = 15.0
depth_range = depth_max - depth_min

# Scale to metric
depth_metric = ((depth - depth_min) / depth_range) * known_depth_ft
```

**Expected Improvements**:
- ✅ Physically accurate depth (for VR, 3D reconstruction)
- ✅ Consistent scaling across images
- ✅ Better for multi-view consistency
- ✅ Enables real-world effects (DoF based on f-stop)

---

## 5. Temporal Consistency (For Video/Multi-Image Sets)

### Strategy: Enforce Depth Consistency Across Similar Views
If multiple angles of same scene exist, enforce depth consistency.

**Implementation**:
```python
# For images i and i+1 with camera motion
flow = compute_optical_flow(img_i, img_i1)
depth_i1_warped = warp_depth(depth_i, flow)

# Blend warped and predicted depth
depth_i1_consistent = 0.7 * depth_i1_predicted + 0.3 * depth_i1_warped
```

**Expected Improvements**:
- ✅ No flickering in video depth sequences
- ✅ Better multi-view 3D reconstruction
- ✅ Smoother depth across camera movements

---

## 6. Normal Map Generation for Enhanced Detail

### Strategy: Compute Surface Normals from Depth Gradients
Derive surface orientation for better material/lighting effects.

**Implementation**:
```python
# Compute depth gradients
dz_dx = np.gradient(depth, axis=1)
dz_dy = np.gradient(depth, axis=0)

# Convert to surface normals
normals = np.stack([
    -dz_dx,
    -dz_dy,
    np.ones_like(depth)
], axis=-1)

# Normalize
normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

# Convert to 8-bit RGB normal map
normal_map = ((normals + 1) / 2 * 255).astype(np.uint8)
```

**Expected Improvements**:
- ✅ Better material response (normal-based lighting)
- ✅ Enhanced detail perception
- ✅ Improved 3D visualization
- ✅ PBR-compatible output

---

## 7. Depth Uncertainty Estimation

### Strategy: Generate Confidence/Uncertainty Maps
Identify regions where depth estimation is unreliable.

**A. Monte Carlo Dropout (Model-Based)**
```python
# Enable dropout at inference time
model.train()  # Enables dropout

# Multiple forward passes
depths = [model(image) for _ in range(10)]

# Uncertainty = variance across predictions
depth_mean = np.mean(depths, axis=0)
depth_uncertainty = np.std(depths, axis=0)
```

**B. Gradient-Based Confidence**
```python
# Low gradients in depth = high confidence
# High gradients = edges (medium confidence)
# Erratic gradients = uncertainty

gradient_magnitude = compute_gradient_magnitude(depth)
confidence = 1.0 / (1.0 + gradient_magnitude)
```

**Expected Improvements**:
- ✅ Identify problematic regions (glass, mirrors, textureless)
- ✅ Weight multi-model fusion by confidence
- ✅ Adaptive post-processing (more filtering in uncertain areas)
- ✅ Quality assurance for production pipelines

---

## 8. Stereo Depth Fusion (If Multiple Viewpoints Available)

### Strategy: Combine Monocular ML with Stereo Geometry
If you have 2+ images of same scene from different angles.

**Implementation**:
```python
# Classical stereo matching
depth_stereo = compute_stereo_depth(left_img, right_img)

# ML monocular depth
depth_mono = model(left_img)

# Weighted fusion (stereo is metric, mono has better coverage)
depth_fused = alpha * depth_stereo + (1 - alpha) * depth_mono_scaled
```

**Expected Improvements**:
- ✅ Metric accuracy from stereo
- ✅ Dense coverage from monocular
- ✅ Better hole filling
- ✅ Physically accurate depth

---

## 9. Scene-Specific Fine-Tuning

### Strategy: Fine-Tune Model on Architectural Interiors
Create small dataset of luxury interiors with depth ground truth.

**Implementation**:
```python
# Collect 50-100 luxury interior images
# Use LiDAR depth or manual annotation for ground truth

# Fine-tune Depth Anything V2 Large
model.train()
for epoch in range(10):
    for img, depth_gt in luxury_dataset:
        loss = criterion(model(img), depth_gt)
        loss.backward()
        optimizer.step()
```

**Expected Improvements**:
- ✅ Domain-specific accuracy (luxury interiors)
- ✅ Better on common materials (marble, glass, metal)
- ✅ Improved on challenging scenes (reflections, lighting)
- ✅ Tailored to 750 Picacho style

**Cost**: Requires ground truth depth data (LiDAR scan or manual)

---

## 10. Post-Processing Pipeline Optimization

### Strategy: Sophisticated Multi-Stage Enhancement

**Full Pipeline**:
```python
# Stage 1: Multi-model ensemble
depth_raw = ensemble_models([DA2_Large, DPT, DA_v1])

# Stage 2: Edge-aware guided filter
depth_refined = guided_filter(depth_raw, rgb_image)

# Stage 3: Super-resolution (if needed)
depth_sr = super_resolve_depth(depth_refined)

# Stage 4: Percentile normalization
depth_norm = percentile_normalize(depth_sr, p_low=0.5, p_high=99.5)

# Stage 5: Adaptive histogram equalization (optional)
depth_enhanced = adaptive_histogram_equalize(
    depth_norm,
    clip_limit=1.5,  # Conservative
    tile_size=128
)

# Stage 6: 16-bit conversion with dithering
depth_16bit = convert_16bit_dithered(depth_enhanced)
```

**Expected Improvements**:
- ✅ Best-in-class quality (combines all techniques)
- ✅ Production-ready output
- ✅ Minimal artifacts
- ✅ Maximum precision

---

## Recommended Implementation Priority

### Tier 1: Immediate High-Impact (< 1 hour effort)
1. **Guided Filter** - Biggest quality gain for minimal effort
2. **Percentile Refinement** - Already implemented, tune parameters
3. **Normal Map Generation** - Adds value with minimal cost

### Tier 2: Significant Improvement (1-3 hours effort)
4. **Model Ensemble** (DA2 Large + DPT) - 2-model fusion is sweet spot
5. **Super-Resolution** (2x) - Process at 8K, downsample to 4K
6. **Depth Uncertainty** - Valuable for QA and selective refinement

### Tier 3: Advanced/Specialized (3+ hours effort)
7. **Metric Depth (ZoeDepth)** - If physical accuracy needed
8. **Temporal Consistency** - If processing video or image sequences
9. **Fine-Tuning** - If you have ground truth depth data

---

## Practical Next Steps

### Option A: Quick Win (15 minutes)
Apply guided filter to existing depth map:
```bash
python enhance_depth_guided.py \
  --depth outputs/depth_4k_dci/750Picacho_Kitchen_4K_DCI_depth_enhanced_16bit.tiff \
  --rgb input_images/750_Picacho/Optimized_TIFFs/750Picacho_Kitchen_4K.tiff \
  --output depth_guided_enhanced.tiff
```

### Option B: Maximum Quality (1 hour)
Multi-model ensemble + guided filter:
```bash
python generate_depth_ensemble.py \
  --input input_images/750_Picacho/Optimized_TIFFs/750Picacho_Kitchen_4K.tiff \
  --models depth-anything-v2-large intel-dpt-large \
  --refine guided-filter \
  --output depth_ultimate_quality.tiff
```

### Option C: Research-Grade (3+ hours)
Full pipeline with all enhancements:
```bash
python depth_ultimate_pipeline.py \
  --input Kitchen_4K.tiff \
  --ensemble 3-models \
  --super-resolution 2x \
  --guided-filter \
  --uncertainty-map \
  --normal-map \
  --output depth_research_grade/
```

---

## Expected Quality Improvements

| Enhancement | Unique Levels | Edge Sharpness | Processing Time | Complexity |
|-------------|--------------|----------------|-----------------|------------|
| **Current** | 65,536 | 98.01 | 0.28s | Baseline |
| **+ Guided Filter** | 65,536 | **~150** | +0.5s | Low |
| **+ Ensemble (2)** | 65,536+ | **~180** | +0.3s | Medium |
| **+ Super-Res (2x)** | 65,536+ | **~200** | +1.5s | Medium |
| **+ Full Pipeline** | 65,536+ | **~250** | +3.0s | High |

---

## Conclusion

**Current state is already excellent** (65K levels, true 16-bit), but several paths exist for further improvement:

**Best ROI**: Guided filter (2x edge sharpness, minimal effort)  
**Best Quality**: Multi-model ensemble + guided filter  
**Best for Production**: Current + guided filter + normal maps

Would you like me to implement any of these enhancements?
