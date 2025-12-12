# Lux Depth V2 - Phase 2 Implementation Guide

**Status**: Foundation Complete (Architectural Scaffolding)  
**Effort Estimate**: 64-86 hours (4-6 weeks)  
**Expected Impact**: 60-80% boundary precision improvement, >85% material classification accuracy  
**Risk Level**: Medium (new ML models, API integration)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Task 1: EfficientSAM Integration (24-32h)](#task-1-efficientsam-integration)
4. [Task 2: CLIP Material Classification (16-24h)](#task-2-clip-material-classification)
5. [Task 3: Expand Material Classes (12-16h)](#task-3-expand-material-classes)
6. [Task 4: Lighting Condition Metadata (12-14h)](#task-4-lighting-condition-metadata)
7. [Integration & Testing](#integration--testing)
8. [Code Templates](#code-templates)
9. [Dependencies](#dependencies)
10. [Timeline & Milestones](#timeline--milestones)

---

## Overview

### Phase 2 Goals

Phase 2 builds on Phase 1's Material Property Schema and Hybrid Depth Zones to deliver:

1. **EfficientSAM Backend**: Segment Anything Model for 60-80% boundary precision improvement
2. **CLIP Classification**: Zero-shot material classification with >85% accuracy
3. **Expanded Taxonomy**: 18-24 material classes for luxury real estate
4. **Lighting Metadata**: Adaptive tone mapping and color grading

### Success Criteria

- ✅ EfficientSAM boundary precision > SegFormer-B5 by 60-80%
- ✅ CLIP material classification accuracy > 85%
- ✅ Expanded taxonomy covers 85%+ of pool/kitchen scenes
- ✅ Lighting detection enables adaptive processing
- ✅ No performance regression (< 2x slowdown vs Phase 1)
- ✅ Backward compatible with Phase 1 configurations

### Dependencies

**Phase 1 (Complete)**:
- ✅ Material Property Schema
- ✅ Hybrid Depth Zones
- ✅ SegFormer-B5 backend
- ✅ Confidence-gated material response

**Phase 2 (In Progress)**:
- 🔨 Architectural scaffolding (THIS COMMIT)
- ⏳ Implementation (4-6 weeks)
- ⏳ Validation & benchmarking
- ⏳ Documentation & examples

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Lux Depth V2 Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐     ┌──────────────────────────────────┐    │
│  │  RGB Input     │────▶│  Lighting Condition Detector     │    │
│  └────────────────┘     │  (Task 4: 12-14h)                │    │
│                         │  - Sky analysis                   │    │
│         │               │  - Time of day classification     │    │
│         │               │  - Adaptive tone mapping rules    │    │
│         ▼               └──────────────────────────────────┘    │
│                                       │                          │
│  ┌────────────────┐                   ▼                          │
│  │ Depth Map      │         ┌──────────────────────────────┐    │
│  │ (Depth Any V2) │         │  Material Segmentation       │    │
│  └────────────────┘         │  (Task 1 + Task 2)           │    │
│                             │                               │    │
│         │                   │  ┌──────────────────────┐    │    │
│         │                   │  │ EfficientSAM         │    │    │
│         ▼                   │  │ (Task 1: 24-32h)     │    │    │
│                             │  │ - Prompt engineering │    │    │
│  ┌────────────────┐         │  │ - Mask generation   │    │    │
│  │ Depth Zones    │         │  │ - Quality filtering │    │    │
│  │ (Phase 1)      │         │  └──────────┬───────────┘    │    │
│  └────────────────┘         │             │                │    │
│                             │             ▼                │    │
│         │                   │  ┌──────────────────────┐    │    │
│         │                   │  │ CLIP Classifier      │    │    │
│         │                   │  │ (Task 2: 16-24h)     │    │    │
│         ▼                   │  │ - Zero-shot classify │    │    │
│                             │  │ - Hybrid SegF fusion │    │    │
│  ┌────────────────┐         │  │ - NL query interface │    │    │
│  │ Materials V2   │◀────────┤  └──────────────────────┘    │    │
│  │ (Phase 1)      │         │                               │    │
│  │ - Property     │         │  ┌──────────────────────┐    │    │
│  │   Schema       │         │  │ Material Taxonomy    │    │    │
│  │ - Confidence   │         │  │ (Task 3: 12-16h)     │    │    │
│  │   Gating       │         │  │ - 18-24 classes      │    │    │
│  └────────────────┘         │  │ - ADE20K mapping     │    │    │
│                             │  │ - Property presets   │    │    │
│         │                   │  └──────────────────────┘    │    │
│         ▼                   └──────────────────────────────┘    │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Enhanced Output                                        │    │
│  │  - Precise material boundaries (EfficientSAM)          │    │
│  │  - Accurate material classification (CLIP)             │    │
│  │  - Expanded material coverage (18-24 classes)          │    │
│  │  - Adaptive processing (lighting metadata)             │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Phase**:
   - RGB image → Lighting detector (sky analysis, time-of-day)
   - RGB image → Depth Anything V2 → Depth map
   - Lighting metadata → Adaptive processing rules

2. **Segmentation Phase**:
   - RGB + Depth → EfficientSAM (prompt generation → mask generation)
   - SAM masks → CLIP (zero-shot classification → material labels)
   - CLIP + SegFormer → Hybrid fusion (confidence-weighted)

3. **Processing Phase**:
   - Material masks → Materials V2 Engine (property schema lookup)
   - Depth zones + Material properties → Per-zone, per-material enhancement
   - Lighting metadata → Adaptive tone mapping, color grading

4. **Output Phase**:
   - Enhanced RGB with precise boundaries
   - Material segmentation metadata (18-24 classes)
   - Lighting condition metadata (JSON)

---

## Task 1: EfficientSAM Integration

**Effort**: 24-32 hours  
**Priority**: High (critical for boundary precision improvement)  
**Status**: Stub created in `lux_depth_v2/material_segmentation.py`

### Objectives

- Integrate EfficientSAM for high-precision material boundary detection
- Achieve 60-80% improvement over SegFormer-B5 boundary precision
- Maintain < 2x processing time overhead

### Implementation Checklist

#### 1.1 Research & Model Selection (4-6h)

- [ ] **Research EfficientSAM variants**:
  - [EfficientSAM-S](https://github.com/yformer/EfficientSAM): ~36MB, fastest
  - EfficientSAM-Ti: ~24MB, ultra-fast
  - Distilled variants: Custom distilled models
- [ ] **Benchmark model variants** on pool/kitchen scenes:
  - Boundary precision (IoU, F1, boundary recall)
  - Processing time per image
  - Memory footprint
- [ ] **Select optimal variant** (recommend: EfficientSAM-S for balance)
- [ ] **Download and cache model checkpoint**

#### 1.2 Model Loading & Initialization (4-6h)

- [ ] **Implement `__init__()` in EfficientSAMSegmenter**:
  ```python
  def __init__(self, cfg, device):
      # Load EfficientSAM checkpoint
      from efficient_sam.build_efficient_sam import build_efficient_sam_vits
      self.model = build_efficient_sam_vits()
      self.model.load_state_dict(torch.load(cfg.efficientSAM_model))
      self.model.to(device)
      self.model.eval()
      
      # Initialize prompt encoder and mask decoder
      self.prompt_encoder = self.model.prompt_encoder
      self.mask_decoder = self.model.mask_decoder
  ```
- [ ] **Validate model loads correctly**
- [ ] **Set up device placement** (CPU/CUDA/MPS)
- [ ] **Enable mixed precision** for speed (fp16 on CUDA)

#### 1.3 Prompt Engineering (8-12h)

- [ ] **Implement `_generate_architectural_prompts()`**:
  - **Grid-based prompts**: Uniform coverage (16x16 grid for 512px)
  - **Edge-aware prompts**: Detect edges, place box prompts on structures
  - **Material-specific prompts**:
    - Water: Lower third of image (pools, water features)
    - Sky: Upper third of image
    - Vegetation: Green-dominant regions
    - Architecture: Vertical/horizontal edges
- [ ] **Adaptive prompt density**: More prompts in complex regions
- [ ] **Prompt format conversion**: (x, y) points, (x1, y1, x2, y2) boxes

**Code Template**:
```python
def _generate_architectural_prompts(self, rgb: torch.Tensor) -> List[Dict]:
    """Generate prompts optimized for architectural scenes."""
    b, c, h, w = rgb.shape
    prompts = []
    
    # Grid-based prompts for uniform coverage
    grid_spacing = max(h, w) // 16
    for i in range(0, h, grid_spacing):
        for j in range(0, w, grid_spacing):
            prompts.append({
                'type': 'point',
                'coords': [[j, i]],  # (x, y)
                'labels': [1],  # Foreground
            })
    
    # Edge-aware prompts (detect edges with Sobel/Canny)
    edges = detect_edges(rgb)  # TODO: Implement
    edge_points = extract_edge_points(edges, num_points=50)
    for (x, y) in edge_points:
        prompts.append({
            'type': 'point',
            'coords': [[x, y]],
            'labels': [1],
        })
    
    # Material-specific prompts
    # Water: bottom third
    prompts.append({
        'type': 'box',
        'coords': [[0, int(h * 0.66), w, h]],  # (x1, y1, x2, y2)
    })
    
    # Sky: top third
    prompts.append({
        'type': 'box',
        'coords': [[0, 0, w, int(h * 0.33)]],
    })
    
    return prompts
```

#### 1.4 Mask Generation (6-8h)

- [ ] **Implement `predict()` method**:
  - Preprocess RGB to EfficientSAM format (1024x1024 recommended)
  - Run inference with prompts
  - Post-process masks (resize, soften, threshold)
  - Return Dict[material_name, mask_tensor]

**Code Template**:
```python
def predict(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Generate material masks using EfficientSAM."""
    # Generate prompts
    prompts = self._generate_architectural_prompts(rgb)
    
    # Preprocess to EfficientSAM input size
    rgb_resized, scale = resize_to_1024(rgb)
    
    # Encode prompts
    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        points=batch_points(prompts),
        boxes=batch_boxes(prompts),
        masks=None,
    )
    
    # Generate masks with mask decoder
    low_res_masks, iou_predictions = self.mask_decoder(
        image_embeddings=self.model.image_encoder(rgb_resized),
        image_pe=self.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    
    # Postprocess masks
    masks = postprocess_masks(low_res_masks, rgb.shape[-2:])
    
    # Classify masks with CLIP (Task 2 integration)
    material_masks = self._classify_masks_with_CLIP(rgb, masks)
    
    return material_masks
```

#### 1.5 Quality Filtering (2-4h)

- [ ] **Filter low-quality masks**:
  - IoU threshold (e.g., > 0.5)
  - Stability score threshold (consistent across scales)
  - Minimum area threshold (avoid tiny spurious masks)
- [ ] **Merge overlapping masks** for same material
- [ ] **Confidence scoring** per mask

#### 1.6 Integration & Testing (4-6h)

- [ ] **Update `create_material_segmenter()` factory**
- [ ] **Test on pool scene** (750Picacho_Pool)
- [ ] **Test on kitchen scene**
- [ ] **Benchmark boundary precision** vs SegFormer-B5
- [ ] **Document processing time** and memory usage

### Expected Outputs

- **Boundary Precision**: 60-80% improvement over SegFormer-B5 (measured on validation set)
- **Processing Time**: < 200ms per image (512px, M4 Max with MPS)
- **Memory**: < 4GB VRAM for 512px images

---

## Task 2: CLIP Material Classification

**Effort**: 16-24 hours  
**Priority**: High (enables zero-shot classification and hybrid fusion)  
**Status**: Stub created in `lux_depth_v2/materials_v2.py`

### Objectives

- Implement CLIP-based zero-shot material classification
- Achieve >85% classification accuracy
- Enable natural language query interface
- Implement hybrid SegFormer+CLIP fusion

### Implementation Checklist

#### 2.1 Research & Model Selection (2-4h)

- [ ] **Research CLIP model variants**:
  - [ViT-B/32](https://github.com/openai/CLIP): Fast, 224px, good accuracy
  - ViT-L/14: Slower, 336px, best accuracy
  - OpenCLIP variants: Community models
- [ ] **Benchmark on material classification task**
- [ ] **Select optimal variant** (recommend: ViT-B/32 for speed)

#### 2.2 Model Loading (2-3h)

- [ ] **Implement `__init__()` in CLIPMaterialClassifier**:
  ```python
  def __init__(self, device, model_name="ViT-B/32"):
      import clip
      self.model, self.preprocess = clip.load(model_name, device=device)
      self.model.eval()
      self.device = device
      
      # Precompute text embeddings for material templates
      self.material_embeddings = self._precompute_embeddings()
  ```
- [ ] **Precompute text embeddings** for efficiency

#### 2.3 Zero-Shot Classification (4-6h)

- [ ] **Implement `classify_image()`**:
  - Encode image with CLIP vision encoder
  - Compute cosine similarity with material embeddings
  - Return confidence scores [0, 1]

**Code Template**:
```python
def classify_image(self, rgb: torch.Tensor, material_classes=None):
    """Classify materials in image using zero-shot CLIP."""
    # Encode image
    with torch.no_grad():
        image_features = self.model.encode_image(rgb)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity with material embeddings
        material_classes = material_classes or self.material_classes
        text_features = self.material_embeddings[material_classes]
        
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        # Return confidence scores
        return {cls: sim.item() for cls, sim in zip(material_classes, similarity[0])}
```

- [ ] **Design material templates** (see Section 8.2)
- [ ] **Test on pool/kitchen scenes**

#### 2.4 Natural Language Query (4-6h)

- [ ] **Implement `query_natural_language()`**:
  - Generate dense image embeddings (patch-level)
  - Compute attention map for query
  - Return mask highlighting relevant regions

**Use Cases**:
- "surfaces that would reflect light" → glass, water, polished metal
- "natural materials like wood or stone" → wood, stone, vegetation
- "water features" → pool, fountains

#### 2.5 Hybrid SegFormer+CLIP Fusion (4-6h)

- [ ] **Implement `fuse_with_segformer()`**:
  - SegFormer provides spatial priors (WHERE)
  - CLIP refines classification (WHAT)
  - Confidence-weighted fusion: `alpha * segformer + (1-alpha) * clip`
  - Resolve conflicts using confidence scores

**Code Template**:
```python
def fuse_with_segformer(self, rgb, segformer_masks, segformer_confidences):
    """Hybrid fusion of SegFormer + CLIP."""
    refined_masks = {}
    
    for material, seg_mask in segformer_masks.items():
        # Extract region features
        region_features = extract_region_features(rgb, seg_mask)
        
        # Classify region with CLIP
        clip_scores = self.classify_image(region_features)
        
        # Compute fusion alpha based on SegFormer confidence
        seg_conf = segformer_confidences[material].mean()
        alpha = min(0.9, seg_conf + 0.2)  # High SegFormer conf → trust SegFormer
        
        # Fuse masks
        clip_material = max(clip_scores, key=clip_scores.get)
        if clip_material == material:
            # Agreement: keep SegFormer mask with high confidence
            refined_masks[material] = seg_mask
        else:
            # Conflict: blend based on confidence
            refined_masks[material] = alpha * seg_mask + (1 - alpha) * clip_mask
    
    return refined_masks
```

#### 2.6 Testing & Benchmarking (2-4h)

- [ ] **Validate accuracy** on ground truth dataset
- [ ] **Benchmark inference time**
- [ ] **Measure hybrid fusion accuracy gain**

### Expected Outputs

- **Classification Accuracy**: >85% on validation set
- **Hybrid Fusion Gain**: +5-10% accuracy improvement over SegFormer alone
- **Inference Time**: <100ms per image (ViT-B/32, batch=1)

---

## Task 3: Expand Material Classes

**Effort**: 12-16 hours  
**Priority**: Medium (enhances coverage and granularity)  
**Status**: Stub created in `lux_depth_v2/materials_v2.py`

### Objectives

- Expand from 8 to 18-24 material classes
- Map to ADE20K semantic classes
- Create property schema presets for each class
- Achieve 85%+ segmentation coverage on pool/kitchen scenes

### Implementation Checklist

#### 3.1 Define Material Taxonomy (3-4h)

- [ ] **Define MaterialClass enum** (see Section 8.3)
- [ ] **Document each class** with description and use cases
- [ ] **Organize into categories**:
  - Architecture (6 classes)
  - Hardscape (4 classes)
  - Water (3 classes)
  - Vegetation (5 classes)
  - Sky (2 classes)

#### 3.2 ADE20K Mapping (3-4h)

- [ ] **Implement `get_ade20k_mapping()`**:
  - Map each material class to ADE20K semantic labels
  - Handle many-to-many relationships (e.g., "tile" → pool_tile or tile_surface)
- [ ] **Validate mappings** against ADE20K label set

#### 3.3 Property Schema Presets (4-6h)

- [ ] **Implement `get_property_schema()`**:
  - Create MaterialPropertySchema preset for each class
  - Tune parameters per material (gloss, roughness, albedo, etc.)
- [ ] **Validate presets** on test images

#### 3.4 Integration & Testing (2-4h)

- [ ] **Update MaterialsV2Config** with `use_expanded_taxonomy` flag
- [ ] **Test backward compatibility** (Phase 1 classes still work)
- [ ] **Measure coverage** on pool/kitchen scenes

### Expected Outputs

- **Material Classes**: 18-24 unique classes
- **Coverage**: 85%+ of pool/kitchen scene area classified
- **Accuracy**: >80% per-class accuracy

---

## Task 4: Lighting Condition Metadata

**Effort**: 12-14 hours  
**Priority**: Low-Medium (enables adaptive processing)  
**Status**: Stub created in `lux_depth_v2/lighting_detector.py`

### Objectives

- Detect lighting conditions (time of day, sky characteristics)
- Enable adaptive tone mapping and color grading
- Improve results for dawn/golden hour/twilight scenes

### Implementation Checklist

#### 4.1 Sky Region Analysis (3-4h)

- [ ] **Implement `_analyze_sky_region()`**:
  - Extract sky pixels using sky_mask from material segmentation
  - Convert to LAB color space
  - Compute average brightness (L channel)
  - Estimate color temperature from A/B channels

#### 4.2 Time-of-Day Classification (3-4h)

- [ ] **Implement `_classify_time_of_day()`**:
  - Decision tree based on sky characteristics
  - Golden hour: warm (>0.5), hue [20, 60]
  - Dawn: cool (<-0.3), hue [200, 260]
  - Twilight: cool, hue [240, 300]
  - Midday: neutral, high brightness

#### 4.3 Adaptive Tone Mapping (3-4h)

- [ ] **Implement `adapt_tone_mapping()`**:
  - Golden hour: Increase highlight preservation
  - Midday: Reduce shadow crushing, increase contrast
  - Overcast: Increase local contrast

#### 4.4 Adaptive Color Grading (2-3h)

- [ ] **Implement `adapt_color_grading()`**:
  - Golden hour: Enhance warm tones
  - Dawn/Twilight: Enhance cool tones
  - Midday: Neutral grading

#### 4.5 Integration (1-2h)

- [ ] **Integrate with pipeline** (pre-analysis phase)
- [ ] **Save metadata** to output JSON

### Expected Outputs

- **Detection Accuracy**: >80% time-of-day classification
- **Adaptive Improvement**: 5-15% subjective quality gain on challenging lighting

---

## Integration & Testing

### End-to-End Pipeline Integration

1. **Update `pipeline.py`**:
   - Add lighting detection pre-analysis
   - Integrate EfficientSAM + CLIP segmentation
   - Apply adaptive processing rules

2. **Configuration**:
   - Add feature gates (enabled=False by default)
   - Ensure backward compatibility

3. **Testing**:
   - Run full pipeline on pool/kitchen scenes
   - Validate all four tasks working together
   - Benchmark end-to-end performance

### Performance Targets

- **Processing Time**: < 2x Phase 1 (acceptable overhead)
- **Memory**: < 1.5x Phase 1 (efficient VRAM usage)
- **Quality**: 60-80% boundary precision improvement

### Validation Metrics

- **Boundary Precision**: IoU, F1, boundary recall (EfficientSAM)
- **Classification Accuracy**: Per-class and overall (CLIP)
- **Coverage**: % of scene area classified (expanded taxonomy)
- **Lighting Detection**: Confusion matrix (time-of-day)

---

## Code Templates

### 8.1 EfficientSAM Prompt Engineering

```python
def _generate_architectural_prompts(self, rgb: torch.Tensor) -> List[Dict]:
    """Generate prompts optimized for architectural scenes."""
    b, c, h, w = rgb.shape
    prompts = []
    
    # Grid-based prompts (uniform coverage)
    grid_spacing = max(h, w) // 16
    for i in range(0, h, grid_spacing):
        for j in range(0, w, grid_spacing):
            prompts.append({'type': 'point', 'coords': [[j, i]], 'labels': [1]})
    
    # Material-specific box prompts
    prompts.append({'type': 'box', 'coords': [[0, int(h*0.66), w, h]]})  # Water (bottom)
    prompts.append({'type': 'box', 'coords': [[0, 0, w, int(h*0.33)]]})  # Sky (top)
    
    return prompts
```

### 8.2 CLIP Material Templates

```python
def _get_material_templates(self) -> Dict[str, List[str]]:
    """Get text templates for material classification."""
    return {
        "pool_water": [
            "a photo of pool water",
            "clear blue swimming pool water",
            "reflective water surface in a luxury pool",
        ],
        "stone_paver": [
            "a photo of stone pavers",
            "natural stone paving in architectural design",
            "textured stone surface",
        ],
        "wood_structure": [
            "a photo of wood structure",
            "wooden architectural elements",
            "natural wood in luxury interior",
        ],
        "sky_gradient": [
            "a photo of sky",
            "clear blue sky gradient",
            "open sky in architectural photography",
        ],
        # TODO: Add remaining 18-24 material templates
    }
```

### 8.3 Material Class Enum

```python
class MaterialClass:
    """Expanded material taxonomy (18-24 classes)."""
    
    # Architecture
    STUCCO_WALL = "stucco_wall"
    STONE_COLUMN = "stone_column"
    ALUMINUM_FRAME = "aluminum_frame"
    WOOD_STRUCTURE = "wood_structure"
    CONCRETE_SURFACE = "concrete_surface"
    TILE_SURFACE = "tile_surface"
    
    # Hardscape
    POOL_TILE_MOSAIC = "pool_tile_mosaic"
    POOL_DECK_PAVER = "pool_deck_paver"
    STONE_PAVER = "stone_paver"
    CONCRETE_DECK = "concrete_deck"
    
    # Water
    POOL_WATER_SURFACE = "pool_water_surface"
    POOL_WATER_VOLUME = "pool_water_volume"
    WATER_FEATURE = "water_feature"
    
    # Vegetation
    TREE_CANOPY = "tree_canopy"
    FLOWERING_TREE = "flowering_tree"
    SHRUB = "shrub"
    GRASS = "grass"
    SUCCULENT = "succulent"
    
    # Sky
    SKY_GRADIENT = "sky_gradient"
    MOUNTAIN_DISTANT = "mountain_distant"
```

### 8.4 Lighting Detection

```python
def detect(self, rgb: torch.Tensor, depth_map=None, sky_mask=None) -> LightingCondition:
    """Detect lighting conditions in scene."""
    # Extract sky region
    sky_coverage, color_temp, brightness = self._analyze_sky_region(rgb, sky_mask)
    
    # Classify time of day
    dominant_hue = compute_dominant_hue(rgb)
    warmth = compute_warmth(rgb)
    time_of_day, confidence = self._classify_time_of_day(
        color_temp, brightness, dominant_hue, warmth
    )
    
    # Detect shadows
    has_shadows, shadow_dir = self._detect_shadows(rgb, depth_map)
    
    return LightingCondition(
        time_of_day=time_of_day,
        confidence=confidence,
        sky_coverage=sky_coverage,
        sky_color_temp=color_temp,
        sky_brightness=brightness,
        has_strong_shadows=has_shadows,
        shadow_direction=shadow_dir,
        dominant_hue=dominant_hue,
        warmth=warmth,
    )
```

---

## Dependencies

### New Python Packages

Add to `lux_depth_v2/requirements-repo.txt`:

```txt
# Phase 2 dependencies
efficient-sam>=0.1.0  # EfficientSAM for boundary precision
clip>=1.0  # OpenAI CLIP for zero-shot classification (ViT-B/32)
# OR: open_clip_torch>=2.0.0  # OpenCLIP alternative
```

### Model Downloads

1. **EfficientSAM**: ~36MB (EfficientSAM-S variant)
   - URL: https://github.com/yformer/EfficientSAM/releases
   - SHA256: (compute after download)

2. **CLIP**: ~350MB (ViT-B/32 variant)
   - Automatically downloaded by `clip.load()`
   - Alternative: OpenCLIP models

### Version Compatibility

- Python 3.10+
- PyTorch 2.0+
- transformers 4.30+ (for SegFormer, already installed)
- CUDA 11.8+ or MPS (Apple Silicon)

---

## Timeline & Milestones

### Week 1: EfficientSAM Research + Prototyping (24-32h)

- **Days 1-2**: Research EfficientSAM variants, download models
- **Days 3-4**: Implement model loading and basic inference
- **Days 5-6**: Prompt engineering and mask generation
- **Day 7**: Testing and debugging

**Deliverable**: Working EfficientSAM backend with boundary precision benchmark

### Week 2: CLIP Integration + Material Expansion (28-40h)

- **Days 1-2**: CLIP model selection and zero-shot classification
- **Days 3-4**: Natural language query and hybrid fusion
- **Days 5-6**: Expand material taxonomy to 18-24 classes
- **Day 7**: Integration testing

**Deliverable**: CLIP classifier with >85% accuracy, expanded material taxonomy

### Week 3: Lighting Detection + Integration (12-14h)

- **Days 1-2**: Lighting detection implementation
- **Days 3-4**: Adaptive tone mapping and color grading
- **Days 5-7**: End-to-end pipeline integration

**Deliverable**: Complete Phase 2 pipeline with all four tasks

### Week 4: Testing + Validation + Documentation (16-20h)

- **Days 1-3**: Comprehensive testing (pool, kitchen, diverse scenes)
- **Days 4-5**: Benchmarking and performance optimization
- **Days 6-7**: Documentation, examples, user guide

**Deliverable**: Production-ready Phase 2 with validation report

### Total Timeline: 4-6 weeks (64-86 hours)

---

## Success Metrics

### Quantitative Metrics

- ✅ **Boundary Precision**: 60-80% improvement over SegFormer-B5
- ✅ **Classification Accuracy**: >85% overall, >75% per-class
- ✅ **Coverage**: 85%+ of pool/kitchen scenes classified
- ✅ **Processing Time**: <2x Phase 1 (acceptable overhead)
- ✅ **Memory**: <1.5x Phase 1 VRAM usage

### Qualitative Metrics

- ✅ **Visual Quality**: Crisp material boundaries, no artifacts
- ✅ **Usability**: Easy configuration, clear documentation
- ✅ **Backward Compatibility**: Phase 1 configs work unchanged

---

## Next Steps

1. **Review this guide** and architectural scaffolding
2. **Prioritize tasks** based on project needs (recommend: Task 1 → Task 2 → Task 3 → Task 4)
3. **Set up development environment** with Phase 2 dependencies
4. **Begin Task 1** (EfficientSAM) with research and model selection
5. **Iterate and test** each component before moving to next task

---

**Phase 2 Foundation Complete** ✅  
**Ready for Implementation** 🚀

For questions or clarification, consult:
- `lux_depth_v2/material_segmentation.py` (EfficientSAM stub)
- `lux_depth_v2/materials_v2.py` (CLIP stub, MaterialClass stub)
- `lux_depth_v2/lighting_detector.py` (Lighting stub)
- `lux_depth_v2/tests/test_*` (Test stubs with expected behavior)
