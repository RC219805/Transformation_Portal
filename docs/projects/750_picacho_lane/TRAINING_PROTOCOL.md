# 750 Picacho Lane Training Protocol

## Executive Summary

This document describes the comprehensive property-specific training protocol for the **750 Picacho Lane luxury estate** in Montecito, CA. The protocol integrates architectural data analysis, material-aware optimization, and depth intelligence to produce a fully operational 4K 16-bit TIFF enhancement pipeline.

**Project Number**: 24098.00  
**Property Type**: Contemporary Mediterranean Luxury Estate  
**Target Output**: 6 enhanced 4K 16-bit TIFF files  
**Protocol Version**: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Component Details](#component-details)
4. [Training Stages](#training-stages)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Output Specifications](#output-specifications)
8. [Quality Metrics](#quality-metrics)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Overview

### Objective

Create a property-specific enhancement model optimized for 750 Picacho Lane that:
- Understands the specific materials present (stone, glass, water, wood, metal, fabric)
- Respects architectural features and lighting conditions
- Produces professional-quality 4K 16-bit TIFF output
- Maintains consistency across all room types

### Property Images

| Room | Type | Primary Materials |
|------|------|-------------------|
| Exterior | Outdoor | Stucco, Stone, Glass, Vegetation |
| Living Room | Interior | Wood, Fabric, Glass, Stone |
| Kitchen | Interior | Stone, Metal, Wood, Glass |
| Pool | Outdoor | Water, Stone, Glass |
| Primary Bathroom | Interior | Stone, Glass, Metal |
| Primary Bedroom | Interior | Fabric, Wood, Glass |

### Key Features

- **Material-Aware Processing**: Automatic detection and enhancement of 6 material types
- **Depth Intelligence**: Depth Anything V2 Large model ensemble for architectural depth
- **Multi-Stage Training**: Progressive resolution training (512→1024→2048)
- **Production-Ready Output**: 16-bit TIFF with full metadata preservation

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     750 PICACHO LANE TRAINING PROTOCOL                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   PROPERTY   │───▶│    DEPTH     │───▶│   DATASET    │                   │
│  │   ANALYSIS   │    │  SYNTHESIS   │    │  GENERATION  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              MULTI-STAGE TRAINING                    │                    │
│  │  ┌─────────┐   ┌─────────────────┐   ┌───────────┐  │                    │
│  │  │Stage 1  │──▶│     Stage 2     │──▶│  Stage 3  │  │                    │
│  │  │Material │   │ Architectural   │   │Full-Res   │  │                    │
│  │  │Learning │   │  Refinement     │   │Fine-tune  │  │                    │
│  │  │ 512px   │   │    1024px       │   │  2048px   │  │                    │
│  │  │20 epochs│   │   20 epochs     │   │10 epochs  │  │                    │
│  │  └─────────┘   └─────────────────┘   └───────────┘  │                    │
│  └─────────────────────────────────────────────────────┘                    │
│                            │                                                 │
│                            ▼                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  VALIDATION  │───▶│  PRODUCTION  │───▶│   FINAL      │                   │
│  │              │    │  INFERENCE   │    │  DELIVERY    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Property Analysis (`picacho_analyzer.py`)

Analyzes all 6 property images to extract:

- **Material Detection**: Identifies stone, glass, water, wood, metal, fabric using color-space analysis
- **Color Palette Extraction**: Dominant colors, temperature, saturation
- **Architectural Features**: Ceiling type, lighting, view type, style
- **Quality Metrics**: Resolution, dynamic range, sharpness

```python
from training.property_specific import PicachoAnalyzer

analyzer = PicachoAnalyzer()
report = analyzer.analyze_property()
report.save("property_analysis.json")
```

### 2. Depth Synthesis (`depth_synthesis.py`)

Generates high-quality depth maps using Depth Anything V2:

- **Model Ensemble**: Large + Base models with weighted averaging
- **Architectural Priors**: Vertical gradient, vignette compensation
- **Edge Enhancement**: Depth discontinuities at detected edges
- **Multi-Format Export**: 16-bit PNG, float32 TIFF, colorized visualization

```python
from training.property_specific import DepthSynthesis

depth_synth = DepthSynthesis()
depth_map = depth_synth.synthesize("image.tiff")
depth_map.save("output/", save_16bit=True, save_float32=True)
```

### 3. Dataset Generation (`dataset_generator.py`)

Creates augmented training dataset:

- **Multi-Scale Crops**: 512px (40%), 1024px (40%), 2048px (20%)
- **Depth Correspondence**: Aligned image-depth pairs
- **Material-Aware Augmentation**: Material-specific color adjustments
- **Quality Augmentation**: Noise, blur, JPEG compression

```python
from training.property_specific import DatasetGenerator

generator = DatasetGenerator(analyzer, depth_synth)
samples = generator.generate_dataset(num_samples=600)
generator.save_dataset("data/training_750picacho")
```

### 4. Multi-Stage Trainer (`picacho_trainer.py`)

Progressive training with three stages:

| Stage | Resolution | Epochs | Focus |
|-------|------------|--------|-------|
| 1 | 512px | 20 | Material patterns, texture learning |
| 2 | 1024px | 20 | Architectural details, spatial relationships |
| 3 | 2048px | 10 | Fine details, full-resolution quality |

```python
from training.property_specific import PicachoTrainer

trainer = PicachoTrainer(config_path="config/training/750_picacho_lane_protocol.yaml")
trainer.train()
```

### 5. Production Inference (`picacho_inference.py`)

Processes full 4K images for final delivery:

- **Tiled Processing**: Memory-efficient processing of large images
- **Material Enhancement**: Material-specific post-processing
- **16-bit Output**: Full quality TIFF with metadata

```python
from training.property_specific import PicachoInference

inference = PicachoInference(model_path="weights/750_picacho/best_model.pth")
result = inference.process("source_image.tiff")
result.save("enhanced.tiff")
```

---

## Training Stages

### Stage 1: Material Learning (512px, 20 epochs)

**Objective**: Learn material-specific enhancement patterns at low resolution.

**Focus Areas**:
- Stone texture enhancement
- Glass clarity optimization
- Water reflection handling
- Wood grain preservation
- Metal highlight control
- Fabric softness rendering

**Configuration**:
```yaml
stage1:
  resolution: 512
  batch_size: 8
  learning_rate: 0.0001
  epochs: 20
```

### Stage 2: Architectural Refinement (1024px, 20 epochs)

**Objective**: Refine architectural details and spatial relationships.

**Focus Areas**:
- Edge sharpness optimization
- Perspective consistency
- Lighting coherence
- Depth-aware processing
- Structural detail preservation

**Configuration**:
```yaml
stage2:
  resolution: 1024
  batch_size: 4
  learning_rate: 0.00005
  epochs: 20
```

### Stage 3: Full-Resolution Fine-tuning (2048px, 10 epochs)

**Objective**: Final quality refinement at production resolution.

**Focus Areas**:
- 4K texture detail
- Fine grain preservation
- Micro-contrast optimization
- Color fidelity verification

**Configuration**:
```yaml
stage3:
  resolution: 2048
  batch_size: 2
  learning_rate: 0.00001
  epochs: 10
```

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install ML extras
pip install -e ".[ml]"

# Optional: Install TIFF support
pip install -e ".[tiff]"
```

### Run Complete Pipeline

```bash
# Full pipeline (recommended)
./scripts/training/750_picacho/run_complete_pipeline.sh

# Quick test mode
./scripts/training/750_picacho/run_complete_pipeline.sh --quick-train

# Skip training (use existing model)
./scripts/training/750_picacho/run_complete_pipeline.sh --skip-training
```

### Run Individual Stages

```bash
# 1. Analyze property
python scripts/training/750_picacho/01_analyze_property.py

# 2. Synthesize depth maps
python scripts/training/750_picacho/02_synthesize_depth.py

# 3. Generate training dataset
python scripts/training/750_picacho/03_generate_dataset.py

# 4. Train model
python scripts/training/750_picacho/04_train_model.py

# 5. Validate model
python scripts/training/750_picacho/05_validate_model.py

# 6. Process final output
python scripts/training/750_picacho/06_process_final_output.py
```

---

## Configuration

### Main Configuration File

`config/training/750_picacho_lane_protocol.yaml`

```yaml
name: "750_picacho_lane_training_protocol"

property:
  name: "750 Picacho Lane"
  project_number: "24098.00"

data:
  total_samples: 600
  crop_sizes: [512, 1024, 2048]

training:
  stage1:
    epochs: 20
    resolution: 512
  stage2:
    epochs: 20
    resolution: 1024
  stage3:
    epochs: 10
    resolution: 2048

loss:
  mse_weight: 1.0
  perceptual_weight: 1.0
  depth_weight: 0.3
  material_weight: 0.5
```

### Environment Variables

```bash
# Compute device
export TP_DEVICE=auto  # auto, cuda, mps, cpu

# Model cache
export TP_MODEL_CACHE=~/.cache/transformation_portal

# Debug mode
export TP_DEBUG=1
```

---

## Output Specifications

### Final Deliverables

| File | Format | Resolution | Bit Depth |
|------|--------|------------|-----------|
| Exterior_enhanced.tiff | TIFF | 4096×3072 | 16-bit |
| LivingRoom_enhanced.tiff | TIFF | 4096×3072 | 16-bit |
| Kitchen_enhanced.tiff | TIFF | 4096×3072 | 16-bit |
| Pool_enhanced.tiff | TIFF | 4096×3072 | 16-bit |
| PrimaryBathroom_enhanced.tiff | TIFF | 4096×3072 | 16-bit |
| PrimaryBedroom_enhanced.tiff | TIFF | 4096×3072 | 16-bit |

### Output Directory Structure

```
output/750_picacho/
├── property_analysis.json
├── validation/
│   ├── validation_results.json
│   └── comparisons/
├── final_deliverables/
│   ├── 20240215_143022/
│   │   ├── Exterior_enhanced.tiff
│   │   ├── LivingRoom_enhanced.tiff
│   │   ├── Kitchen_enhanced.tiff
│   │   ├── Pool_enhanced.tiff
│   │   ├── PrimaryBathroom_enhanced.tiff
│   │   ├── PrimaryBedroom_enhanced.tiff
│   │   └── processing_report.json
│   └── latest -> 20240215_143022/
└── logs/
```

---

## Quality Metrics

### Target Thresholds

| Metric | Target | Minimum |
|--------|--------|---------|
| PSNR | ≥35 dB | ≥30 dB |
| SSIM | ≥0.92 | ≥0.85 |
| LPIPS | ≤0.15 | ≤0.20 |

### Validation Report

```json
{
  "num_samples": 30,
  "psnr": {"mean": 36.5, "std": 2.1},
  "ssim": {"mean": 0.94, "std": 0.02},
  "processing_time": {"mean": 2.3, "total": 69.0}
}
```

---

## Troubleshooting

### Common Issues

#### Model Not Found

```
❌ Model not found: weights/750_picacho/best_model.pth
```

**Solution**: Run the training stage first, or use `--skip-training` to use fallback enhancement.

#### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or use tiled processing:
```bash
python 04_train_model.py --stage1-batch-size 4
```

#### No Property Images Found

```
❌ No images found. Please check the property directory.
```

**Solution**: Ensure images are in `projects/750_picacho_lane/Final_Production_UltraQuality/`

### Debug Mode

```bash
export TP_DEBUG=1
python scripts/training/750_picacho/04_train_model.py --verbose
```

---

## API Reference

### PicachoAnalyzer

```python
class PicachoAnalyzer:
    def __init__(self, property_dir: Optional[Path] = None)
    def analyze_property(self) -> PropertyReport
    def get_image_paths(self) -> List[Path]
    def get_room_types(self) -> List[RoomType]
```

### DepthSynthesis

```python
class DepthSynthesis:
    def __init__(self, config: Optional[DepthSynthesisConfig] = None)
    def synthesize(self, image: Union[Path, Image]) -> SynthesizedDepth
    def synthesize_all(self, images: List[Path]) -> List[SynthesizedDepth]
```

### DatasetGenerator

```python
class DatasetGenerator:
    def __init__(self, analyzer, depth_synthesis, config)
    def generate_dataset(self, num_samples: int) -> List[TrainingSample]
    def save_dataset(self, output_dir: Path) -> Dict[str, Any]
```

### PicachoTrainer

```python
class PicachoTrainer:
    def __init__(self, config: Optional[TrainingConfig] = None)
    def train(self) -> Dict[str, Any]
```

### PicachoInference

```python
class PicachoInference:
    def __init__(self, config: Optional[InferenceConfig] = None)
    def process(self, image: Union[Path, Image]) -> EnhancedOutput
    def process_batch(self, images: List[Path]) -> List[EnhancedOutput]
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12 | Initial release |

---

## References

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Transformation Portal Documentation](../../README.md)
- [750 Picacho Elite Preset](../../../config/750_picacho_elite_preset.yaml)

---

*Document generated by Transformation Portal Training Protocol v1.0.0*
