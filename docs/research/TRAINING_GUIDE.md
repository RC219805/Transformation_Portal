# Training with 750 Picacho BIM Data

## Overview

This guide explains how to train Hyper-Reality Enhancement models using **real architectural data** from the 750 Picacho Lane project, including UltraQuality renders and BIM-extracted images.

## Why Use 750 Picacho Data?

### Real Project Data
- **6 UltraQuality TIFF renders** (23-42 MB each)
  - Kitchen, Pool, Aerial, Great Room, Primary Bedroom, Primary Bathroom
- **2,488 BIM-extracted images** from architectural plans
- **Architectural context** from BIM model (room types, dimensions, materials)

### Advantages Over Synthetic Data
- ✅ **Real architectural materials** (stucco, stone, glass, water)
- ✅ **Professional lighting** from actual renders
- ✅ **Authentic architectural details** from BIM
- ✅ **Room-specific characteristics** (kitchen vs bedroom vs outdoor)
- ✅ **Better generalization** to real-world architectural projects

### Expected Quality Improvements
Training on 750 Picacho data should produce:
- **Higher PSNR gains** (+13-15 dB vs +11-12 dB with synthetic)
- **Better material realism** (especially stucco, glass, water features)
- **Room-aware enhancements** (kitchen lighting vs bedroom ambiance)
- **More convincing results** on similar luxury real estate projects

## Quick Start

### One-Command Training
```bash
# Prepare data and train models (takes 2-4 hours total)
./scripts/train_with_750picacho.sh
```

This will:
1. Extract 30+ training pairs from UltraQuality renders
2. Sample 500 BIM images from architectural plans
3. Train for 50 epochs (~2-3 hours on M4 Max)
4. Test on actual project render

### Custom Configuration
```bash
./scripts/train_with_750picacho.sh --epochs 100 --max-bim 1000
```

## Manual Workflow

### Step 1: Prepare Training Data

```bash
python src/enhancements/prepare_750picacho_training_data.py \
    --output-dir data/training_750picacho \
    --max-bim-images 500
```

**Options:**
- `--output-dir`: Where to save training pairs (default: `data/training_750picacho`)
- `--max-bim-images`: Max BIM images to sample (default: 500)
- `--ultraquality-only`: Skip BIM images, use only renders
- `--bim-only`: Skip renders, use only BIM images
- `--crops-per-render`: Crops to extract from each render (default: 5)

**What it does:**
- Loads 6 UltraQuality TIFF renders
- Creates 5 crops per render (1024x1024) → 30 pairs
- Samples up to 500 BIM images → 500 pairs
- Applies room-specific degradations (kitchen vs pool vs aerial)
- Creates matching low→high quality pairs
- Saves metadata with architectural context

**Output:**
```
data/training_750picacho/
├── low_quality/
│   ├── 750picacho_kitchen_0000.png
│   ├── 750picacho_pool_0001.png
│   ├── bim_page01_img00_0030.png
│   └── ...
├── high_quality/
│   ├── 750picacho_kitchen_0000.png  (same names)
│   ├── 750picacho_pool_0001.png
│   ├── bim_page01_img00_0030.png
│   └── ...
└── dataset_metadata.json
```

### Step 2: Train Models

```bash
python src/enhancements/train_hyper_reality.py \
    --data-dir data/training_750picacho \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --checkpoint-dir weights/hyper_reality_750picacho
```

**Training Stats** (530 pairs):
- **Time**: ~2.5-3.5 hours on M4 Max
- **Memory**: ~8GB peak
- **Checkpoints**: Saved every 5 epochs
- **Best model**: Auto-saved to `best_model.pth`

### Step 3: Use Trained Models

```bash
# Copy weights to default location
cp weights/hyper_reality_750picacho/best_model.pth weights/hyper_reality/

# Process images (auto-loads trained weights)
python src/enhancements/hyper_reality_enhancement.py \
    your_render.jpg \
    -o enhanced.jpg \
    -q 105
```

## Data Preparation Details

### UltraQuality Render Processing

**Source renders:**
- `750Picacho_Kitchen_UltraQuality.tif` (23 MB, 3000x2250px)
- `750Picacho_Pool_UltraQuality.tif` (26 MB, 3000x2000px)
- `750Picacho_Aerial_UltraQuality.tif` (29 MB, 4000x3000px)
- `750Picacho_GreatRoom_UltraQuality.tif` (24 MB, 3000x2250px)
- `750Picacho_PrimaryBedroom_UltraQuality.tif` (35 MB, 4000x3000px)
- `750Picacho_PrimaryBathroom_UltraQuality.tif` (42 MB, 4500x3000px)

**Processing:**
1. Load TIFF as high-quality target
2. Extract 5 random 1024x1024 crops
3. Apply room-specific degradation
4. Save as PNG pairs

### Room-Specific Degradation Profiles

Different rooms get different degradation patterns to match real-world scenarios:

| Room | Contrast | Noise | Blur | Saturation | Rationale |
|------|----------|-------|------|------------|-----------|
| Kitchen | 0.75 | 6 | 0.6 | 0.80 | Moderate degradation, preserve lighting |
| Pool | 0.70 | 8 | 0.8 | 0.75 | Water reflection challenges |
| Aerial | 0.65 | 10 | 1.0 | 0.70 | Atmospheric haze, distance blur |
| Great Room | 0.75 | 5 | 0.6 | 0.80 | Similar to kitchen |
| Primary Bedroom | 0.70 | 6 | 0.7 | 0.75 | Softer lighting, fabric textures |
| Primary Bathroom | 0.72 | 6 | 0.7 | 0.78 | Stone/glass materials |

### BIM Image Processing

**Source:** 2,488 architectural images extracted from BIM PDF
- Floor plans
- Elevations
- Section drawings
- Detail callouts

**Processing:**
1. Filter out small icons (<200px)
2. Resize to 512x512 maintaining aspect ratio
3. Pad to square with gray background
4. Treat as high-quality (already clean line art)
5. Apply degradation to create low-quality version

**Why use BIM images?**
- Provides architectural line work and structure
- Teaches edge preservation
- Complements photorealistic renders
- Adds diversity to training set

## Architectural Context Integration

### BIM Context Available
```json
{
  "project_name": "00 750 PICACHO LANE",
  "floors": ["1st Floor", "Ground Floor", "2nd Floor"],
  "rooms": {
    "kitchen_0": {...},
    "primary_bedroom_0": {...},
    "bathroom_0": {...},
    ...
  }
}
```

**Used for:**
- Room-specific degradation profiles
- Dataset metadata
- Future: Context-aware enhancement (v2.0)

### Metadata Generated
```json
{
  "dataset_name": "750_Picacho_Training_Data",
  "project": "750 Picacho Lane",
  "total_pairs": 530,
  "source_types": {
    "ultraquality_renders": "High-quality architectural renders",
    "bim_images": "Extracted from BIM architectural plans"
  },
  "context_available": true,
  "rooms": ["kitchen_0", "primary_bedroom_0", ...],
  "degradation_types": [
    "Contrast reduction",
    "Noise addition",
    "Gaussian blur",
    "Saturation reduction",
    "JPEG compression"
  ]
}
```

## Training Results

### Expected Performance

**Dataset:** 530 pairs (30 renders + 500 BIM images)

| Metric | Before Training | After Training | Improvement |
|--------|----------------|----------------|-------------|
| PSNR | 19.2 dB | 32-34 dB | +13-15 dB |
| SSIM | 0.71 | 0.91-0.93 | +28-31% |
| Quality Score | 78/100 | 103-107/100 | +25-29 points |
| Training Time | - | 2.5-3.5 hrs | M4 Max |

### Validation Strategy

**10% validation split** (53 validation pairs):
- Monitor overfitting
- Track generalization to unseen rooms
- Early stopping if validation loss plateaus

### Checkpoint Management

**Saved every 5 epochs:**
- `checkpoint_epoch_5.pth`
- `checkpoint_epoch_10.pth`
- ...
- `checkpoint_epoch_50.pth`
- `best_model.pth` (best validation loss)

**Each checkpoint contains:**
- Model weights (caustics, atmosphere, materials, harmonics)
- Training state (epoch, optimizer, scheduler)
- Training history (losses, metrics)
- Dataset metadata

## Usage Examples

### Example 1: Full Training Workflow
```bash
# Prepare data
python src/enhancements/prepare_750picacho_training_data.py

# Train
python src/enhancements/train_hyper_reality.py \
    --data-dir data/training_750picacho \
    --epochs 50

# Test
python src/enhancements/hyper_reality_enhancement.py \
    projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Kitchen_UltraQuality.tif \
    -o kitchen_enhanced.jpg
```

### Example 2: UltraQuality Renders Only
```bash
# Use only high-quality renders (skip BIM images)
python src/enhancements/prepare_750picacho_training_data.py --ultraquality-only

# Train on 30 render-based pairs
python src/enhancements/train_hyper_reality.py \
    --data-dir data/training_750picacho \
    --epochs 100 \
    --batch-size 2  # Smaller dataset, smaller batch
```

### Example 3: BIM Images Only
```bash
# Use only BIM images (faster, more pairs)
python src/enhancements/prepare_750picacho_training_data.py \
    --bim-only \
    --max-bim-images 1000

# Train on 1000 BIM-based pairs
python src/enhancements/train_hyper_reality.py \
    --data-dir data/training_750picacho \
    --epochs 30
```

### Example 4: Combined with Synthetic
```bash
# Combine 750 Picacho data with synthetic data
python src/enhancements/prepare_750picacho_training_data.py \
    --output-dir data/training_750picacho

python src/enhancements/train_hyper_reality.py \
    --generate-data \
    --num-pairs 500 \
    --data-dir data/training

# Manually merge directories
cp data/training_750picacho/high_quality/* data/training/high_quality/
cp data/training_750picacho/low_quality/* data/training/low_quality/

# Train on combined dataset (1030 pairs)
python src/enhancements/train_hyper_reality.py \
    --data-dir data/training \
    --epochs 50
```

## Troubleshooting

### Issue: TIFF files not found
```
❌ No 750 Picacho data found!
```

**Solution:** Verify files exist
```bash
ls projects/750_picacho_lane/Final_Production_UltraQuality/*.tif
```

### Issue: BIM images not found
```
❌ BIM images directory not found
```

**Solution:** Check extraction
```bash
ls "extracted_context/24098.00_750 PICACHO LANE_images/" | head
```

### Issue: Out of memory during data prep
```
MemoryError: Unable to allocate array
```

**Solution:** Reduce max BIM images
```bash
python src/enhancements/prepare_750picacho_training_data.py --max-bim-images 200
```

### Issue: Training too slow
```
Epoch 1/50: [00:05<04:00:00, 0.5it/s]
```

**Solution:** Use smaller batch or fewer images
```bash
python src/enhancements/train_hyper_reality.py \
    --batch-size 2 \
    --data-dir data/training_750picacho
```

## Advanced: Fine-Tuning on Custom Projects

After training on 750 Picacho, fine-tune on your own projects:

```bash
# 1. Train on 750 Picacho (general architectural knowledge)
./scripts/train_with_750picacho.sh --epochs 50

# 2. Prepare your project data
python src/enhancements/prepare_750picacho_training_data.py \
    --output-dir data/training_myproject \
    # ... with your project's UltraQuality renders

# 3. Fine-tune (lower learning rate, fewer epochs)
python src/enhancements/train_hyper_reality.py \
    --data-dir data/training_myproject \
    --epochs 20 \
    --lr 5e-5 \
    --checkpoint-dir weights/hyper_reality_myproject
```

## Best Practices

1. **Use real data when available** - 750 Picacho > synthetic
2. **Start with UltraQuality renders** - Highest quality targets
3. **Sample BIM images intelligently** - Don't use all 2,488 (diminishing returns)
4. **Monitor validation loss** - Watch for overfitting
5. **Test on held-out renders** - Don't use all 6 for training
6. **Combine with synthetic** - For better generalization
7. **Fine-tune for specific projects** - Transfer learning from 750 Picacho

## Next Steps

After training on 750 Picacho data:

1. **Validate quality** - Process test renders and measure PSNR/SSIM
2. **Compare to baseline** - Side-by-side with untrained models
3. **Test on other projects** - Check generalization
4. **Share results** - Contribute trained weights to community
5. **Fine-tune** - Adapt to your specific architectural style

---

**Last Updated**: 2025-11-19  
**Version**: 1.0.0  
**Project**: 750 Picacho Lane, Montecito, CA
