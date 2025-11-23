# Training Scripts for Hyper-Reality Enhancement

This directory contains automated scripts for training the neural networks in the Transformation Portal.

## 🚀 Quick Start

### Recommended: Train on Real Project Data
```bash
./scripts/train_with_750picacho.sh
```

This trains models on the 750 Picacho BIM project data (best quality results).

### Alternative: Train on Synthetic Data
```bash
./scripts/quickstart_training.sh
```

This trains models on generated synthetic data (faster, good baseline).

### Validation: Quick Demo
```bash
python scripts/quick_train_demo.py
```

This runs a minimal training demo (3 epochs, 50 pairs) to validate the infrastructure.

## 📁 Available Scripts

### 1. `train_with_750picacho.sh`
**Purpose:** Train on real architectural project data  
**Dataset:** 750 Picacho Lane BIM data (530+ pairs)  
**Duration:** ~2.5-3.5 hours on M4 Max  
**Quality:** 103-107/100 target  

**Usage:**
```bash
./scripts/train_with_750picacho.sh [options]

Options:
  --epochs N        Train for N epochs (default: 50)
  --batch-size N    Training batch size (default: 4)
  --max-bim N       Max BIM images to use (default: 500)
  --help            Show help message
```

**Example:**
```bash
# Train for 100 epochs with batch size 8
./scripts/train_with_750picacho.sh --epochs 100 --batch-size 8
```

### 2. `quickstart_training.sh`
**Purpose:** Train on synthetic data  
**Dataset:** Generated synthetic pairs (1000 default)  
**Duration:** ~2-3 hours on M4 Max  
**Quality:** 100-103/100 target  

**Usage:**
```bash
./scripts/quickstart_training.sh [options]

Options:
  --num-pairs N     Generate N training pairs (default: 1000)
  --epochs N        Train for N epochs (default: 50)
  --batch-size N    Training batch size (default: 4)
  --help            Show help message
```

**Example:**
```bash
# Generate 2000 pairs and train for 100 epochs
./scripts/quickstart_training.sh --num-pairs 2000 --epochs 100
```

### 3. `quick_train_demo.py`
**Purpose:** Validate training infrastructure  
**Dataset:** 50 synthetic pairs  
**Duration:** ~5-10 minutes  
**Quality:** Demo only (not for production)  

**Usage:**
```bash
python scripts/quick_train_demo.py
```

**What it does:**
1. Generates 50 synthetic training pairs
2. Trains for 3 epochs (demonstration only)
3. Validates the training loop works correctly
4. Saves checkpoints to `weights/hyper_reality_demo/`

**When to use:**
- ✅ Verify training infrastructure is working
- ✅ Test before committing to long training runs
- ✅ Debug training issues
- ❌ NOT for production use (insufficient training)

## 🎯 Which Script Should I Use?

### For Production Quality
→ Use `train_with_750picacho.sh`
- Best quality results (103-107/100)
- Real architectural data
- Material realism
- Room-aware enhancements

### For Quick Baseline
→ Use `quickstart_training.sh`
- Good baseline quality (100-103/100)
- Faster than real data
- Consistent results
- Good for experimentation

### For Testing Only
→ Use `quick_train_demo.py`
- Validates infrastructure
- Quick feedback (~10 minutes)
- Not production-ready
- Debug purposes only

## 📊 Expected Results

| Script | Dataset | Pairs | Epochs | Duration* | Quality | PSNR Gain | SSIM Gain |
|--------|---------|-------|--------|-----------|---------|-----------|-----------|
| `train_with_750picacho.sh` | Real BIM | 530+ | 50 | 2.5-3.5h | 103-107/100 | +13-15 dB | +28-31% |
| `quickstart_training.sh` | Synthetic | 1000 | 50 | 2-3h | 100-103/100 | +11-12 dB | +25-27% |
| `quick_train_demo.py` | Synthetic | 50 | 3 | 5-10min | Demo only | - | - |

*Duration on M4 Max with MPS. CPU training is 4-6x slower.

## 🔧 Requirements

### Hardware
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 10-15GB free space
- **GPU:** CUDA or Apple Silicon (recommended)

### Software
- Python 3.10+
- PyTorch 2.0+
- All ML dependencies:
  ```bash
  pip install -r requirements/ml.txt
  ```

## 💡 Tips

### Optimize Training Speed
```bash
# Use GPU if available
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Apple Silicon
export CUDA_VISIBLE_DEVICES=0         # NVIDIA GPU

# Increase batch size (if you have enough memory)
./scripts/train_with_750picacho.sh --batch-size 8
```

### Monitor Training
```bash
# Watch training progress
tail -f weights/hyper_reality_*/training.log

# Check checkpoints
ls -lh weights/hyper_reality_*/
```

### Resume Training
Training saves checkpoints every 5 epochs. To resume:
```bash
python src/enhancements/train_hyper_reality.py \
    --data-dir data/training_750picacho \
    --checkpoint-dir weights/hyper_reality_750picacho \
    --resume-from weights/hyper_reality_750picacho/checkpoint_epoch_25.pth
```

### Reduce Memory Usage
```bash
# Reduce batch size
./scripts/train_with_750picacho.sh --batch-size 2

# Reduce image size (in training script)
# Edit config: image_size: 256 (default: 512)
```

## 📚 Related Documentation

- **Training Guide:** `../docs/TRAINING_GUIDE.md`
- **750 Picacho Guide:** `../docs/750_PICACHO_TRAINING.md`
- **Execution Guide:** `../docs/migrated/TRAINING_EXECUTION_GUIDE.md`
- **Model Status:** `../docs/MODEL_TRAINING_STATUS.md`

## 🐛 Troubleshooting

### Training is slow
- **Check GPU:** `python -c "import torch; print(torch.cuda.is_available())"`
- **Use smaller batch size:** `--batch-size 2`
- **Consider synthetic data:** `quickstart_training.sh` (faster than BIM data)

### Out of memory
- **Reduce batch size:** `--batch-size 1`
- **Close other applications**
- **Use CPU (slow):** Set environment variable `CUDA_VISIBLE_DEVICES=""`

### Data not found
- **750 Picacho:** Verify `projects/750_picacho_lane/Final_Production_UltraQuality/` exists
- **BIM images:** Verify `extracted_context/24098.00_750 PICACHO LANE_images/` exists
- **Generate synthetic:** Run with `--generate-data` flag

### Script fails
- **Check dependencies:** `pip install -r requirements/ml.txt`
- **Verify Python version:** `python --version` (need 3.10+)
- **Run demo first:** `python scripts/quick_train_demo.py`

## 🏆 After Training

Once training completes, use the trained models:

```bash
# Process images with trained models
python src/enhancements/hyper_reality_enhancement.py \
    input.jpg \
    -o output.jpg \
    -q 105

# Models are automatically loaded from:
# weights/hyper_reality/best_model.pth
# or weights/hyper_reality_750picacho/best_model.pth
```

## 📞 Support

Need help? Check:
1. **Documentation:** `../docs/TRAINING_GUIDE.md`
2. **Issues:** https://github.com/RC219805/Transformation_Portal/issues
3. **Discussions:** Community forum

---

**Last Updated:** 2025-11-19  
**Status:** All scripts validated and working  
**Recommendation:** Use `train_with_750picacho.sh` for best results
