# Holdout Images Directory

**Purpose**: Real-world architectural glass negatives for Phase C completion gate  
**Status**: Awaiting acquisition (15 images required)

## Usage

```bash
# Set environment variable before validation
export WATER_HOLDOUT_DIR="/Users/rc/Transformation_Portal/data/water_v0/holdout_images"

# Run validation
./scripts/validate_holdout.sh

# Results archived in ../holdout_results/run_YYYYMMDD_HHMMSS/
```

See `../HOLDOUT_ACQUISITION_SPEC.md` for image selection requirements.
