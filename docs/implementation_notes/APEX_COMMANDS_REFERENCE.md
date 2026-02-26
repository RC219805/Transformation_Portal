# APEX V2 Enhancement - Command Reference
# Generated: $(date)

## Quick Reference

### Six Source Files to Process
```
/Users/rc/Projects/Transformation_Portal/input_images/source_tiffs/
├── V2_750Picacho_Aerial.tiff           (396 MB)
├── V2_750Picacho_GreatRoom.tiff        ( 69 MB)
├── V2_750Picacho_Kitchen.tiff          (116 MB)
├── V2_750Picacho_Pool.tiff             (116 MB)
├── V2_750Picacho_PrimaryBathroom.tiff  (275 MB)
└── V2_750Picacho_PrimaryBedroom.tiff   (137 MB)
```

Total input size: ~1.1 GB (6 files)

---

## Option 1: Automated Batch Processing (Recommended)

**Script:** `scripts/pipelines/process_source_tiffs_apex.sh`

**Features:**
- Processes all 6 files automatically
- Fail-fast on errors
- Progress tracking
- Individual logs per file
- Summary report

**Run:**
```bash
cd /Users/rc/Projects/Transformation_Portal
./scripts/pipelines/process_source_tiffs_apex.sh
```

**Output:**
```
output_apex_v2_luxury/
├── V2_750Picacho_Aerial_enhanced.png
├── V2_750Picacho_Aerial_report.json
├── V2_750Picacho_GreatRoom_enhanced.png
├── V2_750Picacho_GreatRoom_report.json
├── ... (6 images + 6 reports)
```

---

## Option 2: Individual Commands

**Script:** `scripts/pipelines/process_source_tiffs_individual.sh`

Or run commands individually:

### Command Template
```bash
python3 scripts/enhance_image.py \
    input_images/source_tiffs/[FILENAME].tiff \
    --output-dir output_apex_v2_luxury \
    --preset luxury_estate \
    --device mps \
    --depth-dir depth_maps_apex \
    --verbose
```

### Individual Commands (copy-paste ready)

**1. Aerial View**
```bash
python3 scripts/enhance_image.py \
    input_images/source_tiffs/V2_750Picacho_Aerial.tiff \
    --output-dir output_apex_v2_luxury \
    --preset luxury_estate \
    --device mps \
    --depth-dir depth_maps_apex \
    --verbose
```

**2. Great Room**
```bash
python3 scripts/enhance_image.py \
    input_images/source_tiffs/V2_750Picacho_GreatRoom.tiff \
    --output-dir output_apex_v2_luxury \
    --preset luxury_estate \
    --device mps \
    --depth-dir depth_maps_apex \
    --verbose
```

**3. Kitchen**
```bash
python3 scripts/enhance_image.py \
    input_images/source_tiffs/V2_750Picacho_Kitchen.tiff \
    --output-dir output_apex_v2_luxury \
    --preset luxury_estate \
    --device mps \
    --depth-dir depth_maps_apex \
    --verbose
```

**4. Pool**
```bash
python3 scripts/enhance_image.py \
    input_images/source_tiffs/V2_750Picacho_Pool.tiff \
    --output-dir output_apex_v2_luxury \
    --preset luxury_estate \
    --device mps \
    --depth-dir depth_maps_apex \
    --verbose
```

**5. Primary Bathroom**
```bash
python3 scripts/enhance_image.py \
    input_images/source_tiffs/V2_750Picacho_PrimaryBathroom.tiff \
    --output-dir output_apex_v2_luxury \
    --preset luxury_estate \
    --device mps \
    --depth-dir depth_maps_apex \
    --verbose
```

**6. Primary Bedroom**
```bash
python3 scripts/enhance_image.py \
    input_images/source_tiffs/V2_750Picacho_PrimaryBedroom.tiff \
    --output-dir output_apex_v2_luxury \
    --preset luxury_estate \
    --device mps \
    --depth-dir depth_maps_apex \
    --verbose
```

---

## Advanced Features Enabled

### Preset: `luxury_estate`
- **Enhancement Strength:** 0.8 (80% - premium marketing aesthetic)
- **Clarity Strength:** 0.6 (60% - crisp detail enhancement)
- **Material Strength:** 0.7 (70% - material-specific processing)
- **Depth-Aware Tone Mapping:** Enabled (spatial hierarchy)
- **Atmospheric Effects:** Enabled (ambient occlusion, depth haze)

### Processing Features
✓ **Depth-aware tone mapping** - Creates spatial depth hierarchy
✓ **Material detection & processing** - Wood, metal, glass, textiles, leather
✓ **Clarity enhancement** - Edge-preserving sharpening
✓ **Atmospheric effects** - Ambient occlusion, depth haze, light wrap
✓ **Color grading** - Luxury real estate aesthetic
✓ **Perceptual finishing** - Professional-grade final touches

### Hardware Acceleration
- **Device:** `mps` (Apple Silicon GPU)
- **Expected Performance:** <2 seconds per image (without depth generation)
- **With Depth Generation:** Add ~5-10s per image for Depth Pro inference

### Output Format
- **Enhanced Image:** PNG (high quality, lossless)
- **Report:** JSON with comprehensive metadata
  - Processing parameters
  - Runtime metrics
  - Enhancement settings
  - File paths and sizes

---

## Alternative Presets

### Default (Balanced)
```bash
--preset default
# enhancement=0.7, clarity=0.5, material=0.6
```

### Architectural (Technical)
```bash
--preset architectural
# enhancement=0.6, clarity=0.7, material=0.5
# More clarity, less atmosphere
```

### None (Passthrough - no enhancement)
```bash
--preset none
# Skip V2 enhancement entirely
```

---

## Performance Expectations

### Without Pre-generated Depth Maps
- **First run:** ~7-12s per image (includes depth generation with Depth Pro)
- **Subsequent runs:** <2s per image (reuses cached depth)

### With Pre-generated Depth Maps
- **Processing:** <2s per image (enhancement only)
- **Total batch:** ~12s for all 6 files

### Depth Map Generation (Optional)
If you want to pre-generate depth maps:
```bash
# Requires depth-pro package installed
# pip install depth-pro

for file in input_images/source_tiffs/*.tiff; do
    filename=$(basename "$file" .tiff)
    python scripts/run_depth_estimation.py \
        --input "$file" \
        --output "depth_maps_apex/${filename}_depth.png" \
        --backend depth_pro \
        --device mps
done
```

---

## Troubleshooting

### Missing Depth Maps
If depth maps are not found, enhancement proceeds **without** depth-aware processing:
- Material processing: ✓ Still active
- Clarity enhancement: ✓ Still active
- Depth-aware tone mapping: ✗ Skipped
- Atmospheric effects: ✗ Skipped

### Synthetic Fallback
If ML dependencies are missing, system falls back to synthetic depth:
- Fast processing (no model inference)
- Deterministic output
- Lower quality depth information

### Device Selection
**Recommended order:**
1. `mps` - Apple Silicon (M1/M2/M3) - Fastest
2. `cuda` - NVIDIA GPU - Fast
3. `cpu` - Universal fallback - Slower

---

## Next Steps

1. **Run batch processing:**
   ```bash
   ./scripts/pipelines/process_source_tiffs_apex.sh
   ```

2. **Review outputs:**
   ```bash
   open output_apex_v2_luxury/
   ```

3. **Check reports:**
   ```bash
   jq . output_apex_v2_luxury/*_report.json | less
   ```

4. **Compare quality:**
   - Review enhanced vs. original
   - Check material processing accuracy
   - Validate depth effects (if depth maps used)

---

**Documentation:**
- V2 Enhancement Quickstart: `docs/V2_ENHANCEMENT_QUICKSTART.md`
- Architectural Guidance: `docs/architecture/decisions/V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md`
- CLI Reference: `docs/cli/CLI_REFERENCE.md`
