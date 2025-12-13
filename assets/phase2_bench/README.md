# Phase 2 Validation Benchmark Set

Small, curated set of images for validating Phase 2 CLIP + Lighting detection features.

## Benchmark Images

### Interior Scenes

1. **750Picacho_Kitchen_Ultimate.tif**
   - Scene: Interior kitchen
   - Lighting: Natural daylight
   - Expected: interior_luxury_apex_quality preset
   - Materials: Wood cabinets, stone counters, metal appliances

2. **750Picacho_PrimaryBedroom_Ultimate.tif**
   - Scene: Interior bedroom
   - Lighting: Warm ambient
   - Expected: interior_luxury presets
   - Materials: Textiles, wood, glass

3. **750Picacho_PrimaryBathroom_Ultimate.tif**
   - Scene: Interior bathroom
   - Lighting: Mixed (natural + artificial)
   - Expected: interior_luxury presets
   - Materials: Tile, glass, metal (high specular)

### Exterior Scenes

4. **750Picacho_Pool_Ultimate.tif**
   - Scene: Exterior pool
   - Lighting: Twilight/golden hour
   - Expected: exterior_pool_apex_quality preset
   - Materials: Water, stone, vegetation

5. **750Picacho_Aerial_Ultimate.tif** (optional)
   - Scene: Exterior facade/landscape
   - Lighting: Bright daylight
   - Expected: exterior_showcase preset
   - Materials: Roofing, hardscape, landscape

## Usage

```bash
# Run full benchmark matrix
scripts/run_phase2_bench_matrix.sh

# Run single test case
lux-depth-v2 --input assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif \
             --output-dir outputs/phase2_test \
             --auto-preset --quality-tier apex
```

## Expected Outcomes

- CLIP correctly classifies scene types (>0.7 confidence)
- Lighting detector identifies time of day
- Auto-preset selection matches manual expert choice
- Quality metrics within APEX thresholds
- Phase 2 overhead < 500ms per image
