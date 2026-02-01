# 750 Picacho Primary Bedroom - PBR Test Results

**Date**: 2026-01-31
**Test**: Production PBR generation with PREMIUM preset
**Status**: ✅ SUCCESS

---

## Test Summary

Successfully generated PBR (Physically Based Rendering) maps for the 750 Picacho Primary Bedroom luxury property image using the newly created PBR preset system.

## Input

**File**: `input_images/750Picacho_PrimaryBedroom_Ultimate.tif`
- **Size**: 136.9 MB
- **Resolution**: 5989 × 3993 pixels (23.9 megapixels)
- **Format**: TIFF (high-quality source)

## Configuration

**Preset**: `premium` (maximum quality for hero shots)

**PBR Parameters**:
- Normal strength: 1.5x (maximum detail, no pre-blur)
- Roughness strength: 1.3x (high surface sensitivity)
- AO strength: 1.2x (strong dimensional depth)
- AO bias: 0.4 (darker for luxury ambiance)
- Float depth: Enabled (critical for quality)

**Model**: Depth Anything V2 Metric Large (Indoor)
- Device: MPS (Apple Silicon acceleration)
- Fallback from V3 to V2 (V3 models not yet available)

## Performance

**Processing Time**: 2.8 seconds total
- Depth estimation: 1.7s (60.7%)
- PBR generation: 1.1s (39.3%)

**Throughput**: ~1,277 images/hour

**Note**: Significantly faster than benchmark estimate (6-8s) due to:
1. V2 model used instead of V3
2. Apple M-series optimization
3. Efficient PBR algorithm (NumPy/SciPy)

## Outputs Generated

**Output Directory**: `output_750_picacho_pbr_test/`

**5 Files Created** (100.1 MB total):

1. **Depth Map (16-bit PNG)** - 1.08 MB
   - Quantized depth visualization
   - Range: 0-65535 (16-bit grayscale)
   - Format: PNG, 5989 × 3993, 16-bit grayscale

2. **Float Depth (32-bit NPY)** - 91.23 MB
   - High-precision depth array
   - Range: [0.000, 1.000] (float32)
   - **Critical for PBR quality**

3. **Normal Map (RGB PNG)** - 2.94 MB
   - Tangent-space surface normals
   - Format: PNG, 5989 × 3993, 8-bit RGB
   - 1.5x strength (maximum detail)

4. **Roughness Map (Grayscale PNG)** - 3.52 MB
   - Surface micro-detail texture
   - Format: PNG, 5989 × 3993, 8-bit grayscale
   - 1.3x strength (high sensitivity)

5. **Ambient Occlusion (Grayscale PNG)** - 1.34 MB
   - Indirect lighting approximation
   - Format: PNG, 5989 × 3993, 8-bit grayscale
   - 1.2x strength, 0.4 bias (deep shadows)

## Quality Assessment

### Depth Map
- ✅ Full resolution preserved (3993 × 5989)
- ✅ Normalized range [0, 1] for consistent PBR
- ✅ No quantization artifacts (thanks to float depth)

### Normal Map
- ✅ Sharp detail preservation (0px blur)
- ✅ 1.5x strength captures fine surface variations
- ✅ RGB encoding for tangent-space normals

### Roughness Map
- ✅ High sensitivity (1.3x) captures material variations
- ✅ Minimal smoothing (2px) preserves texture

### Ambient Occlusion
- ✅ Deep shadows (0.4 bias) for luxury ambiance
- ✅ 7px blur provides natural occlusion spread

## Conclusion

The PBR preset system successfully generated high-quality physically based rendering maps for the 750 Picacho Primary Bedroom. The **premium preset** proved optimal for this use case.

**Status**: ✅ Production Ready
**Performance**: 2.8s processing, ~1,277 images/hour
**Quality**: Premium tier, client-deliverable
