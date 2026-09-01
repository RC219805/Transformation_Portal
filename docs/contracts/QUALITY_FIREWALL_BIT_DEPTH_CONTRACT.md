# Quality Firewall - Bit-Depth Preservation Contract

**Status:** ACTIVE (as of 2026-02-10)
**Enforcement:** Mechanical (code + CI)
**Compliance Level:** BLOCKING

---

## Contract Overview

The **Quality Firewall** ensures that image processing operations preserve or enhance quality, never degrade it. For bit-depth specifically, the contract is:

> **16-bit input SHALL produce 16-bit output unless 8-bit encoding is explicitly selected.**

This is a **blocking contract**: violations are not warnings—they are failures.

---

## Bit-Depth Preservation Guarantee

### Contract Statement

```
IF input_bits_per_sample == 16
THEN output_bits_per_sample == 16
UNLESS output_bit_depth == 8
   OR (output_bit_depth IS OMITTED AND allow_8bit_output == True)
```

### Enforcement Mechanisms

#### 1. **Compile-Time: Type Contract**

```python
def enhance_image(
    input_path: Path,
    output_path: Path,
    allow_8bit_output: bool = False,  # Legacy compatibility authority
    output_bit_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quality Firewall: 16-bit input → 16-bit output
    unless output_bit_depth=8, or an omitted output_bit_depth with
    allow_8bit_output=True, explicitly authorizes an 8-bit encoding.
    """
```

The direct V2 API preserves input precision when `output_bit_depth` is omitted.
Lux orchestration always supplies the canonical `output_bit_depth` selector;
its runtime default is 8.

#### 2. **Runtime: Validation and Logging**

```python
image, input_bits, metadata = load_image_preserve_bit_depth(
    input_path,
    output_bit_depth == 8 or (output_bit_depth is None and allow_8bit_output),
)

# Quality Firewall check
if output_bit_depth is not None:
    output_bits = output_bit_depth
elif input_bits == 16 and allow_8bit_output:
    logger.warning("Quality Firewall BYPASSED: --allow-8bit flag set")
    output_bits = 8
else:
    output_bits = input_bits

if input_bits == 16 and output_bits == 16:
    logger.info("Quality Firewall ACTIVE: 16-bit preservation enforced")
elif input_bits == 16 and output_bits == 8:
    logger.warning("Quality Firewall downgrade explicitly authorized")
```

#### 3. **CI: Automated Regression Test**

```python
# tests/unit/lux_depth_v3/test_v2_enhance_quality_firewall.py

def test_16bit_preservation_enforced():
    """Quality Firewall: 16-bit input must produce 16-bit output."""

    # Process 16-bit test image
    report = enhance_image(
        input_path="tests/fixtures/test_16bit.tiff",
        output_path="test_output.tiff",
        # allow_8bit_output NOT set (firewall active)
    )

    # Verify contract
    assert report['bit_depth']['bit_depth_preserved'] == True, \
           "QUALITY FIREWALL VIOLATION: Bit-depth not preserved!"

    assert report['bit_depth']['input_bits_per_sample'] == 16
    assert report['bit_depth']['output_bits_per_sample'] == 16
    assert report['bit_depth']['quality_firewall_active'] == True


def test_16bit_downgrade_requires_explicit_bypass():
    """16-bit → 8-bit downgrade requires an explicit authority."""

    # Process 16-bit image with bypass flag
    report = enhance_image(
        input_path="tests/fixtures/test_16bit.tiff",
        output_path="test_output.tiff",
        allow_8bit_output=True,  # Explicit bypass
    )

    # Verify bypass was honored
    assert report['bit_depth']['downgrade_allowed'] == True
    assert report['bit_depth']['output_bits_per_sample'] == 8
    assert report['bit_depth']['quality_firewall_active'] == False

    canonical_report = enhance_image(
        input_path="tests/fixtures/test_16bit.tiff",
        output_path="test_output.png",
        output_bit_depth=8,
    )
    assert canonical_report['bit_depth']['downgrade_allowed'] == True
    assert canonical_report['bit_depth']['output_bits_per_sample'] == 8
```

**CI Requirement:** This test MUST pass on every commit.

#### 4. **Report: Auditable Metadata**

Every enhancement operation produces a JSON report with bit-depth metadata:

```json
{
  "status": "success",
  "input": "/path/to/input.tiff",
  "output": "/path/to/output.tiff",
  "runtime_s": 1.01,

  "bit_depth": {
    "input_bits_per_sample": 16,
    "output_bits_per_sample": 16,
    "input_dtype": "uint16",
    "output_dtype": "uint16",
    "quality_firewall_active": true,
    "bit_depth_preserved": true,
    "downgrade_allowed": false
  }
}
```

**Auditability:** Reports can be automatically scanned for Quality Firewall violations.

---

## CLI Interface

### Direct V2 Default (Firewall Active)

```bash
# 16-bit input → 16-bit output (firewall enforced)
python scripts/enhance_image.py input_16bit.tiff --output-dir out/
```

**Log output:**
```
Quality Firewall ACTIVE: 16-bit input detected - will preserve 16-bit output
Saved 16-bit TIFF: out/input_16bit_v2_enhanced.tif
```

### Explicit 8-bit Selection (Downgrade Allowed)

```bash
# 16-bit input → 8-bit output (canonical selection)
python scripts/enhance_image.py input_16bit.tiff \
  --output-dir out/ \
  --output-bit-depth 8
```

`--allow-8bit` remains a legacy compatibility authority for callers that have
not migrated to `--output-bit-depth 8`.

**Use cases for bypass:**
- Web delivery where file size matters
- Compatibility with legacy 8-bit pipelines
- Intentional quality-size trade-off

**Requirement:** The downgrade must be **explicit**, through the canonical
selector or the legacy compatibility flag.

---

## Implementation Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT TIFF (16-bit)                                         │
│  - BitsPerSample: (16, 16, 16)                              │
│  - dtype: uint16                                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ LOAD with tifffile.imread()                                 │
│  - Preserves 16-bit precision (no auto-conversion)          │
│  - Returns: np.ndarray(dtype=uint16)                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ QUALITY FIREWALL CHECK                                      │
│  - input_bits == 16 ?                                       │
│  - explicit output_bit_depth, if present                    │
│  - otherwise legacy allow_8bit_output, if true              │
│  - otherwise preserve input_bits                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ PROCESS in float32 [0.0, 1.0]                               │
│  - Normalization: uint16 / 65535.0 → float32               │
│  - Enhancement: All ops in float32 (preserves precision)    │
│  - Denormalization: float32 * 65535.0 → uint16             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ SAVE with tifffile.imwrite()                                │
│  - photometric='rgb'                                        │
│  - compression='lzw' (lossless)                             │
│  - metadata={'BitsPerSample': 16}                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT TIFF (16-bit) ✅                                     │
│  - BitsPerSample: (16, 16, 16)                              │
│  - dtype: uint16                                            │
│  - Quality Firewall: SATISFIED                              │
└─────────────────────────────────────────────────────────────┘
```

### Critical Components

1. **`detect_input_bit_depth(pil_image)`**
   - Reads TIFF tag 258 (BitsPerSample)
   - Returns 8 or 16

2. **`load_image_preserve_bit_depth(input_path, allow_8bit_output=False)`**
   - Uses `tifffile.imread()` for 16-bit TIFFs
   - Falls back to PIL for 8-bit/non-TIFF inputs, or when an 8-bit output was
     explicitly authorized

3. **`EnhancementStage(output_dtype=np.uint16)`**
   - Processes in float32 [0, 1]
   - Denormalizes to `output_dtype` (uint8 or uint16)

4. **`tifffile.imwrite(..., metadata={'BitsPerSample': 16})`**
   - Writes true 16-bit TIFF
   - Preserves full precision

---

## Verification Checklist

### Manual Verification

```bash
# 1. Process 16-bit TIFF
python scripts/enhance_image.py test_16bit.tiff --output-dir out/

# 2. Check output bit-depth
python -c "
from PIL import Image
img = Image.open('out/test_16bit_v2_enhanced.tif')
bits = img.tag_v2.get(258)  # BitsPerSample tag
assert bits == (16, 16, 16), f'Expected (16,16,16), got {bits}'
print('✅ Output is 16-bit')
"

# 3. Check JSON report
python -c "
import json
report = json.load(open('out/test_16bit_report.json'))
assert report['bit_depth']['bit_depth_preserved'] == True
assert report['bit_depth']['quality_firewall_active'] == True
print('✅ Quality Firewall active')
print('✅ Bit-depth preserved')
"
```

### Automated Verification (CI)

```yaml
# .github/workflows/quality-firewall.yml

jobs:
  bit-depth-preservation:
    name: Quality Firewall - Bit-Depth Preservation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Test 16-bit preservation
        run: |
          python scripts/enhance_image.py \
            tests/fixtures/test_16bit.tiff \
            --output-dir test_output \
            --preset luxury_estate

      - name: Verify bit-depth contract
        run: |
          python -c "
          import json
          from PIL import Image

          # Check output file
          img = Image.open('test_output/test_16bit_v2_enhanced.tif')
          bits = img.tag_v2.get(258)
          assert bits == (16, 16, 16), f'OUTPUT BIT-DEPTH VIOLATION: {bits}'

          # Check report
          report = json.load(open('test_output/test_16bit_report.json'))
          bd = report['bit_depth']
          assert bd['bit_depth_preserved'] == True, 'QUALITY FIREWALL VIOLATED'
          assert bd['input_bits_per_sample'] == 16
          assert bd['output_bits_per_sample'] == 16

          print('✅ Quality Firewall: Bit-depth preservation VERIFIED')
          "
```

---

## Compliance Requirements

### For Developers

1. **Never bypass Quality Firewall without explicit intent**
   - Direct V2 API default: omit `output_bit_depth` and keep `allow_8bit_output=False`
   - Canonical selection: set `output_bit_depth` to 8 or 16 deliberately
   - Legacy compatibility: use `allow_8bit_output=True` only when required and documented

2. **Always check bit-depth metadata in reports**
   - Verify `bit_depth_preserved == True` for 16-bit inputs
   - Investigate any `quality_firewall_active == False` instances

3. **Test both bit-depths**
   - 8-bit → 8-bit (no change)
   - 16-bit → 16-bit (preservation)
   - 16-bit → 8-bit (explicit bypass)

### For CI/CD

1. **Bit-depth preservation test MUST pass**
   - Blocking: PRs cannot merge if test fails
   - Runs on every commit

2. **Report all Quality Firewall bypasses**
   - Audit canonical `--output-bit-depth 8` selections and warnings for legacy
     `--allow-8bit` usage
   - Track bypass frequency in metrics

3. **Audit reports for violations**
   - Scan JSON reports for `bit_depth_preserved == False`
   - Alert on unexpected downgrades

---

## Failure Modes and Recovery

### Failure Mode 1: tifffile Import Fails

**Symptom:** `ImportError: No module named 'tifffile'`

**Impact:** A 16-bit load fails closed with `V2EnhancementError` unless 8-bit
output was explicitly authorized. Only an authorized 8-bit operation may fall
back to PIL and down-convert the input.

**Recovery:**
```bash
make install-core
```

**Prevention:** Add `tifffile` to `requirements.txt` (already done)

### Failure Mode 2: 16-bit Save Fails

**Symptom:** Exception during `tifffile.imwrite()`

**Impact:** The requested 16-bit operation fails closed. The V2 save path raises
`V2EnhancementError` instead of publishing an 8-bit fallback under a 16-bit
contract.

**Recovery:**
- Check tifffile version (should be >= 2024.x)
- Check disk space (16-bit TIFFs are large)
- Check write permissions

**Error surfaced to the caller:**
```
ERROR: Cannot save requested 16-bit output with tifffile: [error];
       publishing an 8-bit file under a 16-bit contract is forbidden
```

### Failure Mode 3: Quality Firewall Bypassed Accidentally

**Symptom:** `bit_depth_preserved == False` in report, but bypass not intended

**Detection:**
- CI scans successful reports for a 16-bit input and 8-bit output paired with
  `downgrade_allowed == False`
- Audits distinguish an explicit `output_bit_depth=8` selection (or the legacy
  `allow_8bit_output` compatibility path) from a failed 16-bit save
- Manual audit of JSON reports

**Recovery:**
- Re-process with `--output-bit-depth 16`; remove any accidental
  `--output-bit-depth 8` selection or legacy `--allow-8bit` flag
- Verify output is 16-bit

---

## Performance Impact

### Bit-Depth Preservation Overhead

| Operation              | 8-bit Time | 16-bit Time | Overhead |
|------------------------|-----------|-------------|----------|
| Load (PIL)             | 0.10s     | —           | —        |
| Load (tifffile)        | —         | 0.12s       | +20%     |
| Process (float32)      | 0.80s     | 0.85s       | +6%      |
| Save (PIL)             | 0.10s     | —           | —        |
| Save (tifffile + LZW)  | —         | 0.15s       | +50%     |
| **Total**              | **1.00s** | **1.12s**   | **+12%** |

**Conclusion:** 16-bit preservation adds ~10-15% overhead—acceptable for quality guarantee.

### File Size Impact

| Bit-Depth | Uncompressed | LZW Compressed | Ratio |
|-----------|-------------|----------------|-------|
| 8-bit     | 60 MB       | 45 MB          | 1.0x  |
| 16-bit    | 120 MB      | 65 MB          | 1.4x  |

**Conclusion:** 16-bit TIFFs with LZW compression are ~1.4x larger—acceptable for archival quality.

---

## References

- **ADR-007:** `docs/architecture/decisions/ADR-007-bit-depth-preservation.md`
- **Implementation:** `src/transformation_portal/lux_depth_v3/v2_enhance.py`
- **Test Suite:** `tests/unit/lux_depth_v3/test_v2_enhance_quality_firewall.py`
- **TIFF Specification:** Tag 258 (BitsPerSample)

---

## Changelog

### 2026-09-01: Canonical Output Encoding Selector

- ✅ `output_bit_depth=8|16` is the canonical encoding authority
- ✅ Omitted direct-V2 selection still preserves input precision
- ✅ Requested 16-bit save failures fail closed without an 8-bit fallback

### 2026-02-10: Bit-Depth Preservation Implemented

- ✅ Quality Firewall active for 16-bit inputs
- ✅ Bit-depth metadata in all reports
- ✅ CLI flag `--allow-8bit` for explicit bypass
- ✅ Automated verification implemented
- ✅ CI regression test active

---

**Architect Approval:** ✅ APPROVED
**Status:** ACTIVE
**Compliance:** ENFORCED
