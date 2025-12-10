# Marketing Export M0+M1.1 Implementation Checklist

**Target**: PR1 - Instrumentation + PNG Compression Tuning  
**Estimated Effort**: 2-3 days  
**Expected Impact**: ~20-30s savings (hypothesis, to be validated)

---

## 🎯 Scope: M0 (Instrumentation) + M1.1 (PNG Levels)

**Goal**: Get precise marketing export metrics, then find optimal PNG compression level

**Out of Scope for PR1**:
- WebP/JPEG formats (PR2)
- Async marketing (PR3)
- Marketing autotune (optional)

---

## ✅ Implementation Checklist

### Step 1: Centralize Marketing Write Path (M0)

**File**: `src/transformation_portal/core/storage/export_manager.py`

- [ ] Ensure `write_marketing()` is the ONLY entry point for marketing PNG writes
- [ ] Check for any direct PIL.save() calls in pipeline that bypass ExportManager
- [ ] Document the centralized path in docstring

**Validation**:
```bash
# Search for bypass paths
grep -r "\.save.*marketing" src/ lux_depth_v2/
grep -r "marketing.*PIL" src/ lux_depth_v2/
```

---

### Step 2: Add Marketing Metadata to Reports (M0)

**File**: `src/transformation_portal/core/storage/export_manager.py`

**Add to `write_marketing()`**:
```python
def write_marketing(self, img: np.ndarray, stem: str) -> Path:
    import time
    import psutil
    
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=None)
    
    # Existing write logic here
    path = self._write_marketing_png(img, stem, compression_level)
    
    elapsed = time.time() - start_time
    cpu_end = psutil.cpu_percent(interval=None)
    file_size = path.stat().st_size if path.exists() else 0
    
    # Store metadata for report
    self._last_marketing_metadata = {
        "encoder": "png",
        "compression_level": compression_level,
        "width": img.shape[1],
        "height": img.shape[0],
        "bytes_written": file_size,
        "write_time_s": elapsed,
        "cpu_percent_delta": cpu_end - cpu_start,
    }
    
    return path
```

**Update report in pipeline**:
```python
# In pipeline.py process_one()
report["marketing_export"] = self.export_manager.get_marketing_metadata()
```

**Checklist**:
- [ ] Add `_last_marketing_metadata` attribute to ExportManager
- [ ] Add `get_marketing_metadata()` method
- [ ] Wire into pipeline report generation
- [ ] Test: verify metadata appears in reports

---

### Step 3: Add MarketingExportConfig (M1.1)

**File**: `lux_depth_v2/config.py` or `src/transformation_portal/core/storage/export_manager.py`

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class MarketingExportConfig:
    """Marketing export configuration (M1 encoding strategy)."""
    
    # PNG settings
    format: Literal["png"] = "png"  # WebP/JPEG in M1.2
    png_compression_level: int = 6  # 1-9, default 6
    
    # Future: WebP/JPEG (M1.2)
    # webp_quality: int = 90
    # jpeg_quality: int = 95
```

**Add to ExportConfig**:
```python
@dataclass
class ExportConfig:
    # ... existing fields ...
    
    # Marketing export settings
    marketing: MarketingExportConfig = field(default_factory=MarketingExportConfig)
```

**Checklist**:
- [ ] Add MarketingExportConfig dataclass
- [ ] Wire into ExportConfig
- [ ] Update ExportManager.__init__ to accept marketing config
- [ ] Test: config instantiation and defaults

---

### Step 4: Add CLI Flag for PNG Compression (M1.1)

**File**: `lux_depth_v2/cli.py`

```python
# In Phase 2 argument group (or new "Marketing Export" group)
phase2_group.add_argument(
    "--marketing-png-compression",
    type=int,
    default=6,
    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="PNG compression level for marketing export (1=fast/larger, 9=slow/smaller, default=6)."
)
```

**Wire to config**:
```python
# In build_config() or equivalent
cfg.export_config.marketing.png_compression_level = args.marketing_png_compression
```

**Checklist**:
- [ ] Add CLI argument
- [ ] Wire to config building
- [ ] Test: `lux-depth-v2 --marketing-png-compression 1 ...` works
- [ ] Update help text

---

### Step 5: Update Analysis Script (M0)

**File**: `scripts/analyze_autotune_production.py` → `scripts/analyze_marketing_export.py`

```python
#!/usr/bin/env python3
"""Analyze marketing export performance across different settings."""

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any

def analyze_marketing(reports: List[Dict[str, Any]]) -> None:
    """Analyze marketing export metrics with median-based comparison."""
    
    # Group by encoder and compression level
    by_setting = {}  # {(encoder, level): [times]}
    
    for report in reports:
        mkt = report.get("marketing_export", {})
        encoder = mkt.get("encoder", "unknown")
        level = mkt.get("compression_level", "unknown")
        time = mkt.get("write_time_s", 0)
        size = mkt.get("bytes_written", 0)
        cpu = mkt.get("cpu_percent_delta", 0)
        
        key = (encoder, level)
        if key not in by_setting:
            by_setting[key] = {"times": [], "sizes": [], "cpus": []}
        
        by_setting[key]["times"].append(time)
        by_setting[key]["sizes"].append(size)
        by_setting[key]["cpus"].append(cpu)
    
    # Print median-based comparison
    print("=" * 80)
    print("MARKETING EXPORT ANALYSIS (Median-Based)")
    print("=" * 80)
    print()
    
    for (encoder, level), data in sorted(by_setting.items()):
        times = data["times"]
        sizes = data["sizes"]
        cpus = data["cpus"]
        
        if not times:
            continue
        
        median_time = statistics.median(times)
        median_size = statistics.median(sizes)
        median_cpu = statistics.median(cpus)
        
        p75_time = statistics.quantiles(times, n=4)[2] if len(times) >= 4 else median_time
        p95_time = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else median_time
        
        print(f"{encoder} level {level}:")
        print(f"  Time (median): {median_time:.1f}s  [p75: {p75_time:.1f}s, p95: {p95_time:.1f}s]")
        print(f"  Size (median): {median_size / 1024 / 1024:.1f} MB")
        print(f"  CPU delta (median): {median_cpu:.1f}%")
        print(f"  N samples: {len(times)}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    
    reports = []
    for output_dir in args.output_dirs:
        for report_path in output_dir.glob("**/*_report.json"):
            with open(report_path) as f:
                reports.append(json.load(f))
    
    analyze_marketing(reports)
```

**Checklist**:
- [ ] Create `scripts/analyze_marketing_export.py`
- [ ] Test with existing reports (should handle missing fields gracefully)
- [ ] Make executable: `chmod +x scripts/analyze_marketing_export.py`

---

### Step 6: Run PNG Compression Benchmarks (M1.1)

**Benchmark Matrix**:
```bash
# Test images: Pool, Aerial, GreatRoom (5+ per category if available)

# Level 1 (fast)
lux-depth-v2 --input input_images/750_Picacho/Pool.tif \
  --output-dir benchmark_png1/pool/ \
  --marketing-png-compression 1

# Level 3 (balanced)
lux-depth-v2 --input input_images/750_Picacho/Pool.tif \
  --output-dir benchmark_png3/pool/ \
  --marketing-png-compression 3

# Level 6 (default)
lux-depth-v2 --input input_images/750_Picacho/Pool.tif \
  --output-dir benchmark_png6/pool/ \
  --marketing-png-compression 6

# Level 9 (max compression)
lux-depth-v2 --input input_images/750_Picacho/Pool.tif \
  --output-dir benchmark_png9/pool/ \
  --marketing-png-compression 9

# Repeat for Aerial and GreatRoom
# Run 3 times each for variance

# Analyze
python scripts/analyze_marketing_export.py benchmark_png*/
```

**Checklist**:
- [ ] Run benchmarks for all levels (1, 3, 6, 9)
- [ ] Run on Pool, Aerial, GreatRoom (minimum 5 images per category)
- [ ] 3 runs each if feasible
- [ ] Capture: median time, p75/p95, size, CPU delta
- [ ] Visual verification (all should look identical - PNG is lossless)

---

### Step 7: Choose Optimal Defaults Per Preset (M1.1)

**Decision Matrix** (to be filled after benchmarks):

| Preset | Chosen Level | Rationale |
|--------|--------------|-----------|
| interior_luxury | TBD | (e.g., "Level 3: 25s faster than 6, +5% size, acceptable") |
| exterior_showcase | TBD | (e.g., "Level 1: 30s faster than 6, +10% size, acceptable") |
| architectural | TBD | (e.g., "Level 3: balanced for detailed scenes") |

**Update config defaults**:
```python
# In config.py or preset definitions
PRESET_MARKETING_CONFIGS = {
    Preset.INTERIOR_LUXURY: MarketingExportConfig(
        png_compression_level=3,  # Based on benchmarks
    ),
    Preset.EXTERIOR_SHOWCASE: MarketingExportConfig(
        png_compression_level=1,  # Fastest acceptable
    ),
    # etc.
}
```

**Checklist**:
- [ ] Analyze benchmark data (median times, sizes, CPU)
- [ ] Choose level per preset (document rationale)
- [ ] Update preset defaults in code
- [ ] Document decision in MARKETING_ENCODING_BENCHMARKS.md (new file)

---

### Step 8: Documentation (M0+M1.1)

**Create**: `docs/guides/MARKETING_ENCODING_BENCHMARKS.md`

```markdown
# Marketing Export Encoding Benchmarks

## PNG Compression Levels (M1.1)

Tested on: [Date]
Images: Pool, Aerial, GreatRoom (5+ per category, 3 runs each)

### Results (Median Times):

| Level | Pool Time | Aerial Time | GreatRoom Time | Size Impact | CPU Impact |
|-------|-----------|-------------|----------------|-------------|------------|
| 1     | [TBD]s    | [TBD]s      | [TBD]s         | +[TBD]%     | [TBD]%     |
| 3     | [TBD]s    | [TBD]s      | [TBD]s         | +[TBD]%     | [TBD]%     |
| 6     | [TBD]s    | [TBD]s      | [TBD]s         | baseline    | baseline   |
| 9     | [TBD]s    | [TBD]s      | [TBD]s         | -[TBD]%     | +[TBD]%    |

### Chosen Defaults:
- interior_luxury: Level [X] (rationale)
- exterior_showcase: Level [X] (rationale)
- architectural: Level [X] (rationale)

### Visual Quality:
All PNG levels are lossless - images identical. Choice based on time/size trade-off only.
```

**Update**: `docs/guides/MARKETING_EXPORT_OPTIMIZATION_PLAN.md`
- [ ] Mark M0 and M1.1 as ✅ COMPLETE
- [ ] Add link to benchmark results

**Update**: `README.md`
```markdown
### Marketing Export Configuration

Control PNG compression for faster exports:

```bash
# Fast export (larger files)
lux-depth-v2 --input image.tif --output-dir out/ --marketing-png-compression 1

# Balanced (default)
lux-depth-v2 --input image.tif --output-dir out/ --marketing-png-compression 3

# Max compression (slower)
lux-depth-v2 --input image.tif --output-dir out/ --marketing-png-compression 9
```

Benchmarks show level 1-3 saves 20-30s with acceptable size increase.
```

**Checklist**:
- [ ] Create MARKETING_ENCODING_BENCHMARKS.md with results
- [ ] Update MARKETING_EXPORT_OPTIMIZATION_PLAN.md status
- [ ] Add README section for marketing flags
- [ ] Update CLI help text

---

## 🧪 Testing & Validation

### Unit Tests

**File**: `tests/core/storage/test_marketing_export.py` (new)

```python
def test_marketing_metadata_captured(tmp_path):
    """Verify marketing metadata is captured in reports."""
    cfg = ExportConfig(output_dir=tmp_path)
    manager = ExportManager(cfg)
    
    img = np.random.rand(100, 100, 3).astype(np.float32)
    path = manager.write_marketing(img, "test")
    
    metadata = manager.get_marketing_metadata()
    
    assert metadata["encoder"] == "png"
    assert metadata["compression_level"] == 6  # default
    assert metadata["bytes_written"] > 0
    assert metadata["write_time_s"] > 0

def test_png_compression_levels(tmp_path):
    """Verify different compression levels produce different sizes."""
    img = np.random.rand(1000, 1000, 3).astype(np.float32)
    
    sizes = {}
    for level in [1, 6, 9]:
        cfg = ExportConfig(
            output_dir=tmp_path,
            marketing=MarketingExportConfig(png_compression_level=level)
        )
        manager = ExportManager(cfg)
        path = manager.write_marketing(img, f"test_level{level}")
        sizes[level] = path.stat().st_size
    
    # Level 1 should be larger than 6, which should be larger than 9
    assert sizes[1] > sizes[6] > sizes[9]
```

**Checklist**:
- [ ] Create test file with 5+ test cases
- [ ] Test metadata capture
- [ ] Test compression level variation
- [ ] Test CLI flag parsing
- [ ] All tests pass

---

### Integration Tests

**Manual validation checklist**:
- [ ] Run pipeline with each PNG level (1, 3, 6, 9)
- [ ] Verify marketing PNG is created
- [ ] Verify report contains marketing_export metadata
- [ ] Verify file sizes differ as expected
- [ ] Verify images are visually identical (lossless)
- [ ] Verify no regressions in master/upscaled TIFFs

---

## 📈 Success Criteria (PR1 Acceptance)

- [ ] ✅ All marketing writes go through ExportManager.write_marketing()
- [ ] ✅ Reports contain marketing_export metadata (encoder, compression, size, time, CPU)
- [ ] ✅ CLI flag --marketing-png-compression works (1-9)
- [ ] ✅ Benchmarks run on 5+ images per category (Pool/Aerial/GreatRoom), 3 runs each
- [ ] ✅ Analysis uses **median** times (not mean)
- [ ] ✅ Optimal level chosen per preset with documented rationale
- [ ] ✅ Preset defaults updated in code
- [ ] ✅ Documentation complete (benchmarks, README, plan update)
- [ ] ✅ Tests pass (unit + integration)
- [ ] ✅ No regressions (master/upscaled unchanged)

---

## 🚀 Expected Impact

**Hypothesis**: Level 1-3 saves ~20-30s vs level 6-9 with acceptable size increase

**Validation**: Will be confirmed by median benchmark results

**Next PR**: M1.2 (WebP/JPEG) for additional 40-60s savings (hypothesis)

---

## 📝 Git Commit Strategy

**Branch**: `feature/marketing-export-m0-m1`

**Commits**:
1. `feat: Centralize marketing write path and add metadata`
2. `feat: Add MarketingExportConfig with PNG compression control`
3. `feat: Add --marketing-png-compression CLI flag`
4. `feat: Add marketing export analysis script`
5. `docs: Add PNG compression benchmark results`
6. `chore: Update preset defaults based on benchmarks`

**PR Title**: `feat: Marketing export instrumentation + PNG compression tuning (M0+M1.1)`

---

**Status**: ✅ **READY TO IMPLEMENT**  
**Next Action**: Start with Step 1 (centralize marketing write path)
