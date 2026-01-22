# Depth Zoning Integration

## Purpose
Depth zoning is a diagnostics-first preprocessing layer that converts a depth map into **four soft zone masks** (Z1–Z4).
It is designed to support future depth-conditioned operators while being immediately useful for QA, debugging, and audit trails.

## What it is (today)
- Generates zone masks and stats from **DA3 depth maps**.
- Produces artifacts:
  - `{stem}_zones_Z1.png` … `{stem}_zones_Z4.png` (16-bit)
  - `{stem}_zones_preview.png` (visual overlay)
  - `{stem}_zone_stats.json` (coverage, thresholds, depth convention, valid coverage)

## What it is not (yet)
- It does **not** apply photometric operators (sharpening/clarity/exposure).
- It does **not** guarantee improved final renders until zone-conditioned operators are implemented and validated.

## Canonical workflow (Option 2)
This repository's canonical DA3 path is the **script → depth folder → V2 consume** pattern.

### Step A — Generate DA3 depth maps
```bash
make da3-depth
```

### Step B — Generate depth zones (diagnostics only)
```bash
PYTHONPATH="$PWD" python scripts/da3_depth_zones.py \
  --depth-dir depth_da3_run4 \
  --output-dir zones_da3_run4 \
  --input-dir renders_safe
```

Artifacts produced:
- `zones_da3_run4/{stem}_zones_Z1.png` … `Z4.png`
- `zones_da3_run4/{stem}_zones_preview.png`
- `zones_da3_run4/{stem}_zone_stats.json`

### Step C — Run V2 enhancement (unchanged)
```bash
make da3-v2
```

**Note**: Depth zones are not consumed by V2 yet. They exist for:
- QA and debugging
- depth validation
- future Phase 4 operator conditioning

## Architecture guardrail
Zone masks are intentionally **not wired into V2** until:
1. Zone-conditioned operators are implemented
2. Photometric impact is validated on production data
3. Visual improvement is benchmarked and documented

This ensures diagnostics infrastructure is battle-tested before runtime dependency.
