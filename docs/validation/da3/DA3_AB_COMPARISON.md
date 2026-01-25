# DA3 A/B Comparison

**A (placeholder)**: `output_v3_zones_eval` (legacy local evaluation outputs)

**B (model backend)**: local evaluation directory (`model_backend_eval/`, not committed)

> Note: Raw evaluation artifacts (depth maps, zone images, manifests) are intentionally
> excluded from version control. The CSV and this summary are the source of truth.

## Depth scaling (mean across images)
| Metric | A | B |
|---|---:|---:|
| valid_coverage_pct | 100.00 | 100.00 |
| p10 | 0.2025 | 0.0471 |
| p90 | 0.6688 | 0.7192 |
| clip_low_frac | 0.0000 | 0.0000 |
| clip_high_frac | 0.0000 | 0.0000 |
| invalid_frac | 0.0000 | 0.0000 |

## Zone distribution (mean %)
| Zone | A | B |
|---|---:|---:|
| Z1 | 5.66 | 10.00 |
| Z2 | 25.25 | 25.01 |
| Z3 | 24.35 | 29.99 |
| Z4 | 44.74 | 35.00 |
