# Validation

The validation UI story is:

1) Curate a dataset pack (e.g. `validation_v1`) under:
- `data/benchmark_datasets/<dataset_id>/input/`

2) Generate commercial baselines (Topaz / Adobe) under:
- `data/benchmark_datasets/<dataset_id>/baselines/<baseline_id>/`

3) Generate a manifest with SHA256 hashes for reproducibility.

4) Run benchmark validation:
- **Synthetic reference mode** (for reference-based metrics)
- **Real-world mode** (for aesthetic and no-reference scoring)

Metrics are reported separately for runtime weighting:
- Fidelity: SSIM / PSNR
- Perceptual: LPIPS
- Aesthetic: NIMA

