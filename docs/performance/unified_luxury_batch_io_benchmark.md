# Unified Luxury Batch I/O Benchmark

This benchmark measures the Unified Luxury Pipeline batch I/O scheduler:

- serial batch processing with `parallel_io=False`
- overlapped input load and output save with `parallel_io=True`

It is intentionally CPU-only by default. GPU/backend acceleration work must stay
in separate benchmark lanes because it has different dependencies, hardware
assumptions, and failure modes.

## Run

Use the Make target:

```bash
make benchmark-unified-luxury-batch-io
```

Pass harness arguments through `UNIFIED_LUXURY_BATCH_IO_BENCHMARK_ARGS`:

```bash
UNIFIED_LUXURY_BATCH_IO_BENCHMARK_ARGS="\
  --input-dir /path/to/representative/tiffs \
  --output-json /tmp/unified-luxury-batch-io.json \
  --runs 5 \
  --warmup-runs 1 \
  --output-formats master \
  --io-prefetch-size 2 \
  --io-saver-workers 2 \
  --memory-limit-mib 4096 \
  --representative-input-set" \
make benchmark-unified-luxury-batch-io
```

When `--input-dir` is omitted, the harness creates deterministic synthetic
fixtures. Synthetic runs are useful for smoke checks, but they are not enough to
change defaults.

The harness writes a JSON report with schema
`tp.unified_luxury.batch_io_benchmark.v1`. The report includes per-trial wall
time, summary percentiles, peak RSS when `psutil` is installed, the default
candidate decision, and reuse notes for adjacent pipelines.

## Default Policy

Keep `UnifiedPipelineConfig.parallel_io` defaulted to `False` until measured
evidence proves a default flip is safe for representative production batches.

Minimum evidence for a default flip:

- serial and `parallel_io=True` runs use the same input set and output formats
- representative images include production-sized TIFFs, not only synthetic
  fixtures
- mean speedup is at least the configured `--min-speedup` threshold
- p95 wall time does not regress materially
- `--memory-limit-mib` is set to the operator limit for the target host class
- `--representative-input-set` is passed only after the input set is large and
  varied enough to represent the target production workload
- peak RSS with `parallel_io=True` stays below that memory limit
- no per-image failures, ordering changes, or output-count changes occur

If any item is missing, the report keeps the decision at `keep_false`. If all
items pass, treat `candidate_after_representative_runs` as review evidence, not
an automatic code change.

## Adjacent Pipelines

Do not automatically reuse `ParallelIOPipeline` in the recipe pipeline or TIFF
batch processor.

The recipe pipeline already supports `parallel=True` through isolated
`ThreadPoolExecutor` worker pipeline instances. That model overlaps whole-image
work and has recipe/RAG indexing semantics that differ from load/save overlap.
Reassess it only with a dedicated benchmark proving lower wall time and stable
recipe state.

The TIFF batch processor already supports `--workers` through
`ProcessPoolExecutor`. That model isolates per-image TIFF processing and has a
different memory profile from threaded save workers. Reassess it only with
TIFF-specific measurements against representative source files.

## Out Of Scope

This harness does not evaluate:

- depth backend selection
- GPU, MPS, CUDA, CoreML, or TensorRT acceleration
- model download, warmup, or inference scheduling
- quality metric changes

Those concerns should remain in backend-specific benchmark lanes.
