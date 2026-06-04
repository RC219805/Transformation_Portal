# Performance Optimization Plan

## Status

Phase 1 is partially implemented as a bounded parallel I/O primitive and an
explicit opt-in path for the compatibility Unified Luxury Pipeline. Phase 1.B
adds the dedicated advisory benchmark harness needed to evaluate whether that
opt-in path should ever become the default.

Implemented:

- `transformation_portal.pipelines.parallel_io.ParallelIOPipeline`
- deterministic per-item results for `load`, `process`, and `save` failures
- bounded prefetch and save queues
- background input loading and background output writing
- TIFF inputs use a `tifffile` pixel-load path with PIL fallback
- explicit Unified Luxury Pipeline config controls:
  - `parallel_io`
  - `io_prefetch_size`
  - `io_saver_workers`
- focused unit coverage for ordering, failure isolation, and background-save
  overlap
- `tools/benchmark_unified_luxury_batch_io.py`
- `make benchmark-unified-luxury-batch-io`
- advisory benchmark documentation at
  `docs/performance/unified_luxury_batch_io_benchmark.md`

Not implemented:

- a new public `--parallel-io` CLI flag
- default-on behavior for existing batch callers
- benchmark gate changes or committed performance baselines
- GPU/backend acceleration changes
- memory-mapped TIFF loading

The CLI flag from the earlier draft was intentionally not added because there
is no active Unified Luxury Pipeline CLI surface. The live CLIs already expose
their own parallel controls: `transformation-portal process --parallel` for the
recipe pipeline and `luxury_tiff_batch_processor --workers` for the TIFF batch
processor. Adding a new flag without a live command would create documentation
drift rather than an operator feature.

## Current Bottleneck Hypothesis

Large TIFF and multi-format finishing jobs can spend a meaningful share of wall
time in image load and output writes. The highest-leverage safe change is to
overlap I/O with compute without changing output formats, color transforms,
metadata handling, or existing CLI defaults.

Earlier drafts claimed specific throughput targets such as 30-50 images/minute.
Those numbers should be treated as hypotheses until measured on a current
fixture set with real benchmark artifacts. Prior performance audits found that
green benchmark workflows can still produce empty or schema-incompatible
benchmark output, so optimization claims must be backed by inspected benchmark
JSON and logs.

## Implemented Architecture

`ParallelIOPipeline` is a generic producer/consumer helper:

```text
input paths
  -> loader thread and bounded prefetch queue
  -> caller-thread processor
  -> bounded save queue and background saver workers
  -> ordered per-input results
```

The primitive is intentionally independent of PIL, tifffile, model runtimes, and
pipeline-specific state. Callers supply the loader, processor, and saver
functions. This keeps it reusable for future TIFF, depth, and export workflows
without coupling the helper to one image representation.

Failure behavior is fail-isolated:

- load failures produce a failed item with `stage="load"`
- process failures produce a failed item with `stage="process"`
- save failures produce a failed item with `stage="save"`
- other inputs continue processing
- returned results preserve the original input order

## Unified Luxury Pipeline Integration

The compatibility Unified Luxury Pipeline now supports explicit batch I/O
overlap:

```python
from pathlib import Path

from transformation_portal.pipelines import (
    OutputFormat,
    ProcessingProfile,
    UnifiedLuxuryPipeline,
    UnifiedPipelineConfig,
)

config = UnifiedPipelineConfig(
    profile=ProcessingProfile.BALANCED,
    output_dir=Path("output"),
    output_formats=[OutputFormat.MASTER_TIFF, OutputFormat.WEB_4K],
    parallel_io=True,
    io_prefetch_size=2,
    io_saver_workers=2,
)

pipeline = UnifiedLuxuryPipeline(config)
results = pipeline.batch_process(list(Path("renders").glob("*.tif")))
```

The convenience function exposes the same controls:

```python
from pathlib import Path

from transformation_portal.pipelines import batch_process_luxury_renders

results = batch_process_luxury_renders(
    input_dir=Path("renders"),
    output_dir=Path("output"),
    parallel_io=True,
    io_prefetch_size=2,
    io_saver_workers=2,
)
```

Existing behavior is preserved because `parallel_io` defaults to `False`.

## Why The Integration Is Opt-In

The Unified Luxury Pipeline keeps mutable per-call stage state for timing and
graceful-degradation reporting. The new batch path avoids cross-thread mutation
by running load and compute in a single ordered path and deferring only output
generation to background saver workers. That is safe and testable, but it is a
contract-visible execution-mode change, so it remains explicit rather than
default-on.

Default-on optimization should wait for:

- benchmark evidence on representative TIFF and JPEG batches
- memory profiling for high-resolution batches
- confirmation that optional depth/material stages do not regress due to
  changed I/O scheduling
- confirmation that the `tifffile` loader improves wall time on the target
  storage medium versus the PIL fallback
- production operator approval for changed batch timing and logging semantics

## Benchmark Requirements

Any future performance claim should include:

- fixture set name and image dimensions
- storage medium and host hardware
- pipeline profile and enabled stages
- prefetch and saver worker settings
- wall time and images/minute
- p50/p95 per-image latency
- peak RSS
- output quality/hash comparison when deterministic output is expected
- benchmark artifact path and schema verification
- whether the benchmark input set was explicitly marked representative with
  `--representative-input-set`

Minimum local measurement command shape:

```bash
.venv/bin/python -m pytest tests/test_parallel_io.py tests/test_unified_luxury_pipeline.py tests/test_unified_luxury_batch_io_benchmark.py -q
```

For real throughput measurement, use the dedicated harness:

```bash
UNIFIED_LUXURY_BATCH_IO_BENCHMARK_ARGS="\
  --input-dir /path/to/representative/tiffs \
  --output-json /tmp/unified-luxury-batch-io.json \
  --runs 5 \
  --warmup-runs 1 \
  --output-formats master \
  --memory-limit-mib 4096 \
  --representative-input-set" \
make benchmark-unified-luxury-batch-io
```

Do not treat a green workflow as performance evidence unless the artifact
contains real measurements under schema
`tp.unified_luxury.batch_io_benchmark.v1`.

## Current Evaluation

The default remains `parallel_io=False`.

Local fixture evidence from
`tests/fixtures/pipelines/750_picacho_lane/input` exercises two 800x600 TIFF
files, `output_formats=master`, `runs=3`, `warmup_runs=1`, and
`--memory-limit-mib 4096`. That run produced a non-empty
`tp.unified_luxury.batch_io_benchmark.v1` artifact with:

- serial mean wall time: 0.0209056663s
- `parallel_io=True` mean wall time: 0.0132810000s
- measured mean speedup: 1.5741033222x
- `parallel_io=True` peak RSS: 92.703125 MiB
- failures: 0

This is useful smoke evidence, but it is not enough to change the default
because the input set is only two small fixtures and was not marked
representative. A default flip still requires a larger production-sized batch,
an operator-approved memory limit for the target host class, and review of the
resulting benchmark artifact.

The recipe pipeline and TIFF batch processor should not be switched to
`ParallelIOPipeline` by default. The recipe pipeline already exposes
`parallel=True` through isolated `ThreadPoolExecutor` worker pipeline instances,
and the TIFF batch processor already exposes `--workers` through
`ProcessPoolExecutor`. Shared reuse should be revisited only after separate
benchmarks prove load/save overlap beats those existing execution controls
without changing recipe/RAG semantics or TIFF processing isolation.

GPU/backend acceleration remains out of scope for this I/O scheduling lane. It
belongs in backend-specific benchmark work because dependencies, hardware
assumptions, memory pressure, and failure modes differ from load/save overlap.

## Validation

Focused validation for this phase:

```bash
.venv/bin/pytest tests/test_parallel_io.py tests/test_unified_luxury_pipeline.py -q
.venv/bin/pytest tests/test_unified_luxury_batch_io_benchmark.py -q
make benchmark-unified-luxury-batch-io
git diff --check
```

Broader validation should include `make ci-quick` before merging into `main`.
