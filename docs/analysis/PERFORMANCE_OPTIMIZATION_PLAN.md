# Performance Optimization Plan

## Status

Phase 1 is partially implemented as a bounded parallel I/O primitive and an
explicit opt-in path for the compatibility Unified Luxury Pipeline.

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

Not implemented:

- a new public `--parallel-io` CLI flag
- default-on behavior for existing batch callers
- benchmark gate changes
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

Minimum local measurement command shape:

```bash
.venv/bin/python -m pytest tests/test_parallel_io.py tests/test_unified_luxury_pipeline.py -q
```

For real throughput measurement, add or reuse a benchmark that writes a non-empty
artifact with a schema that the consuming workflow validates. Do not treat a
green workflow as performance evidence unless the artifact contains real
measurements.

## Remaining Work

1. Add a dedicated benchmark harness for Unified Luxury Pipeline batch I/O.
2. Evaluate whether `parallel_io=True` should become the default after measured
   evidence and memory limits are understood.
3. Reassess the recipe pipeline and TIFF batch processor for shared reuse of
   `ParallelIOPipeline`; both already have existing parallel execution controls,
   so changes should be evidence-driven.
4. Keep GPU/backend acceleration work separate from I/O scheduling; it has
   different dependencies, hardware assumptions, and failure modes.

## Validation

Focused validation for this phase:

```bash
.venv/bin/pytest tests/test_parallel_io.py tests/test_unified_luxury_pipeline.py -q
git diff --check
```

Broader validation should include `make ci-quick` before merging into `main`.
