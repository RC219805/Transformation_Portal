# Unified LuxDepth

`transformation_portal.lux_depth` is the common operator API and `lux-depth` is
its CLI. It composes the existing governed engines: V4 ingest/runtime controls,
V5 native depth and Materials evidence, and V6 reconstruction, linear grading,
display rendering, and independent replay. V3 remains the production default
and rollback implementation. Consolidation does not establish photographic,
performance, hosted, or production acceptance.

## Choose a workflow

| Command | Input and behavior | Native authority |
| --- | --- | --- |
| `process` | Original photographs through DA3 Metric, retained V5 evidence, then V6 finishing and depth products | `tp.execution.plan.v5` |
| `infer` | Original photographs through V5; optional source-bound MaterialsV4 evidence and policy | `tp.execution.plan.v4` |
| `finish` | Complete verified V5 generation through V6; depth products enabled by default | `tp.lux.grade.plan.v2`; explicit `--no-depth-maps` selects V1 |
| `depth-pro` | Original JPEG/PNG/TIFF through explicitly acknowledged non-commercial Depth Pro | `tp.lux.depth_pro.plan.v1` |
| `verify` | Detect the recorded bounded native plan and invoke its independent verifier | No inference or new execution authority |
| `legacy` | Forward remaining arguments to the existing V3 CLI | Unchanged V3 contract |

Versioned commands and Python import paths remain supported. No plan, artifact,
cache, or evidence schema is relabeled. The new API also accepts exact V4
request/prepared carriers for compatibility, including verification of plan V2
and V3 generations. V3 verification continues through its existing evidence
tools; the unified replay command does not claim V6 semantic replay for V3.

The [dated forensic audit](../analysis/LUX_DEPTH_UNIFICATION_AUDIT_2026-09-26.md)
records the capability dispositions and review scope. Engine details remain in
the [V5 guide](LUX_DEPTH_V5.md), [V6 guide](LUX_DEPTH_V6.md), and
[MaterialsV4 guide](../guides/MATERIALS_V4.md).

## Plan and run original photographs

From the repository root, use the supported core environment and a governed
DA3 runtime installed with `./scripts/setup/install_da3_runtime.sh --profile baseline`.
Choose a new output directory whose parent exists. Input, output, and cache
roots must be disjoint. Preparation reads and hashes inputs but creates no
output and loads no model. Every request explicitly selects a workflow; there
is no fallback to another backend on failure.

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth process \
  --input-dir /absolute/photographs --output-dir /absolute/output/lux \
  --input-color srgb --device mps --precision fp32 --plan

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth process \
  --input-dir /absolute/photographs --output-dir /absolute/output/lux \
  --input-color srgb --device mps --precision fp32

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth verify \
  --output-dir /absolute/output/lux
```

Defaults are CPU, FP32, target size 518, and V6 `guided_bilinear_v4`
reconstruction. `--target-size 1008` is an explicit experiment; it is not a
quality-accepted default. `--runtime-python`, `--raw-python`, `--cache-dir`,
and `--companions-manifest` use the V5 admission rules. `auto` color requires
recognized source color metadata; use an explicit color assertion only when
it describes the actual input. RAW needs its separately governed decoder.
Convert unsupported profiles to sRGB with a profile-aware editor before upload.
If `auto` still rejects the exported sRGB profile, select `srgb` only for those
converted copies; the input-color option does not convert pixels.

`process` produces `source-v5/`, `v6/`, `execution-plan.json`, and
`execution-evidence.json`. It retains both complete stage inventories. Final
products include the float master, 16-bit sRGB TIFF, draft PNG, native depth,
reconstructed relative depth, support/validity masks, and replay receipts.
Model estimates, calibrated maps, relative finishing maps, and previews retain
distinct meanings. Missing sky/confidence or unsupported surfaces abstain from
depth edits. Reconstructed detail is not newly measured geometry.

The composite uses the same frozen file/count/manifest/total-byte admission
limits as managed generations. Current defaults are 4 GiB per file, 16 GiB
total, a 1 MiB manifest, and the configured indexed-artifact count (normally
200). Preparation records effective limits; `--max-output-bytes` can tighten
the total but cannot bypass publication bounds. Larger batches may require
separate generations. Local admission neither constructs a publisher nor
grants managed visibility.

Canonical plan stdout has **no appended newline**; its SHA-256 is the exact
`plan_sha256` in the completion summary. Progress goes to stderr. Verification
accepts `--expected-plan-sha256 <digest>` to bind an intended plan. V4/V5's
older unsigned-plan fingerprint remains internal to those native contracts;
it is not the new CLI's byte digest. Cancellation returns a nonzero exit and
never reports partial execution as successful. Keep failed outputs for
inspection and retry into a new directory.

Retained-V5 finishing and Depth Pro verification apply caller resource limits
before source-pixel reads and replay. Defaults match the native CLI: 64 GiB of total
source bytes, 100 million pixels per image, and 16,384 MiB of working memory.
Override them with `verify --max-input-bytes`, `--max-pixels`, and `--memory-mib`,
or pass `source_limits=SourceLimits(...)` to the Python `verify` function.
Recorded plan limits can only tighten these caller limits. Explicit source-limit
overrides are rejected for self-contained inference/composite outputs, whose
native verifiers retain their existing admission contracts.

An installed wheel registers `lux-depth`; a checkout with older console scripts
can use the module commands above. Outside a checkout, install the supported
core dependencies and point `TP_MODEL_LOCK_MANIFEST` at the governed
`config/model_lock_manifest.yaml`. Set `--runtime-python` explicitly when the
DA3 runtime is not discoverable. Missing policy/runtime files fail closed;
installing the wheel does not install neural weights or authorize a model.

## Materials, retained finishing, and research

`infer` accepts `--materials-manifest` and a complete, bounded JSON
`--materials-policy` as described by the MaterialsV4 guide. `process` rejects
these flags: applied Materials responses cannot be replayed by the existing
V6 retained-source contract. The unified command does not silently discard
materials or claim that unsupported composition works.

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth infer \
  --input-dir /absolute/photographs --output-dir /absolute/output/inference \
  --input-color srgb

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth finish \
  --input-dir /absolute/output/inference --output-dir /absolute/output/finished \
  --exposure-stops 0.25 --saturation 1.05

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth verify \
  --output-dir /absolute/output/finished --source-root /absolute/output/inference
```

The finish example requires a V5 generation with no applied Materials response;
use a separate V5 generation without Materials if any response was applied.
Retained finishing requires no neural runtime. `--no-depth-maps` is available
only on `finish` and preserves the older V1 photographic artifact set.

Depth Pro keeps its separate research acknowledgments and never uses DA3 cache
identity or automatically performs unsupported depth-driven photographic edits:

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth depth-pro \
  --input-dir /absolute/photographs --output-dir /absolute/output/research \
  --depth-pro-python "$PWD/.venv-depth-pro/bin/python" \
  --depth-pro-checkpoint "$PWD/checkpoints/depth_pro.pt" --device mps \
  --non-commercial-ok --accept-apple-depth-pro-research-license

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth verify \
  --output-dir /absolute/output/research --source-root /absolute/photographs
```

Only pass research acknowledgments when they apply to the intended use. The
optional ML-core `psutil` dependency is required in the parent for this route.
Verification does not require the inference interpreter/checkpoint options.
Replays remain point-in-time checks: protect source and output roots from
concurrent writers. Historical plans bind processing source/dependency hashes;
replay needs the matching historical environment or a freshly prepared run.

## Python and managed integration

```python
from pathlib import Path
from transformation_portal.lux_depth import (
    PhotographyRequest, InferenceRequest, GradeRecipe, prepare, run, verify,
)

prepared = prepare(PhotographyRequest(
    InferenceRequest(Path("/absolute/photographs"), Path("/absolute/output/lux"),
                     input_color="srgb"),
    grade=GradeRecipe(exposure_stops=0.25),
))
result = run(prepared)
verified = verify(result.output_root)
```

`InferenceRequest`, `PhotographyRequest`, `FinishingRequest`, and
`DepthProRequest` are aliases of exact existing immutable request classes.
The prepared carrier and native result are preserved; use `result_summary`
for a normalized completion summary. Unsupported carriers and subclasses fail
closed. Importing the package or CLI help does not import model runtimes.

Managed V5/V6 admission adapters and the worker dispatcher call this same API.
Their existing pipeline IDs, feature flags, native schemas, and artifact labels
remain supported. Existing V3 invocation is available through
`python -m transformation_portal.lux_depth legacy --help`.

## HTTP portal and managed access

Select **Lux Depth Unified (opt-in)** in Build, then select **process** or
**infer**. Each workflow retains its own draft. `process` accepts original
photographs and publishes the verified finished photograph, draft preview,
depth products, and retained inference evidence. `infer` exposes V5 inference
and optional source-bound Materials controls. V3 remains the portal default.
Staged file/folder uploads and authorized server directories use the existing
upload contract. Build requires a successful preview of the current workflow
and controls plus current readiness. Review uses verified PNG derivatives for
display; TIFF/float artifacts retain precision authority.

Preview and admission inspect authorized image headers within bounded read and
inventory limits, reporting unsupported or ambiguous color before dispatch.
This metadata check does not replace execution-time decoding and validation.

Managed `finish`, `depth-pro`, `verify`, and `legacy` workflow values are not
accepted. Retained-generation finishing needs a distinct source-authority
contract, and research execution needs its separate license/runtime boundary;
use the standalone commands for these workflows.

Deploy matching API and worker code, apply `make db-upgrade` through
`0007_photography_bindings`, and load the managed service environment on each
host. It must supply database/Redis URLs, backend/frontdoor authentication,
authorized input/output roots, an executable governed
`TRANSFORMATION_PORTAL_DA3_PYTHON`, and the same protected service-owned mode
0700 `TP_ORCHESTRATOR_EXECUTION_ROOT`. See the
[distributed runbook](../runtimes/orchestrator-postgres.md). Then configure both
API and worker hosts:

```bash
export TP_LUX_DEPTH_MANAGED_ENABLED=1
export TP_ORCHESTRATOR_STATE_BACKEND=postgres
export TP_ORCHESTRATOR_QUEUE_BACKEND=redis
export TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED=0
```

The Unified flag is independent of `TP_LUX_V5_MANAGED_ENABLED` and
`TP_LUX_V6_MANAGED_ENABLED`. In pilot tenant mode, add `lux-depth` to the
existing `TP_PILOT_ALLOWED_PIPELINES` list on API and workers. Keep authenticated
frontdoor sessions, signed actor membership, and tenant-contained paths. The
frontdoor and backend must use the same dedicated identity secret; browser
storage and job bodies must never contain the backend API key. Runtime/RAW
interpreters and the optional `TP_LUX_V5_CACHE_DIR/<tenant_id>` cache remain
server-owned. Start `make run-backend-local-noreload`,
`make run-orchestrator-worker`, and `make run-frontdoor-local` in separate
terminals with the appropriate environment; the frontdoor requires Node 22.

`GET /v1/readiness` includes `lux-depth` with its default `process` workflow.
`GET /v1/config-metadata?pipeline=lux-depth` supplies closed JSON argument
schemas for both supported workflows. Post the same payload to
`POST /v1/config-preview` and `POST /v1/jobs`, correcting preview errors first:

```json
{
  "pipeline": "lux-depth",
  "args": {
    "workflow": "process",
    "input_dir": "/authorized/tenant/photos",
    "output_dir": "/authorized/tenant/output-unified",
    "input_color": "srgb",
    "device": "mps",
    "precision": "fp32",
    "exposure_stops": 0.25
  }
}
```

Omitted `workflow` normalizes to `process`. For `infer`, select that workflow
and omit finishing-only controls such as `exposure_stops`; unsupported fields
are rejected rather than ignored. The preview reports readiness and plan schema
for the selected workflow. Existing response envelopes, job routes,
cancellation, and artifact URLs remain unchanged, including `/v2/jobs`.
Existing SSE events remain supported alongside the ephemeral `heartbeat` event.
See the [portal quickstart](../guides/PORTAL_ORCHESTRATOR_QUICKSTART.md#endpoints)
for heartbeat and same-origin managed artifact delivery behavior.

The immutable `tp.job.lux_depth.bindings.v1` envelope binds `pipeline`,
`workflow`, and the native photography bindings to the existing admitted
binding digest. `process` requires native plan V5; `infer` requires native plan
V4. Workers recheck the Unified flag, exact pipeline grant, tenant paths,
runtime selection, and cache namespace before executing the frozen plan.
Historical V5/V6 grants cannot authorize Unified, or vice versa. Publication
independently verifies the native output under the current dispatch fence and
retains the native artifact/evidence schemas. The run summary reports
`pipeline: lux_depth`, its workflow, and `engine_pipeline`.

Rollback is to disable the Unified flag on API and workers while retaining
matching worker code until Unified jobs are drained or canceled. Existing
V3/V5/V6 selections remain available under their own policy. No database schema
migration beyond the existing photography bindings columns is introduced.

## Validation

```bash
make test-lux-depth-contract
make test-lux-depth-managed-contract
make test-lux-depth-v5-managed-contract
make test-lux-depth-v6-managed-contract
make test-orchestrator-http-contract
make test-portal-contract
make test-frontdoor-contract
```

The unified lane runs the new composition/API/CLI contracts plus the V4, V5,
V6, Materials, and shared depth contracts. Controlled worker fixtures exercise
real decoding, precision, reconstruction, artifacts, and independent replay;
they do not establish model accuracy. Service-backed gates still require their
dedicated migrated Postgres/Redis services. Native photographic and hosted
acceptance evidence must be reported separately.

The service-backed Unified gate is `make test-lux-depth-managed-services` with
`TP_DISPATCH_TEST_DATABASE_URL` naming a migrated dedicated `*_test` database
and `TP_DISPATCH_TEST_REDIS_URL` set. It exercises both workflows through HTTP,
Postgres, Redis, an external worker, and verified artifact retrieval, with a
controlled inference worker. It is integration evidence, not neural-model or
production acceptance. Browser validation uses `make validate-portal-browser`
and `make validate-frontdoor-browser` with the supported browser prerequisites.
