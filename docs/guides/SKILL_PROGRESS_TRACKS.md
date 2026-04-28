# Skill Progress Tracks

This guide turns recurring review feedback from the `skill-progression-map`
automation into concrete practice tracks. Each track is evidence-linked to
recent PR review themes and is meant for additive learning work unless the
drill explicitly fixes a reviewed defect.

Use the tracks after an automation run ranks one of these skills. Keep drills
small, preserve existing route shapes and CLI contracts, and validate with the
focused acceptance tests listed here before widening scope.

## API Contract Parity

Review evidence:

- PR #1567: `src/transformation_portal/api/v1/config.py` needed a fuller
  top-level `ConfigPreviewData` schema and required preset fields so OpenAPI
  matched the stable wire shape.
- PR #1561: `src/transformation_portal/api/v1/errors.py` initially missed
  real orchestrator error codes and did not mirror `_error_obj`'s
  `details=None` coercion.

Drill 1 - derive schemas from handler fixtures:

- Target files: the route model under `src/transformation_portal/api/v1/` and
  its focused `tests/api/v1/` model test.
- Expected behavior: build the Pydantic payload model from an existing handler
  fixture or contract-test payload before adding `response_model=`.
- Acceptance tests: assert required fields, optional fields, extra-key policy,
  and full envelope `model_dump(mode="json")` output against the real fixture.

Drill 2 - replace brittle generic identity checks:

- Target files: API model tests that assert aliases or generic envelope types.
- Expected behavior: prove validation behavior and serialized envelope shape
  instead of relying on `ApiEnvelope[T]` object identity.
- Acceptance tests: include one valid instance per envelope type and one
  negative validation case proving the wrong payload type is rejected.

Review checklist:

- Does the model accept every value the handler can emit?
- Does the test fixture come from the handler or existing HTTP contract?
- Does the test avoid Pydantic cache or object-identity assumptions?

## Fail-Closed Path Governance

Review evidence:

- PR #1555: archive index preflight in `app.py` needed clearer
  archive-root/input-dir error attribution and symlink handling.
- PR #1556: `tools/run_apex_eval.py` needed safe derived-evidence path
  components before constructing output paths from CLI strings.

Drill 1 - validate path components before joining:

- Target files: CLI or pipeline code that builds output paths from user or
  manifest identifiers.
- Expected behavior: reject empty values, `.`/`..`, separators, drive-like
  forms, and characters outside a documented safe component regex before path
  construction; then verify resolved containment as defense in depth.
- Acceptance tests: parametrize unsafe components and assert the command fails
  closed without writing outside the intended directory.

Drill 2 - attribute archive-root preflight errors to the right field:

- Target files: archive-gate preview/job validation code and its runtime/HTTP
  contract tests.
- Expected behavior: missing or non-directory roots surface as
  `archive_root`/`input_dir` errors; only a valid root with non-resolving index
  rows surfaces as an `archive_index` mismatch.
- Acceptance tests: cover missing root, symlink root, matching root, and
  mismatched index/root cases with stable reason codes and field names.

Review checklist:

- Is validation performed before filesystem side effects?
- Are unsafe paths rejected rather than normalized into a different target?
- Are user-facing field errors tied to the field operators can actually fix?

## Deterministic CI And Docs Validation

Review evidence:

- PR #1560: `docs/ci/WORKFLOW_MATRIX.md` had workflow inventory drift around
  unique jobs, branch coverage, schedules, and advisory/required status.
- PR #1557: `docs/governance/audit/archive-gates-2026-04-27.md` needed to
  describe normalized audit JSON accurately and use a reproducible snippet
  that works outside repo root.

Drill 1 - generate workflow inventory from YAML:

- Target files: workflow inventory docs and the script or test that derives
  job names, triggers, cron schedules, and advisory/required status.
- Expected behavior: regenerate inventory facts from `.github/workflows/*.yml`
  instead of hand-copying them into prose.
- Acceptance tests: parse representative workflow fixtures and assert unique
  jobs, push branches, cron entries, and advisory markers are reflected in the
  generated summary.

Drill 2 - make audit normalization reproducible:

- Target files: governance audit docs and any helper snippet that redacts or
  normalizes machine-local fields.
- Expected behavior: derive repo root from `git rev-parse --show-toplevel` or
  another invocation-safe source, remove ephemeral fields explicitly, and
  rewrite absolute fixture paths to repo-relative paths.
- Acceptance tests: run the normalization from a non-root working directory
  and assert no `/Users/`, `/home/`, `/tmp/`, or Windows absolute paths remain.

Review checklist:

- Are generated or inventoried facts sourced from the underlying files?
- Does every reproducibility snippet work when copied verbatim?
- Are normalized artifacts labeled as normalized rather than raw?

## Runtime Bootstrap Determinism

Review evidence:

- PR #1565: `scripts/setup/install_da3_runtime.sh` needed explicit fetch/ref
  behavior and deterministic dependency-profile metadata for optional
  `xformers`.
- PR #1559: `docker-compose.yml` and `.env.example` needed Docker healthcheck,
  interpreter, and environment-variable contract alignment.

Drill 1 - lock runtime ref and dependency profile dry runs:

- Target files: runtime installer scripts and validation tests for bootstrap
  behavior.
- Expected behavior: dry-run output names the source ref, fetch ref,
  dependency profile, interpreter path, pinned dependencies, and any
  intentionally operator-managed dependency.
- Acceptance tests: verify remote-only fetch refs, commit SHA refs, profile
  inclusion/exclusion, and preserved virtualenv paths without downloading
  models.

Drill 2 - compare container/env contracts across layers:

- Target files: Dockerfile, Compose file, `.env.example`, and docs that describe
  runtime environment variables.
- Expected behavior: service commands use interpreters present in the image,
  Compose healthchecks do not silently override Dockerfile healthchecks unless
  documented, and env vars are labeled by the layer that consumes them.
- Acceptance tests: static checks assert documented env vars are consumed by
  the claimed layer and that Compose service commands match image interpreter
  names.

Review checklist:

- Can an operator tell exactly what ref and dependency profile will install?
- Are unpinned optional dependencies either pinned or called out explicitly?
- Do Docker, Compose, and `.env.example` describe the same runtime contract?

## APEX Evidence Semantics

Review evidence:

- PR #1564: `src/transformation_portal/stage_graph/stages/enhancement.py`
  needed material names in mask cache keys and aligned mask predicates for
  run metadata.
- PR #1556: `src/transformation_portal/evals/apex_evidence_bundle.py` and
  `tools/run_apex_eval.py` needed implemented-op inference, idempotent warning
  codes, and canonical JSON artifact writing.

Drill 1 - test mask semantics with adversarial fixtures:

- Target files: material enhancement stage tests and APEX metric/evidence tests.
- Expected behavior: cache keys include material names and bytes; metadata
  flags use the same predicate as the pixel operation path.
- Acceptance tests: cover identical mask arrays assigned to different material
  names, signed or zero-valued masks, malformed masks, and no-pixel-op
  passthrough cases.

Drill 2 - centralize APEX codes and artifact writing:

- Target files: `src/transformation_portal/lux_depth_v3/apex_codes.py`
  (`transformation_portal.lux_depth_v3.apex_codes`), eval helpers, and CLI
  artifact writers.
- Expected behavior: stable warning/failure codes live in the canonical module
  and are re-exported only for compatibility; JSON artifacts use the repo
  canonical helper with `allow_nan=False`, deterministic key order, UTF-8, and
  a trailing newline.
- Acceptance tests: assert object identity for compatibility re-exports,
  idempotent warning insertion across repeated calls, and byte-level JSON
  formatting invariants for derived artifacts.

Review checklist:

- Does evidence semantics distinguish applied ops, blocked implemented ops,
  no implementation, and disabled feature flags?
- Are warning/failure codes imported from one stable owner?
- Are evidence artifacts portable, deterministic JSON?
