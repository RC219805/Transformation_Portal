# Skill Progress Tracks

This guide turns recurring review feedback from the `skill-progression-map`
automation into concrete practice tracks. Each track is evidence-linked to
recent PR review themes and is meant for additive learning work unless the
drill explicitly fixes a reviewed defect.

Use the tracks after an automation run ranks one of these skills. Keep drills
small, preserve existing route shapes and CLI contracts, and validate with the
focused acceptance tests listed here before widening scope.

## 2026-05-02 Review-Thread Refresh

The `2026-04-30T21:19:09Z` automation window surfaced the following current
practice tracks. Treat these as additive drills on top of the original tracks;
where the reviewed defect is already fixed, add contract coverage that prevents
regression instead of rewriting the landed implementation.

### Portal CSS Governance And Parity Isolation

Review evidence:

- PR #1608: `scripts/validation/validate_portal_css_layer_parity.py` needed
  parity probes to restore DOM, localStorage, theme, and review-banner state
  after forcing visual states.
- PR #1610: `web/secure-landing/scripts/check-portal-css-architecture.mjs`
  needed stale historical ownership-report evidence to fail rather than
  self-heal silently.

Drill 1 - isolate parity probe mutations:

- Target files: portal CSS parity validation probes and focused smoke-script
  contract tests.
- Expected behavior: every probe that changes DOM classes/attributes,
  localStorage keys, or global portal state snapshots those fields and restores
  them before the next probe.
- Acceptance tests: assert review-banner, interaction-outline, skeleton,
  snapshot, and runtime class-census probes all use the shared restore guard.

Drill 2 - reject stale ownership evidence:

- Target files: CSS architecture report validation and portal CSS architecture
  tests.
- Expected behavior: historical hash, rendered fingerprint, and deferred-count
  fields are immutable evidence, not auto-repaired baseline data.
- Acceptance tests: mutate each historical field in a fixture and assert the
  validator exits non-zero with a stale-evidence message.

### Deterministic Test And Lint Contracts

Review evidence:

- PR #1609: `scripts/ci/check_no_tautological_tests.py` needed table-driven
  AST coverage for literal containers, real comment-only escapes, tag
  word-boundaries, and dynamic-expression false positives.
- PR #1605: Gaussian opacity validation needed deterministic boundary inputs
  rather than random values that could accidentally avoid the intended edge.

Drill 1 - table-drive lint AST cases:

- Target files: tautological-assert lint tests and the lint helper when needed.
- Expected behavior: constant-only truthy assertions are rejected, while dynamic
  containers, fixture strings, falsey sentinels, and non-comment escape text are
  not misclassified.
- Acceptance tests: one parametrized positive table and one parametrized
  negative table cover the AST shapes and escape hatch boundaries.

Drill 2 - remove RNG from negative boundary checks:

- Target files: one Gaussian/opacity validation test using random inputs for a
  negative case.
- Expected behavior: use fixed valid Gaussian fields plus explicit invalid
  opacity boundaries below 0 and above 1.
- Acceptance tests: repeat the invalid-boundary construction several times and
  assert every run fails for the same validation reason.

### Fail-Fast Input Validation And Path Containment

Review evidence:

- PR #1607: `SkyGANNode.time_of_day` needed to reject unknown values before
  preset construction in `src/transformation_portal/comfyui/custom_nodes.py`.
- PR #1609: CAS DAG lock naming needed traversal containment tests for
  `_get_lock("../escape", ...)`.

Drill 1 - validate user inputs before setup:

- Target files: one ComfyUI or pipeline node with user-controlled inputs.
- Expected behavior: reject invalid user options before importing heavy runtime
  dependencies, converting images, or constructing preset/pipeline objects.
- Acceptance tests: monkeypatch the expensive setup path to raise and prove the
  validation error is raised first.

Drill 2 - lock path containment exactly:

- Target files: CAS DAG lock tests.
- Expected behavior: sanitized lock files resolve directly under the configured
  locks root.
- Acceptance tests: assert both `resolved.is_relative_to(locks_root)` and
  `resolved.parent == locks_root` for traversal and separator inputs.

### Documentation Source-Of-Truth Governance

Review evidence:

- PR #1612: closure references across
  `docs/fixes/BINARY_FILE_BEST_PRACTICES.md`,
  `docs/deliverables/QUICK_WINS.md`, and
  `docs/analysis/TODO_INVENTORY.md` needed valid section anchors.
- PR #1611: `CLAUDE.md` placement/navigation guidance needed to stay aligned
  with the root-file policy and live agent-doc navigation.

Drill 1 - audit closure heading links:

- Target files: docs validation scripts, TODO/quick-win closure docs, and their
  tests.
- Expected behavior: links or section references added during TODO/quick-win
  closure resolve to headings in the referenced markdown file.
- Acceptance tests: fixture docs with a valid heading pass; stale heading names
  fail with the source path, target path, and missing heading.

Drill 2 - update guidance and navigation together:

- Target files: operator docs, documentation map/readme surfaces, and Makefile
  help.
- Expected behavior: introducing operator guidance or a new docs validation
  target updates the live navigation and placement policy in the same patch.
- Acceptance tests: docs checks include the new target and current navigation
  still points at live guidance, not archived notes.

### Security And Coverage Evidence Honesty

Review evidence:

- PR #1604: `tests/security/test_tenant_isolation_helpers.py` needed mutation
  isolation coverage for nested default tenant policy fields.
- PR #1609: `pyproject.toml` coverage configuration needed to avoid omitting
  production packages as a way to improve metrics.

Drill 1 - prove tenant policy mutation isolation:

- Target files: tenant isolation helpers and tests.
- Expected behavior: mutating one tenant's copied default policy cannot mutate
  the manager default or another tenant's policy.
- Acceptance tests: create two tenants from a default policy with mutable
  fields, mutate one, and assert the other/default values are unchanged.

Drill 2 - make zero coverage visible:

- Target files: coverage configuration and coverage roadmap/docs tests.
- Expected behavior: production packages remain in coverage measurement, and
  zero-coverage packages are listed as explicit ratchet targets.
- Acceptance tests: parse `pyproject.toml` and docs to prove production package
  paths are not in `coverage.run.omit` and current zero-coverage packages are
  named as ratchet targets.

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
