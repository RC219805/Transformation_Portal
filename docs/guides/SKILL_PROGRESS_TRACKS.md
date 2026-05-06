# Skill Progress Tracks

This guide turns recurring review feedback from the `skill-progression-map`
automation into concrete practice tracks. Each track is evidence-linked to
recent PR review themes and is meant for additive learning work unless the
drill explicitly fixes a reviewed defect.

Use the tracks after an automation run ranks one of these skills. Keep drills
small, preserve existing route shapes and CLI contracts, and validate with the
focused acceptance tests listed here before widening scope.

## 2026-05-05 Review-Thread Refresh

The `2026-05-03T16:55:21.582Z` through `2026-05-05T21:12:57Z`
automation window surfaced the following current practice tracks from
GitHub-thread-backed evidence. Treat these as additive drills on top of the
earlier tracks; where the reviewed defect is already fixed, add focused
regression coverage that preserves the landed contract.

### Deterministic Validation-System Design

Review evidence:

- PR #1641: `src/transformation_portal/portal/asset_bundle.py` performed
  portal manifest filesystem I/O at import time.
- PR #1636: `app.py` hardcoded FastVLM model roles in
  `_fastvlm_model_role_from_value` instead of deriving them from the governed
  runtime role contract.

Drill 1 - defer import-time resource validation:

- Target files: one portal or runtime helper that currently loads files,
  manifests, or optional resources at import time.
- Expected behavior: importing the module stays side-effect-light, while the
  runtime helper still fails closed when the required resource is actually
  used or explicitly validated.
- Acceptance tests: import the module with the backing file missing, assert the
  import succeeds, then call the runtime helper and assert the same governed
  failure message and path context.

Drill 2 - derive runtime roles from canonical metadata:

- Target files: FastVLM or optional-runtime role resolution helpers and their
  fixtures.
- Expected behavior: role validation reads the canonical role metadata or
  manifest, rejects unknown roles, and does not duplicate role names in app
  code.
- Acceptance tests: table-drive smoke/default/review role inputs plus one
  manifest-only added role and one unknown role; assert accepted roles and
  failure messages match the canonical metadata.

### Documentation Governance Consistency

Review evidence:

- PR #1646: `docs/architecture/agent_governance.md` added `AGENTS.md` to the
  governance-controlled artifact list without first mirroring that artifact in
  the escalation criteria.
- PR #1643: `docs/architecture/MONOLITH_DECOMPOSITION_TARGETS.md` changed the
  ranked-target list inside an extraction PR even though the document limited
  extraction PRs to seam status-table updates.

Drill 1 - lock authority surfaces against escalation criteria:

- Target files: agent governance docs, live agent profiles, and custom-agent
  documentation tests.
- Expected behavior: every governance-controlled artifact appears in the
  corresponding escalation criteria and in the live profiles that delegate to
  those criteria.
- Acceptance tests: build a table of authority artifacts and assert
  `AGENTS.md`, `.github/copilot-instructions.md`, `.github/agents/*`, and
  custom-agent tests are all present in the governance and profile sections.

Drill 2 - separate extraction status from ranking governance:

- Target files: decomposition target docs and docs-governance tests.
- Expected behavior: extraction PRs update only the status table row for the
  shipped seam; ranked-target list changes require a separate governance
  refresh section or PR.
- Acceptance tests: fixture one status-only extraction update and one ranked
  list mutation; assert the former passes and the latter fails with the
  required governance-refresh path.

### Contract-Driven Portal And Frontdoor State Modeling

Review evidence:

- PR #1637: `web/secure-landing/portal-src/review-surface-deferred.js`
  filtered FastVLM captioning artifacts strictly by the selected artifact stem,
  dropping available sidecar/raw/proxy links for other selected artifacts.
- PR #1637: the review evidence strip rendered `FastVLM: Not requested` when
  indexed captioning artifacts existed but `run_summary.captioning_status` was
  missing.

Drill 1 - model captioning artifact fallback states:

- Target files: review-surface artifact grouping helpers and portal smoke
  fixtures.
- Expected behavior: selected-stem matches are preferred, but job-level
  captioning artifacts remain available when the selected artifact has no
  matching captioning evidence.
- Acceptance tests: fixture selected image, compare image, run card, sidecar,
  raw, and proxy artifacts; assert the evidence strip renders stable link
  counts in every selected-artifact state.

Drill 2 - derive visible status from evidence presence:

- Target files: captioning evidence status helpers and review surface smoke
  tests.
- Expected behavior: missing `captioning_status` plus indexed captioning
  artifacts renders an available/evidence-backed state rather than `off` or
  `not requested`.
- Acceptance tests: table-drive missing status, explicit off, pending,
  succeeded, failed, malformed sidecar, and artifact-only inputs; assert
  `data-status`, label text, and link visibility stay coherent.

### Scripts Failure-Mode And Fixture Hygiene

Review evidence:

- PR #1634: `scripts/download_samples.py` generated minimal fixtures locally,
  but `image.save()` and directory creation failures lacked the cleanup path
  used by downloaded sample writes.
- PR #1631: `scripts/ci/check_per_package_coverage.py` let nested
  `lux_depth_v3/validators/` files count toward both the child validator floor
  and the parent `lux_depth_v3` package floor.

Drill 1 - failure-inject local fixture generation:

- Target files: local sample or fixture generation scripts and their tests.
- Expected behavior: partial writes, directory creation failures, and invalid
  generated images clean up temporary outputs and report the failed path
  without leaving stale fixtures.
- Acceptance tests: monkeypatch directory creation and image saving to raise,
  then assert temporary files are removed and the command exits with the
  deterministic cleanup-worthy error.

Drill 2 - prove overlapping path ownership boundaries:

- Target files: package ownership, coverage, or artifact-budget scripts that
  classify files by path prefix.
- Expected behavior: child prefixes with their own floor or budget are excluded
  from the parent rollup unless the contract explicitly opts into aggregation.
- Acceptance tests: use synthetic parent, child, sibling, and similarly named
  prefixes; assert each file contributes to exactly one intended owner.

### Pipeline Re-Export And Import-Surface Discipline

Review evidence:

- PR #1645: `src/transformation_portal/pipelines/rendering_4k_pipeline.py`
  re-exported private helpers from the extracted stages module without making
  the intentional compatibility surface obvious to lint.
- PR #1645: `src/transformation_portal/pipelines/rendering_4k/__init__.py`
  eagerly imported stage functions, expanding the package import surface to
  optional SciPy-backed code.

Drill 1 - make compatibility re-exports explicit:

- Target files: one extracted package initializer or legacy compatibility shim.
- Expected behavior: intentional re-exports are covered by `__all__`, identity
  tests, and explicit lint markers or comments that distinguish compatibility
  exports from unused imports.
- Acceptance tests: assert legacy and extracted symbols are identical objects,
  run the focused lint check, and verify the compatibility shim exports only
  the documented surface.

Drill 2 - lazy-load optional stage dependencies:

- Target files: package `__init__.py` files or stage modules that expose
  optional-dependency-backed functions.
- Expected behavior: importing the package does not import optional heavy
  dependencies; the dependency is touched only when the specific stage function
  runs.
- Acceptance tests: monkeypatch the optional dependency import to fail, import
  the package successfully, then call the dependent stage and assert the
  governed optional-dependency failure is raised.

## 2026-05-03 Review-Thread Refresh

The `2026-05-03T16:55:21.582Z` automation window surfaced the following
current practice tracks from connector-backed GitHub review-thread evidence.
Treat these as additive drills on top of the earlier tracks; where the reviewed
defect is already fixed, add focused regression coverage that preserves the
landed contract.

### Coverage Governance

Review evidence:

- PR #1631: `scripts/ci/check_per_package_coverage.py` needed validator
  files to stop double-counting into the parent `lux_depth_v3` package floor.
- PR #1631: absolute Cobertura filenames needed to normalize before package
  prefix matching, alongside relative, `./`, and platform-separator forms.

Drill 1 - exclude child prefixes from parent floors:

- Target files: per-package coverage checker fixtures and focused coverage
  governance tests.
- Expected behavior: a parent package floor counts only files owned by that
  package, while explicitly configured child package prefixes are measured by
  their own floor.
- Acceptance tests: build nested synthetic package fixtures and assert child
  prefixes cannot inflate or deflate the parent package rollup.

Drill 2 - normalize Cobertura filenames before matching:

- Target files: Cobertura parsing helpers and package-prefix matching tests.
- Expected behavior: absolute paths, repo-relative paths, `./` prefixes, and
  backslash separators all resolve to the same canonical repo-relative file
  key before coverage ownership is evaluated.
- Acceptance tests: feed synthetic Cobertura XML for each filename form and
  assert every variant maps to the expected package owner.

### Dependency Parser Contracts

Review evidence:

- PR #1629: explicit `constraints.txt` paths needed to honor the documented
  exemption in `scripts/validation/check_dependency_pinning.py`.
- PR #1629: arbitrary-equality pins such as `pkg===1.2.3` needed to fail the
  exact-pinning contract instead of being accepted as `==` pins.

Drill 1 - table-drive requirement-line parsing:

- Target files: dependency pinning parser tests and the validation helper when
  needed.
- Expected behavior: only standard `==` exact pins pass, while `===`,
  unpinned requirements, malformed wrapped lines, comments, hash continuations,
  and options are classified according to the documented contract.
- Acceptance tests: use one parametrized table covering `==`, `===`, extras,
  backslash continuations, comments, hashes, and option lines.

Drill 2 - lock CLI scan modes:

- Target files: dependency pinning CLI tests.
- Expected behavior: default scans, explicit file arguments, all-exempt input
  sets, empty input sets, and explicit `constraints.txt` paths produce stable
  exit codes and messages.
- Acceptance tests: run the CLI against temporary fixture trees for each mode
  and assert exempt inputs are skipped without hiding real violations.

### Public Surface Regression Testing

Review evidence:

- PR #1630: `tests/test_comfyui.py` needed to assert
  `BaseNode.__abstractmethods__` instead of relying on descriptor-sensitive
  `__isabstractmethod__` checks.
- PR #1630: depth tool shape coverage needed to cover both
  `src/transformation_portal/depth/tools.py` and
  `src/transformation_portal/pipelines/depth_tools.py`.

Drill 1 - assert canonical runtime invariants:

- Target files: public surface tests for abstract base classes and exported
  node contracts.
- Expected behavior: tests use the metaclass or library's canonical runtime
  invariant rather than inspecting descriptors whose behavior can change across
  Python versions or wrappers.
- Acceptance tests: assert `BaseNode.__abstractmethods__` contains the expected
  abstract hooks and that concrete implementations satisfy the runtime
  contract.

Drill 2 - duplicate-surface regression matrix:

- Target files: tests for paired public implementations and compatibility
  wrappers.
- Expected behavior: every supported implementation path that exposes the same
  public behavior is imported and checked against the same shape invariant.
- Acceptance tests: parametrize both depth tool modules and assert matching
  result shapes, error behavior, and exported symbols.

### CI Signal Efficiency

Review evidence:

- PR #1631: `.github/workflows/build.yml` duplicated frontdoor work by running
  both `npm test` and `npm run test:coverage`.
- PR #1630: PR metadata needed to disclose the depth-tools CI blocker alongside
  the ComfyUI test change so reviewers could see the true scope.

Drill 1 - collapse duplicate CI invocations:

- Target files: workflow tests and the CI workflow under review.
- Expected behavior: each frontend or coverage check has one authoritative CI
  step, with duplicate build/test invocations removed unless they exercise
  distinct contracts.
- Acceptance tests: inspect the workflow and assert the frontdoor test/build
  command set has no duplicate semantic coverage.

Drill 2 - disclose discovered CI blockers:

- Target files: PR template/checklist docs or release-review guidance.
- Expected behavior: when implementation fixes an additional CI-blocking defect,
  the PR title/body names the primary intent, the discovered blocker, and the
  validation commands that prove both.
- Acceptance tests: add a docs contract test or checklist fixture that rejects
  missing discovered-blocker and validation-command fields.

### Docs And Status Truthfulness

Review evidence:

- PR #1629: a script docstring claimed TODO section 5.7 was closed while
  `docs/analysis/TODO_INVENTORY.md` still marked that work partial.
- PR #1628: `CLAUDE.md` needed current FastVLM runtime targets, fail-closed
  `install-ml-raw` behavior, and portal CSS parity guidance.

Drill 1 - validate TODO closure language:

- Target files: docs/status consistency checks and TODO inventory fixtures.
- Expected behavior: "closes TODO" or equivalent completion language is only
  allowed when the referenced TODO inventory entry is actually closed.
- Acceptance tests: fixture one closed TODO and one partial TODO, then assert
  closure language passes only for the closed entry and fails with the stale
  status path for the partial entry.

Drill 2 - pair runtime docs with command contracts:

- Target files: operator docs, Makefile target references, and docs navigation
  tests.
- Expected behavior: every Make target or runtime-state change updates one
  operator-doc anchor and one navigation or command-reference contract in the
  same patch.
- Acceptance tests: assert the FastVLM, `install-ml-raw`, and portal CSS parity
  operator anchors remain linked from current docs and match the documented
  target behavior.

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
