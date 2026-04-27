# Archive Gates A / B / C — State-of-Gates Audit (2026-04-27)

**Status:** PASS — all 12 contract checks satisfied.
**Branch:** `claude/audit-portal-gates-8LYDh`
**Audited commit:** `974429333acf9575f2d4d73f6cfbf4a4d12cefde`
**Generated at (UTC):** `2026-04-27T04:36:30Z`
**Tool:** `scripts/validation/audit_pipeline_readiness.py` (invoked via `make audit-pipeline-readiness`)
**Normalized payload:** [`archive-gates-2026-04-27.json`](./archive-gates-2026-04-27.json)
(schema `tp.orchestrator.pipeline_readiness_audit.v1`; see §1.1 for the
exact normalization applied to the harness's stdout before commit)

## 1. Context & scope

This is a read-only audit of the three archive governance pipelines exposed
by the portal app:

| Pipeline | Canonical command | Description |
|---|---|---|
| `archive-gate-a` | `fixity-scan` | Manifest and provenance assembly |
| `archive-gate-b` | `bag-build` | BagIt packaging and validation workflow |
| `archive-gate-c` | `mets-export` | METS / PROV / STAC export workflow |

Definitions live in `app.py:1220-1234` (`ARCHIVE_GATE_DEFAULT_COMMANDS`,
`ARCHIVE_GATE_ALLOWED_COMMANDS`); per-gate readiness logic in
`app.py:2183-2468` (`_evaluate_pipeline_readiness` /
`_archive_gate_readiness`); command runner in
`tools/archive_governance.py` under the `tp.archive.machine.v1` machine
contract (`docs/api/ARCHIVE_MACHINE_MODE_CONTRACT.md`).

The most recent change to this surface is PR #1555 (commit `5545c5d`,
2026-04-26): *"fix(archive-gate): preflight fixity index roots"*. This
audit verifies the gates are healthy on top of that change. **No code,
tests, fixtures, or CI were modified by this audit.**

`lux-depth-v3` is also emitted by the harness (it runs four pipelines
together) but is out of scope here; for completeness its baseline is
recorded in §6.

### Fixtures used

| Role | Path |
|---|---|
| Archive root | `tests/fixtures/archive_small/archive_root` |
| Archive index | `tests/fixtures/archive_small/archive_index_normalized.csv.gz` |
| Golden hash manifest | `tests/fixtures/archive_small/golden/hash_manifest.csv.gz` |
| Rights policy | `policy/archive/rights_flags.yml` |

3 entries, 55 payload bytes, sha256.

### 1.1 JSON normalization

The committed JSON is a **normalized** snapshot of the harness's stdout
— not its verbatim output — so it remains portable, diffable across
machines, and free of local filesystem layout. The transformations
applied before commit are:

1. **Removed `data.output_dir`.** The harness writes a per-run temp
   directory (e.g., `/tmp/tp-pipeline-readiness-audit-XXXXXX`) which
   is ephemeral and machine-specific.
2. **Removed every `runner_details` block** under
   `data.pipelines.<name>.dispatch_readiness`,
   `data.pipelines.<name>.blocked_without_manifest`, and
   `data.pipelines.lux-depth-v3`. These embedded the runner's Python
   interpreter path and tool path (e.g.,
   `/.../.venv/bin/python`,
   `/.../tools/archive_governance.py`) which differ between
   environments and add no contract value — the canonical command
   that was executed is also recorded as `canonical_command`.
3. **Rewrote `data.fixtures.*`** from absolute paths
   (`/.../tests/fixtures/archive_small/...`) to repo-relative paths
   (`tests/fixtures/archive_small/...`,
   `policy/archive/rights_flags.yml`).

All other fields produced by the harness — `schema`, `success`,
per-gate `baseline_status`, `canonical_command`,
`command_exit_code`, `dispatch_readiness.status`,
`dispatch_readiness.missing_prerequisites`,
`blocked_without_manifest.status`,
`blocked_without_manifest.missing_prerequisites`, `manifest_chain`,
`artifacts`, `generated_at`, and the `lux-depth-v3` baseline metadata
— are preserved verbatim with the harness's `indent=2,
sort_keys=True` formatting. After normalization the JSON re-passes
all 12 contract checks listed in §2. To regenerate the verbatim
(unnormalized) output for an audit, see §9.

## 2. Pass/fail summary

All 12 contract checks against
`docs/governance/audit/archive-gates-2026-04-27.json` passed:

```
PASS  schema == tp.orchestrator.pipeline_readiness_audit.v1
PASS  payload.success == true
PASS  archive-gate-a  command_exit_code == 0           (fixity-scan)
PASS  archive-gate-a  manifest_build_exit_code == 0
PASS  archive-gate-a  rights_apply_exit_code == 0
PASS  archive-gate-a  dispatch_readiness.status == ready
PASS  archive-gate-b  command_exit_code == 0           (bag-build)
PASS  archive-gate-b  dispatch_readiness.status == ready
PASS  archive-gate-b  blocked_without_manifest.status != ready
PASS  archive-gate-c  command_exit_code == 0           (mets-export)
PASS  archive-gate-c  dispatch_readiness.status == ready
PASS  archive-gate-c  blocked_without_manifest.status != ready
```

| Gate | Baseline (no inputs) | Dispatch (with inputs) | Canonical-command exit |
|---|---|---|---|
| archive-gate-a | `degraded` | `ready` | `0` |
| archive-gate-b | `blocked` | `ready` | `0` |
| archive-gate-c | `blocked` | `ready` | `0` |

Baseline `degraded`/`blocked` is the documented healthy resting state —
the gates correctly refuse dispatch until required inputs are supplied.

## 3. Gate A — `fixity-scan` (manifest & provenance assembly)

- **Allowed commands:** `fixity-scan`, `fixity-verify`, `manifest-build`,
  `rights-apply` (`app.py:1228`).
- **Baseline status:** `degraded` (expected — no archive index supplied).
- **Dispatch readiness with `archive_index` supplied:** `ready`,
  `missing_prerequisites: []`. Runner resolved to
  `tools/archive_governance.py --json fixity-scan` via the audit's
  `.venv` Python.
- **End-to-end exit codes:**
  - `fixity-scan` → `0`
  - `manifest-build` → `0`
  - `rights-apply` → `0`
- **Artifacts produced:** `hash_manifest.csv.gz`, `hash_summary.json`,
  `merkle_roots.json`, plus the chained
  `archive_manifest_v2.jsonl` and `archive_manifest_v2.rights.jsonl`
  consumed downstream by gates B and C.
- **Hash summary:** `sha256`, 3/3 rows hashed ok, 55 bytes total, schema
  `1.0`, no missing/skipped/unreadable.
- **Manifest-build summary:** schema `tp.archive.manifest.v2.summary.v1`,
  3 entries, hash status counts `{"ok": 3}`,
  `created_source_counts` `{"unsupported": 3}` (fixture has no
  derivative source — expected for the small fixture).
- **Rights-apply summary:** schema `tp.archive.rights.summary.v1`,
  policy version `1`, all 3 entries default-classified.

## 4. Gate B — `bag-build` (BagIt packaging & validation)

- **Allowed commands:** `bag-build`, `bag-validate`, `dedup-plan`
  (`app.py:1234`).
- **Baseline status:** `blocked` (expected — no `manifest_jsonl`).
- **Blocked-without-manifest probe:** `status: blocked`,
  `missing_prerequisites: [{field: manifest_jsonl, reason:
  rights_manifest_required, severity: blocked, message: "Provide an
  existing rights-manifest JSONL artifact before dispatch."}]`. This
  proves the wiring at `app.py:2373-2417` correctly gates dispatch on
  the rights-manifest produced by gate A.
- **Dispatch readiness with `manifest_jsonl` supplied:** `ready`.
- **`bag-build` exit code:** `0`.
- **Artifacts produced:** complete BagIt structure
  (`bagit.txt`, `bag-info.txt`, `manifest-sha256.txt`,
  `tagmanifest-sha256.txt`, `data/`) plus `bag_build_report.json`.
- **Bag report:** schema `tp.archive.bagit.build_report.v1`,
  `copied_files: 3`, `payload_bytes: 55`, `payload_oxum: "55.3"`,
  source manifest correctly references the gate-A rights manifest.

## 5. Gate C — `mets-export` (METS / PROV / STAC export)

- **Allowed commands:** `mets-export`, `prov-export`, `stac-export`
  (`app.py:1234`).
- **Baseline status:** `blocked` (expected — no `manifest_jsonl`).
- **Blocked-without-manifest probe:** `status: blocked`, same
  `rights_manifest_required` blocker as gate B —
  `app.py:2418-2451` enforces the same prerequisite.
- **Dispatch readiness with `manifest_jsonl` supplied:** `ready`.
- **`mets-export` exit code:** `0`.
- **Artifacts produced:** `mets_export.xml`, `mets_summary.json`.
- **METS summary:** schema `tp.archive.mets_export.v1`,
  `manifest_rows: 3`, `payload_rows: 3`,
  `file_groups: {"access_derivative": 3}`,
  `partitions: {"Part1": 2, "Part2": 1}`.

## 6. Cross-gate observations

- **Sequential dependency proven.** The harness drives gate A's
  `fixity-scan` → `manifest-build` → `rights-apply` first, then feeds
  the rights-manifest into gates B and C. Both downstream gates are
  `blocked` without that manifest and `ready` with it — the documented
  A → {B, C} ordering matches the implementation.
- **Machine contract honored.** Every command emits canonical JSON
  under the `tp.archive.machine.v1` family
  (`hash_manifest_schema_version 1.0`,
  `tp.archive.manifest.v2.summary.v1`,
  `tp.archive.rights.summary.v1`,
  `tp.archive.bagit.build_report.v1`,
  `tp.archive.mets_export.v1`). No non-JSON stdout was observed from
  any invocation.
- **Post-#1555 fixity preflight is stable.** Gate A's
  `dispatch_readiness` reports `ready` with no missing prerequisites
  when an archive index is supplied, and the live `fixity-scan`
  produces a deterministic `merkle_roots.json` against the small
  fixture without errors — consistent with the hardening introduced by
  PR #1555.
- **`lux-depth-v3` (out of scope, recorded for completeness):**
  `base_status: ready`, `canary_status: unavailable`,
  `missing_prerequisites: []`. Canary unavailability is unrelated to
  the archive gates.

## 7. Recommendations (informational only — no fixes applied)

1. **Wire `make audit-pipeline-readiness` into CI.** Today the harness
   runs only locally; nothing in `.github/workflows/` exercises the
   archive-gate JSON contract end-to-end on PRs. A nightly or
   pre-merge job that publishes the readiness matrix as an artifact
   would catch contract drift before it ships. Scoped at
   [`follow-ups/ci-wire-audit-pipeline-readiness.md`](./follow-ups/ci-wire-audit-pipeline-readiness.md).
2. **Persist the readiness matrix on each run.** The script already
   supports `--json-output`; consider adopting a dated path under
   `docs/governance/audit/` (or an artifact bucket) so historical matrices are
   diffable.
3. **Add a non-empty rights-classification fixture.** The current
   fixture leaves all 3 entries `default`-classified
   (`rule_hit_counts: {"default": 3}`); an audit fixture that
   exercises at least one non-default rights rule would strengthen
   coverage of `rights-apply`.

These are observations, not blockers — defer to maintainers on
prioritization.

## 8. References

- `app.py:1137-1234` — pipeline presets and allowed-commands maps.
- `app.py:2183-2468` — `_evaluate_pipeline_readiness` and
  `_archive_gate_readiness`.
- `tools/archive_governance.py` — canonical command runner.
- `scripts/validation/audit_pipeline_readiness.py` — this audit's
  harness.
- `docs/api/ARCHIVE_MACHINE_MODE_CONTRACT.md` — `tp.archive.machine.v1`
  contract.
- `docs/guides/PORTAL_ORCHESTRATOR_QUICKSTART.md` — end-to-end
  walkthrough.
- PR #1555 / commit `5545c5d` — fixity-index preflight hardening.

## 9. Reproduction

```bash
# from repo root, with .venv set up (make venv && make setup)

# (a) Verbatim harness output (with machine-local paths intact -
# useful for local diagnostics, not suitable for committing as-is):
make audit-pipeline-readiness

# (b) Same payload written to a JSON file:
python scripts/validation/audit_pipeline_readiness.py \
    --json-output /tmp/archive-gates-readiness.json \
    --output-dir "$(mktemp -d -t tp-audit-readiness-XXXXXX)" \
    --keep-output
```

Exit code `0` and `payload.success == true` indicate a clean run.

To produce a snapshot equivalent to the committed JSON, apply the
three normalization steps from §1.1 to the verbatim output:

```bash
python - <<'PY'
import json, pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[0]) + "/"  # repo root
src = pathlib.Path("/tmp/archive-gates-readiness.json")
data = json.loads(src.read_text())
data["data"].pop("output_dir", None)
for entry in data["data"]["pipelines"].values():
    entry.pop("runner_details", None)
    for sub in ("dispatch_readiness", "blocked_without_manifest"):
        if isinstance(entry.get(sub), dict):
            entry[sub].pop("runner_details", None)
fx = data["data"]["fixtures"]
for k, v in list(fx.items()):
    if isinstance(v, str) and v.startswith(ROOT):
        fx[k] = v[len(ROOT):]
out = pathlib.Path("docs/governance/audit/archive-gates-2026-04-27.json")
out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
```
