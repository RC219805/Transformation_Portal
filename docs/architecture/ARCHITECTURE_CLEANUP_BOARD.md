# Architecture Cleanup Board

**Status:** Active implementation ledger
**Created:** 2026-05-30
**Owner:** Transformation Portal Architect

This board is the current cleanup execution surface. It intentionally does not
re-open landed Tier 1 / Tier 2 audit work or landed monolith extraction seams.
Use it to choose narrow implementation PRs that preserve route shapes,
selectors, API envelopes, auth posture, and validation semantics.

## Active Work

| Priority | Area | Disposition | Source authority | Gate |
| --- | --- | --- | --- | --- |
| P0 | Jobs route decorators | implemented | `app.py` jobs route family; `tests/test_app_route_inventory.py` | `make test-orchestrator-http-contract` |
| P0 | `JobCreateRequest` adoption | implemented | `src/transformation_portal/api/v1/jobs.py` | malformed payload + unsupported pipeline contract tests |
| P1 | Durable SSE replay | implemented | existing `JobEventStore` storage surface | event-store contract + HTTP SSE replay tests |
| P1 | Runtime licensing evidence | implemented | audit item M-2 and run-card schemas | run-card schema + licensing gate tests |
| P1 | Performance gate policy | implemented | audit item M-3 and ADR-034 benchmark policy | doc/workflow parity review |
| P2 | Plugin trust model | implemented | audit item M-1 | plugin trust-boundary tests + ADR |
| P2 | Next mypy tranche | measure | `docs/ci/TYPE_CHECKING_POLICY.md` | `mypy --config-file=mypy.ini <path>` |
| P2 | Cold-zone coverage ratchet | measure | `docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md` | package/branch coverage scripts |
| P3 | ADR-050 portal evidence | defer intentionally | portal modernization RFC evidence gate | pilot evidence artifact |

## Historical References

| Area | Current status | Do not repeat |
| --- | --- | --- |
| Audit Tier 1 | Complete as of the 2026-05-18 backlog updates | torch-load scanner wiring, SAM2 benchmark baseline alignment, Docker non-root runtime, ADR-032 pip-audit correction |
| Audit Tier 2 | Complete except the explicitly deferred coverage-floor trigger | orchestrator mypy tranche, ML sampled coverage instrumentation, shared segmentation content digest |
| Monolith seams | Landed through Target 5D in `MONOLITH_DECOMPOSITION_TARGETS.md` | lux-depth orchestrator helper extraction, portal asset/path/SAM2/archive/artifact helpers, segmentation split, rendering split, spatial AI result/config/graph seams |
| Portal rewrite decision | Not approved | React/Next operator-console rewrite; ADR-050 evidence remains open |

## Operating Rules

- Pick one row per PR unless two rows share the same test fixture and failure
  mode.
- Preserve existing API envelopes, route templates, selectors, auth behavior,
  and browser smoke observability.
- Add or strengthen one focused gate in the same PR as each behavior change.
- Treat service availability failures for Postgres, Redis, Docker, and browser
  runtimes as environment/tooling blockers; do not weaken product contracts to
  hide them.
- Keep `src/tp` and `src/transformation_portal` as separate public import
  surfaces.

## Validation Backbone

Use the narrowest relevant gate while iterating, then close implementation work
with:

```bash
make ci-quick
```

When the touched surface includes managed frontdoor or live browser behavior,
also run the relevant browser smoke:

```bash
make validate-frontdoor-browser
make validate-portal-browser
```
