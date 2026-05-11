# Transformation Portal Execution Agents - Quick Start

**Status**: Ready to Use
**Scope**: Governed execution inside current repository contracts
**Current documentation baseline**: repo-wide refresh audit dated May 11, 2026, building on `main` through PR #1721

---

## Choose The Right Agent

### `@portal-app-steward`

Use for managed browser-boundary work:

- `web/secure-landing/`
- `/`, `/login`, `/portal`, `/portal/bootstrap`
- `/portal/assets/*`, `/portal/video/*`, managed `/healthz`
- `portal.html`
- `public/portal-assets/*`
- selector stability, bundle rebuilds, Node 22 frontdoor validation, and browser validation

Fast prompt patterns:

```text
@portal-app-steward tighten the /login recovery copy without changing auth, cache, or CSP contracts
```

```text
@portal-app-steward update the portal review shell loading state, keep data-ui anchors stable, and list the exact browser checks to run
```

```text
@portal-app-steward trace why the portal asset manifest is missing a browser-served file and show the smallest safe fix
```

### `@transformation-portal-specialist`

Use for backend and governed execution work:

- `app.py`
- typed `/v1/*` behavior
- backend `/healthz`, `/ready`, and `/v1/readiness`
- Lux Depth V3
- archive-gate
- machine-mode and ingest
- targeted docs/tests/tooling updates tied to those surfaces

Fast prompt patterns:

```text
@transformation-portal-specialist debug why /v1/jobs/{id}/events is missing the final done event in contract tests
```

```text
@transformation-portal-specialist update app.py validation without changing the /ready or typed /v1/* response contracts
```

```text
@transformation-portal-specialist fix this tp.meta.machine.v1 payload drift and update the matching tests
```

### `@transformation-portal-architect`

Use for governance and approval-bound decisions:

- dependency policy
- CI/CD policy
- security posture
- route-contract or public-interface changes
- ADR interpretation and architectural trade-offs

Fast prompt pattern:

```text
@transformation-portal-architect review this proposal to change the managed /portal bootstrap contract and list the required enforcement updates
```

---

## Escalate Instead Of Implementing

Escalate to the Architect when work touches:

- `pyproject.toml`, `requirements/`, lockfiles, dependency bans, or new models/runtimes
- `.github/workflows/*`, CI Gate composition, release automation, or deployment behavior
- documentation topology, live custom-agent role boundaries, or Copilot instructions
- security-sensitive input handling or trust-boundary changes
- public CLI/HTTP/schema/import-surface changes
- ADR ambiguity or rewrite decisions such as reopening the portal migration question

---

## Validation Defaults

Prefer canonical repo entrypoints:

- browser boundary work: `make test-frontdoor-contract`, `make test-portal-contract`
- managed browser smokes: `make validate-frontdoor-browser`, `make validate-portal-browser`
- backend/orchestrator work: `make test-orchestrator-contract`
- Lux Depth and broader execution work: `make test-fast`
- broader repo-impacting work: `make ci`

Use the smallest command set that proves the change. If a broader command is needed, say why.

---

## Related Documents

- `portal-app-steward.md`
- `transformation-portal-specialist.md`
- `transformation-portal-architect.md`
- `README.md`
- `../copilot-instructions.md`
- `../../docs/governance/DOCUMENTATION_MAP.md`
- `../../docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md`
- `../../docs/governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md`
- `docs/guides/CUSTOM_AGENT_GUIDE.md`
- `docs/reference/AGENT_QUICK_REFERENCE.md`
