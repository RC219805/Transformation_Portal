# Custom Agent Quick Reference

Current baseline: `main` through PR #1721. Use the documentation map, May 11
refresh audit, and state audit for current repo navigation; historical
agent/RAG notes are not live instructions unless promoted by the live agent
README.

## Use The Right Agent

### `@portal-app-steward`

Use for managed browser-surface work:

- homepage, login, and portal shell behavior
- `/portal/bootstrap`, managed `/healthz`, managed recovery, and same-origin browser flow checks
- `web/secure-landing/`
- `portal.html`
- `public/portal-assets/*`
- `config/portal_asset_manifest.json`
- Node 22 frontdoor validation, browser-smoke, and selector-stability work

Example:

```text
@portal-app-steward update the managed /portal loading shell, preserve data-ui anchors, and list the exact browser validation commands
```

### `@transformation-portal-specialist`

Use for backend and governed execution work:

- `app.py`
- typed `/v1/*` behavior
- backend `/healthz`, `/ready`, and `/v1/readiness`
- Lux Depth V3
- archive-gate
- machine-mode and ingest

Example:

```text
@transformation-portal-specialist debug why /v1/jobs/{id}/events is missing the terminal done event in contract tests
```

### `@transformation-portal-architect`

Use for governance and approval-bound changes:

- dependency policy
- CI/CD policy
- documentation topology, Copilot instructions, or live custom-agent boundaries
- security posture
- route-contract or public-interface changes
- ADR interpretation

Example:

```text
@transformation-portal-architect review this proposal to change the managed /v1 proxy contract and list the required enforcement updates
```

## Quick Decision Guide

- Browser shell or frontdoor question: use `@portal-app-steward`
- Backend/orchestrator health/readiness, Lux Depth, archive, ingest, or machine-mode question: use `@transformation-portal-specialist`
- Governance, dependency, security, CI/CD, or contract decision: use `@transformation-portal-architect`

## Prompt Tips

- Name the active surface and file when you can.
- Include the failing route, selector, test, or command.
- Ask for tests and docs to move with behavioral changes.
- Call out whether you need implementation, review, or validation guidance.

## More Information

- **Full Guide**: [docs/guides/CUSTOM_AGENT_GUIDE.md](../guides/CUSTOM_AGENT_GUIDE.md)
- **Agent README**: [.github/agents/README.md](../../.github/agents/README.md)
- **Quick Start**: [QUICK_START_v2.md](../../.github/agents/QUICK_START_v2.md)
- **Copilot Instructions**: [.github/copilot-instructions.md](../../.github/copilot-instructions.md)
- **Documentation Map**: [DOCUMENTATION_MAP.md](../governance/DOCUMENTATION_MAP.md)
- **Documentation Refresh Audit**: [DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md](../governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md)
- **Documentation State Audit**: [DOCUMENTATION_STATE_AUDIT_2026-04-27.md](../governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md)

---

**Quick Start**: choose the narrowest live agent and start the prompt with its handle.
