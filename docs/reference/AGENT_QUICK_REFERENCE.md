# Custom Agent Quick Reference

## Use The Right Agent

### `@portal-app-steward`

Use for managed browser-surface work:

- homepage, login, and portal shell behavior
- `/portal/bootstrap`, managed recovery, and same-origin browser flow checks
- `web/secure-landing/`
- `portal.html`
- `public/portal-assets/*`
- `config/portal_asset_manifest.json`
- browser-smoke and selector-stability work

Example:

```text
@portal-app-steward update the managed /portal loading shell, preserve data-ui anchors, and list the exact browser validation commands
```

### `@transformation-portal-specialist`

Use for backend and governed execution work:

- `app.py`
- typed `/v1/*` behavior
- `/ready`
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
- security posture
- route-contract or public-interface changes
- ADR interpretation

Example:

```text
@transformation-portal-architect review this proposal to change the managed /v1 proxy contract and list the required enforcement updates
```

## Quick Decision Guide

- Browser shell or frontdoor question: use `@portal-app-steward`
- Backend/orchestrator, Lux Depth, archive, ingest, or machine-mode question: use `@transformation-portal-specialist`
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

---

**Quick Start**: choose the narrowest live agent and start the prompt with its handle.
