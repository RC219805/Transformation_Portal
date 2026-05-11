# Transformation Portal Custom Agents

This directory is the live custom-agent configuration surface for the Transformation Portal repository.

Current documentation baseline: repo-wide refresh audit dated May 11, 2026,
building on `main` through PR #1721. Keep this directory aligned with
`AGENTS.md`, `.github/copilot-instructions.md`,
`docs/governance/DOCUMENTATION_MAP.md`,
`docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md`, and
`docs/governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md`.

## Live Profiles

### Transformation Portal Architect

- File: `transformation-portal-architect.md`
- Role: final authority for repository-wide contracts, dependency policy, CI/CD enforcement, security posture, and architectural direction
- Use for: ADR-bound trade-offs, public interface compatibility, workflow or packaging policy, and cross-surface governance decisions

### Portal App Steward

- File: `portal-app-steward.md`
- Role: execution-focused browser-surface steward for the managed frontdoor, portal shell, manifest-backed portal assets, and browser validation contract
- Use for: `web/secure-landing/`, `/`, `/login`, `/portal`, `/portal/bootstrap`, `/portal/assets/*`, `/portal/video/*`, managed `/healthz`, `portal.html`, `public/portal-assets/*`, selector stability, Node 22 frontdoor validation, and browser-contract docs/tests that stay within existing contracts

### Transformation Portal Specialist

- File: `transformation-portal-specialist.md`
- Role: execution-focused backend, Lux Depth, archive, ingest, and machine-mode implementation agent inside current repository governance boundaries
- Use for: `app.py`, typed `/v1/*` behavior, backend `/healthz`, `/ready`, `/v1/readiness`, Lux Depth V3, archive-gate, ingest, machine-mode, and targeted docs/tests/tooling work that stays within existing contracts

## Canonical References

The live profiles should stay aligned with:

- [AGENTS.md](../../AGENTS.md)
- [copilot-instructions.md](../copilot-instructions.md)
- [DOCUMENTATION_MAP.md](../../docs/governance/DOCUMENTATION_MAP.md)
- [DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md](../../docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md)
- [DOCUMENTATION_STATE_AUDIT_2026-04-27.md](../../docs/governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md)
- [agent_governance.md](../../docs/architecture/agent_governance.md)
- [CUSTOM_AGENT_GUIDE.md](../../docs/guides/CUSTOM_AGENT_GUIDE.md)
- [AGENT_QUICK_REFERENCE.md](../../docs/reference/AGENT_QUICK_REFERENCE.md)
- [transformation-portal-architect.md](./transformation-portal-architect.md)
- [portal-app-steward.md](./portal-app-steward.md)
- [transformation-portal-specialist.md](./transformation-portal-specialist.md)

When drift appears, the profile files themselves are authoritative for role scope and escalation boundaries.

## Usage Guidance

On GitHub.com and in supported IDEs, select the custom agent from the agents picker/dropdown rather than assuming one universal invocation syntax. If a local workflow or tool supports prompt references, keep examples consistent with the current role boundaries:

- Steward examples should stay within the managed browser boundary and portal-shell scope
- Specialist examples should stay within backend, Lux Depth, archive, ingest, and machine-mode scope
- Architect examples should stay focused on governance, contracts, and escalation decisions

Use the narrowest agent surface that matches the task:

- `@portal-app-steward` for homepage, login, portal shell, bootstrap, selector, asset-manifest, managed frontdoor health, and browser-validation work
- `@transformation-portal-specialist` for backend/orchestrator health/readiness, archive, ingest, machine-mode, and Lux Depth work
- `@transformation-portal-architect` for dependency, CI/CD, security, docs topology, route-contract, or ADR-bound decisions

## Support Material

Living support docs:

- `docs/guides/CUSTOM_AGENT_GUIDE.md`
- `docs/reference/AGENT_QUICK_REFERENCE.md`
- `QUICK_START_v2.md` — agent-selection cheatsheet
- `RAG_QUICK_START.md` — RAG usage quick-start
- `RAG_ENHANCEMENTS_GUIDE.md`, `RAG_SYSTEM_ENHANCEMENTS.md` — RAG capability reference
- `rag_system/README.md` — RAG package architecture and module reference

Historical or milestone-style materials in `.github/agents/` should not be treated as canonical operating docs. When a report becomes historical rather than instructional, move it to `_archive/` (or `rag_system/_archive/` for RAG-specific artifacts) instead of letting it define live agent behavior.
