# Transformation Portal Custom Agents

This directory is the live custom-agent configuration surface for the Transformation Portal repository.

## Live Profiles

### Transformation Portal Architect

- File: `transformation-portal-architect.md`
- Role: final authority for repository-wide contracts, dependency policy, CI/CD enforcement, security posture, and architectural direction
- Use for: ADR-bound trade-offs, public interface compatibility, workflow/packaging policy, and cross-surface governance decisions

### Transformation Portal Specialist

- File: `transformation-portal-specialist.md`
- Role: execution-focused implementation and troubleshooting agent inside current repository governance boundaries
- Use for: Lux Depth V3, portal/orchestrator, archive-gate, ingest, machine-mode, and targeted docs/tests/tooling work that stays within existing contracts

## Canonical References

The live profiles should stay aligned with:

- [AGENTS.md](../../AGENTS.md)
- [DOCUMENTATION_MAP.md](../../docs/governance/DOCUMENTATION_MAP.md)
- [CUSTOM_AGENT_GUIDE.md](../../docs/guides/CUSTOM_AGENT_GUIDE.md)
- [transformation-portal-architect.md](./transformation-portal-architect.md)
- [transformation-portal-specialist.md](./transformation-portal-specialist.md)

When drift appears, the profile files themselves are authoritative for role scope and escalation boundaries.

## Usage Guidance

On GitHub.com and in supported IDEs, select the custom agent from the agents picker/dropdown rather than assuming one universal invocation syntax. If a local workflow or tool supports prompt references, keep examples consistent with the current role boundaries:

- Specialist examples should stay within execution and troubleshooting scope
- Architect examples should stay focused on governance, contracts, and escalation decisions

## Support Material

Living support docs:

- `docs/guides/CUSTOM_AGENT_GUIDE.md`
- `QUICK_START_v2.md`
- `RAG_IMPLEMENTATION_SUMMARY.md`

Historical or milestone-style materials in `.github/agents/` should not be treated as canonical operating docs. When a report becomes historical rather than instructional, move or archive it under the repo's documentation policy instead of letting it define live agent behavior.
