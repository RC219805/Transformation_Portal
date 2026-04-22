# RAG Support Layer Summary

**Status**: Active supporting capability
**Scope**: Repository retrieval and evidence-grounding support for custom agents
**Note**: This file describes the retrieval support layer, not the full Specialist role contract

---

## Why This Exists

The Transformation Portal Specialist now operates across a broader governed repository surface than the original rendering-pipeline-only framing. The retrieval layer remains useful, but it is now a supporting capability rather than the definition of the role.

Use this document to understand the repository-retrieval support system and its boundaries. Use `transformation-portal-specialist.md` as the authoritative source for Specialist scope, escalation rules, and current execution surfaces.

## Current Position In The Agent Stack

The Specialist brief governs:

- Lux Depth V3 execution work
- portal/orchestrator implementation and troubleshooting
- archive-gate flows
- machine-mode and ingest tooling
- docs/tests/developer-tooling changes inside current governance boundaries

The RAG layer supports that work by helping the agent retrieve:

- repository files and docs
- tests and validation entrypoints
- contract references
- existing module and route patterns

It should improve grounding. It should not override repo governance, contracts, or escalation rules.

## What The Retrieval Layer Provides

The repository retrieval tooling under `.github/agents/rag_system/` exists to support:

- repository indexing
- targeted retrieval over docs, code, tests, and agent materials
- citation-friendly context assembly
- reusable prompt/template support for common tasks

This support layer is useful when the Specialist needs repository evidence quickly, but the current Specialist contract should not depend on narrow old examples like depth-only effects, LUT workflows, or rendering-only optimizations.

## What Changed Since The Original Specialist Draft

The earlier Specialist materials and tests overfit to a narrower role:

- luxury-rendering and image/video pipeline emphasis
- older depth-pipeline examples
- technology checklist assertions such as PyTorch, FFmpeg, NumPy, and Pillow mentions

The current repo contract is broader. The synced test surface now validates:

- frontmatter and title shape
- required Specialist sections
- governed operational surfaces
- escalation domains
- existence of referenced governance and contract files

That makes the test suite a contract check for the current brief, not a wording freeze for obsolete domain language.

## Practical Guidance

When updating agent materials:

- keep retrieval support docs distinct from role-definition docs
- do not let historical RAG examples redefine the current Specialist contract
- sync `README.md`, `QUICK_START_v2.md`, `transformation-portal-specialist.md`, and `tests/test_custom_agent_config.py` together when the Specialist role changes
- prefer repository-grounded examples covering Lux Depth V3, portal/orchestrator, archive gates, and machine-mode/ingest surfaces

## Related Files

- `transformation-portal-specialist.md`
- `README.md`
- `QUICK_START_v2.md`
- `.github/agents/rag_system/README.md`
- `tests/test_custom_agent_config.py`
