# Transformation Portal Custom Agent Guide

## Overview

The **Transformation Portal Specialist** is a custom GitHub Copilot agent designed for the repository's current governed execution surfaces.

It is no longer just a luxury-rendering or image/video pipeline assistant. The current role spans:

- Lux Depth V3 implementation and troubleshooting
- portal/orchestrator HTTP and UI surfaces
- archive-gate execution flows
- machine-mode, ingest, provenance, and evidence-adjacent tooling
- tests, docs, and developer-tooling updates that must stay aligned with current contracts

The Specialist implements inside Architect-owned constraints. When work touches dependency policy, CI/CD policy, security posture, or public contract changes, it should escalate instead of freelancing a design decision.

## What Makes This Agent Special?

Unlike general-purpose AI assistants, the Transformation Portal Specialist has:

### 1. **Domain-Specific Knowledge**
- **Lux Depth V3**: orchestration, presets, validators, run cards, artifacts, and backend-aware workflows
- **Portal / Orchestrator**: `/ready`, typed `/v1/*` endpoints, job lifecycle behavior, SSE events, and request-hardening expectations
- **Archive / Ingest / Machine Mode**: archive-gate flows, `tp.meta.machine.v1`, ingest contract behavior, and provenance-adjacent surfaces
- **Validation Discipline**: tests, docs, and commands that move with governed behavior
- **Repository Grounding**: file-path, doc, test, and command-based reasoning rather than generic advice

### 2. **Repository Architecture Understanding**
The agent is expected to understand the current modular structure:
```
app.py                                               # portal/orchestrator HTTP surface
src/transformation_portal/lux_depth_v3/orchestrator.py
src/transformation_portal/lux_depth_v3/config_resolver.py
src/transformation_portal/lux_depth_v3/pipeline_coordinator.py
src/transformation_portal/lux_depth_v3/artifact_manager.py
src/transformation_portal/lux_depth_v3/execution_engine.py
src/transformation_portal/lux_depth_v3/validators/run_card_validator.py
tools/                                               # archive, evidence, parser, and developer tooling
requirements/                                        # layered dependency source of truth
```

### 3. **Best Practices & Standards**
- **Code Quality**: narrow diffs, fast tests, contract-aware docs updates, and CI-aligned validation
- **Performance**: targeted measurement, APEX-aware thinking, and avoiding regressions hidden behind convenience
- **Contracts**: preserve current CLI, HTTP, machine-mode, ingest, and import-surface behavior unless explicitly escalated
- **Imports/Packaging**: keep optional-heavy paths lazy and respect layered requirements and install-profile boundaries

### 4. **Troubleshooting Expertise**
- Lux Depth V3 orchestration and validator failures
- portal/orchestrator contract-test failures
- archive-gate allowlist or payload rejection issues
- machine-mode and ingest schema drift
- environment-sensitive install and import-path problems

## When to Use the Custom Agent

### Perfect For:
✅ **Lux Depth V3 execution work**
- debugging orchestration issues
- updating validators, run cards, or artifact behavior
- refining implementation inside current module boundaries

✅ **Portal / orchestrator work**
- fixing route behavior without changing public contracts
- updating contract tests
- tracing request-hardening or job-lifecycle failures

✅ **Archive / machine-mode / ingest work**
- debugging archive-gate flows
- fixing typed machine-mode behavior
- keeping ingest docs, schemas, tests, and code aligned

✅ **Docs-and-tests sync work**
- updating tests when agent or contract docs change
- keeping examples aligned with current repo commands and surfaces

### Use General Copilot For:
- Basic Python syntax questions
- Generic file operations
- Simple utility functions
- Non-domain-specific tasks

### Escalate To The Architect For:
- dependency policy or lockfile changes
- CI/CD policy or workflow changes
- security posture or trust-boundary changes
- public interface or schema changes
- ADR ambiguity or architectural trade-offs

## How to Use the Agent

### Basic Usage Pattern

In GitHub Copilot Chat, prefix your prompt with the agent name:

```
@transformation-portal-specialist [your request]
```

### Example Requests

#### 1. **Implementing Features**
```
@transformation-portal-specialist Update Lux Depth V3 run-card validation so
the error includes the missing field name and add the smallest matching tests
```

**What you'll get**:
- repository-grounded implementation guidance
- affected files and tests
- smallest safe validation set
- explicit contract and compatibility notes

#### 2. **Troubleshooting**
```
@transformation-portal-specialist Debug why /v1/jobs/{id}/events is missing the
terminal done event in contract tests
```

**What you'll get**:
- likely cause analysis tied to real files and tests
- smallest safe fix path
- commands to prove the change

#### 3. **Machine-Mode / Ingest**
```
@transformation-portal-specialist Fix this tp.meta.machine.v1 payload drift and
list the docs, tests, and schema references that must move together
```

**What you'll get**:
- contract-focused change guidance
- synchronized docs/tests/schema checklist
- escalation callout if the change crosses role boundaries

## Real-World Workflow Examples

### Example 1: Lux Depth V3

**Step 1**: Ask for implementation
```
@transformation-portal-specialist I need to adjust pipeline_coordinator.py so
research-only backend acknowledgements are enforced earlier
```

**Step 2**: Review the response
- Affected Lux Depth V3 modules
- Contract and compatibility notes
- Matching tests and validation commands

**Step 3**: Iterate with follow-ups
```
@transformation-portal-specialist Now update the smallest validator/test set to
cover that acknowledgment path
```

### Example 2: Portal / Orchestrator

**Step 1**: Describe the problem
```
@transformation-portal-specialist The orchestrator is returning the wrong typed
404 envelope for an unknown /v1 route
```

**Step 2**: Get diagnostic commands
```
@transformation-portal-specialist Show me the smallest route-level tests and
runtime checks I should run before changing app.py
```

**Step 3**: Implement the fix
```
@transformation-portal-specialist Patch the typed error response without
changing /ready or job-envelope semantics
```

**Step 4**: Add test coverage
```
@transformation-portal-specialist Add the matching contract test and keep the
validation set minimal
```

### Example 3: Docs-And-Tests Sync

**Step 1**: Identify the stale surface
```
@transformation-portal-specialist The Specialist brief changed. Which tests and
agent docs should change with it?
```

**Step 2**: Apply the update
```
@transformation-portal-specialist Replace the legacy keyword-based test with a
contract test and sync the supporting agent docs
```

**Step 3**: Validate narrowly
```
@transformation-portal-specialist Run the smallest targeted validation set for
that docs-and-tests sync
```

### Automation Support

For recurring review-pattern analysis, use the skill progression automation guide:

- [Skill progression automation](./SKILL_PROGRESSION_AUTOMATION.md)

It documents the GitHub-first evidence flow, the repo-local collector, the ranking rubric, and the required degraded-mode behavior when review-thread data or memory writes fail.

The agent follows a structured response pattern:

1. **Context**: "I understand you're working with [component]..."
2. **Analysis**: "The issue is caused by / The approach would be..."
3. **Implementation**: [Code examples with explanations]
4. **Integration**: "Here's how to integrate this..."
5. **Testing**: "Test this with..."
6. **Performance**: "Expected throughput: X images/hour"
7. **Documentation**: "Update these docs..."

This ensures comprehensive, actionable responses.

## Best Practices for Agent Interaction

### DO:
✅ Be specific about which pipeline or component you're working with
✅ Provide error messages, logs, or code snippets for context
✅ Ask for complete solutions including tests and documentation
✅ Request performance benchmarks and optimization strategies
✅ Ask follow-up questions to refine the solution
✅ Mention hardware constraints (GPU, memory, CPU)

### DON'T:
❌ Ask extremely vague questions like "make it better"
❌ Omit important context (error messages, configurations)
❌ Request changes without considering existing code
❌ Ignore testing and documentation needs
❌ Skip performance considerations

### Example: Good vs. Bad Prompts

**❌ Bad Prompt:**
```
The depth thing isn't working
```

**✅ Good Prompt:**
```
@transformation-portal-specialist The /v1/jobs contract test is failing because
the typed 400 envelope changed after an app.py validation update. What is the
safest fix path and what tests should I run?
```

**❌ Bad Prompt:**
```
Add tests
```

**✅ Good Prompt:**
```
@transformation-portal-specialist Write the smallest contract-focused tests for
this Lux Depth V3 validator change and keep the validation set fast enough for
normal contributor workflows.
```

## Advanced Usage

### Chaining Agent Interactions

For complex tasks, break them into steps:

```
1. @transformation-portal-specialist Design the architecture for a new
   portal/orchestrator artifact-preview flow without changing the public /v1 job contract

2. @transformation-portal-specialist Implement the smallest compatible app.py
   and UI changes inside the approved route shape

3. @transformation-portal-specialist Add the matching contract tests and
   validation commands

4. @transformation-portal-specialist Update the docs that define the changed
   behavior and flag any Architect escalation points

5. @transformation-portal-specialist Summarize rollback risk and follow-up work
```

### Combining with Code Review

Use the agent to review code before committing:

```
@transformation-portal-specialist Review this portal/orchestrator change.
Check contract safety, validation coverage, and alignment with current
repository standards.

[paste code]
```

### Learning from the Agent

Use it as a teacher:

```
@transformation-portal-specialist Explain how the tp.meta.machine.v1 contract
is structured, what fields are stable, and which docs and tests I should read
before changing a machine-mode command
```

## Agent Limitations

The agent acknowledges when:
- GPU resources aren't available for testing
- Changes might impact production workflows (suggests careful testing)
- Real-world profiling data would be beneficial
- optional environment capabilities are not provisioned locally
- a change crosses into Architect-owned dependency, CI/CD, security, or contract decisions

It will guide you toward appropriate testing and validation strategies.

## Measuring Agent Effectiveness

Track these metrics to evaluate the agent:
- **Accuracy**: Does it provide correct, working code?
- **Completeness**: Does it include tests, docs, and error handling?
- **Context**: Does it understand repository patterns and standards?
- **Efficiency**: Does it save time compared to manual implementation?
- **Learning**: Does it help you understand the codebase better?

## Improving the Agent

The agent learns from the repository's evolving patterns. Update it when:
- New pipelines or major features are added
- Coding standards change
- New performance optimization patterns emerge
- Common issues or FAQs are identified
- Dependencies or tools are updated

To update: Edit `.github/agents/transformation-portal-specialist.md`

## Integration with Development Workflow

### Development Cycle with Agent

```
1. Design Phase
   └─ @agent: "Design architecture for [feature]"

2. Implementation Phase
   └─ @agent: "Implement [component] with tests"

3. Optimization Phase
   └─ @agent: "Profile and optimize [code]"

4. Review Phase
   └─ @agent: "Review this implementation"

5. Documentation Phase
   └─ @agent: "Document [feature] with examples"
```

### CI/CD Integration

The agent understands CI/CD constraints:
- Tests must run in < 5 minutes
- Mock heavy dependencies (ML models, FFmpeg for unit tests)
- Python 3.10/3.11/3.12 compatibility
- Linting with flake8 and pylint
- Code coverage expectations

## Resources

- **Agent File**: `.github/agents/transformation-portal-specialist.md`
- **Agent README**: `.github/agents/README.md`
- **Repository Docs**: `docs/`
- **Copilot Instructions**: `.github/copilot-instructions.md`

## Support

If the agent provides incorrect or unhelpful responses:
1. Rephrase your question with more context
2. Break complex requests into smaller steps
3. Ask for alternative approaches
4. Check if you're using the right agent for the task
5. Report persistent issues for agent improvement

---

**Remember**: The Transformation Portal Specialist is designed to be your expert partner in building world-class image and video processing pipelines. Use it actively, iterate on responses, and provide feedback to make it even better!
