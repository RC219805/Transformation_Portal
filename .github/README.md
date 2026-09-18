# GitHub Repository Automation

Use this directory for maintained GitHub configuration and agent entry points.
The [forensic audit](../docs/ci/GITHUB_FORENSIC_AUDIT_2026-09-18.md) records the
2026-09-18 UTC source and remote-state review.

| Surface | Authority and maintenance boundary |
| --- | --- |
| `workflows/*.yml` | Active GitHub Actions. Start with [workflow guidance](workflows/README.md) and the [workflow matrix](../docs/ci/WORKFLOW_MATRIX.md). |
| `dependabot.yml` | Update policy, checked by `scripts/validation/check_dependabot_config.py`. |
| `CODEOWNERS` | Review routing; remote branch protection determines whether code-owner approval is required. |
| `copilot-instructions.md`, `agents/` | Live agent profiles and retrieval tooling; start with [agent guidance](agents/README.md). |
| `apex-workflow-orchestrator.copilot-agent.yml` | APEX helper guidance subordinate to current execution, license, and performance contracts. |
| `copilot-firewall.yml` | Repository Copilot network guidance; it does not prove remote runtime enforcement. |
| `dependency-submission-config.yml` | Configuration intent; `workflows/dependency-submission.yml` owns actual submission behavior. |
| `pre-commit-hook.sh` | Compatibility hook; use `make install-hooks` for maintained hook installation. |
| `agents/**/_archive/`, `workflows-disabled/` | Historical material, excluded from live agent authority or active workflow execution. |
| PR approval, disk-space, and old workflow reports | Dated evidence, never approval or acceptance evidence for a new change. |

Keep required check names stable, preserve the post-CI firewall's trusted
upstream checkout, and validate changes with `make validate-ci` plus the
focused contract tests. Read current GitHub settings before relying on any
branch-protection snapshot.
