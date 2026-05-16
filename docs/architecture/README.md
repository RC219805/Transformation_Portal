# Architecture Documentation Index

**Last Updated:** 2026-05-16
**Classification:** canonical architecture index
**Current overlay:** [May 12 architecture inventory](../governance/audit/architecture-inventory-2026-05-12.csv) on the preserved May 11 repo-wide documentation inventory.
**Decision-currency overlay:** [ADR_DECISION_CURRENCY_REVIEW_2026-05-16.md](ADR_DECISION_CURRENCY_REVIEW_2026-05-16.md) — per-ADR classification (implemented / active / obsolete) and the source of the 2026-05-16 renumbering of `adr-0015` → `ADR-015` and `ADR-030-materials-v3` → `ADR-048-materials-v3`, plus the relocation of the V-JEPA 2 template to `templates/`.

This directory now holds maintained architecture, ADR, roadmap, and review-required planning surfaces only. Historical PR/review records were moved to `docs/pr_archive/architecture/`; point-in-time architecture reports were moved to `docs/historical/architecture/`. No redirect stubs are retained here, so update links to the destination paths directly.

Current repository-wide documentation navigation remains governed by [docs/README.md](../README.md) and [docs/governance/DOCUMENTATION_MAP.md](../governance/DOCUMENTATION_MAP.md). This index is the architecture-specific disposition overlay for files that were in `docs/architecture` before the May 12 cleanup.

## Canonical

These files are maintained as the canonical architecture set named by the May 12 triage plan.

| File | Disposition |
| --- | --- |
| [ADR-019-depth-backend-unification.md](ADR-019-depth-backend-unification.md) | canonical |
| [ADR-030-phase2-deterministic-raw-ingest.md](ADR-030-phase2-deterministic-raw-ingest.md) | canonical |
| [APEX_WORKFLOW_DESIGN.md](APEX_WORKFLOW_DESIGN.md) | canonical |
| [ARCHITECTURE.md](ARCHITECTURE.md) | canonical |
| [PORTAL_FRONTDOOR_ROADMAP.md](PORTAL_FRONTDOOR_ROADMAP.md) | canonical |
| [PORTAL_ORCHESTRATOR_ROADMAP.md](PORTAL_ORCHESTRATOR_ROADMAP.md) | canonical |
| [README.md](README.md) | canonical |
| [ADR-015-da3-1-1-non-commercial-research-tier.md](ADR-015-da3-1-1-non-commercial-research-tier.md) | canonical (renumbered 2026-05-16 from `adr-0015`) |
| [agent_governance.md](agent_governance.md) | canonical |
| [SPEC-DH-001.md](specifications/SPEC-DH-001.md) | canonical |

## Promoted / Current Navigation

These support files remain under `docs/architecture` and are promoted into current navigation or governance cross-references.

| File | Disposition |
| --- | --- |
| [ADR-032-dependency-pinning-strategy.md](ADR-032-dependency-pinning-strategy.md) | promoted-current-navigation |
| [ADR-044-test-marker-enforcement.md](ADR-044-test-marker-enforcement.md) | promoted-current-navigation |
| [DNA_UX_UI_STRATEGY_REBASELINE_2026-04-08.md](DNA_UX_UI_STRATEGY_REBASELINE_2026-04-08.md) | promoted-current-navigation |
| [TODO_INVENTORY_QUICK_REF.md](TODO_INVENTORY_QUICK_REF.md) | promoted-current-navigation |

## Current-Support ADRs

These ADRs remain active support material. Keep them in this directory, and review them when the associated contract, dependency, test, or governance surface changes.

| File | Disposition |
| --- | --- |
| [ADR-001-APPROVAL.md](ADR-001-APPROVAL.md) | current-support-adr |
| [ADR-001-PBR-Integration-Architecture.md](ADR-001-PBR-Integration-Architecture.md) | current-support-adr |
| [ADR-017-parallelization-strategy.md](ADR-017-parallelization-strategy.md) | current-support-adr |
| [ADR-018-depth-pro-integration.md](ADR-018-depth-pro-integration.md) | current-support-adr |
| [ADR-019_FINAL_CHECKLIST.md](ADR-019_FINAL_CHECKLIST.md) | current-support-adr |
| [ADR-019_VERIFICATION_REPORT.md](ADR-019_VERIFICATION_REPORT.md) | current-support-adr |
| [ADR-020-drop-python-3.10.md](ADR-020-drop-python-3.10.md) | current-support-adr |
| [ADR-021-huggingface-revision-policy.md](ADR-021-huggingface-revision-policy.md) | current-support-adr |
| [ADR-022-v2-enhancement-optional.md](ADR-022-v2-enhancement-optional.md) | current-support-adr |
| [ADR-023-spatial-ai-ingest-isolation.md](ADR-023-spatial-ai-ingest-isolation.md) | current-support-adr |
| [ADR-024-apache-iceberg-ban.md](ADR-024-apache-iceberg-ban.md) | current-support-adr |
| [ADR-025-apex-research-workflow.md](ADR-025-apex-research-workflow.md) | current-support-adr |
| [ADR-026-apex-research-ultra.md](ADR-026-apex-research-ultra.md) | current-support-adr |
| [ADR-027-phase2-spatial-ai-extension.md](ADR-027-phase2-spatial-ai-extension.md) | current-support-adr |
| [ADR-029-execution-graph-abstraction.md](ADR-029-execution-graph-abstraction.md) | current-support-adr |
| [ADR-048-materials-v3-production-integration.md](ADR-048-materials-v3-production-integration.md) | current-support-adr (renumbered 2026-05-16 from `ADR-030` to resolve a collision with the canonical ADR-030 Phase II Deterministic RAW Ingest) |
| [ADR-031-test-dependency-isolation.md](ADR-031-test-dependency-isolation.md) | current-support-adr |
| [ADR-033-test-flake-management.md](ADR-033-test-flake-management.md) | current-support-adr |
| [ADR-034-benchmark-exclusion-from-pr-gating.md](ADR-034-benchmark-exclusion-from-pr-gating.md) | current-support-adr |
| [ADR-035-bundle-root-anchoring-invariants.md](ADR-035-bundle-root-anchoring-invariants.md) | current-support-adr |
| [ADR-036-accountability-governance-invariants.md](ADR-036-accountability-governance-invariants.md) | current-support-adr |
| [ADR-037-repo-root-contract.md](ADR-037-repo-root-contract.md) | current-support-adr |
| [ADR-038-operational-determinism-enforcement-layer.md](ADR-038-operational-determinism-enforcement-layer.md) | current-support-adr |
| [ADR-039-branch-staleness-and-selective-integration-policy.md](ADR-039-branch-staleness-and-selective-integration-policy.md) | current-support-adr |
| [ADR-040-remove-multipleof-floats-tp-meta-capture-v1.md](ADR-040-remove-multipleof-floats-tp-meta-capture-v1.md) | current-support-adr |
| [ADR-041-phase4f-external-verification-and-trust-export.md](ADR-041-phase4f-external-verification-and-trust-export.md) | current-support-adr |
| [ADR-042-scene-group-contract.md](ADR-042-scene-group-contract.md) | current-support-adr |
| [ADR-043-orchestrator-decomposition.md](ADR-043-orchestrator-decomposition.md) | current-support-adr |
| [ADR-045-monolith-decomposition-residuals.md](ADR-045-monolith-decomposition-residuals.md) | current-support-adr |
| [ADR-046-app-path-security-helper-extraction.md](ADR-046-app-path-security-helper-extraction.md) | current-support-adr |
| [ADR-047-managed-sam2-checkpoint-security-extraction.md](ADR-047-managed-sam2-checkpoint-security-extraction.md) | current-support-adr |
| [templates/ADR-vjepa2-separate-repo-TEMPLATE.md](templates/ADR-vjepa2-separate-repo-TEMPLATE.md) | template (relocated 2026-05-16 out of the numbered ADR series; not a real ADR) |

## Review-Required Planning Docs

These non-canonical planning, evidence, and support docs remain in `docs/architecture` for now. They need owner review before future promotion, consolidation, or archival.

| File | Disposition |
| --- | --- |
| [APEX_END_TO_END_ARCHITECTURE.md](APEX_END_TO_END_ARCHITECTURE.md) | review-required-planning |
| [APEX_PHASE3_ARCHITECTURE.md](APEX_PHASE3_ARCHITECTURE.md) | review-required-planning |
| [ARCHITECTURAL_CONTEXT_IMPLEMENTATION.md](ARCHITECTURAL_CONTEXT_IMPLEMENTATION.md) | review-required-planning |
| [ARCHITECTURAL_CONTEXT_INTEGRATION.md](ARCHITECTURAL_CONTEXT_INTEGRATION.md) | review-required-planning |
| [ARCHITECTURAL_WORKFLOW.md](ARCHITECTURAL_WORKFLOW.md) | review-required-planning |
| [ARCHITECTURE_PHILOSOPHY.md](ARCHITECTURE_PHILOSOPHY.md) | review-required-planning |
| [ARCHITECT_DIRECTIVE_CI_HEALTH.md](ARCHITECT_DIRECTIVE_CI_HEALTH.md) | review-required-planning |
| [BRANCH_CLEANUP_QUICKREF.md](BRANCH_CLEANUP_QUICKREF.md) | review-required-planning |
| [BRANCH_HYGIENE_ASSESSMENT_2026-02-16.md](BRANCH_HYGIENE_ASSESSMENT_2026-02-16.md) | review-required-planning |
| [CI_003_ARCHITECT_ASSESSMENT.md](CI_003_ARCHITECT_ASSESSMENT.md) | review-required-planning |
| [CODEBASE_AUDIT_2026_Q1.md](CODEBASE_AUDIT_2026_Q1.md) | review-required-planning |
| [DEVELOPMENT_ROADMAP_2026_Q2.md](DEVELOPMENT_ROADMAP_2026_Q2.md) | review-required-planning |
| [DUAL_REQUEST_ARCHITECT_DECISION.md](DUAL_REQUEST_ARCHITECT_DECISION.md) | review-required-planning |
| [DUAL_REQUEST_QUICK_REF.md](DUAL_REQUEST_QUICK_REF.md) | review-required-planning |
| [ML_CI_OPTIMIZATION_QUICKREF.md](ML_CI_OPTIMIZATION_QUICKREF.md) | review-required-planning |
| [ML_CI_OPTIMIZATION_STRATEGIC_REVIEW.md](ML_CI_OPTIMIZATION_STRATEGIC_REVIEW.md) | review-required-planning |
| [MONOLITH_DECOMPOSITION_TARGETS.md](MONOLITH_DECOMPOSITION_TARGETS.md) | review-required-planning |
| [NIGHTLY_CHECKS_POSTMORTEM_2026-02-02.md](NIGHTLY_CHECKS_POSTMORTEM_2026-02-02.md) | review-required-planning |
| [PBR-Integration-Final-Review.md](PBR-Integration-Final-Review.md) | review-required-planning |
| [PBR-Integration-Implementation-Roadmap.md](PBR-Integration-Implementation-Roadmap.md) | review-required-planning |
| [PBR-Integration-Quick-Reference.md](PBR-Integration-Quick-Reference.md) | review-required-planning |
| [PBR-Integration-Visual-Architecture.md](PBR-Integration-Visual-Architecture.md) | review-required-planning |
| [PBR_IMPLEMENTATION_REVIEW_2026-02-01.md](PBR_IMPLEMENTATION_REVIEW_2026-02-01.md) | review-required-planning |
| [PHASE1-3_OPTIMIZATION_REVIEW.md](PHASE1-3_OPTIMIZATION_REVIEW.md) | review-required-planning |
| [PHASE2-AUTHORIZATION.md](PHASE2-AUTHORIZATION.md) | review-required-planning |
| [PHASE2_DOCUMENTATION_INDEX.md](PHASE2_DOCUMENTATION_INDEX.md) | review-required-planning |
| [PHASE2_IMPLEMENTATION_PLAN.md](PHASE2_IMPLEMENTATION_PLAN.md) | review-required-planning |
| [PHASE2_QUICKREF.md](PHASE2_QUICKREF.md) | review-required-planning |
| [PHASE4_PORTAL_INTEGRATION.md](PHASE4_PORTAL_INTEGRATION.md) | review-required-planning |
| [PHASE5_STAGEGRAPH_INTEGRATION.md](PHASE5_STAGEGRAPH_INTEGRATION.md) | review-required-planning |
| [PHASE6_API_CONTRACT.md](PHASE6_API_CONTRACT.md) | review-required-planning |
| [PHASE_C_ARCHITECTURAL_DECISION.md](PHASE_C_ARCHITECTURAL_DECISION.md) | review-required-planning |
| [PORTAL_EDGE_HARDENING_IMPLEMENTATION_STANDARD.md](PORTAL_EDGE_HARDENING_IMPLEMENTATION_STANDARD.md) | review-required-planning |
| [PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md](PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md) | review-required-planning |
| [PORTAL_OPERATOR_CONSOLE_MODERNIZATION_RFC.md](PORTAL_OPERATOR_CONSOLE_MODERNIZATION_RFC.md) | review-required-planning |
| [SPATIAL_AI_PHASE_I_AUTHORIZATION.md](SPATIAL_AI_PHASE_I_AUTHORIZATION.md) | review-required-planning |
| [SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md](SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md) | review-required-planning |
| [TEMPORAL_ARCHITECTURE_QUICKREF.md](TEMPORAL_ARCHITECTURE_QUICKREF.md) | review-required-planning |
| [TRANCHE_EXECUTION_PLAN.md](TRANCHE_EXECUTION_PLAN.md) | review-required-planning |
| [TRANCHE_PHASE1_QUALITY_REVIEW.md](TRANCHE_PHASE1_QUALITY_REVIEW.md) | review-required-planning |
| [TRANCHE_PHASE2_EXECUTION_PLAN.md](TRANCHE_PHASE2_EXECUTION_PLAN.md) | review-required-planning |
| [TRANCHE_PHASE2_QUICKREF.md](TRANCHE_PHASE2_QUICKREF.md) | review-required-planning |
| [V2_0_0_IMPLEMENTATION_PLAN.md](V2_0_0_IMPLEMENTATION_PLAN.md) | review-required-planning |
| [V2_0_0_RELEASE_REVIEW.md](V2_0_0_RELEASE_REVIEW.md) | review-required-planning |
| [V_JEPA_2_DECISION.md](V_JEPA_2_DECISION.md) | review-required-planning |
| [V_JEPA_2_INTEGRATION_ASSESSMENT.md](V_JEPA_2_INTEGRATION_ASSESSMENT.md) | review-required-planning |
| [V_JEPA_2_QUICKREF.md](V_JEPA_2_QUICKREF.md) | review-required-planning |
| [ANALYSIS-DH-001.md](analysis/ANALYSIS-DH-001.md) | review-required-analysis |
| [ci_gate_pattern.md](ci_gate_pattern.md) | review-required-planning |
| [performance_ledger_v1.7_review.md](performance_ledger_v1.7_review.md) | review-required-planning |
| [phase3_l1_cache_invariants.md](phase3_l1_cache_invariants.md) | review-required-planning |
| [transformation_portal_roadmap_rereview_2026-04-07.md](transformation_portal_roadmap_rereview_2026-04-07.md) | review-required-planning |

## Moved Historical Files

These files are retained as historical evidence outside maintained architecture navigation. Their old locations are preserved in the May 12 CSV ledger; use the destination links here for current references.

### PR And Review Evidence

| File | Current path | Disposition |
| --- | --- | --- |
| PHASE2_PR_SUMMARY.md | [PHASE2_PR_SUMMARY.md](../pr_archive/architecture/PHASE2_PR_SUMMARY.md) | moved-to-pr-archive |
| PR778_ARCHITECTURAL_ASSESSMENT.md | [PR778_ARCHITECTURAL_ASSESSMENT.md](../pr_archive/architecture/PR778_ARCHITECTURAL_ASSESSMENT.md) | moved-to-pr-archive |
| PR_804_GOVERNANCE_ANALYSIS.md | [PR_804_GOVERNANCE_ANALYSIS.md](../pr_archive/architecture/PR_804_GOVERNANCE_ANALYSIS.md) | moved-to-pr-archive |
| PR_845_ARCHITECTURAL_REVIEW.md | [PR_845_ARCHITECTURAL_REVIEW.md](../pr_archive/architecture/PR_845_ARCHITECTURAL_REVIEW.md) | moved-to-pr-archive |
| PR_906_ARCHITECTURAL_REVIEW.md | [PR_906_ARCHITECTURAL_REVIEW.md](../pr_archive/architecture/PR_906_ARCHITECTURAL_REVIEW.md) | moved-to-pr-archive |
| PR_906_FINAL_VERIFICATION.md | [PR_906_FINAL_VERIFICATION.md](../pr_archive/architecture/PR_906_FINAL_VERIFICATION.md) | moved-to-pr-archive |
| PR_906_FOLLOWUP_DECISIONS.md | [PR_906_FOLLOWUP_DECISIONS.md](../pr_archive/architecture/PR_906_FOLLOWUP_DECISIONS.md) | moved-to-pr-archive |
| PR_906_FOLLOWUP_ISSUES.md | [PR_906_FOLLOWUP_ISSUES.md](../pr_archive/architecture/PR_906_FOLLOWUP_ISSUES.md) | moved-to-pr-archive |
| PR_906_P1B_FIX_VERIFICATION.md | [PR_906_P1B_FIX_VERIFICATION.md](../pr_archive/architecture/PR_906_P1B_FIX_VERIFICATION.md) | moved-to-pr-archive |
| PR_932_ARCHITECTURAL_VERIFICATION.md | [PR_932_ARCHITECTURAL_VERIFICATION.md](../pr_archive/architecture/PR_932_ARCHITECTURAL_VERIFICATION.md) | moved-to-pr-archive |
| PR_932_CRITICAL_FIXES_VERIFICATION.md | [PR_932_CRITICAL_FIXES_VERIFICATION.md](../pr_archive/architecture/PR_932_CRITICAL_FIXES_VERIFICATION.md) | moved-to-pr-archive |
| PR_932_DOCUMENTATION_ALIGNMENT_COMPLETE.md | [PR_932_DOCUMENTATION_ALIGNMENT_COMPLETE.md](../pr_archive/architecture/PR_932_DOCUMENTATION_ALIGNMENT_COMPLETE.md) | moved-to-pr-archive |
| PR_ARCHITECTURAL_ASSESSMENT_793_792_790.md | [PR_ARCHITECTURAL_ASSESSMENT_793_792_790.md](../pr_archive/architecture/PR_ARCHITECTURAL_ASSESSMENT_793_792_790.md) | moved-to-pr-archive |
| PR_REVIEW_ACTION_ITEMS.md | [PR_REVIEW_ACTION_ITEMS.md](../pr_archive/architecture/PR_REVIEW_ACTION_ITEMS.md) | moved-to-pr-archive |
| pr-review-845-883-superseded.md | [pr-review-845-883-superseded.md](../pr_archive/architecture/pr-review-845-883-superseded.md) | moved-to-pr-archive |

### Point-In-Time Architecture Reports

| File | Current path | Disposition |
| --- | --- | --- |
| DECISION_SUMMARY_2026-02-15.txt | [DECISION_SUMMARY_2026-02-15.txt](../historical/architecture/DECISION_SUMMARY_2026-02-15.txt) | moved-to-historical |
| ISSUE_COMPLETION_AUDIT_879_852.md | [ISSUE_COMPLETION_AUDIT_879_852.md](../historical/architecture/ISSUE_COMPLETION_AUDIT_879_852.md) | moved-to-historical |
| MATERIALS_V3_ARCHITECTURAL_APPROVAL.md | [MATERIALS_V3_ARCHITECTURAL_APPROVAL.md](../historical/architecture/MATERIALS_V3_ARCHITECTURAL_APPROVAL.md) | moved-to-historical |
| ML_CI_OPTIMIZATION_CHECKLIST.md | [ML_CI_OPTIMIZATION_CHECKLIST.md](../historical/architecture/ML_CI_OPTIMIZATION_CHECKLIST.md) | moved-to-historical |
| NIGHTLY_CI_RESTORATION_REPORT.md | [NIGHTLY_CI_RESTORATION_REPORT.md](../historical/architecture/NIGHTLY_CI_RESTORATION_REPORT.md) | moved-to-historical |
| PBR-Integration-Executive-Summary.md | [PBR-Integration-Executive-Summary.md](../historical/architecture/PBR-Integration-Executive-Summary.md) | moved-to-historical |
| PHASE2_PLANNING_SUMMARY.md | [PHASE2_PLANNING_SUMMARY.md](../historical/architecture/PHASE2_PLANNING_SUMMARY.md) | moved-to-historical |
| PHASE3_L1_IMPLEMENTATION_SUMMARY.md | [PHASE3_L1_IMPLEMENTATION_SUMMARY.md](../historical/architecture/PHASE3_L1_IMPLEMENTATION_SUMMARY.md) | moved-to-historical |
| PHASE4_CAPTURE_PROVENANCE_FLAWLESS_ROADMAP.md | [PHASE4_CAPTURE_PROVENANCE_FLAWLESS_ROADMAP.md](../historical/architecture/PHASE4_CAPTURE_PROVENANCE_FLAWLESS_ROADMAP.md) | moved-to-historical |
| REPOSITORY_HEALTH_REPORT_2026-02-03.md | [REPOSITORY_HEALTH_REPORT_2026-02-03.md](../historical/architecture/REPOSITORY_HEALTH_REPORT_2026-02-03.md) | moved-to-historical |
| TEMPORAL_ARCHITECTURE_SUMMARY.md | [TEMPORAL_ARCHITECTURE_SUMMARY.md](../historical/architecture/TEMPORAL_ARCHITECTURE_SUMMARY.md) | moved-to-historical |
| TRANCHE_PHASE1_EXECUTIVE_SUMMARY.md | [TRANCHE_PHASE1_EXECUTIVE_SUMMARY.md](../historical/architecture/TRANCHE_PHASE1_EXECUTIVE_SUMMARY.md) | moved-to-historical |

## Delete Candidates

Deletion is limited to local or tracked non-documentation artifacts. Historical docs are moved, not deleted.

| Former path | Disposition | Reason |
| --- | --- | --- |
| `docs/architecture/.DS_Store` | delete | Untracked local metadata file; not documentation or audit evidence. |
