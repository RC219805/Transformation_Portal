# PR #655 Remediation Checklist

**Document**: `docs/DEPTH_ESTIMATION_ANALYSIS.md`
**Commit**: `5afa0329`
**Status**: 🔴 **BLOCKED - Remediation Required**
**Owner**: TBD
**Due Date**: TBD
**Review Document**: [PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md](./PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md)

---

## Phase 1: Critical Fixes (REQUIRED FOR MERGE)

**Estimated Effort**: 4-6 hours
**Priority**: 🔴 CRITICAL
**Blocks**: PR #655 merge

### 1.1 Remove or Qualify Unprovable Claims

- [ ] **Line 37**: Replace "1,348 tests passing" with link to CI or specific V2 test count
  - Option A: Link to CI badge `[![Tests](https://github.com/jknohr/transformation-portal/workflows/tests/badge.svg)](../../.github/workflows/ci.yml)`
  - Option B: State specific count: "Comprehensive test coverage (44 test suites covering...)"
  - **Assignee**: ____________
  - **Verification**: Run `find lux_depth_v2 -name "test_*.py" | wc -l` confirms count

- [ ] **Lines 39, 251**: Remove "127-400 images/hour" or add benchmark methodology
  - Option A: Remove claim entirely
  - Option B: Add methodology block (see ADR-004, Section 5)
  - **Assignee**: ____________
  - **Verification**: Benchmark script in `bench/` can reproduce numbers

- [ ] **Line 38**: Link CVE-2024-27763 claim to SECURITY.md
  - Replace: `- ✅ Security-hardened (CVE-2024-27763 mitigated)`
  - With: `- ✅ Security-hardened ([CVE-2024-27763 mitigated](../../SECURITY.md#cve-2024-27763-basicsr-command-injection-vulnerability))`
  - **Assignee**: ____________
  - **Verification**: Link resolves to SECURITY.md section

### 1.2 Fix License Terminology Conflict

- [ ] **Lines 88-92**: Rename "COMMERCIAL USE" to match code contract
  - Replace model variants section with two subsections:
    - "Non-Commercial Models (CC-BY-NC-4.0)"
    - "Commercial-Friendly Models (Apache 2.0)"
  - Remove ambiguous "COMMERCIAL USE" annotation
  - **Template**: See [Appendix B in review doc](./PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md#appendix-b-aligned-license-documentation-template)
  - **Assignee**: ____________
  - **Verification**: grep "COMMERCIAL USE" returns no matches

- [ ] **Add License Compliance Warning**
  - Insert after model variants table
  - Template:
    ```markdown
    ⚠️ **License Compliance Note**: Non-commercial models (CC-BY-NC-4.0) require explicit permission for commercial deployment. See [lux_depth_v3/SECURITY.md](../../lux_depth_v3/SECURITY.md) for license validation workflow.
    ```
  - **Assignee**: ____________
  - **Verification**: Warning is visible in rendered markdown

### 1.3 Add Output Format Contract Section

- [ ] **Create new Section 6**: "Output Format Contract"
  - Insert before "Recommended Configuration Summary" (current Section 7)
  - Include 3 subsections:
    - 6.1 Output Artifacts by Purpose (table)
    - 6.2 Critical Usage Warnings
    - 6.3 Metric Depth Output Constraints
  - **Template**: See [Issue 4.1 in review doc](./PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md#issue-41-no-separation-of-preview-vs-contract-vs-research-formats)
  - **Assignee**: ____________
  - **Verification**: Section is in table of contents, table renders correctly

- [ ] **Add "DO NOT use preview PNG" warning**
  - Must be visually prominent (⚠️ emoji, bold text)
  - Include code example showing CORRECT vs WRONG usage
  - **Assignee**: ____________
  - **Verification**: Warning appears in Section 6.2

- [ ] **Document metric depth constraints**
  - List 3 requirements: model variant, intrinsics, scene assumptions
  - List 3 failures: sky, reflections, monocular limits
  - **Assignee**: ____________
  - **Verification**: Constraints appear in Section 6.3

### 1.4 Fix CLI Command Name Conflicts

- [ ] **Rename duplicate `benchmark` commands in CLI**
  - File: `lux_depth_v3/cli.py`
  - Line 440: Rename to `benchmark_inference` or use subcommand `benchmark inference`
  - Line 620: Rename to `benchmark_quality` or use subcommand `benchmark quality`
  - **Assignee**: ____________
  - **Verification**: `lux-depth-v3 --help` shows both commands without conflicts

- [ ] **Update documentation to match final CLI command names**
  - Search doc for all `benchmark` references
  - Update to match actual command registration
  - **Assignee**: ____________
  - **Verification**: grep "benchmark" in doc matches `lux-depth-v3 --help` output

---

## Phase 2: Quality Enhancements (RECOMMENDED)

**Estimated Effort**: 6-8 hours
**Priority**: 🟡 HIGH
**Blocks**: Promotion to primary documentation

### 2.1 Add "Known Failure Modes" Section

- [ ] **Create new Section 7**: "Known Failure Modes and Limitations"
  - Insert after "Quality Validation Metrics"
  - Include 3 subsections:
    - 7.1 Monocular Depth Limitations (table)
    - 7.2 Configuration Pitfalls
    - 7.3 Metric Depth Constraints
  - **Template**: See [Issue 5.1 in review doc](./PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md#issue-51-no-known-limitations-section)
  - **Assignee**: ____________
  - **Verification**: Section is in table of contents, renders correctly

- [ ] **Document 5 common failure modes**
  - Texture embossing (flat walls with patterns)
  - Sky instability (gradient/noise in sky)
  - Reflection mirroring (water/glass)
  - Vegetation saturation (trees/bushes)
  - Uniform wall collapse (blank walls)
  - **Assignee**: ____________
  - **Verification**: All 5 modes in table with mitigation strategies

- [ ] **Add configuration pitfall examples**
  - Double edge-snapping (code example: WRONG vs CORRECT)
  - Texture imprinting (low overlap example)
  - **Assignee**: ____________
  - **Verification**: Code examples render correctly, syntax highlighting works

### 2.2 Add Benchmark Methodology Block

- [ ] **Expand Performance Benchmarks section (3.3)**
  - Add methodology subsection before table
  - Include 7 required context items:
    1. Hardware specification
    2. Warmup iterations
    3. Measurement iterations
    4. Precision (FP16/FP32)
    5. Batch size
    6. Input resolution
    7. Link to benchmark script
  - **Template**: See [Issue 6.1 in review doc](./PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md#issue-61-performance-table-lacks-reproducibility-context)
  - **Assignee**: ____________
  - **Verification**: All 7 context items present

- [ ] **Add 95th percentile and throughput columns to table**
  - Existing table has 5 columns (Model, Device, Resolution, Time, Notes)
  - Add: 95th %ile, Throughput (images/min)
  - **Assignee**: ____________
  - **Verification**: Table has 7 columns, calculations correct

### 2.3 Qualify Validation Metrics

- [ ] **Update Edge Alignment section (6.1)**
  - Replace absolute threshold `> 0.5` with context-sensitive thresholds
  - Add calibration guidance
  - **Template**: See [Issue 7.1 in review doc](./PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md#issue-71-edge-alignment-and-seam-energy-thresholds-are-context-sensitive)
  - **Assignee**: ____________
  - **Verification**: 3 context-specific thresholds listed

- [ ] **Update Seam Energy section (6.2)**
  - Replace absolute threshold `< 1.2` with context-sensitive thresholds
  - Add "skip if low variance" logic
  - Document when validation is not applicable (sky, uniform regions)
  - **Assignee**: ____________
  - **Verification**: Variance check example present

---

## Phase 3: Structural Refactoring (FUTURE WORK)

**Estimated Effort**: 12-16 hours
**Priority**: 🟢 MEDIUM
**Blocks**: None (optimization)

### 3.1 Split into Focused Documents

- [ ] **Create `docs/depth_pipeline/` directory**
  - **Assignee**: ____________

- [ ] **Extract `overview.md`** (High-level capabilities, system comparison)
  - Sections: Executive Summary, System Overview (1.1-1.3), Quality Comparison (3.1-3.2)
  - **Assignee**: ____________

- [ ] **Extract `recipes.md`** (Configuration examples)
  - Sections: Optimal Configuration (2.1-2.3), Recommended Configuration Summary (7.1-7.4)
  - **Assignee**: ____________

- [ ] **Extract `validation.md`** (Quality metrics and thresholds)
  - Sections: Quality Validation Metrics (6.1-6.3), Known Failure Modes (7.1-7.3)
  - **Assignee**: ____________

- [ ] **Extract `benchmarks.md`** (Performance data with methodology)
  - Sections: Performance Benchmarks (3.3), Critical Configuration Parameters (4.1-4.4)
  - **Assignee**: ____________

- [ ] **Extract `failure_modes.md`** (Known limitations and workarounds)
  - Sections: Common Quality Issues (5.1-5.4), Known Failure Modes (7.1-7.3)
  - **Assignee**: ____________

- [ ] **Create index/landing page**
  - Link to all 5 focused documents
  - Add navigation guide
  - **Assignee**: ____________

- [ ] **Update all cross-references**
  - Replace internal section links with inter-document links
  - Verify all links resolve
  - **Assignee**: ____________

---

## Verification & Sign-Off

### Phase 1 Verification (Required for Merge)

- [ ] **All Phase 1 tasks complete** (Verified by: ____________)
- [ ] **No unprovable claims without citations** (grep check)
- [ ] **License terminology aligns with code** (manual review)
- [ ] **Output format contract present** (Section 6 exists)
- [ ] **CLI commands verified** (`lux-depth-v3 --help` matches doc)

**Phase 1 Sign-Off**: ____________ (Architect) | Date: ____________

### Phase 2 Verification (Required for Promotion)

- [ ] **All Phase 2 tasks complete** (Verified by: ____________)
- [ ] **Failure modes documented** (5+ modes in table)
- [ ] **Benchmark methodology present** (7 context items)
- [ ] **Validation metrics qualified** (context-sensitive thresholds)

**Phase 2 Sign-Off**: ____________ (Architect) | Date: ____________

### Phase 3 Verification (Optional)

- [ ] **All Phase 3 tasks complete** (Verified by: ____________)
- [ ] **5 focused documents created** (overview, recipes, validation, benchmarks, failure_modes)
- [ ] **All cross-references updated** (no broken links)

**Phase 3 Sign-Off**: ____________ (Architect) | Date: ____________

---

## Related Documents

- [PR #655 Architectural Review](./PR_655_DEPTH_DOC_ARCHITECTURAL_REVIEW.md) - Full technical analysis
- [PR #655 Executive Summary](./PR_655_EXECUTIVE_SUMMARY.md) - High-level summary
- [ADR-004: Documentation Quality Standards](./adrs/ADR-004-DOCUMENTATION-QUALITY-STANDARDS.md) - Long-term standards
- [SECURITY.md](../../SECURITY.md) - CVE-2024-27763 details

---

## Notes & Issues

**Blocker Log**:
- (Add any blockers encountered during remediation)

**Clarifications Needed**:
- (Add any questions or ambiguities)

**Decisions Made**:
- (Document any architectural decisions made during remediation)

---

**Checklist Created**: 2026-01-05
**Last Updated**: 2026-01-05
**Status**: 🔴 **BLOCKED - Awaiting Phase 1 Remediation**
