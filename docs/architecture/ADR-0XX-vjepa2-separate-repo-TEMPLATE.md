# ADR-0XX: V-JEPA 2 Integration via Separate Repository Pattern

**Status:** TEMPLATE (for use if separate repository approach approved)
**Date:** TBD
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** V-JEPA 2 Integration Assessment, ADR-023 (Pipeline Isolation)

---

## Context

V-JEPA 2 video world model training data export capabilities were proposed for integration into the Transformation Portal repository. Architectural assessment (see `docs/architecture/V_JEPA_2_INTEGRATION_ASSESSMENT.md`) identified:

1. **Mission Misalignment:** V-JEPA 2 world model training is ML research infrastructure, orthogonal to luxury real estate rendering
2. **Capacity Constraints:** Single-maintainer operation cannot sustain two parallel major expansions
3. **Maintenance Burden:** +36% LoC, +67% dependencies, +125% CI time exceeds sustainable threshold
4. **Architectural Health:** Repository mid-transition with Spatial AI Phase I in review (PR #946)

**Architect Recommendation:** Create separate `transformation-portal-world-model` repository consuming Transformation Portal's spatial AI exports as inputs.

---

## Decision

V-JEPA 2 video world model training data export capabilities **shall be implemented in a separate repository** (`transformation-portal-world-model`), not integrated into the Transformation Portal repository.

### Repository Structure

```
transformation-portal/                    # Main Repository (Existing)
├── Mission: Luxury real estate rendering + spatial data preparation
├── Exports: Linear tensors, provenance manifests, spatial catalog
└── Users: ArchViz professionals, real estate marketers

transformation-portal-world-model/        # New Repository (V-JEPA 2)
├── Mission: Video world model training data export
├── Imports: Transformation Portal exports via defined contract
└── Users: ML researchers, foundation model trainers
```

### Integration Contract

**Transformation Portal Responsibilities:**

1. **Export Linear Tensors**
   - Format: `.npy` or `.safetensors` (float32, gamma=1.0)
   - Layout: `(H, W, 3)` for images, `(T, H, W, 3)` for video clips
   - Location: User-specified output directory
   - Metadata: Sidecar JSON with content hash, provenance chain

2. **Export Provenance Manifests**
   - Format: JSON (schema versioned)
   - Content: EXIF, lineage, decode recipe, content hash
   - Schema: Defined in `docs/spatial_ai/EXPORT_CONTRACT.md`
   - Versioning: Semantic versioning (MAJOR.MINOR.PATCH)

3. **Export Spatial Catalog**
   - Format: Parquet or SQLite (TBD in Phase I completion)
   - Content: Spatiotemporal index (property_id, visit_id, timeline_index)
   - Access: Read-only query interface
   - Schema: Defined in `docs/spatial_ai/CATALOG_SCHEMA.md`

**World Model Repository Responsibilities:**

1. **Consume Exports via Manifest**
   - Never access Transformation Portal internals directly
   - Load data via manifest-driven data loaders
   - Respect versioned schema compatibility

2. **Handle Format Changes Gracefully**
   - Pin to specific export schema version
   - Fail loudly if schema incompatible
   - Document required Transformation Portal version

3. **Independent Dependency Management**
   - No shared dependencies (except Python stdlib)
   - ML dependencies isolated to world-model repo
   - Supply chain audits independent

### Export Contract Schema (v1.0.0 Draft)

```python
# docs/spatial_ai/EXPORT_CONTRACT.md
# Schema version: 1.0.0

@dataclass
class SpatialExport:
    """Export contract from Transformation Portal to world-model consumers."""

    # Version control
    schema_version: str = "1.0.0"  # Semantic versioning

    # Data payload
    tensor_path: Path              # Linear tensor (.npy or .safetensors)
    tensor_dtype: str              # "float32" (enforced)
    tensor_shape: Tuple[int, ...]  # (H, W, 3) or (T, H, W, 3)
    gamma: float                   # Always 1.0 (linear light)

    # Provenance chain
    content_hash: str              # SHA-256 of input file
    decode_recipe: str             # Decoder version + config
    exif_metadata: Dict[str, Any]  # Camera, lens, capture settings
    lineage: List[str]             # Parent → derived chain

    # Spatiotemporal index
    property_id: Optional[str]     # Multi-visit grouping
    visit_id: Optional[str]        # Visit identifier
    timeline_index: Optional[int]  # Temporal ordering
    capture_timestamp: Optional[datetime]

    # Quality gates
    bit_depth_original: int        # 12, 14, 16 (source fidelity)
    is_derived: bool               # True if synthetic/augmented
    validation_status: str         # "passed" | "failed" | "unchecked"
```

**Breaking Changes:**

- Major version bump: Schema structure changes (fields removed/renamed)
- Minor version bump: Schema extension (new optional fields added)
- Patch version bump: Documentation clarification only

**Compatibility Guarantee:**

- Transformation Portal will maintain backward compatibility for 1 year after major version bump
- World-model repo must pin to specific major version
- Forward compatibility not guaranteed (world-model repo must validate schema version)

---

## Rationale

### 1. Clean Mission Separation

**Transformation Portal:**
- Core competency: High-fidelity rendering and spatial data preparation
- User persona: ArchViz professionals, real estate marketers
- Output artifacts: Enhanced images, PBR maps, graded videos
- Quality posture: Production-grade, stable, contract-bound

**World Model Repository:**
- Core competency: ML training data export for video world models
- User persona: ML researchers, foundation model trainers
- Output artifacts: Training tuples, token-mask schedules, motion trajectories
- Quality posture: Research-grade, experimental, evolving

**Verdict:** Orthogonal missions, different users, incompatible quality/stability requirements.

### 2. Dependency Isolation

**Transformation Portal Dependencies (Current):**
- Minimal ML: Depth models only (~20-30 packages)
- Pinned to commit SHAs (HuggingFace revision policy)
- Supply chain audit quarterly

**World Model Dependencies (Projected):**
- Heavy ML: SAM 3, tokenization models, action conditioning, optical flow (~40-50 packages)
- Evolving research ecosystem (frequent updates)
- Supply chain audit monthly (higher churn)

**Verdict:** Mixing dependencies increases CVE exposure and audit burden for production rendering toolkit.

### 3. Independent Evolution

**Scenario:** World model research discovers better tokenization strategy.

**In-Repo Integration:**
- Change requires regression testing entire Transformation Portal
- Risk of accidental coupling (shared utilities)
- Merge conflicts between rendering and world-model work

**Separate Repo:**
- World-model repo iterates freely
- Transformation Portal unaffected
- Export contract versioning handles compatibility

**Verdict:** Separate repos enable independent velocity without coordination tax.

### 4. Reduced Blast Radius

**Scenario:** Experimental world-model feature introduces CUDA IPC memory leak.

**In-Repo Integration:**
- Bug affects entire repository CI
- All contributors blocked until fix
- Production rendering stability jeopardized

**Separate Repo:**
- Bug isolated to world-model repo
- Transformation Portal unaffected
- Production rendering stability protected

**Verdict:** Isolation reduces risk of experimental work destabilizing production code.

### 5. Organizational Scalability

**Current Capacity:** Single maintainer, part-time

**In-Repo Integration:**
- Maintainer must understand rendering + world-model domains
- Context-switching overhead
- Bottleneck on all decisions

**Separate Repo:**
- Can recruit world-model specialist maintainer independently
- Each repo has focused expertise
- Parallel progress without coordination bottleneck

**Verdict:** Separate repos enable recruiting specialists without domain overlap requirement.

---

## Consequences

### Positive

✅ **Mission Clarity**
- Each repository has single, focused purpose
- No confusion about scope or user persona
- Clear ownership and decision authority

✅ **Dependency Isolation**
- Heavy ML dependencies stay in world-model repo
- Production rendering toolkit remains lightweight
- Reduced supply chain exposure

✅ **Independent Versioning**
- Breaking changes don't cascade across repos
- Each repo versioned independently
- Clear compatibility contracts

✅ **Reduced Coupling**
- No shared code surface (except export contract)
- ADR-023 isolation boundary preserved
- Minimal coordination overhead

✅ **Organizational Scalability**
- Can recruit specialist maintainers per repo
- Domain expertise focused
- Parallel progress without bottleneck

### Negative

⚠️ **Code Duplication**
- Manifest parsing duplicated (~100 LoC)
- Hashing utilities duplicated (~50 LoC)
- Export/import logic (~50 LoC)
- **Total:** ~200 LoC duplication

**Mitigation:** Acceptable duplication cost for clean separation. Alternative (shared library) creates coupling.

⚠️ **Coordination Overhead**
- Export contract changes require coordination
- Schema versioning discipline required
- Breaking changes need communication

**Mitigation:** Versioned schema + backward compatibility guarantee (1 year) + clear deprecation process.

⚠️ **Discovery Overhead**
- Users must discover two repositories
- Documentation must link clearly
- Install instructions more complex

**Mitigation:** Clear README cross-links, `transformation-portal` docs mention world-model repo, install guides reference both.

### Neutral

- More repositories to manage (marginal cost)
- Clearer architecture makes maintenance easier (net positive)
- Independent CI optimization possible (net positive)

---

## Alternatives Considered

### Alternative 1: In-Repository Integration (Rejected)

**Structure:**
```
src/transformation_portal/
  lux_depth_v3/          # Rendering
  spatial_ai/            # Spatial data prep
  world_model/           # V-JEPA 2 (NEW)
```

**Rejected Because:**
- ❌ Mission misalignment (rendering vs ML training infrastructure)
- ❌ Organizational capacity insufficient (single maintainer → 2-3 required)
- ❌ Maintenance burden unsustainable (+36% LoC, +67% deps, +125% CI time)
- ❌ Architectural health jeopardized (mid-transition, premature expansion)

**Assessment:** See `docs/architecture/V_JEPA_2_INTEGRATION_ASSESSMENT.md` for full analysis.

### Alternative 2: Monorepo with Strict Isolation (Rejected)

**Structure:**
```
transformation-portal/  (monorepo)
  packages/
    rendering/          # Lux Depth V3
    spatial-ai/         # Data preparation
    world-model/        # V-JEPA 2
```

**Rejected Because:**
- ❌ Python ecosystem lacks mature monorepo tooling (unlike Node.js/Yarn workspaces)
- ❌ Dependency isolation still requires separate `requirements/` per package (complexity)
- ❌ CI still runs all tests (no granular triggering in GitHub Actions without custom scripts)
- ❌ Same maintenance burden as in-repo integration

**Assessment:** Complexity of monorepo tooling exceeds benefits in Python ecosystem.

### Alternative 3: Shared Library for Common Code (Rejected)

**Structure:**
```
transformation-portal/              # Main repo
transformation-portal-common/       # Shared utilities (manifest, hashing)
transformation-portal-world-model/  # V-JEPA 2
```

**Rejected Because:**
- ❌ Creates coupling via shared dependency
- ❌ Breaking changes in `common` affect both repos
- ❌ Coordination overhead for `common` changes
- ❌ Violates ADR-023 isolation principle (duplication preferred over coupling)

**Assessment:** ~200 LoC duplication is cheaper than shared library coupling.

**Decision:** Simple separate repositories (Alternative chosen) balances isolation, simplicity, and maintainability.

---

## Implementation Plan

### Phase 1: Export Contract Definition (Week 1-2)

**Deliverables:**

1. ✅ `docs/spatial_ai/EXPORT_CONTRACT.md`
   - Schema v1.0.0 definition
   - Versioning policy
   - Compatibility guarantees
   - Breaking change process

2. ✅ Export contract implementation in Transformation Portal
   - Location: `src/transformation_portal/spatial_ai/export/`
   - Module: `contract.py` (schema definition)
   - Module: `exporter.py` (export utilities)
   - Tests: `tests/spatial_ai/export/` (≥90% coverage)

3. ✅ Documentation updates
   - README.md: Link to world-model repo
   - `docs/spatial_ai/INTEGRATION_GUIDE.md`: Export workflow
   - `docs/architecture/EXPORT_CONTRACT.md`: Technical spec

**Acceptance Gates:**

- Export contract schema validated (Pydantic or dataclasses)
- Backward compatibility guarantee documented
- Breaking change process documented
- Export utilities tested (round-trip export → import)

### Phase 2: World Model Repository Creation (Week 3-4)

**Deliverables:**

1. ✅ Create `transformation-portal-world-model` repository
   - GitHub repo initialization
   - README.md with mission statement
   - LICENSE (match main repo or specify research-only)
   - Basic CI scaffolding (.github/workflows/)

2. ✅ Import contract implementation
   - Module: `world_model/ingest/import_contract.py`
   - Validate export schema version compatibility
   - Fail loudly if schema incompatible
   - Tests: ≥90% coverage

3. ✅ Documentation
   - `docs/INTEGRATION.md`: How to consume Transformation Portal exports
   - `docs/SCHEMA_COMPATIBILITY.md`: Version pinning guide
   - Examples: `examples/load_spatial_export.py`

**Acceptance Gates:**

- Repository created and accessible
- Import contract validates export schema
- Round-trip export → import works end-to-end
- Documentation complete

### Phase 3: V-JEPA 2 Milestone Execution (Ongoing)

**Milestones (in world-model repo):**

1. M0: Contract lockdown (schemas)
2. M1: Durable writes (filesystem utilities)
3. M2: Deterministic hashing (reproducibility)
4. M3: Video ingest (clip extraction)
5. M4: SAM 3 perception (masklet tracking)
6. M5: Tokenization (token-mask schedules)
7. M6: Motion summaries (trajectories)
8. M7: Tier B streams (action conditioning)
9. M8: Transport layer (CUDA IPC)
10. M9: CLI integration (commands, docs)

**Execution Authority:** World-model repo maintainers (independent of Transformation Portal)

**Coordination Points:**
- Export contract changes require sync with Transformation Portal (schema versioning)
- Major version bumps require advance notice (1 month deprecation)

---

## Success Criteria

This ADR is successful when:

### Technical Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Transformation Portal LoC** | ≤22,000 | Remains stable (no V-JEPA 2 code) |
| **Transformation Portal deps** | ≤35 packages | No V-JEPA 2 dependencies added |
| **Transformation Portal CI time** | ≤10 min | Unaffected by world-model work |
| **Export contract stability** | ≥12 months | No breaking changes for 1 year |
| **Round-trip export → import** | 100% success | All exported data importable |

### Organizational Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Maintainer focus** | 1 maintainer/repo | No cross-domain expertise required |
| **PR velocity** | Independent | World-model PRs don't block rendering PRs |
| **Coordination overhead** | ≤4 hours/month | Minimal sync meetings required |

### User Experience Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Installation clarity** | 2-step process | Clear docs for both repos |
| **Discovery time** | ≤5 min | Users find world-model repo from main README |
| **Integration setup** | ≤15 min | Users export → import successfully |

---

## Migration Plan

### For Existing Users

**N/A:** No existing V-JEPA 2 integration exists, so no migration required.

### For Future Contributors

**If proposing world-model features:**

1. Check if feature belongs in `transformation-portal` or `transformation-portal-world-model`
2. Use export contract for integration (no direct code sharing)
3. Submit PRs to correct repository
4. Coordinate with Architect if export contract changes required

---

## Enforcement

### CI Gates (Transformation Portal)

**Gate 1: Dependency Freeze Check**

```yaml
# .github/workflows/dependency-audit.yml
- name: Verify No World-Model Dependencies
  run: |
    python scripts/security/verify_banned_dependencies.py
    # Fails if SAM 3, tokenization models, or V-JEPA 2 deps detected
```

**Gate 2: LoC Growth Check**

```yaml
# .github/workflows/complexity-budget.yml
- name: Verify LoC Budget
  run: |
    python scripts/metrics/check_loc_budget.py
    # Fails if total LoC exceeds 22,000 (current: ~21,300)
```

**Gate 3: Export Contract Stability**

```yaml
# .github/workflows/export-contract-check.yml
- name: Verify Export Contract Unchanged
  run: |
    python scripts/validation/check_export_contract_version.py
    # Warns if schema version bumped without ADR update
```

### Review Requirements

**Export Contract Changes:**
- MUST escalate to Architect (per governance policy)
- MUST update schema version (semantic versioning)
- MUST document breaking changes
- MUST provide 1-month deprecation for breaking changes

---

## Review and Maintenance

**Review Date:** 2027-02-15 (12 months from approval)

**Review Criteria:**
- Export contract stability maintained
- User feedback on separation model
- Organizational capacity assessment
- Dependency isolation verified

**Maintenance Responsibility:**
- **Transformation Portal Architect:** Export contract stability, versioning policy
- **World Model Repo Maintainer:** Import contract compatibility, consumer experience

**Amendments:**
- Export contract changes require ADR amendment
- Breaking schema changes require new major version ADR

---

## Approval

**Status:** TEMPLATE (for use if separate repository approach approved)

**Approver:** Transformation Portal Architect (required)

**Implementation Start:** Upon approval

**Review Interval:** 12 months (2027-02-15)

---

## References

- **Assessment:** `docs/architecture/V_JEPA_2_INTEGRATION_ASSESSMENT.md`
- **Decision Summary:** `docs/architecture/V_JEPA_2_DECISION_SUMMARY.md`
- **Quick Reference:** `docs/architecture/V_JEPA_2_QUICKREF.md`
- **Related ADRs:** ADR-023 (Pipeline Isolation), ADR-027 (Spatial AI Phase II)
- **Governance:** `docs/architecture/agent_governance.md`

---

**This ADR is binding once approved. Deviations require explicit superseding ADR with migration plan.**
