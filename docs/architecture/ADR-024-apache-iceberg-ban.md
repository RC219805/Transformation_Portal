# ADR-024: Apache Iceberg Dependency Ban (Pending Supply Chain Audit)

**Status:** Approved (Mandatory Ban)
**Date:** 2026-02-11
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** ADR-023 (Spatial AI Ingest Isolation), Spatial AI Roadmap Milestone 1
**Enforcement:** `requirements/constraints.txt` hard block pin

---

## Executive Summary

**Decision:** Apache Iceberg (and PyIceberg wrapper) are BANNED from Transformation Portal until supply chain audit is complete.

**Rationale:**
- Complex Java interop via py4j (Python ↔ JVM bridge)
- Transitive dependency chain includes Java libraries (Hadoop, Parquet-MR, etc.)
- Unaudited supply chain risk for security-critical repository
- SQLite Tier A backend is sufficient for Phase I catalog requirements

**Enforcement:**
```txt
# requirements/constraints.txt
apache-iceberg>=9999.0.0  # HARD-BLOCKED
pyiceberg>=9999.0.0       # HARD-BLOCKED
```

**Unblock Criteria:** See Section 6 (Supply Chain Audit Checklist)

---

## Context

### Spatial AI Roadmap Proposal (Milestone 1)

The Spatial AI Foundation roadmap proposes a **two-tier catalog strategy**:

1. **Tier A (Phase I):** SQLite + Parquet metadata
   - Local-first, simple, no external dependencies
   - Sufficient for 10K-100K samples (Phase I scale)
   - Proven technology (SQLite in Python stdlib, Parquet via pandas/pyarrow)

2. **Tier B (Phase II+):** Apache Iceberg tables
   - Distributed, cloud-native, petabyte-scale
   - Schema evolution, time travel, snapshot isolation
   - Industry standard for ML data lakes (Netflix, Airbnb, Apple ML)

**Roadmap assumes:** Iceberg as "future-proof" choice for scale.

**Architect concern:** Supply chain risk not assessed, dependency complexity high.

### Apache Iceberg Dependency Tree

```
pyiceberg (Python client)
  ├─ py4j (Python-JVM bridge)
  │   └─ Requires JVM runtime (Spark/Java 11+)
  ├─ pyarrow (Parquet I/O)
  ├─ fsspec (filesystem abstraction)
  └─ Transitive Java dependencies:
      ├─ iceberg-core.jar
      ├─ hadoop-common.jar
      ├─ parquet-mr.jar
      ├─ aws-java-sdk (if S3 backend)
      └─ ... (50+ JARs in typical deployment)
```

**Risk Factors:**

1. **Supply Chain Complexity:**
   - PyPI package (pyiceberg) wraps Java ecosystem
   - Java JARs downloaded at runtime or install time
   - Maven Central as transitive dependency source (not audited by this repo)

2. **JVM Requirement:**
   - Adds JVM to deployment footprint (security surface expansion)
   - Version skew risk (Python 3.11 + Java 11/17/21 compatibility matrix)

3. **Reproducibility:**
   - pip freeze doesn't capture Java JAR versions
   - Docker builds need separate Java layer
   - Lock file (poetry/pdm) doesn't cover Maven artifacts

4. **Maintenance Posture:**
   - PyIceberg is relatively new (2022+, compared to SQLite's 20+ years)
   - Apache Iceberg Java core is mature, but Python wrapper less proven
   - Breaking changes more likely than SQLite stdlib

### Current Repository Dependency Posture

Per `docs/architecture/agent_governance.md` Section D:

> **Escalation to Architect REQUIRED for:**
> - Introducing new ML models, model weights, binary tools, or external runtimes
> - Any dependency with unclear license, provenance, or maintenance status

**Iceberg triggers both:**
- "External runtime" (JVM)
- "Unclear provenance" (transitive Java dependencies from Maven Central)

---

## Decision

### 1. Hard Block via Constraints

**DECISION:** Add to `requirements/constraints.txt`:

```txt
# Spatial AI Foundation: Supply chain risks pending audit
# apache-iceberg  # Java interop (py4j), supply chain risk not assessed (ADR-024 pending)
apache-iceberg>=9999.0.0
# pyiceberg  # Wrapper for apache-iceberg, same risk profile
pyiceberg>=9999.0.0
```

**Enforcement:**
```bash
# Any install attempt fails:
$ pip install pyiceberg
ERROR: ResolutionImpossible, pyiceberg requires >=0.x but constraint pins >=9999.0.0
```

**Scope:**
- Blocks `pyiceberg` (official Apache Python client)
- Blocks `apache-iceberg` (alternate name, if any)
- Does NOT block `pyarrow` (used independently for Parquet, no Iceberg dependency)

### 2. Phase I Alternative: SQLite Tier A Only

**DECISION:** Phase I catalog implementation uses SQLite exclusively.

**Implementation:**
```python
# src/transformation_portal/spatial_ai/catalog/backends.py
class StubCatalogBackend:
    """No-op catalog for testing."""

class SQLiteCatalogBackend:
    """Production Tier A: Local SQLite catalog."""
    # Uses stdlib sqlite3, no external deps

# BLOCKED until ADR-024 is superseded:
# class IcebergCatalogBackend:
#     """Production Tier B: Distributed Iceberg catalog."""
```

**Justification:**

| Scale | Samples | SQLite Performance | Iceberg Needed? |
|-------|---------|-------------------|-----------------|
| Phase I | 1K-10K | <100ms queries | ❌ No |
| Phase II | 10K-100K | <500ms queries | ❌ No |
| Phase III | 100K-1M | <2s queries | ⚠️ Maybe |
| Production | 1M-10M+ | Slow (>10s) | ✅ Yes |

**SQLite is sufficient until 1M+ samples** (Phase III at earliest).

### 3. Unblock Process (Supersede This ADR)

**To unblock Iceberg in the future:**

1. **Create ADR-024-SUPERSEDED.md** documenting:
   - Supply chain audit results (Section 6 checklist)
   - Java runtime governance policy
   - Reproducible build strategy (Docker + Java layer)
   - Migration plan (SQLite → Iceberg)

2. **Remove from constraints.txt:**
   ```diff
   - apache-iceberg>=9999.0.0
   - pyiceberg>=9999.0.0
   ```

3. **Add to the optional spatial dependency surface after a current
   dependency-lane decision:**
   ```txt
   # Optional Tier B catalog (requires JVM)
   pyiceberg>=0.6.0  # Pinned after audit
   py4j>=0.10.9.7    # Explicit pin (not transitive)
   ```

4. **CI enforcement:**
   - Add nightly job: "Iceberg smoke test" (Docker-based, JVM included)
   - Verify reproducible build (lock Java JAR versions)

**Until then:** Iceberg MUST NOT be installed.

---

## Consequences

### Positive

✅ **Supply chain risk deferred until needed**
- JVM not added to deployment until 1M+ sample scale reached
- Transitive Java dependencies audited only when value is proven

✅ **Simpler Phase I implementation**
- No py4j debugging, no JVM version skew
- SQLite in Python stdlib (zero install risk)
- Faster CI (no Iceberg setup overhead)

✅ **Defer complexity until architecture is proven**
- Phase I validates catalog schema and query patterns
- Migration to Iceberg is schema-preserving (same SQL semantics)
- No premature optimization

### Negative

⚠️ **Migration cost if scale exceeds SQLite**
- SQLite → Iceberg migration is non-trivial (table export + import)
- Downtime during migration (mitigated by snapshot/time-travel)
- Query syntax differences (SQLite SQL vs Iceberg SQL)

**Mitigation:**
- Design schema to be Iceberg-compatible from day 1
- Use SQLAlchemy or similar ORM (portable SQL)
- Test migration on synthetic dataset before production cutover

⚠️ **Potential performance cliff at 100K-1M samples**
- SQLite FTS5 (full-text search) degrades >500K rows
- Spatial index (H3/geohash) slower than Iceberg's partitioning

**Mitigation:**
- Monitor query latency in Phase II
- Trigger Iceberg audit when p95 latency >2s
- Hybrid strategy: SQLite for metadata, Parquet for timeseries data

### Neutral

- Roadmap already planned Tier A (SQLite) as default
- This ADR formalizes "Tier B deferred" as architectural policy
- No immediate impact on Phase I timeline

---

## Alternatives Considered

### Alternative 1: Allow Iceberg with Audit Requirement

**Proposal:** Unblock Iceberg immediately, require audit before first use.

**Rejected:**
- Risk of "forgot to audit" before production deployment
- Hard block is safer than "honor system"
- Phase I doesn't need Iceberg (no value for cost)

### Alternative 2: Use DuckDB Instead of Iceberg

**Proposal:** DuckDB supports Parquet + SQL + spatial extensions, no JVM.

**Analysis:**

| Feature | SQLite | DuckDB | Iceberg |
|---------|--------|--------|---------|
| Deployment | stdlib | Pip install | JVM required |
| Scale | 100K-1M | 1M-10M | 10M-1B+ |
| Spatial index | R-tree | H3 extension | Partitioned |
| ACID | Yes | Yes | Yes (via snapshots) |
| Time travel | No | No | Yes |

**Decision:** DuckDB is viable alternative for Phase II (100K-1M scale).

**Recommendation:**
- Evaluate DuckDB in Phase II before triggering Iceberg audit
- DuckDB avoids JVM but adds dependency (not stdlib)
- Defer decision until scale requirements are concrete

### Alternative 3: Cloud-Managed Catalog (BigQuery/Snowflake)

**Proposal:** Use managed data warehouse instead of self-hosted Iceberg.

**Rejected:**
- Vendor lock-in (anti-pattern for moat data)
- Export/portability concerns (query billing, egress costs)
- Spatial AI dataset is competitive advantage (must remain self-hosted)

**Exception:** If enterprise deployment requires Snowflake, integration is acceptable (but catalog must also exist in self-hosted form for R&D).

---

## Supply Chain Audit Checklist (Unblock Criteria)

**To supersede this ADR and unblock Iceberg:**

### License Audit
- [ ] Verify PyIceberg license (Apache 2.0 or compatible)
- [ ] Verify iceberg-core.jar license (Apache 2.0)
- [ ] Verify all transitive Java dependencies (no GPL/AGPL)
- [ ] Document license compatibility in ADR-024-SUPERSEDED.md

### Provenance Audit
- [ ] Verify PyPI package signature (PEP 740 attestations if available)
- [ ] Verify Maven Central JAR signatures (GPG verification)
- [ ] Document source-to-binary build reproducibility
- [ ] Identify maintainer accountability (Apache PMC governance)

### Security Audit
- [ ] Review CVE database for PyIceberg + iceberg-core (2022-present)
- [ ] Verify JVM security updates (Java 11 LTS or 17 LTS)
- [ ] Assess py4j security posture (Python ↔ JVM marshalling risks)
- [ ] Document security update policy (Dependabot for Python, manual for Java)

### Reproducibility Audit
- [ ] Create Dockerfile with pinned JVM + Java JARs
- [ ] Lock PyIceberg version + py4j version in requirements
- [ ] Verify lock file captures Java dependencies (Maven lock plugin or equivalent)
- [ ] Test reproducible build (same Dockerfile → same SHA256 artifacts)

### Integration Testing
- [ ] Implement `IcebergCatalogBackend` with feature parity to SQLite
- [ ] Add nightly CI job: Iceberg smoke test (Docker-based)
- [ ] Verify SQLite → Iceberg migration on synthetic dataset
- [ ] Benchmark query performance (compare to SQLite baseline)

### Documentation
- [ ] ADR-024-SUPERSEDED.md with audit results
- [ ] Migration guide: SQLite → Iceberg (runbook)
- [ ] Java runtime governance policy (version policy, security patching)
- [ ] Rollback plan if Iceberg causes production issues

**Sign-off Required:** Transformation Portal Architect

**Estimated Effort:** 2-3 weeks for complete audit (defer until Phase III)

---

## Migration Plan (When Audit Complete)

### Step 1: Audit and Approval
1. Complete checklist above
2. Create ADR-024-SUPERSEDED.md
3. Architect approval (explicit sign-off)

### Step 2: Dependency Integration
1. Remove ban from `requirements/constraints.txt`
2. Add to the approved optional spatial dependency surface. As of 2026-05-12,
   that surface is the `pyproject.toml` `spatial-ai` extra unless a checked-in
   spatial lock lane is introduced first:
   ```txt
   pyiceberg==0.6.1  # Audited 2026-XX-XX, see ADR-024-SUPERSEDED
   py4j==0.10.9.7    # Explicit pin
   ```
3. Update Dockerfile to include JVM layer

### Step 3: Implementation
1. Implement `IcebergCatalogBackend` in `src/transformation_portal/spatial_ai/catalog/backends.py`
2. Add registry entry: `SpatialBackendRegistry.register("iceberg", IcebergCatalogBackend)`
3. Add integration tests (nightly lane)

### Step 4: Migration Tooling
1. Create `scripts/spatial_ai/migrate_sqlite_to_iceberg.py`
2. Test on staging data (synthetic fixtures)
3. Document rollback procedure

### Step 5: Production Cutover
1. Enable Iceberg backend in production config
2. Monitor query latency (compare to SQLite baseline)
3. Keep SQLite as fallback for 1 month (dual-write)

---

## References

- Spatial AI Foundation Roadmap: `docs/spatial_ai/ROADMAP.md` (Milestone 1)
- Governance Policy: `docs/architecture/agent_governance.md` (Section D: Dependency Escalation)
- Constraint Enforcement: `requirements/constraints.txt`
- Architectural Review: `docs/architecture/SPATIAL_AI_ROADMAP_ARCHITECTURAL_REVIEW.md` (Section 4.1)

---

**Approval:** Transformation Portal Architect
**Enforcement:** Hard block in `requirements/constraints.txt` (immediate)
**Review Date:** 2026-08-11 (6 months, or when Phase III begins)
**Supersede Trigger:** Supply chain audit complete (see Section 6)
