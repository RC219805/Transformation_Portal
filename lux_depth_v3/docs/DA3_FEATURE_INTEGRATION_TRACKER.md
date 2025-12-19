# DA3 Feature Integration Tracker

**Status:** Planning Complete | Implementation Pending  
**Last Updated:** December 19, 2025

---

## Executive Summary

Based on comprehensive gap analysis of the official Depth Anything 3 repository, we have identified **9 potential features** for integration. Of these:

- **2 are Priority 1 (Critical)** - Immediate implementation recommended
- **2 are Priority 2 (High-Value)** - Next sprint
- **3 are Priority 3 (Nice-to-Have)** - Future consideration
- **2 are Priority 4 (Out-of-Scope)** - Not recommended

**Total Estimated Effort:** 29-34 hours for P1+P2 features (4-5 developer days)

---

## Priority 1: Critical Features (Sprint 1)

### ✅ 1.1 Model Versioning Support

**Status:** 📋 **PLANNED**  
**Effort:** 6 hours  
**Value:** HIGH  
**Risk:** LOW

**Description:**
Add support for `-1.1` model versions (bug fixes, better street scene performance).

**Implementation Checklist:**
- [ ] Extend `ModelVariant` enum with `-1.1` versions (2h)
  - `GIANT_V1_1`, `LARGE_V1_1`, `NESTED_GIANT_LARGE_V1_1`
- [ ] Add `auto_upgrade_to_latest` configuration flag (2h)
- [ ] Update documentation with version notes (1h)
- [ ] Add tests for version resolution (1h)

**Files to Modify:**
- `lux_depth_v3/config.py` - Add new enum values
- `lux_depth_v3/da3_wrapper.py` - Update model name resolution
- `lux_depth_v3/docs/API_REFERENCE.md` - Document versions
- `lux_depth_v3/tests/test_config.py` - Add tests

**Acceptance Criteria:**
- [ ] Users can select `-1.1` models via CLI: `--model da3-giant-1.1`
- [ ] Auto-upgrade flag works: `--auto-upgrade-to-latest`
- [ ] Documentation lists version differences
- [ ] All tests passing

---

### ✅ 1.2 Metric Depth Conversion Utilities

**Status:** 📋 **PLANNED**  
**Effort:** 5 hours  
**Value:** HIGH  
**Risk:** LOW

**Description:**
Add utilities to convert DA3METRIC output to metric depth (meters) using official formula: `metric_depth = focal * net_output / 300.0`

**Implementation Checklist:**
- [ ] Implement `convert_to_metric_depth()` function (2h)
- [ ] Implement `estimate_focal_length()` helper (1h)
- [ ] Add `convert_metric_depth()` to Postprocessor (1h)
- [ ] Add CLI flag `--metric-conversion` (1h)

**Files to Modify:**
- `lux_depth_v3/postprocessing.py` - Add conversion functions
- `lux_depth_v3/cli.py` - Add CLI flag
- `lux_depth_v3/docs/API_REFERENCE.md` - Document utilities
- `lux_depth_v3/examples/metric_depth_example.py` - Add example (new file)
- `lux_depth_v3/tests/test_postprocessing.py` - Add tests

**Acceptance Criteria:**
- [ ] `convert_to_metric_depth()` function works correctly
- [ ] CLI supports: `lux-depth-v3 process --metric-conversion --focal 1000`
- [ ] Documentation includes EXIF extraction examples
- [ ] Tests validate formula accuracy

---

## Priority 2: High-Value Features (Sprint 2)

### ✅ 2.1 License Validation & Warnings

**Status:** 📋 **PLANNED**  
**Effort:** 8 hours  
**Value:** MEDIUM-HIGH  
**Risk:** LOW

**Description:**
Validate model licenses (Apache 2.0 vs CC-BY-NC 4.0) and warn users about commercial restrictions.

**Implementation Checklist:**
- [ ] Create license mapping and validation functions (2h)
- [ ] Integrate validation into DA3Config (2h)
- [ ] Add CLI flags: `--commercial-use`, `--strict-license` (2h)
- [ ] Add license table to documentation (2h)

**Files to Modify:**
- `lux_depth_v3/config.py` - Add license enums and validation
- `lux_depth_v3/cli.py` - Add CLI flags
- `lux_depth_v3/README.md` - Add license comparison table
- `lux_depth_v3/tests/test_config.py` - Add license tests

**Acceptance Criteria:**
- [ ] Warning appears for CC-BY-NC models in commercial mode
- [ ] CLI supports: `lux-depth-v3 process --commercial-use --strict-license`
- [ ] Documentation includes clear license table
- [ ] Tests cover all model variants

---

### ✅ 2.2 XFormers Support & Fallback

**Status:** 📋 **PLANNED**  
**Effort:** 10 hours  
**Value:** MEDIUM  
**Risk:** MEDIUM

**Description:**
Detect XFormers availability and GPU compatibility, with graceful fallback for older GPUs.

**Implementation Checklist:**
- [ ] Implement XFormers detection utilities (3h)
- [ ] Add fallback logic to wrapper initialization (3h)
- [ ] Add configuration flags for XFormers control (2h)
- [ ] Document XFormers requirements (2h)

**Files to Modify:**
- `lux_depth_v3/da3_wrapper.py` - Add detection and fallback
- `lux_depth_v3/config.py` - Add XFormers config flags
- `lux_depth_v3/docs/TROUBLESHOOTING.md` - Add XFormers section (new file)
- `lux_depth_v3/tests/test_da3_wrapper.py` - Add tests

**Acceptance Criteria:**
- [ ] XFormers detection automatic on initialization
- [ ] Graceful fallback with clear warning message
- [ ] Documentation links to GPU compatibility requirements
- [ ] Tests cover detection logic

---

## Priority 3: Nice-to-Have Features (Future)

### ⏸️ 3.1 DA3-Streaming (Conditional)

**Status:** 🔍 **EVALUATING**  
**Effort:** 20 hours  
**Value:** MEDIUM (if users request)  
**Risk:** MEDIUM-HIGH

**Decision Point:**
- **IF** users request long video support (>1000 frames) in next 2 months
- **THEN** implement minimal wrapper
- **OTHERWISE** defer indefinitely

**Current Status:**
- No user requests for ultra-long video processing
- Manual chunking acceptable for now
- Monitor feedback after Sprint 1-2 completion

---

### 📝 3.2 Gradio/Gallery CLI Passthrough

**Status:** 📋 **PLANNED** (Low Priority)  
**Effort:** 2 hours  
**Value:** LOW-MEDIUM  
**Risk:** LOW

**Description:**
Add CLI commands to launch Gradio UI and Gallery server.

**Implementation:**
```python
# Simple passthrough to official DA3 CLI
lux-depth-v3 gradio --model da3-large --port 7860
lux-depth-v3 gallery output/ --port 8080
```

**Deferred until:** User demand emerges for web UI

---

### 📊 3.3 Model Performance Documentation

**Status:** 📋 **PLANNED** (Low Priority)  
**Effort:** 8 hours  
**Value:** LOW-MEDIUM  
**Risk:** LOW

**Description:**
Create comprehensive model comparison guide with AUC3 results, speed benchmarks, and use case recommendations.

**Deliverable:**
- `lux_depth_v3/docs/MODEL_PERFORMANCE.md`
- Model selection flowchart
- Benchmark tables

**Deferred until:** Sprint 2 completion

---

## Priority 4: Out-of-Scope Features

### ❌ 4.1 Custom Model Architecture Configs

**Status:** 🚫 **REJECTED**  
**Rationale:**
- Not aligned with luxury real estate user base
- High maintenance burden
- Pre-trained models cover use cases

**Alternative:**
Document how to use official DA3 API directly for research use cases.

---

### ❌ 4.2 Community Tools Integration

**Status:** 🚫 **REJECTED**  
**Rationale:**
- Different target audience (plugin developers)
- High maintenance across multiple platforms
- Better handled by community

**Alternative:**
Document compatible export formats (PLY, GLB, NPZ) for community tool integration.

---

## Implementation Timeline

### Sprint 1 (Week 1) - Critical
**Duration:** 1.5-2 developer days  
**Features:** Model Versioning + Metric Depth Utilities

```
Day 1-2: Implementation (11h)
Day 3: Testing & Documentation (3h)
Total: 14 hours
```

**Deliverables:**
- ✅ `-1.1` model support
- ✅ Metric depth conversion utilities
- ✅ CLI flags and examples
- ✅ Test coverage ≥85%

---

### Sprint 2 (Week 2-3) - High-Value
**Duration:** 2.5-3 developer days  
**Features:** License Validation + XFormers Fallback

```
Week 2: License Validation (8h)
Week 3: XFormers Fallback (10h)
Total: 18 hours
```

**Deliverables:**
- ✅ License warnings for CC-BY-NC models
- ✅ `--commercial-use` flag
- ✅ XFormers detection and fallback
- ✅ Troubleshooting documentation

---

### Future Sprints (Month 2+) - Optional
**Duration:** TBD based on user feedback  
**Features:** DA3-Streaming (conditional), Gradio UI, Performance Docs

**Decision Gates:**
1. Review user feedback after Sprint 1-2
2. Prioritize based on actual demand
3. Re-evaluate DA3-Streaming need

---

## Success Metrics

### Sprint 1 Success Criteria
- [ ] ≥50% of users adopt `-1.1` models within 2 weeks
- [ ] Metric depth conversion used in ≥30% of DA3METRIC workflows
- [ ] Zero user confusion on version selection
- [ ] Documentation rated ≥4.5/5 by users

### Sprint 2 Success Criteria
- [ ] Zero license violations reported
- [ ] XFormers fallback works on ≥95% of tested GPUs
- [ ] Support tickets reduced by 20% (compatibility issues)

### Overall Success Criteria
- [ ] Feature adoption rate ≥60%
- [ ] User satisfaction score ≥4.0/5
- [ ] Zero critical bugs introduced
- [ ] Test coverage maintained ≥85%

---

## Risk Register

### Active Risks

| Risk | Probability | Impact | Mitigation | Owner |
|------|------------|---------|-----------|-------|
| Upstream API breaking changes | MEDIUM | HIGH | Pin DA3 version, test before upgrades | Backend Dev |
| License mapping becomes outdated | LOW | HIGH | Monitor official repo monthly | Architect |
| XFormers compatibility matrix incomplete | MEDIUM | MEDIUM | Conservative detection, user override | Backend Dev |
| Metric depth formula changes | LOW | MEDIUM | Make scale_factor configurable | ML Engineer |

### Resolved Risks
(None yet - will update after implementation)

---

## Next Steps

### Immediate (This Week)
1. **Review this document** with team
2. **Assign owners** for Sprint 1 features
3. **Set up development branch:** `feature/da3-enhancements`
4. **Create GitHub issues** for P1 features

### Short-Term (Next Week)
1. **Begin Sprint 1 implementation**
2. **Monitor upstream DA3 repo** for changes
3. **Collect user feedback** on current integration

### Long-Term (Month 2+)
1. **Evaluate DA3-Streaming demand**
2. **Consider Gradio UI** if requested
3. **Benchmark model performance** for documentation

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-12-19 | 1.0 | Initial planning document | Architect |

---

## References

- **Gap Analysis:** `lux_depth_v3/docs/DA3_FEATURE_GAP_ANALYSIS.md`
- **Official DA3 Repo:** https://github.com/ByteDance-Seed/Depth-Anything-3
- **API Reference:** `lux_depth_v3/docs/API_REFERENCE.md`
- **Integration Guide:** `lux_depth_v3/INTEGRATION_GUIDE.md`

---

**Status:** Ready for Implementation  
**Next Review:** After Sprint 1 completion  
**Owner:** Transformation Portal Architect
