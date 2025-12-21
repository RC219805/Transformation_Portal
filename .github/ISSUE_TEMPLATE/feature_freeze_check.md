---
name: Feature Freeze Check
about: Review proposed changes during feature freeze period
title: "[FREEZE CHECK] "
labels: ["feature-freeze", "review-required"]
assignees: []
---

## Feature Freeze Period

**Active**: December 20, 2025 - January 10, 2026  
**Policy**: [FEATURE_FREEZE_POLICY.md](../../docs/FEATURE_FREEZE_POLICY.md)

---

## Change Description

<!-- Describe the proposed change -->

---

## Freeze Compliance Check

### Change Category (check one):

- [ ] **ALLOWED** - Bug fix (correctness issue)
- [ ] **ALLOWED** - Security fix
- [ ] **ALLOWED** - Documentation improvement
- [ ] **ALLOWED** - Test improvement
- [ ] **ALLOWED** - Performance optimization (no behavior change)
- [ ] **BLOCKED** - New feature
- [ ] **BLOCKED** - Breaking change
- [ ] **BLOCKED** - Refactoring (non-critical)
- [ ] **DEFERRED** - Non-urgent improvement

### Justification

<!-- If claiming ALLOWED, explain why this can't wait until Jan 10 -->

---

## Impact Assessment

**Files Changed**:
<!-- List files -->

**Lines Changed**: <!-- +X/-Y -->

**Breaking Changes**: Yes / No

**Test Coverage**: <!-- % or N/A -->

---

## Review Checklist

- [ ] Change is minimal and surgical
- [ ] No Golden Path workflow disruption
- [ ] Tests pass (`make test-lux-depth-v2`)
- [ ] Documentation updated (if applicable)
- [ ] Security review (if applicable)

---

## Recommendation

**Reviewer**: <!-- @username -->  
**Decision**: APPROVE / DEFER / REJECT  
**Reason**:

---

## Notes

<!-- Additional context -->
