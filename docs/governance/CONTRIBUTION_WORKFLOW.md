# Contribution Workflow

## Golden Path (lux_depth_v2) is Feature-Frozen

As of 2025-12-23, `lux_depth_v2` is **feature-frozen**.

See: `lux_depth_v2/FEATURE_FREEZE.md`

### Allowed Changes:
- Security fixes
- Performance improvements
- Bug fixes

### NOT Allowed:
- New parameters/knobs
- New presets
- Feature expansion

### Exception Process:
Any PR touching `lux_depth_v2` must:
1. Reference FEATURE_FREEZE.md
2. Justify against allowed change types
3. Get architect approval

---

## New Ideas Routing

All new ideas follow this path:

1. **Document first**: Add to `docs/research/`
2. **Validate need**: Does it solve production problem?
3. **Graduate carefully**: Only if proven essential

**Default answer: NO**

Credibility comes from restraint, not expansion.

---

## Experimental Work

All experimental work:
- Lives in `experimental/`
- Cannot be imported by production code
- CI enforces boundary
- Explicitly marked as unstable

See: `.github/workflows/experimental-boundary.yml`

---

## Cultural Principle

> "We are no longer proving capability. We are protecting credibility."

Fewer changes. Fewer entry points. Stronger 'no' muscle.
