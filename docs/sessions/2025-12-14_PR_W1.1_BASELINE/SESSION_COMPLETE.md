----------------------------------------------------------------------------
---

✅ SESSION COMPLETE: PR-W1.1 Water Baseline Infrastructure

Status (2025-12-15)

  - PR #559: ✅ MERGED (all checks green)
  - Repository: Clean, on main, ready for next work
  - CI: Water regression job active (warn-only, non-blocking)

----------------------------------------------------------------------------
---

What Shipped (PR-W1.1)

  - ✅ Deterministic CI fixture generator (scripts/gen_water_ci_fixture.py)
  - ✅ Ground-truth v0 schema + validator (data/water_v0/)
  - ✅ Validation harness executes detector path (Materials V3 master gate enabled)
  - ✅ Warn-only CI regression job (non-blocking + artifact upload)
  - ✅ Baseline v0 pinned (data/water_v0/baseline_ci_v0.json)

Key achievement: CI can run water validation on every PR without committing images.

----------------------------------------------------------------------------
---

Baseline V0 Characteristics (intentional "plumbing + contract" baseline)

Pinned baseline metrics (from baseline_ci_v0.json):

  - Total images: 14
  - Positives: 12 (pool=6, ocean=6)
  - Negatives: 2
  - Pool recall: 100% (6/6)
  - Ocean recall: 100% (6/6)
  - False trigger rate: 100% (2/2)  ← expected for uncalibrated heuristic + synthetic negatives
  - Avg processing time: ~97.6ms

Interpretation:

  - v0 proves: detector executes, schema stable, determinism works.
  - v0 is NOT a quality baseline yet (fixtures are mostly full-frame; negatives are deliberately hard).

Known issue to fix next:

  - Metrics naming/aggregation: baseline summary reports false_trigger_count=2
    but false_positive_count=0 despite per-image is_false_positive=true on both negatives.
    Align definitions or remove the redundant field.

----------------------------------------------------------------------------
---

Next Priority: PR-W1.2 (Calibration)

Goal: materially reduce false triggers while preserving recall.

Workstreams:

  A) Confidence shaping / suppressors
     - Add explicit suppression for "flat blue painted surfaces"
     - Add explicit suppression for "architectural glass / grid-like edges"

  B) Improve synthetic fixtures (make metrics meaningful)
     - Positives: partial water coverage with non-water context (deck / horizon)
     - Negatives: structured glass grid, realistic wall seams/shadows
     - Target: median coverage ≠ 1.0 for most samples

  C) Baseline versioning
     - Keep baseline_ci_v0.json as audit trail ("detector runs")
     - Generate baseline_ci_v1.json after suppressors + improved fixtures
     - Point CI regression to v1 when ready

----------------------------------------------------------------------------
---

Materials V3 Progress (merged)

  - PR #552: Glass pixel ops
  - PR #555: Stone pixel ops
  - PR #558: Water detector + integration
  - PR #559: Water baseline infrastructure

Next: PR-W1.2 calibration, then PR-4E wood pixel ops.

----------------------------------------------------------------------------
---

Documentation

  - docs/sessions/2025-12-14_PR_W1.1_BASELINE/SESSION_COMPLETE.md
  - STATUS_REPORT_2025-12-15.md

----------------------------------------------------------------------------
---

Session Status: ✅ COMPLETE
Next Session: PR-W1.2 Calibration
