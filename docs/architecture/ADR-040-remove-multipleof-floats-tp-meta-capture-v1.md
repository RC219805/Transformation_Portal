# ADR-040 Remove `multipleOf` Constraints for Float Fields in `tp.meta.capture.v1`

## Status
Accepted

## Date
2026-02-27

## Executive Summary

This ADR removes JSON Schema `multipleOf` constraints from floating-point
capture fields in `tp.meta.capture.v1`. The schema keeps semantic ranges and
nullability, while precision enforcement remains in the canonicalization layer.
This avoids IEEE-754 representation edge cases that can trigger false-negative
schema validation failures for values that are otherwise valid after
deterministic rounding.

## Context

Phase 4C canonicalization already enforces deterministic precision using
Decimal half-even rounding rules from `tools/capture_metadata_config.json`.
Schema-level `multipleOf` checks on floats can still reject common values such
as `2.8` and `0.7` because binary floating-point representation is not exact.

That creates a contract-layer brittleness: numerically correct canonical output
can be rejected for representation reasons unrelated to extraction correctness.

## Decision

### D1. Remove `multipleOf` on Float Capture Fields

`multipleOf` is removed from the following `tp.meta.capture.v1` schema fields:

- `gps_latitude`
- `gps_longitude`
- `focal_length_mm`
- `aperture_fnumber`
- `shutter_speed_seconds`
- `exposure_compensation_ev`

### D2. Preserve Existing Semantic Guards

The schema continues to enforce:

- existing `minimum`/`maximum`/`exclusiveMinimum` bounds,
- existing nullability and `oneOf` structure,
- existing deterministic object constraints.

### D3. Precision Ownership Remains in Canonicalization

Precision and rounding policy remains owned by the canonicalization layer
(`Decimal`, half-even) rather than schema arithmetic constraints.

### D4. Regression Coverage Is Mandatory

The repository keeps a DJI float edge-case regression in
`tests/test_extract_capture_metadata.py`
(`test_phase4c_dji_float_case_schema_and_rounding`) to lock behavior for:

- float normalization (`2.8`, `+0.7`),
- GPS float precision acceptance,
- timezone warning semantics and strict-mode failure policy.

## Consequences

Positive:

- removes false schema rejects caused by IEEE-754 representation artifacts,
- keeps determinism and warning behavior unchanged,
- clarifies contract boundaries between schema and canonicalization.

Trade-off:

- schema no longer encodes decimal precision granularity directly; this is now
  a canonicalization-policy responsibility and must stay test-covered.

## References

- `schemas/phase4/metadata.schema.json`
- `tests/test_extract_capture_metadata.py`
- `tools/capture_metadata_config.json`
