# Lux Depth Config Knob Checklist

Use this checklist whenever a new `EnhanceConfig` knob is added or an existing
one changes behavior.

- Normalize or clamp the value in `EnhanceConfig.__post_init__` when the runtime
  treats part of the input domain as equivalent.
- Decide whether the knob changes the APEX gate fingerprint, the broader config
  fingerprint, or neither, and add focused tests for that decision.
- Keep cache keys narrow: if the knob changes review or policy behavior but does
  not change depth output generation, prove the depth-cache payload stays
  unchanged.
- Check telemetry and preview/readout surfaces so the effective normalized value
  is what operators see in diagnostics and review output.
- Update the contributor-facing docs when the knob changes normalization rules,
  fingerprint behavior, cache scope, or validation expectations.

Current representative example: `apex_depth_threshold_epsilon` now normalizes at
config construction so runtime gate math and fingerprint generation use the same
effective policy surface.
