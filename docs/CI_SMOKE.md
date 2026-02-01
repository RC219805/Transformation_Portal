# CI Smoke Test Marker

This file exists as a minimal, non-functional documentation artifact that can be safely modified
when we need a trivial change to exercise CI (for example, validating that documentation-only
diffs still run the expected pipelines).

## Purpose

- Provides a dedicated file for testing CI workflows without modifying production code
- Useful for validating branch protection rules, required checks, and workflow triggers
- Safe to modify for testing purposes

## Usage Guidelines

Do **not** update this file solely to change timestamps or other time-varying content. Prefer:

- Making a meaningful change to documentation or code, or
- Using an empty commit or CI-specific trigger mechanism if you need to test CI wiring

## History

- 2026-02-01: Created for quality firewall validation (PR #771)
