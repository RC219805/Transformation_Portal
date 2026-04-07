# Transformation Portal Roadmap Re-review

_Date: 2026-04-07_

## Summary

This re-review narrows the roadmap to current repo truth. Several earlier
"next steps" are already implemented and verified in code and tests, so they
should remain closed. The direct-debug portal CSP unlock is complete, and the
hash-strategy lane now has an explicit repo decision instead of an open-ended
evaluation posture.

## Closed Now

- Pull-request dependency review is already present in
  `.github/workflows/dependency-review.yml` and covered by
  `tests/test_dependency_review_workflow.py`.
- The secure-install hash pilot is already present in
  `.github/workflows/secure-install-pilot.yml`, documented in
  `requirements/README.md`, and covered by
  `tests/test_secure_install_pilot_workflow.py`.
- Runtime version alignment is already enforced through
  `src/transformation_portal/__init__.py`,
  `pyproject.toml`, and `tests/test_package_version.py`.
- The direct-debug portal no longer depends on the Tailwind CDN.
- Direct-debug API-key persistence is already session-only rather than
  long-lived `localStorage` persistence.
- The direct-debug portal CSP unlock is complete:
  - the portal shell references same-origin CSS and JS assets;
  - repo-local font assets replace third-party font hosts;
  - the FastAPI CSP removes inline script/style allowances for `/`;
  - regression tests pin the tightened shell and header behavior.
- The hash strategy decision is now explicit:
  - hash-enriched lock generation remains a CI-only advisory control for the
    non-ML layered locks;
  - the checked-in dependency contract remains pinned-without-hashes for
    standard install flows;
  - root wrapper files and ML platform locks remain outside hash enforcement
    until a later policy decision promotes them.

## Active Now

- No new immediate remediation lane is promoted after the CSP unlock and
  hash-policy closure work.
- The current expectation is to preserve the managed/front-door split, the
  direct-debug browser hardening, and the CI-only hash policy through docs and
  regression coverage rather than opening a new platform refactor.

## Trigger-based Backlog

- `app.py` modularization remains backlog work, not urgent remediation.
- Externalized state remains backlog work until deployment or operational
  triggers make multi-instance behavior necessary.
- Promotion from CI-only advisory hash validation to mandatory
  `--require-hashes` enforcement remains deferred until wrapper flows, ML
  platform locks, and toolchain friction justify a broader contract change.
- Any broader CodeQL language-surface expansion remains optional coverage work,
  not a baseline gap.

## Current Execution Outcome

This re-review now records two completed execution slices:

- the direct-debug portal shell references same-origin CSS and JS assets under
  `/portal/assets/`;
- repo-local font assets replace third-party font hosts for the direct-debug
  surface;
- the FastAPI CSP for `/` removes `'unsafe-inline'` from `script-src` and
  `style-src`;
- regression tests pin the new direct-debug shell, asset routes, and tightened
  CSP/header behavior.
- the secure-install lane now has an explicit policy of CI-only advisory hash
  generation for non-ML layered locks;
- the checked-in dependency contract remains pinned-without-hashes for normal
  local setup and CI install flows;
- README and workflow-contract tests now pin that policy so later changes must
  deliberately opt into any broader enforcement move.
