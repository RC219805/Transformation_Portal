# Transformation Portal Roadmap Re-review

_Date: 2026-04-07_

## Summary

This re-review narrows the roadmap to current repo truth. Several earlier
"next steps" are already implemented and verified in code and tests, so they
should remain closed. The primary active browser-security lane is now the
direct-debug portal CSP posture, which was previously constrained by a single
inline HTML shell.

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

## Active Now

- The direct-debug portal remains the active browser-security surface.
- The highest-value next implementation is to keep the managed/front-door
  contract unchanged while tightening the FastAPI-served direct-debug CSP.
- The chosen implementation lane is `CSP Unlock`:
  - move direct-debug CSS and JS into same-origin portal assets;
  - replace third-party font hosts with repo-local font assets;
  - remove the HTML meta CSP and rely on the FastAPI response header as the
    single policy source;
  - tighten the direct-debug CSP to remove inline script/style allowances.

## Trigger-based Backlog

- `app.py` modularization remains backlog work, not urgent remediation.
- Externalized state remains backlog work until deployment or operational
  triggers make multi-instance behavior necessary.
- Any broader CodeQL language-surface expansion remains optional coverage work,
  not a baseline gap.

## Current Execution Outcome

This re-review is now paired with the first active implementation step:

- the direct-debug portal shell references same-origin CSS and JS assets under
  `/portal/assets/`;
- repo-local font assets replace third-party font hosts for the direct-debug
  surface;
- the FastAPI CSP for `/` removes `'unsafe-inline'` from `script-src` and
  `style-src`;
- regression tests pin the new direct-debug shell, asset routes, and tightened
  CSP/header behavior.
