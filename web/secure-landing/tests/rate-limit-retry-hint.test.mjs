// Unit tests for parseRateLimitRetryHint and the portal.template.js
// retry-decision contracts that consume it (bootstrap retry +
// config-preview service retry).
//
// Backend contract (asserted by tests/test_app_rejection_paths.py and
// tests/test_app_orchestrator_contract_http.py): a 429 response carries
//   Retry-After
//   X-RateLimit-Limit
//   X-RateLimit-Remaining (== "0")
//   X-RateLimit-Reset
//
// Front-door proxy contract (tests/rate-limit-headers.test.mjs +
// tests/routes.test.mjs): those four headers reach the browser
// unchanged.
//
// This file pins the *consumer* side: the parser produces the right
// timing object and the portal source threads it through bootstrap
// retry and config-preview retry without breaking 401/403, 5xx, or
// non-429 paths, and without touching native EventSource.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  parseRateLimitRetryHint,
  RATE_LIMIT_HINT_MAX_RETRY_MS
} from "../portal-src/internal/rate-limit-retry.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORTAL_TEMPLATE_PATH = path.resolve(__dirname, "../portal-src/portal.template.js");

function _headersFromFixture(fixture) {
  const headers = new Headers();
  for (const [key, value] of Object.entries(fixture || {})) {
    if (value !== undefined && value !== null) {
      headers.set(key, String(value));
    }
  }
  return { headers };
}

// -----------------------------------------------------------------------
// Parser contract
// -----------------------------------------------------------------------

test("parser: Retry-After integer seconds drives the schedule", () => {
  const nowMs = 1_700_000_000_000;
  const hint = parseRateLimitRetryHint(
    _headersFromFixture({ "Retry-After": "7", "X-RateLimit-Reset": "1700000600" }),
    nowMs
  );
  assert.ok(hint, "expected a hint when Retry-After is present");
  assert.equal(hint.source, "retry-after-seconds");
  assert.equal(hint.retryAfterSeconds, 7);
  assert.equal(hint.retryAfterMs, 7000);
  assert.equal(hint.resetEpochSeconds, 1700000600);
  assert.equal(hint.retryAtMs, nowMs + 7000);
});

test("parser: Retry-After HTTP-date form is honoured when integer parser fails", () => {
  const nowMs = Date.UTC(2026, 9, 21, 7, 28, 0); // 2026-10-21T07:28:00Z
  const hint = parseRateLimitRetryHint(
    _headersFromFixture({ "Retry-After": "Wed, 21 Oct 2026 07:28:42 GMT" }),
    nowMs
  );
  assert.ok(hint, "expected a hint when Retry-After is an HTTP-date");
  assert.equal(hint.source, "retry-after-date");
  // 42 seconds difference, ceil-rounded.
  assert.equal(hint.retryAfterSeconds, 42);
});

test("parser: X-RateLimit-Reset fallback applies when Retry-After is absent", () => {
  const nowMs = 1_700_000_000_000;
  const hint = parseRateLimitRetryHint(
    _headersFromFixture({ "X-RateLimit-Reset": "1700000060" }),
    nowMs
  );
  assert.ok(hint, "expected a hint from X-RateLimit-Reset fallback");
  assert.equal(hint.source, "x-ratelimit-reset");
  assert.equal(hint.retryAfterSeconds, 60);
  assert.equal(hint.resetEpochSeconds, 1700000060);
});

test("parser: invalid Retry-After values fall back to null when no usable headers remain", () => {
  const nowMs = 1_700_000_000_000;
  const cases = [
    { "Retry-After": "0" },
    { "Retry-After": "-5" },
    { "Retry-After": "NaN" },
    { "Retry-After": "" },
    { "Retry-After": "soon" },
    {} // no headers at all
  ];
  for (const fixture of cases) {
    const hint = parseRateLimitRetryHint(_headersFromFixture(fixture), nowMs);
    assert.equal(hint, null, `expected null for fixture ${JSON.stringify(fixture)}`);
  }
});

test("parser: zero/negative X-RateLimit-Reset is rejected", () => {
  const hint = parseRateLimitRetryHint(
    _headersFromFixture({ "X-RateLimit-Reset": "0" }),
    1_700_000_000_000
  );
  assert.equal(hint, null);
});

test("parser: past X-RateLimit-Reset is rejected (must point to the future)", () => {
  const nowMs = 1_700_000_000_000;
  const hint = parseRateLimitRetryHint(
    _headersFromFixture({ "X-RateLimit-Reset": "1699999000" }),
    nowMs
  );
  assert.equal(hint, null);
});

test("parser: absurdly large Retry-After is clamped to RATE_LIMIT_HINT_MAX_RETRY_MS", () => {
  const nowMs = 1_700_000_000_000;
  const hint = parseRateLimitRetryHint(
    _headersFromFixture({ "Retry-After": "100000" }), // 100,000 seconds
    nowMs
  );
  assert.ok(hint);
  assert.equal(hint.retryAfterMs, RATE_LIMIT_HINT_MAX_RETRY_MS);
  assert.equal(hint.retryAfterSeconds, Math.ceil(RATE_LIMIT_HINT_MAX_RETRY_MS / 1000));
});

test("parser accepts a Headers object directly as well as a Response-shaped object", () => {
  const nowMs = 1_700_000_000_000;
  const headers = new Headers({ "Retry-After": "11" });
  const hintFromHeaders = parseRateLimitRetryHint(headers, nowMs);
  const hintFromResponseShape = parseRateLimitRetryHint({ headers }, nowMs);
  assert.deepEqual(hintFromHeaders, hintFromResponseShape);
  assert.equal(hintFromHeaders.retryAfterSeconds, 11);
});

test("parser: header lookup is case-insensitive", () => {
  const nowMs = 1_700_000_000_000;
  const hint = parseRateLimitRetryHint(
    _headersFromFixture({ "retry-after": "9" }),
    nowMs
  );
  assert.ok(hint);
  assert.equal(hint.retryAfterSeconds, 9);
});

// -----------------------------------------------------------------------
// portal.template.js retry-decision wiring contract
// -----------------------------------------------------------------------
//
// These assertions guard the seam where the parser is consumed. They are
// source-level (the function bodies are not executable here) but they
// pin down the contract that implementation changes have to preserve.

const portalTemplate = readFileSync(PORTAL_TEMPLATE_PATH, "utf8");

test("bootstrap retry threads the rate-limit hint into the scheduler", () => {
  // 429 must be retryable for bootstrap.
  assert.match(
    portalTemplate,
    /BOOTSTRAP_RETRIABLE_HTTP_STATUSES = new Set\(\[429,/
  );
  // Explicit failure-classifier branch for 429 produces a retryable
  // 'rate_limited' reason.
  assert.match(
    portalTemplate,
    /normalizedReason === 'rate_limited' \|\| normalizedStatus === 429/
  );
  assert.match(
    portalTemplate,
    /reason: 'rate_limited',\s*\n\s*retryable: true/
  );
  // The fetch site captures the hint only on 429 and forwards it.
  assert.match(
    portalTemplate,
    /res\.status === 429\s*\n?\s*\?\s*portalInternals\.parseRateLimitRetryHint\(res\)\s*\n?\s*:\s*null/
  );
  // Scheduler accepts the hint as an explicit third argument.
  assert.match(
    portalTemplate,
    /function _scheduleBootstrapRetry\(reason = '', httpStatus = 0, rateLimitHint = null\)/
  );
  // Scheduler prefers hint-derived delay over exponential when valid,
  // and clamps to the existing BOOTSTRAP_RETRY_MAX_DELAY_MS upper bound.
  assert.match(
    portalTemplate,
    /delayMs = Math\.min\(Number\(rateLimitHint\.retryAfterMs\), BOOTSTRAP_RETRY_MAX_DELAY_MS\)/
  );
  // Telemetry distinguishes the source of the chosen delay so observers
  // can tell exponential backoff from header-derived schedules.
  assert.match(
    portalTemplate,
    /delaySource = String\(rateLimitHint\.source \|\| 'rate_limit_hint'\)/
  );
});

test("bootstrap fetch does not consume Retry-After on 401/403 paths", () => {
  // The 429 branch is gated on res.status === 429 only; the 401/403
  // branch above it returns before reaching the rate-limit-aware code.
  const sourceSnippet = portalTemplate.slice(
    portalTemplate.indexOf("if (res.status === 401 || res.status === 403)"),
    portalTemplate.indexOf("if (res.status === 401 || res.status === 403)") + 1500
  );
  assert.ok(
    sourceSnippet.includes("window.location.assign(_managedLoginUrlForCurrentRoute());"),
    "401/403 branch must redirect to login, not retry"
  );
  assert.ok(
    !/parseRateLimitRetryHint/.test(sourceSnippet.split("if (!res.ok)")[0]),
    "Retry-After must not be parsed on the auth branch"
  );
});

test("config-preview service retry honours the rate-limit hint and re-reads the form", () => {
  // The scheduler accepts the hint and uses it for the delay.
  assert.match(
    portalTemplate,
    /function _scheduleConfigPreviewServiceRetry\(rateLimitHint = null\)/
  );
  assert.match(
    portalTemplate,
    /delay = Math\.min\(\s*Number\(rateLimitHint\.retryAfterMs\),\s*maxScheduledDelay\s*\);/
  );
  // The retry timer always re-reads generatePayload() so the cooldown
  // does not replay a stale snapshot.
  const retryFn = portalTemplate.slice(
    portalTemplate.indexOf("function _scheduleConfigPreviewServiceRetry"),
    portalTemplate.indexOf("async function fetchConfigPreview")
  );
  assert.ok(
    retryFn.includes("const payload = generatePayload();"),
    "retry must re-read the current form state"
  );
  // 429 routes into service_failure (so the scheduler runs) instead of
  // the validation_error bucket which does not retry.
  assert.match(
    portalTemplate,
    /isRateLimited\s*\n\s*\?\s*'service_failure'\s*\n\s*:\s*res\.status >= 400 && res\.status < 500/
  );
  // The hint capture is gated on isRateLimited, so non-429 4xx paths
  // do not call the parser.
  assert.match(
    portalTemplate,
    /const rateLimitHint = isRateLimited\s*\n\s*\?\s*portalInternals\.parseRateLimitRetryHint\(res\)\s*\n\s*:\s*null;/
  );
});

test("native EventSource path is not changed by this commit", () => {
  // The single native EventSource construction site stays in main
  // bundle and is *not* paired with parseRateLimitRetryHint.
  const eventSourceUsage = portalTemplate.match(/new EventSource\(streamUrl\);/);
  assert.ok(eventSourceUsage, "expected at least one native EventSource site");
  // EventSource error handling cannot read response headers (browser
  // limitation), so the parser must not be invoked there.
  const eventSourceBlockStart = portalTemplate.indexOf("new EventSource(streamUrl);");
  const nextFunctionDefinition = portalTemplate.indexOf("\nfunction ", eventSourceBlockStart);
  const eventSourceBlock = portalTemplate.slice(eventSourceBlockStart, nextFunctionDefinition);
  assert.ok(
    !/parseRateLimitRetryHint/.test(eventSourceBlock),
    "EventSource path must not consume the rate-limit hint"
  );
});

test("mutating POSTs (job dispatch / upload) are not silently auto-replayed", () => {
  // Sanity guard: the parser is only consumed in the two retry sites
  // wired above. Searching the source for parseRateLimitRetryHint should
  // surface exactly those two call sites — bootstrap and config-preview.
  const callSites = portalTemplate.match(/portalInternals\.parseRateLimitRetryHint\(/g) || [];
  assert.equal(
    callSites.length,
    2,
    "parseRateLimitRetryHint should be consumed by exactly two retry sites today; "
      + "if you add a new consumer, update this guard and confirm the new call site does not "
      + "auto-replay a non-idempotent POST."
  );
  // Job dispatch is implemented in submitJob; it must not invoke the
  // parser (a rate-limited dispatch should surface the cooldown to the
  // user, not silently replay the POST).
  const submitJobStart = portalTemplate.indexOf("async function submitJob");
  if (submitJobStart >= 0) {
    const submitJobEnd = portalTemplate.indexOf("\nasync function ", submitJobStart + 1);
    const submitJobBody = portalTemplate.slice(
      submitJobStart,
      submitJobEnd > submitJobStart ? submitJobEnd : portalTemplate.length
    );
    assert.ok(
      !/parseRateLimitRetryHint/.test(submitJobBody),
      "submitJob must not auto-replay rate-limited POSTs"
    );
  }
});
