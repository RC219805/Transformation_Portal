// Parse Retry-After / X-RateLimit-* hints emitted by the orchestrator's
// 429 contract (see app.py:_rate_limit_response_headers and the contract
// tests under tests/test_app_rejection_paths.py + tests/web/secure-landing
// /tests/rate-limit-headers.test.mjs).
//
// Resolution order:
//   1. Retry-After: integer seconds (HTTP/1.1 standard).
//   2. Retry-After: HTTP-date (RFC 1123) -> seconds-from-now.
//   3. X-RateLimit-Reset: epoch seconds -> seconds-from-now.
//
// Returns null when no usable hint is present so callers can fall back to
// their existing backoff/exponential schedule. Callers are responsible for
// only applying the hint on HTTP 429 (the parser does not enforce status,
// since it accepts a Headers object directly).

// Hard cap so a misconfigured upstream cannot stall the portal forever;
// per-call clients clamp further to their own existing max delay.
export const RATE_LIMIT_HINT_MAX_RETRY_MS = 120000;

function _toIntOrNaN(rawValue) {
  if (rawValue === null || rawValue === undefined) return Number.NaN;
  const trimmed = String(rawValue).trim();
  if (!trimmed) return Number.NaN;
  // Reject anything that is not a pure non-negative integer literal so that
  // HTTP-date Retry-After values fall through to the date branch instead of
  // partially parsing (e.g. "Wed," -> NaN before this guard, but "21" out
  // of "21 Oct ..." would otherwise satisfy Number()).
  if (!/^[0-9]+$/.test(trimmed)) return Number.NaN;
  const numeric = Number(trimmed);
  if (!Number.isFinite(numeric)) return Number.NaN;
  return numeric;
}

function _retryAfterDateSeconds(rawValue, nowMs) {
  if (rawValue === null || rawValue === undefined) return Number.NaN;
  const trimmed = String(rawValue).trim();
  if (!trimmed) return Number.NaN;
  if (/^[0-9]+$/.test(trimmed)) return Number.NaN;
  const parsed = Date.parse(trimmed);
  if (!Number.isFinite(parsed)) return Number.NaN;
  return Math.ceil((parsed - nowMs) / 1000);
}

function _resolveHeaders(source) {
  if (!source) return null;
  if (typeof source.headers !== "undefined" && source.headers && typeof source.headers.get === "function") {
    return source.headers;
  }
  if (typeof source.get === "function") {
    return source;
  }
  return null;
}

export function parseRateLimitRetryHint(source, nowMs = Date.now()) {
  const headers = _resolveHeaders(source);
  if (!headers) return null;

  const retryAfterRaw = headers.get("Retry-After") || headers.get("retry-after") || "";
  const resetRaw = headers.get("X-RateLimit-Reset") || headers.get("x-ratelimit-reset") || "";

  let retryAfterSeconds = Number.NaN;
  let appliedSource = null;

  const integerSeconds = _toIntOrNaN(retryAfterRaw);
  if (Number.isFinite(integerSeconds) && integerSeconds > 0) {
    retryAfterSeconds = integerSeconds;
    appliedSource = "retry-after-seconds";
  } else {
    const dateSeconds = _retryAfterDateSeconds(retryAfterRaw, nowMs);
    if (Number.isFinite(dateSeconds) && dateSeconds > 0) {
      retryAfterSeconds = dateSeconds;
      appliedSource = "retry-after-date";
    }
  }

  let resetEpochSeconds = null;
  const resetInt = _toIntOrNaN(resetRaw);
  if (Number.isFinite(resetInt) && resetInt > 0) {
    resetEpochSeconds = resetInt;
    if (!Number.isFinite(retryAfterSeconds)) {
      const computedSeconds = Math.ceil(resetInt - nowMs / 1000);
      if (computedSeconds > 0) {
        retryAfterSeconds = computedSeconds;
        appliedSource = "x-ratelimit-reset";
      }
    }
  }

  if (!Number.isFinite(retryAfterSeconds) || retryAfterSeconds <= 0) return null;

  const cappedSeconds = Math.min(
    retryAfterSeconds,
    Math.ceil(RATE_LIMIT_HINT_MAX_RETRY_MS / 1000)
  );
  const retryAfterMs = cappedSeconds * 1000;

  return {
    retryAfterSeconds: cappedSeconds,
    retryAfterMs,
    resetEpochSeconds,
    retryAtMs: nowMs + retryAfterMs,
    source: appliedSource
  };
}
