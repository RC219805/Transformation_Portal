// Contract tests for rate-limit / Retry-After header preservation through
// the managed front-door /v1 proxy seam.
//
// Backend contract (asserted in tests/test_app_rejection_paths.py and
// tests/test_app_orchestrator_contract_http.py): a 429 response from the
// FastAPI origin always carries:
//   Retry-After
//   X-RateLimit-Limit
//   X-RateLimit-Remaining (== "0")
//   X-RateLimit-Reset
//
// This file pins down the proxy-helper side of that contract:
// copyUpstreamResponseHeaders() must preserve these headers (i.e. they are
// not in the response strip-list) so that 429s reach the browser with the
// retry hints intact. End-to-end behaviour through the full /v1 route
// handler (auth, sessions, CSRF) is covered alongside the existing
// upstream-failure tests in routes.test.mjs.

import test from "node:test";
import assert from "node:assert/strict";

import { copyUpstreamResponseHeaders } from "../lib/proxy.js";

test("proxy copyUpstreamResponseHeaders preserves Retry-After and X-RateLimit-* headers", () => {
  const upstream = new Headers({
    "Retry-After": "37",
    "X-RateLimit-Limit": "60",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1700000000",
    "Content-Type": "application/json"
  });

  const forwarded = copyUpstreamResponseHeaders(upstream);

  assert.equal(forwarded.get("retry-after"), "37");
  assert.equal(forwarded.get("x-ratelimit-limit"), "60");
  assert.equal(forwarded.get("x-ratelimit-remaining"), "0");
  assert.equal(forwarded.get("x-ratelimit-reset"), "1700000000");
});

test("proxy copyUpstreamResponseHeaders strips hop-by-hop headers without dropping rate-limit headers", () => {
  const upstream = new Headers({
    "Retry-After": "5",
    "X-RateLimit-Limit": "4",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1700000005",
    "Connection": "close",
    "Transfer-Encoding": "chunked",
    "Content-Encoding": "gzip"
  });

  const forwarded = copyUpstreamResponseHeaders(upstream);

  // Rate-limit contract preserved.
  assert.equal(forwarded.get("retry-after"), "5");
  assert.equal(forwarded.get("x-ratelimit-limit"), "4");
  assert.equal(forwarded.get("x-ratelimit-remaining"), "0");
  assert.equal(forwarded.get("x-ratelimit-reset"), "1700000005");

  // Hop-by-hop headers stripped per existing proxy denylist.
  assert.equal(forwarded.get("connection"), null);
  assert.equal(forwarded.get("transfer-encoding"), null);
  assert.equal(forwarded.get("content-encoding"), null);
});
