import assert from "node:assert/strict";
import test from "node:test";

import worker from "../src/index.ts";

const FRONTDOOR_ORIGIN = "https://frontdoor.example:8443";

for (const pathname of [
  "//attacker.example/capture",
  "///attacker.example/capture",
  "/\\attacker.example/capture",
  "\\\\attacker.example/capture",
  "//attacker.example\\capture",
  "/%2f%2fattacker.example/capture"
]) {
  test(`request path ${JSON.stringify(pathname)} cannot replace the configured origin`, async (t) => {
    const incoming = new Request(`https://portal.example${pathname}?view=review&job=a%2Fb`, {
      headers: {
        Cookie: "__Host-tp_session=test-session",
        "Cf-Access-Jwt-Assertion": "test-access-assertion"
      }
    });
    const incomingUrl = new URL(incoming.url);
    const fetchMock = t.mock.method(globalThis, "fetch", async (upstreamRequest) => {
      const upstreamUrl = new URL(upstreamRequest.url);
      assert.equal(upstreamUrl.origin, FRONTDOOR_ORIGIN);
      assert.equal(upstreamUrl.pathname, incomingUrl.pathname);
      assert.equal(upstreamUrl.search, incomingUrl.search);
      assert.equal(upstreamRequest.headers.get("cookie"), "__Host-tp_session=test-session");
      assert.equal(upstreamRequest.headers.get("cf-access-jwt-assertion"), "test-access-assertion");
      return new Response("frontdoor");
    });

    const response = await worker.fetch(incoming, { FRONTDOOR_ORIGIN });

    assert.equal(await response.text(), "frontdoor");
    assert.equal(fetchMock.mock.callCount(), 1);
  });
}

test("normal path, query, credentials, and manual redirects keep their proxy contract", async (t) => {
  const upstreamResponse = new Response(null, {
    status: 302,
    headers: { Location: "https://external.example/", "Cache-Control": "no-store" }
  });
  const fetchMock = t.mock.method(globalThis, "fetch", async (upstreamRequest) => {
    assert.equal(upstreamRequest.url, `${FRONTDOOR_ORIGIN}/portal?view=review&job=a%2Fb`);
    assert.equal(upstreamRequest.method, "GET");
    assert.equal(upstreamRequest.redirect, "manual");
    assert.equal(upstreamRequest.headers.get("cookie"), "__Host-tp_session=test-session");
    assert.equal(upstreamRequest.headers.get("cf-access-jwt-assertion"), "test-access-assertion");
    assert.equal(upstreamRequest.headers.get("x-forwarded-host"), "portal.example");
    assert.equal(upstreamRequest.headers.get("x-forwarded-proto"), "https");
    return upstreamResponse;
  });

  const response = await worker.fetch(
    new Request("https://portal.example/portal?view=review&job=a%2Fb", {
      headers: {
        Cookie: "__Host-tp_session=test-session",
        "Cf-Access-Jwt-Assertion": "test-access-assertion"
      }
    }),
    { FRONTDOOR_ORIGIN }
  );

  assert.equal(response, upstreamResponse);
  assert.equal(fetchMock.mock.callCount(), 1);
});
