import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { signActorAssertion } from "../lib/frontdoor-identity.js";
import { buildUpstreamHeaders } from "../lib/proxy.js";

const fixture = JSON.parse(readFileSync(new URL("../../../tests/fixtures/frontdoor_identity_v1.json", import.meta.url), "utf8"));

test("frontdoor identity signature matches the Python verifier fixture", () => {
  assert.equal(signActorAssertion(fixture), fixture.assertion);
});

test("all browser-supplied identity and tenant headers are stripped even without an actor", () => {
  const source = new Headers({ "x-tp-tenant-id":"tenant_b", "x-tp-actor":"attacker", "x-tp-actor-email":"other@example.com", "x-tp-actor-role":"admin", "x-tp-actor-assertion":"forged" });
  const headers = buildUpstreamHeaders(source, { backendApiKey:"backend-secret", actor:null });
  for (const [key] of source.entries()) assert.equal(headers.has(key), false);
  assert.equal(headers.get("x-api-key"), "backend-secret");
});

test("proxy reconstructs identity only from authenticated session and dedicated signer", () => {
  const headers = buildUpstreamHeaders(new Headers({"x-tp-actor-assertion":"forged", "x-tp-tenant-id":"tenant_b"}), {
    backendApiKey:"backend-secret", actor:fixture.actor, identity:fixture
  });
  assert.equal(headers.get("x-tp-actor-assertion"), fixture.assertion);
  assert.equal(headers.get("x-tp-actor-email"), fixture.actor.accessEmail);
  assert.equal(headers.has("x-tp-tenant-id"), false);
});

test("proxy strips a configured legacy tenant header", () => {
  const previous = process.env.TP_PILOT_TENANT_HEADER;
  process.env.TP_PILOT_TENANT_HEADER = "x-custom-tenant";
  try {
    const headers = buildUpstreamHeaders(new Headers({"x-custom-tenant":"tenant_b"}), {backendApiKey:"key", actor:null});
    assert.equal(headers.has("x-custom-tenant"), false);
  } finally {
    if (previous === undefined) delete process.env.TP_PILOT_TENANT_HEADER;
    else process.env.TP_PILOT_TENANT_HEADER = previous;
  }
});

for (const change of [{secret:"short"}, {actor:{username:"admin"}}, {target:"/"+"a".repeat(8192)}]) {
  test(`signing rejects incomplete or oversized authority: ${Object.keys(change)[0]}`, () => {
    assert.throws(() => signActorAssertion({...fixture,...change}));
  });
}
