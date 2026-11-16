import test from "node:test";
import assert from "node:assert/strict";
import { backendRecoveryBootstrap } from "../lib/recovery-bootstrap.js";

const actor = { username: "alice", accessEmail: "alice@example.test", role: "operator" };
const context = { schema: "tp.portal.recovery_context.v1", tenantId: "tenant-a", ownerId: "a".repeat(64) };
const unavailable = { recoveryContext: null, recoveryContextStatus: "unavailable", features: { jobSubmissionReceipts: false, uploadReceipts: false } };
function env(run) {
  const names = ["TP_PILOT_CONTROL_PLANE_ENABLED", "TP_FRONTDOOR_IDENTITY_SECRET", "TP_BACKEND_API_KEY", "TP_FASTAPI_ORIGIN"];
  const previous = names.map((name) => process.env[name]);
  Object.assign(process.env, { TP_PILOT_CONTROL_PLANE_ENABLED: "1", TP_FRONTDOOR_IDENTITY_SECRET: "separate signing authority secret 123456789", TP_BACKEND_API_KEY: "server-private-key", TP_FASTAPI_ORIGIN: "https://backend.example.test" });
  return Promise.resolve().then(run).finally(() => names.forEach((name, index) => { if (previous[index] === undefined) delete process.env[name]; else process.env[name] = previous[index]; }));
}
const json = (body) => new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });

test("recovery bootstrap uses fresh signed server identity and reconstructs only bounded public context", () => env(async () => {
  let observed;
  const output = await backendRecoveryBootstrap(actor, "", { fetchImpl: async (url, options) => {
    observed = { url, options };
    return json({ recoveryContext: { ...context, private: "never forward" }, features: { jobSubmissionReceipts: true, uploadReceipts: "true", backendApiKey: "never forward" }, csrfToken: "never forward" });
  } });
  assert.equal(observed.url.toString(), "https://backend.example.test/portal/bootstrap");
  assert.equal(observed.options.cache, "no-store"); assert.equal(observed.options.redirect, "manual");
  const headers = new Headers(observed.options.headers);
  assert.equal(headers.get("x-api-key"), "server-private-key");
  assert.equal(headers.get("cookie"), null); assert.equal(headers.get("x-tp-tenant-id"), null);
  const signed = JSON.parse(Buffer.from(headers.get("x-tp-actor-assertion").split(".")[0], "base64url").toString());
  assert.deepEqual(signed.actor, actor); assert.equal(signed.target, "/portal/bootstrap"); assert.equal(signed.method, "GET");
  assert.deepEqual(output, { recoveryContext: context, recoveryContextStatus: "ready", features: { jobSubmissionReceipts: true, uploadReceipts: false } });
  assert.doesNotMatch(JSON.stringify(output), /private|csrf|server-private-key/);
}));

test("backend outages and malformed contexts disable new capabilities without changing managed auth", () => env(async () => {
  for (const recoveryContext of [null, { ...context, ownerId: "A".repeat(64) }, { ...context, ownerId: [context.ownerId] }, { ...context, tenantId: "\n" }, { ...context, schema: "unknown" }]) {
    assert.deepEqual(await backendRecoveryBootstrap(actor, "", { fetchImpl: async () => json({ recoveryContext, features: { jobSubmissionReceipts: true, uploadReceipts: true } }) }), unavailable);
  }
  assert.deepEqual(await backendRecoveryBootstrap(actor, "", { fetchImpl: async () => { throw new Error("offline"); } }), unavailable);
  assert.deepEqual(await backendRecoveryBootstrap(actor, "", { fetchImpl: async () => json({ recoveryContext: context, padding: "x".repeat(17000) }) }), unavailable);
}));

test("bootstrap deadline includes stalled body reads and aborts upstream", () => env(async () => {
  let signal;
  const result = await backendRecoveryBootstrap(actor, "", { timeoutMs: 5, fetchImpl: async (_, options) => {
    signal = options.signal;
    return new Response(new ReadableStream({ start() {} }), { headers: { "content-type": "application/json" } });
  } });
  assert.deepEqual(result, unavailable); assert.equal(signal.aborted, true);
}));

test("disabled pilot or invalid server signing setup never contacts backend", () => env(async () => {
  let calls = 0; const fetchImpl = async () => { calls += 1; return json({}); };
  process.env.TP_PILOT_CONTROL_PLANE_ENABLED = "0";
  assert.deepEqual(await backendRecoveryBootstrap(actor, "", { fetchImpl }), unavailable);
  process.env.TP_PILOT_CONTROL_PLANE_ENABLED = "1";
  process.env.TP_FRONTDOOR_IDENTITY_SECRET = "too short";
  assert.deepEqual(await backendRecoveryBootstrap(actor, "", { fetchImpl }), unavailable);
  assert.equal(calls, 0);
}));
