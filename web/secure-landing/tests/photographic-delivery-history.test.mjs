import test from "node:test";
import assert from "node:assert/strict";
import { photographicDeliveryAttempt, photographicDeliveryRecovered, photographicDeliveryRecoveryStorage } from "../portal-src/photographic-delivery-history.js";

const SHA = "a".repeat(64);
const preset = { schema: "tp.photo.export_preset.v1", name: "website", format: "jpeg", intended_use: "Website", long_edge: 2560,
  jpeg_quality: 92, sharpen_amount: 0.2, render: "perceptual_srgb", shoulder: 0.8 };
const source = { artifact_path: "final.tif", sha256: SHA, source_identity_sha256: SHA };
const body = { property_name: "House", expected_version: 3, approved_sources: [source], presets: ["website"], scene_labels: { scene: "Room" }, request_correlation_id: "attempt" };
const scope = (actor) => JSON.stringify(["job", JSON.stringify(["managed", { username: actor }, null])]);

test("frozen delivery intent includes presentation and expanded presets without sharing mutable draft objects", () => {
  const request = structuredClone(body);
  const attempt = photographicDeliveryAttempt(request, [{ ...source, presentation_policy_sha256: "b".repeat(64) }], [preset]);
  request.property_name = "Changed";
  assert.equal(attempt.body.property_name, "House");
  assert.equal(attempt.intent.sources[0].presentation_policy_sha256, "b".repeat(64));
  assert.deepEqual(attempt.intent.presets, [preset]);
});

test("recovery requires the original correlation, version, complete intent and server fingerprint", () => {
  const attempt = photographicDeliveryAttempt(body, [source], [preset]);
  const entry = { request_correlation_id: "attempt", approval_version: 3, request_fingerprint_schema: "tp.photo.delivery_intent.v1",
    request_fingerprint_sha256: SHA, intent: attempt.intent };
  assert.equal(photographicDeliveryRecovered(entry, attempt), true);
  for (const change of [{ request_correlation_id: "other" }, { approval_version: 4 }, { request_fingerprint_sha256: "bad" },
    { intent: { ...attempt.intent, scene_labels: { scene: "Changed" } } },
    { intent: { ...attempt.intent, sources: [{ ...attempt.intent.sources[0], presentation_policy_sha256: "c".repeat(64) }] } }]) {
    assert.equal(photographicDeliveryRecovered({ ...entry, ...change }, attempt), false);
  }
});

test("session recovery is scope bound, size bounded and tolerates unavailable or malformed storage", () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  const recovery = photographicDeliveryRecoveryStorage(storage);
  const attempt = photographicDeliveryAttempt(body, [source], [preset]);
  assert.equal(recovery.set(scope("a"), attempt), true);
  assert.deepEqual(recovery.get(scope("a")), attempt);
  assert.equal(recovery.get(scope("b")), null);
  assert.equal(recovery.set(scope("oversized"), { ...attempt, padding: "x".repeat(128 * 1024) }), false);
  assert.deepEqual(recovery.get(scope("a")), attempt);
  const changed = { ...attempt, body: { ...attempt.body, request_correlation_id: "different" } };
  assert.equal(recovery.set(scope("a"), changed), false, "An unresolved attempt cannot be overwritten");
  assert.equal(recovery.set(scope("a"), null, changed), false, "A different expected attempt cannot be cleared");
  assert.equal(recovery.set(scope("a"), null, attempt), true);
  assert.equal(recovery.get(scope("a")), null);
  for (const value of ["bad", "[null]", "[{}]", "x".repeat(128 * 1024 + 1)]) {
    storage.setItem("tp.photo.delivery_recovery.v1", value);
    assert.equal(recovery.get(scope("a")), null);
    assert.equal(recovery.inspect(scope("a")).status, "invalid");
    assert.equal(recovery.set(scope("a"), attempt), false);
    assert.equal(values.get("tp.photo.delivery_recovery.v1"), value);
  }
  assert.equal(photographicDeliveryRecoveryStorage({ getItem() { throw new Error("unavailable"); } }).get("scope"), null);
  assert.equal(photographicDeliveryRecoveryStorage(null).set("scope", attempt), false);
});

test("invalid scopes, duplicate scopes and malformed frozen attempts cannot be silently dropped by a write", () => {
  const attempt = photographicDeliveryAttempt(body, [source], [preset]);
  for (const entries of [
    [{ attempt }], [{ scope: null, attempt }], [{ scope: "", attempt }], [{ scope: "actor:job", attempt }],
    [{ scope: ` ${scope("a")}`, attempt }],
    [{ scope: JSON.stringify(["job", '["managed", null, null]']), attempt }],
    [{ scope: JSON.stringify(["job", "not-an-actor-scope"]), attempt }],
    [{ scope: scope("a"), attempt }, { scope: scope("a"), attempt }],
    [{ scope: scope("a"), attempt: { body: attempt.body, intent: {} } }],
  ]) {
    let raw = JSON.stringify(entries);
    const original = raw;
    const recovery = photographicDeliveryRecoveryStorage({ getItem: () => raw, setItem: (_key, value) => { raw = value; } });
    assert.equal(recovery.inspect(scope("b")).status, "invalid");
    assert.equal(recovery.set(scope("b"), attempt), false);
    assert.equal(raw, original);
  }
});
