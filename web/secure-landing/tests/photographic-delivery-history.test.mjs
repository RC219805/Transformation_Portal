import test from "node:test";
import assert from "node:assert/strict";
import { photographicDeliveryAttempt, photographicDeliveryRecovered, photographicDeliveryRecoveryStorage } from "../portal-src/photographic-delivery-history.js";

const SHA = "a".repeat(64);
const preset = { schema: "tp.photo.export_preset.v1", name: "website", format: "jpeg", intended_use: "Website", long_edge: 2560,
  jpeg_quality: 92, sharpen_amount: 0.2, render: "perceptual_srgb", shoulder: 0.8 };
const source = { artifact_path: "final.tif", sha256: SHA, source_identity_sha256: SHA };
const body = { property_name: "House", expected_version: 3, approved_sources: [source], presets: ["website"], scene_labels: { scene: "Room" }, request_correlation_id: "attempt" };

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
  assert.equal(recovery.set("actor-a:job", attempt), true);
  assert.deepEqual(recovery.get("actor-a:job"), attempt);
  assert.equal(recovery.get("actor-b:job"), null);
  assert.equal(recovery.set("oversized", { ...attempt, padding: "x".repeat(128 * 1024) }), false);
  assert.deepEqual(recovery.get("actor-a:job"), attempt);
  recovery.set("actor-a:job", null);
  assert.equal(recovery.get("actor-a:job"), null);
  for (const value of ["bad", "[null]", "[{}]", "x".repeat(128 * 1024 + 1)]) {
    storage.setItem("tp.photo.delivery_recovery.v1", value);
    assert.equal(recovery.get("actor-a:job"), null);
  }
  assert.equal(photographicDeliveryRecoveryStorage({ getItem() { throw new Error("unavailable"); } }).get("scope"), null);
  assert.equal(photographicDeliveryRecoveryStorage(null).set("scope", attempt), false);
});
