import test from "node:test";
import assert from "node:assert/strict";

import { evaluateSessionScaling } from "../lib/session-scaling.js";

test("session scaling defaults to supported single-instance SQLite posture", () => {
  const result = evaluateSessionScaling({});

  assert.equal(result.ok, true);
  assert.equal(result.backend, "sqlite");
  assert.equal(result.mode, "single_instance");
  assert.equal(result.reason, null);
});

test("session scaling rejects multi-instance mode until an external store exists", () => {
  const result = evaluateSessionScaling({
    sessionScalingMode: "multi_instance"
  });

  assert.equal(result.ok, false);
  assert.equal(result.backend, "sqlite");
  assert.equal(result.mode, "multi_instance");
  assert.equal(result.reason, "multi_instance_requires_external_session_store");
});

test("session scaling rejects ephemeral runtime mode until an external store exists", () => {
  const result = evaluateSessionScaling({
    sessionScalingMode: "ephemeral-runtime"
  });

  assert.equal(result.ok, false);
  assert.equal(result.backend, "sqlite");
  assert.equal(result.mode, "ephemeral_runtime");
  assert.equal(result.reason, "ephemeral_runtime_requires_external_session_store");
});

test("session scaling rejects invalid mode declarations", () => {
  const result = evaluateSessionScaling({
    sessionScalingMode: "planet-scale"
  });

  assert.equal(result.ok, false);
  assert.equal(result.backend, "sqlite");
  assert.equal(result.mode, "planet_scale");
  assert.equal(result.reason, "invalid_session_scaling_mode");
});
