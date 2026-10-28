import test from "node:test";
import assert from "node:assert/strict";
import { photographicSelectionFor } from "../portal-src/internal/photographic-selection.js";

function fixture() {
  const auth = { mode: "managed", actor: { username: "reviewer" }, tenant: { id: "one" }, features: { photographicPagedCatalogue: true } };
  const job = { id: "photo", pipeline: "lux-depth" };
  const selected = { jobId: job.id, sceneId: "input-0016", artifactPath: "original/input-0016.tif",
    railPath: "v6/input-0016/delivery.tif", actorScope: JSON.stringify([auth.mode, auth.actor, auth.tenant]) };
  return { state: { auth, artifactUi: { photographicSelection: selected } }, job, selected };
}

test("original and published rail selectors retain one exact actor/job-scoped photographic choice", () => {
  const { state, job, selected } = fixture();
  for (const path of [selected.artifactPath, selected.railPath]) {
    assert.equal(photographicSelectionFor(state, job, path), selected);
  }
  // Rehydration replaces the job object, without changing the selection.
  assert.equal(photographicSelectionFor(state, { ...job }, selected.artifactPath), selected);
});

test("another rail choice cannot inherit the previous photo's version", () => {
  const { state, job } = fixture();
  assert.deepEqual(photographicSelectionFor(state, job, "v6/input-0031/delivery.tif"), {
    artifactPath: "v6/input-0031/delivery.tif", railPath: ""
  });
});

test("actor, tenant, job and rollout changes cannot reuse a retained rail mapping", () => {
  for (const mutate of [
    (state) => { state.auth.actor = { username: "another" }; },
    (state) => { state.auth.tenant = { id: "two" }; },
    (_state, job) => { job.id = "other"; }
  ]) {
    const { state, job, selected } = fixture();
    mutate(state, job);
    const result = photographicSelectionFor(state, job, selected.artifactPath);
    assert.notEqual(result, selected);
    assert.equal(result.railPath, "");
  }
  const { state, job, selected } = fixture();
  state.auth.features.photographicPagedCatalogue = false;
  assert.equal(photographicSelectionFor(state, job, selected.artifactPath), null);
});

test("photographic rollout leaves unrelated jobs' missing-artifact recovery unchanged", () => {
  const { state } = fixture();
  assert.equal(photographicSelectionFor(state, { id: "archive", pipeline: "archive-gate-a", artifacts: [{ path: "index.json" }] }, "missing.json"), null);
});
