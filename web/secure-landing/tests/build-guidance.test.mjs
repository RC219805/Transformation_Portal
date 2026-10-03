import test from "node:test";
import assert from "node:assert/strict";
import { buildWorkspaceGuidance } from "../portal-src/internal/build-guidance.js";

const ready = { accessReady: true, pipeline: "lux-depth-v3", readiness: { canRun: true } };

test("a server action preserves its field and warning without authorizing dispatch", () => {
  const guide = buildWorkspaceGuidance({ ...ready, nextAction: {
    tone: "blocked", field: "output_dir", label: "Choose an output", detail: "Use an authorized destination."
  } });
  assert.equal(guide.field, "output_dir");
  assert.equal(guide.tone, "error");
  assert.equal(guide.detail, "Use an authorized destination.");
  assert.equal(guide.step, 4);
});

test("an optimistic next action cannot override current readiness", () => {
  for (const overview of [false, true]) {
    const guide = buildWorkspaceGuidance({ ...ready, step: 4, overview,
      readiness: { canRun: false, detail: "Preview is refreshing." },
      nextAction: { action: "dispatch_ready", tone: "ready" }
    });
    assert.equal(guide.tone, "info");
    assert.equal(guide.title, "Checking this draft");
    assert.equal(guide.detail, "Preview is refreshing.");
    assert.equal(guide.field, undefined);
  }
});

test("unavailable access retains its recovery message and disables guidance", () => {
  const guide = buildWorkspaceGuidance({ ...ready, accessReady: false,
    accessSummary: { tone: "blocked", badge: "Recovery required", detail: "Sign in again." }
  });
  assert.equal(guide.title, "Recovery required");
  assert.equal(guide.detail, "Sign in again.");
  assert.equal(guide.tone, "error");
  assert.equal(guide.disabled, true);
});

test("submission guidance follows pending and confirmed handoff states", () => {
  const pending = buildWorkspaceGuidance({ ...ready, dispatchPending: true });
  assert.equal(pending.disabled, true);
  assert.equal(pending.title, "Submitting your run");
  const submitted = buildWorkspaceGuidance({ ...ready, handoffJobId: "job-current" });
  assert.equal(submitted.jobId, "job-current");
  assert.equal(submitted.title, "Your run has been submitted");
  assert.equal(submitted.field, undefined);
});

test("photography help preserves encoding and authorization boundaries", () => {
  for (const pipeline of ["lux-depth", "lux-depth-v5", "lux-depth-v6"]) {
    const paths = buildWorkspaceGuidance({ ...ready, pipeline, step: 2 });
    const outputs = buildWorkspaceGuidance({ ...ready, pipeline, step: 3 });
    assert.match(paths.answer, /Uploading files fills the input path only/);
    assert.match(paths.answer, /server preview checks access/);
    assert.match(outputs.answer, /do not convert another profile/);
  }
});
