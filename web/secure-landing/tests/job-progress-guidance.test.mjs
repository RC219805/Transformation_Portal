import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { jobProgressSnapshot, renderJobProgress } from "../portal-src/internal/job-progress.js";
import { createDeferredOperateSurfaceApi } from "../portal-src/operate-surface-deferred.js";

function progressElement() {
  return {
    value: 0, max: 100, dataset: {}, attributes: new Map(),
    setAttribute(name, value) { this.attributes.set(name, value); },
    removeAttribute(name) { this.attributes.delete(name); },
  };
}

test("active jobs without measured progress show waiting without inferring work from heartbeats", () => {
  for (const [state, label] of [["queued", "Queued"], ["running", "In progress"]]) {
    const job = { state, progress: 0 };
    const initial = jobProgressSnapshot(job);
    assert.deepEqual(initial, { value: 0, label, waiting: true });
    job.lastEventAt = Date.now();
    job.logs = ["Heartbeat received"];
    job.artifacts = [{ path: "output.json" }];
    assert.deepEqual(jobProgressSnapshot(job), initial);
    assert.equal(job.progress, 0, "presentation must not mutate backend progress");
  }
});

test("reported progress is bounded and terminal success completes before the final progress event", () => {
  for (const [progress, expected] of [[37, 37], ["42", 42], [140, 100], [-4, 0], [Infinity, 0], [NaN, 0], [undefined, 0]]) {
    assert.equal(jobProgressSnapshot({ state: "running", progress }).value, expected);
  }
  assert.deepEqual(jobProgressSnapshot({ state: "succeeded", progress: 0 }), { value: 100, label: "100%", waiting: false });
  for (const [state, label] of [["failed", "Failed"], ["canceled", "Canceled"], ["partial", "Partial result"], ["offline", "Offline"]]) {
    assert.equal(jobProgressSnapshot({ state, progress: 0 }).waiting, false, `${state} must not appear active`);
    assert.equal(jobProgressSnapshot({ state, progress: 0 }).label, label, `${state} must not imply measured zero progress`);
    assert.equal(jobProgressSnapshot({ state, progress: 42 }).label, "42%");
  }
});

test("failed and canceled runs clear waiting state without presenting an unmeasured percentage", () => {
  for (const [state, label] of [["failed", "Failed"], ["canceled", "Canceled"]]) {
    const element = progressElement();
    renderJobProgress(element, jobProgressSnapshot({ state: "running", progress: 0 }));
    renderJobProgress(element, jobProgressSnapshot({ state, progress: 0 }));
    assert.equal(element.value, 0);
    assert.equal(element.dataset.progressState, "unavailable");
    assert.equal(element.attributes.get("aria-valuetext"), label);
  }
});

test("progress rendering clears obsolete measurements and restores numeric state after waiting", () => {
  const element = progressElement();
  element.attributes.set("value", "40");
  renderJobProgress(element, jobProgressSnapshot({ state: "running", progress: 0 }));
  assert.equal(element.attributes.has("value"), false);
  assert.equal(element.dataset.progressState, "waiting");
  assert.equal(element.attributes.get("aria-valuetext"), "In progress");
  renderJobProgress(element, jobProgressSnapshot({ state: "running", progress: 37 }));
  assert.equal(element.value, 37);
  assert.equal(element.dataset.progressState, "measured");
  assert.equal(element.attributes.get("aria-valuetext"), "37%");
  renderJobProgress(element, jobProgressSnapshot({ state: "succeeded", progress: 0 }));
  assert.equal(element.value, 100);
  assert.equal(element.attributes.get("aria-valuetext"), "100%");
});

function inspector(job) {
  const els = {
    selectedJobRecoveryTitle: {}, selectedJobRecoveryDetail: {}, selectedJobProgressText: {},
    selectedJobProgressBar: progressElement(),
  };
  const state = { jobs: job ? [job] : [], selectedJobId: job?.id, inspectorTab: "overview" };
  const api = createDeferredOperateSurfaceApi({
    state, els,
    jobProgressSnapshot, renderJobProgress,
    EVENT_SOURCE_READY_STATE_CONNECTING: 0, EVENT_SOURCE_READY_STATE_OPEN: 1, EVENT_SOURCE_READY_STATE_CLOSED: 2,
    _isJobsHydrationPending: () => false, _toggleSurfaceSkeleton() {},
    _reconcileJobTimeline() {}, _nativeEventSourceReadyState: () => null,
    _jobHasActiveStream: () => false,
    _latestVisibleTransportWarning: (selected) => selected?.transportWarnings?.at(-1),
    _displayJobState: (selected) => selected.state,
    formatDuration: () => "1m", formatTransportLabel: () => "closed", titleCaseToken: (value) => value,
    getReadableError: (error) => error?.message || "", jobOutcomeSummary: () => "",
    renderSelectedJobRecoveryActions() {}, renderConsoleContextRibbon() {},
  });
  api.renderSelectedJobInspector();
  return els;
}

test("inspector distinguishes queued, processing, completed, and canceled recovery", () => {
  const job = { id: "job-guidance", progress: 0, artifacts: [] };
  const queued = inspector({ ...job, state: "queued" });
  assert.equal(queued.selectedJobProgressText.textContent, "Queued");
  assert.equal(queued.selectedJobRecoveryTitle.textContent, "Waiting for a worker");
  const running = inspector({ ...job, state: "running" });
  assert.equal(running.selectedJobProgressText.textContent, "In progress");
  assert.match(running.selectedJobRecoveryDetail.textContent, /connectivity, not processing progress/);
  const completed = inspector({ ...job, state: "succeeded" });
  assert.equal(completed.selectedJobProgressText.textContent, "100%");
  assert.match(completed.selectedJobRecoveryDetail.textContent, /finished without indexed outputs/);
  assert.match(completed.selectedJobRecoveryDetail.textContent, /Logs and Run Details/);
  const canceled = inspector({ ...job, state: "canceled" });
  assert.equal(canceled.selectedJobProgressText.textContent, "Canceled");
  assert.equal(canceled.selectedJobRecoveryTitle.textContent, "Run canceled");
  assert.doesNotMatch(canceled.selectedJobRecoveryDetail.textContent, /failure|failed/);
  assert.equal(inspector({ ...job, state: "failed" }).selectedJobProgressText.textContent, "Failed");
  assert.equal(inspector(null).selectedJobProgressText.textContent, "No run selected");
});

test("specific errors and authentication recovery retain precedence over waiting guidance", () => {
  const job = { id: "job-guidance", state: "running", progress: 0, artifacts: [] };
  assert.equal(inspector({ ...job, reconnectBlocked: true }).selectedJobRecoveryTitle.textContent, "Restore authentication");
  assert.equal(inspector({ ...job, cancelPending: true }).selectedJobRecoveryTitle.textContent, "Cancel request pending");
  const warning = { detail: "The stream disconnected. Retrying." };
  assert.equal(inspector({ ...job, transportWarnings: [warning] }).selectedJobRecoveryDetail.textContent, warning.detail);
  const error = { code: "RUNNER_EXIT_NONZERO", message: "Convert the source profile before rerunning.", details: { stage: "preprocess", reason: "unsupported_icc_profile" } };
  assert.equal(inspector({ ...job, state: "failed", error }).selectedJobRecoveryDetail.textContent, error.message);
});

const reviewSource = readFileSync(new URL("../portal-src/review-surface-deferred.js", import.meta.url), "utf8");
function reviewSection(start, end) {
  const startIndex = reviewSource.indexOf(start);
  const endIndex = reviewSource.indexOf(end, startIndex);
  assert.ok(startIndex >= 0 && endIndex > startIndex);
  return reviewSource.slice(startIndex, endIndex);
}
const reviewCopy = new Function(`
  ${reviewSection("const REVIEW_STATUS_BUILDERS", "function _reviewStatusSnapshot")}
  ${reviewSection("function _artifactEmptyStateCopy", "function _renderInlinePreview")}
  return { builders: REVIEW_STATUS_BUILDERS, empty: _artifactEmptyStateCopy, state: _reviewStatusState };
`)();

test("Review does not promise future artifacts for a terminal run without outputs", () => {
  for (const state of ["succeeded", "ready"]) {
    const job = { state, artifacts: [] };
    const empty = reviewCopy.empty(job);
    const banner = reviewCopy.builders[reviewCopy.state(job, false, null)]({ artifactCount: 0 });
    for (const copy of [empty, banner]) {
      assert.equal(copy.title, "Run completed without indexed outputs");
      assert.match(copy.action, /Logs and Run Details/);
      assert.doesNotMatch(copy.detail, /will appear|ready for.*review/);
    }
  }
  assert.equal(reviewCopy.empty({ state: "queued" }).title, "Run is waiting to start");
  assert.equal(reviewCopy.empty({ state: "offline" }).title, "Reconnect to check outputs");
});

test("Review retains success and partial-output guidance when outputs exist", () => {
  assert.equal(reviewCopy.builders.ready({ artifact: { path: "output.png" }, artifactCount: 1 }).title, "Outputs ready for review");
  const partial = reviewCopy.builders.partial_reviewable({ artifactCount: 0 });
  assert.match(partial.detail, /no indexed outputs/);
  assert.match(partial.action, /Logs in Operate/);
  assert.match(reviewCopy.builders.partial_reviewable({ artifactCount: 1 }).action, /review the retained outputs/);
});
