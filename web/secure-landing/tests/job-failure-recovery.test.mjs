import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
function section(start, end) {
  return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
}

const guidance = {
  code: "RUNNER_EXIT_NONZERO",
  message: "Convert the image to sRGB with a profile-aware editor, then create a new job.",
  details: { exit_code: 1, stage: "preprocess", reason: "unsupported_icc_profile" },
};

function streamHarness() {
  const context = vm.createContext({
    state: { selectedJobId: "job-color" }, els: {},
    _markJobEventActivity() {}, _recordProgressTimeline() {}, _reconcileJobTimeline() {},
    logToPane() {}, stopJobActivity() {}, scheduleRenderJobQueue() {}, createToast() {},
    refreshJobStatus(job) { job.finalLogsRequested = true; },
    appendJobLog(job, line) { job.logs.push(line); },
  });
  vm.runInContext(section("function getReadableError(", "const CAPTIONING_RUN_STATUS_VALUES")
    + section("function _applyJobStreamEvent(", "async function _startAuthorizedFetchSse("), context);
  return context;
}

test("historical generic done events retain projected color diagnostics and log the useful error", () => {
  const h = streamHarness();
  const job = { id: "job-color", state: "failed", error: guidance, logs: [], progress: 0 };
  h._applyJobStreamEvent(job, "done", {
    state: "failed", exit_code: 1,
    error: { code: guidance.code, message: "runner exited with code 1", details: { exit_code: 1 } },
  });
  assert.equal(job.error, guidance);
  assert.match(job.logs[0], /profile-aware editor/);
  assert.equal(job.finalLogsRequested, true);
});

test("new diagnostic or different exit replaces prior guidance", () => {
  const h = streamHarness();
  for (const error of [
    { ...guidance, details: { exit_code: 2 } },
    { ...guidance, details: { exit_code: 1, reason: "other_failure", stage: "preprocess" } },
    { code: "AUTH_DENIED", message: "Access denied." },
  ]) {
    assert.equal(h._mergeJobStreamError(guidance, error), error);
  }
  const ambiguous = { ...guidance, details: { ...guidance.details, reason: "ambiguous_input_color" } };
  assert.equal(h._mergeJobStreamError(guidance, ambiguous), ambiguous);
});

test("list refresh preserves retrieved logs while detail refresh replaces the saved tail", () => {
  let refreshes = 0;
  const context = vm.createContext({
    _ensureJobStreamState() {}, _reconcileJobTimeline() {},
    _isJobStreamRecoverable: (job) => ['running', 'queued'].includes(job.state),
    _clearSseRetry() {}, _teardownJobEventStream() {},
    state: { selectedJobId: "job-color", currentView: "overview", jobs: [] }, els: {},
    _rememberSelectedJob() {}, renderReviewSurfaces() {}, _primeDeferredReviewSurface() {},
    _primeDeferredOperateSurface() {}, _primeDeferredBuildSurface() {}, scheduleRenderJobQueue() {},
    refreshJobStatus() { refreshes += 1; },
  });
  vm.runInContext(section("function _syncHydratedJob(", "function refreshJobStatus(")
    + section("function selectJob(", "function hydrateJobFromServer("), context);
  const job = { id: "job-color", state: "running", logsLoadStatus: "ready", logs: ["saved diagnostic"] };
  context.state.jobs.push(job);
  const hydrated = { id: "job-color", state: "failed", logs: [] };
  context._syncHydratedJob(job, hydrated, {});
  assert.deepEqual(job.logs, ["saved diagnostic"]);
  assert.equal(job.logsLoadStatus, "pending", "terminal transition requires the final saved log tail");
  context.selectJob(job.id);
  assert.equal(refreshes, 1, "selecting the completed job fetches its final logs despite earlier running hydration");
  context._syncHydratedJob(job, { ...hydrated, logs: ["updated diagnostic"] });
  assert.deepEqual(job.logs, ["updated diagnostic"]);
});

test("concurrent detail refreshes share a bounded request and allow retry after completion", async () => {
  let calls = 0;
  let release;
  const context = vm.createContext({
    _isProtectedFamilySuppressed: () => false,
    _isJobStreamRecoverable: (job) => ['running', 'queued'].includes(job.state),
    _fetchJobStatus: () => { calls += 1; return new Promise((resolve) => { release = resolve; }); },
  });
  vm.runInContext(section("function refreshJobStatus(", "async function _fetchJobStatus("), context);
  const job = { id: "job-color" };
  const first = context.refreshJobStatus(job);
  const second = context.refreshJobStatus(job);
  assert.equal(first, second);
  assert.equal(calls, 1);
  release();
  await first;
  const retry = context.refreshJobStatus(job);
  assert.equal(calls, 2);
  release();
  await retry;
});

for (const finalState of ['failed', 'running', 'unavailable']) {
  test(`terminal SSE rejects an in-flight running snapshot and performs one final detail refresh (${finalState})`, async () => {
    const h = streamHarness();
    let requests = 0;
    let releaseRunning;
    const runningSnapshot = { id: 'job-color', state: 'running', logs_tail: ['partial saved logs'], progress: 1 };
    const result = (rawJob) => ({ response: { ok: true }, body: JSON.stringify({ data: rawJob }) });
    Object.assign(h, {
      API_BASE: '', BOOTSTRAP_TIMEOUT_MS: 3500,
      _isJobStreamRecoverable: (job) => ['running', 'queued'].includes(job.state),
      _isProtectedFamilySuppressed: () => false,
      _buildAuthHeaders: (headers) => headers,
      _parseJsonResponseBody: JSON.parse,
      _ensureJobStreamState() {}, _clearSseRetry() {}, _teardownJobEventStream() {},
      _maybeSuppressOnProtectedResponse() {},
      hydrateJobFromServer: (rawJob) => ({ ...rawJob, logs: rawJob.logs_tail, artifacts: [], run_summary: null }),
      fetchBodyWithTimeout: () => {
        requests += 1;
        if (requests === 1) return new Promise((resolve) => { releaseRunning = () => resolve(result(runningSnapshot)); });
        assert.equal(requests, 2, 'final refresh must not loop when the backend remains stale or unavailable');
        return Promise.resolve(finalState === 'unavailable'
          ? { response: { ok: false } }
          : result({ ...runningSnapshot, state: finalState, error: guidance, logs_tail: ['final saved diagnostic'] }));
      },
    });
    vm.runInContext(section('function _syncHydratedJob(', 'function scheduleSseReconnect('), h);
    const job = { id: 'job-color', state: 'running', logs: ['existing logs'], progress: 0 };
    const pending = h.refreshJobStatus(job);
    h._applyJobStreamEvent(job, 'done', { state: 'failed', exit_code: 1, error: guidance });
    assert.equal(requests, 1, 'done coalesces with the in-flight detail request');
    releaseRunning();
    await pending;
    assert.equal(requests, 2);
    assert.equal(job.state, 'failed', 'a stale running response cannot regress the terminal state');
    assert.equal(job.detailRequest, null);
    assert.equal(job.logsLoadStatus, finalState === 'failed' ? 'ready' : 'error');
    assert.ok(!job.logs.includes('partial saved logs'));
    if (finalState === 'failed') assert.deepEqual(Array.from(job.logs), ['final saved diagnostic']);
    else assert.ok(job.logs.includes('existing logs'), 'failed final refresh retains existing logs');
  });
}
