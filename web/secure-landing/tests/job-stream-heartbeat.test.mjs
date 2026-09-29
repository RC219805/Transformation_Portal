import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
function section(start, end) {
  return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
}

function harness() {
  let now = 100_000;
  let watchdog;
  let reconnects = 0;
  const warnings = [];
  class NativeStream {
    constructor() { this.readyState = 1; this.listeners = new Map(); }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    close() { this.readyState = 2; }
    emit(name, data = '{}') { this.listeners.get(name)?.({ data }); }
  }
  const context = vm.createContext({
    Date: class extends Date { static now() { return now; } },
    EventSource: NativeStream, API_BASE: '',
    state: { jobs: [] }, document: { hidden: false }, sseWatchdogIntervalId: null,
    SSE_STALL_THRESHOLD_MS: 45_000, SSE_STALL_CHECK_INTERVAL_MS: 10_000,
    EVENT_SOURCE_READY_STATE_CONNECTING: 0, EVENT_SOURCE_READY_STATE_CLOSED: 2,
    _ensureJobStreamState(job) { job.sseRetry ||= { attempt: 0, timer: null }; },
    _isJobStreamRecoverable: (job) => ['running', 'queued'].includes(job.state),
    _isProtectedFamilySuppressed: () => false,
    _clearSseRetry() {}, _currentApiToken: () => '',
    _jobHasActiveStream: (job) => job.eventSource?.readyState === 1,
    _nativeEventSourceReadyState: (stream) => stream.readyState,
    _noteTransportWarning: (_job, code) => warnings.push(code),
    _reconcileJobTimeline() {}, scheduleRenderJobQueue() {}, logToPane() {},
    appendJobLog(job, line) { job.logs.push(line); },
    scheduleSseReconnect() { reconnects += 1; },
    setInterval(callback) { watchdog = callback; return 1; },
  });
  vm.runInContext(
    section('function _teardownJobEventStream(', 'function refreshJobStatus(')
    + section('function _applyJobStreamEvent(', 'async function _startAuthorizedFetchSse(')
    + section('function startJobEventStream(', 'async function recoverJobs(')
    + section('function startSseWatchdog(', 'function stopSseWatchdog('),
    context,
  );
  const job = { id: 'job-quiet-inference', state: 'running', progress: 20, logs: [], timeline: [] };
  context.state.jobs.push(job);
  context.startJobEventStream(job, '/v1/jobs/job-quiet-inference/events');
  job.eventSource.onopen();
  context.startSseWatchdog();
  return {
    context, job, stream: job.eventSource, warnings,
    advance(ms) { now += ms; }, tick: () => watchdog(), reconnects: () => reconnects,
  };
}

test('quiet native SSE heartbeats prevent false stalls without creating progress, logs, or timeline entries', () => {
  const h = harness();
  assert.deepEqual([...h.stream.listeners.keys()], ['log', 'progress', 'state', 'artifact', 'done', 'heartbeat']);
  for (let interval = 0; interval < 12; interval += 1) {
    h.advance(15_000);
    h.stream.emit('heartbeat');
    h.tick();
  }
  assert.equal(h.reconnects(), 0);
  assert.deepEqual(h.warnings, []);
  assert.equal(h.job.progress, 20);
  assert.deepEqual(h.job.logs, []);
  assert.deepEqual(h.job.timeline, []);
  h.advance(45_001);
  h.tick();
  assert.equal(h.reconnects(), 1, 'a genuinely silent transport still reconnects');
  assert.deepEqual(h.warnings, ['stream_stalled']);
});

test('stale status timestamps cannot erase fresh heartbeat activity', () => {
  const h = harness();
  h.advance(60_000);
  h.stream.emit('heartbeat');
  const freshTimestamp = h.job.lastEventAt;
  h.context._syncHydratedJob(h.job, { ...h.job, lastEventAt: 100_000, updatedAt: 100_000 });
  assert.equal(h.job.lastEventAt, freshTimestamp);
  h.tick();
  assert.equal(h.reconnects(), 0);
});

test('malformed heartbeats and events from a replaced native stream cannot extend liveness', () => {
  const h = harness();
  const previousTimestamp = h.job.lastEventAt;
  h.advance(30_000);
  h.stream.emit('heartbeat', 'not-json');
  assert.equal(h.job.lastEventAt, previousTimestamp);
  h.job.eventSource = { readyState: 1 };
  h.stream.emit('heartbeat');
  h.stream.emit('log', '{"line":"late event"}');
  assert.equal(h.job.lastEventAt, previousTimestamp);
  assert.deepEqual(h.job.logs, []);
});
