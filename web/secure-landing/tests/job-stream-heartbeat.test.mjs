import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { createOperateTools } from "../portal-src/operate-tools.js";

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
    section('function _isJobTerminal(', 'function _latestVisibleTransportWarning(')
    + section('function _teardownJobEventStream(', 'function refreshJobStatus(')
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

test('live log pane appends lines, stays bounded and coalesces autoscroll per frame', () => {
  const frames = [];
  let layoutReads = 0;
  let hydrated = false;
  const pane = {
    childNodes: [],
    scrollTop: 0,
    get firstChild() { return this.childNodes[0]; },
    get scrollHeight() { layoutReads += 1; return this.childNodes.length * 10; },
    get textContent() { return this.childNodes.map((node) => node.data).join(''); },
    set textContent(_value) {
      if (hydrated) throw new Error('logToPane must not rewrite the whole pane per line');
    },
    appendChild(node) { this.childNodes.push(node); return node; },
    removeChild(node) { assert.equal(node, this.childNodes[0]); this.childNodes.shift(); return node; },
  };
  const state = { selectedJobId: 'job-live', jobs: [{ id: 'job-live', logs: [] }] };
  const els = { logPane: pane };
  // The operate tools own the log pane (search and follow); logToPane hands each
  // live line to them. Creating the tools hydrates the pane once.
  const tools = createOperateTools({
    state, els, maxLogLines: 1200,
    document: { getElementById: () => null, createTextNode: (data) => ({ data }) },
    requestAnimationFrame: (callback) => { frames.push(callback); return frames.length; },
    _displayJobState: (job) => job.state, _portalPrivilegesReady: () => true,
  });
  hydrated = true;
  layoutReads = 0;
  const context = vm.createContext({ state, els, _deferredOperateSurfaceApi: () => tools });
  vm.runInContext(section('function logToPane(', '// ===='), context);

  for (let index = 0; index < 3000; index += 1) context.logToPane('job-live', `[INFO] line ${index}`);
  context.logToPane('job-other', '[INFO] not for the selected job');

  assert.equal(pane.childNodes.length, 1200, 'the pane is capped like job.logs');
  assert.equal(pane.childNodes[0].data, '[INFO] line 1800\n');
  assert.equal(pane.childNodes.at(-1).data, '[INFO] line 2999\n');
  assert.equal(layoutReads, 0, 'no forced reflow per appended line');
  assert.equal(frames.length, 1, 'autoscroll is coalesced into one animation frame');
  frames[0]();
  assert.equal(pane.scrollTop, 12_000);
  context.logToPane('job-live', '[INFO] next frame');
  assert.equal(frames.length, 2, 'a later frame schedules a new autoscroll');
});
