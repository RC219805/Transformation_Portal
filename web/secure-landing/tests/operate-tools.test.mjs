import test from 'node:test';
import assert from 'node:assert/strict';
import { createOperateTools, filterLoadedJobs, loadedLogSnapshot } from '../portal-src/operate-tools.js';

const jobs = [
  { id: 'job-b', pipeline: 'lux-depth-v3', state: 'running', createdAt: 100 },
  { id: 'job-c', pipeline: 'archive-gate-a', state: 'failed', createdAt: 200 },
  { id: 'job-a', pipeline: 'lux-depth', state: 'succeeded', createdAt: 100 },
];

test('loaded run ordering is newest first with a deterministic ID tie and no state mutation', () => {
  const original = structuredClone(jobs);
  assert.deepEqual(filterLoadedJobs(jobs).map(job => job.id), ['job-c', 'job-a', 'job-b']);
  assert.deepEqual(jobs, original);
});

test('loaded search and status compose without changing the selected run or job collection', () => {
  const state = { jobs: structuredClone(jobs), selectedJobId: 'job-c' };
  const result = filterLoadedJobs(state.jobs, { query: 'LUX', status: 'running' });
  assert.deepEqual(result.map(job => job.id), ['job-b']);
  assert.equal(state.selectedJobId, 'job-c');
  assert.equal(state.jobs.length, 3);
  assert.deepEqual(filterLoadedJobs(jobs, { query: 'missing' }), []);
});

test('search only considers IDs and workflows rather than unbounded private request/log content', () => {
  const input = [{ ...jobs[0], logs: ['hidden-secret'], request: { private: 'hidden-secret' } }];
  assert.deepEqual(filterLoadedJobs(input, { query: 'hidden-secret' }), []);
  assert.equal(filterLoadedJobs(input, { query: 'JOB-B' }).length, 1);
});

test('display statuses drive filters including worker loss and indexing', () => {
  const states = [{ id: 'lost', state: 'worker_lost' }, { id: 'finished', state: 'succeeded' }];
  const displayState = job => job.state === 'succeeded' ? 'indexing' : job.state;
  assert.deepEqual(filterLoadedJobs(states, { status: 'indexing', displayState }).map(job => job.id), ['finished']);
  assert.deepEqual(filterLoadedJobs(states, { status: 'worker_lost', displayState }).map(job => job.id), ['lost']);
});

test('server timestamps and missing timestamps sort deterministically', () => {
  assert.deepEqual(filterLoadedJobs([{ id: 'c', created_at: '2026-01-01T00:00:00Z' }, { id: 'b' }, { id: 'a' }]).map(job => job.id), ['c', 'a', 'b']);
});

test('log search keeps copy/download scoped to the complete bounded loaded buffer', () => {
  const job = { logs: Array.from({ length: 1205 }, (_, index) => `line ${index}`) };
  const result = loadedLogSnapshot(job, 'LINE 1204');
  assert.equal(result.lines.length, 1200);
  assert.equal(result.lines[0], 'line 5');
  assert.equal(result.text, 'line 1204\n');
  assert.equal(result.allText.split('\n').length, 1201);
  assert.equal(job.logs.length, 1205);
});

test('empty and filtered-empty logs stay distinct without implying durable full-history export', () => {
  assert.deepEqual(loadedLogSnapshot(null).lines, []);
  const result = loadedLogSnapshot({ logs: ['first', 'second'] }, 'missing');
  assert.equal(result.text, '');
  assert.equal(result.allText, 'first\nsecond\n');
  assert.equal(result.lines.length, 2);
});


function controlsFixture() {
  class Control {
    value = ''; textContent = ''; disabled = false; scrollTop = 0; scrollHeight = 1000;
    listeners = new Map(); attributes = new Map(); dataset = {};
    addEventListener(name, handler) { this.listeners.set(name, handler); }
    setAttribute(name, value) { this.attributes.set(name, value); }
    replaceChildren(...children) { this.children = children; }
    trigger(name) { return this.listeners.get(name)?.(); }
  }
  const controls = new Map(['loadedRunSearch', 'loadedRunState', 'loadedLogSearch', 'toggleLogFollowBtn', 'copyLoadedLogsBtn', 'downloadLoadedLogsBtn', 'loadedLogSummary'].map(id => [id, new Control()]));
  const state = { auth: { mode: 'managed', actor: { accessEmail: 'alice@example.test', username: 'alice' } },
    selectedJobId: 'job-b', jobs: structuredClone(jobs).map(job => ({ ...job, logs: ['first', 'second'] })) };
  const logPane = new Control(), copied = [];
  let filters = 0, privileged = true;
  const api = createOperateTools({ state, els: { logPane }, maxLogLines: 1200,
    document: { getElementById: id => controls.get(id), createElement: () => new Control() },
    _displayJobState: job => job.state, _portalPrivilegesReady: () => privileged,
    copyToClipboard: value => copied.push(value), onFilterChange: () => { filters += 1; } });
  return { api, state, logPane, controls, copied, filters: () => filters, deny: () => { privileged = false; } };
}

test('pause follow retains the scroll position while continuing to show loaded lines', () => {
  const h = controlsFixture();
  h.controls.get('toggleLogFollowBtn').trigger('click');
  h.logPane.scrollTop = 123;
  h.state.jobs.find(job => job.id === 'job-b').logs.push('third');
  h.api.renderLogs();
  assert.equal(h.logPane.scrollTop, 123);
  assert.equal(h.logPane.textContent, 'first\nsecond\nthird\n');
  assert.equal(h.controls.get('toggleLogFollowBtn').attributes.get('aria-pressed'), 'false');
  h.controls.get('toggleLogFollowBtn').trigger('click');
  assert.equal(h.logPane.scrollTop, 1000);
});

test('filter controls leave selected inspector identity intact and log search does not truncate copied data', () => {
  const h = controlsFixture();
  const search = h.controls.get('loadedRunSearch'); search.value = 'archive'; search.trigger('input');
  assert.equal(h.filters(), 1);
  assert.deepEqual(h.api.visibleJobs().map(job => job.id), ['job-c']);
  assert.equal(h.state.selectedJobId, 'job-b');
  const logs = h.controls.get('loadedLogSearch'); logs.value = 'second'; logs.trigger('input');
  assert.equal(h.logPane.textContent, 'second\n');
  h.controls.get('copyLoadedLogsBtn').trigger('click');
  assert.deepEqual(h.copied, ['first\nsecond\n']);
});

test('clearing loaded logs is local and newly selected runs reset log search and following', () => {
  const h = controlsFixture();
  h.api.clearLogs();
  assert.equal(h.logPane.textContent, '');
  assert.deepEqual(h.state.jobs.find(job => job.id === 'job-b').logs, []);
  h.state.selectedJobId = 'job-c'; h.api.renderLogs();
  assert.equal(h.logPane.textContent, 'first\nsecond\n');
  assert.equal(h.controls.get('loadedLogSearch').value, '');
  assert.equal(h.controls.get('toggleLogFollowBtn').attributes.get('aria-pressed'), 'true');
});

test('exports recheck current privileges at the click and searches clear across account changes', () => {
  const h = controlsFixture();
  h.deny(); h.controls.get('copyLoadedLogsBtn').trigger('click');
  assert.deepEqual(h.copied, []);
  const search = h.controls.get('loadedRunSearch'); search.value = 'archive'; search.trigger('input');
  h.state.auth.actor.username = 'bob'; h.api.renderQueueControls();
  assert.equal(search.value, '');
  assert.equal(h.api.visibleJobs().length, 3);
});

test('live appended lines honour log search and paused follow without rewriting the pane', () => {
  const frames = [];
  const h = controlsFixture();
  const pane = h.logPane;
  pane.childNodes = [];
  pane.appendChild = node => { pane.childNodes.push(node); return node; };
  pane.removeChild = node => { pane.childNodes.splice(pane.childNodes.indexOf(node), 1); return node; };
  Object.defineProperty(pane, 'firstChild', { get: () => pane.childNodes[0] });
  // A second tool set over the same controls takes over their listeners.
  const api = createOperateTools({ state: h.state, els: { logPane: pane }, maxLogLines: 1200,
    document: { getElementById: id => h.controls.get(id), createElement: () => ({}), createTextNode: data => ({ data }) },
    requestAnimationFrame: callback => { frames.push(callback); return frames.length; },
    _displayJobState: job => job.state, _portalPrivilegesReady: () => true, onFilterChange: () => {} });
  const job = h.state.jobs.find(item => item.id === 'job-b');
  const logSearch = h.controls.get('loadedLogSearch'); logSearch.value = 'warn'; logSearch.trigger('input');
  const hydratedText = pane.textContent;
  for (const line of ['[INFO] skipped', '[WARN] shown']) { job.logs.push(line); api.appendLog('job-b', line); }
  assert.equal(pane.textContent, hydratedText, 'appending never rewrites the hydrated pane');
  assert.deepEqual(pane.childNodes.map(node => node.data), ['[WARN] shown\n']);
  assert.equal(frames.length, 1, 'autoscroll and summary are coalesced into one frame');
  h.controls.get('toggleLogFollowBtn').trigger('click');
  pane.scrollTop = 7;
  frames[0]();
  assert.equal(pane.scrollTop, 7, 'paused follow keeps the scroll position');
  assert.match(h.controls.get('loadedLogSummary').textContent, /^1 matching \/ 4 loaded lines/);
  api.appendLog('job-c', '[WARN] another run');
  assert.deepEqual(pane.childNodes.map(node => node.data), ['[WARN] shown\n'], 'lines of another run are not appended');
});
