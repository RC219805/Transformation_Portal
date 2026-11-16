import test from 'node:test';
import assert from 'node:assert/strict';
import { createPortalHistoryNavigation } from '../portal-src/internal/history-navigation.js';

function fixture() {
  const entries = [{ url: 'build', state: null }];
  let index = 0, dirty = false, pending, applied = [];
  const history = {
    get state() { return entries[index].state; },
    replaceState(state, _title, url) { entries[index] = { state, url: url || entries[index].url }; },
    pushState(state, _title, url) { entries.splice(++index); entries.push({ state, url }); },
    go(offset) { queueMicrotask(() => { index += offset; router.pop({ state: entries[index].state }); }); },
  };
  const router = createPortalHistoryNavigation({ history, hasEdits: () => dirty,
    requestNavigation: action => { pending = action; }, apply: () => applied.push(entries[index].url) });
  return { router, entries, history, applied, dirty: value => { dirty = value; },
    current: () => entries[index].url, discard() { dirty = false; const action = pending; pending = null; action(); },
    cancel() { pending = null; } };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('cancelled Back preserves Operate and the complete forward history', async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate' }, 'operate');
  h.router.record(false, { view: 'review' }, 'review');
  h.dirty(true);
  h.history.go(-1); await settle();
  assert.equal(h.current(), 'review'); assert.deepEqual(h.applied, []);
  h.cancel();
  assert.deepEqual(h.entries.map(row => row.url), ['build', 'operate', 'review']);
  h.history.go(-1); await settle(); h.discard(); await settle();
  assert.equal(h.current(), 'operate'); assert.deepEqual(h.applied, ['operate']);
  assert.deepEqual(h.entries.map(row => row.url), ['build', 'operate', 'review']);
  h.history.go(1); await settle(); assert.equal(h.current(), 'review');
});

test('multi-entry traversal restores position until explicit discard', async () => {
  const h = fixture(); h.router.record(false, {}, 'operate'); h.router.record(false, {}, 'review');
  h.dirty(true); h.history.go(-2); await settle(); assert.equal(h.current(), 'review');
  h.discard(); await settle(); assert.equal(h.current(), 'build');
  assert.deepEqual(h.applied, ['build']);
});

test('replacements preserve index and view/job metadata without adding entries', () => {
  const h = fixture(); h.router.record(false, { view: 'review', jobId: 'job-a' }, 'review');
  h.router.record(true, { view: 'review', jobId: 'job-b' }, 'review-b');
  assert.equal(h.entries.length, 2);
  assert.deepEqual(h.history.state, { view: 'review', jobId: 'job-b', portalIndex: 1 });
});
