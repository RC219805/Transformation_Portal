import test from 'node:test';
import assert from 'node:assert/strict';
import { createPortalHistoryNavigation } from '../portal-src/internal/history-navigation.js';

function fixture() {
  const entries = [{ url: 'build', state: null }];
  let index = 0, dirty = false, pending, afterTraversal, prompts = 0, applied = [];
  const traversals = [];
  const history = {
    get state() { return entries[index].state; },
    replaceState(state, _title, url) { entries[index] = { state, url: url || entries[index].url }; },
    pushState(state, _title, url) { entries.splice(++index); entries.push({ state, url }); },
    go(offset) {
      traversals.push(offset);
      if (offset === 0) return; // Browsers reload instead of emitting another same-document popstate.
      queueMicrotask(() => {
        index += offset;
        const after = afterTraversal; afterTraversal = null;
        router.pop({ state: entries[index].state }); after?.();
      });
    },
  };
  const router = createPortalHistoryNavigation({ history, hasEdits: () => dirty,
    requestNavigation: action => { prompts += 1; pending = action; }, apply: () => applied.push(entries[index].url) });
  return { router, entries, history, applied, traversals, dirty: value => { dirty = value; },
    fragment: url => history.pushState({ ...history.state }, '', url),
    current: () => entries[index].url, discard({ clearEdits = true } = {}) {
      if (clearEdits) dirty = false;
      const action = pending; pending = null; action();
    },
    afterNextTraversal: action => { afterTraversal = action; },
    hasPrompt: () => !!pending, promptCount: () => prompts,
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

test('dirty fragment Back stays on Review without reloading and preserves the next owned Back guard', async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate' }, 'operate');
  h.router.record(false, { view: 'review' }, 'review');
  h.fragment('review#main-content');
  h.dirty(true);
  const original = structuredClone(h.entries);
  h.history.go(-1); await settle();
  assert.equal(h.current(), 'review');
  assert.deepEqual(h.traversals, [-1], 'A same-route fragment traversal must never request history.go(0)');
  assert.deepEqual(h.applied, ['review']);
  assert.equal(h.hasPrompt(), false); assert.equal(h.promptCount(), 0);

  h.history.go(-1); await settle();
  assert.equal(h.current(), 'review'); assert.equal(h.hasPrompt(), true); assert.equal(h.promptCount(), 1);
  assert.deepEqual(h.traversals, [-1, -1, 1]);
  h.cancel();
  assert.deepEqual(h.entries, original);
  h.history.go(-1); await settle(); h.discard(); await settle();
  assert.equal(h.current(), 'operate'); assert.deepEqual(h.applied, ['review', 'operate']);
  assert.equal(h.promptCount(), 2); assert.equal(h.hasPrompt(), false);
  assert.equal(h.traversals.includes(0), false); assert.deepEqual(h.entries, original);
});

test('an equal-index pop cannot revoke an already permitted owned Back', async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate' }, 'operate');
  h.router.record(false, { view: 'review' }, 'review');
  h.dirty(true);
  h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true);
  h.discard({ clearEdits: false });
  h.router.pop({ state: h.history.state });
  await settle();
  assert.equal(h.current(), 'operate'); assert.equal(h.hasPrompt(), false); assert.equal(h.promptCount(), 1);
  assert.deepEqual(h.applied, ['review', 'operate']);
  assert.equal(h.traversals.includes(0), false);
});

for (const offset of [-1, -2]) test(`a background Review replacement while restoring dirty Back by ${-offset} entries preserves history and its discard prompt`, async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate', jobId: 'job-a' }, 'operate');
  h.router.record(false, { view: 'review', jobId: 'job-a' }, 'review');
  const original = structuredClone(h.entries), target = offset === -1 ? 'operate' : 'build';
  h.dirty(true);
  h.afterNextTraversal(() => h.router.record(true, { view: 'review', jobId: 'job-b' }, 'review-background'));
  h.history.go(offset); await settle();
  assert.equal(h.current(), 'review'); assert.equal(h.hasPrompt(), true); assert.equal(h.promptCount(), 1);
  assert.deepEqual(h.entries, original); assert.deepEqual(h.traversals, [offset, -offset]);
  h.discard({ clearEdits: false }); await settle();
  assert.equal(h.current(), target); assert.deepEqual(h.applied, [target]);
  assert.equal(h.hasPrompt(), false); assert.equal(h.promptCount(), 1);
  assert.deepEqual(h.entries, original); assert.deepEqual(h.traversals, [offset, -offset, offset]);
});

for (const offset of [-1, -2]) test(`a background Review replacement cannot revoke permitted Back by ${-offset} entries`, async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate', jobId: 'job-a' }, 'operate');
  h.router.record(false, { view: 'review', jobId: 'job-a' }, 'review');
  const original = structuredClone(h.entries), target = offset === -1 ? 'operate' : 'build';
  h.dirty(true); h.history.go(offset); await settle();
  assert.equal(h.current(), 'review'); assert.equal(h.hasPrompt(), true);
  h.discard({ clearEdits: false });
  h.router.record(true, { view: 'review', jobId: 'job-b' }, 'review-background');
  await settle();
  assert.equal(h.current(), target); assert.deepEqual(h.applied, [target]);
  assert.equal(h.hasPrompt(), false); assert.equal(h.promptCount(), 1);
  assert.deepEqual(h.entries, original); assert.deepEqual(h.traversals, [offset, -offset, offset]);
});

test('replacements preserve index and view/job metadata without adding entries', () => {
  const h = fixture(); h.router.record(false, { view: 'review', jobId: 'job-a' }, 'review');
  h.router.record(true, { view: 'review', jobId: 'job-b' }, 'review-b');
  assert.equal(h.entries.length, 2);
  assert.deepEqual(h.history.state, { view: 'review', jobId: 'job-b', portalIndex: 1 });
});
