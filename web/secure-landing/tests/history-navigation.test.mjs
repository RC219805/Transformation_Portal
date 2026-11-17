import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createPortalHistoryNavigation } from '../portal-src/internal/history-navigation.js';
import { createPortalRouteHelpers } from '../portal-src/internal/routing.js';

function fixture({ initialEntries = [{ url: 'build', state: null }], initialIndex = 0, withLocation = true, locationHref } = {}) {
  const entries = structuredClone(initialEntries);
  let index = initialIndex, dirty = false, pending, afterTraversal, prompts = 0, applied = [];
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
    ...(withLocation ? { location: { get href() { return locationHref ?? new URL(entries[index].url, 'https://portal.invalid/').href; } } } : {}),
    requestNavigation: action => { prompts += 1; pending = action; }, apply: () => applied.push(entries[index].url) });
  return { router, entries, history, applied, traversals, dirty: value => { dirty = value; },
    fragment: url => history.pushState({ ...history.state }, '', url),
    current: () => entries[index].url, discard({ clearEdits = true } = {}) {
      if (clearEdits) dirty = false;
      const action = pending; pending = null; return action();
    },
    afterNextTraversal: action => { afterTraversal = action; },
    hasPrompt: () => !!pending, promptCount: () => prompts,
    cancel() { pending = null; } };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

function routeHostFixture({ throwAfterApply = false } = {}) {
  const origin = 'https://portal.test';
  const entries = [
    { url: `${origin}/portal?view=build`, state: { view: 'build', portalIndex: 0 } },
    { url: `${origin}/portal?view=review&job=job-a`, state: { view: 'review', jobId: 'job-a', portalIndex: 1 } },
  ];
  let index = 1, dirty = false, pending, applied = 0, hostRoute;
  const location = Object.fromEntries(['href', 'pathname', 'search', 'hash'].map(key => [key, null]));
  for (const key of Object.keys(location)) Object.defineProperty(location, key, {
    get: () => key === 'href' ? entries[index].url : new URL(entries[index].url)[key],
  });
  const history = {
    get state() { return entries[index].state; },
    replaceState(state, _title, url) {
      entries[index] = { state, url: url === undefined ? location.href : new URL(url, location.href).href };
    },
    pushState(state, _title, url) {
      const next = { state, url: new URL(url, location.href).href };
      entries.splice(++index); entries.push(next);
    },
    go(offset) {
      assert.notEqual(offset, 0);
      queueMicrotask(() => { index += offset; router.pop({ state: history.state }); });
    },
  };
  const router = createPortalHistoryNavigation({ history, location, hasEdits: () => dirty,
    requestNavigation: action => { pending = action; },
    apply: () => {
      applied += 1; hostRoute.applyConsoleRouteFromLocation(true);
      if (throwAfterApply) throw new Error('host render failed');
    },
  });
  const state = { currentView: 'review', selectedJobId: 'job-a', jobs: [
    { id: 'job-a', logsLoadStatus: 'ready', logs: [] }, { id: 'job-b', logsLoadStatus: 'ready', logs: [] },
  ] };
  const routeContexts = {};
  const context = {
    window: { location }, state, portalHistoryNavigation: router,
    portalRoute: createPortalRouteHelpers({ location }), deferredProfileOpenIntent: 0,
    resolveConsoleView: value => value,
    _normalizeSelectedJobId: value => String(value || ''),
    _normalizeArtifactRoutePath: value => String(value || ''),
    _normalizeCompareQueryValue: value => value === '1',
    _activeRouteContext: jobId => routeContexts[jobId] || { artifactPath: '', compareEnabled: false },
    _preferredSelectedJobId: () => 'job-a',
    _rememberSelectedJob() {}, _rememberArtifactSelection() {}, _rememberComparePreference() {},
    _primeDeferredReviewSurface() {}, _primeDeferredOperateSurface() {}, _primeDeferredBuildSurface() {},
    applyConsoleViewLayout() {}, els: {}, _isJobStreamRecoverable: () => false,
    renderReviewSurfaces() {}, setInspectorTab() {}, updateConsoleViewContext() {},
    emitPortalEvent() {}, scheduleRenderJobQueue() {},
    _requestReviewNavigation(action) { if (!dirty) return false; pending = action; return true; },
  };
  const source = readFileSync(new URL('../portal-src/portal.template.js', import.meta.url), 'utf8');
  const section = (start, end) => {
    const first = source.indexOf(start), last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first);
    return source.slice(first, last);
  };
  hostRoute = vm.runInNewContext(section('function _routeUrlForView(', 'function _managedReturnToPath(')
    + section('function applyConsoleRouteFromLocation(', 'function setupSectionRail(')
    + section('function selectJob(', 'function hydrateJobFromServer(')
    + '\n({ applyConsoleRouteFromLocation, sync: _syncConsoleRoute, selectJob })', context);
  return { router, history, entries, state, location, hostRoute, routeContexts, dirty: value => { dirty = value; },
    applied: () => applied, hasPrompt: () => !!pending,
    discard() { dirty = false; const action = pending; pending = null; return action(); },
    cancel() { pending = null; },
  };
}

for (const browserState of [null, {}]) test(`an accepted unindexed Operate route synchronizes its selected job without stamping an index (${JSON.stringify(browserState)})`, () => {
  const h = routeHostFixture();
  h.history.pushState(browserState, '', '/portal?view=operate');
  h.router.pop({ state: null });
  assert.equal(h.state.currentView, 'operate'); assert.equal(h.state.selectedJobId, 'job-a');
  assert.equal(h.location.search, '?view=operate&job=job-a');
  assert.equal(h.history.state, browserState);
  h.hostRoute.selectJob('job-b');
  assert.equal(h.state.selectedJobId, 'job-b');
  assert.equal(h.location.search, '?view=operate&job=job-b', 'A later accepted queue selection synchronizes the active route');
  assert.equal(h.history.state, browserState);
});

test('active unindexed inspection synchronization preserves the browser state and selected job', () => {
  const h = routeHostFixture(); h.history.pushState({}, '', '/portal?view=review');
  h.router.pop({ state: null });
  h.routeContexts['job-a'] = { artifactPath: 'output/final.png', compareEnabled: true };
  h.hostRoute.sync(true);
  assert.equal(h.location.search, '?view=review&job=job-a&artifact=output%2Ffinal.png&compare=1');
  assert.equal(h.history.state.portalIndex, undefined);
});

for (const replace of [false, true]) test(`external ${replace ? 'replacement' : 'push'} from active unindexed authoring fences the new destination before a render`, () => {
  const h = routeHostFixture(); h.history.pushState({}, '', '/portal?view=operate');
  h.router.pop({ state: null });
  h.history[replace ? 'replaceState' : 'pushState'](null, '', '/portal?view=review&job=other');
  const original = structuredClone(h.entries);
  h.hostRoute.sync(true);
  assert.equal(h.location.search, '?view=review&job=other'); assert.deepEqual(h.entries, original);
  h.dirty(true); h.router.pop({ state: null });
  assert.equal(h.hasPrompt(), true); assert.equal(h.applied(), 1);
  h.cancel(); h.hostRoute.sync(true);
  assert.deepEqual(h.entries, original);
});

test('an active unindexed job selection still obeys the existing dirty draft guard', () => {
  const h = routeHostFixture(); h.history.pushState({}, '', '/portal?view=operate');
  h.router.pop({ state: null }); h.dirty(true);
  h.hostRoute.selectJob('job-b');
  assert.equal(h.hasPrompt(), true); assert.equal(h.state.selectedJobId, 'job-a');
  assert.equal(h.location.search, '?view=operate&job=job-a');
  h.cancel(); assert.equal(h.state.selectedJobId, 'job-a');
  h.hostRoute.selectJob('job-b'); h.discard();
  assert.equal(h.state.selectedJobId, 'job-b'); assert.equal(h.location.search, '?view=operate&job=job-b');
  assert.equal(h.history.state.portalIndex, undefined);
});

test('dirty active unindexed Review keeps hash-only Back and Forward within the same authoring source', async () => {
  const h = routeHostFixture(); h.history.pushState({}, '', '/portal?view=review');
  h.router.pop({ state: null }); h.dirty(true);
  h.history.pushState(null, '', '/portal?view=review&job=job-a#main-content');
  h.router.pop({ state: null });
  assert.equal(h.hasPrompt(), false); assert.equal(h.state.selectedJobId, 'job-a');
  h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), false); assert.equal(h.location.hash, '');
  h.history.go(1); await settle();
  assert.equal(h.hasPrompt(), false); assert.equal(h.location.hash, '#main-content');
  assert.equal(h.applied(), 4); assert.equal(h.history.state, null);
});

for (const browserState of [null, {}]) test(`a dirty unknown route canonicalizes only after explicit discard (${JSON.stringify(browserState)})`, () => {
  const h = routeHostFixture(); h.dirty(true);
  h.history.pushState(browserState, '', '/portal?view=operate');
  h.router.pop({ state: null });
  assert.equal(h.hasPrompt(), true); assert.equal(h.applied(), 0);
  h.hostRoute.sync(true); h.cancel(); h.hostRoute.sync(true);
  assert.equal(h.location.search, '?view=operate'); assert.equal(h.history.state, browserState);
  h.router.pop({ state: null }); h.discard();
  assert.equal(h.applied(), 1); assert.equal(h.location.search, '?view=operate&job=job-a');
  assert.equal(h.history.state, browserState);
});

test('an unknown canonical route remains unindexed and guards a newly edited return to the owned source', async () => {
  const h = routeHostFixture(); h.history.pushState({}, '', '/portal?view=operate');
  h.router.pop({ state: null });
  assert.equal(h.location.search, '?view=operate&job=job-a');
  h.dirty(true); h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true); assert.equal(h.applied(), 1);
  assert.equal(h.entries[2].state.portalIndex, undefined);
  h.cancel(); h.hostRoute.sync(true);
  assert.equal(h.location.search, '?view=review&job=job-a');
  assert.equal(h.entries[2].url, 'https://portal.test/portal?view=operate&job=job-a');
});

test('a throwing authorized host apply cannot leave later background synchronization enabled', () => {
  const h = routeHostFixture({ throwAfterApply: true });
  h.history.pushState({}, '', '/portal?view=operate');
  assert.throws(() => h.router.pop({ state: null }), /host render failed/);
  assert.equal(h.location.search, '?view=operate&job=job-a');
  h.state.selectedJobId = 'background-job'; h.hostRoute.sync(true);
  assert.equal(h.location.search, '?view=operate&job=job-a');
  assert.equal(h.history.state.portalIndex, undefined);
});

for (const initialIndex of [0, 3]) test(`an unknown fork after Back and a forward jump advances the ${initialIndex ? 'initial' : 'observed'} owned sequence`, async () => {
  const h = fixture({ initialEntries: [
    { url: 'build', state: { view: 'build', portalIndex: 0 } },
    { url: 'operate', state: { view: 'operate', portalIndex: 5 } },
    { url: 'browser-review', state: null },
    { url: 'old-review', state: { view: 'review', portalIndex: 7 } },
  ], initialIndex });
  if (initialIndex === 0) { h.history.go(3); await settle(); }
  h.history.go(-3); await settle();
  h.history.go(2); await settle();
  assert.equal(h.current(), 'browser-review');
  h.router.record(false, { view: 'review' }, 'new-review');
  assert.equal(h.history.state.portalIndex, 8);
  assert.equal(h.entries[2].state, null);
  assert.equal(h.entries.some(entry => entry.url === 'old-review'), false);
  const original = structuredClone(h.entries);
  h.applied.length = 0; h.dirty(true); h.history.go(-2); await settle();
  assert.equal(h.current(), 'new-review'); assert.equal(h.hasPrompt(), true);
  assert.deepEqual(h.applied, []); h.cancel();
  h.history.go(-2); await settle(); h.discard(); await settle();
  assert.equal(h.current(), 'operate'); assert.deepEqual(h.applied, ['operate']);
  h.applied.length = 0; h.dirty(true); h.history.go(2); await settle();
  assert.equal(h.current(), 'operate'); assert.equal(h.hasPrompt(), true);
  h.discard(); await settle();
  assert.equal(h.current(), 'new-review'); assert.deepEqual(h.applied, ['new-review']);
  assert.deepEqual(h.entries, original); assert.equal(h.traversals.includes(0), false);
});

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

test('Back across an earlier fragment replays the requested physical destination after cancel and discard', async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate' }, 'operate');
  h.fragment('operate#main-content');
  h.router.record(false, { view: 'review' }, 'review');
  const original = structuredClone(h.entries);
  h.dirty(true); h.history.go(-3); await settle();
  assert.equal(h.current(), 'review'); assert.equal(h.hasPrompt(), true);
  assert.deepEqual(h.applied, []); assert.deepEqual(h.entries, original);
  h.cancel();
  h.history.go(-3); await settle(); h.discard(); await settle();
  assert.equal(h.current(), 'build'); assert.deepEqual(h.applied, ['build']);
  assert.equal(h.hasPrompt(), false); assert.deepEqual(h.entries, original);
  assert.equal(h.traversals.includes(0), false);
  h.history.go(3); await settle(); assert.equal(h.current(), 'review');
});

test('restoring and permitted fragment-spanning traversal retain the target during background replacements', async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate' }, 'operate');
  h.fragment('operate#main-content');
  h.router.record(false, { view: 'review' }, 'review');
  const original = structuredClone(h.entries);
  h.dirty(true);
  h.afterNextTraversal(() => h.router.record(true, { view: 'review' }, 'review-background'));
  h.history.go(-3); await settle();
  assert.equal(h.current(), 'review'); assert.equal(h.promptCount(), 1);
  assert.deepEqual(h.entries, original);
  h.discard({ clearEdits: false });
  h.router.record(true, { view: 'review' }, 'review-background');
  await settle();
  assert.equal(h.current(), 'build'); assert.deepEqual(h.applied, ['build']);
  assert.equal(h.promptCount(), 1); assert.equal(h.hasPrompt(), false);
  assert.deepEqual(h.entries, original); assert.equal(h.traversals.includes(0), false);
});

test('a router push after Forward truncation still counts an intervening fragment during dirty Back', async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate' }, 'operate');
  h.router.record(false, { view: 'review' }, 'old-review');
  h.history.go(-1); await settle();
  h.fragment('operate#main-content');
  h.router.record(false, { view: 'review' }, 'new-review');
  const original = structuredClone(h.entries);
  h.applied.length = 0; h.dirty(true);
  h.history.go(-3); await settle();
  assert.equal(h.current(), 'new-review'); assert.equal(h.hasPrompt(), true);
  h.discard(); await settle();
  assert.equal(h.current(), 'build'); assert.deepEqual(h.applied, ['build']);
  assert.deepEqual(h.entries, original);
  assert.equal(h.entries.some(row => row.url === 'old-review'), false);
  assert.equal(h.traversals.includes(0), false);
});

for (const fragments of [1, 3]) test(`dirty Forward restores and replays across ${fragments} physical fragment entries`, async () => {
  const h = fixture();
  h.router.record(false, { view: 'review' }, 'first-review');
  h.router.record(false, { view: 'operate' }, 'operate');
  for (let n = 0; n < fragments; n += 1) h.fragment(`operate#${n % 2 ? 'alternate' : 'main-content'}`);
  h.router.record(false, { view: 'review' }, 'last-review');
  const distance = fragments + 2;
  h.history.go(-distance); await settle();
  assert.equal(h.current(), 'first-review');
  h.applied.length = 0; h.dirty(true);
  const original = structuredClone(h.entries);
  h.history.go(distance); await settle();
  assert.equal(h.current(), 'first-review'); assert.equal(h.hasPrompt(), true);
  assert.deepEqual(h.applied, []); assert.deepEqual(h.entries, original);
  h.cancel();
  h.history.go(distance); await settle(); h.discard(); await settle();
  assert.equal(h.current(), 'last-review'); assert.deepEqual(h.applied, ['last-review']);
  assert.equal(h.hasPrompt(), false); assert.deepEqual(h.entries, original);
  assert.equal(h.traversals.includes(0), false);
});

test('restoration counts an intermediate browser-owned fragment without applying its route', async () => {
  const h = fixture({ initialEntries: [
    { url: 'build', state: { portalIndex: 0 } },
    { url: 'operate', state: { portalIndex: 1 } },
    { url: 'operate#main-content', state: null },
    { url: 'review', state: { portalIndex: 2 } },
  ], initialIndex: 3 });
  const original = structuredClone(h.entries);
  h.dirty(true); h.history.go(-3); await settle();
  assert.equal(h.current(), 'review'); assert.equal(h.hasPrompt(), true);
  assert.deepEqual(h.applied, []);
  h.discard(); await settle();
  assert.equal(h.current(), 'build'); assert.deepEqual(h.applied, ['build']);
  assert.deepEqual(h.entries, original); assert.equal(h.traversals.includes(0), false);
});

for (const forward of [false, true]) test(`direct ${forward ? 'Forward' : 'Back'} to a null-state fragment is adopted before a later router push`, async () => {
  const h = fixture();
  h.router.record(false, { view: 'operate', jobId: 'job-a' }, 'operate');
  h.history.pushState(null, '', 'operate#main-content');
  h.router.pop({ state: null });
  h.router.record(false, { view: 'review', jobId: 'job-a' }, 'review');
  assert.deepEqual(h.entries[2], { url: 'operate#main-content', state: { view: 'operate', jobId: 'job-a', portalIndex: 1 } });
  if (forward) { h.history.go(-3); await settle(); }
  h.applied.length = 0;
  const original = structuredClone(h.entries), source = h.current(), offset = forward ? 2 : -1;
  h.dirty(true); h.history.go(offset); await settle();
  assert.equal(h.current(), source); assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  h.cancel(); assert.deepEqual(h.entries, original);
  h.history.go(offset); await settle(); h.discard(); await settle();
  assert.equal(h.current(), 'operate#main-content'); assert.deepEqual(h.applied, ['operate#main-content']);
  assert.deepEqual(h.entries, original); assert.equal(h.traversals.includes(0), false);
});

for (const offset of [-1, 1]) test(`pre-router unknown ${offset < 0 ? 'Back' : 'Forward'} keeps authoring and the destination fenced through cancel and background replacement`, async () => {
  const unknown = { url: 'operate#main-content', state: null }, owned = { url: 'review', state: null };
  const h = fixture({ initialEntries: offset < 0 ? [unknown, owned] : [owned, unknown], initialIndex: offset < 0 ? 1 : 0 });
  const original = structuredClone(h.entries);
  h.dirty(true); h.history.go(offset); await settle();
  assert.equal(h.current(), unknown.url, 'Unknown direction leaves the browser-owned URL intact');
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []); assert.deepEqual(h.entries, original);
  h.cancel();
  h.router.record(true, { view: 'review', jobId: 'background' }, 'review-background');
  assert.deepEqual(h.entries, original); assert.deepEqual(h.applied, []);
  h.dirty(false);
  h.router.record(true, { view: 'review', jobId: 'background' }, 'review-background');
  assert.deepEqual(h.entries, original, 'A persistence-only render cannot release the canceled fence');
  h.history.go(-offset); await settle();
  assert.equal(h.current(), owned.url);
  h.router.record(true, { view: 'review', jobId: 'fresh' }, 'review-current');
  assert.equal(h.current(), 'review-current');
  assert.equal(h.entries[offset < 0 ? 0 : 1].url, unknown.url);
  assert.equal(h.traversals.includes(0), false);
});

test('explicit discard applies an unknown destination once and canonicalizes it without assigning an index', async () => {
  const h = fixture({ initialEntries: [{ url: 'operate#main-content', state: null }, { url: 'review', state: null }], initialIndex: 1 });
  h.dirty(true); h.history.go(-1); await settle();
  h.discard();
  assert.deepEqual(h.applied, ['operate#main-content']);
  h.router.record(true, { view: 'operate' }, 'operate-current');
  assert.deepEqual(h.entries[0], { url: 'operate-current', state: null });
  h.history.go(1); await settle();
  h.router.record(true, { view: 'review' }, 'review-current');
  assert.equal(h.current(), 'review-current'); assert.equal(h.entries[0].url, 'operate-current');
  assert.deepEqual(h.traversals, [-1, 1]);
});

for (const discard of [false, true]) test(`${discard ? 'discarded' : 'clean'} unknown visit cannot stamp a duplicate index that bypasses a later dirty Back`, async () => {
  const h = fixture({ initialEntries: [{ url: 'operate#main-content', state: null }, { url: 'review', state: null }], initialIndex: 1 });
  if (discard) h.dirty(true);
  h.history.go(-1); await settle();
  if (discard) h.discard();
  h.router.record(true, { view: 'operate' }, 'operate#main-content');
  assert.equal(h.entries[0].state, null);
  h.history.go(1); await settle();
  assert.equal(h.current(), 'review'); h.dirty(true);
  h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true);
  assert.deepEqual(h.applied, ['operate#main-content', 'review']);
  h.cancel();
  h.router.record(true, { view: 'review' }, 'review-background');
  assert.deepEqual(h.entries[0], { url: 'operate#main-content', state: null });
  assert.equal(h.traversals.includes(0), false);
});

for (const target of [
  'https://portal.test/portal?view=review&job=other&artifact=one#main-content',
  'https://portal.test/portal?view=review&job=one&artifact=other#main-content',
  'https://portal.test/portal?view=review&job=one&artifact=one&compare=1#main-content',
]) test(`unknown query identity ${target} cannot use the same-fragment fast path`, async () => {
  const h = fixture({ initialEntries: [{ url: target, state: null },
    { url: 'https://portal.test/portal?view=review&job=one&artifact=one', state: null }], initialIndex: 1 });
  h.dirty(true); h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  h.cancel(); h.router.record(true, { view: 'review' }, 'review-background');
  assert.deepEqual(h.entries[0], { url: target, state: null });
});

test('a caller without a location seam keeps an unknown destination fenced after discard', async () => {
  const h = fixture({ withLocation: false,
    initialEntries: [{ url: 'operate#main-content', state: null }, { url: 'review', state: null }], initialIndex: 1 });
  h.dirty(true); h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  h.discard(); h.router.record(true, { view: 'operate' }, 'operate-background');
  assert.deepEqual(h.applied, ['operate#main-content']);
  assert.deepEqual(h.entries[0], { url: 'operate#main-content', state: null });
});

test('an invalid relative location seam cannot authorize or index an unknown destination', async () => {
  const h = fixture({ locationHref: 'portal?view=review',
    initialEntries: [{ url: 'operate#main-content', state: null }, { url: 'review', state: null }], initialIndex: 1 });
  h.dirty(true); h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  h.discard(); h.router.record(true, { view: 'operate' }, 'operate-background');
  assert.deepEqual(h.applied, ['operate#main-content']);
  assert.deepEqual(h.entries[0], { url: 'operate#main-content', state: null });
  assert.deepEqual(h.traversals, [-1]);
});

test('canceling owned Back retains the source URL identity for a later unknown destination', async () => {
  const h = fixture({ initialEntries: [
    { url: 'build', state: { portalIndex: 0 } },
    { url: 'operate', state: { portalIndex: 1 } },
    { url: 'operate#main-content', state: null },
    { url: 'review', state: { portalIndex: 2 } },
  ], initialIndex: 3 });
  h.dirty(true); h.history.go(-2); await settle(); h.cancel();
  assert.equal(h.current(), 'review');
  h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  assert.deepEqual(h.entries[2], { url: 'operate#main-content', state: null });
});

for (const discard of [false, true]) test(`new dirty authoring on a ${discard ? 'discarded' : 'cleanly applied'} unknown route guards return to the old owned index`, async () => {
  const h = fixture({ initialEntries: [{ url: 'review-b#main-content', state: null }, { url: 'review-a', state: null }], initialIndex: 1 });
  if (discard) h.dirty(true);
  h.history.go(-1); await settle();
  if (discard) h.discard();
  assert.deepEqual(h.applied, ['review-b#main-content']);
  h.dirty(true); h.history.go(1); await settle();
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, ['review-b#main-content']);
  h.cancel(); h.router.record(true, { view: 'review' }, 'review-b#main-content');
  assert.equal(h.current(), 'review-a'); assert.equal(h.entries[1].state.portalIndex, 0);
  h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true, 'An unindexed source remains guarded until its explicit application');
  h.discard();
  assert.equal(h.current(), 'review-b#main-content');
  assert.equal(h.entries[0].state, null); assert.equal(h.traversals.includes(0), false);
});

test('same-index entries with different artifact URLs require explicit discard without go(0)', async () => {
  const a = 'https://portal.test/portal?view=review&job=one&artifact=a';
  const b = 'https://portal.test/portal?view=review&job=one&artifact=b#main-content';
  const h = fixture(); h.router.record(false, { view: 'review', jobId: 'one' }, a);
  h.fragment(a + '#main-content');
  h.router.record(true, { view: 'review', jobId: 'one' }, b);
  const original = structuredClone(h.entries);
  h.dirty(true); h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  h.cancel(); h.router.record(true, { view: 'review', jobId: 'one' }, b);
  assert.deepEqual(h.entries, original);
  h.history.go(1); await settle();
  assert.equal(h.current(), b);
  h.history.go(-1); await settle(); h.discard();
  assert.equal(h.current(), a); assert.deepEqual(h.entries, original);
  assert.equal(h.traversals.includes(0), false);
});

test('a pending replacement updates active authoring URL without stamping the browser destination', async () => {
  const a = 'https://portal.test/portal?view=review&job=a';
  const b = 'https://portal.test/portal?view=review&job=b';
  const h = fixture({ initialEntries: [{ url: a, state: { portalIndex: 0 } }, { url: b + '&raw=1', state: null }] });
  h.dirty(true); h.history.go(1); await settle(); h.cancel();
  h.router.record(true, { view: 'review', jobId: 'a' }, a);
  assert.deepEqual(h.entries[1], { url: b + '&raw=1', state: null });
  h.dirty(true); h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), false, 'The host retained authoring on A, so returning to A keeps the draft');
  assert.equal(h.current(), a);
  assert.deepEqual(h.entries[1], { url: b + '&raw=1', state: null });
});

for (const replace of [false, true]) test(`an external different-route ${replace ? 'replace' : 'push'} is fenced before a background record`, () => {
  const a = 'https://portal.test/portal?view=review&job=a';
  const b = 'https://portal.test/portal?view=review&job=b#main-content';
  const h = fixture(); h.router.record(false, { view: 'review', jobId: 'a' }, a);
  h.history[replace ? 'replaceState' : 'pushState'](null, '', b);
  const original = structuredClone(h.entries);
  h.router.record(true, { view: 'review', jobId: 'a' }, a);
  assert.equal(h.current(), b); assert.deepEqual(h.entries, original);
  assert.deepEqual(h.applied, []);
});

for (const forward of [false, true]) test(`${forward ? 'Forward' : 'Back'} restoration walks past a same-index clone with a different source URL`, async () => {
  const a = 'https://portal.test/portal?view=review&job=one&artifact=a';
  const b = 'https://portal.test/portal?view=review&job=one&artifact=b#main-content';
  const h = fixture(); h.router.record(false, { view: 'review', jobId: 'one' }, a);
  h.fragment(a + '#main-content'); h.router.record(true, { view: 'review', jobId: 'one' }, b);
  if (forward) { h.router.record(false, { view: 'operate' }, 'operate'); h.history.go(-2); await settle(); }
  h.applied.length = 0;
  const source = h.current(), original = structuredClone(h.entries), offset = forward ? 2 : -2;
  h.dirty(true); h.history.go(offset); await settle();
  assert.equal(h.current(), source); assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  h.cancel(); assert.deepEqual(h.entries, original);
  h.history.go(offset); await settle(); h.discard(); await settle();
  assert.equal(h.current(), forward ? 'operate' : 'build');
  assert.deepEqual(h.entries, original); assert.equal(h.traversals.includes(0), false);
});

test('a caller without location uses bounded confirmation rather than guessing a restoration source', async () => {
  const h = fixture({ withLocation: false }); h.router.record(false, { view: 'review' }, 'review');
  h.dirty(true); h.history.go(-1); await settle();
  assert.equal(h.hasPrompt(), true); assert.deepEqual(h.applied, []);
  assert.deepEqual(h.traversals, [-1]);
  h.cancel(); h.router.record(true, { view: 'review' }, 'review-background');
  assert.equal(h.current(), 'build'); assert.equal(h.entries[0].state.portalIndex, 0);
});

test('a permitted router push after unknown cancellation releases the fence without losing its destination', async () => {
  const h = fixture({ initialEntries: [{ url: 'operate#main-content', state: null }, { url: 'review', state: null }], initialIndex: 1 });
  h.dirty(true); h.history.go(-1); await settle(); h.cancel();
  h.dirty(false); h.router.record(false, { view: 'build' }, 'build');
  assert.equal(h.current(), 'build'); assert.equal(h.entries[0].url, 'operate#main-content');
  assert.equal(h.entries[0].state, null, 'An explicit push must not relabel the fenced destination as its retained source');
  h.router.record(true, { view: 'build' }, 'build-current');
  assert.equal(h.current(), 'build-current');
  assert.deepEqual(h.entries[0], { url: 'operate#main-content', state: null }); assert.deepEqual(h.applied, []);
});

test('returning to an owned entry invalidates the previous unknown-entry permission', async () => {
  const h = fixture({ initialEntries: [{ url: 'operate#main-content', state: null }, { url: 'review', state: null }], initialIndex: 1 });
  h.dirty(true); h.history.go(-1); await settle();
  h.history.go(1); await settle();
  const applied = [...h.applied];
  assert.equal(h.discard({ clearEdits: false }), false);
  assert.deepEqual(h.applied, applied); assert.equal(h.current(), 'review');
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
  const walk = Array(-offset).fill(1);
  assert.deepEqual(h.entries, original); assert.deepEqual(h.traversals, [offset, ...walk]);
  h.discard({ clearEdits: false }); await settle();
  assert.equal(h.current(), target); assert.deepEqual(h.applied, [target]);
  assert.equal(h.hasPrompt(), false); assert.equal(h.promptCount(), 1);
  assert.deepEqual(h.entries, original); assert.deepEqual(h.traversals, [offset, ...walk, offset]);
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
  assert.deepEqual(h.entries, original); assert.deepEqual(h.traversals, [offset, ...Array(-offset).fill(1), offset]);
});

test('replacements preserve index and view/job metadata without adding entries', () => {
  const h = fixture(); h.router.record(false, { view: 'review', jobId: 'job-a' }, 'review');
  h.router.record(true, { view: 'review', jobId: 'job-b' }, 'review-b');
  assert.equal(h.entries.length, 2);
  assert.deepEqual(h.history.state, { view: 'review', jobId: 'job-b', portalIndex: 1 });
});
