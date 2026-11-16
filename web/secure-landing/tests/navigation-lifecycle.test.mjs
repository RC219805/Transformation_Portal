import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('cancelled unload warns without stopping streams, health or bootstrap recovery', () => {
  const source = readFileSync(new URL('../portal-src/portal.template.js', import.meta.url), 'utf8');
  const start = source.indexOf("window.addEventListener('beforeunload', (event)");
  const end = source.indexOf("window.addEventListener('pageshow'", start);
  assert.ok(start >= 0 && end > start);
  const listeners = new Map(); let cleanup = 0;
  new Function('window', 'deferredReviewSurfaceApi', '_flushPendingTransientPortalDraftPersist', 'cleanupActiveJobHandles', '_flushPortalRumOnPagehide', source.slice(start, end))(
    { addEventListener: (name, action) => listeners.set(name, [...(listeners.get(name) || []), action]) },
    { hasUnsavedChanges: () => true }, () => {}, () => { cleanup += 1; }, () => {},
  );
  const event = { prevented: false, preventDefault() { this.prevented = true; } };
  for (const action of listeners.get('beforeunload')) action(event);
  assert.equal(event.prevented, true); assert.equal(event.returnValue, '');
  assert.equal(cleanup, 0, 'choosing to stay retains all active services');
  for (const action of listeners.get('pagehide')) action({});
  assert.equal(cleanup, 1, 'actual page exit releases the services');
});
