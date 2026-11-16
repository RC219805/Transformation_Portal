import test from 'node:test';
import assert from 'node:assert/strict';
import { installThumbnailFetchOwnership } from './e2e/helpers/thumbnail-fetch-ownership.mjs';

const thumbnail = '/v1/jobs/example/photographic-review/thumbnail?scene_id=room';
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
function fixture(fetch) {
  const target = { fetch, location: { href: 'http://localhost/portal' } };
  installThumbnailFetchOwnership(target);
  return { target, metrics: target.__tpThumbnailFetchOwnership };
}

test('ownership spans headers and every body chunk without changing the response', async () => {
  let source;
  const response = new Response(new ReadableStream({ start(controller) { source = controller; } }), { headers: { 'X-Evidence': 'unchanged' } });
  const { target, metrics } = fixture(async () => response);
  const received = await target.fetch(thumbnail);
  assert.equal(received, response); assert.equal(received.headers.get('X-Evidence'), 'unchanged');
  assert.equal(metrics.active, 1);
  const reader = received.body.getReader(); source.enqueue(new Uint8Array([1, 2]));
  assert.deepEqual((await reader.read()).value, new Uint8Array([1, 2])); assert.equal(metrics.active, 1);
  source.close(); assert.equal((await reader.read()).done, true);
  assert.equal(metrics.active, 0); assert.equal(metrics.peak, 1);
});

test('a third genuinely owned fetch is detected before any body completes', async () => {
  const { target, metrics } = fixture(async () => new Response('data'));
  const responses = await Promise.all([target.fetch(thumbnail), target.fetch(thumbnail), target.fetch(thumbnail)]);
  assert.equal(metrics.started, 3); assert.equal(metrics.active, 3); assert.equal(metrics.peak, 3);
  for (const response of responses) await response.body.getReader().cancel();
  assert.equal(metrics.active, 0);
});

test('abort ends ownership once even when the fetch rejection arrives later', async () => {
  const pending = deferred(), controller = new AbortController();
  const { target, metrics } = fixture(() => pending.promise);
  const result = target.fetch(thumbnail, { signal: controller.signal });
  assert.equal(metrics.active, 1); controller.abort(); assert.equal(metrics.active, 0);
  pending.reject(new Error('late transport terminal'));
  await assert.rejects(result, /late transport/);
  assert.equal(metrics.active, 0); assert.equal(metrics.events.length, 2);
});

test('a 503 abort releases ownership before an independent browser terminal notification', async () => {
  const { target, metrics } = fixture(async () => new Response('unavailable', { status: 503 }));
  const controllers = [new AbortController(), new AbortController(), new AbortController()];
  const failed = await target.fetch(thumbnail, { signal: controllers[0].signal });
  await target.fetch(thumbnail, { signal: controllers[1].signal });
  assert.equal(failed.status, 503); assert.equal(metrics.active, 2);
  controllers[0].abort();
  await target.fetch(thumbnail, { signal: controllers[2].signal });
  assert.equal(metrics.active, 2); assert.equal(metrics.peak, 2);
  controllers[1].abort(); controllers[2].abort(); assert.equal(metrics.active, 0);
});

test('reader cancellation retains ownership until cancellation settles', async () => {
  const drain = deferred();
  const { target, metrics } = fixture(async () => new Response(new ReadableStream({ cancel: () => drain.promise })));
  const response = await target.fetch(thumbnail), cancellation = response.body.getReader().cancel();
  assert.equal(metrics.active, 1); drain.resolve(); await cancellation; assert.equal(metrics.active, 0);
});

test('unrelated fetches remain untouched and uncounted', async () => {
  const response = new Response('ok');
  const { target, metrics } = fixture(async () => response);
  assert.equal(await target.fetch('/healthz'), response); assert.equal(metrics.started, 0); assert.equal(metrics.active, 0);
});
