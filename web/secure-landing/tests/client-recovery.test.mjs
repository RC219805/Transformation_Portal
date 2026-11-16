import test from "node:test";
import assert from "node:assert/strict";
import { createDeferredClientRecoveryApi, createRecoveryReferenceStore, boundedRecoveryJson } from "../portal-src/client-recovery-deferred.js";

const scope = "a".repeat(64);
const id = (n) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const flush = async () => { for (let n = 0; n < 25; n += 1) await Promise.resolve(); };
function clock() {
  let now = 0, next = 0;
  const tasks = new Map();
  return { now: () => now, tasks,
    setTimeout(fn, ms) { const key = ++next; tasks.set(key, { fn, at: now + ms }); return key; },
    clearTimeout(key) { tasks.delete(key); },
    async tick(ms) {
      const end = now + ms;
      for (;;) {
        await flush();
        const task = [...tasks].sort((a, b) => a[1].at - b[1].at)[0];
        if (!task || task[1].at > end) break;
        now = task[1].at; tasks.delete(task[0]); task[1].fn();
      }
      now = end; await flush();
    }
  };
}
function storage(initial = null) {
  let raw = initial;
  return { getItem: () => raw, setItem: (_, value) => { raw = value; }, raw: () => raw };
}
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
const job = { id: "job_accepted", state: "queued", events_url: "/v1/jobs/job_accepted/events" };
const upload = { batch_id: "batch-one", input_dir: "/authorized/staged", metadata_dir: null,
  summary: { file_count: 1, total_bytes: 4 } };
const selection = { entries: [{ file: {}, relativePath: "image.jpg", name: "image.jpg", sizeBytes: 4 }], totalBytes: 4 };
function harness(fetchImpl = async () => { throw new Error("lost"); }, restored = null) {
  const timers = clock(), saved = storage(restored), requests = [], accepted = [], applied = [], states = [];
  let serial = 0;
  const context = { ready: true, scope, intent: "private draft", features: { jobSubmissionReceipts: true, uploadReceipts: true } };
  class Xhr {
    constructor() { this.upload = {}; this.headers = {}; requests.push(this); }
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(key, value) { this.headers[key] = value; }
    send(form) { this.form = form; }
    abort() { this.aborted = true; this.onabort?.(); }
  }
  const calls = [];
  const api = createDeferredClientRecoveryApi({ context: () => context, sessionStorage: saved,
    uuid: () => id(++serial), timers, now: timers.now, headers: (extra) => ({ ...extra, "X-CSRF-Token": "private token" }),
    fetchImpl: (...args) => { calls.push(args); return fetchImpl(...args); }, xhrFactory: () => new Xhr(),
    formDataFactory: () => ({ append() {} }), acceptJob: (data) => accepted.push(data), applyUpload: (data) => applied.push(data),
    uploadState: (patch) => states.push(patch),
  });
  return { api, context, timers, saved, requests, accepted, applied, states, calls };
}

test("tab references retain only bounded opaque actor-scoped identities", async () => {
  const h = harness(() => new Promise(() => {}));
  void h.api.submit({ args: { input_dir: "/secret", api_key: "never-store" } });
  const raw = h.saved.raw();
  assert.doesNotMatch(raw, /secret|token|never-store|draft|input_dir|api_key/);
  const parsed = JSON.parse(raw);
  assert.deepEqual(Object.keys(parsed.attempts[0]).sort(), ["created_at", "request_id", "scope", "type"]);
  assert.equal(parsed.attempts[0].scope, scope);
  assert.equal(h.calls[0][1].headers["Idempotency-Key"], id(2));
  await h.timers.tick(30000);
  h.context.ready = false; h.api.sync(); await h.timers.tick(5000);
});

test("storage denial or malformed persisted references prevents a durable POST", async () => {
  for (const raw of ['{"schema":"wrong"}', JSON.stringify({ schema: "tp.portal.recovery_refs.v1", tab_id: id(1), attempts: [{ type: "job", request_id: id(2), scope, created_at: 0, payload: "private" }] })]) {
    const h = harness(undefined, raw);
    await assert.rejects(h.api.submit({}), /storage/);
    assert.equal(h.calls.length, 0);
  }
  const denied = createRecoveryReferenceStore({ getItem() { throw new Error("denied"); } });
  assert.equal(denied.available(), false);
});

test("the 30 second deadline includes a stalled response body and rejects oversized bodies", async () => {
  const timers = clock(); let signal;
  const pending = boundedRecoveryJson(async (_, options) => { signal = options.signal; return new Response(new ReadableStream({ start() {} })); }, "/post", {}, 30000, timers);
  await timers.tick(29999); assert.equal(signal.aborted, false);
  const rejected = assert.rejects(pending, { name: "TimeoutError" });
  await timers.tick(1); await rejected; assert.equal(signal.aborted, true);
  await assert.rejects(boundedRecoveryJson(async () => json({ value: "x".repeat(262144) }), "/read"), /bound/);
});

test("lost job response reconciles with GET only and attaches exactly one accepted run", async () => {
  const h = harness(async (url, options) => {
    if (options.method === "POST") throw new Error("response lost after commit");
    const requestId = url.split("/").at(-1);
    return json({ schema: "tp.orchestrator.job_submission.v1", success: true, data: { request_id: requestId, status: "accepted", job } });
  });
  assert.equal(await h.api.submit({ pipeline: "lux-depth-v3" }), null);
  await flush();
  assert.deepEqual(h.calls.map(([, options]) => options.method), ["POST", "GET"]);
  assert.deepEqual(h.accepted, [job]); assert.deepEqual(h.api.snapshot(), []);
  assert.deepEqual(JSON.parse(h.saved.raw()).attempts, []);
});

test("malformed success never invents a queued/failed run; unchanged and restored attempts cannot repost", async () => {
  const h = harness(async (_, options) => options.method === "POST" ? json({ success: true, data: { id: "unverified" } }) : json({ success: false }, 404));
  await h.api.submit({}); await flush();
  assert.equal(h.accepted.length, 0); assert.equal(h.api.snapshot()[0].status, "unknown");
  await assert.rejects(h.api.submit({}), /unresolved/);
  await h.timers.tick(60000);
  assert.equal(h.calls.filter(([, options]) => options.method === "POST").length, 1);
  const count = h.calls.length; await h.timers.tick(60000); assert.equal(h.calls.length, count);
  await h.api.check(id(2)); assert.equal(h.calls.length, count + 1);
  const restored = harness(undefined, h.saved.raw());
  await assert.rejects(restored.api.submit({}, { separate: true }), /unresolved/);
});

test("a changed draft needs explicit separate intent while preserving earlier unknown", async () => {
  const h = harness(async (_, options) => { if (options.method === "POST") throw new Error("lost"); return json({ success: false }, 404); });
  await h.api.submit({ old: true }); h.context.intent = "changed";
  await assert.rejects(h.api.submit({ new: true }), /unresolved/);
  await h.api.submit({ new: true }, { separate: true });
  assert.equal(h.calls.filter(([, options]) => options.method === "POST").length, 2);
  assert.equal(h.api.snapshot().length, 2);
  h.context.ready = false; h.api.sync(); await h.timers.tick(5000);
});

test("access loss and actor change mask results and prohibit stale response attachment", async () => {
  let complete;
  const h = harness(() => new Promise((resolve) => { complete = resolve; }));
  const posting = h.api.submit({});
  h.context.ready = false; h.api.sync();
  complete(json({ schema: "tp.orchestrator.job.v1", success: true, data: job }));
  await posting; assert.equal(h.accepted.length, 0);
  assert.equal(h.api.snapshot()[0].result, null);
  h.context.scope = "b".repeat(64); h.context.ready = true; h.api.sync();
  assert.deepEqual(h.api.snapshot(), []);
  h.context.scope = scope; h.api.sync();
  await assert.rejects(h.api.submit({}, { separate: true }), /unresolved/);
});

test("upload progress is transport-only, with separate 90 second idle and 120 second staging limits", async () => {
  const h = harness(async () => json({ success: false }, 404));
  h.api.upload(selection); const xhr = h.requests[0];
  assert.equal(xhr.headers["X-TP-Upload-Request-ID"], id(2));
  await h.timers.tick(89000); assert.equal(xhr.aborted, undefined);
  xhr.upload.onprogress({ lengthComputable: true, loaded: 4, total: 4 });
  assert.equal(h.applied.length, 0);
  xhr.upload.onload(); assert.equal(h.states.at(-1).status, "staging");
  await h.timers.tick(119999); assert.equal(xhr.aborted, undefined);
  await h.timers.tick(1); assert.equal(xhr.aborted, true);
  assert.equal(h.api.snapshot()[0].status, "unknown"); assert.equal(h.applied.length, 0);
  h.context.ready = false; h.api.sync(); await h.timers.tick(5000);
});

test("upload idle timeout aborts transport and never starts a second upload automatically", async () => {
  const h = harness(async () => json({ success: false }, 404));
  h.api.upload(selection); await h.timers.tick(90000);
  assert.equal(h.requests[0].aborted, true);
  assert.throws(() => h.api.upload(selection), /unresolved/);
  assert.equal(h.requests.length, 1); assert.equal(h.applied.length, 0);
  h.context.ready = false; h.api.sync(); await h.timers.tick(5000);
});

test("cancel reconciles late committed upload but applies paths only through explicit acceptance", async () => {
  const h = harness(async (url) => json({ schema: "tp.orchestrator.upload_receipt.v1", success: true,
    data: { request_id: url.split("/").at(-1), status: "ready", receipt: upload } }));
  const requestId = h.api.upload(selection); h.api.cancelUpload(); await flush();
  const xhr = h.requests[0]; assert.equal(xhr.aborted, true);
  Object.assign(xhr, { status: 200, response: { schema: "tp.orchestrator.upload_staging.v1", success: true, data: upload } }); xhr.onload();
  assert.equal(h.applied.length, 0); assert.equal(h.api.snapshot()[0].status, "ready");
  assert.ok(h.api.accept(requestId)); assert.deepEqual(h.applied, [upload]);
  assert.equal(h.requests.length, 1); assert.equal(h.calls[0][1].method, "GET");
});

test("changed upload context cannot auto-apply late success, and auth loss aborts without exposing paths", async () => {
  const h = harness(async (url) => json({ schema: "tp.orchestrator.upload_receipt.v1", success: true,
    data: { request_id: url.split("/").at(-1), status: "ready", receipt: upload } }));
  const requestId = h.api.upload(selection); h.context.intent = "edited while transferring"; h.api.sync();
  const xhr = h.requests[0]; Object.assign(xhr, { status: 200, response: { schema: "tp.orchestrator.upload_staging.v1", success: true, data: upload } }); xhr.onload();
  assert.equal(h.applied.length, 0); assert.equal(h.api.snapshot()[0].status, "ready");
  h.context.ready = false; h.api.sync(); assert.equal(h.api.snapshot()[0].result, null); assert.equal(h.api.accept(requestId), false);
  h.context.ready = true; h.api.sync(); assert.equal(h.api.accept(requestId), false);
  await h.api.check(requestId); assert.equal(h.api.accept(requestId), true);
  h.api.upload(selection); h.context.ready = false; h.api.sync();
  assert.equal(h.requests[1].aborted, true); assert.equal(h.applied.length, 1);
});
