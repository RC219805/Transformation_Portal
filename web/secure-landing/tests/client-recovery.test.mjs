import test from "node:test";
import assert from "node:assert/strict";
import { createDeferredClientRecoveryApi, createRecoveryReferenceStore, boundedRecoveryJson, recoveryStateSummary } from "../portal-src/client-recovery-deferred.js";

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
function harness(fetchImpl = async () => { throw new Error("lost"); }, restored = null, hostOptions = {}) {
  const timers = clock(), saved = storage(restored), requests = [], accepted = [], applied = [], states = [];
  let serial = 0;
  const context = { ready: true, scope, intent: "private draft", draftGeneration: 0, features: { jobSubmissionReceipts: true, uploadReceipts: true } };
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
    formDataFactory: () => ({ append() {} }), acceptJob: (data) => { accepted.push(data); return true; }, applyUpload: (data) => applied.push(data),
    uploadState: (patch) => states.push(patch), ...hostOptions,
  });
  return { api, context, timers, saved, requests, accepted, applied, states, calls };
}

test("recovery summary preserves empty defaults, context blocking, latest-job order and settled uploads", () => {
  assert.deepEqual(recoveryStateSummary([]), { outcome: "idle", jobBlocked: false, uploadBlocked: false, uploadPresent: false });
  assert.deepEqual(recoveryStateSummary([], true), { outcome: "paused", jobBlocked: true, uploadBlocked: true, uploadPresent: false });
  const attempts = [{ type: "job", status: "unknown" }, { type: "upload", status: "ready" }, { type: "job", status: "accepted" }];
  const before = structuredClone(attempts);
  assert.deepEqual(recoveryStateSummary(attempts), { outcome: "accepted", jobBlocked: true, uploadBlocked: false, uploadPresent: true });
  assert.deepEqual(attempts, before);
});

for (const status of ["accepted", "ready", "expired", "rejected", "posting", "pending", "unknown", "paused"]) {
  test(`recovery summary retains both request blocking boundaries for ${status}`, () => {
    const settled = ["accepted", "ready", "expired", "rejected"].includes(status);
    assert.deepEqual(recoveryStateSummary([{ type: "job", status }, { type: "upload", status }]),
      { outcome: status, jobBlocked: !settled, uploadBlocked: !settled, uploadPresent: true });
  });
}

test("onState adds the derived view without changing the existing snapshot or context arguments", () => {
  const calls = [];
  const h = harness(undefined, null, { onState: (...args) => calls.push(args) });
  const [snapshot, needsContext, view] = calls.at(-1);
  assert.deepEqual(snapshot, h.api.snapshot()); assert.equal(needsContext, false);
  assert.deepEqual(view, recoveryStateSummary(snapshot, needsContext));
});

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

for (const status of [400, 404, 413, 422]) {
  test(`confirmed job rejection ${status} does not become an unresolved attempt after reload`, async () => {
    const h = harness(async () => json({ success: false, error: { message: "Invalid request" } }, status));
    await h.api.submit({ pipeline: "lux-depth-v3", args: {} });
    assert.equal(h.api.snapshot()[0].status, "rejected");
    assert.deepEqual(JSON.parse(h.saved.raw()).attempts, []);
    const reloaded = harness(async () => json({ success: true, schema: "tp.orchestrator.job.v1", data: job }), h.saved.raw());
    assert.deepEqual(reloaded.api.snapshot(), []);
    assert.deepEqual(await reloaded.api.submit({ pipeline: "lux-depth-v3", args: {} }), { ...job, pipeline: "lux-depth-v3" });
    assert.equal(reloaded.calls.length, 1);
    assert.equal(reloaded.calls[0][1].method, "POST");
  });

  test(`confirmed upload rejection ${status} releases only its own opaque reference before reload`, async () => {
    const h = harness();
    void h.api.submit({ pipeline: "lux-depth-v3", args: {} });
    const uploadId = h.api.upload(selection);
    h.requests[0].status = status;
    h.requests[0].response = { success: false, error: { message: "Invalid upload" } };
    h.requests[0].onload();
    assert.equal(h.api.snapshot().find((row) => row.request_id === uploadId).status, "rejected");
    const retained = JSON.parse(h.saved.raw()).attempts;
    assert.equal(retained.length, 1);
    assert.equal(retained[0].type, "job");
    assert.deepEqual(Object.keys(retained[0]).sort(), ["created_at", "request_id", "scope", "type"]);
    const reloaded = harness(undefined, h.saved.raw());
    assert.equal(reloaded.api.snapshot().length, 1);
    assert.equal(reloaded.api.snapshot()[0].type, "job");
    reloaded.api.upload(selection);
    assert.equal(reloaded.requests.length, 1);
  });
}

for (const status of [401, 403, 408, 409, 410, 429, 503]) {
  test(`job response ${status} retains its recovery reference across reload`, async () => {
    const h = harness(async () => json({ success: false }, status));
    await h.api.submit({});
    assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
    const reloaded = harness(undefined, h.saved.raw());
    await assert.rejects(reloaded.api.submit({}, { separate: true }), /unresolved/);
    assert.equal(reloaded.calls.length, 0);
  });
}

for (const status of [401, 403, 408, 409, 410, 429, 503]) {
  test(`upload response ${status} retains its recovery reference across reload`, () => {
    const h = harness();
    h.api.upload(selection);
    h.requests[0].status = status;
    h.requests[0].response = { success: false };
    h.requests[0].onload();
    assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
    const reloaded = harness(undefined, h.saved.raw());
    assert.throws(() => reloaded.api.upload(selection), /unresolved/);
    assert.equal(reloaded.requests.length, 0);
  });
}

test("failed rejection-reference removal retains recovery and blocks another durable POST", async () => {
  let raw = null, writes = 0;
  const saved = { getItem: () => raw, setItem(_, value) { if (++writes > 1) throw new Error("Storage denied"); raw = value; } };
  const h = harness(async () => json({ success: false }, 422), null, { sessionStorage: saved });
  await h.api.submit({});
  assert.equal(writes, 2);
  assert.equal(JSON.parse(raw).attempts.length, 1);
  assert.equal(h.api.snapshot()[0].status, "rejected");
  await assert.rejects(h.api.submit({}), /storage/);
  assert.equal(h.calls.length, 1);
  const reloaded = harness(undefined, raw);
  await assert.rejects(reloaded.api.submit({}), /unresolved/);
});

test("malformed rejection bodies retain an unresolved reference across reload", async () => {
  const h = harness(async () => new Response("invalid JSON", { status: 400 }));
  await h.api.submit({});
  assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
  const reloaded = harness(undefined, h.saved.raw());
  await assert.rejects(reloaded.api.submit({}, { separate: true }), /unresolved/);
  assert.equal(reloaded.calls.length, 0);
});

test("unapplied accepted runs and later expired tombstones retain explicit recovery across reload", async () => {
  const h = harness(async (_, options) => options.method === "POST"
    ? json({ success: true, schema: "tp.orchestrator.job.v1", data: job })
    : json({ success: true, schema: "tp.orchestrator.job_submission.v1", data: { request_id: id(2), status: "expired" } }),
  null, { acceptJob: () => false });
  await h.api.submit({});
  assert.equal(h.api.snapshot()[0].status, "accepted");
  assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
  await h.api.check(id(2));
  assert.equal(h.api.snapshot()[0].status, "expired");
  assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
  const reloaded = harness(async () => json({ success: true, schema: "tp.orchestrator.job_submission.v1", data: { request_id: id(2), status: "expired" } }), h.saved.raw());
  assert.equal(reloaded.api.snapshot()[0].status, "unknown");
  await reloaded.api.check(id(2));
  assert.equal(reloaded.api.snapshot()[0].status, "expired");
  assert.equal(reloaded.calls.length, 1);
  assert.equal(reloaded.calls[0][1].method, "GET");
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
  assert.deepEqual(h.accepted, [{ ...job, pipeline: "lux-depth-v3" }]); assert.deepEqual(h.api.snapshot(), []);
  assert.deepEqual(JSON.parse(h.saved.raw()).attempts, []);
});

for (const type of ['job', 'upload']) {
  test(`restored ${type} digest references block both POST types until backend ownership is verified`, async () => {
    const context = { ready: true, scope: '["managed","alice@example.test","alice"]', intent: 'private draft', draftGeneration: 0,
      features: { jobSubmissionReceipts: false, uploadReceipts: false } };
    const required = [];
    const raw = JSON.stringify({ schema: 'tp.portal.recovery_refs.v1', tab_id: id(1),
      attempts: [{ type, request_id: id(2), scope, created_at: 0 }] });
    const h = harness(async () => json(type === 'job'
      ? { schema: 'tp.orchestrator.job_submission.v1', success: true, data: { request_id: id(2), status: 'accepted', job } }
      : { schema: 'tp.orchestrator.upload_receipt.v1', success: true, data: { request_id: id(2), status: 'ready', receipt: upload } }), raw,
    { context: () => context, onState: (_state, needsContext) => required.push(needsContext) });
    assert.equal(required.at(-1), true);
    await assert.rejects(h.api.submit({}), /Confirm access/);
    assert.throws(() => h.api.upload(selection), /Confirm access/);
    await h.api.check(id(2));
    assert.equal(h.calls.length, 0); assert.equal(h.requests.length, 0);
    assert.equal(h.saved.raw(), raw, 'Unverifiable actor names cannot clear or rewrite another scoped reference');
    context.scope = scope; h.api.sync();
    assert.equal(required.at(-1), false);
    assert.equal(h.api.snapshot()[0].restored, true);
    assert.equal(h.api.snapshot()[0].status, 'unknown');
    await h.api.check(id(2));
    assert.equal(h.accepted.length, 0); assert.equal(h.applied.length, 0);
    assert.equal(h.api.accept(id(2)), true);
    assert.deepEqual(h.calls.map(([, options]) => options.method), ['GET']);
    assert.deepEqual(JSON.parse(h.saved.raw()).attempts, []);
  });
}

test('legacy sessions without durable references retain their existing keyless request path', async () => {
  const context = { ready: true, scope: '["managed","alice@example.test","alice"]', intent: 'legacy draft', draftGeneration: 0, features: {} };
  const required = [];
  const h = harness(async () => json({ schema: 'tp.orchestrator.job.v1', success: true, data: job }), null,
    { context: () => context, onState: (_state, needsContext) => required.push(needsContext) });
  await h.api.submit({ pipeline: 'archive-gate-a' });
  assert.equal(required.at(-1), false);
  assert.equal(h.calls[0][1].headers['Idempotency-Key'], undefined);
  assert.equal(h.accepted[0].pipeline, 'archive-gate-a');
  h.api.upload(selection);
  assert.equal(h.requests[0].headers['X-TP-Upload-Request-ID'], undefined);
  Object.assign(h.requests[0], { status: 200, response: { schema: 'tp.orchestrator.upload_staging.v1', success: true, data: upload } });
  h.requests[0].onload();
  assert.deepEqual(h.applied, [upload]);
});

for (const reconciled of [false, true]) {
  test(`${reconciled ? 'reconciled' : 'normal'} accepted jobs use the captured submitted pipeline after draft changes`, async () => {
    let complete;
    const h = harness((_url, options) => {
      if (reconciled && options.method === 'POST') throw new Error('lost after commit');
      return new Promise(resolve => { complete = resolve; });
    });
    const payload = { pipeline: 'archive-gate-a', args: { input_dir: '/source' } };
    const posting = h.api.submit(payload);
    await flush();
    payload.pipeline = 'archive-gate-b'; h.context.intent = 'new workflow draft'; h.context.draftGeneration += 1;
    complete(json(reconciled
      ? { schema: 'tp.orchestrator.job_submission.v1', success: true, data: { request_id: id(2), status: 'accepted', job } }
      : { schema: 'tp.orchestrator.job.v1', success: true, data: job }));
    await posting; await flush();
    assert.equal(h.accepted.length, 0);
    assert.equal(h.api.snapshot()[0].result.pipeline, 'archive-gate-a');
    assert.equal(h.api.accept(id(2)), true);
    assert.equal(h.accepted[0].pipeline, 'archive-gate-a');
    assert.equal(JSON.parse(h.calls[0][1].body).pipeline, 'archive-gate-a');
    assert.doesNotMatch(h.saved.raw(), /archive-gate|source|workflow/);
    assert.equal(h.calls.filter(([, options]) => options.method === 'POST').length, 1);
  });
}

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

test("a stale job POST failure cannot suppress the next actor's requests", async () => {
  let complete, blocked = false, inspected = 0;
  const h = harness(() => h.calls.length === 1 ? new Promise(resolve => { complete = resolve; })
    : json({ schema: "tp.orchestrator.job.v1", success: true, data: job }), null, {
    canRequest: () => !blocked,
    onProtectedFailure: () => { inspected += 1; blocked = true; }
  });
  const posting = h.api.submit({ actor: "a" });
  h.context.scope = "b".repeat(64); h.api.sync();
  complete(json({ error: { details: { retryable: false } } }, 503));
  await posting;
  assert.equal(inspected, 0); assert.equal(blocked, false);
  assert.deepEqual(await h.api.submit({ actor: "b" }), job);
  assert.equal(h.calls.filter(([, options]) => options.method === "POST").length, 2);
  assert.deepEqual(h.accepted, [job]);
});

test("a job POST failure from before an access outage cannot suppress recovered access", async () => {
  let complete, inspected = 0;
  const h = harness(() => new Promise(resolve => { complete = resolve; }), null, {
    onProtectedFailure: () => { inspected += 1; }
  });
  const posting = h.api.submit({});
  h.context.ready = false; h.api.sync();
  h.context.ready = true; h.api.sync();
  complete(json({ error: { details: { retryable: false } } }, 503));
  await posting;
  assert.equal(inspected, 0);
  assert.equal(h.api.snapshot()[0].status, "unknown");
  assert.equal(h.calls.length, 1);
});

for (const outcome of [false, undefined, Promise.resolve(true)]) {
  test(`accepted job receipt survives a host result of ${outcome instanceof Promise ? "Promise" : outcome}`, async () => {
    let apply = false;
    const modes = [], h = harness(async () => json({ schema: "tp.orchestrator.job.v1", success: true, data: job }), null, {
      acceptJob: (_data, options) => { modes.push(options.automatic); return apply ? true : outcome; }
    });
    await h.api.submit({});
    assert.equal(h.api.snapshot()[0].status, "accepted");
    assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
    assert.equal(h.api.accept(id(2)), false);
    assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
    apply = true;
    assert.equal(h.api.accept(id(2)), true);
    assert.deepEqual(h.api.snapshot(), []);
    assert.deepEqual(JSON.parse(h.saved.raw()).attempts, []);
    assert.deepEqual(modes, [true, false, false]);
  });
}

for (const reconciled of [false, true]) {
  test(`job ${reconciled ? "GET reconciliation" : "POST success"} preserves its receipt after a same-intent draft ABA`, async () => {
    let complete;
    const h = harness((_url, options) => {
      if (reconciled && options.method === "POST") throw new Error("lost after commit");
      return new Promise(resolve => { complete = resolve; });
    });
    const posting = h.api.submit({});
    await flush();
    h.context.draftGeneration += 2;
    complete(json(reconciled ? { schema: "tp.orchestrator.job_submission.v1", success: true,
      data: { request_id: id(2), status: "accepted", job } } : { schema: "tp.orchestrator.job.v1", success: true, data: job }));
    await posting; await flush();
    assert.equal(h.accepted.length, 0);
    assert.equal(h.api.snapshot()[0].status, "accepted");
    assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
    assert.equal(h.api.accept(id(2)), true);
    assert.deepEqual(h.accepted, [job]);
    assert.equal(h.calls.filter(([, options]) => options.method === "POST").length, 1);
  });
}

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

test('upload edit-undo ABA retains its ready receipt until explicit input acceptance', () => {
  const h = harness();
  const requestId = h.api.upload(selection), original = h.context.intent;
  h.context.intent = 'temporary edit'; h.context.draftGeneration += 1; h.api.sync();
  h.context.intent = original; h.context.draftGeneration += 1; h.api.sync();
  Object.assign(h.requests[0], { status: 200, response: { schema: 'tp.orchestrator.upload_staging.v1', success: true, data: upload } });
  h.requests[0].onload();
  assert.equal(h.applied.length, 0);
  assert.equal(h.api.snapshot()[0].status, 'ready');
  assert.deepEqual(h.api.snapshot()[0].result, upload);
  assert.equal(h.states.at(-1).status, 'recovery_ready');
  assert.equal(JSON.parse(h.saved.raw()).attempts.length, 1);
  assert.equal(h.api.accept(requestId), true);
  assert.deepEqual(h.applied, [upload]);
  assert.deepEqual(JSON.parse(h.saved.raw()).attempts, []);
  assert.equal(h.requests.length, 1); assert.equal(h.calls.length, 0);
});


test("nonretryable frontdoor configuration failures suppress receipt polling and new requests", async () => {
  let blocked = false;
  const h = harness(async () => json({ error: { details: { retryable: false } } }, 503), null, {
    canRequest: () => !blocked,
    onProtectedFailure: (_type, envelope) => { blocked = envelope.error.details.retryable === false; }
  });
  await h.api.submit({ pipeline: "archive-gate-a" }); await flush();
  assert.equal(h.calls.length, 1); assert.equal(h.api.snapshot()[0].status, "unknown");
  await h.timers.tick(60000); await h.api.check(h.api.snapshot()[0].request_id);
  assert.equal(h.calls.length, 1);
  await assert.rejects(h.api.submit({}));
  assert.equal(h.calls.length, 1);
  assert.throws(() => h.api.upload(selection), /configuration requires repair/);
});

test("Retry-After is honored by automatic and manual reconciliation without another POST", async () => {
  const h = harness(async (url, options) => options.method === "POST"
    ? new Response(JSON.stringify({ error: { code: "RATE_LIMITED" } }), { status: 429, headers: { "Retry-After": "60" } })
    : json({ schema: "tp.orchestrator.job_submission.v1", success: true, data: { request_id: url.split("/").at(-1), status: "accepted", job } }),
    null, { retryDelay: () => 60000 });
  await h.api.submit({}); await flush();
  const attempt = h.api.snapshot()[0];
  await h.timers.tick(59000); await h.api.check(attempt.request_id);
  assert.equal(h.calls.length, 1);
  await h.timers.tick(1000); await h.api.check(attempt.request_id);
  assert.deepEqual(h.calls.map(([, options]) => options.method), ["POST", "GET"]);
  assert.deepEqual(h.accepted, [job]);
});

test("upload configuration failures never cause repeated receipt reads", async () => {
  let blocked = false;
  const h = harness(undefined, null, { canRequest: () => !blocked,
    onProtectedFailure: (_type, envelope) => { blocked = envelope?.error?.details?.retryable === false; } });
  h.api.upload(selection);
  Object.assign(h.requests[0], { status: 503, response: { error: { details: { retryable: false } } } });
  h.requests[0].onload(); await flush(); await h.timers.tick(60000);
  assert.equal(h.requests.length, 1); assert.equal(h.calls.length, 0);
});
