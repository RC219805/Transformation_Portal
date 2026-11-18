import test from "node:test";
import assert from "node:assert/strict";
import { createPhotographicThumbnails, photographicThumbnailEvidence } from "../portal-src/photographic-thumbnails.js";

const sha = "a".repeat(64), catalogue = "c".repeat(64);
const expected = { sceneId: "room one", catalogue, artifactPath: "room one/delivery.tif", sha256: sha, sourceIdentity: sha, policy: null, view: "clip_srgb" };
function png(size = 33) {
  const bytes = new Uint8Array(size); bytes.set([137, 80, 78, 71, 13, 10, 26, 10]);
  const data = new DataView(bytes.buffer); data.setUint32(8, 13); bytes.set([73, 72, 68, 82], 12); data.setUint32(16, 256); data.setUint32(20, 128); return bytes;
}
function response(proof = expected, overrides = {}, bytes = png()) {
  return new Response(bytes, { headers: { "Content-Type": "image/png", "X-TP-Catalogue-Identity-SHA256": proof.catalogue,
    "X-TP-Scene-ID": encodeURIComponent(proof.sceneId), "X-TP-Artifact-Path": encodeURIComponent(proof.artifactPath),
    "X-TP-Source-SHA256": sha, "X-TP-Source-Identity-SHA256": sha, "X-TP-Presentation-Policy-SHA256": "none",
    "X-TP-Presentation-View": "clip_srgb", "X-TP-Display-Transform": "linear_srgb_to_8bit_srgb_clip_v1",
    "X-TP-Thumbnail-Max-Edge": "256", "X-TP-Thumbnail-Max-Bytes": "1048576", "X-TP-Thumbnail-Renderer": "tp.photo.thumbnail.v1",
    "X-TP-Thumbnail-Width": "256", "X-TP-Thumbnail-Height": "128", "X-TP-Native-Width": "8000", "X-TP-Native-Height": "4000", ...overrides } });
}
class Element {
  constructor(tag) { this.tag = tag; this.dataset = {}; this.style = {}; this.children = []; }
  prepend(...nodes) { this.children.unshift(...nodes); }
}
const document = { createElement: (tag) => new Element(tag) };
const settle = async () => { for (let n = 0; n < 50; n += 1) await new Promise((resolve) => setImmediate(resolve)); };
function page(start, count = 8) {
  const scenes = Array.from({ length: count }, (_, n) => ({ scene_id: `room-${start + n}`, label: `Room ${start + n}`,
    thumbnail: { url: "/v1/jobs/job-one/photographic-review/thumbnail", artifact_path: `room-${start + n}/delivery.tif`, sha256: sha, source_identity_sha256: sha, presentation_policy_sha256: null } }));
  return { catalogue_identity_sha256: catalogue, scenes };
}
const cards = (snapshot) => snapshot.scenes.map((scene) => ({ scene, control: new Element("button") }));
const indexedPage = (start, count = 8) => {
  const snapshot = page(start, count);
  snapshot.scenes.forEach((scene) => { scene.thumbnail.presentation_policy_sha256 = "b".repeat(64); });
  return snapshot;
};
function harness(fetchImpl, options = {}) {
  const context = { ready: true, enabled: true, owner: "verified-owner", jobId: "job-one", view: "clip_srgb" };
  const created = [], revoked = [], calls = [];
  const api = createPhotographicThumbnails({ document, context: () => context, origin: "http://localhost", headers: () => ({ "X-CSRF-Token": "never-on-disk" }),
    fetchImpl: (...args) => { calls.push(args); return fetchImpl(...args); },
    urls: { createObjectURL: (blob) => { const url = `blob:fixture-${created.length}`; created.push({ url, blob }); return url; }, revokeObjectURL: (url) => revoked.push(url) }, ...options });
  return { api, context, calls, created, revoked };
}
function proof(url) {
  const params = new URL(url, "http://localhost").searchParams;
  return { ...expected, sceneId: params.get("scene_id"), artifactPath: `${params.get("scene_id")}/delivery.tif`, catalogue: params.get("catalogue_identity_sha256") };
}

test("thumbnail proof binds encoded scene/path, source identities, transform, catalogue and exact PNG dimensions", () => {
  assert.equal(photographicThumbnailEvidence(response(), png(), expected).width, 256);
  for (const change of [
    { "X-TP-Catalogue-Identity-SHA256": "d".repeat(64) }, { "X-TP-Scene-ID": "different" }, { "X-TP-Artifact-Path": "wrong.tif" },
    { "X-TP-Source-SHA256": "d".repeat(64) }, { "X-TP-Source-Identity-SHA256": "d".repeat(64) },
    { "X-TP-Presentation-View": "recorded" }, { "X-TP-Display-Transform": "wrong" }, { "X-TP-Thumbnail-Width": "257" },
    { "X-TP-Thumbnail-Width": "128" }, { "X-TP-Thumbnail-Max-Bytes": "2097152" }, { "X-TP-Native-Width": "0" }, { "Content-Type": "image/jpeg" },
  ]) assert.throws(() => photographicThumbnailEvidence(response(expected, change), png(), expected), /verification|dimensions/);
  assert.throws(() => photographicThumbnailEvidence(response(), png(1048577), expected), /verification/);
  assert.throws(() => photographicThumbnailEvidence(response(), new Uint8Array(33), expected), /PNG/);
});

test("unknown metadata identities require verified private response digests and never become indexed hints", () => {
  const unknown = { ...expected, sha256: null, sourceIdentity: null };
  assert.equal(photographicThumbnailEvidence(response(), png(), unknown).sha256, sha);
  assert.throws(() => photographicThumbnailEvidence(response(expected, { "X-TP-Source-SHA256": "unknown" }), png(), unknown), /verification/);
  assert.equal(unknown.sha256, null); assert.equal(unknown.sourceIdentity, null);
});

test("recorded thumbnails require their exact policy and recorded display transform", () => {
  const policy = "b".repeat(64), recorded = { ...expected, view: "recorded", policy };
  const headers = { "X-TP-Presentation-View": "recorded", "X-TP-Presentation-Policy-SHA256": policy, "X-TP-Display-Transform": "recorded_presentation_to_8bit_srgb_v1" };
  assert.equal(photographicThumbnailEvidence(response(expected, headers), png(), recorded).policy, policy);
  assert.throws(() => photographicThumbnailEvidence(response(expected, { ...headers, "X-TP-Presentation-Policy-SHA256": "none" }), png(), recorded), /verification/);
});

test("missing indexed presentation authority uses an explicitly labelled clipping fallback", async () => {
  const h = harness(async (url) => response(proof(url))); h.context.view = "recorded";
  const snapshot = page(0, 1), rows = cards(snapshot); h.api.render(snapshot, rows); await settle();
  assert.match(h.calls[0][0], /view=clip_srgb/);
  assert.match(rows[0].control.children[0].alt, /clipping fallback/); assert.equal(rows[0].control.dataset.thumbnailState, "ready");
  assert.equal(snapshot.scenes[0].thumbnail.presentation_policy_sha256, null);
});

test("capability off and invalid metadata issue no thumbnail requests", async () => {
  const h = harness(async (url) => response(proof(url)));
  const snapshot = page(0), rows = cards(snapshot); h.context.enabled = false; h.api.render(snapshot, rows); await settle();
  assert.equal(h.calls.length, 0); assert.equal(rows[0].control.children.length, 0);
  h.context.enabled = "true"; h.api.render(snapshot, rows); await settle(); assert.equal(h.calls.length, 0);
  h.context.enabled = true; snapshot.scenes[0].thumbnail.url = "https://elsewhere.test/v1/jobs/job-one/photographic-review/thumbnail";
  snapshot.scenes = snapshot.scenes.slice(0, 1); h.api.render(snapshot, cards(snapshot)); await settle(); assert.equal(h.calls.length, 0);
});

test("at most two visible-page requests run, and a page change aborts old work without adjacent prefetch", async () => {
  const waiting = [];
  const h = harness((url, options) => new Promise((resolve, reject) => {
    const request = { url, resolve: () => resolve(response(proof(url))), aborted: false }; waiting.push(request);
    options.signal.addEventListener("abort", () => { request.aborted = true; reject(new Error("aborted")); }, { once: true });
  }));
  const first = page(0); h.api.render(first, cards(first)); await settle();
  assert.equal(waiting.length, 2); assert.equal(h.api.stats().running, 2);
  waiting[0].resolve(); await settle(); assert.equal(waiting.length, 3); assert.equal(h.api.stats().running, 2);
  const next = page(8); h.api.render(next, cards(next)); await settle();
  assert.equal(waiting[1].aborted, true); assert.equal(waiting[2].aborted, true);
  assert.equal(h.api.stats().running, 2);
  assert.ok(waiting.slice(3).every((row) => ["room-8", "room-9"].includes(new URL(row.url, "http://localhost").searchParams.get("scene_id"))));
  assert.equal(h.calls.some(([url]) => /room-(?:1[6-9]|2\d)/.test(url)), false);
  h.api.reset(); await settle(); assert.equal(h.api.stats().running, 0);
});

test("cache is capped at sixteen entries and sixteen MiB; scope changes revoke every object URL", async () => {
  const h = harness(async (url) => response(proof(url), { "X-TP-Presentation-Policy-SHA256": "b".repeat(64) }, png(1048576)));
  for (const start of [0, 8, 16]) { const snapshot = indexedPage(start); h.api.render(snapshot, cards(snapshot)); await settle(); }
  assert.equal(h.calls.length, 24); assert.equal(h.api.stats().entries, 16); assert.equal(h.api.stats().bytes, 16 * 1048576); assert.equal(h.revoked.length, 8);
  const snapshot = indexedPage(16); h.api.render(snapshot, cards(snapshot)); await settle(); assert.equal(h.calls.length, 24);
  h.context.owner = "another-owner"; h.api.render(snapshot, cards(snapshot)); await settle();
  assert.equal(h.revoked.length, 24); h.api.reset(); assert.equal(h.revoked.length, h.created.length);
});

test("late responses cannot paint a changed owner, and malformed/decoder-failed cards retain navigation", async () => {
  let finish;
  const h = harness((url) => new Promise((resolve) => { finish = () => resolve(response(proof(url))); }));
  const snapshot = page(0, 1), rows = cards(snapshot); h.api.render(snapshot, rows); h.context.owner = "another-owner";
  finish(); await settle(); assert.equal(h.created.length, 0);
  const bad = harness(async (url) => response(proof(url), { "X-TP-Source-SHA256": "d".repeat(64) }));
  const failed = cards(snapshot); bad.api.render(snapshot, failed); await settle();
  assert.equal(failed[0].control.dataset.thumbnailState, "failed"); assert.equal(failed[0].control.disabled, undefined);
  const indexed = indexedPage(0, 1), good = harness(async (url) => response(proof(url), { "X-TP-Presentation-Policy-SHA256": "b".repeat(64) })), decoded = cards(indexed);
  good.api.render(indexed, decoded); await settle(); const originalUrl = decoded[0].control.children[0].src; decoded[0].control.children[0].onerror();
  assert.equal(decoded[0].control.dataset.thumbnailState, "failed"); assert.equal(good.api.stats().bytes, 0); assert.equal(good.revoked.length, 1);
  assert.deepEqual(good.revoked, [originalUrl], "Replacing a cached URL with failure revokes its exact previous URL once");
  good.api.render(indexed, cards(indexed)); await settle(); assert.equal(good.calls.length, 1, "A failed indexed card does not retry on unrelated rerenders");
  good.api.reset(); assert.deepEqual(good.revoked, [originalUrl], "Reset cannot revoke an already replaced URL twice");
});

test("IntersectionObserver queues only intersecting cards from the current page", async () => {
  let observer;
  class Observer {
    constructor(callback) { this.callback = callback; this.targets = []; observer = this; }
    observe(node) { this.targets.push(node); } unobserve() {} disconnect() { this.targets = []; }
  }
  const h = harness(async (url) => response(proof(url)), { IntersectionObserver: Observer });
  const snapshot = page(0); h.api.render(snapshot, cards(snapshot)); await settle(); assert.equal(h.calls.length, 0);
  observer.callback([{ isIntersecting: true, target: observer.targets[2] }]); await settle();
  assert.equal(h.calls.length, 1); assert.match(h.calls[0][0], /scene_id=room-2/);
  h.api.reset();
});

test("a delayed replacement-image observer reuses an indexed fetch that completed after rerender", async (t) => {
  const observers = [], pending = [];
  class Observer {
    constructor(callback) { this.callback = callback; this.targets = []; observers.push(this); }
    observe(node) { this.targets.push(node); } unobserve() {} disconnect() { this.targets = []; }
  }
  const h = harness((url, options) => new Promise((resolve, reject) => {
    pending.push(() => resolve(response(proof(url), { "X-TP-Presentation-Policy-SHA256": "b".repeat(64) })));
    options.signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
  }), { IntersectionObserver: Observer });
  t.after(() => h.api.reset());
  const snapshot = indexedPage(0, 1), first = cards(snapshot);
  h.api.render(snapshot, first);
  observers[0].callback([{ isIntersecting: true, target: observers[0].targets[0] }]); await settle();
  assert.equal(h.calls.length, 1);
  const replacement = cards(snapshot); h.api.render(snapshot, replacement);
  const observer = observers.at(-1), image = observer.targets[0];
  pending[0](); await settle();
  assert.equal(replacement[0].control.dataset.thumbnailState, "ready");
  const originalUrl = image.src;
  observer.callback([{ isIntersecting: true, target: image }]); await settle();
  assert.equal(h.calls.length, 1, "The later observer callback cannot create another source render");
  assert.equal(h.created.length, 1); assert.equal(image.src, originalUrl);
  assert.deepEqual(h.api.stats(), { entries: 1, cachedEntries: 1, bytes: 33, running: 0, queued: 0 });
  assert.deepEqual(h.revoked, []);
  h.api.reset(); assert.deepEqual(h.revoked, [originalUrl], "Every created blob URL remains owned by reset cleanup");
});

test("the response deadline includes a stalled body and the streaming byte limit rejects overflow", async () => {
  let timeout, aborted = false;
  const h = harness(async (_, options) => {
    options.signal.addEventListener("abort", () => { aborted = true; });
    return new Response(new ReadableStream({ start() {} }), { headers: { "Content-Type": "image/png" } });
  }, { timers: { setTimeout(fn) { timeout = fn; return 1; }, clearTimeout() {} } });
  const snapshot = page(0, 1), rows = cards(snapshot); h.api.render(snapshot, rows); await settle();
  timeout(); await settle(); assert.equal(aborted, true); assert.equal(rows[0].control.dataset.thumbnailState, "failed"); assert.equal(h.created.length, 0);
  const tooLarge = harness(async (url) => response(proof(url), {}, png(1048577))), oversized = cards(snapshot);
  tooLarge.api.render(snapshot, oversized); await settle();
  assert.equal(oversized[0].control.dataset.thumbnailState, "failed"); assert.equal(tooLarge.created.length, 0);
});

test("catalogue conflicts revoke stale URLs, stop queued work and require explicit refresh", async () => {
  let conflict = false, notified = 0;
  const h = harness(async (url) => conflict ? new Response("changed", { status: 409 }) : response(proof(url)), { onCatalogueChanged: () => { notified += 1; } });
  let snapshot = page(0, 1); h.api.render(snapshot, cards(snapshot)); await settle(); assert.equal(h.created.length, 1);
  conflict = true; snapshot = page(8); const failed = cards(snapshot); h.api.render(snapshot, failed); await settle();
  assert.equal(notified, 1); assert.equal(h.revoked.length, 1); assert.equal(h.api.stats().running, 0); assert.equal(h.api.stats().queued, 0);
  assert.ok(failed.every((row) => row.control.dataset.thumbnailState === "failed"));
  const count = h.calls.length; h.api.render(snapshot, cards(snapshot)); await settle(); assert.equal(h.calls.length, count);
  h.api.reset(); h.api.render(snapshot, cards(snapshot)); await settle(); assert.equal(notified, 2);
});

test("an authorization failure freezes pending thumbnail work before any bytes become visible", async () => {
  const codes = [];
  const h = harness(async () => new Response("sign in", { status: 401 }), { onAuthorizationFailure: (code) => { codes.push(code); h.context.ready = false; h.api.reset(); } });
  const snapshot = page(0); h.api.render(snapshot, cards(snapshot)); await settle();
  assert.ok(codes.every((code) => code === 401)); assert.ok(codes.length > 0);
  assert.equal(h.created.length, 0); assert.equal(h.api.stats().running, 0); assert.equal(h.api.stats().entries, 0);
});

test("a delayed conflict from an aborted old page cannot invalidate the current page", async () => {
  let finish, notified = 0;
  const h = harness((url) => url.includes("scene_id=room-0") ? new Promise((resolve) => { finish = () => resolve(new Response("changed", { status: 409 })); })
    : Promise.resolve(response(proof(url))), { onCatalogueChanged: () => { notified += 1; } });
  let snapshot = page(0, 1); h.api.render(snapshot, cards(snapshot)); await settle();
  snapshot = page(8, 1); const current = cards(snapshot); h.api.render(snapshot, current); await settle();
  assert.equal(current[0].control.dataset.thumbnailState, "ready");
  finish(); await settle(); assert.equal(notified, 0); assert.equal(h.revoked.length, 0);
  assert.equal(current[0].control.dataset.thumbnailState, "ready"); h.api.reset();
});

test("nullable legacy source identities reread changed private bytes even when catalogue metadata stays unchanged", async () => {
  let current = sha;
  const h = harness(async (url) => response(proof(url), { "X-TP-Source-SHA256": current, "X-TP-Source-Identity-SHA256": current }));
  const snapshot = page(0, 1); snapshot.scenes[0].thumbnail.sha256 = snapshot.scenes[0].thumbnail.source_identity_sha256 = null;
  const first = cards(snapshot); h.api.render(snapshot, first); await settle();
  assert.equal(h.api.stats().cachedEntries, 0); assert.equal(first[0].control.dataset.thumbnailState, "ready");
  current = "d".repeat(64); const second = cards(snapshot); h.api.render(snapshot, second); await settle();
  assert.equal(h.calls.length, 2); assert.equal(h.revoked.length, 1); assert.equal(h.created.length, 2);
  assert.notEqual(second[0].control.children[0].src, first[0].control.children[0].src);
  assert.equal(second[0].control.dataset.thumbnailState, "ready"); assert.equal(h.api.stats().cachedEntries, 0); h.api.reset();
});

test("either nullable source digest or policy remains binding-local instead of entering reusable cache", async () => {
  for (const field of ["sha256", "source_identity_sha256", "presentation_policy_sha256"]) {
    const snapshot = indexedPage(0, 1); snapshot.scenes[0].thumbnail[field] = null;
    const h = harness(async (url) => response(proof(url), { "X-TP-Presentation-Policy-SHA256": "b".repeat(64) }));
    h.api.render(snapshot, cards(snapshot)); await settle(); h.api.render(snapshot, cards(snapshot)); await settle();
    assert.equal(h.calls.length, 2, field); assert.equal(h.api.stats().cachedEntries, 0, field);
    assert.equal(h.revoked.length, 1, field); h.api.reset();
  }
});

test("a new private binding starts a fresh read instead of attaching a pending old response", async () => {
  let finish, count = 0;
  const h = harness((url) => ++count === 1 ? new Promise((resolve) => { finish = () => resolve(response(proof(url))); })
    : Promise.resolve(response(proof(url), { "X-TP-Source-SHA256": "d".repeat(64), "X-TP-Source-Identity-SHA256": "d".repeat(64) })));
  const snapshot = page(0, 1); snapshot.scenes[0].thumbnail.sha256 = snapshot.scenes[0].thumbnail.source_identity_sha256 = null;
  h.api.render(snapshot, cards(snapshot)); const current = cards(snapshot); h.api.render(snapshot, current); await settle();
  assert.equal(h.calls.length, 2); assert.equal(h.created.length, 1); const url = current[0].control.children[0].src;
  finish(); await settle(); assert.equal(h.created.length, 1); assert.equal(current[0].control.children[0].src, url); h.api.reset();
});

test("ephemeral private bytes and reusable indexed bytes share the same sixteen MiB retention bound", async () => {
  const h = harness(async (url) => response(proof(url), { "X-TP-Presentation-Policy-SHA256": "b".repeat(64) }, png(1048576)));
  for (const start of [0, 8]) { const snapshot = indexedPage(start); h.api.render(snapshot, cards(snapshot)); await settle(); }
  const privatePage = page(16); h.api.render(privatePage, cards(privatePage)); await settle();
  assert.equal(h.api.stats().entries, 16); assert.equal(h.api.stats().cachedEntries, 8); assert.equal(h.api.stats().bytes, 16 * 1048576);
  h.api.render(privatePage, cards(privatePage)); await settle();
  assert.equal(h.api.stats().entries, 16); assert.equal(h.api.stats().bytes, 16 * 1048576); assert.equal(h.calls.length, 32); h.api.reset();
});
