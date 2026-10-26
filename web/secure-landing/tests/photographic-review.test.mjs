import test from "node:test";
import assert from "node:assert/strict";
import {
  createPhotographicReview, photographicVariantLabel, photographicProperties,
  photographicDecisionMatches, nativeCropGeometry, photographicAssetUrl, photographicRecipe, photographicPreviewMatches, photographicRetryAfterMs,
} from "../portal-src/photographic-review.js";
import { photographicDeliveryAttempt } from "../portal-src/photographic-delivery-history.js";

const SHA = "a".repeat(64);
const ORIGINAL_SHA = "b".repeat(64);
const variant = { role: "final", artifact_path: "input-0000/delivery.tif", sha256: SHA,
  width: 8000, height: 6000, bit_depth: 16, color_profile: "sRGB", intended_use: "Photographic master" };
const original = { ...variant, role: "original", artifact_path: "original/input-0000/source.tif", sha256: ORIGINAL_SHA };
const WEBSITE_PRESET = { schema: "tp.photo.export_preset.v1", name: "website", format: "jpeg", intended_use: "Website encoded-sRGB delivery",
  long_edge: 2560, jpeg_quality: 92, sharpen_amount: 0.2, render: "perceptual_srgb", shoulder: 0.8 };

test("photographic properties identify the master independently from a reduced preview", () => {
  assert.equal(photographicVariantLabel(variant), "Final · input-0000/delivery.tif");
  assert.match(photographicProperties(variant), /8000 × 6000 px · 16-bit · sRGB · Photographic master/);
  assert.equal(photographicVariantLabel({ role: "previous_revision", artifact_path: "r1.tif" }), "Previous revision · r1.tif");
  assert.match(photographicProperties({}), /Dimensions not reported.*Precision not reported.*Profile not reported/);
});

test("review decisions match exact artifact path and full SHA, never a renamed or modified image", () => {
  const saved = { status: "approved", artifact_path: variant.artifact_path, sha256: SHA };
  assert.equal(photographicDecisionMatches(saved, variant), true);
  assert.equal(photographicDecisionMatches(saved, { ...variant, sha256: ORIGINAL_SHA }), false);
  assert.equal(photographicDecisionMatches(saved, { ...variant, artifact_path: "other.tif" }), false);
  assert.equal(photographicDecisionMatches({ ...saved, status: "probably_good" }, variant), false);
  assert.equal(photographicDecisionMatches({ ...saved, sha256: "a" }, { ...variant, sha256: "a" }), false);
});

test("native crop geometry uses native dimensions, clamps edges, and cannot exceed the source", () => {
  assert.deepEqual(nativeCropGeometry(variant, { x: 0.5, y: 0.5 }, 512), { x: 3744, y: 2744, width: 512, height: 512 });
  assert.deepEqual(nativeCropGeometry(variant, { x: 0, y: 1 }, 512), { x: 0, y: 5488, width: 512, height: 512 });
  assert.deepEqual(nativeCropGeometry({ width: 21, height: 10 }, { x: 1, y: 0 }, 2048), { x: 0, y: 0, width: 21, height: 10 });
  assert.equal(nativeCropGeometry(variant, { x: 1, y: 1 }, 100000).width, 2048);
  assert.throws(() => nativeCropGeometry({ width: "8000", height: 6000 }), /unavailable/);
});

test("photographic assets stay on the current authenticated job and reject cross-origin or other-job links", () => {
  const origin = "https://portal.example";
  assert.equal(photographicAssetUrl("/v1/jobs/job-one/photographic-review/files/id/export.zip?q=1", "job-one", origin), "/v1/jobs/job-one/photographic-review/files/id/export.zip?q=1");
  for (const url of ["https://other.example/v1/jobs/job-one/file.png", "/v1/jobs/job-two/file.png", "javascript:alert(1)", "https://name:secret@portal.example/v1/jobs/job-one/file.png", "/v1/jobs/job-one/../job-two/file.png"]) {
    assert.equal(photographicAssetUrl(url, "job-one", origin), "");
  }
});

test("bounded adjustments are explicit and cannot silently authorize unsupported values", () => {
  assert.deepEqual(photographicRecipe({ exposure: -0.1, warmth: 0.05, saturation: 1.1 }), {
    exposure_stops: -0.1, white_balance: [1.05, 1, 0.95], saturation: 1.1, materials: [],
  });
  for (const adjustment of [{ exposure: NaN }, { exposure: 1.01 }, { warmth: -0.16 }, { saturation: 0.79 }, { saturation: 1.21 }]) {
    assert.throws(() => photographicRecipe(adjustment), /outside/);
  }
});

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = []; this.dataset = {}; this.attributes = {}; this.listeners = {};
    this.className = ""; this.textContent = ""; this.value = ""; this.style = { setProperty() {} };
    this.classList = {
      add: (name) => { if (!this.className.split(" ").includes(name)) this.className += ` ${name}`; },
      remove: (name) => { this.className = this.className.split(" ").filter((item) => item !== name).join(" "); },
      toggle: (name, force) => { const value = force ?? !this.className.split(" ").includes(name); this.classList[value ? "add" : "remove"](name); },
      contains: (name) => this.className.split(" ").includes(name),
    };
  }
  get options() { return this.children; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = [...items]; }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; if (key === "src") this.src = ""; }
  remove() { this.removed = true; }
  click() { this.clicked = true; }
  addEventListener(key, handler) { this.listeners[key] = handler; }
  querySelectorAll(selector) {
    const key = selector.match(/^\[data-(.+)\]$/)?.[1]?.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    const all = this.children.flatMap((child) => [child, ...child.querySelectorAll(selector)]);
    return all.filter((child) => key && Object.hasOwn(child.dataset, key));
  }
  find(ui) { return this.dataset.ui === ui ? this : this.children.map((child) => child.find(ui)).find(Boolean); }
  async trigger(event = "click") { this.listeners[event]?.({ preventDefault() {}, key: "" }); await settle(); }
}
async function settle() { for (let count = 0; count < 8; count += 1) await new Promise((resolve) => setImmediate(resolve)); }
function fixture({ responseHash = null, mutationError = false, extraFetch = null, renderDebounceMs = 0, authMode = "managed", sessionStorage = undefined, historyEnabled = true } = {}) {
  const root = new Element("main");
  const body = new Element("body");
  const state = { auth: { mode: authMode, features: { photographicDeliveryHistory: historyEnabled } } };
  const requests = [];
  const recoveryValues = new Map();
  const recoveryStorage = sessionStorage === undefined ? { getItem: (key) => recoveryValues.get(key), setItem: (key, value) => recoveryValues.set(key, value) } : sessionStorage;
  let saved = [];
  const data = () => ({ state_version: saved.length, scenes: [{ scene_id: "input-0000", label: "Living room", variants: [variant, original] }],
    decisions: saved, groups: [], presets: [WEBSITE_PRESET] });
  const fetchImpl = async (url, options) => {
    requests.push({ url, options });
    const extra = extraFetch?.(url, options);
    if (extra) return extra;
    if (options.method === "POST") {
      if (mutationError) return new Response(JSON.stringify({ success: false, error: { code: "REVIEW_CONFLICT", message: "stale" } }), { status: 409 });
      saved = [JSON.parse(options.body)];
      return new Response(JSON.stringify({ success: true, data: { saved: true } }));
    }
    if (url.includes("/quality?")) return new Response(JSON.stringify({ success: true, data: {
      metrics: { banding_risk: 0.15, color_drift: 0.02 }, scope: "reference_relative_not_photographic_acceptance", algorithm_revision: "spatial_v2",
    } }));
    if (url.includes("/render?") || url.includes("/native-crop?")) {
      const params = new URL(url, "http://localhost").searchParams;
      return new Response(new Blob(["png"], { type: "image/png" }), { headers: {
        "X-TP-Source-SHA256": url.includes("/native-crop?") ? responseHash || params.get("sha256") : params.get("sha256"),
        "X-TP-Presentation-View": params.get("view") || "clip_srgb", "X-TP-Presentation-Policy-SHA256": "none",
        "X-TP-Native-Width": params.get("width") || "1600", "X-TP-Native-Height": params.get("height") || "1200",
      } });
    }
    return new Response(JSON.stringify({ success: true, data: data() }));
  };
  const api = createPhotographicReview({ state, els: { artifactShellContent: root }, document: { body, createElement: (tag) => new Element(tag) }, fetchImpl, renderDebounceMs, sessionStorage: recoveryStorage,
    _buildAuthHeaders: (headers, method) => ({ ...headers, ...(method === "POST" ? { "X-CSRF-Token": "fixture-csrf" } : { "x-api-key": "fixture-key" }) }) });
  const job = { id: "job-one", pipeline: "lux-depth", artifacts: [{ ...variant, path: variant.artifact_path }] };
  return { root, body, state, requests, api, job };
}

test("Review exposes explicit versions, fit-preview truth, and sends authenticated SHA-bound render requests", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  const primary = h.root.find("photographic-primary");
  assert.equal(primary.value, variant.artifact_path);
  assert.deepEqual(primary.options.map((option) => option.textContent), ["Final · input-0000/delivery.tif", "Original · original/input-0000/source.tif"]);
  assert.match(h.root.find("photographic-resolution-status").textContent, /reduced.*does not show native detail/);
  const render = h.requests.find((request) => request.url.includes("/render?"));
  assert.match(render.url, new RegExp(`sha256=${SHA}`));
  assert.match(render.url, /max_edge=1600/);
  assert.equal(render.options.credentials, "same-origin");
  assert.equal(render.options.headers["x-api-key"], "fixture-key");
});

test("a retained photographic catalogue may finish after 43.51 seconds within its bounded 90-second budget", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let complete;
  const h = fixture({ extraFetch: (url) => url.endsWith("/photographic-review")
    ? new Promise((resolve) => { complete = resolve; }) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, variant);
  const catalogue = h.requests.find((request) => request.url.endsWith("/photographic-review"));
  t.mock.timers.tick(43510);
  assert.equal(catalogue.options.signal.aborted, false);
  complete(new Response(JSON.stringify({ success: true, data: { state_version: 0,
    scenes: [{ scene_id: "input-0000", variants: [variant, original] }], decisions: [], groups: [], presets: [] } })));
  await settle();
  assert.match(h.root.find("photographic-review-status").textContent, /Review loaded/);
  assert.equal(h.root.find("photographic-primary").value, variant.artifact_path);
});

test("a catalogue deadline is visible at 90 seconds and Refresh review recovers without replaying changes", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let first = true;
  const h = fixture({ extraFetch: (url, options) => {
    if (!url.endsWith("/photographic-review") || !first) return null;
    first = false;
    return new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))));
  } });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, variant);
  const catalogue = h.requests.find((request) => request.url.endsWith("/photographic-review"));
  t.mock.timers.tick(89999);
  assert.equal(catalogue.options.signal.aborted, false);
  t.mock.timers.tick(1);
  await settle();
  assert.equal(catalogue.options.signal.aborted, true);
  assert.match(h.root.find("photographic-review-status").textContent, /timed out.*Refresh review/);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "error");
  assert.equal(h.root.find("photographic-review").classList.contains("hidden"), false);
  await h.root.find("photographic-review-refresh").trigger();
  assert.match(h.root.find("photographic-review-status").textContent, /Review refreshed/);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "info");
  assert.equal(h.requests.filter((request) => request.options.method === "POST").length, 0);
});

test("an intentional catalogue abort cannot publish a stale timeout or loading status in the next job", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = fixture({ extraFetch: (url, options) => url === "/v1/jobs/job-one/photographic-review"
    ? new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(new DOMException("Scope changed", "AbortError")))) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, variant);
  const previous = h.requests[0];
  h.api.show({ ...h.job, id: "job-two" }, variant);
  await settle();
  assert.equal(previous.options.signal.aborted, true);
  assert.match(h.root.find("photographic-review-status").textContent, /Review loaded/);
  t.mock.timers.tick(90000);
  await settle();
  assert.match(h.root.find("photographic-review-status").textContent, /Review loaded/);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "info");
});

test("saved human decisions use the exact selected master, state version, and managed CSRF transport", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  h.root.find("photographic-primary-image").onload();
  h.root.find("photographic-decision-note").value = "Window edges and sky inspected.";
  await h.root.find("photographic-decision-approved").trigger();
  const mutation = h.requests.find((request) => request.url.endsWith("/decisions"));
  assert.deepEqual(JSON.parse(mutation.options.body), { artifact_path: variant.artifact_path, sha256: SHA,
    source_identity_sha256: SHA, status: "approved", note: "Window edges and sky inspected.", expected_version: 0 });
  assert.equal(mutation.options.headers["X-CSRF-Token"], "fixture-csrf");
  assert.match(h.root.find("photographic-decision-status").textContent, /Approved.*exact SHA-256/);
});

test("recorded presentation is explicit, policy-bound in requests/cache, and clipping fallback is labelled", async () => {
  let policySha = "c".repeat(64);
  const h = fixture({ extraFetch: (url) => {
    if (url.endsWith("/photographic-review")) return new Response(JSON.stringify({ success: true, data: {
      state_version: 0, scenes: [{ scene_id: "input-0000", label: "Room", variants: [
        { ...variant, presentation_policy_sha256: policySha, presentation: { available_views: ["recorded", "clip_srgb"],
          policy: { stage: "display_referred" }, policy_sha256: policySha } }, original,
      ] }], decisions: [], groups: [], presets: [],
    } }));
    if (url.includes("/render?") || url.includes("/native-crop?")) {
      const params = new URL(url, "http://localhost").searchParams;
      return new Response(new Blob(["png"], { type: "image/png" }), { headers: {
        "X-TP-Source-SHA256": params.get("sha256"),
        "X-TP-Presentation-View": params.get("view"),
        "X-TP-Presentation-Policy-SHA256": params.get("sha256") === SHA ? policySha : "none",
        "X-TP-Native-Width": params.get("width") || "1600", "X-TP-Native-Height": params.get("height") || "1200",
      } });
    }
    return null;
  } });
  h.state.auth.features = { photographicRecordedPresentation: true };
  h.api.show(h.job, variant); await settle();
  let requests = h.requests.filter((request) => request.url.includes("/render?"));
  assert.equal(new URL(requests.find((request) => request.url.includes(`sha256=${SHA}`)).url, "http://localhost").searchParams.get("view"), "recorded");
  assert.match(h.root.find("photographic-resolution-status").textContent, /authoritative recorded presentation.*display_referred/);
  assert.match(h.root.find("photographic-resolution-status").textContent, /labelled clipping fallback/);
  const before = requests.length;
  h.root.find("photographic-presentation").value = "clip_srgb";
  await h.root.find("photographic-presentation").trigger("change");
  assert.ok(h.requests.filter((request) => request.url.includes("/render?")).length > before);
  h.root.find("photographic-presentation").value = "recorded";
  await h.root.find("photographic-presentation").trigger("change");
  const afterViews = h.requests.filter((request) => request.url.includes("/render?")).length;
  policySha = "d".repeat(64);
  await h.root.find("photographic-review-refresh").trigger();
  assert.ok(h.requests.filter((request) => request.url.includes("/render?")).length > afterViews,
    "A policy change with unchanged source bytes must not reuse old rendered pixels");
  h.root.find("photographic-primary-image").onload();
  await h.root.find("photographic-decision-approved").trigger();
  const mutation = h.requests.find((request) => request.url.endsWith("/decisions"));
  assert.deepEqual(JSON.parse(mutation.options.body).inspection_presentation, { view: "recorded", policy_sha256: policySha });
});

test("a stale policy response fails inspection closed without replacing source approval", async () => {
  const digest = "c".repeat(64);
  const approved = { artifact_path: variant.artifact_path, sha256: SHA, source_identity_sha256: SHA, status: "approved" };
  const h = fixture({ extraFetch: (url) => url.endsWith("/photographic-review") ? new Response(JSON.stringify({ success: true, data: {
    state_version: 1, scenes: [{ scene_id: "input-0000", variants: [{ ...variant, presentation_policy_sha256: digest,
      presentation: { available_views: ["recorded", "clip_srgb"], policy_sha256: digest } }] }], decisions: [approved],
  } })) : null });
  h.state.auth.features = { photographicRecordedPresentation: true };
  h.api.show(h.job, variant); await settle();
  assert.match(h.root.find("photographic-review-status").textContent, /identity or native dimensions could not be verified/);
  assert.match(h.root.find("photographic-decision-status").textContent, /Approved.*exact SHA-256/);
});

test("recorded presentation rollout is disabled independently of advertised source authority", async () => {
  const h = fixture({ extraFetch: (url) => url.endsWith("/photographic-review") ? new Response(JSON.stringify({ success: true, data: {
    state_version: 0, scenes: [{ scene_id: "input-0000", variants: [{ ...variant, presentation_policy_sha256: "c".repeat(64),
      presentation: { available_views: ["recorded", "clip_srgb"], policy_sha256: "c".repeat(64) } }] }],
  } })) : null });
  h.api.show(h.job, variant); await settle();
  assert.equal(h.root.find("photographic-presentation").disabled, true);
  assert.match(h.root.find("photographic-resolution-status").textContent, /rollout is disabled/);
  assert.ok(h.requests.filter((request) => request.url.includes("/render?")).every((request) =>
    new URL(request.url, "http://localhost").searchParams.get("view") === "clip_srgb"));
});

test("legacy catalogue and image headers keep clipping and original mutation shapes even when rollout is requested", async () => {
  const h = fixture({ extraFetch: (url) => {
    if (!url.includes("/render?") && !url.includes("/native-crop?")) return null;
    const params = new URL(url, "http://localhost").searchParams;
    return new Response(new Blob(["png"], { type: "image/png" }), { headers: {
      "X-TP-Source-SHA256": params.get("sha256"),
      "X-TP-Native-Width": params.get("width") || "1600", "X-TP-Native-Height": params.get("height") || "1200",
    } });
  } });
  h.state.auth.features = { photographicRecordedPresentation: true };
  h.api.show(h.job, variant); await settle();
  h.root.find("photographic-primary-image").onload();
  assert.equal(h.root.find("photographic-primary-image").classList.contains("hidden"), false);
  assert.ok(h.requests.filter((request) => request.url.includes("/render?")).every((request) =>
    !new URL(request.url, "http://localhost").searchParams.has("view")));
  assert.match(h.root.find("photographic-resolution-status").textContent, /labelled clipping fallback/);
  await h.root.find("photographic-decision-approved").trigger();
  const payload = JSON.parse(h.requests.find((request) => request.url.endsWith("/decisions")).options.body);
  assert.equal(Object.hasOwn(payload, "inspection_presentation"), false);
});

test("native 1:1 fails closed on a changed source hash and never falls back to an enlarged preview", async () => {
  const h = fixture({ responseHash: "c".repeat(64) });
  h.api.show(h.job, variant); await settle();
  const inspection = h.root.find("photographic-inspection");
  inspection.value = "native"; await inspection.trigger("change");
  assert.match(h.root.find("photographic-review-status").textContent, /identity or native dimensions could not be verified/);
  assert.equal(h.root.find("photographic-primary-image").src, "");
  const crop = h.requests.find((request) => request.url.includes("/native-crop?"));
  assert.match(crop.url, /width=512&height=512/);
  assert.match(crop.url, /x=3744&y=2744/);
});

test("concurrent review conflicts refresh authoritative decisions without replaying the mutation", async () => {
  const h = fixture({ mutationError: true });
  h.api.show(h.job, variant); await settle();
  await h.root.find("photographic-decision-approved").trigger();
  assert.equal(h.requests.filter((request) => request.options.method === "POST").length, 1);
  assert.match(h.root.find("photographic-review-status").textContent, /changed elsewhere.*before retrying/);
  assert.equal(h.root.find("photographic-decision-approved").disabled, false);
  assert.equal(h.root.find("photographic-create-revision").disabled, true, "a failed decision cannot enable an unpreviewed revision");
});

test("unsupported jobs do not call photographic APIs or inherit another run's workbench", async () => {
  const h = fixture();
  h.api.show({ id: "unrelated", pipeline: "archive", artifacts: [] }, null); await settle();
  assert.equal(h.requests.length, 0);
  assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
});

test("explicit local material rectangles stay bounded and retain their own source-bound recipe", () => {
  const recipe = photographicRecipe({ materials: [{ name: "Ceiling cove", region: [0.1, 0.05, 0.9, 0.4], exposure_stops: -0.1, warmth: 0.02, feather: 0.03 }] });
  assert.deepEqual(recipe.materials, [{ name: "Ceiling cove", region: [0.1, 0.05, 0.9, 0.4], exposure_stops: -0.1, white_balance: [1.02, 1, 0.98], feather: 0.03 }]);
  for (const material of [
    { name: "Sky", region: [0.9, 0, 0.1, 1], exposure_stops: 0, feather: 0.03 },
    { name: "Sky", region: [0, 0, 1, 1], exposure_stops: 0.51, feather: 0.03 },
    { name: "Sky", region: [0, 0, 1, 1], exposure_stops: 0, feather: 0.2 },
  ]) assert.throws(() => photographicRecipe({ materials: [material] }));
});

test("original inspection cannot be approved or silently reused as an export master", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  const primary = h.root.find("photographic-primary");
  primary.value = original.artifact_path; await primary.trigger("change");
  assert.equal(h.root.find("photographic-decision-approved").disabled, true);
  assert.match(h.root.find("photographic-decision-status").textContent, /Original inspection only/);
  await h.root.find("photographic-decision-approved").trigger();
  assert.equal(h.requests.filter((request) => request.options.method === "POST").length, 0);
});

import { createLazyPhotographicReview } from "../portal-src/photographic-review-loader.js";

test("photographic JavaScript and styles are lazy, same-origin, and resolve the latest selected job", async () => {
  const origin = "https://portal.example";
  const shows = [];
  const imports = [];
  let finish;
  let styleLoads = 0;
  const moduleGate = new Promise((resolve) => { finish = resolve; });
  const loader = createLazyPhotographicReview({
    origin, els: {}, document: { body: { dataset: { photographicReviewJsUrl: "/portal/assets/portal-photographic.js?v=abc" } } },
    importPhotographicModule(url) { imports.push(url); return moduleGate; },
    loadPhotographicCss() { styleLoads += 1; return Promise.resolve(); },
  });
  loader.show({ id: "archive", pipeline: "archive" }, null);
  assert.equal(imports.length, 0);
  loader.show({ id: "first", pipeline: "lux-depth" }, null);
  loader.show({ id: "second", pipeline: "lux-depth" }, null);
  finish({ createPhotographicReview: () => ({ show: (job) => shows.push(job.id) }) });
  await settle();
  assert.deepEqual(imports, ["/portal/assets/portal-photographic.js?v=abc"]);
  assert.equal(styleLoads, 1);
  assert.deepEqual(shows, ["second"]);
});

test("an untrusted workbench script URL never reaches dynamic import", async () => {
  let calls = 0;
  const loader = createLazyPhotographicReview({ origin: "https://portal.example", els: {},
    document: { body: { dataset: { photographicReviewJsUrl: "https://outside.example/portal-photographic.js" } } },
    importPhotographicModule() { calls += 1; }, loadPhotographicCss() { calls += 1; },
  });
  loader.show({ id: "job", pipeline: "lux-depth" }, null); await settle();
  assert.equal(calls, 0);
});

test("decoded native image geometry must also match the server before 1:1 is verified", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  const inspection = h.root.find("photographic-inspection");
  inspection.value = "native"; await inspection.trigger("change");
  const image = h.root.find("photographic-primary-image");
  assert.equal(image.classList.contains("hidden"), true, "a crop stays hidden until decoded dimensions are checked");
  image.naturalWidth = 512; image.naturalHeight = 512; image.onload();
  assert.equal(image.classList.contains("hidden"), false);
  assert.equal(image.style.width, "512px");
  assert.match(h.root.find("photographic-primary-properties").textContent, /native 1:1 verified/);
  image.naturalWidth = 511; image.onload();
  assert.equal(image.classList.contains("hidden"), true);
  assert.match(h.root.find("photographic-review-status").textContent, /decoded crop dimensions do not match/);
});

test("quality signals use the explicitly chosen comparison and cannot auto-approve", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  await h.root.find("photographic-measure-quality").trigger();
  const quality = h.requests.find((request) => request.url.includes("/quality?"));
  assert.match(quality.url, new RegExp(`sha256=${SHA}`));
  assert.match(quality.url, new RegExp(`reference_sha256=${ORIGINAL_SHA}`));
  assert.match(h.root.find("photographic-review-status").textContent, /Human review and approval remain separate/);
  assert.equal(h.requests.filter((request) => request.options.method === "POST").length, 0);
  assert.match(h.root.find("photographic-decision-status").textContent, /Unreviewed/);
});

test("a rendered revision preview must match its master, geometry, recipe and unapproved state", () => {
  const recipe = photographicRecipe({ exposure: -0.1 });
  const receipt = { schema: "tp.photo.revision.v1", source_master_sha256: SHA, width: 8000, height: 6000,
    output_bit_depth: 32, review_status: "unreviewed", approval_transferred: false, reference_sha256: null, recipe };
  assert.equal(photographicPreviewMatches(receipt, variant, recipe), true);
  for (const change of [{ source_master_sha256: ORIGINAL_SHA }, { width: 1600 }, { output_bit_depth: 8 },
    { approval_transferred: true }, { recipe: photographicRecipe({ exposure: 0 }) }]) {
    assert.equal(photographicPreviewMatches({ ...receipt, ...change }, variant, recipe), false);
  }
});

test("invalidating or committing a rendered preview clears its stale receipt with the preview", async () => {
  const h = fixture({ extraFetch: (url, options) => {
    if (!url.endsWith("/revisions")) return null;
    const payload = JSON.parse(options.body);
    return new Response(JSON.stringify({ success: true, data: payload.preview_only ? {
      preview_data_url: "data:image/png;base64,cG5n",
      receipt: { schema: "tp.photo.revision.v1", source_master_sha256: SHA, width: variant.width, height: variant.height,
        output_bit_depth: 32, review_status: "unreviewed", approval_transferred: false, reference_sha256: null, recipe: payload.recipe },
    } : { state_version: 1 } }));
  } });
  h.api.show(h.job, variant); await settle();
  const receipt = h.root.find("photographic-revision-receipt");
  const image = h.root.find("photographic-variant-preview");
  const commit = h.root.find("photographic-create-revision");
  await h.root.find("photographic-preview-revision").trigger();
  assert.match(receipt.textContent, new RegExp(`Rendered preview from master ${SHA}`));
  const exposure = h.root.find("photographic-revision-exposure"); exposure.value = "-0.1"; await exposure.trigger("input");
  assert.equal(receipt.textContent, "");
  assert.equal(image.src, "");
  assert.equal(image.classList.contains("hidden"), true);
  assert.equal(commit.disabled, true);
  await h.root.find("photographic-preview-revision").trigger();
  assert.match(receipt.textContent, /Rendered preview/);
  await commit.trigger();
  assert.equal(receipt.textContent, "");
  assert.equal(image.classList.contains("hidden"), true);
  assert.equal(commit.disabled, true);
});

test("decisions and revision previews bind native composite identity as well as unchanged physical bytes", () => {
  const native = { ...variant, role: "working_master", source_identity_sha256: "e".repeat(64) };
  const saved = { artifact_path: native.artifact_path, sha256: SHA, source_identity_sha256: native.source_identity_sha256, status: "approved" };
  assert.equal(photographicDecisionMatches(saved, native), true);
  assert.equal(photographicDecisionMatches({ ...saved, source_identity_sha256: undefined }, { ...native, source_identity_sha256: SHA }), true);
  assert.equal(photographicDecisionMatches({ ...saved, source_identity_sha256: undefined }, native), false);
  assert.equal(photographicDecisionMatches(saved, { ...native, source_identity_sha256: "f".repeat(64) }), false);
  assert.match(photographicVariantLabel(native), /^Native working master/);
  const recipe = photographicRecipe();
  const receipt = { schema: "tp.photo.revision.v1", source_master_sha256: SHA, source_identity_sha256: native.source_identity_sha256,
    width: 8000, height: 6000, output_bit_depth: 32, review_status: "unreviewed", approval_transferred: false, recipe };
  assert.equal(photographicPreviewMatches(receipt, native, recipe), true);
  assert.equal(photographicPreviewMatches({ ...receipt, source_identity_sha256: SHA }, native, recipe), false);
});

test("Retry-After respects both delta seconds and HTTP dates", () => {
  assert.equal(photographicRetryAfterMs("7", 0), 7000);
  assert.equal(photographicRetryAfterMs("Thu, 01 Jan 1970 00:00:08 GMT", 1000), 7000);
  assert.equal(photographicRetryAfterMs(null, 0), 30000);
});

test("rendered pixels are reused across metadata refreshes and comparison gestures", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  const initial = h.requests.filter((request) => request.url.includes("/render?")).length;
  assert.equal(initial, 2);
  const mode = h.root.find("photographic-comparison-mode");
  mode.value = "wipe"; await mode.trigger("change");
  mode.value = "blink"; await mode.trigger("change");
  await h.root.find("photographic-review-refresh").trigger();
  assert.equal(h.requests.filter((request) => request.url.includes("/render?")).length, initial);
});

test("rapid native pan is debounced into one request per distinct selected image", async () => {
  const h = fixture({ renderDebounceMs: 20 });
  h.api.show(h.job, variant); await settle();
  await new Promise((resolve) => setTimeout(resolve, 25)); await settle();
  const inspection = h.root.find("photographic-inspection");
  inspection.value = "native"; await inspection.trigger("change");
  await h.root.find("photographic-pan-right").trigger();
  await h.root.find("photographic-pan-right").trigger();
  await h.root.find("photographic-pan-right").trigger();
  await new Promise((resolve) => setTimeout(resolve, 25)); await settle();
  const crops = h.requests.filter((request) => request.url.includes("/native-crop?"));
  assert.equal(crops.length, 2);
  assert.ok(crops.every((request) => new URL(request.url, "http://localhost").searchParams.get("x") === "4512"));
});

test("a rate-limited native request preserves the review decision and honors Retry-After", async () => {
  const h = fixture({ extraFetch: (url) => url.includes("/native-crop?")
    ? new Response("rate limited", { status: 429, headers: { "Retry-After": "60" } }) : null });
  h.api.show(h.job, variant); await settle();
  const inspection = h.root.find("photographic-inspection");
  inspection.value = "native"; await inspection.trigger("change");
  const before = h.requests.length;
  await h.root.find("photographic-review-refresh").trigger();
  assert.equal(h.requests.length, before, "refresh must wait instead of repeating a throttled request");
  assert.match(h.root.find("photographic-review-status").textContent, /Retry after.*selection and saved decisions are retained/);
  assert.equal(h.root.find("photographic-primary").value, variant.artifact_path);
  assert.match(h.root.find("photographic-decision-status").textContent, /Unreviewed/);
});

test("a saved previous-revision lighting reference remains selectable for a new final", async () => {
  const final = { ...variant, artifact_path: "review/new-final.tif", sha256: "c".repeat(64) };
  const previous = { ...variant, role: "previous_revision" };
  const h = fixture({ extraFetch: (url, options) => url.endsWith("/photographic-review") && options.method === "GET"
    ? new Response(JSON.stringify({ success: true, data: { state_version: 2, decisions: [], presets: [],
      scenes: [{ scene_id: "input-0000", label: "Living room", lighting_group: "Daylight", reference_sha256: SHA, variants: [previous, final, original] }],
      groups: [{ name: "Daylight", reference_sha256: SHA, members: [SHA] }] } })) : null });
  h.api.show(h.job, final); await settle();
  assert.equal(h.root.find("photographic-primary").value, final.artifact_path);
  assert.equal(h.root.find("photographic-reference").value, SHA);
  assert.ok(h.root.find("photographic-reference").options.some((option) => option.value === SHA && /Previous revision/.test(option.textContent)));
});

test("clearing a destination long edge restores the supported source-dimensions override", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  await h.root.find("photographic-decision-approved").trigger();
  const checkbox = h.root.find("photographic-preset"); checkbox.checked = true; await checkbox.trigger("change");
  const edge = h.root.find("photographic-preset-long_edge"); edge.value = ""; await edge.trigger("input");
  await h.root.find("photographic-deliver-approved").trigger();
  const sent = h.requests.find((request) => request.url.endsWith("/delivery"));
  assert.deepEqual(JSON.parse(sent.options.body).presets, [{ name: "website", long_edge: null }]);
});

test("a delivery link is invalidated when native identity changes despite renewed approval of unchanged physical bytes", async (t) => {
  const previousLocation = globalThis.location;
  globalThis.location = { origin: "http://localhost" };
  t.after(() => { if (previousLocation === undefined) delete globalThis.location; else globalThis.location = previousLocation; });
  let sourceIdentity = "e".repeat(64);
  const h = fixture({ extraFetch: (url, options) => {
    if (url.endsWith("/photographic-review")) return new Response(JSON.stringify({ success: true, data: {
      state_version: 1, scenes: [{ scene_id: "input-0000", label: "Living room", variants: [{ ...variant, role: "working_master", source_identity_sha256: sourceIdentity }] }],
      decisions: [{ artifact_path: variant.artifact_path, sha256: SHA, source_identity_sha256: sourceIdentity, status: "approved" }],
      groups: [], presets: [WEBSITE_PRESET],
    } }));
    if (url.endsWith("/delivery") && options.method === "POST") return new Response(JSON.stringify({ success: true, data: {
      download_url: "/v1/jobs/job-one/photographic-review/files/export/photos.zip", filename: "photos.zip", sha256: ORIGINAL_SHA, size_bytes: 1234,
    } }));
    return null;
  } });
  h.api.show(h.job, variant); await settle();
  const checkbox = h.root.find("photographic-preset"); checkbox.checked = true; await checkbox.trigger("change");
  await h.root.find("photographic-deliver-approved").trigger();
  assert.ok(h.root.find("photographic-delivery-download"));
  await h.root.find("photographic-review-refresh").trigger();
  assert.equal(h.root.find("photographic-delivery-result").textContent, "", "unchanged approval and identity retain the verified link");
  sourceIdentity = "f".repeat(64);
  await h.root.find("photographic-review-refresh").trigger();
  assert.match(h.root.find("photographic-delivery-result").textContent, /approval changed.*newly verified bundle/);
  assert.match(h.root.find("photographic-decision-status").textContent, /Approved/, "new approval does not authorize an old bundle with different interpretation identity");
});

test("native variants with identical array bytes remain independently selectable and export only the chosen exact source", async () => {
  const first = { ...variant, role: "working_master", artifact_path: "first/master.npy", source_identity_sha256: "e".repeat(64) };
  const second = { ...first, artifact_path: "second/master.npy", source_identity_sha256: "f".repeat(64) };
  const h = fixture({ extraFetch: (url) => url.endsWith("/photographic-review") ? new Response(JSON.stringify({ success: true, data: {
    state_version: 1, scenes: [{ scene_id: "input-0000", label: "Living room", variants: [first, second] }],
    decisions: [first, second].map((source) => ({ ...source, status: "approved" })), groups: [], presets: [WEBSITE_PRESET],
  } })) : null });
  h.api.show(h.job, variant); await settle();
  const primary = h.root.find("photographic-primary");
  assert.deepEqual(primary.options.map((option) => option.value), [first.artifact_path, second.artifact_path]);
  assert.deepEqual(h.root.find("photographic-comparison").options.map((option) => option.value), ["", second.artifact_path]);
  primary.value = second.artifact_path; await primary.trigger("change");
  await h.root.find("photographic-decision-selected").trigger();
  const decision = JSON.parse(h.requests.find((request) => request.url.endsWith("/decisions")).options.body);
  assert.equal(decision.artifact_path, second.artifact_path);
  assert.equal(decision.source_identity_sha256, second.source_identity_sha256);
  const delivery = h.root.find("photographic-delivery-version");
  assert.deepEqual(delivery.options.map((option) => option.value), ["", first.artifact_path, second.artifact_path]);
  delivery.value = second.artifact_path; await delivery.trigger("change");
  const checkbox = h.root.find("photographic-preset"); checkbox.checked = true; await checkbox.trigger("change");
  await h.root.find("photographic-deliver-approved").trigger();
  const payload = JSON.parse(h.requests.find((request) => request.url.endsWith("/delivery")).options.body);
  assert.deepEqual(payload.approved_sources, [{ artifact_path: second.artifact_path, sha256: SHA, source_identity_sha256: second.source_identity_sha256 }]);
  assert.deepEqual(payload.approved_sha256, [SHA]);
  assert.deepEqual(payload.scene_labels, { "input-0000": "Living room" });
});

function directDownloadFixture(t, extraFetch = null) {
  const before = globalThis.location;
  globalThis.location = { origin: "http://localhost" };
  t.after(() => { if (before === undefined) delete globalThis.location; else globalThis.location = before; });
  const base = "/v1/jobs/job-one/photographic-review";
  const h = fixture({ authMode: "direct_debug", extraFetch: (url, options) => {
    const custom = extraFetch?.(url, options);
    if (custom) return custom;
    if (url === base) return new Response(JSON.stringify({ success: true, data: { state_version: 1,
      scenes: [{ scene_id: "input-0000", label: "Living room", variants: [variant] }],
      decisions: [{ ...variant, status: "approved" }], groups: [], presets: [WEBSITE_PRESET],
    } }));
    if (url.endsWith("/delivery")) return new Response(JSON.stringify({ success: true, data: {
      download_url: `${base}/files/export/photos.zip`, filename: "photos.zip", sha256: ORIGINAL_SHA, size_bytes: 7,
      files: [{ filename: "inventory.json", url: `${base}/files/export/inventory.json` }],
    } }));
    if (url.includes("/files/")) return new Response("archive", { headers: { "Content-Type": "application/octet-stream", "Content-Length": "7", "X-TP-Artifact-SHA256": ORIGINAL_SHA } });
    return null;
  } });
  t.after(() => h.api.show(null, null));
  return h;
}

async function prepareDirectDownload(h) {
  h.api.show(h.job, variant); await settle();
  const checkbox = h.root.find("photographic-preset"); checkbox.checked = true; await checkbox.trigger("change");
  await h.root.find("photographic-deliver-approved").trigger();
  return h.root.find("photographic-delivery-download");
}

test("direct-debug thumbnails and streamed ZIP/companion downloads use authenticated headers and retain filenames", async (t) => {
  const previousPicker = globalThis.showSaveFilePicker;
  const saved = [];
  globalThis.showSaveFilePicker = async ({ suggestedName }) => {
    const record = { filename: suggestedName, chunks: [], closed: false };
    saved.push(record);
    const stream = new WritableStream({ write: (chunk) => record.chunks.push(chunk), close: () => { record.closed = true; } });
    return { createWritable: async () => stream };
  };
  t.after(() => { if (previousPicker === undefined) delete globalThis.showSaveFilePicker; else globalThis.showSaveFilePicker = previousPicker; });
  const h = directDownloadFixture(t);
  const link = await prepareDirectDownload(h);
  const thumbnail = h.requests.find((request) => request.url.includes("max_edge=256"));
  assert.equal(thumbnail.options.headers["x-api-key"], "fixture-key");
  await link.trigger();
  await h.root.find("photographic-delivery-companion").trigger();
  assert.deepEqual(saved.map((record) => [record.filename, record.closed, new TextDecoder().decode(record.chunks[0])]), [
    ["photos.zip", true, "archive"], ["inventory.json", true, "archive"],
  ]);
  const requests = h.requests.filter((request) => request.url.includes("/files/"));
  assert.equal(requests.length, 2);
  assert.ok(requests.every(({ options }) => options.headers["x-api-key"] === "fixture-key" && options.credentials === "same-origin" && options.cache === "no-store"));
  assert.ok(h.requests.every(({ url }) => !url.includes("fixture-key")));
});

test("direct-debug fallback refuses unknown or oversized lengths before reading a download body", async (t) => {
  const previousPicker = globalThis.showSaveFilePicker;
  delete globalThis.showSaveFilePicker;
  t.after(() => { if (previousPicker !== undefined) globalThis.showSaveFilePicker = previousPicker; });
  for (const length of [null, String(64 * 1024 * 1024 + 1)]) {
    let read = false;
    const h = directDownloadFixture(t, (url) => {
      if (!url.includes("/files/")) return null;
      return { ok: true, headers: new Headers({ ...(length ? { "Content-Length": length } : {}), "X-TP-Artifact-SHA256": ORIGINAL_SHA }),
        body: { getReader() { read = true; throw new Error("unbounded reader must not open"); } } };
    });
    await (await prepareDirectDownload(h)).trigger();
    assert.equal(read, false);
    assert.match(h.root.find("photographic-review-status").textContent, /known file size up to 64 MiB.*managed front door/);
    assert.equal(h.requests.find((request) => request.url.includes("/files/")).options.signal.aborted, true);
  }
});

test("bounded direct-debug fallback revokes blob URLs and cancels stale job downloads", async (t) => {
  const previousPicker = globalThis.showSaveFilePicker;
  delete globalThis.showSaveFilePicker;
  t.after(() => { if (previousPicker !== undefined) globalThis.showSaveFilePicker = previousPicker; });
  const revoked = t.mock.method(URL, "revokeObjectURL");
  const h = directDownloadFixture(t);
  await (await prepareDirectDownload(h)).trigger();
  const anchor = h.body.children[0];
  assert.equal(anchor.download, "photos.zip");
  assert.equal(anchor.clicked, true);
  assert.ok(anchor.href.startsWith("blob:"));
  h.api.show(null, null); await settle();
  assert.ok(revoked.mock.calls.some(({ arguments: args }) => args[0] === anchor.href));
  let resolveFetch;
  const pending = directDownloadFixture(t, (url) => url.includes("/files/") ? new Promise((resolve) => { resolveFetch = resolve; }) : null);
  const link = await prepareDirectDownload(pending);
  await link.trigger();
  const download = pending.requests.find((request) => request.url.includes("/files/"));
  pending.api.show(null, null);
  assert.equal(download.options.signal.aborted, true);
  resolveFetch(new Response("archive", { headers: { "Content-Length": "7", "X-TP-Artifact-SHA256": ORIGINAL_SHA } }));
  await settle();
  assert.equal(pending.body.children.length, 0, "an old job response cannot create a download in a new scope");
});

test("direct-debug download refuses a mismatched artifact hash and a stream exceeding its advertised size", async (t) => {
  const previousPicker = globalThis.showSaveFilePicker;
  delete globalThis.showSaveFilePicker;
  t.after(() => { if (previousPicker !== undefined) globalThis.showSaveFilePicker = previousPicker; });
  for (const [hash, message] of [[SHA, /Verified download unavailable/], [ORIGINAL_SHA, /exceeded its bounded size/]]) {
    const h = directDownloadFixture(t, (url) => url.includes("/files/") ? new Response("too long", {
      headers: { "Content-Length": "7", "X-TP-Artifact-SHA256": hash },
    }) : null);
    await (await prepareDirectDownload(h)).trigger();
    assert.match(h.root.find("photographic-review-status").textContent, message);
    assert.equal(h.body.children.length, 0);
    assert.equal(h.requests.find((request) => request.url.includes("/files/")).options.signal.aborted, true);
  }
});

function recoveredDelivery(body) {
  return { id: "committed", created_at: 1, approval_version: body.expected_version,
    request_correlation_id: body.request_correlation_id, request_fingerprint_schema: "tp.photo.delivery_intent.v1",
    request_fingerprint_sha256: SHA, intent: photographicDeliveryAttempt(body, [variant], [WEBSITE_PRESET]).intent,
    approval_at_creation: "recorded", current_recorded_approval: "approved", source_validation: "not_checked", file_validation: "not_checked",
    files: [{ filename: "recovered.zip", url: "/v1/jobs/job-one/photographic-review/files/committed/recovered.zip", sha256: SHA, size_bytes: 20, content_type: "application/zip" }] };
}

function reviewAfterRecovery(entry, stateVersion = entry.approval_version + 1) {
  return new Response(JSON.stringify({ success: true, data: { state_version: stateVersion,
    scenes: [{ scene_id: "input-0000", label: "Living room", variants: [variant, original] }],
    decisions: entry.intent.sources.map((source) => ({ ...source, status: "approved" })), groups: [], presets: [WEBSITE_PRESET] } }));
}

async function prepareExport(h) {
  h.api.show(h.job, variant); await settle();
  await h.root.find("photographic-decision-approved").trigger();
  const checkbox = h.root.find("photographic-preset"); checkbox.checked = true; await checkbox.trigger("change");
}

test("a lost export response refreshes review after GET recovery before the next explicit export", async (t) => {
  const previous = globalThis.location; globalThis.location = { origin: "http://localhost" };
  t.after(() => { globalThis.location = previous; });
  let committed = null;
  const submitted = [];
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  let finishCurrentReview;
  const currentReview = new Promise((resolve) => { finishCurrentReview = resolve; });
  let refreshReleased = false;
  const h = fixture({ sessionStorage: storage, extraFetch: (url, options) => {
    if (url.endsWith("/delivery") && options.method === "POST") {
      submitted.push(JSON.parse(options.body));
      if (submitted.length > 1) return new Response(JSON.stringify({ success: true, data: {
        state_version: 3, filename: "next.zip", download_url: "/v1/jobs/job-one/photographic-review/files/next/next.zip",
        sha256: SHA, size_bytes: 20 } }));
      committed = recoveredDelivery(submitted[0]);
      return Promise.reject(new TypeError("Response lost after commit"));
    }
    if (committed && url.endsWith("/photographic-review")) return refreshReleased
      ? reviewAfterRecovery(committed, submitted.length > 1 ? 3 : 2) : currentReview;
    if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: {
      state_version: committed ? 2 : 0, snapshot_count: committed ? 1 : 0, items: committed ? [committed] : [], next_cursor: null } }));
    return null;
  } });
  await prepareExport(h);
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(h.root.find("photographic-deliver-approved").disabled, true, "A resolved history lookup must wait for fresh review metadata");
  assert.match(h.root.find("photographic-delivery-history-status").textContent, /Refreshing current review before another export/);
  assert.deepEqual(JSON.parse(values.get("tp.photo.delivery_recovery.v1"))[0].attempt.body, submitted[0]);
  refreshReleased = true;
  finishCurrentReview(reviewAfterRecovery(committed));
  await settle();
  assert.equal(h.requests.filter((request) => request.url.endsWith("/delivery")).length, 1);
  assert.match(h.root.find("photographic-delivery-history-status").textContent, /Recovered.*original request.*No export was retried/);
  assert.match(h.root.find("photographic-delivery-history-download").href, /committed\/recovered.zip/);
  assert.equal(h.root.find("photographic-deliver-approved").disabled, false);
  assert.equal(submitted[0].expected_version, 1);
  const afterRecovery = h.requests.length;
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(submitted.length, 2);
  assert.equal(submitted[1].expected_version, 2);
  assert.notEqual(submitted[1].request_correlation_id, submitted[0].request_correlation_id);
  assert.equal(submitted[0].expected_version, 1, "The frozen original attempt must never advance");
  assert.equal(h.requests.slice(afterRecovery).filter((request) => request.options.method === "POST").length, 1);
  assert.match(h.root.find("photographic-delivery-download").href, /next\/next.zip/);
});

for (const failure of ["unavailable", "stale version"]) {
  test(`recovered export stays blocked when current review is ${failure} until a fresh GET succeeds`, async () => {
    const values = new Map();
    const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
    let committed = null;
    let fresh = false;
    const h = fixture({ sessionStorage: storage, extraFetch: (url, options) => {
      if (url.endsWith("/delivery") && options.method === "POST") {
        committed = recoveredDelivery(JSON.parse(options.body));
        return Promise.reject(new TypeError("Response lost after commit"));
      }
      if (committed && url.endsWith("/photographic-review")) {
        if (fresh) return reviewAfterRecovery(committed);
        return failure === "stale version" ? reviewAfterRecovery(committed, committed.approval_version)
          : new Response(JSON.stringify({ success: false, error: { message: "Review temporarily unavailable" } }), { status: 503 });
      }
      if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: {
        state_version: committed ? 2 : 0, snapshot_count: committed ? 1 : 0, items: committed ? [committed] : [], next_cursor: null } }));
      return null;
    } });
    await prepareExport(h);
    await h.root.find("photographic-deliver-approved").trigger();
    assert.match(h.root.find("photographic-delivery-history-status").textContent, /Recovered.*Current review refresh failed.*Export remains blocked/);
    assert.equal(h.root.find("photographic-deliver-approved").disabled, true);
    assert.equal(JSON.parse(values.get("tp.photo.delivery_recovery.v1"))[0].attempt.body.expected_version, 1);
    await h.root.find("photographic-deliver-approved").trigger();
    assert.equal(h.requests.filter((request) => request.url.endsWith("/delivery")).length, 1);
    fresh = true;
    const beforeFreshGet = h.requests.length;
    await h.root.find("photographic-delivery-reconcile").trigger();
    assert.equal(h.root.find("photographic-deliver-approved").disabled, false);
    assert.match(h.root.find("photographic-delivery-history-status").textContent, /Recovered.*No export was retried/);
    assert.equal(h.requests.slice(beforeFreshGet).every((request) => request.options.method !== "POST"), true);
    assert.deepEqual(JSON.parse(values.get("tp.photo.delivery_recovery.v1")), []);
  });
}

test("an unknown export preserves its original version and blocks another POST through refresh and reconciliation", async () => {
  let sent = null;
  const h = fixture({ extraFetch: (url, options) => {
    if (url.endsWith("/delivery") && options.method === "POST") { sent = JSON.parse(options.body); return Promise.reject(new TypeError("network lost")); }
    if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: { state_version: sent?.expected_version ?? 0, snapshot_count: 0, items: [], next_cursor: null } }));
    return null;
  } });
  await prepareExport(h);
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(sent.expected_version, 1);
  assert.equal(h.root.find("photographic-deliver-approved").disabled, true);
  assert.match(h.root.find("photographic-delivery-history-status").textContent, /still unknown.*no automatic export retry/);
  await h.root.find("photographic-review-refresh").trigger();
  await h.root.find("photographic-delivery-reconcile").trigger();
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(h.requests.filter((request) => request.url.endsWith("/delivery")).length, 1);
  assert.equal(sent.expected_version, 1);
});

test("reloading restores a scoped unknown attempt and reconciles without issuing any POST", async () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  let sent = null;
  const first = fixture({ sessionStorage: storage, extraFetch: (url, options) => {
    if (url.endsWith("/delivery") && options.method === "POST") { sent = JSON.parse(options.body); return Promise.reject(new TypeError("lost")); }
    if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: { state_version: sent?.expected_version ?? 0, snapshot_count: 0, items: [], next_cursor: null } }));
    return null;
  } });
  await prepareExport(first); await first.root.find("photographic-deliver-approved").trigger();
  const entry = recoveredDelivery(sent);
  const second = fixture({ sessionStorage: storage, extraFetch: (url) => url.includes("/deliveries?")
    ? new Response(JSON.stringify({ success: true, data: { state_version: 2, snapshot_count: 1, items: [entry], next_cursor: null } }))
    : url.endsWith("/photographic-review") ? reviewAfterRecovery(entry) : null });
  second.api.show(second.job, variant); await settle();
  assert.equal(second.requests.filter((request) => request.options.method === "POST").length, 0);
  assert.match(second.root.find("photographic-delivery-history-status").textContent, /Recovered/);
  assert.deepEqual(JSON.parse(values.get("tp.photo.delivery_recovery.v1")), []);
});

test("history is visible independently of a pending catalogue and never renders withdrawn download links", async () => {
  let finish;
  const pendingCatalogue = new Promise((resolve) => { finish = resolve; });
  const entry = { id: "old", approval_at_creation: "recorded", current_recorded_approval: "withdrawn", files: [
    { filename: "old.zip", url: "/v1/jobs/job-one/photographic-review/files/old/old.zip", sha256: SHA } ] };
  const h = fixture({ extraFetch: (url) => {
    if (url.endsWith("/photographic-review")) return pendingCatalogue;
    if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: { state_version: 2, snapshot_count: 1, items: [entry], next_cursor: null } }));
    return null;
  } });
  h.api.show(h.job, variant); await settle();
  assert.ok(h.root.find("photographic-delivery-history-entry"));
  assert.equal(h.root.find("photographic-delivery-history-download"), undefined);
  finish(new Response(JSON.stringify({ success: true, data: { state_version: 2, scenes: [], decisions: [], presets: [] } })));
  await settle();
});

test("late export transport failure cannot regress a delivery already recovered while the response was pending", async () => {
  let rejectResponse;
  let committed = null;
  const deferred = new Promise((_resolve, reject) => { rejectResponse = reject; });
  const h = fixture({ extraFetch: (url, options) => {
    if (url.endsWith("/delivery") && options.method === "POST") { committed = recoveredDelivery(JSON.parse(options.body)); return deferred; }
    if (committed && url.endsWith("/photographic-review")) return reviewAfterRecovery(committed);
    if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: {
      state_version: committed ? 2 : 0, snapshot_count: committed ? 1 : 0, items: committed ? [committed] : [], next_cursor: null } }));
    return null;
  } });
  await prepareExport(h);
  await h.root.find("photographic-deliver-approved").trigger();
  await h.root.find("photographic-delivery-reconcile").trigger();
  assert.match(h.root.find("photographic-delivery-history-status").textContent, /Recovered/);
  rejectResponse(new TypeError("Late transport failure")); await settle();
  assert.match(h.root.find("photographic-delivery-history-status").textContent, /Recovered/);
  assert.equal(h.root.find("photographic-deliver-approved").disabled, false);
  assert.equal(h.requests.filter((request) => request.url.endsWith("/delivery")).length, 1);
});

test("disabled history UI retains the legacy export payload and does not call the history API", async () => {
  const h = fixture({ historyEnabled: false });
  await prepareExport(h);
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(h.requests.some((request) => request.url.includes("/deliveries?")), false);
  const sent = JSON.parse(h.requests.find((request) => request.url.endsWith("/delivery")).options.body);
  assert.equal(Object.hasOwn(sent, "request_correlation_id"), false);
  assert.equal(h.root.find("photographic-delivery-history-controls").classList.contains("hidden"), true);
});

for (const [name, storage] of [
  ["unavailable", null],
  ["throwing read", { getItem() { throw new Error("Disabled"); }, setItem() {} }],
  ["quota rejected", { getItem() { return null; }, setItem() { throw new Error("QuotaExceededError"); } }],
  ["write not retained", { getItem() { return null; }, setItem() {} }],
]) {
  test(`history-enabled export refuses ${name} session recovery before any POST and remains safe after reload`, async () => {
    const first = fixture({ sessionStorage: storage });
    await prepareExport(first);
    await first.root.find("photographic-deliver-approved").trigger();
    assert.equal(first.requests.some((request) => request.url.endsWith("/delivery")), false);
    assert.match(first.root.find("photographic-review-status").textContent, /Session recovery storage.*No export was submitted/);
    const reloaded = fixture({ sessionStorage: storage });
    await prepareExport(reloaded);
    await reloaded.root.find("photographic-deliver-approved").trigger();
    assert.equal(reloaded.requests.some((request) => request.url.endsWith("/delivery")), false);
  });
}

test("a full bounded recovery catalogue blocks a new export before POST", async () => {
  const entries = Array.from({ length: 16 }, (_, index) => ({
    scope: JSON.stringify([`other-${index}`, JSON.stringify(["managed", null, null])]),
    attempt: photographicDeliveryAttempt({ property_name: "Earlier", expected_version: 0, request_correlation_id: `prior-${index}`,
      approved_sources: [{ artifact_path: variant.artifact_path, sha256: SHA, source_identity_sha256: SHA }], presets: ["website"], scene_labels: {} }, [variant], [WEBSITE_PRESET]) }));
  const values = new Map([["tp.photo.delivery_recovery.v1", JSON.stringify(entries)]]);
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  const h = fixture({ sessionStorage: storage });
  await prepareExport(h);
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(h.requests.some((request) => request.url.endsWith("/delivery")), false);
  assert.match(h.root.find("photographic-review-status").textContent, /storage is unavailable or full.*No export was submitted/);
  assert.deepEqual(JSON.parse(values.get("tp.photo.delivery_recovery.v1")), entries);
});

for (const failure of ["throw", "invalid JSON", "invalid schema"]) {
  test(`a restored ${failure} recovery store cannot hide an earlier attempt or overwrite it before GET reconciliation`, async () => {
    const values = new Map();
    let unreadable = false;
    const storage = { getItem: (key) => {
      if (unreadable && failure === "throw") throw new Error("Temporarily unreadable");
      if (unreadable) return failure === "invalid JSON" ? "{" : JSON.stringify([{ scope: null, attempt: {} }]);
      return values.get(key);
    }, setItem: (key, value) => values.set(key, value) };
    let original = null;
    const first = fixture({ sessionStorage: storage, extraFetch: (url, options) => {
      if (url.endsWith("/delivery") && options.method === "POST") { original = JSON.parse(options.body); return Promise.reject(new TypeError("Unknown response")); }
      if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: {
        state_version: original?.expected_version ?? 0, items: [], next_cursor: null } }));
      return null;
    } });
    await prepareExport(first); await first.root.find("photographic-deliver-approved").trigger();
    const retained = values.get("tp.photo.delivery_recovery.v1");
    const entry = recoveredDelivery(original);
    unreadable = true;
    const second = fixture({ sessionStorage: storage, extraFetch: (url) => {
      if (url.endsWith("/photographic-review")) return reviewAfterRecovery(entry);
      if (url.includes("/deliveries?")) return new Response(JSON.stringify({ success: true, data: {
        state_version: 2, items: [entry], next_cursor: null } }));
      return null;
    } });
    second.api.show(second.job, variant); await settle();
    const checkbox = second.root.find("photographic-preset"); checkbox.checked = true; await checkbox.trigger("change");
    assert.equal(second.root.find("photographic-deliver-approved").disabled, true);
    await second.root.find("photographic-deliver-approved").trigger();
    assert.equal(values.get("tp.photo.delivery_recovery.v1"), retained);
    assert.equal(second.requests.some((request) => request.options.method === "POST"), false);
    unreadable = false;
    await second.root.find("photographic-deliver-approved").trigger();
    assert.equal(second.requests.some((request) => request.options.method === "POST"), false, "Restoring reads alone must not authorize an export");
    await second.root.find("photographic-delivery-reconcile").trigger();
    assert.match(second.root.find("photographic-delivery-history-status").textContent, /Recovered.*No export was retried/);
    assert.equal(second.requests.some((request) => request.options.method === "POST"), false);
    assert.equal(original.expected_version, 1);
    assert.equal(second.requests.some((request) => request.url.includes(`request_correlation_id=${original.request_correlation_id}`)), true);
    assert.deepEqual(JSON.parse(values.get("tp.photo.delivery_recovery.v1")), []);
    assert.equal(second.root.find("photographic-deliver-approved").disabled, false);
  });
}

test("an active local cooldown creates no delivery attempt or POST and expiry permits an explicit fresh export", async (t) => {
  let now = 100000;
  t.mock.method(Date, "now", () => now);
  let correlations = 0;
  t.mock.method(globalThis.crypto, "randomUUID", () => `explicit-after-cooldown-${++correlations}`);
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  let submitted = null;
  const h = fixture({ sessionStorage: storage, extraFetch: (url, options) => {
    if (url.includes("/quality?")) return new Response("", { status: 429, headers: { "Retry-After": "2" } });
    if (url.endsWith("/delivery") && options.method === "POST") {
      submitted = JSON.parse(options.body);
      return new Response(JSON.stringify({ success: true, data: { state_version: 2, filename: "explicit.zip",
        download_url: "/v1/jobs/job-one/photographic-review/files/new/explicit.zip", sha256: SHA, size_bytes: 20 } }));
    }
    return null;
  } });
  await prepareExport(h);
  await h.root.find("photographic-measure-quality").trigger();
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(correlations, 0);
  assert.equal(values.has("tp.photo.delivery_recovery.v1"), false);
  assert.equal(submitted, null);
  assert.match(h.root.find("photographic-review-status").textContent, /Rate limited.*Retry after/);
  now += 2001;
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(correlations, 1);
  assert.equal(submitted.expected_version, 1);
  assert.equal(submitted.request_correlation_id, "explicit-after-cooldown-1");
  assert.equal(h.requests.filter((request) => request.url.endsWith("/delivery")).length, 1);
});
