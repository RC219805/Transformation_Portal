import test from "node:test";
import assert from "node:assert/strict";
import {
  createPhotographicReview, photographicVariantLabel, photographicProperties,
  photographicDecisionMatches, nativeCropGeometry, photographicAssetUrl, photographicRecipe, photographicPreviewMatches, photographicRetryAfterMs,
} from "../portal-src/photographic-review.js";

const SHA = "a".repeat(64);
const ORIGINAL_SHA = "b".repeat(64);
const variant = { role: "final", artifact_path: "input-0000/delivery.tif", sha256: SHA,
  width: 8000, height: 6000, bit_depth: 16, color_profile: "sRGB", intended_use: "Photographic master" };
const original = { ...variant, role: "original", artifact_path: "original/input-0000/source.tif", sha256: ORIGINAL_SHA };

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
function fixture({ responseHash = null, mutationError = false, extraFetch = null, renderDebounceMs = 0 } = {}) {
  const root = new Element("main");
  const requests = [];
  let saved = [];
  const data = () => ({ state_version: saved.length, scenes: [{ scene_id: "input-0000", label: "Living room", variants: [variant, original] }],
    decisions: saved, groups: [], presets: [{ name: "website", intended_use: "Website delivery" }] });
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
        "X-TP-Native-Width": params.get("width") || "1600", "X-TP-Native-Height": params.get("height") || "1200",
      } });
    }
    return new Response(JSON.stringify({ success: true, data: data() }));
  };
  const api = createPhotographicReview({ state: {}, els: { artifactShellContent: root }, document: { createElement: (tag) => new Element(tag) }, fetchImpl, renderDebounceMs,
    _buildAuthHeaders: (headers, method) => ({ ...headers, ...(method === "POST" ? { "X-CSRF-Token": "fixture-csrf" } : { "x-api-key": "fixture-key" }) }) });
  const job = { id: "job-one", pipeline: "lux-depth", artifacts: [{ ...variant, path: variant.artifact_path }] };
  return { root, requests, api, job };
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

test("saved human decisions use the exact selected master, state version, and managed CSRF transport", async () => {
  const h = fixture();
  h.api.show(h.job, variant); await settle();
  h.root.find("photographic-decision-note").value = "Window edges and sky inspected.";
  await h.root.find("photographic-decision-approved").trigger();
  const mutation = h.requests.find((request) => request.url.endsWith("/decisions"));
  assert.deepEqual(JSON.parse(mutation.options.body), { artifact_path: variant.artifact_path, sha256: SHA,
    source_identity_sha256: SHA, status: "approved", note: "Window edges and sky inspected.", expected_version: 0 });
  assert.equal(mutation.options.headers["X-CSRF-Token"], "fixture-csrf");
  assert.match(h.root.find("photographic-decision-status").textContent, /Approved.*exact SHA-256/);
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
      groups: [], presets: ["website"],
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
    decisions: [first, second].map((source) => ({ ...source, status: "approved" })), groups: [], presets: ["website"],
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
