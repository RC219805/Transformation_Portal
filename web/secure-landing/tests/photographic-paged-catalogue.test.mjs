import test from "node:test";
import assert from "node:assert/strict";
import { createPhotographicReview } from "../portal-src/photographic-review.js";
import { createPhotographicPagedCatalogue, photographicSelectionStorage } from "../portal-src/photographic-review-support.js";

const CATALOGUE_SHA = "c".repeat(64);
const NEXT_CATALOGUE_SHA = "d".repeat(64);
const PAGE_SIZE = 8;
const WEBSITE_PRESET = { schema: "tp.photo.export_preset.v1", name: "website", format: "jpeg", intended_use: "Website",
  long_edge: 2560, jpeg_quality: 92, sharpen_amount: 0.2, render: "perceptual_srgb", shoulder: 0.8 };

class Element {
  constructor(tag, ownerDocument) {
    this.tagName = tag;
    this.ownerDocument = ownerDocument;
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
  get disabled() { return this._disabled || false; }
  set disabled(value) {
    this._disabled = Boolean(value);
    if (this._disabled && this.ownerDocument?.activeElement === this) this.ownerDocument.activeElement = this.ownerDocument.body;
  }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) {
    if (this.ownerDocument && this.ownerDocument.activeElement !== this && this.contains(this.ownerDocument.activeElement)) this.ownerDocument.activeElement = this.ownerDocument.body;
    this.children = [...items];
  }
  contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
  focus() { this.ownerDocument.activeElement = this; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  removeAttribute(key) { delete this.attributes[key]; if (key === "src") this.src = ""; }
  remove() { this.removed = true; }
  click() { this.clicked = true; }
  addEventListener(key, handler) { this.listeners[key] = handler; }
  querySelectorAll(selector) {
    const key = selector.match(/^\[data-(.+)\]$/)?.[1]?.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    const all = this.children.flatMap((child) => [child, ...child.querySelectorAll(selector)]);
    return all.filter((child) => key && Object.hasOwn(child.dataset, key));
  }
  findAll(ui) { return [this, ...this.children.flatMap((child) => child.findAll(ui))].filter((child) => child.dataset.ui === ui); }
  find(ui) { return this.findAll(ui)[0]; }
  async trigger(event = "click", fields = {}) {
    this.listeners[event]?.({ preventDefault() {}, key: "", ...fields });
    await settle();
  }
}

async function settle() { for (let count = 0; count < 12; count += 1) await new Promise((resolve) => setImmediate(resolve)); }
function json(data, status = 200) { return new Response(JSON.stringify({ success: true, data }), { status }); }
function error(code, message, status = 409) { return new Response(JSON.stringify({ success: false, error: { code, message } }), { status }); }
function storageFixture() {
  const values = new Map();
  return { values, storage: { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: (key) => values.delete(key) } };
}
function makeScene(index) {
  const sceneId = `input-${String(index).padStart(4, "0")}`;
  const variant = { role: "final", artifact_path: `${sceneId}/delivery.tif`, sha256: (index + 1).toString(16).padStart(64, "0"),
    width: 8000, height: 6000, bit_depth: 16, color_profile: "sRGB", intended_use: "Photographic master" };
  const original = { ...variant, role: "original", artifact_path: `original/${sceneId}/source.tif`, sha256: (index + 101).toString(16).padStart(64, "0") };
  return { scene_id: sceneId, label: `Room ${String(50 - index).padStart(2, "0")}`, variants: [variant, original] };
}

function fixture({ sceneCount = 50, paged = true, actor = "actor-one", jobId = "job-one", sessionStorage,
  extraFetch, restoredSelection = null } = {}) {
  const root = new Element("main");
  const body = new Element("body");
  const document = { body, activeElement: body, createElement: (tag) => new Element(tag, document) };
  root.ownerDocument = document; body.ownerDocument = document;
  const state = { auth: { mode: "managed", actor: { username: actor }, tenant: "tenant-one", features: { photographicPagedCatalogue: paged } } };
  const requests = [];
  const selections = [];
  const scenes = Array.from({ length: sceneCount }, (_, index) => makeScene(index));
  const decisions = scenes.filter((_scene, index) => index % 3 !== 0).map((scene, index) => ({
    artifact_path: scene.variants[0].artifact_path, sha256: scene.variants[0].sha256, source_identity_sha256: scene.variants[0].sha256,
    status: index % 2 ? "needs_revision" : "approved", note: "", scene_id: scene.scene_id, label: scene.label, role: "final",
  }));
  const server = { catalogueIdentitySha256: CATALOGUE_SHA, stateVersion: 4, failedScene: "", failedPage: false,
    decisions, groups: [], presets: [WEBSITE_PRESET] };
  const publicState = () => ({ state_version: server.stateVersion, decisions: server.decisions, groups: server.groups, presets: server.presets });
  const summary = (scene) => ({ scene_id: scene.scene_id, label: scene.label,
    review_status: server.decisions.find((decision) => decision.scene_id === scene.scene_id)?.status || "unreviewed",
    variant_count: scene.variants.length, recorded_decision: server.decisions.find((decision) => decision.scene_id === scene.scene_id) || null,
    variant_paths: scene.variants.map((variant) => variant.artifact_path),
    related_artifacts: [...scene.variants.filter((variant) => variant.role !== "original").map((variant) => ({ role: "master", artifact_path: variant.artifact_path })),
      { role: "depth", artifact_path: `${scene.scene_id}/depth.npy` }, { role: "evidence", artifact_path: `${scene.scene_id}/provenance.json` }],
  });
  const page = (params) => {
    const filter = params.get("status") || "all";
    const compare = (left, right) => Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
    const key = params.get("sort") || "scene_id";
    const filtered = scenes.map(summary).filter((scene) => filter === "all" || scene.review_status === filter)
      .sort((left, right) => compare(left[key], right[key]) || compare(left.scene_id, right.scene_id));
    const offset = Number(params.get("cursor") || 0);
    const limit = Number(params.get("limit") || PAGE_SIZE);
    return { schema: "tp.photo.scene_page.v1", catalogue_identity_sha256: server.catalogueIdentitySha256,
      ...publicState(), scenes: filtered.slice(offset, offset + limit), total_scenes: scenes.length, total_filtered: filtered.length,
      next_cursor: offset + limit < filtered.length ? String(offset + limit) : null };
  };
  const detail = (sceneId) => ({ schema: "tp.photo.scene_detail.v1", catalogue_identity_sha256: server.catalogueIdentitySha256,
    ...publicState(), decisions: server.decisions.filter((row) => row.scene_id === sceneId), scenes: scenes.filter((scene) => scene.scene_id === sceneId) });
  const fetchImpl = async (url, options) => {
    const parsed = new URL(url, "http://localhost");
    const request = { url, options, path: parsed.pathname, params: parsed.searchParams };
    requests.push(request);
    const extra = extraFetch?.(request, { server, page, detail });
    if (extra) return extra;
    if (options.method === "POST" && parsed.pathname.endsWith("/decisions")) {
      const body = JSON.parse(options.body);
      const scene = scenes.find((item) => item.variants.some((variant) => variant.artifact_path === body.artifact_path));
      server.decisions = [...server.decisions.filter((decision) => decision.artifact_path !== body.artifact_path), { ...body, scene_id: scene.scene_id }];
      server.stateVersion += 1; server.catalogueIdentitySha256 = NEXT_CATALOGUE_SHA;
      return json({ saved: true, state_version: server.stateVersion });
    }
    if (parsed.pathname.endsWith("/scenes")) {
      if (server.failedPage) return error("CATALOGUE_CHANGED", "Catalogue changed; reset pagination.");
      return json(page(parsed.searchParams));
    }
    if (parsed.pathname.endsWith("/scene")) {
      if (server.failedScene === parsed.searchParams.get("scene_id")) return error("SCENE_UNAVAILABLE", "Scene detail unavailable.", 503);
      if (parsed.searchParams.get("catalogue_identity_sha256") !== server.catalogueIdentitySha256) return error("CATALOGUE_CHANGED", "Catalogue changed; reset pagination.");
      return json(detail(parsed.searchParams.get("scene_id")));
    }
    if (parsed.pathname.endsWith("/render") || parsed.pathname.endsWith("/native-crop")) {
      return new Response(new Blob(["png"], { type: "image/png" }), { headers: {
        "X-TP-Source-SHA256": parsed.searchParams.get("sha256"), "X-TP-Presentation-View": parsed.searchParams.get("view") || "clip_srgb",
        "X-TP-Source-Identity-SHA256": parsed.searchParams.get("source_identity_sha256") || parsed.searchParams.get("sha256"),
        "X-TP-Presentation-Policy-SHA256": "none", "X-TP-Native-Width": parsed.searchParams.get("width") || "1600",
        "X-TP-Native-Height": parsed.searchParams.get("height") || "1200",
      } });
    }
    if (parsed.pathname.endsWith("/photographic-review")) {
      assert.notEqual(state.auth.features.photographicPagedCatalogue, true, "Paged review must not fetch the unbounded legacy catalogue");
      return json({ ...publicState(), scenes });
    }
    throw new Error(`Unexpected photographic request: ${url}`);
  };
  const recovery = sessionStorage === undefined ? storageFixture().storage : sessionStorage;
  const api = createPhotographicReview({ state, els: { artifactShellContent: root }, document,
    fetchImpl, renderDebounceMs: 0, sessionStorage: recovery,
    _buildAuthHeaders: (headers, method) => ({ ...headers, ...(method === "POST" ? { "X-CSRF-Token": "fixture-csrf" } : { "x-api-key": "fixture-key" }) }),
    onPhotographicSelection: (selection) => selections.push(selection), getPhotographicSelection: () => restoredSelection,
  });
  const job = { id: jobId, pipeline: "lux-depth", artifacts: scenes.map((scene) => ({ ...scene.variants[0], path: scene.variants[0].artifact_path })) };
  return { root, document, state, requests, selections, scenes, server, api, job, page, detail };
}

function requested(h, suffix) { return h.requests.filter((request) => request.path.endsWith(suffix)); }
function selected(h) { return h.root.find("photographic-scene").value; }

test("concurrent refreshes coalesce the selected catalogue read and finish with a ready status", async (t) => {
  let finish;
  const h = fixture({ extraFetch: (request, response) => request.path.endsWith("/scenes")
    ? new Promise((resolve) => { finish = () => resolve(json(response.page(request.params))); }) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]);
  const refreshes = [h.api.refresh(), h.api.refresh()];
  assert.equal(requested(h, "/scenes").length, 1);
  finish(); await Promise.all(refreshes); await settle();
  assert.equal(requested(h, "/scenes").length, 1);
  assert.equal(requested(h, "/scene").length, 1);
  assert.equal(h.root.find("photographic-page-status").dataset.state, "ready");
  assert.match(h.root.find("photographic-review-status").textContent, /Review refreshed/);
});

test("next unreviewed follows the selected name sort across pages", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const sort = h.root.find("photographic-scene-sort"); sort.value = "label"; await sort.trigger("change");
  await h.root.find("photographic-next-unreviewed").trigger();
  assert.equal(selected(h), h.scenes[48].scene_id, "Last sorted unreviewed photograph wraps to the first sorted unreviewed photograph");
  await h.root.find("photographic-next-unreviewed").trigger();
  assert.equal(selected(h), h.scenes[45].scene_id);
});

for (const tieBreak of [false, true]) {
  test(`next unreviewed follows server code-point ${tieBreak ? "scene ID tie-break" : "label"} order and wraps across pages`, async (t) => {
    const h = fixture({ sceneCount: 10 });
    h.server.decisions = [];
    h.scenes.forEach((scene, index) => { scene.label = tieBreak ? "Same room" : `A Room ${index}`; });
    if (tieBreak) {
      for (const [index, id] of [[8, "１-room"], [9, "🏠-room"]]) {
        h.scenes[index].scene_id = id;
        h.scenes[index].variants[0].artifact_path = `${id}/delivery.tif`;
      }
    } else {
      h.scenes[8].label = "１ Kitchen"; h.scenes[9].label = "🏠 Exterior";
    }
    t.after(() => h.api.show(null, null));
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    await h.root.find("photographic-page-next").trigger();
    const sort = h.root.find("photographic-scene-sort"); sort.value = "label"; await sort.trigger("change");
    assert.equal(selected(h), h.scenes[8].scene_id);
    const before = requested(h, "/scene").length;
    await h.root.find("photographic-next-unreviewed").trigger();
    assert.equal(selected(h), h.scenes[9].scene_id, "Astral text follows fullwidth text in Python's code-point order");
    assert.match(h.root.find("photographic-page-status").textContent, /page 2/);
    await h.root.find("photographic-next-unreviewed").trigger();
    assert.equal(selected(h), h.scenes[0].scene_id, "The last server-ordered photograph wraps to the first page");
    assert.match(h.root.find("photographic-page-status").textContent, /page 1/);
    assert.equal(requested(h, "/scene").length, before + 2, "Search opens only each selected rich scene");
  });
}

test("a same-inventory rail selection locates a relative artifact without a scene identifier", async (t) => {
  const restoredSelection = { sceneId: "input-0000", artifactPath: "input-0000/delivery.tif" };
  const h = fixture({ restoredSelection });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const previousDetails = requested(h, "/scene").length;
  restoredSelection.sceneId = "";
  restoredSelection.artifactPath = h.scenes[31].variants[1].artifact_path;
  h.api.show(h.job, h.scenes[31].variants[1]); await settle();
  assert.equal(selected(h), h.scenes[31].scene_id);
  assert.equal(h.root.find("photographic-primary").value, h.scenes[31].variants[1].artifact_path);
  assert.equal(requested(h, "/scene").length, previousDetails + 1);
  assert.equal(h.root.find("photographic-contact-sheet").findAll("photographic-contact-item").some((node) => node.attributes["aria-pressed"] === "true"), true);
});

test("selection recovery validates raw selectors without interpreting filename percent escapes", () => {
  const storage = storageFixture();
  const selections = photographicSelectionStorage(storage.storage);
  for (const path of ["/etc/passwd", "input/../private", "https://other.example/a", "a\\b", "input/\x00secret", "input/./secret", "input//secret"]) {
    selections.set("scope", { sceneId: "input-0000", primaryPath: path, comparePath: path, apiKey: "should-never-persist" });
    const restored = selections.get("scope");
    assert.equal(restored.primaryPath, ""); assert.equal(restored.comparePath, "");
    assert.equal([...storage.values.values()].some((raw) => raw.includes("should-never-persist")), false);
  }
  for (const path of ["room_100%_enhanced.tif", "room:west_enhanced.tif", "input/%2e%2e/private", "a%5cb", "input/%00secret"]) {
    selections.set("scope", { sceneId: path, primaryPath: path, comparePath: path });
    const restored = selections.get("scope");
    assert.equal(restored.sceneId, path); assert.equal(restored.primaryPath, path); assert.equal(restored.comparePath, path);
  }
});

for (const sceneId of ["room_100%_enhanced.tif", "room_%00_enhanced.tif", "room:west_enhanced.tif"]) {
  test(`raw photograph filename ${sceneId} loads without a second URI decode`, async (t) => {
    const h = fixture({ sceneCount: 1 });
    h.scenes[0].scene_id = sceneId;
    h.scenes[0].variants[0].artifact_path = sceneId;
    h.scenes[0].variants[1].artifact_path = `original/${sceneId}`;
    t.after(() => h.api.show(null, null));
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    assert.equal(selected(h), sceneId);
    assert.equal(h.root.find("photographic-primary").value, sceneId);
    assert.equal(requested(h, "/scene").at(-1).params.get("scene_id"), sceneId);
    assert.equal(h.selections.at(-1).artifactPath, sceneId);
    assert.ok(requested(h, "/scene").at(-1).url.includes(new URLSearchParams({ scene_id: sceneId }).toString()));
  });
}

test("paged catalogue rollout defaults to the existing photographic review contract", async (t) => {
  const h = fixture({ paged: false });
  delete h.state.auth.features.photographicPagedCatalogue;
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]);
  await settle();
  assert.equal(requested(h, "/photographic-review").length, 1);
  assert.equal(requested(h, "/scenes").length, 0);
  assert.equal(requested(h, "/scene").length, 0);
  assert.equal(h.root.find("photographic-primary").value, h.scenes[0].variants[0].artifact_path);
});

test("fifty photographs traverse bounded metadata pages and load rich detail only for the selected scene", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]);
  await settle();
  const firstPage = requested(h, "/scenes")[0];
  assert.equal(firstPage.params.get("limit"), "8");
  assert.equal(firstPage.params.get("status"), "all");
  assert.equal(firstPage.params.get("sort"), "scene_id");
  assert.equal(firstPage.options.credentials, "same-origin");
  assert.equal(firstPage.options.headers["x-api-key"], "fixture-key");
  assert.equal(requested(h, "/scene").length, 1);
  assert.equal(h.root.findAll("photographic-contact-item").length, PAGE_SIZE);
  for (let index = 0; index < h.scenes.length; index += 1) {
    assert.equal(selected(h), h.scenes[index].scene_id);
    assert.equal(h.root.find("photographic-primary").value, h.scenes[index].variants[0].artifact_path);
    assert.equal(h.root.find("photographic-primary").options.length, 2);
    assert.ok(h.root.findAll("photographic-contact-item").length <= PAGE_SIZE);
    assert.equal(h.root.find("photographic-contact-sheet").findAll("photographic-primary-image").length, 0);
    assert.equal(requested(h, "/scene").at(-1).params.get("scene_id"), h.scenes[index].scene_id);
    if (index + 1 < h.scenes.length) await h.root.find("photographic-scene-next").trigger();
  }
  assert.equal(requested(h, "/scenes").length, Math.ceil(h.scenes.length / PAGE_SIZE));
  assert.equal(requested(h, "/scene").length, h.scenes.length);
  assert.equal(requested(h, "/photographic-review").length, 0);
  assert.match(h.root.find("photographic-page-status").textContent, /50 of 50 photographs.*page 7/);
  const related = h.root.findAll("photographic-related-artifacts");
  assert.equal(related.length, 2);
  assert.match(related[0].children.map((child) => child.textContent).join(" "), /master:.*delivery.tif.*depth:.*depth.npy.*evidence:.*provenance.json/);
  const details = requested(h, "/scene");
  assert.ok(details.every((request) => request.params.get("catalogue_identity_sha256") === CATALOGUE_SHA));
  assert.equal(h.selections.at(-1).jobId, h.job.id);
  assert.equal(h.selections.at(-1).sceneId, h.scenes.at(-1).scene_id);
  assert.equal(h.selections.at(-1).artifactPath, h.scenes.at(-1).variants[0].artifact_path);
  assert.equal(h.selections.at(-1).stateVersion, 4);
  assert.equal(h.selections.at(-1).catalogueIdentitySha256, CATALOGUE_SHA);
});

test("page and scene navigation cross both boundaries without materializing adjacent rich scenes", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-page-next").trigger();
  assert.equal(selected(h), h.scenes[8].scene_id);
  assert.equal(requested(h, "/scenes").at(-1).params.get("cursor"), "8");
  await h.root.find("photographic-scene-previous").trigger();
  assert.equal(selected(h), h.scenes[7].scene_id);
  assert.equal(requested(h, "/scenes").at(-1).params.get("cursor"), null);
  await h.root.find("photographic-page-next").trigger();
  await h.root.find("photographic-page-previous").trigger();
  assert.equal(selected(h), h.scenes[0].scene_id);
  assert.deepEqual(requested(h, "/scene").map((request) => request.params.get("scene_id")), ["input-0000", "input-0008", "input-0007", "input-0008", "input-0000"]);
});

test("status filters and sorting reset cursors while retaining the selected photograph", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-page-next").trigger();
  const retained = selected(h);
  for (const value of ["approved", "needs_revision", "unreviewed", "all"]) {
    const before = requested(h, "/scenes").length;
    const filter = h.root.find("photographic-scene-filter"); filter.value = value; await filter.trigger("change");
    const request = requested(h, "/scenes")[before];
    assert.equal(request.params.get("status"), value);
    assert.equal(request.params.get("cursor"), null, "Changing status starts a new cursor sequence");
    assert.equal(selected(h), retained, "A metadata filter does not silently replace inspected detail");
  }
  const beforeSort = requested(h, "/scenes").length;
  const sort = h.root.find("photographic-scene-sort"); sort.value = "label"; await sort.trigger("change");
  assert.equal(requested(h, "/scenes")[beforeSort].params.get("sort"), "label");
  assert.equal(requested(h, "/scenes")[beforeSort].params.get("cursor"), null);
  assert.equal(selected(h), retained);
});

test("next unreviewed scans summary pages and opens only the next eligible photograph", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-next-unreviewed").trigger();
  assert.equal(selected(h), "input-0003");
  assert.equal(h.root.find("photographic-scene-filter").value, "unreviewed");
  assert.equal(requested(h, "/scenes").at(-1).params.get("status"), "unreviewed");
  assert.deepEqual(requested(h, "/scene").map((request) => request.params.get("scene_id")), ["input-0000", "input-0003"]);
  assert.equal(h.root.findAll("photographic-contact-item").length, PAGE_SIZE);
});

for (const endpoint of ["/scenes", "/scene"]) {
  test(`a stale ${endpoint} cursor reloads metadata and retains selected detail without replaying navigation`, async (t) => {
    let failed = false;
    const h = fixture({ extraFetch: (request, { server }) => {
      const stalePage = endpoint === "/scenes" && request.path.endsWith("/scenes") && request.params.get("cursor") === "8";
      const staleDetail = endpoint === "/scene" && request.path.endsWith("/scene") && request.params.get("scene_id") === "input-0001";
      if (failed || !(stalePage || staleDetail)) return null;
      failed = true; server.catalogueIdentitySha256 = NEXT_CATALOGUE_SHA;
      return error("CATALOGUE_CHANGED", "Catalogue changed; reset pagination.");
    } });
    t.after(() => h.api.show(null, null));
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    await h.root.find(endpoint === "/scenes" ? "photographic-page-next" : "photographic-scene-next").trigger();
    assert.equal(failed, true);
    assert.equal(selected(h), "input-0000");
    assert.equal(h.root.find("photographic-primary").value, h.scenes[0].variants[0].artifact_path);
    assert.equal(requested(h, "/scenes").at(-1).params.get("cursor"), null);
    assert.equal(requested(h, "/scene").at(-1).params.get("catalogue_identity_sha256"), NEXT_CATALOGUE_SHA);
    assert.match(h.root.find("photographic-review-status").textContent, /catalogue changed.*selection.*retained/i);
    assert.equal(h.root.find("photographic-review-status").dataset.tone, "error", "Catalogue recovery retains its warning until the operator inspects the refreshed state");
    assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
  });
}

test("unsaved notes and revision adjustments require explicit discard and survive failed navigation", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const note = h.root.find("photographic-decision-note");
  note.value = "Check the window edge before approval."; await note.trigger("input");
  const exposure = h.root.find("photographic-revision-exposure");
  exposure.value = "0.15"; await exposure.trigger("input");
  const before = h.requests.length;
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(h.requests.length, before, "Blocked navigation cannot fetch or discard the current draft");
  assert.equal(selected(h), "input-0000");
  assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
  assert.match(h.root.find("photographic-review-status").textContent, /unsaved.*retained/i);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "error");
  h.server.failedScene = "input-0001";
  await h.root.find("photographic-discard-navigation").trigger();
  assert.equal(selected(h), "input-0000");
  assert.equal(note.value, "Check the window edge before approval.");
  assert.equal(exposure.value, "0.15");
  assert.match(h.root.find("photographic-review-status").textContent, /Scene detail unavailable/);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "error");
  h.server.failedScene = "";
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0000", "A failed discard transition does not authorize a later implicit discard");
  await h.root.find("photographic-discard-navigation").trigger();
  assert.equal(selected(h), "input-0001");
  assert.equal(note.value, "");
  assert.equal(exposure.value, "0");
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "info");
  assert.match(h.root.find("photographic-review-status").textContent, /Photograph navigation ready/);
  assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
});

test("explicitly discarding for a status filter clears draft edits even when selected detail stays the same", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const note = h.root.find("photographic-decision-note"); note.value = "Unsaved draft"; await note.trigger("input");
  const exposure = h.root.find("photographic-revision-exposure"); exposure.value = "0.15"; await exposure.trigger("input");
  const filter = h.root.find("photographic-scene-filter"); filter.value = "unreviewed"; await filter.trigger("change");
  assert.equal(filter.value, "all");
  await h.root.find("photographic-discard-navigation").trigger();
  assert.equal(selected(h), "input-0000");
  assert.equal(filter.value, "unreviewed");
  assert.equal(note.value, "");
  assert.equal(exposure.value, "0");
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "info");
  assert.match(h.root.find("photographic-review-status").textContent, /Photograph navigation ready/);
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0003", "A completed explicit discard cannot leave a false unsaved blocker");
});

test("refresh preserves a missing photograph's draft until its replacement is explicitly accepted", async (t) => {
  const h = fixture({ sceneCount: 2 });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const note = h.root.find("photographic-decision-note"), exposure = h.root.find("photographic-revision-exposure");
  note.value = "Retain the missing photograph's review"; await note.trigger("input");
  exposure.value = "0.15"; await exposure.trigger("input");
  h.scenes.shift();
  await h.api.refresh(); await settle();
  assert.equal(selected(h), "input-0000");
  assert.equal(note.value, "Retain the missing photograph's review");
  assert.equal(exposure.value, "0.15");
  assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
  await h.root.find("photographic-discard-navigation").trigger();
  assert.equal(selected(h), "input-0001");
  assert.equal(note.value, "");
  assert.equal(exposure.value, "0");
  assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
});

test("refresh preserves a removed native version's draft when the same photograph still has a delivery TIFF", async (t) => {
  const h = fixture({ sceneCount: 1 });
  const nativePath = "working/input-0000/native.npy";
  h.scenes[0].variants.push({ ...h.scenes[0].variants[0], role: "working_master", artifact_path: nativePath, sha256: "e".repeat(64) });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const primary = h.root.find("photographic-primary"); primary.value = nativePath; await primary.trigger("change");
  const note = h.root.find("photographic-decision-note"), exposure = h.root.find("photographic-revision-exposure");
  note.value = "Retain this native version's draft"; await note.trigger("input");
  exposure.value = "0.15"; await exposure.trigger("input");
  h.scenes[0].variants = h.scenes[0].variants.filter((variant) => variant.artifact_path !== nativePath);
  await h.api.refresh(); await settle();
  assert.equal(selected(h), "input-0000");
  assert.equal(primary.value, nativePath);
  assert.equal(note.value, "Retain this native version's draft");
  assert.equal(exposure.value, "0.15");
  assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
  await h.root.find("photographic-discard-navigation").trigger();
  assert.equal(selected(h), "input-0000");
  assert.equal(primary.value, "input-0000/delivery.tif");
  assert.equal(note.value, ""); assert.equal(exposure.value, "0");
  assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
});

test("manual refresh cannot read pre-save metadata while a decision owns review", async (t) => {
  let finish;
  const h = fixture({ extraFetch: (request) => request.path.endsWith("/decisions")
    ? new Promise((resolve) => { finish = () => resolve(error("UNAVAILABLE", "Injected save unavailable.", 503)); }) : null });
  t.after(() => { finish?.(); h.api.show(null, null); });
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-decision-approved").trigger();
  const before = h.requests.length, refresh = h.root.find("photographic-review-refresh");
  await refresh.trigger();
  assert.equal(h.requests.length, before, "A directly dispatched manual refresh must respect mutation ownership");
  assert.equal(refresh.disabled, true);
  finish(); await settle();
  assert.equal(refresh.disabled, false);
});

test("failed committed-version reconciliation blocks stale mutations until a current manual refresh succeeds", async (t) => {
  let first = true;
  const h = fixture({ extraFetch: (request) => {
    if (!first || !request.path.endsWith("/decisions")) return null;
    first = false; return json({ saved: true, state_version: 5 });
  } });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const approve = h.root.find("photographic-decision-approved"), refresh = h.root.find("photographic-review-refresh");
  await approve.trigger();
  assert.equal(requested(h, "/decisions").length, 1);
  assert.equal(approve.disabled, true, "A committed receipt cannot authorize mutations against older review metadata");
  assert.equal(refresh.disabled, false, "Manual refresh remains available for recovery");
  await approve.trigger();
  assert.equal(requested(h, "/decisions").length, 1, "A direct event cannot post the stale expected_version");
  await refresh.trigger();
  assert.equal(approve.disabled, true, "Manual refresh must inherit the committed version floor");
  h.server.stateVersion = 5; h.server.catalogueIdentitySha256 = NEXT_CATALOGUE_SHA;
  const source = h.scenes[0].variants[0];
  h.server.decisions.push({ scene_id: h.scenes[0].scene_id, artifact_path: source.artifact_path, sha256: source.sha256,
    source_identity_sha256: source.sha256, status: "approved", note: "" });
  await refresh.trigger();
  assert.equal(approve.disabled, false);
  assert.match(h.root.find("photographic-decision-status").textContent, /Approved/);
  await approve.trigger();
  assert.equal(requested(h, "/decisions").length, 2);
  assert.equal(JSON.parse(requested(h, "/decisions")[1].options.body).expected_version, 5);
});

for (const reset of [false, true]) {
  test(`mutation reconciliation ${reset ? "cannot cross a reset while awaiting old refresh" : "fetches committed metadata after an older refresh settles"}`, async () => {
    const h = fixture({ sceneCount: 1 }), applied = [], requests = [];
    const node = (tag, text = "", className = "", ui = "") => {
      const element = new Element(tag); element.textContent = text; element.className = className; element.dataset.ui = ui; return element;
    };
    let finish, hold = false;
    const pager = createPhotographicPagedCatalogue({ el: node, button: (text, ui, handler) => {
      const control = node("button", text, "", ui); control.addEventListener("click", handler); return control;
    }, field: (_label, control) => control, select: (ui) => node("select", "", "", ui), status() {}, hasEdits: () => false,
    selection: () => ({ sceneId: "input-0000" }), apply: (loaded) => applied.push(loaded.state_version), request: async (path) => {
      const url = new URL(path, "http://localhost"); requests.push(url.pathname);
      const result = structuredClone(url.pathname === "/scenes" ? h.page(url.searchParams) : h.detail("input-0000"));
      if (hold && url.pathname === "/scene") {
        hold = false; return new Promise((resolve) => { finish = () => resolve(result); });
      }
      return result;
    } });
    await pager.refresh("input-0000");
    hold = true; const older = pager.refresh("input-0000"); await settle();
    pager.lock(true); h.server.stateVersion = 5; h.server.catalogueIdentitySha256 = NEXT_CATALOGUE_SHA;
    const fresh = pager.refresh("input-0000", { reconcile: true, minimumVersion: 5 }); await settle();
    if (reset) pager.reset();
    finish(); await older; const reconciled = await fresh;
    assert.equal(reconciled, !reset);
    assert.equal(requests.filter((path) => path === "/scenes").length, reset ? 2 : 3, "Reconciliation requires its own read in the original scope");
    assert.equal(applied.at(-1), reset ? 4 : 5);
  });
}

for (const operation of ["decision", "refresh"]) {
  test(`discard navigation preserves its pending action while a ${operation} is in progress`, async (t) => {
    let finish, delay = false;
    const h = fixture({ extraFetch: (request) => {
      if (!delay || !request.path.endsWith(operation === "decision" ? "/decisions" : "/scenes")) return null;
      delay = false;
      return new Promise((resolve) => { finish = () => resolve(error("UNAVAILABLE", "Injected request unavailable.", 503)); });
    } });
    t.after(() => { finish?.(); h.api.show(null, null); });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const note = h.root.find("photographic-decision-note"); note.value = "Retain this draft"; await note.trigger("input");
    await h.root.find("photographic-scene-next").trigger();
    const discard = h.root.find("photographic-discard-navigation");
    assert.equal(discard.hidden, false);
    delay = true;
    const pending = operation === "decision" ? h.root.find("photographic-decision-approved").trigger() : h.api.refresh();
    await settle();
    assert.ok(finish);
    const before = h.requests.length;
    const disabled = discard.disabled;
    await discard.trigger();
    assert.equal(h.requests.length, before, "Even a directly dispatched event cannot start competing navigation");
    assert.equal(discard.hidden, false, "A blocked discard must retain its pending action");
    assert.equal(disabled, true, "Discard follows the active mutation or navigation lock");
    assert.equal(selected(h), "input-0000");
    assert.equal(note.value, "Retain this draft");
    finish(); await pending; await settle();
    assert.equal(discard.disabled, false);
    await discard.trigger();
    assert.equal(selected(h), "input-0001", "The explicit pending navigation remains available after the lock clears");
    assert.equal(note.value, "");
    assert.equal(requested(h, "/decisions").length, operation === "decision" ? 1 : 0);
  });
}

test("a navigation that finds no unreviewed scene preserves its specific result message", async (t) => {
  const h = fixture({ sceneCount: 1 });
  h.server.decisions = [{ scene_id: h.scenes[0].scene_id, label: h.scenes[0].label, role: "final", status: "approved", note: "",
    artifact_path: h.scenes[0].variants[0].artifact_path, sha256: h.scenes[0].variants[0].sha256, source_identity_sha256: h.scenes[0].variants[0].sha256 }];
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-next-unreviewed").trigger();
  assert.equal(selected(h), "input-0000");
  assert.match(h.root.find("photographic-review-status").textContent, /No unreviewed photographs remain/);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "info");
});

test("session restoration retains selected scene, comparison mode and native ROI only within its actor and job", async (t) => {
  const recovery = storageFixture();
  const first = fixture({ sessionStorage: recovery.storage });
  t.after(() => first.api.show(null, null));
  first.api.show(first.job, first.scenes[0].variants[0]); await settle();
  await first.root.find("photographic-page-next").trigger();
  const mode = first.root.find("photographic-comparison-mode"); mode.value = "wipe"; await mode.trigger("change");
  const inspection = first.root.find("photographic-inspection"); inspection.value = "native"; await inspection.trigger("change");
  const crop = first.root.find("photographic-crop-size"); crop.value = "1024"; await crop.trigger("change");
  const x = first.root.find("photographic-crop-x"); x.value = "73"; await x.trigger("change");
  const y = first.root.find("photographic-crop-y"); y.value = "24"; await y.trigger("change");
  const note = first.root.find("photographic-decision-note"); note.value = "Private unsaved review note"; await note.trigger("input");
  const stored = recovery.values.get("tp.photo.review_selection.v1");
  assert.ok(stored && stored.length <= 32768);
  assert.doesNotMatch(stored, /fixture-key|fixture-csrf|blob:|Private unsaved review note/);
  first.api.show(null, null);
  const restored = fixture({ sessionStorage: recovery.storage });
  t.after(() => restored.api.show(null, null));
  restored.api.show(restored.job, restored.scenes[0].variants[0]); await settle();
  assert.equal(selected(restored), "input-0008");
  assert.equal(restored.root.find("photographic-comparison-mode").value, "wipe");
  assert.equal(restored.root.find("photographic-inspection").value, "native");
  assert.equal(restored.root.find("photographic-crop-size").value, "1024");
  assert.equal(restored.root.find("photographic-crop-x").value, "73");
  assert.equal(restored.root.find("photographic-crop-y").value, "24");
  assert.equal(restored.root.find("photographic-decision-note").value, "");
  assert.deepEqual(requested(restored, "/scene").map((request) => request.params.get("scene_id")), ["input-0008"]);
  const native = requested(restored, "/native-crop")[0];
  assert.equal(native.params.get("x"), "5328");
  assert.equal(native.params.get("y"), "928");
  assert.equal(native.params.get("width"), "1024");
  for (const scope of [{ actor: "actor-two" }, { jobId: "job-two" }]) {
    const other = fixture({ sessionStorage: recovery.storage, ...scope });
    t.after(() => other.api.show(null, null));
    other.api.show(other.job, other.scenes[0].variants[0]); await settle();
    assert.equal(selected(other), "input-0000");
    assert.equal(other.root.find("photographic-comparison-mode").value, "side_by_side");
    assert.equal(other.root.find("photographic-inspection").value, "fit");
  }
});

test("optional host selection restoration opens one authoritative detail without requiring a legacy catalogue", async (t) => {
  const h = fixture({ restoredSelection: { jobId: "job-one", sceneId: "input-0031", artifactPath: "input-0031/delivery.tif",
    stateVersion: 3, catalogueIdentitySha256: CATALOGUE_SHA } });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  assert.equal(selected(h), "input-0031");
  assert.deepEqual(requested(h, "/scene").map((request) => request.params.get("scene_id")), ["input-0031"]);
  assert.equal(h.selections.at(-1).stateVersion, h.server.stateVersion, "Restoration metadata cannot replace current server review state");
});

for (const role of ["original", "previous_revision"]) {
  test(`host scene navigation retains the exact ${role} artifact path`, async (t) => {
    const artifactPath = role === "original" ? "original/input-0031/source.tif" : "revisions/input-0031/prior.tif";
    const h = fixture({ restoredSelection: { jobId: "job-one", sceneId: "input-0031", artifactPath } });
    const revisionPath = "revisions/input-0031/prior.tif", workingPath = "working/input-0031/master.npy";
    h.scenes[31].variants.push(
      { ...h.scenes[31].variants[0], role: "previous_revision", artifact_path: revisionPath, sha256: "f".repeat(64) },
      { ...h.scenes[31].variants[0], role: "working_master", artifact_path: workingPath, sha256: "e".repeat(64) });
    if (role === "original") {
      const row = h.page(new URLSearchParams("limit=8&cursor=24")).scenes.find((scene) => scene.scene_id === "input-0031");
      assert.equal(row.related_artifacts.some((artifact) => artifact.artifact_path === artifactPath), false, "Originals are absent from related-artifact hints in the backend DTO");
      assert.ok(row.variant_paths.includes(artifactPath));
      assert.equal(h.server.decisions.some((decision) => decision.artifact_path === artifactPath), false);
    }
    t.after(() => h.api.show(null, null));
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    assert.equal(selected(h), "input-0031");
    assert.equal(h.root.find("photographic-primary").value, artifactPath);
    assert.equal(h.selections.at(-1).artifactPath, artifactPath);
    assert.deepEqual(h.selections.at(-1).railPaths, [h.scenes[31].variants[0].artifact_path, revisionPath]);
    assert.equal(h.selections.at(-1).railPaths.includes(h.scenes[31].variants[1].artifact_path), false, "Original inspection does not bind a published rail artifact");
    assert.equal(h.selections.at(-1).railPaths.includes(workingPath), false, "Working masters do not bind published rail artifacts");
    assert.deepEqual(requested(h, "/scene").map((request) => request.params.get("scene_id")), ["input-0031"]);
  });
}

test("an artifact-only host link beyond the first page locates its scene using metadata alone", async (t) => {
  const path = "original/input-0031/source.tif";
  const h = fixture({ restoredSelection: { jobId: "job-one", artifactPath: path } });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  assert.equal(selected(h), "input-0031");
  assert.equal(h.root.find("photographic-primary").value, path);
  assert.equal(requested(h, "/scenes").length, 4);
  assert.deepEqual(requested(h, "/scene").map((request) => request.params.get("scene_id")), ["input-0031"]);
  assert.equal(requested(h, "/photographic-review").length, 0);
});

test("selection recovery stores bounded view metadata and tolerates malformed or unavailable session storage", () => {
  const recovery = storageFixture();
  const selections = photographicSelectionStorage(recovery.storage);
  const view = { sceneId: "input-0008", primaryPath: "input-0008/delivery.tif", comparePath: "original/input-0008/source.tif",
    mode: "wipe", inspection: "native", presentationView: "clip_srgb", center: { x: 0.73, y: 0.24 }, cropSize: 1024 };
  for (let index = 0; index < 20; index += 1) selections.set(`actor-job-${index}`, view);
  const stored = recovery.values.get("tp.photo.review_selection.v1");
  assert.ok(stored.length <= 32768);
  assert.ok(JSON.parse(stored).length <= 16);
  assert.equal(selections.get("actor-job-0"), null);
  assert.deepEqual(selections.get("actor-job-19"), view);
  assert.equal(selections.get("different-actor-job"), null);
  for (const raw of ["{", "null", "{}", "x".repeat(32769)]) {
    recovery.values.set("tp.photo.review_selection.v1", raw);
    assert.equal(selections.get("actor-job-19"), null);
  }
  selections.set("actor-job-invalid", { ...view, sceneId: "../other-scene" });
  assert.equal(selections.get("actor-job-invalid"), null);
  const unavailable = photographicSelectionStorage({ getItem() { throw new Error("storage disabled"); }, setItem() { throw new Error("storage disabled"); } });
  assert.equal(unavailable.get("actor-job"), null);
  assert.doesNotThrow(() => unavailable.set("actor-job", view));
});

test("metadata navigation is keyboard accessible without taking arrow keys from editable controls", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const controls = h.root.find("photographic-paged-controls");
  await controls.trigger("keydown", { key: "ArrowRight", target: { tagName: "TEXTAREA" } });
  assert.equal(selected(h), "input-0000");
  await controls.trigger("keydown", { key: "ArrowRight", target: controls });
  assert.equal(selected(h), "input-0001");
  await controls.trigger("keydown", { key: "ArrowLeft", target: controls });
  assert.equal(selected(h), "input-0000");
});

test("focused photograph cards retain consecutive arrow navigation across both page boundaries", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const contact = h.root.find("photographic-contact-sheet");
  h.root.find("photographic-contact-item").focus();
  for (const [key, indexes] of [["ArrowRight", [1, 2, 3, 4, 5, 6, 7, 8, 9]], ["ArrowLeft", [8, 7, 6]]]) {
    for (const index of indexes) {
      await contact.trigger("keydown", { key, target: h.document.activeElement });
      assert.equal(selected(h), h.scenes[index].scene_id);
      assert.ok(h.document.activeElement === h.root.findAll("photographic-contact-item").find((control) => control.attributes["aria-pressed"] === "true"), "The selected card must own focus after every arrow transition");
    }
  }
  const next = h.root.find("photographic-scene-next"); next.focus(); await next.trigger();
  assert.equal(h.document.activeElement, next, "Rendering cards must retain focus on a persistent navigation control");
  const primary = h.root.find("photographic-primary"); primary.focus(); await h.api.refresh();
  assert.equal(h.document.activeElement, primary, "Rendering cards must retain focus on a persistent input");
});

for (const moveFocus of [false, true]) {
  test(`delayed photograph navigation ${moveFocus ? "preserves another input's focus" : "restores its disabled navigation control's focus"}`, async (t) => {
    let finish, delay = false;
    const h = fixture({ extraFetch: (request, response) => {
      if (!delay || !request.path.endsWith("/scene")) return null;
      delay = false;
      return new Promise((resolve) => { finish = () => resolve(json(response.detail(request.params.get("scene_id")))); });
    } });
    t.after(() => { finish?.(); h.api.show(null, null); });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    await h.root.find("photographic-page-next").trigger();
    const previous = h.root.find("photographic-scene-previous"), primary = h.root.find("photographic-primary");
    previous.focus(); delay = true; await previous.trigger();
    assert.ok(finish);
    assert.equal(previous.disabled, true);
    assert.equal(h.document.activeElement, h.document.body, "Disabling a focused native control moves focus to the document");
    assert.equal(selected(h), "input-0008");
    if (moveFocus) primary.focus();
    finish(); await settle();
    assert.equal(selected(h), "input-0007");
    assert.equal(previous.disabled, false);
    assert.ok(h.document.activeElement === (moveFocus ? primary : previous), "Navigation restores only focus lost to its own disabled control");
  });
}

test("saving an exact decision refreshes cursor metadata and preserves the selected photograph", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-page-next").trigger();
  const before = requested(h, "/scenes").length;
  const note = h.root.find("photographic-decision-note"); note.value = "Native detail checked."; await note.trigger("input");
  await h.root.find("photographic-decision-approved").trigger();
  const saved = requested(h, "/decisions");
  assert.equal(saved.length, 1);
  assert.deepEqual(JSON.parse(saved[0].options.body), { artifact_path: "input-0008/delivery.tif", sha256: h.scenes[8].variants[0].sha256,
    source_identity_sha256: h.scenes[8].variants[0].sha256, status: "approved", note: "Native detail checked.", expected_version: 4 });
  assert.equal(saved[0].options.headers["X-CSRF-Token"], "fixture-csrf");
  assert.equal(requested(h, "/scenes")[before].params.get("cursor"), null);
  assert.equal(selected(h), "input-0008");
  assert.equal(note.value, "Native detail checked.");
  assert.match(h.root.find("photographic-decision-status").textContent, /Approved/);
  assert.equal(h.selections.at(-1).stateVersion, 5);
  assert.equal(h.selections.at(-1).catalogueIdentitySha256, NEXT_CATALOGUE_SHA);
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0009", "A saved note does not leave a false unsaved-navigation blocker");
});

test("in-flight decisions keep navigation bound to the source being saved", async (t) => {
  let complete;
  const h = fixture({ extraFetch: (request) => request.options.method === "POST" && request.path.endsWith("/decisions")
    ? new Promise((resolve) => { complete = resolve; }) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-decision-approved").trigger();
  assert.ok(complete);
  assert.equal(h.root.find("photographic-scene-next").disabled, true);
  const before = requested(h, "/scene").length;
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0000");
  assert.equal(requested(h, "/scene").length, before);
  complete(json({ saved: true })); await settle();
  assert.equal(h.root.find("photographic-scene-next").disabled, false);
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0001");
});

test("in-flight scene detail blocks draft edits and mutations until the new source owns review", async (t) => {
  let complete;
  let delayed = true;
  const h = fixture({ sceneCount: 2, extraFetch: (request, response) => {
    if (!delayed || !request.path.endsWith("/scene") || request.params.get("scene_id") !== "input-0001") return null;
    delayed = false;
    return new Promise((resolve) => { complete = () => resolve(json(response.detail("input-0001"))); });
  } });
  h.server.decisions = [{ scene_id: h.scenes[0].scene_id, label: h.scenes[0].label, role: "final", status: "approved", note: "",
    artifact_path: h.scenes[0].variants[0].artifact_path, sha256: h.scenes[0].variants[0].sha256, source_identity_sha256: h.scenes[0].variants[0].sha256 }];
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const preset = h.root.find("photographic-preset"); preset.checked = true; await preset.trigger("change");
  assert.equal(h.root.find("photographic-deliver-approved").disabled, false);
  assert.equal(h.root.find("photographic-preview-revision").disabled, false);
  await h.root.find("photographic-scene-next").trigger();
  assert.ok(complete);
  assert.equal(selected(h), "input-0000");
  for (const ui of ["photographic-decision-approved", "photographic-deliver-approved", "photographic-preview-revision",
    "photographic-decision-note", "photographic-revision-exposure"]) {
    assert.equal(h.root.find(ui).disabled, true, `${ui} must stay bound while selected detail is loading`);
  }
  for (const ui of ["photographic-decision-approved", "photographic-deliver-approved", "photographic-preview-revision"]) {
    await h.root.find(ui).trigger();
  }
  assert.equal(h.requests.filter((request) => request.options.method === "POST").length, 0);
  complete(); await settle();
  assert.equal(selected(h), "input-0001");
  assert.equal(h.root.find("photographic-decision-note").disabled, false);
  assert.equal(h.root.find("photographic-revision-exposure").disabled, false);
  await h.root.find("photographic-decision-approved").trigger();
  assert.equal(requested(h, "/decisions").length, 1);
  const body = JSON.parse(requested(h, "/decisions")[0].options.body);
  assert.equal(body.artifact_path, "input-0001/delivery.tif");
  assert.equal(body.sha256, h.scenes[1].variants[0].sha256);
  assert.equal(h.root.find("photographic-delivery-version").value, "input-0001/delivery.tif");
  assert.match(h.root.find("photographic-deliver-approved").textContent, /\(2\)/);
});

test("an old mutation's pending refresh cannot publish success into a new actor scope", async (t) => {
  let holdRefresh = false;
  let complete;
  const h = fixture({ extraFetch: (request, response) => {
    if (request.options.method === "POST" && request.path.endsWith("/decisions")) holdRefresh = true;
    if (!holdRefresh || !request.path.endsWith("/scenes")) return null;
    holdRefresh = false;
    return new Promise((resolve) => { complete = () => resolve(json(response.page(request.params))); });
  } });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-decision-approved").trigger();
  assert.equal(requested(h, "/decisions").length, 1);
  assert.ok(complete, "The old mutation has committed and its metadata refresh is pending");
  h.state.auth.actor = { username: "actor-two" };
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const readyStatus = h.root.find("photographic-review-status").textContent;
  assert.match(readyStatus, /Review loaded/);
  complete(); await settle();
  assert.equal(h.root.find("photographic-review-status").textContent, readyStatus);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "info");
  assert.equal(requested(h, "/decisions").length, 1, "A scope change never replays the old mutation");
});

test("refreshing unchanged selected metadata reuses verified image pixels", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-page-next").trigger();
  const beforeRenders = requested(h, "/render").length;
  const beforeDetails = requested(h, "/scene").length;
  const source = h.root.find("photographic-primary-image").src;
  await h.root.find("photographic-review-refresh").trigger();
  assert.equal(selected(h), "input-0008");
  assert.equal(requested(h, "/scene").length, beforeDetails + 1, "Refresh revalidates current detail");
  assert.equal(requested(h, "/render").length, beforeRenders, "Identical artifact identity and view settings reuse the verified image cache");
  assert.equal(h.root.find("photographic-primary-image").src, source);
});

test("delivery includes recorded approvals across pages and revalidates every selected source before export", async (t) => {
  const previousLocation = globalThis.location;
  globalThis.location = { origin: "http://localhost" };
  t.after(() => { if (previousLocation === undefined) delete globalThis.location; else globalThis.location = previousLocation; });
  const h = fixture({ extraFetch: (request, { server }) => request.options.method === "POST" && request.path.endsWith("/delivery")
    ? json({ state_version: server.stateVersion, download_url: "/v1/jobs/job-one/photographic-review/files/export/photos.zip",
      filename: "photos.zip", sha256: "e".repeat(64), size_bytes: 1234 }) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const approvalCount = h.server.decisions.filter((decision) => decision.status === "approved").length;
  assert.equal(requested(h, "/scene").length, 1, "Recorded delivery choices do not prefetch their rich scenes");
  await h.root.find("photographic-scene-next").trigger();
  assert.match(h.root.find("photographic-deliver-approved").textContent, new RegExp(`\\(${approvalCount}\\)`));
  await h.root.find("photographic-page-next").trigger();
  await h.root.find("photographic-scene-next").trigger();
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0010");
  assert.match(h.root.find("photographic-deliver-approved").textContent, new RegExp(`\\(${approvalCount}\\)`));
  const preset = h.root.find("photographic-preset"); preset.checked = true; await preset.trigger("change");
  const before = h.requests.length;
  await h.root.find("photographic-deliver-approved").trigger();
  const delivery = requested(h, "/delivery");
  assert.equal(delivery.length, 1);
  const body = JSON.parse(delivery[0].options.body);
  const approved = h.server.decisions.filter((decision) => decision.status === "approved");
  assert.deepEqual(body.approved_sources.map((source) => source.artifact_path).sort(), approved.map((decision) => decision.artifact_path).sort());
  assert.deepEqual(body.scene_labels, Object.fromEntries(approved.map((decision) => [decision.scene_id, decision.label])));
  const beforePost = h.requests.slice(before, h.requests.indexOf(delivery[0]));
  for (const decision of approved) {
    assert.ok(beforePost.some((request) => request.path.endsWith("/scene") && request.params.get("scene_id") === decision.scene_id));
  }
  assert.ok(h.root.find("photographic-delivery-download"));
  const exported = delivery.length;
  h.server.decisions = h.server.decisions.filter((decision) => decision.scene_id !== "input-0001");
  h.server.stateVersion += 1;
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(requested(h, "/delivery").length, exported, "A source revoked on a different page must block another export");
  assert.match(h.root.find("photographic-review-status").textContent, /no longer approved|currently approved/i);
});

test("a retained delivery choice whose approval was revoked can be explicitly excluded before exporting the remaining page", async (t) => {
  const previousLocation = globalThis.location;
  globalThis.location = { origin: "http://localhost" };
  t.after(() => { if (previousLocation === undefined) delete globalThis.location; else globalThis.location = previousLocation; });
  const h = fixture({ sceneCount: 10, extraFetch: (request, { server }) => request.path.endsWith("/delivery")
    ? json({ state_version: server.stateVersion, download_url: "/v1/jobs/job-one/photographic-review/files/export/photos.zip",
      filename: "photos.zip", sha256: "e".repeat(64), size_bytes: 1234 }) : null });
  h.server.decisions = [h.scenes[0], h.scenes[9]].map((scene) => ({ scene_id: scene.scene_id, label: scene.label, role: "final",
    artifact_path: scene.variants[0].artifact_path, sha256: scene.variants[0].sha256,
    source_identity_sha256: scene.variants[0].sha256, status: "approved", note: "" }));
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const preset = h.root.find("photographic-preset"); preset.checked = true; await preset.trigger("change");
  assert.match(h.root.find("photographic-delivery-selection-summary").textContent, /2 selected across all pages \(2 currently recorded approved\)/);
  h.server.decisions[0].status = "needs_revision";
  h.server.stateVersion += 1; h.server.catalogueIdentitySha256 = NEXT_CATALOGUE_SHA;
  await h.api.refresh(); await settle();
  assert.match(h.root.find("photographic-delivery-selection-summary").textContent, /2 selected across all pages \(1 currently recorded approved\)/, "Revocation must retain the operator's exact selection until explicit exclusion");
  assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(requested(h, "/delivery").length, 0, "Revocation blocks export until the operator resolves the retained choice");
  const choice = h.root.find("photographic-delivery-version");
  assert.ok(choice, "Every retained choice needs an exclusion control even with no approved variants");
  assert.equal(choice.value, "input-0000/delivery.tif");
  assert.equal(choice.options.some((option) => option.value === ""), true);
  choice.value = ""; await choice.trigger("change");
  assert.match(h.root.find("photographic-delivery-selection-summary").textContent, /1 selected across all pages \(1 currently recorded approved\)/);
  assert.equal(requested(h, "/delivery").length, 0, "Explicit exclusion changes only the local delivery selection");
  await h.root.find("photographic-page-next").trigger();
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0009");
  assert.match(h.root.find("photographic-delivery-selection-summary").textContent, /1 selected across all pages/);
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(requested(h, "/delivery").length, 1);
  const body = JSON.parse(requested(h, "/delivery")[0].options.body);
  assert.deepEqual(body.approved_sources.map((source) => source.artifact_path), ["input-0009/delivery.tif"]);
  assert.deepEqual(body.approved_sha256, [h.scenes[9].variants[0].sha256]);
});

test("flag-off review still clears revoked delivery choices automatically", async (t) => {
  const previousLocation = globalThis.location;
  globalThis.location = { origin: "http://localhost" };
  t.after(() => { if (previousLocation === undefined) delete globalThis.location; else globalThis.location = previousLocation; });
  const h = fixture({ paged: false, sceneCount: 2, extraFetch: (request, { server }) => request.path.endsWith("/delivery")
    ? json({ state_version: server.stateVersion, download_url: "/v1/jobs/job-one/photographic-review/files/export/photos.zip",
      filename: "photos.zip", sha256: "e".repeat(64), size_bytes: 1234 }) : null });
  h.server.decisions = h.scenes.map((scene) => ({ scene_id: scene.scene_id, role: "final", status: "approved", note: "",
    artifact_path: scene.variants[0].artifact_path, sha256: scene.variants[0].sha256, source_identity_sha256: scene.variants[0].sha256 }));
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  assert.equal(h.root.findAll("photographic-delivery-version").length, 2);
  h.server.decisions[0].status = "needs_revision"; h.server.stateVersion += 1;
  await h.api.refresh(); await settle();
  assert.deepEqual(h.root.findAll("photographic-delivery-version").map((control) => control.value), ["input-0001/delivery.tif"], "Legacy review must not retain an exclusion control for the revoked source");
  assert.equal(h.root.find("photographic-delivery-selection-summary").hidden, true);
  const preset = h.root.find("photographic-preset"); preset.checked = true; await preset.trigger("change");
  assert.equal(requested(h, "/delivery").length, 0);
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(requested(h, "/delivery").length, 1);
  assert.deepEqual(JSON.parse(requested(h, "/delivery")[0].options.body).approved_sources.map((source) => source.artifact_path), ["input-0001/delivery.tif"]);
  assert.equal(requested(h, "/scenes").length, 0);
});

test("recorded lighting references remain selectable without opening their unrelated rich scene", async (t) => {
  const h = fixture();
  h.server.groups = [{ name: "Interiors", reference_sha256: h.scenes[40].variants[0].sha256,
    members: [h.scenes[1].variants[0].sha256, h.scenes[40].variants[0].sha256] }];
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(h.root.find("photographic-lighting-group").value, "Interiors");
  assert.equal(h.root.find("photographic-reference").value, h.scenes[40].variants[0].sha256);
  assert.ok(h.root.find("photographic-reference").options.some((option) => /Recorded lighting reference/.test(option.textContent)));
  assert.equal(requested(h, "/scene").some((request) => request.params.get("scene_id") === "input-0040"), false);
});

test("renewed approval cannot reuse a delivery choice after its same-path source interpretation changes", async (t) => {
  const h = fixture();
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const preset = h.root.find("photographic-preset"); preset.checked = true; await preset.trigger("change");
  const previous = h.scenes[1].variants[0];
  h.scenes[1].variants[0] = { ...previous, source_identity_sha256: "f".repeat(64) };
  h.server.decisions = h.server.decisions.map((decision) => decision.scene_id === "input-0001"
    ? { ...decision, source_identity_sha256: "f".repeat(64) } : decision);
  h.server.catalogueIdentitySha256 = NEXT_CATALOGUE_SHA;
  h.server.stateVersion += 1;
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(requested(h, "/delivery").length, 0);
  assert.match(h.root.find("photographic-review-status").textContent, /no longer approved or available/);
  assert.equal(h.scenes[1].variants[0].artifact_path, previous.artifact_path);
  assert.equal(h.scenes[1].variants[0].sha256, previous.sha256);
});

test("a metadata response arriving after a new actor scope cannot replace its current page or detail", async (t) => {
  let complete;
  let delayed = true;
  const h = fixture({ extraFetch: (request) => {
    if (!delayed || !request.path.endsWith("/scenes")) return null;
    delayed = false;
    return new Promise((resolve) => { complete = resolve; });
  } });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const stale = requested(h, "/scenes")[0];
  h.state.auth.actor = { username: "actor-two" };
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  assert.equal(stale.options.signal.aborted, true);
  const callbackCount = h.selections.length;
  const oldPage = h.page(new URLSearchParams("limit=8&status=all&sort=scene_id"));
  oldPage.scenes = [{ ...oldPage.scenes[0], scene_id: "old-scope-scene" }];
  complete(json(oldPage)); await settle();
  assert.equal(h.selections.length, callbackCount);
  assert.equal(selected(h), "input-0000");
  assert.equal(h.root.findAll("photographic-contact-item").length, PAGE_SIZE);
  assert.equal(requested(h, "/scene").some((request) => request.params.get("scene_id") === "old-scope-scene"), false);
});

test("a discarded navigation request cannot replace the new actor's ready status", async (t) => {
  let complete;
  const h = fixture({ extraFetch: (request) => request.path.endsWith("/scenes") && request.params.get("cursor") === "8"
    ? new Promise((resolve) => { complete = resolve; }) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-page-next").trigger();
  const stale = requested(h, "/scenes").at(-1);
  assert.ok(complete);
  h.state.auth.actor = { username: "actor-two" };
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  assert.equal(stale.options.signal.aborted, true);
  const readyStatus = h.root.find("photographic-review-status").textContent;
  complete(json(h.page(new URLSearchParams("limit=8&status=all&sort=scene_id&cursor=8")))); await settle();
  assert.equal(selected(h), "input-0000");
  assert.equal(h.root.find("photographic-review-status").textContent, readyStatus);
  assert.equal(h.root.find("photographic-review-status").dataset.tone, "info");
});

test("disabling paged review aborts old navigation and late metadata cannot replace the legacy selection", async (t) => {
  let complete;
  const h = fixture({ extraFetch: (request) => request.path.endsWith("/scenes") && request.params.get("cursor") === "8"
    ? new Promise((resolve) => { complete = resolve; }) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-page-next").trigger();
  const stale = requested(h, "/scenes").at(-1);
  assert.ok(complete);
  h.state.auth.features.photographicPagedCatalogue = false;
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  assert.equal(stale.options.signal.aborted, true);
  assert.equal(requested(h, "/photographic-review").length, 1);
  assert.equal(h.root.find("photographic-paged-controls").hidden, true);
  assert.equal(selected(h), "input-0000");
  const detailCount = requested(h, "/scene").length;
  complete(json(h.page(new URLSearchParams("limit=8&status=all&sort=scene_id&cursor=8")))); await settle();
  assert.equal(selected(h), "input-0000");
  assert.equal(h.root.find("photographic-primary").value, "input-0000/delivery.tif");
  assert.equal(requested(h, "/scene").length, detailCount, "A revoked paged request cannot start a rich detail read");
});

test("an old actor's late rate limit cannot establish a cooldown for the current actor", async (t) => {
  let complete;
  let delayed = true;
  const h = fixture({ extraFetch: (request) => {
    if (!delayed || !request.path.endsWith("/scenes") || request.params.get("cursor") !== "8") return null;
    delayed = false;
    return new Promise((resolve) => { complete = resolve; });
  } });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await h.root.find("photographic-page-next").trigger();
  assert.ok(complete);
  h.state.auth.actor = { username: "actor-two" };
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const readyStatus = h.root.find("photographic-review-status").textContent;
  complete(new Response(JSON.stringify({ success: false, error: { code: "RATE_LIMITED", message: "Previous actor is rate limited." } }),
    { status: 429, headers: { "Retry-After": "600" } })); await settle();
  assert.equal(h.root.find("photographic-review-status").textContent, readyStatus);
  await h.root.find("photographic-scene-next").trigger();
  assert.equal(selected(h), "input-0001");
  assert.doesNotMatch(h.root.find("photographic-review-status").textContent, /Rate limited/);
});

for (const endpoint of ["/decisions", "/revisions"]) {
  test(`an old actor's late ${endpoint} result cannot release the current actor's pending controls`, async (t) => {
    const pending = [];
    const h = fixture({ extraFetch: (request) => request.options.method === "POST" && request.path.endsWith(endpoint)
      ? new Promise((resolve) => pending.push({ resolve, body: JSON.parse(request.options.body) })) : null });
    t.after(() => h.api.show(null, null));
    const control = endpoint === "/decisions" ? "photographic-decision-approved" : "photographic-preview-revision";
    const finish = (index) => {
      const request = pending[index];
      request.resolve(json(endpoint === "/decisions" ? { saved: true, state_version: h.server.stateVersion } : {
        preview_data_url: "data:image/png;base64,cG5n", receipt: { schema: "tp.photo.revision.v1",
          source_master_sha256: request.body.sha256, width: 8000, height: 6000, output_bit_depth: 32,
          review_status: "unreviewed", approval_transferred: false, reference_sha256: null, recipe: request.body.recipe },
      }));
    };
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    await h.root.find(control).trigger();
    assert.equal(pending.length, 1);
    h.state.auth.actor = { username: "actor-two" };
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    await h.root.find(control).trigger();
    assert.equal(pending.length, 2);
    assert.equal(h.root.find("photographic-scene-next").disabled, true);
    const currentStatus = h.root.find("photographic-review-status").textContent;
    finish(0); await settle();
    assert.equal(h.root.find("photographic-scene-next").disabled, true);
    assert.equal(h.root.find("photographic-decision-approved").disabled, true);
    assert.equal(h.root.find("photographic-review-status").textContent, currentStatus);
    assert.equal(h.root.find("photographic-create-revision").disabled, true);
    finish(1); await settle();
    assert.equal(h.root.find("photographic-scene-next").disabled, false);
    assert.equal(h.root.find("photographic-decision-approved").disabled, false);
    if (endpoint === "/revisions") {
      assert.equal(h.root.find("photographic-variant-preview").classList.contains("hidden"), false);
      assert.equal(h.root.find("photographic-create-revision").disabled, false);
    }
  });
}

for (const changedScope of ["actor", "tenant", "job", "revocation"]) {
  test(`a late rich scene response cannot publish after ${changedScope} changes the review scope`, async (t) => {
    let complete;
    let delayed = true;
    const h = fixture({ extraFetch: (request) => {
      if (!delayed || !request.path.endsWith("/scene")) return null;
      delayed = false;
      return new Promise((resolve) => { complete = resolve; });
    } });
    t.after(() => h.api.show(null, null));
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const stale = requested(h, "/scene")[0];
    assert.ok(complete);
    if (changedScope === "actor") h.state.auth.actor = { username: "actor-two" };
    else if (changedScope === "tenant") h.state.auth.tenant = "tenant-two";
    const nextJob = changedScope === "job" ? { ...h.job, id: "job-two" } : h.job;
    h.api.show(changedScope === "revocation" ? null : nextJob, h.scenes[0].variants[0]); await settle();
    assert.equal(stale.options.signal.aborted, true);
    const callbackCount = h.selections.length;
    const staleDetail = h.detail("input-0000");
    staleDetail.scenes = [{ ...staleDetail.scenes[0], variants: [{ ...h.scenes[0].variants[0], artifact_path: "old-scope/delivery.tif" }] }];
    complete(json(staleDetail)); await settle();
    assert.equal(h.selections.length, callbackCount, "A response from a discarded scope cannot publish a host selection");
    if (changedScope === "revocation") assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
    else {
      assert.equal(h.root.find("photographic-primary").value, "input-0000/delivery.tif");
      assert.equal(h.selections.at(-1).jobId, nextJob.id);
    }
  });
}
