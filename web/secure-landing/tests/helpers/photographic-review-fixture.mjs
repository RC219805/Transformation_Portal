import assert from "node:assert/strict";
import { createPhotographicReview } from "../../portal-src/photographic-review.js";

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
  extraFetch, restoredSelection = null, getReviewAuthorization = null, onReviewAuthorizationFailure = null } = {}) {
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
    ...(getReviewAuthorization ? { getReviewAuthorization } : {}),
    ...(onReviewAuthorizationFailure ? { onReviewAuthorizationFailure } : {}),
    fetchImpl, renderDebounceMs: 0, sessionStorage: recovery,
    _buildAuthHeaders: (headers, method) => ({ ...headers, ...(method === "POST" ? { "X-CSRF-Token": "fixture-csrf" } : { "x-api-key": "fixture-key" }) }),
    onPhotographicSelection: (selection) => selections.push(selection), getPhotographicSelection: () => restoredSelection,
  });
  const job = { id: jobId, pipeline: "lux-depth", artifacts: scenes.map((scene) => ({ ...scene.variants[0], path: scene.variants[0].artifact_path })) };
  return { root, document, state, requests, selections, scenes, server, api, job, page, detail };
}

function requested(h, suffix) { return h.requests.filter((request) => request.path.endsWith(suffix)); }
function selected(h) { return h.root.find("photographic-scene").value; }

export { CATALOGUE_SHA, NEXT_CATALOGUE_SHA, PAGE_SIZE, WEBSITE_PRESET, Element, settle, json, error, storageFixture, makeScene, fixture, requested, selected };
