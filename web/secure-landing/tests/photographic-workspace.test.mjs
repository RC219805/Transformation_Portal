import test from "node:test";
import assert from "node:assert/strict";
import { photographicRegionChange, photographicDeliverySummary, createPhotographicRegionOverlay } from "../portal-src/photographic-workspace.js";
import { Element } from "./helpers/photographic-review-fixture.mjs";

test("moving a normalized source region preserves its area at all image boundaries", () => {
  assert.deepEqual(photographicRegionChange([0.25, 0.25, 0.75, 0.75], 1, -1, "move", 8000, 6000), [0.5, 0, 1, 0.5]);
  assert.deepEqual(photographicRegionChange([0, 0, 0.5, 0.5], -0.1, -0.1, "move", 8000, 6000), [0, 0, 0.5, 0.5]);
});

test("resizing enforces positive source-pixel area and never crosses the opposite corner", () => {
  assert.deepEqual(photographicRegionChange([0.25, 0.25, 0.75, 0.75], 1, 1, "nw", 100, 200), [0.74, 0.745, 0.75, 0.75]);
  assert.deepEqual(photographicRegionChange([0.25, 0.25, 0.75, 0.75], -1, -1, "se", 100, 200), [0.25, 0.25, 0.26, 0.255]);
  for (const region of [[0.5, 0, 0.5, 1], [0, -1, 1, 1], [0, 0, 2, 1], [0, 0, Infinity, 1]]) {
    assert.equal(photographicRegionChange(region, 0.1, 0, "move", 100, 100), null);
  }
});

test("keyboard increments represent source pixels for fit and native image contexts", () => {
  const initial = [0.25, 0.25, 0.75, 0.75];
  assert.deepEqual(photographicRegionChange(initial, 1 / 8000, 0, "move", 8000, 6000), [0.250125, 0.25, 0.750125, 0.75]);
  assert.deepEqual(photographicRegionChange(initial, 10 / 8000, 0, "se", 8000, 6000), [0.25, 0.25, 0.75125, 0.75]);
});

test("large source regions retain exact one-pixel movement and minimum rectangle precision", () => {
  const initial = [0.25, 0.25, 0.75, 0.75], pixel = 1 / 30000;
  const moved = photographicRegionChange(initial, pixel, 0, "move", 30000, 1000);
  assert.deepEqual(moved, [0.25 + pixel, 0.25, 0.75 + pixel, 0.75]);
  assert.equal(moved[0] * 30000, 7501);
  const minimum = photographicRegionChange(initial, 1, 1, "nw", 30000, 1000);
  assert.deepEqual(minimum, [0.75 - pixel, 0.749, 0.75, 0.75]);
  assert.equal(minimum[0] * 30000, 22499);
  assert.equal(photographicRegionChange(minimum, 0, 0, "move", 30000, 1000)?.length, 4);
});

test("region geometry rejects empty and invalid raw edges without inferring zero", () => {
  for (const invalid of ["", " ", null, undefined, true, [], {}, "invalid", NaN, Infinity]) {
    assert.equal(photographicRegionChange([invalid, 0.25, 0.75, 0.75], 1 / 30000, 0, "move", 30000, 1000), null);
  }
  assert.deepEqual(photographicRegionChange(["0.25", "0.25", "0.75", "0.75"], 1 / 30000, 0, "move", 30000, 1000),
    [0.25 + 1 / 30000, 0.25, 0.75 + 1 / 30000, 0.75]);
});

test("delivery summary exposes excluded count, products, set name and real download destination", () => {
  const summary = photographicDeliverySummary({ total: 10, selected: 3, approved: 2, products: ["website", "print"], property: "Hill House" });
  for (const expected of ["2 approved sources of 10", "8 excluded", "3 selected", "website, print", "Hill House", "ZIP download", "contact sheet", "revalidated"]) assert.ok(summary.includes(expected));
});

function overlayHarness(native = false) {
  const body = new Element("body"), document = { body, activeElement: body };
  body.ownerDocument = document;
  const el = (tag, text = "", className = "", ui = "") => {
    const node = new Element(tag, document); node.textContent = text; node.className = className; if (ui) node.dataset.ui = ui; return node;
  };
  const image = el("img"), viewport = el("div"), materials = [{ name: "Region 1", region: [0.25, 0.25, 0.75, 0.75] }];
  image.clientWidth = native ? 512 : 800; image.clientHeight = native ? 512 : 400;
  image.getBoundingClientRect = () => ({ x: 0, y: 0, width: image.clientWidth, height: image.clientHeight });
  const context = { key: "source-and-inspection", ready: true, visible: true, blocked: false, width: 2000, height: 1000,
    crop: native ? { x: 744, y: 244, width: 512, height: 512 } : { x: 0, y: 0, width: 2000, height: 1000 } };
  let edits = 0, numeric = 0;
  const api = createPhotographicRegionOverlay({ nodes: { figures: [{ image, viewport }] }, el,
    button: (text, ui, action) => { const button = el("button", text, "", ui); button.addEventListener("click", action); return button; },
    getMaterials: () => materials, context: () => context, changed: () => { edits += 1; }, syncNumeric: () => { numeric += 1; } });
  api.render();
  const control = viewport.find("photographic-region-move"), captures = new Set();
  control.setPointerCapture = (id) => captures.add(id); control.hasPointerCapture = (id) => captures.has(id); control.releasePointerCapture = (id) => captures.delete(id);
  const event = (type, patch = {}) => control.listeners[type]({ preventDefault() {}, stopPropagation() {}, pointerId: 1, button: 0, clientX: 100, clientY: 100, ...patch });
  return { api, context, materials, image, control, event, captures, layer: viewport.find("photographic-region-overlay"), counts: () => ({ edits, numeric }) };
}

test("invalid numeric edges stay untouched by overlay gestures until explicit repair", () => {
  for (const invalid of ["", " ", "invalid"]) {
    const h = overlayHarness(); h.materials[0].region[0] = invalid; h.api.sync();
    assert.equal(h.layer.children[0].hidden, true);
    h.event("keydown", { key: "ArrowRight" }); h.event("pointerdown"); h.event("pointermove", { clientX: 200 });
    assert.deepEqual(h.materials[0].region, [invalid, 0.25, 0.75, 0.75]);
    assert.deepEqual(h.counts(), { edits: 0, numeric: 0 }); assert.equal(h.captures.size, 0);
    h.materials[0].region[0] = "0.25"; h.api.sync();
    assert.equal(h.layer.children[0].hidden, false);
    h.event("keydown", { key: "ArrowRight" });
    assert.deepEqual(h.materials[0].region, [0.2505, 0.25, 0.7505, 0.75]);
    assert.deepEqual(h.counts(), { edits: 1, numeric: 1 });
  }
});

test("an invalid numeric edge entered during a drag cancels capture without restoring an inferred region", () => {
  const h = overlayHarness(); h.event("pointerdown");
  h.materials[0].region[0] = ""; h.api.sync();
  assert.equal(h.captures.size, 0); assert.equal(h.layer.dataset.interaction, "idle");
  h.event("pointermove", { clientX: 200 }); h.event("keydown", { key: "Escape", pointerId: undefined });
  assert.deepEqual(h.materials[0].region, ["", 0.25, 0.75, 0.75]);
  assert.deepEqual(h.counts(), { edits: 0, numeric: 0 });
});

for (const native of [false, true]) test(`${native ? "native crop" : "fit preview"} pointer motion uses source geometry and Escape restores the original rectangle`, () => {
  const h = overlayHarness(native);
  h.event("pointerdown"); h.event("pointermove", { clientX: 110 });
  assert.deepEqual(h.materials[0].region, native ? [0.255, 0.25, 0.755, 0.75] : [0.2625, 0.25, 0.7625, 0.75]);
  assert.deepEqual(h.counts(), { edits: 1, numeric: 1 });
  h.event("keydown", { key: "Escape", pointerId: undefined });
  assert.deepEqual(h.materials[0].region, [0.25, 0.25, 0.75, 0.75]); assert.equal(h.captures.size, 0);
  h.event("pointerup"); assert.deepEqual(h.counts(), { edits: 2, numeric: 2 });
});

test("a source-context change or pending mutation prevents stale pointer/keyboard writes", () => {
  const h = overlayHarness(); h.event("keydown", { key: "ArrowRight", ctrlKey: true }); assert.equal(h.counts().edits, 0);
  h.event("pointerdown");
  h.context.key = "another-source"; h.api.sync(); h.event("pointermove", { clientX: 200 });
  assert.equal(h.captures.size, 0);
  assert.deepEqual(h.materials[0].region, [0.25, 0.25, 0.75, 0.75]); assert.equal(h.counts().edits, 0);
  h.context.blocked = true; h.event("keydown", { key: "ArrowRight" }); h.event("pointerdown"); h.event("pointermove", { clientX: 300 });
  assert.equal(h.counts().edits, 0);
});

test("hiding overlays or changing comparison mode ends a drag and prohibits hidden control edits", () => {
  for (const hide of [(h) => { h.context.visible = false; h.api.sync(); }, (h) => h.api.toggle.listeners.click()]) {
    const h = overlayHarness(); h.event("pointerdown"); hide(h);
    h.event("pointermove", { clientX: 200 }); h.event("keydown", { key: "ArrowRight" });
    assert.equal(h.counts().edits, 0); assert.equal(h.captures.size, 0);
    assert.deepEqual(h.materials[0].region, [0.25, 0.25, 0.75, 0.75]);
  }
});

test("an unchanged ResizeObserver notification retains a drag; real image resizing ends it", () => {
  const previous = globalThis.ResizeObserver; let notify;
  globalThis.ResizeObserver = class { constructor(callback) { notify = callback; } observe() {} };
  try {
    const h = overlayHarness(); h.event("pointerdown"); notify();
    h.event("pointermove", { clientX: 110 }); assert.equal(h.counts().edits, 1); assert.equal(h.captures.size, 1);
    h.image.clientWidth += 20; notify(); h.event("pointermove", { clientX: 200 });
    assert.equal(h.counts().edits, 1); assert.equal(h.captures.size, 0);
  } finally { if (previous === undefined) delete globalThis.ResizeObserver; else globalThis.ResizeObserver = previous; }
});
