import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import * as artifactSecurity from "../portal-src/internal/artifact-security.js";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
const helpers = source.slice(source.indexOf("function artifactMediaKind("), source.indexOf("function _timelineEntry("));
const context = vm.createContext({ URL, window: { location: { origin: "https://portal.example" } }, portalInternals: artifactSecurity, API_BASE: "" });
vm.runInContext(helpers, context);
const artifact = (path, options = {}) => ({ path, relative_path: path, media_kind: "image", previewable: true, browser_previewable: true, url: `/v1/jobs/job-v5/artifacts/${path}`, ...options });
const preview = artifact("input-0000/preview.png");
const delivery = artifact("input-0000/delivery.tif", { browser_previewable: false, preview_url: preview.url });

test("Review excludes a TIFF's own browser preview in both selection directions", () => {
  assert.equal(context.findCompareArtifact(delivery, [delivery, preview]), null);
  assert.equal(context.findCompareArtifact(preview, [delivery, preview]), null);
});

test("same-preview exclusion applies before explicit comparison-group ranking", () => {
  const hint = { compare_group: "photography", priority: 1000 };
  const groupedDelivery = { ...delivery, display_hint: hint };
  const groupedPreview = { ...preview, display_hint: hint };
  assert.equal(context.findCompareArtifact(groupedDelivery, [groupedDelivery, groupedPreview]), null);
  assert.equal(context.findCompareArtifact(groupedPreview, [groupedDelivery, groupedPreview]), null);
});

test("Review retains distinct comparisons while skipping same-proxy candidates", () => {
  const distinct = artifact("input-0000/source.png");
  assert.equal(context.findCompareArtifact(delivery, [delivery, preview, distinct]), distinct);
  assert.equal(context.findCompareArtifact(preview, [delivery, preview, distinct]), distinct);
  const grouped = { ...distinct, display_hint: { compare_group: "photography", priority: 500 } };
  const groupedDelivery = { ...delivery, display_hint: { compare_group: "photography", priority: 1000 } };
  assert.equal(context.findCompareArtifact(groupedDelivery, [groupedDelivery, preview, grouped]), grouped);
});

test("Review normalizes same-origin absolute proxy URLs and does not equate invalid or missing URLs", () => {
  assert.equal(context.findCompareArtifact({ ...delivery, preview_url: `https://portal.example${preview.url}` }, [preview]), null);
  assert.equal(context._artifactsSharePreviewUrl(artifact("first.png", { url: "" }), artifact("second.png", { url: "" })), false);
  assert.equal(context._artifactsSharePreviewUrl({ preview_url: "https://outside.example/x" }, { url: "https://outside.example/x" }), false);
});

test("explicit comparison groups prevent fallback from photographs to depth or validity maps", () => {
  const photo = { ...delivery, display_hint: { compare_group: 'input-0000/photography', priority: 1200 } };
  const photoProxy = { ...preview, display_hint: { compare_group: 'input-0000/photography', priority: 1100 } };
  const depth = artifact('input-0000/depth-relative.tif', { preview_url: '/depth-preview.png', display_hint: { compare_group: 'input-0000/depth-relative', priority: 650 } });
  const validity = artifact('input-0000/depth-preview-valid.png', { display_hint: { compare_group: 'input-0000/validity', priority: 550 } });
  const ungrouped = artifact('input-0000/unrelated.png');
  const artifacts = [photo, photoProxy, depth, validity, ungrouped];
  for (const primary of [photo, photoProxy, depth, validity]) {
    assert.equal(context.findCompareArtifact(primary, artifacts), null);
  }
});

test("comparison copy identifies the selected artifacts without inferring before and after roles", () => {
  vm.runInContext(source.slice(source.indexOf("function _compareSurfaceCopy("), source.indexOf("function _findJobById(")), context);
  const selected = artifact("input-0000/final.png", { display_hint: { role: "primary_preview", compare_group: "photograph" } });
  const comparison = artifact("input-0000/source.png", { display_hint: { role: "supporting_preview", compare_group: "photograph" } });
  const copy = context._compareSurfaceCopy(selected, comparison, true);
  assert.equal(copy.summaryTitle, "Side-by-side comparison");
  assert.equal(copy.summaryDetail, "Selected: input-0000/final.png. Comparison: input-0000/source.png.");
  assert.doesNotMatch(copy.summaryDetail, /Before|After/);
});

test("download accessibility describes the full selected file while retaining its original target", () => {
  const review = readFileSync(new URL("../portal-src/review-surface-deferred.js", import.meta.url), "utf8");
  const downloadBlock = review.slice(review.indexOf("if (els.downloadArtifactBtn) {"), review.indexOf("if (els.copyArtifactPathBtn) {"));
  const renderDownload = new Function("els", "selectedArtifact", "selected", "buildArtifactUrl", "artifactNameParts", "formatBytes", downloadBlock);
  const button = {
    dataset: {}, attributes: new Map(), disabled: true,
    setAttribute(name, value) { this.attributes.set(name, value); },
    removeAttribute(name) { this.attributes.delete(name); },
  };
  const els = { downloadArtifactBtn: button };
  renderDownload(els, { ...delivery, size_bytes: 4096 }, { id: "job-v5" }, context.buildArtifactUrl, context.artifactNameParts, (bytes) => `${bytes} bytes`);
  assert.equal(button.disabled, false);
  assert.equal(button.dataset.url, delivery.url, "TIFF download must not switch to its PNG display derivative");
  assert.equal(button.attributes.get("aria-label"), "Download full file: delivery.tif (4096 bytes)");

  const resetSource = source.slice(source.indexOf("function _resetArtifactActionButtons("), source.indexOf("function _reviewSurfaceDeferredEnabled("));
  new Function("els", `${resetSource}; _resetArtifactActionButtons();`)(els);
  assert.equal(button.disabled, true);
  assert.equal(button.dataset.url, undefined);
  assert.equal(button.attributes.has("aria-label"), false, "a previous artifact filename must not survive clearing selection");
});
