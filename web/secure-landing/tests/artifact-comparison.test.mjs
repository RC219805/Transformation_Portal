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
