import { test, expect } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { crc32, deflateSync } from "node:zlib";
import { expectNoHorizontalOverflow, expectNoWcagViolations } from "./helpers/accessibility.mjs";

const assets = new URL("../../../../public/portal-assets/", import.meta.url);
const diagnostics = new WeakMap();
async function expectCoordinate(control, expected) {
  await expect.poll(async () => Number(await control.inputValue())).toBeCloseTo(expected, 12);
}
async function expectPointerCoordinate(control, expected) {
  // Chromium quantizes CSS pointer coordinates; tolerate less than 0.01 of this fixture's 2000 source pixels.
  await expect.poll(async () => Math.abs(Number(await control.inputValue()) - expected) * 2000).toBeLessThan(0.01);
}
async function centerRegion(control) {
  await control.evaluate((node) => node.scrollIntoView({ block: "center", inline: "center", behavior: "instant" }));
  await expect.poll(() => control.evaluate((node) => {
    const box = node.getBoundingClientRect();
    return node.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
  })).toBe(true);
}
test.beforeEach(({ page }) => {
  const detail = { errors: [], failedAssets: [] }; diagnostics.set(page, detail);
  page.on("pageerror", (error) => detail.errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400 && /\/(?:portal\/assets|__mock-portal)\//.test(response.url())) detail.failedAssets.push({ url: response.url(), status: response.status() });
  });
});
test.afterEach(async ({ page }, info) => {
  if (info.status !== info.expectedStatus) await info.attach("workspace diagnostics", { contentType: "application/json", body: Buffer.from(JSON.stringify(diagnostics.get(page))) });
});
function png(width, height) {
  const chunk = (name, body) => {
    const kind = Buffer.from(name), size = Buffer.alloc(4), checksum = Buffer.alloc(4);
    size.writeUInt32BE(body.length); checksum.writeUInt32BE(crc32(Buffer.concat([kind, body])));
    return Buffer.concat([size, kind, body, checksum]);
  };
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 2;
  const rows = Buffer.alloc((width * 3 + 1) * height, 128);
  for (let y = 0; y < height; y += 1) rows[y * (width * 3 + 1)] = 0;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header), chunk("IDAT", deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]);
}
const reply = (route, body) => route.fulfill({ contentType: "application/json", body: JSON.stringify(body) });
async function workspace(page, { thumbnails = false, sceneCount = 1 } = {}) {
  const variant = { role: "final", artifact_path: "scene-one/delivery.tif", sha256: "a".repeat(64), width: 2000, height: 1000, bit_depth: 16, color_profile: "sRGB",
    presentation_policy_sha256: "b".repeat(64), presentation: { available_views: ["clip_srgb", "recorded"], policy_sha256: "b".repeat(64), policy: { stage: "display_referred" } } };
  const scene = { scene_id: "scene-one", label: "Living room", variants: [variant] };
  const scenes = sceneCount === 1 ? [scene] : Array.from({ length: sceneCount }, (_, index) => ({
    scene_id: `scene-${index}`, label: `Room ${index}`, variants: [{ ...variant, artifact_path: `scene-${index}/delivery.tif` }] }));
  const job = { id: "job-workspace", pipeline: "lux-depth", state: "succeeded", progress: 100, created_at: "2026-11-16T10:00:00Z",
    logs_tail: [], artifacts: { items: scenes.map((row) => ({ ...row.variants[0], path: row.variants[0].artifact_path,
      relative_path: row.variants[0].artifact_path, media_kind: "image", artifact_type: "image", previewable: false, browser_previewable: false,
      url: `/v1/jobs/job-workspace/artifacts/${row.variants[0].artifact_path}`, size_bytes: 2048, content_type: "image/tiff" })) } };
  const catalogue = "c".repeat(64), mutations = [], thumbnailRequests = [], detailRequests = [];
  let activeThumbnails = 0, peakThumbnails = 0;
  const saved = { state_version: 1, catalogue_identity_sha256: catalogue, decisions: [], groups: [], presets: [{ schema: "tp.photo.export_preset.v1", name: "website", format: "jpeg", intended_use: "Website", long_edge: 2560, jpeg_quality: 92, sharpen_amount: 0.2, render: "perceptual_srgb", shoulder: 0.8 }] };
  await page.addInitScript(() => { sessionStorage.clear(); localStorage.removeItem("tp_portal_transient_draft"); });
  await page.route(/\/(?:__mock-portal|portal\/assets)\/portal(?:-[a-z0-9-]+)?\.js(?:\?.*)?$/, async (route) => {
    await route.fulfill({ contentType: "text/javascript", body: await readFile(new URL(new URL(route.request().url()).pathname.split("/").pop(), assets)) });
  });
  await page.route("**/portal/bootstrap", (route) => reply(route, { authMode: "managed", csrfToken: "workspace-csrf",
    actor: { username: "smoke-admin", accessEmail: "smoke-admin@local.invalid", role: "admin" },
    features: { artifactViewerModal: true, reviewSurfaceDeferred: true, photographicPagedCatalogue: true, photographicCatalogueThumbnails: thumbnails } }));
  await page.route(/\/(?:healthz|ready)$/, (route) => route.fulfill({ body: "ok" }));
  await page.route("**/v1/**", async (route) => {
    const request = route.request(), url = new URL(request.url()), path = url.pathname;
    if (request.method() === "POST") mutations.push({ path, body: request.postDataJSON() });
    if (path === "/v1/jobs") return reply(route, { success: true, data: { jobs: [job] } });
    if (path === `/v1/jobs/${job.id}`) return reply(route, { success: true, data: job });
    if (path.endsWith("/scenes")) {
      const offset = Number(url.searchParams.get("cursor") || 0);
      return reply(route, { success: true, data: { ...saved, schema: "tp.photo.scene_page.v1", scenes: scenes.slice(offset, offset + 8).map((row) => ({
        scene_id: row.scene_id, label: row.label, review_status: "unreviewed", variant_count: 1, variant_paths: [row.variants[0].artifact_path], related_artifacts: [],
        ...(thumbnails ? { thumbnail: { url: `/v1/jobs/${job.id}/photographic-review/thumbnail`, artifact_path: row.variants[0].artifact_path, sha256: variant.sha256, source_identity_sha256: variant.sha256, presentation_policy_sha256: variant.presentation_policy_sha256 } } : {}),
      })), total_scenes: scenes.length, total_filtered: scenes.length, next_cursor: offset + 8 < scenes.length ? String(offset + 8) : null } });
    }
    if (path.endsWith("/scene")) {
      detailRequests.push(url.searchParams.get("scene_id"));
      return reply(route, { success: true, data: { ...saved, schema: "tp.photo.scene_detail.v1", scenes: scenes.filter((row) => row.scene_id === url.searchParams.get("scene_id")) } });
    }
    if (path.endsWith("/thumbnail")) {
      const id = url.searchParams.get("scene_id"), source = scenes.find((row) => row.scene_id === id)?.variants[0];
      thumbnailRequests.push({ id, catalogue: url.searchParams.get("catalogue_identity_sha256"), view: url.searchParams.get("view") });
      activeThumbnails += 1; peakThumbnails = Math.max(peakThumbnails, activeThumbnails);
      await new Promise((resolve) => setTimeout(resolve, 80)); activeThumbnails -= 1;
      if (id === "scene-3") return route.fulfill({ status: 503, body: "Injected thumbnail failure" });
      return route.fulfill({ contentType: "image/png", body: png(256, 128), headers: {
        "X-TP-Catalogue-Identity-SHA256": catalogue, "X-TP-Scene-ID": encodeURIComponent(id), "X-TP-Artifact-Path": encodeURIComponent(source.artifact_path),
        "X-TP-Source-SHA256": source.sha256, "X-TP-Source-Identity-SHA256": source.sha256, "X-TP-Presentation-Policy-SHA256": variant.presentation_policy_sha256,
        "X-TP-Presentation-View": "clip_srgb", "X-TP-Display-Transform": "linear_srgb_to_8bit_srgb_clip_v1",
        "X-TP-Thumbnail-Max-Edge": "256", "X-TP-Thumbnail-Max-Bytes": "1048576", "X-TP-Thumbnail-Renderer": "tp.photo.thumbnail.v1",
        "X-TP-Thumbnail-Width": "256", "X-TP-Thumbnail-Height": "128", "X-TP-Native-Width": "2000", "X-TP-Native-Height": "1000",
      } });
    }
    if (path.endsWith("/render") || path.endsWith("/native-crop")) {
      const width = path.endsWith("/native-crop") ? Number(url.searchParams.get("width")) : 800;
      const height = path.endsWith("/native-crop") ? Number(url.searchParams.get("height")) : 400;
      return route.fulfill({ contentType: "image/png", body: png(width, height), headers: {
        "X-TP-Source-SHA256": variant.sha256, "X-TP-Source-Identity-SHA256": variant.sha256,
        "X-TP-Presentation-View": "clip_srgb", "X-TP-Presentation-Policy-SHA256": variant.presentation_policy_sha256, "X-TP-Native-Width": String(width), "X-TP-Native-Height": String(height),
      } });
    }
    if (path === "/v1/readiness") return reply(route, { success: true, data: { server: { status: "ready" }, pipelines: { "lux-depth-v3": { status: "ready", missing_prerequisites: [] } } } });
    return reply(route, { success: true, data: {} });
  });
  await page.goto(`/portal?view=review&job=${job.id}`);
  await expect(page.locator('[data-ui="photographic-workspace"]')).toBeVisible();
  await expect(page.locator('[data-ui="photographic-primary-image"]')).toBeVisible();
  return { mutations, thumbnailRequests, detailRequests, peakThumbnails: () => peakThumbnails };
}

for (const [width, height] of [[1440, 1000], [900, 1000], [390, 844]]) {
  test(`@portal-browser image-led review fits ${width}px and retains accessible decision controls`, async ({ page }) => {
    await page.setViewportSize({ width, height }); await workspace(page);
    const grid = page.locator('[data-ui="photographic-workspace"]');
    const columns = await grid.evaluate((node) => getComputedStyle(node).gridTemplateColumns.split(" ").length);
    expect(columns).toBe(width >= 1200 ? 3 : width >= 768 ? 2 : 1);
    await expect(page.locator('[data-ui="photographic-decision-note"]')).toBeVisible();
    await expect(page.locator('[data-ui="photographic-image-evidence"]')).not.toHaveAttribute("open", "");
    await page.locator('[data-ui="photographic-image-evidence"] summary').click();
    await expect(page.locator('[data-ui="photographic-primary-properties"]')).toContainText("SHA-256");
    await expect(page.locator('[data-ui="photographic-delivery-selection-summary"]')).toContainText("excluded");
    await expectNoHorizontalOverflow(page);
    await expectNoWcagViolations(page, '[data-ui="photographic-review"]');
  });
}

test("@portal-browser pointer and keyboard region edits share source-normalized numeric fields without dispatch", async ({ page }, info) => {
  await page.setViewportSize({ width: 1600, height: 1100 }); const state = await workspace(page);
  await page.locator('[data-ui="photographic-add-material"]').click();
  const move = page.locator('[data-ui="photographic-region-move"]');
  await expect(move).toBeVisible();
  const left = page.locator('[data-ui="photographic-material-left"]');
  const right = page.locator('[data-ui="photographic-material-right"]');
  await move.focus(); await page.keyboard.press("ArrowRight");
  await expectCoordinate(left, 0.2505); await expectCoordinate(right, 0.7505);
  await move.evaluate((control) => {
    window.__workspacePointerTrace = [];
    for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel", "gotpointercapture", "lostpointercapture"]) control.addEventListener(type, (event) => {
      window.__workspacePointerTrace.push({ type, pointerId: event.pointerId, button: event.button, x: event.clientX, y: event.clientY,
        captured: control.hasPointerCapture(event.pointerId), interaction: control.closest('[data-ui="photographic-region-overlay"]').dataset.interaction });
    });
  });
  await centerRegion(move);
  const image = await page.locator('[data-ui="photographic-primary-image"]').boundingBox();
  const box = await move.boundingBox();
  diagnostics.get(page).drag = { image, box, hit: await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.outerHTML, { x: box.x + box.width / 2, y: box.y + box.height / 2 }) };
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + image.width / 10, box.y + box.height / 2); await page.mouse.up();
  diagnostics.get(page).pointer = await page.evaluate(() => window.__workspacePointerTrace);
  const trace = JSON.stringify(diagnostics.get(page), null, 2);
  await writeFile(info.outputPath("pointer-trace.json"), trace);
  await expectPointerCoordinate(left, 0.3505); await expectPointerCoordinate(right, 0.8505);
  const corner = page.locator('[data-ui="photographic-region-se"]');
  const beforeResize = Number(await right.inputValue());
  await corner.focus(); await page.keyboard.press("Shift+ArrowLeft");
  await expectCoordinate(right, beforeResize - 10 / 2000);
  await centerRegion(move);
  const beforeCancelLeft = Number(await left.inputValue());
  const resized = await move.boundingBox();
  await page.mouse.move(resized.x + resized.width / 2, resized.y + resized.height / 2); await page.mouse.down();
  await page.mouse.move(resized.x + resized.width / 2 + image.width / 20, resized.y + resized.height / 2);
  await expectPointerCoordinate(left, 0.4005);
  await page.keyboard.press("Escape"); await page.mouse.up();
  await expectCoordinate(left, beforeCancelLeft);
  await left.fill("0.2");
  await expect(move).toBeVisible();
  await expect(page.locator('[data-ui="photographic-draft-status"]')).toContainText("Unsaved");
  await expect(page.locator('[data-ui="photographic-create-revision"]')).toBeDisabled();
  expect(state.mutations.filter((entry) => /photographic-review/.test(entry.path))).toEqual([]);
  await page.locator('[data-ui="photographic-region-overlays-toggle"]').click();
  await expect(move).not.toBeVisible(); await expect(left).toBeVisible();
  await page.locator('[data-ui="photographic-region-overlays-toggle"]').click();
  await page.locator('[data-ui="photographic-inspection"]').selectOption("native");
  await expect(page.locator('[data-ui="photographic-primary-image"]')).toHaveJSProperty("naturalWidth", 512);
  const beforeNativeRight = Number(await right.inputValue());
  await move.focus(); await page.keyboard.press("ArrowRight");
  await expectCoordinate(left, 0.2005); await expectCoordinate(right, beforeNativeRight + 1 / 2000);
  const expectedRecipeRegion = await Promise.all([left.inputValue(), right.inputValue()]);
  await page.locator('[data-ui="photographic-preview-revision"]').click();
  await expect.poll(() => state.mutations.filter((entry) => entry.path.endsWith("/revisions")).length).toBe(1);
  const preview = state.mutations.find((entry) => entry.path.endsWith("/revisions")).body;
  expect(preview.sha256).toBe("a".repeat(64)); expect(preview.source_identity_sha256).toBe("a".repeat(64));
  expect(preview.preview_only).toBe(true);
  expect(preview.recipe.materials[0].region).toEqual([Number(expectedRecipeRegion[0]), 0.25, Number(expectedRecipeRegion[1]), 0.75]);
  await expect(page.locator('[data-ui="photographic-review-status"]')).toContainText("identity or recipe could not be verified");
  await expect(page.locator('[data-ui="photographic-create-revision"]')).toBeDisabled();
});

test("@portal-browser paged thumbnails keep two request slots, never prefetch adjacent detail and retain failed-card navigation", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  const state = await workspace(page, { thumbnails: true, sceneCount: 12 });
  await expect(page.locator('[data-ui="photographic-page-thumbnail"]').first()).toBeVisible();
  await expect.poll(() => state.thumbnailRequests.length).toBeGreaterThan(0);
  expect(state.peakThumbnails()).toBeLessThanOrEqual(2);
  expect(state.detailRequests).toEqual(["scene-0"]);
  expect(state.thumbnailRequests.every((row) => Number(row.id.split("-")[1]) < 8)).toBe(true);
  const failed = page.locator('[data-scene-id="scene-3"] [data-ui="photographic-contact-item"]');
  await failed.scrollIntoViewIfNeeded(); await expect(failed).toHaveAttribute("data-thumbnail-state", "failed");
  await failed.click(); await expect(page.locator('[data-ui="photographic-scene"]')).toHaveValue("scene-3");
  const rail = page.locator('[data-ui="photographic-contact-sheet"]');
  await rail.evaluate((node) => { node.scrollTop = node.scrollHeight; });
  await expect.poll(() => rail.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);
  await page.locator('[data-ui="photographic-page-next"]').click();
  await expect(page.locator('[data-ui="photographic-scene"]')).toHaveValue("scene-8");
  await expect.poll(() => rail.evaluate((node) => node.scrollTop)).toBe(0);
  await expect(page.locator('[data-scene-id="scene-8"] [data-ui="photographic-contact-item"]')).toBeInViewport({ ratio: 1 });
  await expect(page.locator('[data-ui="photographic-page-thumbnail"]').first()).toBeVisible();
  expect(state.peakThumbnails()).toBeLessThanOrEqual(2);
  expect(state.detailRequests).toEqual(["scene-0", "scene-3", "scene-8"]);
  expect(state.thumbnailRequests.every((row) => row.view === "clip_srgb" && row.catalogue === "c".repeat(64))).toBe(true);
});
