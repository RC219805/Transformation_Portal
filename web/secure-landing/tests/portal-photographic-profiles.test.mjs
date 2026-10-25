import test from "node:test";
import assert from "node:assert/strict";
import { createPortalConfigState } from "../portal-src/internal/state.js";
import { applyProcessingProfile, resolveProcessingProfile, processingProfileGuidance } from "../portal-src/internal/photographic-profiles.js";
import { createDeferredBuildSurfaceApi } from "../portal-src/build-surface-deferred.js";

test("new photographic master intent requests16 bits and explicit saved8 stays8", () => {
  const config = createPortalConfigState();
  assert.equal(config.outputBitDepth, 16);
  assert.equal(resolveProcessingProfile(config), "photographic_master");
  const historical = { outputBitDepth: 8 };
  assert.equal(resolveProcessingProfile(historical), "web_delivery");
  assert.equal(historical.outputBitDepth, 8);
  applyProcessingProfile(config, "web_delivery");
  assert.equal(config.outputBitDepth, 8);
  applyProcessingProfile(config, "retouch_print");
  assert.equal(config.outputBitDepth, 16);
  assert.match(processingProfileGuidance("retouch_print", "lux-depth-v3"), /16-bit.*Unified\/V5\/V6.*float32/);
  assert.match(processingProfileGuidance("photographic_master", "lux-depth-v6"), /float32 extended linear sRGB/);
  assert.throws(() => applyProcessingProfile(config, "imaginary"), /known photographic/);
});

test("deferred Build profile changes update canonical encoding and refresh preview once", () => {
  let change;
  let previews = 0;
  let saves = 0;
  let renders = 0;
  const state = { pipeline: "lux-depth-v3", config: createPortalConfigState() };
  const els = { processingProfile: { addEventListener: (name, handler) => { assert.equal(name, "change"); change = handler; } },
    processingProfileHint: {}, emits: { outputBitDepth: {} } };
  const api = createDeferredBuildSurfaceApi({ state, els, renderCLI: () => renders++,
    scheduleConfigPreview: () => previews++, persistDraft: () => saves++ });
  const before = JSON.stringify(state);
  api.syncPhotoProfiles();
  api.syncPhotoProfiles();
  assert.equal(JSON.stringify(state), before);
  assert.equal(els.processingProfile.value, "photographic_master");
  change({ target: { value: "web_delivery" } });
  assert.equal(state.config.outputBitDepth, 8);
  assert.equal(els.emits.outputBitDepth.value, "8");
  assert.match(els.processingProfileHint.textContent, /8-bit encoded-sRGB PNG/);
  assert.deepEqual([previews, saves, renders], [1, 1, 1]);
});
