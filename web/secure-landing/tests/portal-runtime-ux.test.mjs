import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const source = (name) => readFileSync(path.resolve(__dirname, `../portal-src/${name}`), "utf8");
const portal = source("portal.template.js");
const operate = source("operate-surface-deferred.js");
const review = source("review-surface-deferred.js");
const profile = source("profile-surface-deferred.js");
const buildScript = readFileSync(path.resolve(__dirname, "../scripts/build-portal-bundle.mjs"), "utf8");

function between(text, startNeedle, endNeedle) {
  const start = text.indexOf(startNeedle);
  assert.notEqual(start, -1, `missing start marker: ${startNeedle}`);
  const end = text.indexOf(endNeedle, start + startNeedle.length);
  assert.notEqual(end, -1, `missing end marker: ${endNeedle}`);
  return text.slice(start, end);
}

test("workspace navigation uses page-current semantics and keeps Review reachable empty", () => {
  const activeLink = between(portal, "function setActiveWorkspaceLink", "function _portalStatusRegion");
  const rail = between(portal, "function setupSectionRail", "function _minimumBuildStep");
  const navigate = between(portal, "function navigateConsoleView", "function applyConsoleRouteFromLocation");

  assert.match(activeLink, /removeAttribute\('aria-selected'\)/);
  assert.match(activeLink, /setAttribute\('aria-current', 'page'\)/);
  assert.doesNotMatch(rail, /nextView === 'review'.*selectedJobId/s);
  assert.match(navigate, /_focusConsoleViewHeading\(state\.currentView\)/);
  assert.match(portal, /heading\.scrollIntoView\(\{ behavior: 'instant', block: 'center', inline: 'nearest' \}\)/);
  assert.match(portal, /Promise\.allSettled\(pending\)/);
});

test("Build steps use toolbar buttons instead of false multi-panel tab semantics", () => {
  const sync = between(portal, "function syncBuildStepUi", "function setBuildStep");

  assert.match(sync, /button\.setAttribute\('aria-pressed', active \? 'true' : 'false'\)/);
  assert.match(sync, /button\.setAttribute\('aria-current', 'step'\)/);
  assert.match(sync, /panel\.removeAttribute\('role'\)/);
  assert.match(sync, /panel\.removeAttribute\('tabindex'\)/);
  assert.doesNotMatch(sync, /role', 'tabpanel'/);
  assert.match(sync, /button\.removeAttribute\('aria-selected'\)/);
});

test("deferred surface loading stays scoped to its visible workspace", () => {
  const queueVisible = between(portal, "function _isOperateQueuePanelVisible", "function _shouldLoadDeferredOperateSurface");
  const shouldLoad = between(portal, "function _shouldLoadDeferredOperateSurface", "function _primeDeferredOperateSurface");

  assert.match(queueVisible, /state\.currentView === 'operate'/);
  assert.match(queueVisible, /els\.jobsShell/);
  assert.match(queueVisible, /!els\.jobsShell\.classList\.contains\('hidden'\)/);
  assert.match(shouldLoad, /state\.currentView === 'operate'\s*\|\| state\.currentView === 'review'/);
  assert.match(shouldLoad, /state\.portalUi\.dispatchPending \|\| Boolean\(state\.portalUi\.dispatchHandoffJobId\)/);
});

test("deferred profile opening is canceled by navigation, hidden triggers, and active overlays", () => {
  const handler = between(portal, "if (els.saveProfileBtn)", "if (els.exportBtn)");
  const overlayFocus = between(portal, "function _isPortalFocusTargetAvailable", "function _setPortalBackgroundInert");

  assert.match(handler, /const openIntent = \+\+deferredProfileOpenIntent/);
  assert.match(handler, /openIntent !== deferredProfileOpenIntent/);
  assert.match(handler, /state\.currentView !== 'build'/);
  assert.match(handler, /_activeOverlayPanel\(\)/);
  assert.match(handler, /_isPortalFocusTargetAvailable\(trigger\)/);
  assert.match(overlayFocus, /closest\?\.\('\[hidden\], \[aria-hidden="true"\], \.hidden'\)/);
  assert.match(overlayFocus, /_focusConsoleViewHeading\(state\.currentView\)/);
});

test("poll-driven health and empty Review copy avoid unchanged live-region mutations", () => {
  const health = between(portal, "async function checkBackend", "async function extractApiError");
  const emptyState = between(portal, "function _setTextContentIfChanged", "// ============================================================================\n// 9. JOB RENDERING");
  const artifactPanel = between(review, "function renderArtifactPanel", "function _artifactViewerContext");

  assert.match(health, /_setTextContentIfChanged\(els\.healthText, 'Backend Online'\)/);
  assert.match(health, /_setTextContentIfChanged\(els\.healthText, 'Backend Offline'\)/);
  assert.doesNotMatch(health, /healthText\.textContent\s*=/);
  assert.match(emptyState, /element\.textContent === nextText/);
  assert.match(artifactPanel, /setTextContentIfChanged\(els\.emptyArtifactAction, emptyCopy\.action\)/);
  assert.doesNotMatch(artifactPanel, /emptyArtifactAction\.textContent\s*=/);
});

test("queue rows retain keyed semantics and expose separate inspect and cancel actions", () => {
  const queue = between(operate, "function renderJobQueue", "function scheduleRenderJobQueue");
  const displayState = between(portal, "function _displayJobState", "function _displayJobStateTone");

  assert.match(queue, /setAttribute\('role', 'list'\)/);
  assert.match(queue, /existingRows = new Map/);
  assert.match(queue, /inspectButton\.dataset\.action = 'inspect-job'/);
  assert.match(queue, /inspectButton\.setAttribute\('aria-pressed'/);
  assert.match(queue, /cancelButton\.dataset\.action = 'cancel-job'/);
  assert.match(queue, /job\.cancelPending \? 'Canceling…' : job\.cancelError \? 'Retry Cancel'/);
  assert.match(queue, /replaceChildren\(fragment\)/);
  assert.doesNotMatch(queue, /li\.setAttribute\('role', 'option'\)/);
  assert.doesNotMatch(queue, /aria-live.*jobList/);
  assert.match(displayState, /rawState === 'canceled'\) return 'canceled'/);
  assert.doesNotMatch(displayState, /rawState === 'failed' \|\| rawState === 'canceled'/);
});

test("Review uses button selection semantics and contextual image alternatives", () => {
  const artifactPanel = between(review, "function renderArtifactPanel", "function _artifactViewerContext");

  assert.match(artifactPanel, /setAttribute\(["']role["'], ["']group["']\)/);
  assert.match(artifactPanel, /button\.setAttribute\(["']aria-pressed["']/);
  assert.match(artifactPanel, /artifactPreviewImage\.alt =/);
  assert.match(artifactPanel, /artifactCompareImage\.alt =/);
  assert.doesNotMatch(artifactPanel, /setAttribute\(["']role["'], ["']option["']\)/);
  assert.doesNotMatch(artifactPanel, /setAttribute\(["']aria-selected["']/);
});

test("dispatch and copy shortcuts are scoped to ready Build step four", () => {
  const ready = between(portal, "function _isBuildStepFourShortcutReady", "function _artifactViewerContext");
  const keydown = between(portal, "document.addEventListener('keydown'", "document.addEventListener('visibilitychange'");

  assert.match(ready, /state\.currentView !== 'build'/);
  assert.match(ready, /resolveBuildStep\(state\.portalUi\.buildStep\) !== 4/);
  assert.match(ready, /_isTypingTarget\(target\) \|\| _activeOverlayPanel\(\)/);
  assert.match(ready, /state\.portalUi\.dispatchPending/);
  assert.match(keydown, /_isBuildStepFourShortcutReady\(e\.target, 'dispatch'\)/);
  assert.match(keydown, /_isBuildStepFourShortcutReady\(e\.target, 'copy-cli'\)/);
  assert.match(keydown, /!_activeOverlayPanel\(\).*WORKSPACE_VIEW_SHORTCUTS/s);
});

test("deferred profile manager validates names and confirms destructive and legacy import actions", () => {
  assert.match(portal, /profile: \{ datasetKey: 'profileSurfaceJsUrl'/);
  assert.match(profile, /function nameValidationMessage/);
  assert.match(profile, /PROFILE_NAME_MAX_LENGTH = 64/);
  assert.match(profile, /'overwrite'/);
  assert.match(profile, /'rename-overwrite'/);
  assert.match(profile, /'delete'/);
  assert.match(profile, /'import-legacy'/);
  assert.match(profile, /Confirm Claim & Import/);
  assert.match(profile, /hasProtectedChanges/);
  assert.match(profile, /missingActiveProfile/);
  assert.match(profile, /if \(!hasProfile\(profiles, name\)\)/);
  assert.match(
    portal,
    /_isBootstrapReady\(\) \|\| pendingLegacyTransientDraft \|\| state\.currentView !== 'build'/
  );
  assert.match(portal, /transientDraftRestoredForProfile = _restoreTransientPortalDraft\(\)/);
  assert.match(profile, /dataset\.profileState/);
  assert.match(profile, /dataset\.draftState/);
  assert.match(profile, /setPortalBackgroundInert\(true\)/);
  assert.doesNotMatch(profile, /\b(?:prompt|confirm)\s*\(/);
});

test("deprecated Lux output keys migrate out of drafts, profiles, payloads, and CLI previews", () => {
  const migrationSource = between(
    portal,
    "function _migrateDeprecatedLuxOutputConfig",
    "function _copyTransientDraftConfig",
  );
  const migrate = new Function(`
    function createToast() {}
    function parseBoolLike(value, fallback = false) {
      if (typeof value === 'boolean') return value;
      return ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase()) || fallback;
    }
    ${migrationSource}; return _migrateDeprecatedLuxOutputConfig;
  `)();
  const legacyConfig = {
    emits: { master16: true, marketing: true, report: false, runCard: true },
    emit_marketing: true,
    emit_report: false,
    emitMarketing: true,
    emitReport: false,
  };

  const migrated = migrate(structuredClone(legacyConfig));

  assert.deepEqual(migrated.emits, { runCard: true });
  assert.equal(migrated.outputBitDepth, 16);
  for (const key of ["emit_marketing", "emit_report", "emitMarketing", "emitReport"]) {
    assert.equal(Object.hasOwn(migrated, key), false);
  }
  const conflicting = migrate({ output_bit_depth: 8, emit_master16: true });
  assert.equal(conflicting.emit_master16, true);
  const disabled = migrate({ emit_upscaled16: false });
  assert.equal(disabled.output_bit_depth, 8);
  assert.equal(disabled.emit_master16, false);
  const camelCase = migrate({ emitUpscaled16: "on" });
  assert.equal(camelCase.output_bit_depth, 16);
  assert.equal(camelCase.emit_master16, true);
  assert.equal(Object.hasOwn(camelCase, "emitUpscaled16"), false);
  assert.match(profile, /state\.config = copyDraftConfig\(profiles\[name\]\.config\)/);
  assert.match(portal, /\[deprecated_output_flag\] Report on; no marketing\./);
  assert.match(portal, /_migrateDeprecatedLuxOutputConfig\(data\.args\)/);

  const canonicalArgs = between(portal, "function buildCanonicalLuxDepthArgs", "function generatePayload");
  const cliPreview = between(portal, "function renderCLI", "function bindInputs");
  for (const deprecatedToken of ["emit_marketing", "emit_report", "--emit-marketing", "--emit-report"]) {
    assert.doesNotMatch(canonicalArgs, new RegExp(deprecatedToken));
    assert.doesNotMatch(cliPreview, new RegExp(deprecatedToken));
  }
});

test("bundle compaction preserves diagnostic names and property keys", () => {
  const compact = between(buildScript, "const compactPortalBundle", "const portalChanged");
  const operateBuild = between(buildScript, "const deferredOperateSurfaceBuild", "const deferredBuildSurfaceBuild");

  assert.match(compact, /keepNames: true/);
  assert.match(compact, /minifyIdentifiers: true/);
  assert.doesNotMatch(compact, /mangleProps|mangleQuoted/);
  assert.match(operateBuild, /keepNames: true/);
  assert.match(operateBuild, /minifyIdentifiers: true/);
  assert.match(buildScript, /PORTAL_PROFILE_SURFACE_ENTRY/);
  assert.match(buildScript, /PORTAL_PROFILE_SURFACE_ASSET_PATH/);
});

test("bounded response readers keep the request timeout active through body consumption", async () => {
  const helperSource = between(portal, "async function fetchWithTimeout", "function _portalRumNow");
  const loadHelpers = new Function(`${helperSource}; return { fetchWithTimeout, fetchBodyWithTimeout };`);
  const { fetchBodyWithTimeout } = loadHelpers();
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (_url, options) => ({
    ok: true,
    text: () => new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => {
        const error = new Error("aborted while reading response body");
        error.name = "AbortError";
        reject(error);
      }, { once: true });
    }),
  });

  try {
    await assert.rejects(
      fetchBodyWithTimeout("https://portal.invalid/cancel", {}, 15, "body_timeout", null),
      (error) => error?.name === "AppTimeoutError" && error?.reason === "body_timeout",
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
