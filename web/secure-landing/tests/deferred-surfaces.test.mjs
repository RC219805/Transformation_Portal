import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORTAL_TEMPLATE_PATH = path.resolve(__dirname, "../portal-src/portal.template.js");
const OPERATE_SURFACE_PATH = path.resolve(__dirname, "../portal-src/operate-surface-deferred.js");

const portalTemplate = readFileSync(PORTAL_TEMPLATE_PATH, "utf8");
const operateSurface = readFileSync(OPERATE_SURFACE_PATH, "utf8");

function sourceBetween(source, startNeedle, endNeedle) {
  const start = source.indexOf(startNeedle);
  assert.notEqual(start, -1, `missing start marker: ${startNeedle}`);
  const end = source.indexOf(endNeedle, start + startNeedle.length);
  assert.notEqual(end, -1, `missing end marker: ${endNeedle}`);
  return source.slice(start, end);
}

test("deferred Build load reconciles state fetched before lazy import", () => {
  const helper = sourceBetween(
    portalTemplate,
    "function _reconcileDeferredBuildSurface",
    "function _primeDeferredBuildSurface"
  );

  assert.match(helper, /api\.applyLuxMetadataToControls/);
  assert.match(helper, /api\.renderFieldPreviewStatuses/);
  assert.match(helper, /api\.syncRuntimeWorkerModeControls/);
  assert.match(helper, /api\.refreshArchiveFieldVisibility/);

  const prime = sourceBetween(
    portalTemplate,
    "function _primeDeferredBuildSurface",
    "function _renderDeferredReviewSurfaceFallback"
  );
  assert.match(
    prime,
    /_loadDeferredBuildSurface\(\)\.then\(\(loaded\) => \{\s*_reconcileDeferredBuildSurface\(loaded\);/
  );
});

test("renderJobQueue does not import Operate surface outside active Operate views", () => {
  const shouldLoad = sourceBetween(
    portalTemplate,
    "function _shouldLoadDeferredOperateSurface()",
    "function _primeDeferredOperateSurface"
  );
  assert.match(shouldLoad, /if \(!_isBootstrapReady\(\)\) return false;/);
  assert.match(shouldLoad, /_isOperateQueuePanelVisible\(\)/);

  const renderJobQueue = sourceBetween(
    portalTemplate,
    "function renderJobQueue(includeReviewSurfaces = true)",
    "function handleJobListKeydown"
  );

  const guardIndex = renderJobQueue.indexOf("if (!_shouldLoadDeferredOperateSurface()) return;");
  const loadIndex = renderJobQueue.indexOf("_loadDeferredOperateSurface()");
  assert.ok(guardIndex >= 0, "renderJobQueue must guard deferred Operate load");
  assert.ok(loadIndex >= 0, "renderJobQueue must still lazy-load Operate when eligible");
  assert.ok(guardIndex < loadIndex, "Operate load guard must run before import");
});

test("Operate deferred host does not expose unused SAFE_JOB_STATES wiring", () => {
  const host = sourceBetween(
    portalTemplate,
    "function _createDeferredOperateSurfaceHost()",
    "function _deferredOperateSurfaceApi()"
  );

  assert.doesNotMatch(host, /SAFE_JOB_STATES/);
  assert.doesNotMatch(operateSurface, /SAFE_JOB_STATES/);
  assert.doesNotMatch(operateSurface, /\bsafeState\b/);
});
