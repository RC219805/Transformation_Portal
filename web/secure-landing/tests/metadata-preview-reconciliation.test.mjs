import test, { after } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { createDeferredBuildSurfaceApi } from "../portal-src/build-surface-deferred.js";
import { createLatestRequestCoordinator } from "../portal-src/internal/latest-request.js";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
function section(start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first + start.length);
  assert.ok(first >= 0 && last > first, `missing source section: ${start}`);
  return source.slice(first, last);
}

const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
globalThis.document = { createElement: () => ({ value: "", textContent: "" }) };
after(() => {
  if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
  else delete globalThis.document;
});

function select(value) {
  let selected = value;
  const options = [{ value, textContent: value || "Default" }];
  return {
    options,
    get value() { return selected; },
    set value(next) { selected = options.some((option) => option.value === next) ? next : ""; },
    set innerHTML(_value) { options.length = 0; selected = ""; },
    appendChild(option) {
      options.push(option);
      if (options.length === 1) selected = option.value;
    },
  };
}

function buildFixture({ logLevel = "", modelKey = "da3-metric", fields = {} } = {}) {
  const state = {
    pipeline: "lux-depth-v3", currentView: "build", backendOk: true, preview: null,
    config: { modelKey, reconstruction: {}, raw: {}, runtime: { logLevel } },
    metadata: { fields }, auth: { mode: "managed" },
  };
  const els = {
    modelKey: select(modelKey), reconstruction: {}, raw: {},
    runtime: { logLevel: select(logLevel) },
  };
  // Deliberately small request fixture: the real request-key helper, scheduler,
  // transport, metadata API and stale-response guards below remain executable.
  const generatePayload = () => {
    const args = { input_dir: "./input_images", output_dir: "./output/run", model_key: els.modelKey.value };
    if (els.runtime.logLevel.value) args.log_level = els.runtime.logLevel.value;
    return { pipeline: state.pipeline, args };
  };
  const api = createDeferredBuildSurfaceApi({
    state, els, generatePayload,
    _metadataField: (name) => state.metadata.fields[name] || null,
    _normalizeWorkerMode: (mode) => mode,
    _previewIssueForField: () => null,
    _renderIssueStatus() {},
    _resolveDa3ModelKey: (value) => value,
    canonicalArchiveCommand: () => "",
    _stagedUploadsEnabledForState: () => false,
  });
  return { state, els, api, generatePayload };
}

const logOptions = [
  null, undefined, {}, { label: "Missing value" },
  { value: null, label: "Null value" }, { value: undefined, label: "Undefined value" },
  { value: "", label: "Default" }, { value: " DEBUG ", label: " Debug " },
  { value: "INFO", label: "Info" },
];

test("exported metadata API preserves explicit empty Default and omits log_level from the request", () => {
  const h = buildFixture({ fields: { log_level: { options: logOptions } } });
  const before = h.generatePayload();
  h.api.applyLuxMetadataToControls();
  assert.deepEqual(h.els.runtime.logLevel.options.map((option) => [option.value, option.textContent]),
    [["", "Default"], ["DEBUG", "Debug"], ["INFO", "Info"]]);
  assert.equal(h.els.runtime.logLevel.value, "");
  assert.equal(h.state.config.runtime.logLevel, "");
  assert.deepEqual(h.generatePayload(), before);
  assert.equal(Object.hasOwn(h.generatePayload().args, "log_level"), false);
});

test("exported metadata API preserves an explicit supported selection and rejects missing-value entries", () => {
  const h = buildFixture({ logLevel: "INFO", fields: { log_level: { options: logOptions } } });
  h.api.applyLuxMetadataToControls();
  assert.deepEqual(h.els.runtime.logLevel.options.map((option) => option.value), ["", "DEBUG", "INFO"]);
  assert.equal(h.els.runtime.logLevel.value, "INFO");
  assert.equal(h.state.config.runtime.logLevel, "INFO");
  assert.equal(h.generatePayload().args.log_level, "INFO");

  h.state.metadata.fields.log_level.options = logOptions.slice(0, 6);
  h.api.applyLuxMetadataToControls();
  assert.deepEqual(h.els.runtime.logLevel.options.map((option) => option.value), ["", "DEBUG", "INFO"]);
  assert.equal(h.state.config.runtime.logLevel, "INFO", "malformed-only metadata must not replace existing choices");
});

test("exported metadata API clears an unsupported choice when the backend selects empty Default", () => {
  const h = buildFixture({ logLevel: "REMOVED", fields: { log_level: { options: logOptions } } });
  h.api.applyLuxMetadataToControls();
  assert.equal(h.els.runtime.logLevel.value, "");
  assert.equal(h.state.config.runtime.logLevel, "", "state must match the selected empty option");
  assert.equal(Object.hasOwn(h.generatePayload().args, "log_level"), false);
});

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function previewResponse(payload) {
  return new Response(JSON.stringify({ success: true, data: {
    pipeline: payload.pipeline, normalized_args: payload.args,
    field_errors: [{ field: "input_dir", code: "tenant_path_outside_workspace", message: "Unauthorized Input Directory" }],
    readiness: { status: "blocked" },
  } }));
}

function reconciliationHarness({ changed, loading }) {
  const h = buildFixture({ modelKey: changed ? "da3-retired" : "da3-metric",
    fields: { model_key: { options: [{ value: "da3-metric", label: "DA3 Metric" }] } } });
  const bundle = deferred();
  const oldResponse = deferred();
  const calls = [];
  const previewWork = [];
  let loaded = null;
  let dispatchResets = 0;
  const context = vm.createContext({
    state: h.state, generatePayload: h.generatePayload,
    API_BASE: "", CONFIG_PREVIEW_TIMEOUT_MS: 1000, CONFIG_PREVIEW_DEBOUNCE_MS: 1,
    CONFIG_PREVIEW_SUPPORTED_PIPELINES: new Set(["lux-depth-v3"]), configPreviewTimerId: null,
    configPreviewRequests: createLatestRequestCoordinator(),
    AbortController, setTimeout, clearTimeout, window: { setTimeout },
    fetch: async (_url, options) => {
      const payload = JSON.parse(options.body);
      calls.push({ payload, signal: options.signal });
      if (loading && calls.length === 1) return oldResponse.promise;
      return previewResponse(payload);
    },
    _buildAuthHeaders: (headers) => headers,
    _configPreviewEnabledForPipeline: () => true,
    _isProtectedFamilySuppressed: () => false,
    _nonRetryableProtectedDetails: () => null,
    _previewFailureDetails: (value) => ({ reason: value.error_reason }),
    _clearConfigPreviewServiceRetry() {}, _scheduleConfigPreviewServiceRetry() {},
    _emptyPreviewState: (status, pipeline) => ({ status, pipeline }),
    _setPreviewState: (value) => { h.state.preview = value; },
    _reconcilePreviewRepairedPaths: (value) => value,
    _normalizeNextBestAction: (value) => value,
    _resetDispatchHandoff: () => { dispatchResets += 1; },
    _portalPrivilegesReady: () => true,
    currentPipelineDispatchStatus: () => "ready",
    isLuxPipeline: () => true,
    _effectiveDebugBundleEnabled: () => false,
    renderCLI() {}, renderPreRunDiagnostics() {}, _syncBootstrapGuardedControls() {}, emitPortalEvent() {},
    _deferredBuildSurfaceApi: () => loaded,
    _shouldLoadDeferredBuildSurface: () => h.state.currentView === "build",
    _primeDeferredProfileSurface() {},
    _loadDeferredBuildSurface: () => bundle.promise,
  });
  vm.runInContext([
    section("async function fetchWithTimeout(", "function _portalRumNow("),
    section("function _configPreviewRequestKey(", "function _effectivePreviewSnapshot("),
    section("function _reconcileDeferredBuildSurface(", "function _createDeferredProfileSurfaceHost("),
    section("function applyLuxMetadataToControls()", "function _previewIssueForField("),
    section("function _previewIssueForField(", "function _syncIssueAccessibility("),
    section("function fetchConfigPreview(", "function _formatWorkerSummary("),
    section("function _dispatchReadinessSnapshot(", "function updateConsoleViewContext("),
  ].join("\n"), context);
  const fetchConfigPreview = context.fetchConfigPreview;
  context.fetchConfigPreview = (payload) => {
    const work = fetchConfigPreview(payload);
    previewWork.push(work);
    return work;
  };
  return { ...h, context, calls, previewWork, dispatchResets: () => dispatchResets,
    resolveBundle() { loaded = h.api; bundle.resolve(h.api); },
    releaseOld() { oldResponse.resolve(previewResponse(calls[0].payload)); },
  };
}

for (const entry of ["prime", "metadata"]) {
  for (const loading of [false, true]) {
    test(`delayed ${entry} reconciliation refreshes a changed ${loading ? "loading" : "ready"} request after leaving Build`,
      { timeout: 2000 }, async () => {
        const h = reconciliationHarness({ changed: true, loading });
        const oldPayload = h.generatePayload();
        const previous = h.context.fetchConfigPreview(oldPayload);
        if (loading) await Promise.resolve();
        else await previous;
        assert.equal(h.state.preview.status, loading ? "loading" : "ready");
        if (entry === "prime") h.context._primeDeferredBuildSurface();
        else h.context.applyLuxMetadataToControls();
        h.state.currentView = "overview";
        h.resolveBundle();
        try {
          await Promise.resolve();
          await Promise.resolve();
          assert.equal(h.generatePayload().args.model_key, "da3-metric");
          assert.equal(h.dispatchResets(), 1, "changed metadata must use the existing preview scheduler");
          await h.previewWork.at(-1);
          assert.deepEqual(h.calls.map((call) => call.payload.args.model_key), ["da3-retired", "da3-metric"]);
          assert.notEqual(h.context._configPreviewRequestKey(oldPayload), h.context._configPreviewRequestKey(h.generatePayload()));
          assert.equal(h.context._currentPreviewForPayload().status, "ready");
          assert.equal(h.context._previewIssueForField("input_dir").tone, "error", "current server denial remains blocking");
          assert.equal(h.state.preview.readiness.status, "blocked");
          assert.equal(h.context._dispatchReadinessSnapshot().canRun, false);
          assert.equal(h.context._dispatchReadinessSnapshot().detail, "Unauthorized Input Directory");
          if (loading) assert.equal(h.calls[0].signal.aborted, true);
        } finally {
          h.releaseOld();
          await Promise.all(h.previewWork);
        }
        assert.equal(h.context._currentPreviewForPayload().normalized_args.model_key, "da3-metric",
          "a late response for the former request must not replace the reconciled preview");
      });
  }

  test(`delayed ${entry} reconciliation without a request change does not refresh ready or loading work`,
    { timeout: 2000 }, async () => {
      for (const loading of [false, true]) {
        const h = reconciliationHarness({ changed: false, loading });
        const previous = h.context.fetchConfigPreview(h.generatePayload());
        if (loading) await Promise.resolve();
        else await previous;
        const oldKey = h.state.preview.requestKey;
        if (entry === "prime") h.context._primeDeferredBuildSurface();
        else h.context.applyLuxMetadataToControls();
        h.resolveBundle();
        await Promise.resolve();
        await Promise.resolve();
        assert.equal(h.dispatchResets(), 0);
        assert.equal(h.calls.length, 1);
        assert.equal(h.state.preview.requestKey, oldKey);
        assert.equal(h.calls[0].signal.aborted, false);
        h.releaseOld();
        await previous;
        assert.equal(h.context._currentPreviewForPayload().status, "ready");
      }
    });
}
