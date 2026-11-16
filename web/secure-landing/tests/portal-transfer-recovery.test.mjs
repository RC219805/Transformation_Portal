import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { createDeferredConfigTransferApi } from "../portal-src/config-transfer-deferred.js";
import { createPortalConfigState } from "../portal-src/internal/state.js";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
function section(start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  assert.ok(first >= 0 && last > first);
  return source.slice(first, last);
}

function uploadHarness() {
  const state = { config: { inputDir: "/original", outputDir: "/output" }, upload: {} };
  const requests = [];
  const toasts = [];
  const context = vm.createContext({
    state, els: {}, API_BASE: "",
    FormData: class { append() {} },
    XMLHttpRequest: class {
      constructor() { this.upload = {}; requests.push(this); }
      open() {}
      setRequestHeader() {}
      send() {}
    },
    _blockManagedUnavailableAction: () => false,
    _stagedUploadsEnabledForState: () => true,
    _isProtectedFamilySuppressed: () => false,
    _collectStagedUploadSelection: () => ({ entries: [{ file: {}, relativePath: "image.jpg", sizeBytes: 4, name: "image.jpg" }], totalBytes: 4 }),
    _setStagedUploadState: (patch) => Object.assign(state.upload, patch),
    _syncStagedUploadUi() {},
    _portalRumTraceparent() {},
    portalInternals: { createChildTraceparent() {} },
    _buildAuthHeaders: () => ({}),
    _nonRetryableProtectedDetails: () => null,
    _stagedUploadErrorMessage: () => "Failed to stage uploads.",
    createToast: (...args) => toasts.push(args),
    formatBytes: (bytes) => `${bytes} bytes`,
    renderCLI() {}, scheduleConfigPreview() {}, renderFieldPreviewStatuses() {},
  });
  vm.runInContext(section("function _applyStagedUploadResult(", "function resumeBlockedJobStreamsAfterAuthUpdate("), context);
  return { context, state, requests, toasts };
}

test("malformed staged-upload successes release busy controls and preserve draft paths for a retry", () => {
  for (const data of [{}, { input_dir: "   " }, { input_dir: {} }]) {
    const h = uploadHarness();
    h.context._submitStagedUploadSelection([]);
    assert.equal(h.state.upload.busy, true);
    Object.assign(h.requests[0], { status: 200, response: { success: true, data } });
    assert.doesNotThrow(() => h.requests[0].onload());
    assert.equal(h.state.upload.busy, false);
    assert.equal(h.state.upload.status, "idle");
    assert.match(h.state.upload.error, /invalid response/i);
    assert.equal(h.state.config.inputDir, "/original");
    assert.equal(h.state.config.outputDir, "/output");

    h.context._submitStagedUploadSelection([]);
    Object.assign(h.requests[1], { status: 200, response: { success: true, data: {
      input_dir: "/staged/input", batch_id: "batch-retry", summary: { file_count: 1, total_bytes: 4 },
    } } });
    h.requests[1].onload();
    assert.equal(h.state.upload.busy, false);
    assert.equal(h.state.upload.status, "ready");
    assert.equal(h.state.upload.error, "");
    assert.equal(h.state.config.inputDir, "/staged/input");
    assert.equal(h.state.config.outputDir, "/output");
  }
});

function clipboardHarness(copy) {
  const toasts = [];
  const children = new Set();
  const context = vm.createContext({
    navigator: {}, window: { isSecureContext: false },
    document: {
      createElement: () => ({ style: {}, select() {} }),
      body: { appendChild: (node) => children.add(node), removeChild: (node) => children.delete(node) },
      execCommand: copy,
    },
    createToast: (...args) => toasts.push(args),
  });
  vm.runInContext(section("async function copyToClipboard(", "function appendJobLog("), context);
  return { context, toasts, children };
}

test("clipboard fallback reports a refused copy and always removes its temporary text", async () => {
  for (const copy of [() => false, () => { throw new Error("copy denied"); }]) {
    const h = clipboardHarness(copy);
    await h.context.copyToClipboard("operator command");
    assert.deepEqual(h.toasts, [["Failed to copy text.", "error"]]);
    assert.equal(h.children.size, 0);
  }
});

test("clipboard fallback reports success only after the browser confirms the copy", async () => {
  const h = clipboardHarness((command) => { assert.equal(command, "copy"); return true; });
  await h.context.copyToClipboard("operator command");
  assert.deepEqual(h.toasts, [["Copied to clipboard.", "success"]]);
  assert.equal(h.children.size, 0);
});

function importHostHarness() {
  const state = { pipeline: 'lux-depth-v3', config: createPortalConfigState(), portalUi: { activeProfileName: 'Saved profile', buildStep: 1 } };
  const timers = new Map(), idle = new Map(), saves = [], notices = [];
  let nextTimer = 0;
  const context = vm.createContext({
    state, els: { profileSelect: { value: 'Saved profile' } }, configImportGeneration: 0, pendingLegacyTransientDraft: null,
    transientDraftPersistTimerId: null, transientDraftPersistIdleId: null,
    TRANSIENT_DRAFT_SCHEMA: 'draft', TRANSIENT_DRAFT_STORAGE_KEY: 'draft', TRANSIENT_DRAFT_PERSIST_DEBOUNCE_MS: 10,
    sessionStorage: { setItem: (...args) => saves.push(args) },
    window: { setTimeout: cb => { timers.set(++nextTimer, cb); return nextTimer; }, clearTimeout: id => timers.delete(id),
      requestIdleCallback: cb => { idle.set(++nextTimer, cb); return nextTimer; }, cancelIdleCallback: id => idle.delete(id), confirm: () => true },
    portalInternals: { createPortalConfigState }, CONFIG_PREVIEW_SUPPORTED_PIPELINES: new Set(['lux-depth-v3','archive-gate-a']),
    _transientDraftOwnerKey: () => 'managed-owner', _copyTransientDraftConfig: (value = state.config) => structuredClone(value),
    resolveBuildStep: value => value, _migrateDeprecatedLuxOutputConfig: value => value,
    _loadDeferredProfileSurface: async () => ({ syncDraftState: () => ({ hasProtectedChanges: false }) }),
    updateUIFromState() {}, fetchPresetsForPipeline() {}, fetchConfigMetadata() {}, createToast: (...args) => notices.push(args),
    _normalizeVerboseQuietFlags() {}, parseBoolLike: (value, fallback) => typeof value === 'boolean' ? value : fallback,
    _parseNonNegativeIntOrNull: () => null, _parsePositiveIntOrNull: () => null, _parseProbabilityOrNull: () => null,
    _textOrFallback: (value, fallback) => value || fallback,
    ...Object.fromEntries(['Da3ModelKey','DepthBackend','GroupingMode','QualityTier','RunCardVersion','Sam2ModelSize','SegmentationBackend','VlmCaptioningProxyFormat'].map(name => [`_resolve${name}`, value => value])),
  });
  vm.runInContext(section('function _persistTransientPortalDraft(', 'function _applyTransientPortalDraft('), context);
  vm.runInContext(section('function _configImportSnapshot(', 'function _createDeferredBuildSurfaceHost('), context);
  const host = context._createConfigTransferHost();
  return { context, state, saves, notices, api: createDeferredConfigTransferApi(host),
    flush: () => { for (const cb of [...timers.values()]) cb(); timers.clear(); for (const cb of [...idle.values()]) cb(); idle.clear(); } };
}

for (const phase of ['module loading', 'file loading']) {
  test(`pending autosave during ${phase} does not invalidate an otherwise current import`, async () => {
    const h = importHostHarness(); h.context._scheduleTransientPortalDraftPersist();
    const expected = h.context._configImportSnapshot();
    let finish;
    if (phase === 'module loading') h.flush();
    const result = h.api.importFile({ size: 100, text: () => new Promise(resolve => { finish = resolve; }) }, expected);
    if (phase === 'file loading') h.flush();
    assert.equal(h.context._configImportSnapshot(), expected);
    finish(JSON.stringify({ args: { input_dir: '/imported/input' } }));
    assert.equal(await result, true);
    assert.equal(h.state.config.inputDir, '/imported/input');
    assert.equal(h.state.portalUi.activeProfileName, '');
    assert.equal(h.context.els.profileSelect.value, '', 'Import must clear the selected profile as well as its state');
  });
}

test('a real edit and undo still supersede a pending import', async () => {
  const h = importHostHarness(), original = h.state.config.inputDir;
  let finish;
  const result = h.api.importFile({ size: 100, text: () => new Promise(resolve => { finish = resolve; }) });
  h.state.config.inputDir = '/edited'; h.context._scheduleTransientPortalDraftPersist();
  h.state.config.inputDir = original; h.context._scheduleTransientPortalDraftPersist(); h.flush();
  finish(JSON.stringify({ args: { input_dir: '/stale/import' } }));
  assert.equal(await result, false);
  assert.equal(h.state.config.inputDir, original); assert.equal(h.state.portalUi.activeProfileName, 'Saved profile');
  assert.equal(h.context.els.profileSelect.value, 'Saved profile');
});

function eventControl(value = '') {
  const listeners = new Map();
  return { value, files: [], addEventListener: (type, listener) => listeners.set(type, listener),
    dispatch(type) { return listeners.get(type)?.({ target: this }); } };
}

for (const phase of ['module loading', 'file loading']) {
  for (const undo of [false, true]) {
    test(`unblurred photographic text ${undo ? 'edit and undo' : 'input'} supersedes import during ${phase}`, async () => {
      const h = importHostHarness(), before = structuredClone(h.state);
      const control = eventControl(h.state.config.photographyV6.companionsManifest);
      const fileInput = eventControl('selected.json');
      h.context.importTextControl = control;
      h.context.els.fileInput = fileInput;
      vm.runInContext(section('    const safeBindText = (', '    const safeBindInput = (')
        + '\nsafeBindText(importTextControl, () => state.config.photographyV6, "companionsManifest");', h.context);
      let releaseModule, releaseFile;
      const json = JSON.stringify({ args: { input_dir: '/stale/import' } });
      fileInput.files = [{ size: json.length, text: () => phase === 'file loading'
        ? new Promise(resolve => { releaseFile = resolve; }) : Promise.resolve(json) }];
      h.context.loadDeferredSurface = () => phase === 'module loading'
        ? new Promise(resolve => { releaseModule = resolve; }) : Promise.resolve(h.api);
      vm.runInContext(section("if (els.fileInput) els.fileInput.addEventListener('change'", 'if (els.copyCliBtn)'), h.context);
      const importing = fileInput.dispatch('change');
      await new Promise(resolve => setImmediate(resolve));
      control.value = '/unblurred/manifest.json'; control.dispatch('input');
      if (undo) { control.value = before.config.photographyV6.companionsManifest; control.dispatch('input'); }
      assert.deepEqual(h.state, before, 'input must retain existing change-only state semantics');
      assert.equal(h.saves.length, 0, 'input must not persist or normalize the uncommitted control');
      if (phase === 'module loading') releaseModule(h.api); else releaseFile(json);
      await importing;
      assert.deepEqual(h.state, before, 'the stale candidate must not replace the current draft');
      assert.equal(control.value, undo ? before.config.photographyV6.companionsManifest : '/unblurred/manifest.json');
      assert.equal(h.saves.length, 0); assert.deepEqual(h.notices, []);
      assert.equal(h.context.els.profileSelect.value, 'Saved profile');
      assert.equal(fileInput.value, '');
    });
  }
}
