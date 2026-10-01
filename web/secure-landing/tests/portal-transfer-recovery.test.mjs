import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

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
