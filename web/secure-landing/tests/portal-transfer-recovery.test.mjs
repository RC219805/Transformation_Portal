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
    _loadClientRecoverySurface: async () => ({ upload: (selection) => requests.push(selection) }),
    _clientRecoveryOperationStamp: () => "current-upload-operation",
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

test("upload selection delegates once to receipt recovery while retaining current paths", async () => {
  const h = uploadHarness();
  h.context._submitStagedUploadSelection([]);
  await Promise.resolve();
  assert.equal(h.state.upload.busy, true);
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].entries[0].relativePath, "image.jpg");
  assert.equal(h.state.config.inputDir, "/original");
  assert.equal(h.state.config.outputDir, "/output");
});

test("a verified explicit upload apply changes only the input directory", () => {
  const h = uploadHarness();
  h.context._applyStagedUploadResult({ input_dir: "/staged/input", batch_id: "batch-one", summary: { file_count: 1, total_bytes: 4 } });
  assert.equal(h.state.upload.busy, false);
  assert.equal(h.state.upload.status, "ready");
  assert.equal(h.state.config.inputDir, "/staged/input");
  assert.equal(h.state.config.outputDir, "/output");
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
