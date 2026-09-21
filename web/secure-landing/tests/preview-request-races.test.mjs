import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { createLatestRequestCoordinator } from "../portal-src/internal/latest-request.js";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const transport = section("async function fetchWithTimeout(", "function _portalRumNow(");
const preview = section("function fetchConfigPreview(", "function scheduleConfigPreview(");
const recovery = section("function _resetProtectedFamilySuppression(", "function _queueBootstrapOnlineFollowup(");
const success = () => new Response(JSON.stringify({ success: true, data: { pipeline: "lux-depth-v5", normalized_args: {}, readiness: { status: "ready" } } }));

function harness(fetch) {
  const state = { pipeline: "lux-depth-v5", backendOk: true, preview: null };
  const suppression = new Map();
  let retries = 0;
  const context = vm.createContext({
    state, API_BASE: "", CONFIG_PREVIEW_TIMEOUT_MS: 20, HEALTH_CHECK_TIMEOUT_MS: 20,
    configPreviewRequests: createLatestRequestCoordinator(), _protectedFamilySuppression: suppression,
    fetch, AbortController, setTimeout, clearTimeout, currentApiKey: "old-key",
    generatePayload: () => ({ pipeline: "lux-depth-v5", args: { input_dir: "input", output_dir: "output" } }),
    _configPreviewRequestKey: JSON.stringify,
    _configPreviewEnabledForPipeline: () => true,
    _isProtectedFamilySuppressed: (family) => suppression.has(family),
    _nonRetryableProtectedDetails: (body) => body?.error?.details?.retryable === false ? body.error.details : null,
    _recordProtectedFamilySuppression: (family, details) => suppression.set(family, details),
    _clearConfigPreviewServiceRetry() {},
    _scheduleConfigPreviewServiceRetry() { retries += 1; },
    _emptyPreviewState: (status, pipeline) => ({ status, pipeline }),
    _setPreviewState: (value) => { state.preview = value; },
    _previewFailureDetails: ({ error_reason }) => ({ reason: error_reason }),
    _reconcilePreviewRepairedPaths: (value) => value,
    _normalizeNextBestAction: (value) => value,
    renderCLI() {}, renderPreRunDiagnostics() {}, _syncBootstrapGuardedControls() {}, emitPortalEvent() {},
    _syncApiKeyInputState() {}, resumeBlockedJobStreamsAfterAuthUpdate() {}, checkBackend() {},
    fetchConfigMetadata() {}, fetchPresetsForPipeline() {}, fetchReadiness() {},
    portalInternals: { parseRateLimitRetryHint: () => null },
  });
  vm.runInContext(`function _buildAuthHeaders(headers) { return { ...headers, 'X-API-Key': currentApiKey }; }\n${transport}\n${preview}\n${recovery}`, context);
  return { context, state, suppression, retries: () => retries };
}

test("new-key refresh sends a second request and late old-key 401 cannot suppress or overwrite ready preview", async () => {
  let releaseOld;
  const calls = [];
  const h = harness(async (_url, options) => {
    calls.push(options.headers["X-API-Key"]);
    if (calls.length === 1) return new Promise((resolve) => { releaseOld = resolve; });
    return success();
  });
  const previous = h.context.fetchConfigPreview(h.context.generatePayload());
  await Promise.resolve();
  h.context.currentApiKey = "valid-key";
  h.context._handleDirectDebugApiKeyUpdate({ resumeStreams: true });
  await h.context.fetchConfigPreview(h.context.generatePayload());
  releaseOld(new Response(JSON.stringify({ error: { details: { retryable: false, reason: "invalid_key" } } }), { status: 401 }));
  await previous;
  assert.deepEqual(calls, ["old-key", "valid-key"]);
  assert.equal(h.state.preview.status, "ready");
  assert.equal(h.suppression.size, 0);
  assert.equal(h.retries(), 0);
});

test("a stalled preview response body times out, releases same-key work, and permits a fresh retry", async () => {
  let calls = 0;
  let firstSignal;
  const h = harness(async (_url, options) => {
    calls += 1;
    if (calls > 1) return success();
    firstSignal = options.signal;
    const body = new ReadableStream({ start(controller) {
      options.signal.addEventListener("abort", () => controller.error(new Error("aborted body")), { once: true });
    } });
    return new Response(body);
  });
  await h.context.fetchConfigPreview(h.context.generatePayload());
  assert.equal(firstSignal.aborted, true);
  assert.equal(h.state.preview.status, "error");
  assert.equal(h.state.preview.error_reason, "service_failure");
  assert.equal(h.retries(), 1);
  await h.context.fetchConfigPreview(h.context.generatePayload());
  assert.equal(calls, 2);
  assert.equal(h.state.preview.status, "ready");
});
