import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { createLatestRequestCoordinator } from "../portal-src/internal/latest-request.js";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const transport = section("async function fetchWithTimeout(", "function _portalRumNow(");
const preview = section("function fetchConfigPreview(", "function scheduleConfigPreview(");
const failureDetails = section("function _previewFailureDetails(", "function _nextBestActionLabel(");
const recovery = section("function _resetProtectedFamilySuppression(", "function _queueBootstrapOnlineFollowup(");
const success = () => new Response(JSON.stringify({ success: true, data: { pipeline: "lux-depth-v5", normalized_args: {}, readiness: { status: "ready" } } }));

function harness(fetch) {
  const state = { pipeline: "lux-depth-v5", backendOk: true, preview: null };
  const suppression = new Map();
  const events = [];
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
    _reconcilePreviewRepairedPaths: (value) => value,
    _normalizeNextBestAction: (value) => value,
    titleCaseToken: (value) => String(value).replaceAll('_', ' '),
    renderCLI() {}, renderPreRunDiagnostics() {}, _syncBootstrapGuardedControls() {},
    emitPortalEvent: (name, details) => events.push(JSON.parse(JSON.stringify({ name, ...details }))),
    _syncApiKeyInputState() {}, resumeBlockedJobStreamsAfterAuthUpdate() {}, checkBackend() {},
    fetchConfigMetadata() {}, fetchPresetsForPipeline() {}, fetchReadiness() {},
    portalInternals: { parseRateLimitRetryHint: () => null },
  });
  vm.runInContext(`function _buildAuthHeaders(headers) { return { ...headers, 'X-API-Key': currentApiKey }; }\n${transport}\n${failureDetails}\n${preview}\n${recovery}`, context);
  return { context, state, suppression, events, retries: () => retries };
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

test("tenant path denial is editable validation and does not suppress later previews", async () => {
  let allowed = false;
  const h = harness(async () => allowed ? success() : new Response(JSON.stringify({
    error: { code: 'FORBIDDEN', message: 'tenant admission failed', details: {
      field: 'input_dir', reason: 'tenant_path_outside_workspace'
    } }
  }), { status: 403 }));
  await h.context.fetchConfigPreview(h.context.generatePayload());
  assert.equal(h.state.preview.error_reason, 'validation_error');
  assert.equal(h.state.preview.field_errors[0].field, 'input_dir');
  assert.match(h.state.preview.field_errors[0].message, /path authorized for this workspace/);
  assert.equal(h.suppression.size, 0);
  assert.equal(h.retries(), 0);
  allowed = true;
  await h.context.fetchConfigPreview(h.context.generatePayload());
  assert.equal(h.state.preview.status, 'ready');
});

test("unknown forbidden and frontdoor auth configuration failures remain auth failures", async () => {
  for (const [status, code] of [[403, 'FORBIDDEN'], [503, 'AUTH_CONFIGURATION_ERROR']]) {
    const h = harness(async () => new Response(JSON.stringify({ error: { code } }), { status }));
    await h.context.fetchConfigPreview(h.context.generatePayload());
    assert.equal(h.state.preview.error_reason, 'auth_failure');
    assert.equal(h.state.preview.field_errors.length, 0);
    assert.equal(h.retries(), 0);
  }
});

test("only an exact unsupported-pipeline response requires a backend update without reflecting server text", async () => {
  for (const [status, code, reason, expected, retryCount] of [
    [400, 'INVALID_ARGUMENT', 'unsupported_pipeline', 'backend_update_required', 0],
    [400, 'INVALID_ARGUMENT', 'invalid_request', 'validation_error', 0],
    [400, 'OTHER_ERROR', 'unsupported_pipeline', 'validation_error', 0],
    [422, 'INVALID_ARGUMENT', 'unsupported_pipeline', 'validation_error', 0],
    [401, 'INVALID_ARGUMENT', 'unsupported_pipeline', 'auth_failure', 0],
    [403, 'INVALID_ARGUMENT', 'unsupported_pipeline', 'auth_failure', 0],
    [429, 'INVALID_ARGUMENT', 'unsupported_pipeline', 'service_failure', 1],
    [503, 'AUTH_CONFIGURATION_ERROR', 'unsupported_pipeline', 'auth_failure', 0],
    [500, 'INVALID_ARGUMENT', 'unsupported_pipeline', 'service_failure', 1],
  ]) {
    const h = harness(async () => new Response(JSON.stringify({ error: {
      code, message: 'UNTRUSTED secret traceback <script>bad()</script> <SCRIPT>bad()</SCRIPT> <ScRiPt data-origin="backend">bad()</ScRiPt>',
      details: { field: 'payload', reason, traceback: 'UNTRUSTED secret path' }
    } }), { status }));
    const draft = h.context.generatePayload();
    await h.context.fetchConfigPreview(draft);
    assert.equal(h.state.preview.status, 'error');
    assert.equal(h.state.preview.error_reason, expected, `${status}/${code}/${reason}`);
    assert.equal(h.state.preview.field_errors.length, 0);
    assert.equal(h.suppression.size, 0);
    assert.equal(h.retries(), retryCount);
    assert.deepEqual(h.context.generatePayload(), draft);
    const details = h.context._previewFailureDetails(h.state.preview);
    assert.doesNotMatch(JSON.stringify([h.state.preview, details]), /UNTRUSTED|<script\b/i);
    if (expected === 'backend_update_required') {
      assert.equal(details.summaryLabel, 'Backend update required');
      assert.match(details.luxBlockedMessage, /Update and restart the backend API and workers/);
      assert.match(details.toastMessage, /refresh the preview\. Your draft is preserved\./);
      assert.deepEqual(h.events, [{ name: 'preview_error_seen', surface: 'reconstruction_runtime',
        reasons: ['unsupported_pipeline'], metadata: { status: 400 } }]);
    }
  }
});

test("backend update rejection permits a fresh preview and a late rejection cannot replace success", async () => {
  const rejection = () => new Response(JSON.stringify({ error: {
    code: 'INVALID_ARGUMENT', details: { field: 'payload', reason: 'unsupported_pipeline' }
  } }), { status: 400 });
  let restored = false;
  const h = harness(async () => restored ? success() : rejection());
  await h.context.fetchConfigPreview(h.context.generatePayload());
  assert.equal(h.state.preview.error_reason, 'backend_update_required');
  restored = true;
  await h.context.fetchConfigPreview(h.context.generatePayload());
  assert.equal(h.state.preview.status, 'ready');
  assert.equal(h.retries(), 0);
  assert.equal(h.events.length, 1);

  let releaseOld;
  let calls = 0;
  const racing = harness(async () => ++calls === 1
    ? new Promise((resolve) => { releaseOld = resolve; }) : success());
  const oldRequest = racing.context.fetchConfigPreview(racing.context.generatePayload());
  await Promise.resolve();
  racing.context.configPreviewRequests.invalidate();
  await racing.context.fetchConfigPreview(racing.context.generatePayload());
  releaseOld(rejection());
  await oldRequest;
  assert.equal(racing.state.preview.status, 'ready');
  assert.equal(racing.suppression.size, 0);
  assert.equal(racing.retries(), 0);
  assert.deepEqual(racing.events, []);
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
