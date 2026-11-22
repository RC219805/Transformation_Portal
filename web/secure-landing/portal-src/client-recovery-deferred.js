const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const DIGEST = /^[0-9a-f]{64}$/;
const STORAGE_KEY = "tp.portal.recovery_refs.v1";
const TERMINAL = new Set(["accepted", "ready", "expired", "rejected"]);
export function recoveryStateSummary(attempts, needsContext = false) {
  const unresolved = (type) => attempts.some((row) => row.type === type && !TERMINAL.has(row.status));
  return {
    outcome: needsContext ? "paused" : attempts.filter((row) => row.type === "job").at(-1)?.status || "idle",
    jobBlocked: needsContext || unresolved("job"),
    uploadBlocked: needsContext || unresolved("upload"),
    uploadPresent: attempts.some((row) => row.type === "upload"),
  };
}
function validReference(row) {
  return row && Object.keys(row).sort().join(",") === "created_at,request_id,scope,type" && ["job", "upload"].includes(row.type)
    && typeof row.request_id === "string" && UUID.test(row.request_id) && typeof row.scope === "string" && DIGEST.test(row.scope)
    && Number.isSafeInteger(row.created_at) && row.created_at >= 0;
}

export function createRecoveryReferenceStore(storage, uuid = () => crypto.randomUUID()) {
  let tabId = "", rows = [], available = true;
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    if (raw) {
      if (raw.length > 16384) throw new Error("Oversized recovery references");
      const parsed = JSON.parse(raw);
      if (parsed?.schema !== STORAGE_KEY || typeof parsed.tab_id !== "string" || !UUID.test(parsed.tab_id) || !Array.isArray(parsed.attempts) || parsed.attempts.length > 16) throw new Error("Invalid recovery references");
      tabId = parsed.tab_id;
      const unique = new Set();
      rows = parsed.attempts.map((row) => {
        if (!validReference(row) || unique.has(row.request_id)) throw new Error("Invalid recovery reference");
        unique.add(row.request_id); return { ...row };
      });
    } else { tabId = uuid(); if (typeof tabId !== "string" || !UUID.test(tabId)) throw new Error("Secure tab identity unavailable"); }
  } catch { available = false; }
  function write(next) {
    if (!available || !storage || next.length > 16 || next.some((row) => !validReference(row)) || new Set(next.map((row) => row.request_id)).size !== next.length) return false;
    const raw = JSON.stringify({ schema: STORAGE_KEY, tab_id: tabId, attempts: next });
    try {
      storage.setItem(STORAGE_KEY, raw);
      if (storage.getItem(STORAGE_KEY) !== raw) throw new Error("Recovery reference not retained");
      rows = next; return true;
    } catch { available = false; return false; }
  }
  return {
    available: () => available,
    entries: () => rows.map((row) => ({ ...row })),
    add(row) { return write([...rows, row]); },
    remove(id) { return write(rows.filter((row) => row.request_id !== id)); },
  };
}

export async function boundedRecoveryJson(fetchImpl, url, options = {}, timeoutMs = 30000, timers = globalThis) {
  const controller = new AbortController();
  let timer, reader;
  const deadline = new Promise((_, reject) => { timer = timers.setTimeout(() => {
    controller.abort(); const error = new Error("Request outcome is unknown after the response deadline"); error.name = "TimeoutError"; reject(error);
  }, timeoutMs); });
  const abort = () => controller.abort();
  options.signal?.addEventListener("abort", abort, { once: true });
  if (options.signal?.aborted) controller.abort();
  try {
    const response = await Promise.race([fetchImpl(url, { ...options, signal: controller.signal }), deadline]);
    let text = "", bytes = 0;
    const decoder = new TextDecoder("utf-8", { fatal: true });
    if (response.body) {
      reader = response.body.getReader();
      for (;;) {
        const { value, done } = await Promise.race([reader.read(), deadline]);
        if (done) break;
        bytes += value.byteLength;
        if (bytes > 262144) throw new Error("Recovery response exceeds its bound");
        text += decoder.decode(value, { stream: true });
      }
      text += decoder.decode();
    } else {
      text = await Promise.race([response.text(), deadline]);
      if (new TextEncoder().encode(text).byteLength > 262144) throw new Error("Recovery response exceeds its bound");
    }
    return { response, envelope: JSON.parse(text) };
  } finally {
    timers.clearTimeout(timer); options.signal?.removeEventListener("abort", abort);
    if (reader) { void reader.cancel().catch(() => {}); try { reader.releaseLock(); } catch { /* A deadline may leave a read pending. */ } }
    controller.abort();
  }
}

export function createDeferredClientRecoveryApi(host) {
  const fetchImpl = host.fetchImpl || ((...args) => fetch(...args));
  const uuid = host.uuid || (() => crypto.randomUUID());
  const timers = host.timers || globalThis;
  const now = host.now || (() => Date.now());
  let storage;
  try { storage = Object.hasOwn(host, "sessionStorage") ? host.sessionStorage : globalThis.sessionStorage; } catch { storage = null; }
  const store = createRecoveryReferenceStore(storage, uuid);
  const attempts = store.entries().map((reference) => ({ ...reference, status: "unknown", restored: true, durable: true, automatic: false }));
  const polls = new Map();
  let activeUpload = null, separateConfirmation = "", generation = 0, previousScope = "";
  const needsVerifiedContext = () => attempts.some((attempt) => attempt.durable) && !DIGEST.test(host.context().scope);
  const context = () => {
    const current = host.context();
    return needsVerifiedContext() ? { ...current, ready: false } : current;
  };
  const canRequest = (type) => host.canRequest?.(type) !== false;
  function inspectFailure(attempt, response, envelope) {
    if (response.status === 429) attempt.retryAt = now() + Math.max(0, host.retryDelay?.(response, now()) || 0);
    if (!response.ok) host.onProtectedFailure?.(attempt.type, envelope);
  }
  const sameOwner = (attempt) => context().ready && context().scope === attempt.scope;
  const currentIntent = (attempt) => sameOwner(attempt) && attempt.intent === context().intent;
  const unchangedDraft = (attempt) => currentIntent(attempt) && attempt.draftGeneration === context().draftGeneration;
  const headers = (method, extra = {}) => host.headers(extra, method);
  const endpoint = (attempt) => `${host.apiBase || ""}/v1/${attempt.type === "job" ? "job-submissions" : "uploads/staging/receipts"}/${attempt.request_id}`;

  function snapshot() {
    const current = context();
    return attempts.filter((attempt) => attempt.scope === current.scope).map(({ type, request_id, status, restored, canceled, result, message }) =>
      ({ type, request_id, status: current.ready ? status : "paused", restored: !!restored, canceled: !!canceled, result: current.ready ? result : null, message: current.ready ? message : "Recovery paused while access is checked." }));
  }
  function render() {
    const state = snapshot(), needsContext = needsVerifiedContext();
    host.onState?.(state, needsContext, recoveryStateSummary(state, needsContext));
    if (!host.document || !host.panels) return;
    const focused = host.document.activeElement;
    const focusId = focused?.dataset?.recoveryRequestId, focusAction = focused?.dataset?.ui;
    let focusTarget;
    for (const type of ["job", "upload"]) {
      const panel = host.panels[type]; if (!panel) continue;
      panel.replaceChildren(); panel.hidden = !state.some((attempt) => attempt.type === type);
      for (const item of state.filter((attempt) => attempt.type === type)) {
        const row = host.document.createElement("div"); row.dataset.ui = `${type}-recovery-attempt`; row.dataset.outcome = item.status;
        const status = host.document.createElement("p"); status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
        status.textContent = item.message || (item.status === "posting" ? type === "job" ? "Submitting this run…" : "Transferring files…"
          : item.status === "pending" ? "The server is still checking this attempt."
          : item.status === "accepted" ? "A run was accepted. Open the recovered run when ready."
          : item.status === "ready" ? "Staging completed. Apply the recovered input explicitly."
          : item.status === "expired" ? "This receipt expired. Check retained runs or files before starting new work."
          : "Outcome unknown. Check status before starting this work again.");
        row.append(status);
        const action = (label, ui, run, disabled = false) => {
          const button = host.document.createElement("button"); button.type = "button"; button.className = "operator-action-btn operator-action-btn-secondary";
          button.dataset.ui = ui; button.dataset.recoveryRequestId = item.request_id; button.textContent = label; button.disabled = disabled; button.addEventListener("click", run); row.append(button);
          if (focusId === item.request_id && focusAction === ui && !disabled) focusTarget = button;
        };
        if (!TERMINAL.has(item.status)) action("Check status", `${type}-recovery-check`, () => void check(item.request_id), !context().ready || polls.has(item.request_id));
        if (item.status === "accepted" || item.status === "ready") action(type === "job" ? "Open recovered run" : "Apply recovered input to this draft", `${type}-recovery-accept`, () => accept(item.request_id), !context().ready);
        if (item.status === "expired" || item.status === "rejected") action("Dismiss settled receipt", `${type}-recovery-dismiss`, () => forget(attempts.find((attempt) => attempt.request_id === item.request_id)), !context().ready);
        if (type === "upload" && activeUpload?.attempt.request_id === item.request_id) action("Cancel upload", "upload-recovery-cancel", cancelUpload);
        panel.append(row);
      }
      if (type === "job" && attempts.some((attempt) => attempt.type === "job" && attempt.scope === context().scope && !TERMINAL.has(attempt.status)
        && attempt.intent && attempt.intent !== context().intent)) {
        const button = host.document.createElement("button"); button.type = "button"; button.className = "operator-action-btn operator-action-btn-secondary";
        button.disabled = !context().ready;
        button.dataset.ui = "submission-separate-intent"; button.textContent = separateConfirmation === context().intent ? "Confirm a separate run; keep the earlier attempt" : "Start a separate run from the changed draft";
        button.addEventListener("click", () => {
          if (separateConfirmation !== context().intent) { separateConfirmation = context().intent; render(); return; }
          separateConfirmation = ""; host.submitSeparateIntent?.();
        }); panel.append(button);
      }
    }
    focusTarget?.focus({ preventScroll: true });
  }
  function change(attempt, patch) { Object.assign(attempt, patch); render(); }
  function forget(attempt) {
    if (attempt.durable && !store.remove(attempt.request_id)) { change(attempt, { message: "The completed recovery reference could not be cleared. Check status before new work." }); return false; }
    const index = attempts.indexOf(attempt); if (index >= 0) attempts.splice(index, 1); render(); return true;
  }
  function releaseRejectedReference(attempt, status) {
    if (attempt.durable && ![401, 403, 410].includes(status) && store.remove(attempt.request_id)) attempt.durable = false;
  }
  function newAttempt(type, durable) {
    const current = context();
    if (!canRequest(type)) throw new Error("Frontdoor configuration requires repair before another request.");
    if (!current.ready || !current.scope) throw new Error("Confirm access before starting work.");
    const requestId = uuid(); if (typeof requestId !== "string" || !UUID.test(requestId)) throw new Error("Secure request identity is unavailable.");
    const reference = { type, request_id: requestId, scope: current.scope, created_at: now() };
    if (durable && (!DIGEST.test(current.scope) || !store.add(reference))) throw new Error("Recovery storage is unavailable or full. No request was sent.");
    const attempt = { ...reference, intent: current.intent, draftGeneration: current.draftGeneration, durable, status: "posting", restored: false, generation };
    attempts.push(attempt); render(); return attempt;
  }
  function acceptedJob(value) {
    return value && typeof value.id === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(value.id)
      && ["queued", "running", "succeeded", "partial", "failed", "canceled", "worker_lost"].includes(value.state)
      && typeof value.events_url === "string" && value.events_url === `/v1/jobs/${value.id}/events`;
  }
  function jobResult(attempt, value) {
    return attempt.pipeline ? { ...value, pipeline: attempt.pipeline } : value;
  }
  function uploadReceipt(value) {
    return value && typeof value.batch_id === "string" && value.batch_id.length > 0 && value.batch_id.length <= 128
      && typeof value.input_dir === "string" && value.input_dir.trim() && value.input_dir.length <= 4096
      && value.summary && Number.isSafeInteger(value.summary.file_count) && value.summary.file_count > 0
      && Number.isSafeInteger(value.summary.total_bytes) && value.summary.total_bytes >= 0;
  }
  function accept(id, automatic = false) {
    const attempt = attempts.find((row) => row.request_id === id);
    if (!attempt || !sameOwner(attempt)) return false;
    try {
      if (attempt.status === "accepted" && acceptedJob(attempt.result)) {
        if (automatic && !unchangedDraft(attempt)) return false;
        if (host.acceptJob(attempt.result, { automatic }) !== true) return false;
      }
      else if (attempt.status === "ready" && uploadReceipt(attempt.result)) host.applyUpload(attempt.result);
      else return false;
    } catch { change(attempt, { message: "The recovered result could not be applied. Check the current draft and try the explicit recovery action again." }); return false; }
    return forget(attempt);
  }
  async function readReceipt(attempt, timeoutMs = 5000) {
    if (!sameOwner(attempt) || !attempt.durable || !canRequest(attempt.type) || now() < (attempt.retryAt || 0)) return false;
    const expectedGeneration = generation;
    const { response, envelope } = await boundedRecoveryJson(fetchImpl, endpoint(attempt), { method: "GET", credentials: "same-origin", headers: headers("GET", { Accept: "application/json" }) }, timeoutMs, timers);
    if (generation !== expectedGeneration || !sameOwner(attempt)) return false;
    inspectFailure(attempt, response, envelope);
    if (response.status === 401 || response.status === 403) { host.onAuthorizationFailure?.(response.status); return false; }
    if (!response.ok || !envelope?.success || envelope.data?.request_id !== attempt.request_id) return false;
    const receipt = envelope.data;
    if (attempt.type === "job" && envelope.schema === "tp.orchestrator.job_submission.v1") {
      if (receipt.status === "accepted" && acceptedJob(receipt.job)) {
        change(attempt, { status: "accepted", result: jobResult(attempt, receipt.job), message: "" });
        if (!attempt.restored) accept(attempt.request_id, true);
        return true;
      }
      if (receipt.status === "expired") { change(attempt, { status: "expired", message: "Submission receipt expired. Inspect retained runs before starting another run." }); return true; }
    }
    if (attempt.type === "upload" && envelope.schema === "tp.orchestrator.upload_receipt.v1") {
      if (receipt.status === "ready" && uploadReceipt(receipt.receipt)) {
        change(attempt, { status: "ready", result: receipt.receipt, message: "Staging completed. Apply the recovered input explicitly." }); return true;
      }
      if (receipt.status === "expired") { change(attempt, { status: "expired", message: "Upload receipt expired. Previously selected paths were preserved." }); return true; }
      if (receipt.status === "pending") change(attempt, { status: "pending", message: "The server is still staging this attempt." });
    }
    return false;
  }
  async function check(id, automatic = false) {
    const attempt = attempts.find((row) => row.request_id === id);
    if (!attempt || polls.has(id) || !sameOwner(attempt)) return false;
    if (!attempt.durable) { change(attempt, { message: "Receipt recovery is unavailable for this attempt. Check existing runs or staged files before starting new work." }); return false; }
    const expected = generation, expiresAt = now() + (automatic ? 60000 : 5000);
    const operation = (async () => {
      do {
        try { if (await readReceipt(attempt, Math.max(1, Math.min(5000, expiresAt - now())))) return true; } catch { /* A failed read never authorizes another POST. */ }
        const delay = Math.max(2000, (attempt.retryAt || 0) - now());
        if (!automatic || !sameOwner(attempt) || !canRequest(attempt.type) || expected !== generation || now() + delay >= expiresAt) break;
        await new Promise((resolve) => { timers.setTimeout(resolve, delay); });
      } while (now() < expiresAt);
      if (sameOwner(attempt) && attempts.includes(attempt) && !TERMINAL.has(attempt.status)) change(attempt, { status: "unknown", message: "Outcome still unknown. Use Check status; this request will not be repeated." });
      return false;
    })();
    polls.set(id, operation); render();
    try { return await operation; } finally { polls.delete(id); render(); }
  }
  async function submit(payload, { separate = false } = {}) {
    const current = context();
    const unresolved = attempts.filter((attempt) => attempt.type === "job" && attempt.scope === current.scope && !TERMINAL.has(attempt.status));
    if (unresolved.length && (!separate || unresolved.some((attempt) => !attempt.intent || attempt.intent === current.intent))) {
      render(); throw new Error("Submission outcome is unresolved. Check status before submitting this draft again.");
    }
    const attempt = newAttempt("job", current.features.jobSubmissionReceipts === true);
    attempt.pipeline = typeof payload?.pipeline === "string" ? payload.pipeline : null;
    const startedAt = now(); let metricRecorded = false;
    function metric(response) {
      metricRecorded = true;
      host.onSubmissionMetric?.({ durationMs: Math.max(0, now() - startedAt), outcome: response?.ok ? "ok" : "error", ...(response ? { status: response.status } : {}) });
    }
    try {
      const { response, envelope } = await boundedRecoveryJson(fetchImpl, `${host.apiBase || ""}/v1/jobs`, {
        method: "POST", credentials: "same-origin", headers: headers("POST", { "Content-Type": "application/json", ...(attempt.durable ? { "Idempotency-Key": attempt.request_id } : {}) }),
        body: JSON.stringify(payload)
      }, 30000, timers);
      metric(response);
      if (generation !== attempt.generation || !sameOwner(attempt)) { change(attempt, { status: "unknown", message: "Submission belongs to an earlier access context. Recheck it after access is restored." }); return null; }
      inspectFailure(attempt, response, envelope);
      if (response.ok && envelope?.success && envelope.schema === "tp.orchestrator.job.v1" && acceptedJob(envelope.data)) {
        const result = jobResult(attempt, envelope.data);
        change(attempt, { status: "accepted", result, message: "" });
        accept(attempt.request_id, true);
        return result;
      }
      if (response.status >= 400 && response.status < 500 && ![408, 409, 410, 429].includes(response.status)) {
        releaseRejectedReference(attempt, response.status);
        change(attempt, { status: "rejected", message: String(envelope?.error?.message || "Submission was rejected.").slice(0, 240) });
        if (response.status === 401 || response.status === 403) host.onAuthorizationFailure?.(response.status);
        return null;
      }
      throw new Error("Submission response did not establish an accepted run");
    } catch {
      if (!metricRecorded) metric();
      change(attempt, { status: "unknown", message: "Submission outcome unknown — checking for your run. The request will not be repeated." });
      if (sameOwner(attempt)) void check(attempt.request_id, true);
      return null;
    }
  }

  function upload(selection) {
    const current = context();
    if (activeUpload || attempts.some((attempt) => attempt.type === "upload" && attempt.scope === current.scope && !TERMINAL.has(attempt.status))) throw new Error("An upload is unresolved. Check its status before selecting another batch.");
    const attempt = newAttempt("upload", current.features.uploadReceipts === true);
    const xhr = host.xhrFactory ? host.xhrFactory() : new XMLHttpRequest();
    const form = host.formDataFactory ? host.formDataFactory() : new FormData();
    selection.entries.forEach(({ file, relativePath }) => form.append("files", file, relativePath));
    form.append("client_manifest", JSON.stringify({ schema: "tp.portal.upload_manifest.v1", files: selection.entries.map((entry) => ({ relative_path: entry.relativePath, size_bytes: entry.sizeBytes, name: entry.name })) }));
    let idleTimer, responseTimer, finished = false, transferred = false, loadedBytes = 0;
    function clear() { timers.clearTimeout(idleTimer); timers.clearTimeout(responseTimer); }
    function unknown(message) {
      if (finished) return;
      finished = true; clear(); activeUpload = null;
      if (sameOwner(attempt)) host.uploadState({ busy: false, status: "outcome_unknown", error: message, receipt: null });
      change(attempt, { status: "unknown", message });
      if (sameOwner(attempt)) void check(attempt.request_id, true);
    }
    function activity() {
      timers.clearTimeout(idleTimer);
      if (!transferred) idleTimer = timers.setTimeout(() => { unknown("Upload stalled. Its server outcome is unknown; checking the receipt."); xhr.abort(); }, 90000);
    }
    xhr.open("POST", `${host.apiBase || ""}/v1/uploads/staging`); xhr.responseType = "json";
    for (const [key, value] of Object.entries(headers("POST", attempt.durable ? { "X-TP-Upload-Request-ID": attempt.request_id } : {}))) if (typeof value === "string" && value) xhr.setRequestHeader(key, value);
    xhr.upload.onprogress = (event) => {
      if (finished || !sameOwner(attempt)) return;
      if (Number.isFinite(event.loaded) && event.loaded > loadedBytes) { loadedBytes = event.loaded; activity(); }
      const progressPercent = event.lengthComputable && event.total > 0 ? Math.max(0, Math.min(100, Math.floor(event.loaded / event.total * 100))) : 0;
      host.uploadState({ progressPercent, status: "transferring", error: "" });
    };
    xhr.upload.onload = () => {
      if (finished || !sameOwner(attempt)) return;
      transferred = true; timers.clearTimeout(idleTimer);
      host.uploadState({ progressPercent: 100, status: "staging", error: "" });
      responseTimer = timers.setTimeout(() => { unknown("Files transferred, but staging has not replied. Checking the receipt."); xhr.abort(); }, 120000);
    };
    xhr.onerror = () => unknown("Upload outcome unknown after a network failure. Checking the receipt.");
    xhr.onabort = () => unknown("Upload transport stopped. Server staging may still finish; checking its receipt.");
    xhr.onload = () => {
      if (finished) return;
      if (!sameOwner(attempt) || generation !== attempt.generation) { unknown("Upload belongs to an earlier access context. Check its receipt after access is restored."); return; }
      const envelope = xhr.response;
      inspectFailure(attempt, { status: xhr.status, ok: xhr.status >= 200 && xhr.status < 300,
        headers: { get: (key) => xhr.getResponseHeader?.(key) || null } }, envelope);
      if (xhr.status >= 200 && xhr.status < 300 && envelope?.success && envelope.schema === "tp.orchestrator.upload_staging.v1" && uploadReceipt(envelope.data)) {
        finished = true; clear(); activeUpload = null;
        change(attempt, { status: "ready", result: envelope.data, message: "" });
        if (!attempt.canceled && unchangedDraft(attempt)) accept(attempt.request_id);
        else host.uploadState({ busy: false, status: "recovery_ready", error: "Apply the recovered input explicitly.", receipt: null });
      } else if (xhr.status >= 400 && xhr.status < 500 && ![408, 409, 429].includes(xhr.status)) {
        finished = true; clear(); activeUpload = null;
        releaseRejectedReference(attempt, xhr.status);
        host.uploadState({ busy: false, status: "rejected", error: "Staging rejected this upload. Existing paths were preserved.", receipt: null });
        change(attempt, { status: "rejected", message: "Staging rejected this upload. Existing paths were preserved." });
        if (xhr.status === 401 || xhr.status === 403) host.onAuthorizationFailure?.(xhr.status);
      } else unknown("Upload returned no verified receipt. Its outcome is unknown; checking status.");
    };
    activeUpload = { xhr, attempt, unknown };
    host.uploadState({ busy: true, progressPercent: 0, status: "transferring", error: "", receipt: null, fileCount: selection.entries.length, totalBytes: selection.totalBytes });
    activity(); render();
    try { xhr.send(form); } catch { unknown("Upload outcome unknown because its transport failed."); }
    return attempt.request_id;
  }
  function cancelUpload() {
    if (!activeUpload) return;
    const active = activeUpload; active.attempt.canceled = true;
    active.unknown("Upload transport canceled. Staging may still finish; apply recovered input only after checking."); active.xhr.abort();
  }
  function sync() {
    const current = context();
    if (!current.ready || (previousScope && previousScope !== current.scope)) {
      generation += 1;
      if (activeUpload) { const active = activeUpload; active.attempt.canceled = true; active.unknown("Upload paused after an access change. Check its receipt after access is restored."); active.xhr.abort(); }
      if (!current.ready) for (const attempt of attempts) if (attempt.result) {
        delete attempt.result; attempt.status = "unknown"; attempt.restored = true;
      }
      if (current.ready && previousScope && previousScope !== current.scope) {
        for (const attempt of attempts) if (attempt.scope !== current.scope) { delete attempt.intent; delete attempt.pipeline; delete attempt.result; attempt.restored = true; if (["posting", "accepted", "ready"].includes(attempt.status)) attempt.status = "unknown"; }
      }
    }
    if (current.ready) previousScope = current.scope;
    render();
  }
  sync();
  return { submit, upload, check, accept, cancelUpload, sync, snapshot };
}
