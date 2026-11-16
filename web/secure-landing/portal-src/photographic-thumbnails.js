const SHA256 = /^[a-f0-9]{64}$/;
const MAX_BYTES = 1048576;
const MAX_ENTRIES = 16;
const MAX_CACHE_BYTES = 16 * MAX_BYTES;
const RENDERER = "tp.photo.thumbnail.v1";
const TRANSFORMS = { clip_srgb: "linear_srgb_to_8bit_srgb_clip_v1", recorded: "recorded_presentation_to_8bit_srgb_v1" };
const digest = (value) => typeof value === "string" && SHA256.test(value);

export function photographicThumbnailEvidence(response, bytes, expected) {
  const headers = response.headers;
  const decoded = (name) => { try { return decodeURIComponent(headers.get(name) || ""); } catch { return ""; } };
  const width = Number(headers.get("X-TP-Thumbnail-Width")), height = Number(headers.get("X-TP-Thumbnail-Height"));
  const nativeWidth = Number(headers.get("X-TP-Native-Width")), nativeHeight = Number(headers.get("X-TP-Native-Height"));
  const sha256 = headers.get("X-TP-Source-SHA256"), source = headers.get("X-TP-Source-Identity-SHA256"), policy = headers.get("X-TP-Presentation-Policy-SHA256");
  if (!response.ok || !/^image\/png(?:\s*;|$)/i.test(headers.get("Content-Type") || "")
    || headers.get("X-TP-Catalogue-Identity-SHA256") !== expected.catalogue
    || decoded("X-TP-Scene-ID") !== expected.sceneId || decoded("X-TP-Artifact-Path") !== expected.artifactPath
    || headers.get("X-TP-Presentation-View") !== expected.view || headers.get("X-TP-Display-Transform") !== TRANSFORMS[expected.view]
    || headers.get("X-TP-Thumbnail-Max-Edge") !== "256" || headers.get("X-TP-Thumbnail-Max-Bytes") !== String(MAX_BYTES)
    || headers.get("X-TP-Thumbnail-Renderer") !== RENDERER
    || !digest(sha256) || !digest(source) || (policy !== "none" && !digest(policy))
    || (expected.view === "recorded" && !digest(policy))
    || (expected.sha256 !== null && sha256 !== expected.sha256)
    || (expected.sourceIdentity !== null && source !== expected.sourceIdentity)
    || (expected.policy !== null && policy !== expected.policy)
    || !Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 256 || height > 256
    || !Number.isSafeInteger(nativeWidth) || !Number.isSafeInteger(nativeHeight) || nativeWidth < 1 || nativeHeight < 1
    || bytes.byteLength < 33 || bytes.byteLength > MAX_BYTES) throw new Error("Thumbnail identity or bound verification failed.");
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (!signature.every((value, index) => bytes[index] === value) || data.getUint32(8) !== 13
    || String.fromCharCode(...bytes.slice(12, 16)) !== "IHDR" || data.getUint32(16) !== width || data.getUint32(20) !== height) throw new Error("Thumbnail PNG dimensions differ from verified evidence.");
  return { sha256, sourceIdentity: source, policy, width, height };
}

export function createPhotographicThumbnails(host) {
  const cache = new Map(), ephemeral = new Map(), tasks = new Map();
  const fetchImpl = host.fetchImpl || fetch;
  const urls = host.urls || URL;
  const timers = host.timers || globalThis;
  let scope = "", pageKey = "", generation = 0, renderVersion = 0, running = 0, bindings = [], observer = null, catalogueNotified = false;
  const context = () => host.context();
  const immutable = (proof) => digest(proof.sha256) && digest(proof.sourceIdentity) && digest(proof.policy);
  const entries = () => [...cache.values(), ...ephemeral.values()];
  function clearEphemeral() {
    ephemeral.forEach((entry) => { if (entry.url) urls.revokeObjectURL(entry.url); }); ephemeral.clear();
  }
  function currentTask(task) {
    const current = context();
    return task.generation === generation && (!task.volatile || task.renderVersion === renderVersion) && current.ready === true && current.enabled === true
      && JSON.stringify([current.owner, current.jobId, task.expected.catalogue, current.view]) === scope;
  }
  function clearWork() {
    generation += 1; renderVersion += 1; clearEphemeral(); observer?.disconnect(); observer = null;
    tasks.forEach((task) => task.controller?.abort()); tasks.clear(); bindings = [];
  }
  function reset() {
    clearWork(); cache.forEach((entry) => { if (entry.url) urls.revokeObjectURL(entry.url); }); cache.clear(); scope = pageKey = "";
  }
  function keep(key, entry, reusable = true) {
    const target = reusable ? cache : ephemeral;
    target.delete(key); target.set(key, entry);
    let bytes = entries().reduce((total, row) => total + row.bytes, 0);
    while (cache.size + ephemeral.size > MAX_ENTRIES || bytes > MAX_CACHE_BYTES) {
      const victim = cache.size ? cache : ephemeral;
      const first = victim.keys().next().value, evicted = victim.get(first);
      victim.delete(first); bytes -= evicted.bytes; if (evicted.url) urls.revokeObjectURL(evicted.url);
    }
  }
  function paint(binding, entry) {
    if (entry.failed) {
      binding.image.style.visibility = "hidden"; binding.status.textContent = "Preview unavailable. Open photograph.";
      binding.control.dataset.thumbnailState = "failed"; return;
    }
    binding.image.src = entry.url; binding.image.width = entry.width; binding.image.height = entry.height;
    binding.image.style.visibility = ""; binding.status.textContent = ""; binding.control.dataset.thumbnailState = "ready";
  }
  function expected(scene, page, current) {
    const hint = scene.thumbnail;
    if (!hint || typeof hint.url !== "string" || typeof scene.scene_id !== "string" || !scene.scene_id
      || typeof hint.artifact_path !== "string" || !hint.artifact_path || hint.artifact_path.startsWith("/") || /[\\\x00-\x1f\x7f]/.test(hint.artifact_path)
      || hint.artifact_path.split("/").some((part) => ["", ".", ".."].includes(part))
      || ![hint.sha256, hint.source_identity_sha256, hint.presentation_policy_sha256].every((value) => value === null || digest(value))) return null;
    const base = `/v1/jobs/${encodeURIComponent(current.jobId)}/photographic-review/thumbnail`;
    try {
      const url = new URL(hint.url, host.origin || globalThis.location?.origin || "http://localhost");
      if (url.pathname !== base || url.search || url.hash || url.username || url.password
        || url.origin !== (host.origin || globalThis.location?.origin || "http://localhost")) return null;
    } catch { return null; }
    const view = current.view === "recorded" && hint.presentation_policy_sha256 === null ? "clip_srgb" : current.view;
    return { catalogue: page.catalogue_identity_sha256, sceneId: scene.scene_id, artifactPath: hint.artifact_path,
      sha256: hint.sha256, sourceIdentity: hint.source_identity_sha256, policy: hint.presentation_policy_sha256, view,
      url: `${base}?${new URLSearchParams({ scene_id: scene.scene_id, catalogue_identity_sha256: page.catalogue_identity_sha256, view })}` };
  }
  async function load(task) {
    const controller = new AbortController(); task.controller = controller;
    let timer, reader;
    const deadline = new Promise((_, reject) => { timer = timers.setTimeout(() => { controller.abort(); reject(new Error("Thumbnail timed out.")); }, 15000); });
    try {
      const response = await Promise.race([fetchImpl(task.expected.url, { method: "GET", credentials: "same-origin", cache: "no-store", headers: host.headers(), signal: controller.signal }), deadline]);
      if (!currentTask(task) || controller.signal.aborted) return;
      if ([401, 403].includes(response.status)) { host.onAuthorizationFailure?.(response.status); throw new Error("Thumbnail access must be checked again."); }
      if (response.status === 409) {
        if (!catalogueNotified) {
          catalogueNotified = true;
          const oldBindings = bindings;
          clearWork(); cache.forEach((entry) => { if (entry.url) urls.revokeObjectURL(entry.url); }); cache.clear();
          oldBindings.forEach((binding) => { const failed = { failed: true, bytes: 0, url: null }; keep(binding.key, failed, immutable(binding.expected)); paint(binding, failed); });
          host.onCatalogueChanged?.();
        }
        throw new Error("Catalogue changed; refresh review.");
      }
      if (!response.ok || !response.body) throw new Error("Thumbnail unavailable.");
      const size = response.headers.get("Content-Length");
      if (size !== null && (!/^\d+$/.test(size) || Number(size) < 1 || Number(size) > MAX_BYTES)) throw new Error("Thumbnail exceeds its byte bound.");
      reader = response.body.getReader();
      const chunks = []; let length = 0;
      for (;;) {
        const { value, done } = await Promise.race([reader.read(), deadline]); if (done) break;
        length += value.byteLength; if (length > MAX_BYTES) throw new Error("Thumbnail exceeds its byte bound."); chunks.push(value);
      }
      if (size !== null && Number(size) !== length) throw new Error("Thumbnail transfer is incomplete.");
      const bytes = new Uint8Array(length); let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
      const evidence = photographicThumbnailEvidence(response, bytes, task.expected);
      if (!currentTask(task) || controller.signal.aborted) return;
      const entry = { ...evidence, bytes: length, url: urls.createObjectURL(new Blob([bytes], { type: "image/png" })) };
      keep(task.key, entry, !task.volatile);
      bindings.filter((binding) => binding.key === task.key).forEach((binding) => paint(binding, entry));
    } catch {
      if (currentTask(task)) {
        const failed = { failed: true, bytes: 0, url: null }; keep(task.key, failed, !task.volatile);
        bindings.filter((binding) => binding.key === task.key).forEach((binding) => paint(binding, failed));
      }
    } finally {
      timers.clearTimeout(timer); controller.abort();
      if (reader) { void reader.cancel().catch(() => {}); try { reader.releaseLock(); } catch { /* Deadline can leave a read pending. */ } }
    }
  }
  function pump() {
    while (running < 2) {
      const task = [...tasks.values()].find((item) => item.queued && !item.started);
      if (!task) return;
      task.started = true; running += 1;
      void load(task).finally(() => { running -= 1; if (tasks.get(task.key) === task) tasks.delete(task.key); pump(); });
    }
  }
  function queue(binding) {
    if (ephemeral.has(binding.key)) { paint(binding, ephemeral.get(binding.key)); return; }
    if (!tasks.has(binding.key)) tasks.set(binding.key, { key: binding.key, expected: binding.expected, generation, renderVersion, volatile: !immutable(binding.expected), queued: true, started: false });
    pump();
  }
  function render(page, cards) {
    const current = context();
    if (current.ready !== true || current.enabled !== true || !current.jobId || !current.owner || !digest(page?.catalogue_identity_sha256) || !TRANSFORMS[current.view]) { reset(); return; }
    const nextScope = JSON.stringify([current.owner, current.jobId, page.catalogue_identity_sha256, current.view]);
    if (scope !== nextScope) { reset(); scope = nextScope; catalogueNotified = false; }
    const nextPage = JSON.stringify((page.scenes || []).map((scene) => scene.scene_id));
    if (pageKey !== nextPage) { clearWork(); pageKey = nextPage; }
    // A null indexed identity cannot authorize reuse: reread private bytes for
    // every new DOM binding, even when legacy metadata keeps the same hash.
    renderVersion += 1; clearEphemeral();
    tasks.forEach((task, key) => { if (task.volatile) { task.controller?.abort(); tasks.delete(key); } });
    observer?.disconnect(); observer = null; bindings = [];
    const Observer = host.IntersectionObserver || globalThis.IntersectionObserver;
    if (Observer) observer = new Observer((entries) => entries.filter((entry) => entry.isIntersecting).forEach((entry) => {
      observer?.unobserve(entry.target); const binding = bindings.find((item) => item.image === entry.target); if (binding) queue(binding);
    }), { rootMargin: "150px" });
    for (const { scene, control } of cards) {
      if (!(page.scenes || []).some((row) => row.scene_id === scene.scene_id)) continue;
      const verified = expected(scene, page, current);
      if (!verified) {
        const notice = host.document.createElement("small"); notice.textContent = "Preview unavailable. Open photograph.";
        control.prepend(notice); control.dataset.thumbnailState = "failed"; continue;
      }
      const key = JSON.stringify([scope, verified.sceneId, verified.artifactPath, verified.sha256, verified.sourceIdentity, verified.policy, RENDERER]);
      const image = host.document.createElement("img"), status = host.document.createElement("small");
      image.alt = `${scene.label || scene.scene_id} · ${verified.view === "recorded" ? "recorded presentation" : current.view === "recorded" ? "clipping fallback" : "clipping view"} thumbnail`;
      image.width = 256; image.height = 144; image.style.visibility = "hidden"; image.decoding = "async"; image.dataset.ui = "photographic-page-thumbnail";
      status.textContent = "Preview pending"; status.dataset.ui = "photographic-thumbnail-status";
      control.prepend(image, status); control.dataset.thumbnailState = "pending";
      const binding = { key, expected: verified, image, status, control }; bindings.push(binding);
      image.onerror = () => {
        image.style.visibility = "hidden"; status.textContent = "Preview unavailable. Open photograph."; control.dataset.thumbnailState = "failed";
        const bad = cache.get(key) || ephemeral.get(key); if (bad?.url === image.src) { urls.revokeObjectURL(bad.url); keep(key, { failed: true, bytes: 0, url: null }, immutable(verified)); }
      };
      if (cache.has(key)) { const entry = cache.get(key); keep(key, entry); paint(binding, entry); }
      else if (catalogueNotified) paint(binding, { failed: true });
      else if (observer) observer.observe(image); else queue(binding);
    }
  }
  return { render, reset, stats: () => ({ entries: cache.size + ephemeral.size, cachedEntries: cache.size, bytes: entries().reduce((sum, row) => sum + row.bytes, 0), running, queued: [...tasks.values()].filter((row) => !row.started).length }) };
}
