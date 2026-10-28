const SHA256 = /^[a-f0-9]{64}$/;
const DECISIONS = new Set(["selected", "needs_revision", "approved"]);
const ROLES = Object.freeze({ original: "Original", final: "Final", previous_revision: "Previous revision", working_master: "Native working master" });

export function photographicSourceIdentity(variant) { return variant?.source_identity_sha256 || variant?.sha256 || ""; }
export function photographicDeliveryFingerprint(variant) { return JSON.stringify([variant.sha256, photographicSourceIdentity(variant)]); }

export function photographicVariantLabel(variant) {
  return `${ROLES[variant?.role] || "Revision"} · ${String(variant?.label || variant?.source_name || variant?.artifact_path || "Image")}${variant?.version ? ` · ${variant.version}` : ""}`;
}

export function photographicProperties(variant) {
  const dimensions = Number.isInteger(variant?.width) && Number.isInteger(variant?.height)
    ? `${variant.width} × ${variant.height} px` : "Dimensions not reported";
  const depth = /^float(\d+)$/.exec(String(variant?.bit_depth || ""));
  const precision = depth ? `${depth[1]}-bit float` : variant?.bit_depth ? `${variant.bit_depth}-bit` : "Precision not reported";
  return [dimensions, precision,
    variant?.color_profile || "Profile not reported", variant?.intended_use || "Intended use not reported"].join(" · ");
}

export function photographicDecisionMatches(decision, variant) {
  return Boolean(variant && decision && DECISIONS.has(decision.status)
    && decision.artifact_path === variant.artifact_path && SHA256.test(variant.sha256)
    && decision.sha256 === variant.sha256
    && (!variant.source_identity_sha256 || (decision.source_identity_sha256 || decision.sha256) === variant.source_identity_sha256));
}

export function nativeCropGeometry(variant, center = { x: 0.5, y: 0.5 }, size = 512) {
  if (!Number.isInteger(variant?.width) || !Number.isInteger(variant?.height)
    || variant.width < 1 || variant.height < 1) throw new Error("Native dimensions are unavailable.");
  const edge = Math.max(1, Math.min(2048, Math.floor(Number(size) || 512)));
  const width = Math.min(edge, variant.width);
  const height = Math.min(edge, variant.height);
  const bounded = (value) => Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0.5));
  return {
    x: Math.max(0, Math.min(variant.width - width, Math.round(bounded(center.x) * variant.width - width / 2))),
    y: Math.max(0, Math.min(variant.height - height, Math.round(bounded(center.y) * variant.height - height / 2))),
    width, height,
  };
}

export function photographicJobUrl(jobId, path = "") {
  return `/v1/jobs/${encodeURIComponent(String(jobId || ""))}/photographic-review${path}`;
}

export function photographicAssetUrl(raw, jobId, origin = globalThis.location?.origin) {
  try {
    const url = new URL(String(raw || ""), origin);
    const prefix = `/v1/jobs/${encodeURIComponent(String(jobId || ""))}/`;
    return url.origin === origin && ["http:", "https:"].includes(url.protocol)
      && url.pathname.startsWith(prefix) && !url.username && !url.password
      ? `${url.pathname}${url.search}` : "";
  } catch { return ""; }
}

export function photographicRecipe({ exposure = 0, warmth = 0, saturation = 1, materials = [] } = {}) {
  const numeric = (value, min, max, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number < min || number > max) throw new Error(`${label} is outside the supported range.`);
    return Math.round(number * 10000) / 10000;
  };
  const balance = numeric(warmth, -0.15, 0.15, "Warmth");
  const materialNames = new Set();
  if (!Array.isArray(materials) || materials.length > 16) throw new Error("At most 16 material regions are supported.");
  const regions = materials.map((material, index) => {
    const name = String(material.name || "").trim();
    if (!name || materialNames.has(name.toLowerCase())) throw new Error("Material regions require unique names.");
    materialNames.add(name.toLowerCase());
    const region = (material.region || []).map((value) => numeric(value, 0, 1, `Region ${index + 1}`));
    if (region.length !== 4 || region[0] >= region[2] || region[1] >= region[3]) throw new Error("Material regions require a positive-area normalized rectangle.");
    const regionWarmth = numeric(material.warmth || 0, -0.1, 0.1, "Material warmth");
    return { name, region, exposure_stops: numeric(material.exposure_stops, -0.5, 0.5, "Material exposure"),
      white_balance: [1 + regionWarmth, 1, 1 - regionWarmth], feather: numeric(material.feather, 0.005, 0.1, "Material feather") };
  });
  return {
    exposure_stops: numeric(exposure, -1, 1, "Exposure"),
    white_balance: [1 + balance, 1, 1 - balance],
    saturation: numeric(saturation, 0.8, 1.2, "Saturation"),
    materials: regions,
  };
}

export function photographicRetryAfterMs(value, now = Date.now()) {
  const seconds = Number(value);
  if (value !== null && String(value).trim() && Number.isFinite(seconds)) return Math.max(1000, Math.ceil(seconds * 1000));
  const date = Date.parse(String(value || ""));
  return Number.isFinite(date) ? Math.max(1000, date - now) : 30000;
}

export function photographicPreviewMatches(receipt, variant, recipe, referenceSha = null) {
  return Boolean(receipt && ["tp.photo.revision.v1", "tp.photo.revision.v2"].includes(receipt.schema) && variant
    && receipt.source_master_sha256 === variant.sha256
    && (!variant.source_identity_sha256 || (receipt.source_identity_sha256 || receipt.source_master_sha256) === variant.source_identity_sha256) && receipt.width === variant.width && receipt.height === variant.height
    && receipt.output_bit_depth === 32 && receipt.review_status === "unreviewed" && receipt.approval_transferred === false
    && (receipt.schema !== "tp.photo.revision.v2" || ((receipt.input_presentation_policy_sha256 || null) === (variant.presentation_policy_sha256 || null)
      && receipt.output_presentation?.schema === "tp.photo.revision_presentation.v1"
      && ["working_linear", "display_referred"].includes(receipt.output_presentation.stage)))
    && (receipt.reference_sha256 || null) === (referenceSha || null)
    && Object.entries(recipe).every(([key, value]) => JSON.stringify(receipt.recipe?.[key]) === JSON.stringify(value)));
}

const REVIEW_SELECTION_KEY = "tp.photo.review_selection.v1";
const relative = (value) => {
  if (typeof value !== "string" || value.length > 1024) return false;
  return !value.startsWith("/") && !/[\\\x00-\x1f\x7f]/.test(value)
    && !value.split("/").some((part) => !part || part === ".." || part === ".");
};

export function photographicReviewAttribution(recorded) {
  const actor = typeof recorded?.actor === "object" ? recorded.actor?.display_name || recorded.actor?.username || recorded.actor?.id : recorded?.actor;
  const date = recorded?.created_at ? new Date(typeof recorded.created_at === "number" ? recorded.created_at * 1000 : recorded.created_at).toLocaleString() : "Date not recorded";
  return `${actor || "Reviewer not recorded"} · ${date}`;
}

export function photographicPageReview(loaded, page) {
  const paths = new Set((loaded.scenes || []).flatMap((scene) => scene.variants || []).map((variant) => variant.artifact_path));
  const recordedScenes = new Map();
  for (const row of page.decisions || []) {
    if (!row.scene_id || row.status !== "approved" || !SHA256.test(row.sha256 || "")) continue;
    if (!recordedScenes.has(row.scene_id)) recordedScenes.set(row.scene_id, { scene_id: row.scene_id, label: row.label || row.scene_id, variants: [] });
    recordedScenes.get(row.scene_id).variants.push({ ...row, role: row.role || "previous_revision" });
  }
  const scenes = (loaded.scenes || []).map((scene) => ({ ...scene,
    railPaths: scene.variants.filter((variant) => ["final", "previous_revision"].includes(variant.role)).map((variant) => variant.artifact_path) }));
  return { review: { ...loaded, scenes, decisions: [...(page.decisions || []).filter((row) => !paths.has(row.artifact_path)), ...(loaded.decisions || [])] }, recordedScenes };
}

// Persist selectors and view settings only. The server still authorizes every
// detail, image and mutation; neither bytes nor credentials enter this store.
export function photographicSelectionStorage(storage) {
  function clean(value) {
    if (!value || typeof value !== "object" || !relative(value.sceneId) || !value.sceneId) return null;
    return { sceneId: value.sceneId,
      primaryPath: relative(value.primaryPath) ? value.primaryPath : "",
      comparePath: relative(value.comparePath) ? value.comparePath : "",
      mode: ["side_by_side", "wipe", "blink"].includes(value.mode) ? value.mode : "side_by_side",
      inspection: value.inspection === "native" ? "native" : "fit",
      presentationView: value.presentationView === "clip_srgb" ? "clip_srgb" : "recorded",
      center: { x: Math.max(0, Math.min(1, Number(value.center?.x) || 0)), y: Math.max(0, Math.min(1, Number(value.center?.y) || 0)) },
      cropSize: [256, 512, 1024, 2048].includes(value.cropSize) ? value.cropSize : 512 };
  }
  function read() {
    try {
      const raw = storage?.getItem(REVIEW_SELECTION_KEY);
      const rows = raw && raw.length <= 32768 ? JSON.parse(raw) : [];
      return Array.isArray(rows) && rows.length <= 16 ? rows.filter((row) => typeof row.scope === "string" && row.scope.length <= 8192 && clean(row.selection)) : [];
    } catch { return []; }
  }
  return {
    get(scope) { return clean(read().find((row) => row.scope === scope)?.selection); },
    set(scope, value) {
      const selection = clean(value);
      if (!selection || typeof scope !== "string" || scope.length > 8192) return;
      try {
        const rows = [...read().filter((row) => row.scope !== scope), { scope, selection }].slice(-16);
        const raw = JSON.stringify(rows);
        if (raw.length <= 32768) storage?.setItem(REVIEW_SELECTION_KEY, raw);
      } catch { /* Selection recovery is optional; mutations use durable state. */ }
    },
  };
}

export function createPhotographicPagedCatalogue(host) {
  const { el, button, field, select, request, status, selection, apply, hasEdits } = host;
  let page = null, previous = [], cursor = null, filter = "all", sort = "scene_id", generation = 0, busy = false, locked = false, requiredVersion = 0;
  let pendingNavigation = null;
  let discarding = false;
  let refreshPromise = null;
  const details = new Map();
  const controls = el("div", "", "photo-review-controls", "photographic-paged-controls");
  const filterControl = select("photographic-scene-filter", [["all", "All photographs"], ["unreviewed", "Unreviewed"], ["needs_revision", "Needs revision"], ["approved", "Approved"]], () => {
    const next = filterControl.value; filterControl.value = filter;
    void navigate(() => refresh(selection().sceneId, { filter: next }));
  });
  const sortControl = select("photographic-scene-sort", [["scene_id", "Photograph order"], ["label", "Photograph name"]], () => {
    const next = sortControl.value; sortControl.value = sort;
    void navigate(() => refresh(selection().sceneId, { sort: next }));
  });
  const pageStatus = el("p", "", "photo-review-help", "photographic-page-status");
  pageStatus.setAttribute("role", "status"); pageStatus.setAttribute("aria-live", "polite");
  const previousPage = button("Previous page", "photographic-page-previous", () => void navigate(() => turnPage(-1)));
  const nextPage = button("Next page", "photographic-page-next", () => void navigate(() => turnPage(1)));
  const previousScene = button("Previous photograph", "photographic-scene-previous", () => void navigate(() => advance(-1)));
  const nextScene = button("Next photograph", "photographic-scene-next", () => void navigate(() => advance(1)));
  const nextUnreviewed = button("Next unreviewed", "photographic-next-unreviewed", () => void navigate(() => unreviewed()));
  const discard = button("Discard unsaved edits and continue", "photographic-discard-navigation", () => {
    if (busy || locked) return;
    const action = pendingNavigation; pendingNavigation = null; discard.hidden = true;
    if (action) void run(async () => { discarding = true; try { return await action(); } finally { discarding = false; } });
  });
  discard.hidden = true;
  controls.append(field("Review status", filterControl), field("Sort photographs", sortControl), previousScene, nextScene, nextUnreviewed, previousPage, nextPage, discard, pageStatus);
  function handleKeydown(event) {
    if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(String(event.target.tagName).toUpperCase())) return;
    if (!["ArrowLeft", "ArrowRight"].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey) return;
    event.preventDefault(); void navigate(() => advance(event.key === "ArrowLeft" ? -1 : 1));
  }
  controls.addEventListener("keydown", handleKeydown);
  function renderState() {
    const selectedIndex = page?.scenes.findIndex((row) => row.scene_id === selection().sceneId) ?? -1;
    controls.dataset.loading = String(busy); controls.setAttribute("aria-busy", String(busy));
    filterControl.value = filter; sortControl.value = sort;
    previousPage.disabled = busy || locked || !previous.length; nextPage.disabled = busy || locked || !page?.next_cursor;
    previousScene.disabled = busy || locked || !page?.total_filtered || (selectedIndex === 0 && !previous.length);
    nextScene.disabled = busy || locked || !page?.total_filtered || (selectedIndex === page.scenes.length - 1 && !page.next_cursor);
    nextUnreviewed.disabled = filterControl.disabled = sortControl.disabled = discard.disabled = busy || locked;
    pageStatus.dataset.state = busy ? "loading" : page ? "ready" : "empty";
    pageStatus.textContent = busy ? "Loading photograph metadata…" : page
      ? `${page.total_filtered} of ${page.total_scenes} photographs · page ${previous.length + 1}. Recorded status; sources are checked when opened or exported.` : "No photograph metadata loaded.";
  }
  async function run(action) {
    if (busy) return false;
    const expected = generation, focused = controls.ownerDocument?.activeElement;
    busy = true; host.onLoading?.(true); renderState();
    try {
      const result = await action();
      if (result === true && generation === expected) status("Photograph navigation ready.");
      return result;
    }
    catch (error) { if (generation === expected && error.name !== "AbortError") status(error.message, true); return false; }
    finally { if (generation === expected) {
      busy = false; host.onLoading?.(false); renderState();
      if (controls.contains(focused) && !focused.disabled && focused.ownerDocument.activeElement === focused.ownerDocument.body) focused.focus();
    } }
  }
  function retainEdits(action) {
    if (hasEdits() && !discarding) {
      pendingNavigation = action; discard.hidden = false;
      status("Unsaved edits are retained. Save or explicitly discard them to continue.", true);
      return true;
    }
    return false;
  }
  async function navigate(action) {
    if (busy || locked || retainEdits(action)) return false;
    return run(action);
  }
  async function getPage(nextCursor = null, nextFilter = filter, nextSort = sort) {
    const query = new URLSearchParams({ limit: "8", status: nextFilter, sort: nextSort });
    if (nextCursor) query.set("cursor", nextCursor);
    const result = await request(`/scenes?${query}`);
    if (result.schema !== "tp.photo.scene_page.v1" || !SHA256.test(result.catalogue_identity_sha256 || "")
      || !Array.isArray(result.scenes) || result.scenes.length > 16 || !Number.isInteger(result.state_version)
      || result.scenes.some((row) => !relative(row.scene_id) || !row.scene_id)) throw new Error("Photograph page verification failed. Refresh review.");
    return result;
  }
  function getDetail(id, identity) {
    const key = JSON.stringify([generation, id, identity]);
    if (!details.has(key)) {
      const loading = request(`/scene?${new URLSearchParams({ scene_id: id, catalogue_identity_sha256: identity })}`)
        .then((result) => {
          if (result.schema !== "tp.photo.scene_detail.v1" || result.catalogue_identity_sha256 !== identity
            || result.scenes?.length !== 1 || result.scenes[0].scene_id !== id) throw new Error("Photograph detail and catalogue differ. Refresh review.");
          return result;
        }).finally(() => details.delete(key));
      details.set(key, loading);
    }
    return details.get(key);
  }
  async function commit(nextPage, nextCursor, nextPrevious, id, nextFilter = filter, nextSort = sort, requestedPath = "") {
    const expected = generation;
    const detail = id ? await getDetail(id, nextPage.catalogue_identity_sha256) : null;
    if (generation !== expected) return false;
    if (nextPage.state_version < requiredVersion) throw new Error("Review is stale. Refresh review.");
    if (detail && detail.state_version !== nextPage.state_version) throw new Error("Photograph detail and page versions differ. Refresh review.");
    const selected = selection();
    if (selected.sceneId && (selected.sceneId !== id || (selected.primaryPath && !detail?.scenes[0].variants.some((variant) => variant.artifact_path === selected.primaryPath)))
      && retainEdits(() => refresh(id, { filter: nextFilter, sort: nextSort, artifactPath: requestedPath }))) return false;
    page = nextPage; cursor = nextCursor; previous = nextPrevious; filter = nextFilter; sort = nextSort;
    pendingNavigation = null; discard.hidden = true;
    apply(detail || { ...page, scenes: [] }, { page, sceneChanged: selection().sceneId !== id, requestedPath, discardEdits: discarding });
    renderState(); return true;
  }
  async function refresh(id = selection().sceneId, options = {}) {
    const expected = generation, nextFilter = options.filter || filter, nextSort = options.sort || sort;
    let result = await getPage(null, nextFilter, nextSort), nextCursor = null;
    const stack = [];
    const requestedPath = relative(options.artifactPath) ? options.artifactPath : "";
    if (requestedPath && !id) id = result.decisions?.find((row) => row.artifact_path === requestedPath)?.scene_id || "";
    const matches = (row) => id ? row.scene_id === id : requestedPath && ((row.variant_paths || []).includes(requestedPath)
      || (row.related_artifacts || []).some((item) => item.artifact_path === requestedPath));
    // Recovery locates a saved selection using metadata pages only. A filter
    // may exclude it; retain its detail independently of the visible page.
    if ((id || requestedPath) && options.locate !== false && nextFilter === "all") {
      while (!result.scenes.some(matches) && result.next_cursor && stack.length < 63) {
        stack.push(nextCursor); nextCursor = result.next_cursor;
        result = await getPage(nextCursor, nextFilter, nextSort);
        if (generation !== expected) return false;
      }
      id = result.scenes.find(matches)?.scene_id || "";
    }
    if (generation !== expected) return false;
    return commit(result, nextCursor, stack, id || result.scenes[0]?.scene_id || "", nextFilter, nextSort, requestedPath);
  }
  async function recover(action) {
    try { return await action(); }
    catch (error) {
      if (error.code !== "CATALOGUE_CHANGED") throw error;
      const selected = selection().sceneId;
      await refresh(selected);
      status("Photograph catalogue changed. Metadata reloaded; selection retained. No change was replayed.", true);
      return false;
    }
  }
  async function turnPage(direction) {
    return recover(async () => {
      if (!page || (direction > 0 ? !page.next_cursor : !previous.length)) return false;
      const nextCursor = direction > 0 ? page.next_cursor : previous.at(-1);
      const stack = direction > 0 ? [...previous, cursor] : previous.slice(0, -1);
      const result = await getPage(nextCursor);
      return commit(result, nextCursor, stack, result.scenes[0]?.scene_id || "");
    });
  }
  async function choose(id, requestedPath = "") {
    return recover(async () => {
      if (!page) return false;
      return commit(page, cursor, previous, id, filter, sort, requestedPath);
    });
  }
  async function advance(direction) {
    if (!page) return false;
    const index = page.scenes.findIndex((row) => row.scene_id === selection().sceneId);
    const target = page.scenes[index + direction];
    if (target) return choose(target.scene_id);
    if (index < 0 && page.scenes.length) return choose(page.scenes[direction > 0 ? 0 : page.scenes.length - 1].scene_id);
    if (direction < 0 && previous.length) {
      return recover(async () => {
        const nextCursor = previous.at(-1), result = await getPage(nextCursor);
        return commit(result, nextCursor, previous.slice(0, -1), result.scenes.at(-1)?.scene_id || "");
      });
    }
    return turnPage(direction);
  }
  function compareText(left, right) {
    const a = Array.from(left), b = Array.from(right);
    for (let index = 0; index < a.length && index < b.length; index += 1) {
      const order = a[index].codePointAt(0) - b[index].codePointAt(0);
      if (order) return order;
    }
    return a.length - b.length;
  }
  async function unreviewed() {
    return recover(async () => {
      const selected = selection();
      const afterSelection = (row) => {
        const order = compareText(row[sort], sort === "label" ? selected.label : selected.sceneId);
        return order > 0 || (!order && compareText(row.scene_id, selected.sceneId) > 0);
      };
      let result = await getPage(null, "unreviewed"), nextCursor = null;
      const stack = [], first = result;
      let target = result.scenes.find(afterSelection);
      while (!target && result.next_cursor && stack.length < 63) {
        stack.push(nextCursor); nextCursor = result.next_cursor; result = await getPage(nextCursor, "unreviewed");
        target = result.scenes.find(afterSelection);
      }
      if (!target) { result = first; nextCursor = null; stack.length = 0; target = first.scenes[0]; }
      if (!target) { status("No unreviewed photographs remain in this catalogue."); return false; }
      return commit(result, nextCursor, stack, target.scene_id, "unreviewed");
    });
  }
  function renderCards(container, selectedId, selectedCard) {
    const focused = container.ownerDocument?.activeElement;
    const restoreFocus = focused?.dataset?.ui === "photographic-contact-item" && container.contains(focused);
    container.replaceChildren();
    for (const scene of page?.scenes || []) {
      const card = el("article", "", "photo-review-contact-item"); card.dataset.sceneId = scene.scene_id;
      const control = button("", "photographic-contact-item", () => void navigate(() => choose(scene.scene_id)));
      control.className = "photo-review-contact-select"; control.setAttribute("aria-pressed", String(scene.scene_id === selectedId));
      control.append(el("span", scene.label || scene.scene_id), el("small", String(scene.review_status || "unreviewed").replaceAll("_", " ")));
      const recorded = scene.recorded_decision;
      control.append(el("small", photographicReviewAttribution(recorded))); card.append(control);
      const disclosure = el("details", "", "photo-review-artifact-summary", "photographic-related-artifacts");
      disclosure.append(el("summary", `${scene.variant_count || 0} versions · master, preview, depth and evidence`));
      for (const role of ["master", "preview", "depth", "evidence"]) {
        const items = (scene.related_artifacts || []).filter((row) => row.role === role && relative(row.artifact_path));
        disclosure.append(el("p", `${role}: ${items.length ? items.map((row) => row.artifact_path).join(", ") : "Not recorded"}`, "photo-review-help"));
      }
      disclosure.append(el("p", "Recorded metadata; sources are checked when opened or exported.", "photo-review-help"));
      card.append(disclosure); if (scene.scene_id === selectedId) selectedCard?.(card, scene);
      container.append(card);
      if (restoreFocus && scene.scene_id === selectedId) control.focus();
    }
    renderState();
  }
  renderState();
  return { controls, renderCards, navigate, handleKeydown, refresh: async (id, options) => {
      const expected = generation;
      if (locked && !options?.reconcile) return false;
      requiredVersion = Math.max(requiredVersion, options?.minimumVersion || 0);
      if (options?.reconcile) await refreshPromise;
      if (expected !== generation) return false;
      if (busy && !refreshPromise) { status("Review is busy. Refresh when ready."); return false; }
      if (!refreshPromise) {
        refreshPromise = run(() => recover(() => refresh(id, options))).finally(() => { if (generation === expected) refreshPromise = null; });
      }
      return refreshPromise;
    },
    locate: (id, artifactPath) => navigate(() => recover(() => refresh(id, { artifactPath, filter: "all" }))),
    choose: (id, requestedPath) => navigate(() => choose(id, requestedPath)), getDetail: (id) => page ? getDetail(id, page.catalogue_identity_sha256) : Promise.reject(new Error("Refresh the photograph catalogue before export.")),
    page: () => page, current: () => page?.state_version >= requiredVersion, lock(value) { locked = value; renderState(); }, reset() { generation += 1; page = null; previous = []; cursor = null; filter = "all"; sort = "scene_id"; busy = false; host.onLoading?.(false); locked = false; requiredVersion = 0; refreshPromise = null; pendingNavigation = null; discard.hidden = true; details.clear(); renderState(); } };
}

export { photographicDeliveryAttempt, photographicDeliveryRecovered, photographicDeliveryRecoveryStorage } from "./photographic-delivery-history.js";
