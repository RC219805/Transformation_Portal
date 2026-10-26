import { photographicDeliveryAttempt, photographicDeliveryRecovered, photographicDeliveryRecoveryStorage } from "./photographic-delivery-history.js";

const SHA256 = /^[a-f0-9]{64}$/;
const DECISIONS = new Set(["selected", "needs_revision", "approved"]);
const ROLES = Object.freeze({ original: "Original", final: "Final", previous_revision: "Previous revision", working_master: "Native working master" });

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

function normalizedReview(data) {
  const scenes = Array.isArray(data?.scenes) ? data.scenes.map((scene) => ({
    ...scene,
    scene_id: String(scene.scene_id || ""),
    variants: (Array.isArray(scene.variants) ? scene.variants : []).filter((variant) =>
      variant && typeof variant.artifact_path === "string" && SHA256.test(variant.sha256 || "")),
  })).filter((scene) => scene.scene_id && scene.variants.length) : [];
  return { ...data, scenes, decisions: Array.isArray(data?.decisions) ? data.decisions : [],
    groups: Array.isArray(data?.groups) ? data.groups : [], presets: Array.isArray(data?.presets) ? data.presets : [] };
}

export function createPhotographicReview(host) {
  const { state, els, _buildAuthHeaders, createToast } = host;
  const fetchImpl = host.fetchImpl || ((...args) => fetch(...args));
  const doc = host.document || globalThis.document;
  let panel = null;
  let jobId = "";
  let inventoryKey = "";
  let actorScope = "";
  let data = null;
  let sceneId = "";
  let primaryPath = "";
  let comparePath = "";
  let mode = "side_by_side";
  let inspection = "fit";
  let presentationView = "recorded";
  let center = { x: 0.5, y: 0.5 };
  let cropSize = 512;
  let requests = new Set();
  let imageRequest = 0;
  let dataRequest = 0;
  const imageControllers = new Map();
  const imageCache = new Map();
  const imageInFlight = new Map();
  const thumbnailImages = new Map();
  const thumbnailLoads = new WeakMap();
  const thumbnailObserver = globalThis.IntersectionObserver ? new IntersectionObserver((entries) => {
    entries.filter((entry) => entry.isIntersecting).forEach((entry) => { thumbnailObserver.unobserve(entry.target); thumbnailLoads.get(entry.target)?.(); });
  }, { rootMargin: "200px" }) : null;
  let downloadController = null;
  let downloadObjectUrl = "";
  let downloadRevoke = null;
  let activeImageKey = "";
  let displayedInspectionKey = "";
  let imageTimer = null;
  let finishImageDelay = null;
  let retryAt = 0;
  let pending = false;
  let blinkTimer = null;
  let blinkAlternate = false;
  let variantPreview = null;
  let revisionDraft = { exposure: 0, warmth: 0, saturation: 1 };
  let materialDrafts = [];
  const presetEdits = new Map();
  const deliverySelections = new Map();
  const sceneLabels = new Map();
  let deliverySources = [];
  let selectedPresets = new Set();
  let deliveryAttempt = null;
  let deliveryOutcome = "idle";
  let deliveryHistory = [];
  let deliveryCursor = null;
  let historyRequest = 0;
  let scopeGeneration = 0;
  let recoveryStorage;
  try { recoveryStorage = photographicDeliveryRecoveryStorage(Object.hasOwn(host, "sessionStorage") ? host.sessionStorage : globalThis.sessionStorage); }
  catch { recoveryStorage = photographicDeliveryRecoveryStorage(null); }
  let nodes = {};
  function deliveryHistoryEnabled() { return state.auth?.features?.photographicDeliveryHistory === true; }

  function el(tag, text = "", className = "", ui = "") {
    const element = doc.createElement(tag);
    if (text) element.textContent = text;
    if (className) element.className = className;
    if (ui) element.dataset.ui = ui;
    return element;
  }
  function button(text, ui, action) {
    const element = el("button", text, "operator-action-btn operator-action-btn-secondary", ui);
    element.type = "button";
    element.addEventListener("click", action);
    return element;
  }
  function field(label, control) {
    const wrapper = el("label", "", "photo-review-field");
    wrapper.append(el("span", label), control);
    return wrapper;
  }
  function select(ui, choices, change) {
    const element = el("select", "", "photo-review-input", ui);
    choices.forEach(([value, label]) => { const option = el("option", label); option.value = value; element.append(option); });
    element.addEventListener("change", change);
    return element;
  }
  function input(ui, type = "text", value = "") {
    const element = el("input", "", "photo-review-input", ui);
    element.type = type;
    element.value = value;
    return element;
  }
  function status(message, error = false) {
    if (!nodes.status) return;
    nodes.status.textContent = message;
    nodes.status.dataset.tone = error ? "error" : "info";
  }
  function stopBlink() {
    if (blinkTimer) clearInterval(blinkTimer);
    blinkTimer = null;
    blinkAlternate = false;
    if (nodes.stage) nodes.stage.dataset.blinkAlternate = "false";
    if (nodes.blink) { nodes.blink.textContent = "Start blink"; nodes.blink.setAttribute("aria-pressed", "false"); }
  }
  function releaseImages() {
    displayedInspectionKey = "";
    clearDownload();
    thumbnailObserver?.disconnect();
    thumbnailImages.forEach((image) => { if (image.src?.startsWith("blob:")) URL.revokeObjectURL(image.src); });
    imageCache.forEach((entry) => URL.revokeObjectURL(entry.url));
    imageCache.clear(); imageInFlight.clear(); thumbnailImages.clear(); activeImageKey = "";
    if (imageTimer) clearTimeout(imageTimer);
    finishImageDelay?.(); finishImageDelay = null; imageTimer = null;
  }
  function checkBackoff() {
    if (Date.now() < retryAt) throw new Error(`Rate limited. Retry after ${Math.ceil((retryAt - Date.now()) / 1000)} seconds; your selection and saved decisions are retained.`);
  }
  function checkRateLimit(response) {
    if (response.status !== 429) return;
    retryAt = Math.max(retryAt, Date.now() + photographicRetryAfterMs(response.headers.get("Retry-After")));
    checkBackoff();
  }
  function identity(variant) { return variant?.source_identity_sha256 || variant?.sha256 || ""; }
  function recordedPresentationEnabled() { return state.auth?.features?.photographicRecordedPresentation === true; }
  function presentationAware(variant) { return Array.isArray(variant?.presentation?.available_views) && variant.presentation.available_views.includes("clip_srgb"); }
  function viewFor(variant) {
    return recordedPresentationEnabled() && presentationView === "recorded" && variant?.presentation?.available_views?.includes("recorded")
      && SHA256.test(variant.presentation_policy_sha256 || "") ? "recorded" : "clip_srgb";
  }
  function imageKey(variant, native, edge = 1600) {
    return JSON.stringify([jobId, actorScope, variant.artifact_path, variant.sha256, identity(variant),
      viewFor(variant), variant.presentation_policy_sha256 || null,
      native ? nativeCropGeometry(variant, center, cropSize) : { max_edge: edge }]);
  }
  function currentScene() { return data?.scenes.find((scene) => scene.scene_id === sceneId) || null; }
  function currentVariant(path = primaryPath) { return currentScene()?.variants.find((variant) => variant.artifact_path === path) || null; }
  function decision(variant) { return data?.decisions.find((item) => photographicDecisionMatches(item, variant)) || null; }
  function setPending(value) {
    pending = value;
    panel?.querySelectorAll("[data-review-mutation]").forEach((control) => { control.disabled = value; });
    if (nodes.commitRevision) nodes.commitRevision.disabled = value || !variantPreview;
    nodes.decisions?.querySelectorAll("[data-decision]").forEach((control) => { control.disabled = value || currentVariant()?.role === "original"; });
  }
  function resetPreview() {
    variantPreview = null;
    if (nodes.variantPreview) { nodes.variantPreview.classList.add("hidden"); nodes.variantPreview.removeAttribute("src"); nodes.revisionReceipt.textContent = ""; }
    if (nodes.commitRevision) nodes.commitRevision.disabled = true;
  }

  async function request(path = "", body = null, timeout = 20000) {
    checkBackoff();
    const controller = new AbortController();
    requests.add(controller);
    const requestJob = jobId;
    const requestActor = actorScope;
    const requestGeneration = scopeGeneration;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    try {
      const method = body ? "POST" : "GET";
      const response = await fetchImpl(photographicJobUrl(requestJob, path), {
        method, credentials: "same-origin", signal: controller.signal,
        headers: _buildAuthHeaders({ Accept: "application/json", ...(body ? { "Content-Type": "application/json" } : {}) }, method),
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      checkRateLimit(response);
      const payload = await response.json();
      if (requestJob !== jobId || requestActor !== actorScope || requestGeneration !== scopeGeneration) throw new Error("Review selection changed.");
      if (!response.ok || payload.success === false || !payload.data) {
        const error = new Error(payload.error?.message || payload.detail || `Review request failed (${response.status}).`);
        error.code = payload.error?.code;
        error.httpStatus = response.status;
        throw error;
      }
      return payload.data;
    } catch (error) {
      if (timedOut) {
        const deadline = new Error("Review request timed out. Refresh review before retrying; a submitted change may have completed.");
        deadline.name = "TimeoutError";
        throw deadline;
      }
      throw error;
    } finally { clearTimeout(timer); requests.delete(controller); }
  }

  function mount() {
    if (panel) return;
    panel = el("section", "", "photo-review hidden", "photographic-review");
    panel.setAttribute("aria-label", "Photographic review and delivery");
    const heading = el("div", "", "photo-review-heading");
    heading.append(el("h3", "Photographic review", "photo-review-title"),
      el("p", "Inspect native detail, record an image decision, and deliver an approved set. Verification and human approval remain distinct.", "photo-review-help"));
    nodes.status = el("p", "Loading photographic review…", "photo-review-status", "photographic-review-status");
    nodes.status.setAttribute("role", "status");
    nodes.status.setAttribute("aria-live", "polite");
    const refresh = button("Refresh review", "photographic-review-refresh", () => void load(true));
    nodes.scene = select("photographic-scene", [], () => { sceneId = nodes.scene.value; chooseVariants(); render(); });
    nodes.primary = select("photographic-primary", [], () => { primaryPath = nodes.primary.value; resetRecipe(); render(); });
    nodes.compare = select("photographic-comparison", [], () => { comparePath = nodes.compare.value; render(); });
    nodes.mode = select("photographic-comparison-mode", [["side_by_side", "Side by side"], ["wipe", "Wipe"], ["blink", "Blink"]], () => { mode = nodes.mode.value; stopBlink(); renderStage(); });
    nodes.inspection = select("photographic-inspection", [["fit", "Fit preview"], ["native", "Native 1:1 crop"]], () => { inspection = nodes.inspection.value; stopBlink(); void renderImages(); });
    nodes.presentation = select("photographic-presentation", [["recorded", "Recorded presentation when verified"], ["clip_srgb", "Legacy clipping comparison"]], () => {
      presentationView = nodes.presentation.value; stopBlink(); renderContactSheet(); void renderImages();
    });
    const controls = el("div", "", "photo-review-controls");
    controls.append(field("Photograph", nodes.scene), field("Selected version", nodes.primary), field("Compare with", nodes.compare),
      field("Comparison", nodes.mode), field("Inspection", nodes.inspection), field("Presentation", nodes.presentation), refresh);
    nodes.nativeControls = el("div", "", "photo-review-controls hidden", "photographic-native-controls");
    nodes.cropSize = select("photographic-crop-size", [["256", "256 px crop"], ["512", "512 px crop"], ["1024", "1024 px crop"], ["2048", "2048 px crop"]], () => { cropSize = Number(nodes.cropSize.value); void renderImages(); });
    nodes.cropSize.value = String(cropSize);
    nodes.x = input("photographic-crop-x", "range", "50");
    nodes.y = input("photographic-crop-y", "range", "50");
    for (const [axis, control] of [["x", nodes.x], ["y", nodes.y]]) {
      control.min = "0"; control.max = "100"; control.step = "1";
      control.addEventListener("change", () => { center[axis] = Number(control.value) / 100; void renderImages(); });
    }
    nodes.nativeControls.append(field("Crop size", nodes.cropSize), field("Horizontal center (%)", nodes.x), field("Vertical center (%)", nodes.y));
    for (const [name, dx, dy] of [["Left", -1, 0], ["Right", 1, 0], ["Up", 0, -1], ["Down", 0, 1]]) {
      nodes.nativeControls.append(button(name, `photographic-pan-${name.toLowerCase()}`, () => pan(dx, dy)));
    }
    nodes.nativeNotice = el("p", "", "photo-review-help", "photographic-resolution-status");
    nodes.stage = el("div", "", "photo-review-stage", "photographic-comparison-stage");
    nodes.metadata = el("div", "", "photo-review-metadata");
    nodes.figures = ["primary", "comparison"].map((name) => {
      const figure = el("figure", "", "photo-review-figure");
      const viewport = el("div", "", "photo-review-viewport", `photographic-${name}-viewport`);
      viewport.tabIndex = 0;
      viewport.setAttribute("aria-label", `${name === "primary" ? "Selected" : "Comparison"} image. Use arrow keys to pan native crops.`);
      const image = el("img", "", "photo-review-image", `photographic-${name}-image`);
      image.alt = "";
      image.decoding = "async";
      const caption = el("figcaption", "", "photo-review-image-caption", `photographic-${name}-properties`);
      viewport.append(image);
      figure.append(viewport);
      nodes.metadata.append(caption);
      viewport.addEventListener("keydown", (event) => {
        const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
        if (inspection === "native" && direction) { event.preventDefault(); pan(...direction); }
      });
      viewport.addEventListener("scroll", () => {
        const peer = nodes.figures.find((item) => item.viewport !== viewport)?.viewport;
        if (!peer) return;
        if (Math.abs(peer.scrollLeft - viewport.scrollLeft) > 1) peer.scrollLeft = viewport.scrollLeft;
        if (Math.abs(peer.scrollTop - viewport.scrollTop) > 1) peer.scrollTop = viewport.scrollTop;
      });
      nodes.stage.append(figure);
      return { figure, viewport, image, caption };
    });
    nodes.wipe = input("photographic-wipe", "range", "50");
    nodes.wipe.min = "0"; nodes.wipe.max = "100";
    nodes.wipe.addEventListener("input", () => nodes.stage.style.setProperty("--photo-wipe", `${nodes.wipe.value}%`));
    nodes.wipeField = field("Wipe position", nodes.wipe);
    nodes.blink = button("Start blink", "photographic-blink", () => {
      if (blinkTimer) { stopBlink(); return; }
      blinkTimer = setInterval(() => { blinkAlternate = !blinkAlternate; nodes.stage.dataset.blinkAlternate = String(blinkAlternate); }, 750);
      nodes.blink.textContent = "Stop blink";
      nodes.blink.setAttribute("aria-pressed", "true");
    });
    nodes.blink.setAttribute("aria-pressed", "false");
    const comparisonTools = el("div", "", "photo-review-controls");
    comparisonTools.append(nodes.wipeField, nodes.blink);
    nodes.decisions = el("div", "", "photo-review-decisions", "photographic-decisions");
    const quality = el("section", "", "photo-review-section", "photographic-quality");
    nodes.measureQuality = button("Measure local quality signals", "photographic-measure-quality", () => void measureQuality());
    nodes.quality = el("div", "", "photo-review-help", "photographic-quality-results");
    quality.append(nodes.measureQuality, nodes.quality);
    nodes.decisionStatus = el("p", "Unreviewed", "photo-review-help", "photographic-decision-status");
    nodes.note = el("textarea", "", "photo-review-input", "photographic-decision-note");
    nodes.note.maxLength = 2000;
    nodes.note.rows = 2;
    for (const [value, label] of [["selected", "Selected"], ["needs_revision", "Needs revision"], ["approved", "Approved"]]) {
      const control = button(label, `photographic-decision-${value}`, () => void saveDecision(value));
      control.dataset.reviewMutation = "true";
      control.dataset.decision = value;
      nodes.decisions.append(control);
    }
    nodes.decisions.append(nodes.decisionStatus, field("Review note", nodes.note));
    nodes.contact = el("div", "", "photo-review-contact", "photographic-contact-sheet");
    nodes.groups = el("div", "", "photo-review-controls", "photographic-lighting-groups");
    nodes.groupName = input("photographic-lighting-group", "text");
    nodes.groupName.maxLength = 80;
    nodes.reference = select("photographic-reference", [], () => resetPreview());
    const saveGroup = button("Set lighting reference", "photographic-save-reference", () => void saveLightingGroup());
    saveGroup.dataset.reviewMutation = "true";
    nodes.groups.append(field("Lighting group for this photograph", nodes.groupName), field("Reference photograph", nodes.reference), saveGroup);
    nodes.outliers = el("p", "", "photo-review-help", "photographic-quality-flags");
    const revisions = buildRevisionControls();
    const delivery = buildDeliveryControls();
    panel.append(heading, controls, nodes.nativeControls, nodes.nativeNotice, nodes.stage, nodes.metadata, comparisonTools,
      quality, nodes.decisions, el("h4", "Set coherence", "photo-review-subtitle"), nodes.contact, nodes.groups, nodes.outliers, revisions, delivery, nodes.status);
    if (els.artifactShellContent?.prepend) els.artifactShellContent.prepend(panel);
    else els.artifactShellContent?.append(panel);
  }

  function buildRevisionControls() {
    const section = el("section", "", "photo-review-section", "photographic-revision");
    section.append(el("h4", "Create a separate revision", "photo-review-subtitle"),
      el("p", "Bounded adjustments create a new unapproved artifact. Originals and existing deliveries stay unchanged. Inspect the rendered variant before creating it.", "photo-review-help"));
    const controls = el("div", "", "photo-review-controls");
    nodes.revisionInputs = {};
    for (const [key, label, min, max, step] of [["exposure", "Exposure (EV)", -1, 1, 0.05], ["warmth", "Warmth balance", -0.15, 0.15, 0.01], ["saturation", "Saturation", 0.8, 1.2, 0.05]]) {
      const control = input(`photographic-revision-${key}`, "number", String(revisionDraft[key]));
      control.min = String(min); control.max = String(max); control.step = String(step);
      control.addEventListener("input", () => { revisionDraft[key] = control.value; resetPreview(); });
      nodes.revisionInputs[key] = control;
      controls.append(field(label, control));
    }
    nodes.previewRevision = button("Preview variant", "photographic-preview-revision", () => void revise(true));
    nodes.commitRevision = button("Create revision", "photographic-create-revision", () => void revise(false));
    nodes.commitRevision.disabled = true;
    nodes.previewRevision.dataset.reviewMutation = "true";
    nodes.commitRevision.dataset.reviewMutation = "true";
    nodes.variantPreview = el("img", "", "photo-review-variant hidden", "photographic-variant-preview");
    nodes.variantPreview.alt = "Rendered adjustment preview; reduced resolution. New revision has not been created.";
    nodes.revisionReceipt = el("p", "", "photo-review-help", "photographic-revision-receipt");
    controls.append(nodes.previewRevision, nodes.commitRevision);
    nodes.materials = el("div", "", "photo-review-materials", "photographic-material-regions");
    const addRegion = button("Add local material region", "photographic-add-material", () => {
      if (materialDrafts.length >= 16) { status("At most 16 local regions are supported.", true); return; }
      materialDrafts.push({ name: `Region ${materialDrafts.length + 1}`, region: [0.25, 0.25, 0.75, 0.75], exposure_stops: 0, warmth: 0, feather: 0.03 });
      resetPreview(); renderMaterials();
    });
    section.append(controls, el("p", "Local material regions are explicit feathered rectangles, not automatic semantic masks. Keep adjustments subtle; overlapping exposure is checked by the renderer.", "photo-review-help"),
      addRegion, nodes.materials, nodes.variantPreview, nodes.revisionReceipt);
    return section;
  }

  function buildDeliveryControls() {
    const section = el("section", "", "photo-review-section", "photographic-delivery");
    section.append(el("h4", "Deliver approved set", "photo-review-subtitle"),
      el("p", "Choose destination requirements. The bundle includes meaningful filenames, a contact sheet, inventory, and verified source-master links. Approval is checked again at export.", "photo-review-help"));
    nodes.property = input("photographic-property-name", "text", "Property");
    nodes.property.maxLength = 100;
    nodes.presets = el("div", "", "photo-review-controls", "photographic-delivery-presets");
    nodes.deliver = button("Deliver approved set", "photographic-deliver-approved", () => void deliver());
    nodes.deliver.dataset.reviewMutation = "true";
    nodes.deliver.disabled = true;
    nodes.deliveryResult = el("div", "", "photo-review-help", "photographic-delivery-result");
    nodes.deliveryHistory = el("div", "", "photo-review-help", "photographic-delivery-history");
    nodes.deliveryHistoryStatus = el("p", "", "photo-review-help", "photographic-delivery-history-status");
    nodes.reconcileDelivery = button("Refresh delivery history / reconcile", "photographic-delivery-reconcile", () => void loadDeliveryHistory());
    nodes.moreDeliveries = button("Load older deliveries", "photographic-delivery-history-more", () => void loadDeliveryHistory(true));
    nodes.moreDeliveries.disabled = true;
    nodes.deliveryHistoryControls = el("section", "", "hidden", "photographic-delivery-history-controls");
    nodes.deliveryHistoryControls.append(el("h5", "Committed delivery history"), nodes.reconcileDelivery,
      nodes.deliveryHistoryStatus, nodes.deliveryHistory, nodes.moreDeliveries);
    section.append(field("Property name", nodes.property), nodes.presets, nodes.deliver, nodes.deliveryResult, nodes.deliveryHistoryControls);
    return section;
  }

  function resetRecipe() {
    revisionDraft = { exposure: 0, warmth: 0, saturation: 1 };
    materialDrafts = [];
    Object.entries(nodes.revisionInputs || {}).forEach(([key, control]) => { control.value = String(revisionDraft[key]); });
    renderMaterials();
    resetPreview();
  }

  function renderMaterials() {
    if (!nodes.materials) return;
    nodes.materials.replaceChildren();
    materialDrafts.forEach((material, index) => {
      const controls = el("fieldset", "", "photo-review-material-region");
      controls.append(el("legend", `Local region ${index + 1}`));
      const name = input("photographic-material-name", "text", material.name);
      name.maxLength = 80;
      name.addEventListener("input", () => { material.name = name.value; resetPreview(); });
      controls.append(field("Material / region name", name));
      for (const [position, label] of ["Left", "Top", "Right", "Bottom"].entries()) {
        const control = input(`photographic-material-${label.toLowerCase()}`, "number", String(material.region[position]));
        control.min = "0"; control.max = "1"; control.step = "0.01";
        control.addEventListener("input", () => { material.region[position] = control.value; resetPreview(); });
        controls.append(field(`${label} (0–1)`, control));
      }
      for (const [key, label, min, max, step] of [["exposure_stops", "Local exposure (EV)", -0.5, 0.5, 0.05], ["warmth", "Local warmth", -0.1, 0.1, 0.01], ["feather", "Edge feather", 0.005, 0.1, 0.005]]) {
        const control = input(`photographic-material-${key}`, "number", String(material[key]));
        control.min = String(min); control.max = String(max); control.step = String(step);
        control.addEventListener("input", () => { material[key] = control.value; resetPreview(); });
        controls.append(field(label, control));
      }
      controls.append(button("Remove region", "photographic-remove-material", () => { materialDrafts.splice(index, 1); resetPreview(); renderMaterials(); }));
      nodes.materials.append(controls);
    });
  }

  function chooseVariants() {
    const scene = currentScene() || data?.scenes[0];
    sceneId = scene?.scene_id || "";
    primaryPath = scene?.variants.find((variant) => variant.role === "final")?.artifact_path || scene?.variants[0]?.artifact_path || "";
    comparePath = scene?.variants.find((variant) => variant.role === "original")?.artifact_path
      || scene?.variants.find((variant) => variant.artifact_path !== primaryPath)?.artifact_path || "";
    center = { x: 0.5, y: 0.5 };
    resetRecipe();
  }

  async function load(force = false) {
    if (!jobId) return;
    if (force) void loadDeliveryHistory();
    const expectedJob = jobId;
    const sequence = ++dataRequest;
    status("Loading photographic review…");
    try {
      const loaded = normalizedReview(await request("", null, 90000));
      if (jobId !== expectedJob || sequence !== dataRequest) return;
      data = loaded;
      if (deliverySources.some((source) => !data.scenes.some((scene) => scene.variants.some((variant) =>
        photographicDecisionMatches(source, variant) && decision(variant)?.status === "approved")))) {
        clearDownload();
        nodes.deliveryResult.textContent = "A delivery approval changed. Recheck the selected versions and create a newly verified bundle.";
        deliverySources = [];
      }
      if (!currentVariant()) chooseVariants();
      if (!data.scenes.length) { panel.classList.toggle("hidden", !deliveryHistory.length); return; }
      panel.classList.remove("hidden");
      render();
      status(force ? "Review refreshed from verified artifacts and saved decisions." : "Review loaded. A saved approval applies only to the exact SHA-256 displayed here.");
    } catch (error) {
      if (expectedJob !== jobId || sequence !== dataRequest || error.name === "AbortError") return;
      if (error.httpStatus === 404 || error.httpStatus === 422) { panel.classList.toggle("hidden", !deliveryHistory.length); return; }
      panel.classList.remove("hidden");
      status(error.message, true);
    }
  }

  function render() {
    if (!data || !currentScene()) return;
    const scene = currentScene();
    const replaceOptions = (control, choices, value) => {
      control.replaceChildren();
      choices.forEach(([key, label]) => { const option = el("option", label); option.value = key; control.append(option); });
      control.value = value;
    };
    replaceOptions(nodes.scene, data.scenes.map((item) => [item.scene_id, item.label || item.scene_id]), sceneId);
    replaceOptions(nodes.primary, scene.variants.map((variant) => [variant.artifact_path, photographicVariantLabel(variant)]), primaryPath);
    replaceOptions(nodes.compare, [["", "No comparison"], ...scene.variants.filter((variant) => variant.artifact_path !== primaryPath)
      .map((variant) => [variant.artifact_path, photographicVariantLabel(variant)])], comparePath === primaryPath ? "" : comparePath);
    if (comparePath === primaryPath) comparePath = "";
    const currentDecision = decision(currentVariant());
    nodes.decisionStatus.textContent = currentVariant()?.role === "original"
      ? "Original inspection only. Select a final or revision to record an approval."
      : currentDecision
      ? `${currentDecision.status === "needs_revision" ? "Needs revision" : currentDecision.status[0].toUpperCase() + currentDecision.status.slice(1)} · exact SHA-256 ${currentVariant().sha256}`
      : `Unreviewed · exact SHA-256 ${currentVariant().sha256}`;
    nodes.decisions.querySelectorAll("[data-decision]").forEach((control) => control.setAttribute("aria-pressed", String(control.dataset.decision === currentDecision?.status)));
    nodes.note.value = currentDecision?.note || "";
    const group = data.groups.find((item) => item.members?.includes(identity(currentVariant())))
      || data.groups.find((item) => item.name === scene.lighting_group);
    nodes.groupName.value = group?.name || scene.lighting_group || "";
    const referenceVariants = data.scenes.flatMap((item) => item.variants.filter((variant) => variant.role !== "original").map((variant) => ({ scene: item, variant })));
    replaceOptions(nodes.reference, [["", "No reference selected"], ...referenceVariants.map(({ scene: item, variant }) => [identity(variant), `${item.label || item.scene_id} · ${photographicVariantLabel(variant)}`])], group?.reference_sha256 || scene.reference_sha256 || "");
    const cohesion = (data.cohesion?.groups || []).flatMap((group) => group.images || []).find((row) => (row.source_identity_sha256 || row.source_master_sha256) === identity(currentVariant()));
    nodes.outliers.textContent = (scene.quality_flags || scene.outlier_flags || cohesion?.flags || []).map((flag) => typeof flag === "string" ? flag : flag.message || flag.code || "").filter(Boolean).join(" · ")
      || "No automatic quality flags reported. Review sky, fine edges, foliage, texture, and intentional lighting manually.";
    if (cohesion) nodes.outliers.textContent += ` Reference-relative exposure ${Number(cohesion.exposure_delta_ev).toFixed(2)} EV; chromaticity difference ${Number(cohesion.chromaticity_delta).toFixed(3)}. Composition and intentional warmth or twilight can explain these advisory signals.`;
    nodes.quality.replaceChildren();
    nodes.measureQuality.disabled = !currentVariant(comparePath);
    setPending(pending);
    renderContactSheet();
    renderPresets();
    void renderImages();
  }

  function renderContactSheet() {
    nodes.contact.replaceChildren();
    data.scenes.forEach((scene) => {
      const variant = scene.variants.find((item) => item.role === "final") || scene.variants[0];
      const card = el("article", "", "photo-review-contact-item");
      const control = button("", "photographic-contact-item", () => { sceneId = scene.scene_id; chooseVariants(); render(); });
      control.className = "photo-review-contact-select";
      control.setAttribute("aria-pressed", String(scene.scene_id === sceneId));
      const query = new URLSearchParams({ artifact_path: variant.artifact_path, sha256: variant.sha256, source_identity_sha256: identity(variant), max_edge: "256",
        ...(presentationAware(variant) ? { view: viewFor(variant) } : {}) });
      const url = photographicAssetUrl(`${photographicJobUrl(jobId, "/render")}?${query}`, jobId);
      if (url) {
        const key = imageKey(variant, false, 256);
        let image = thumbnailImages.get(key);
        if (!image) {
          image = el("img"); image.alt = ""; image.loading = "lazy"; thumbnailImages.set(key, image);
          if (state.auth?.mode === "direct_debug") {
            const target = image;
            const load = async () => {
              if (thumbnailImages.get(key) !== target) return;
              try {
                const source = await loadImage(variant, false, 256);
                if (!source) return;
                if (thumbnailImages.get(key) !== target) { URL.revokeObjectURL(source.url); return; }
                target.src = source.url;
              } catch (error) { if (thumbnailImages.get(key) === target && error.name !== "AbortError") status(error.message, true); }
            };
            thumbnailLoads.set(target, load);
            if (thumbnailObserver) thumbnailObserver.observe(target); else void load();
          } else image.src = url;
        }
        while (thumbnailImages.size > 64) {
          const oldest = thumbnailImages.keys().next().value;
          const removed = thumbnailImages.get(oldest);
          thumbnailObserver?.unobserve(removed);
          if (removed.src?.startsWith("blob:")) URL.revokeObjectURL(removed.src);
          thumbnailImages.delete(oldest);
        }
        control.append(image);
      }
      control.append(el("span", scene.label || scene.scene_id), el("small", decision(variant)?.status?.replaceAll("_", " ") || "Unreviewed"));
      card.append(control);
      const sceneLabel = input("photographic-scene-delivery-label", "text", sceneLabels.get(scene.scene_id) || scene.label || scene.scene_id);
      sceneLabel.maxLength = 80;
      sceneLabel.addEventListener("input", () => sceneLabels.set(scene.scene_id, sceneLabel.value));
      card.append(field("Delivery scene name", sceneLabel));
      const approved = scene.variants.filter((item) => item.role !== "original" && decision(item)?.status === "approved");
      if (approved.length) {
        if (!deliverySelections.has(scene.scene_id) || (deliverySelections.get(scene.scene_id) && !approved.some((item) => item.artifact_path === deliverySelections.get(scene.scene_id)))) {
          deliverySelections.set(scene.scene_id, (approved.find((item) => item.role === "final") || approved[0]).artifact_path);
        }
        const deliveryChoice = select("photographic-delivery-version", [["", "Exclude from delivery"], ...approved.map((item) => [item.artifact_path, photographicVariantLabel(item)])], () => {
          deliverySelections.set(scene.scene_id, deliveryChoice.value); updateDeliveryButton();
        });
        deliveryChoice.value = deliverySelections.get(scene.scene_id);
        card.append(field(`Approved delivery version · ${scene.label || scene.scene_id}`, deliveryChoice));
      } else deliverySelections.delete(scene.scene_id);
      nodes.contact.append(card);
    });
  }

  function renderPresets() {
    nodes.presets.replaceChildren();
    const names = new Set();
    const labels = { photographic_master: "Photographic master", website: "Website", mls: "MLS", social: "Social", print: "Print", retouch_print: "Retouch / print" };
    data.presets.forEach((preset) => {
      const id = typeof preset === "string" ? preset : preset.name || preset.id;
      if (!id) return;
      names.add(id);
      const checkbox = input("photographic-preset", "checkbox");
      checkbox.value = id;
      checkbox.checked = selectedPresets.has(id);
      checkbox.addEventListener("change", () => { checkbox.checked ? selectedPresets.add(id) : selectedPresets.delete(id); updateDeliveryButton(); });
      const group = el("fieldset", "", "photo-review-preset");
      group.append(field(typeof preset === "string" ? labels[id] || id : preset.label || labels[id] || id, checkbox));
      if (preset.description || preset.intended_use) group.append(el("p", preset.description || preset.intended_use, "photo-review-help"));
      if (typeof preset === "object" && !["photographic_master", "retouch_print"].includes(id)) {
        const edit = presetEdits.get(id) || {};
        const controls = el("div", "", "photo-review-controls");
        for (const [key, label, min, max, step] of [["long_edge", "Long edge (px)", 1, 16384, 1], ["jpeg_quality", "JPEG quality", 50, 100, 1], ["sharpen_amount", "Output sharpening", 0, 1, 0.05]]) {
          if (key === "jpeg_quality" && preset.format !== "jpeg") continue;
          const control = input(`photographic-preset-${key}`, "number", String(edit[key] ?? preset[key] ?? ""));
          control.min = String(min); control.max = String(max); control.step = String(step);
          if (key === "long_edge") control.placeholder = "Source dimensions";
          control.addEventListener("input", () => { presetEdits.set(id, { ...presetEdits.get(id), [key]: control.value }); });
          controls.append(field(label, control));
        }
        group.append(controls);
      }
      nodes.presets.append(group);
    });
    selectedPresets = new Set([...selectedPresets].filter((name) => names.has(name)));
    updateDeliveryButton();
  }

  function deliveryPresets() {
    return [...selectedPresets].map((name) => {
      if (!presetEdits.has(name)) return name;
      const result = { name };
      for (const [key, value] of Object.entries(presetEdits.get(name))) {
        if (key === "long_edge" && !String(value).trim()) { result[key] = null; continue; }
        const number = Number(value);
        const [min, max] = key === "long_edge" ? [1, 16384] : key === "jpeg_quality" ? [50, 100] : [0, 1];
        if (!Number.isFinite(number) || number < min || number > max || (key !== "sharpen_amount" && !Number.isInteger(number))) {
          throw new Error(`${name}: ${key.replaceAll("_", " ")} is outside the supported range.`);
        }
        result[key] = number;
      }
      return result;
    });
  }
  function approvedVariants() {
    return (data?.scenes || []).flatMap((scene) => scene.variants.filter((variant) => variant.role !== "original"
      && variant.artifact_path === deliverySelections.get(scene.scene_id) && decision(variant)?.status === "approved"));
  }
  function updateDeliveryButton() {
    const count = approvedVariants().length;
    nodes.deliver.disabled = pending || ["submitting", "outcome_unknown", "reconciling"].includes(deliveryOutcome) || !count || !selectedPresets.size;
    nodes.deliver.textContent = `Deliver approved set (${count})`;
  }

  function renderStage() {
    const primary = currentVariant();
    const comparison = currentVariant(comparePath);
    const aligned = primary && comparison && primary.width === comparison.width && primary.height === comparison.height;
    if (!aligned && mode !== "side_by_side") { mode = "side_by_side"; stopBlink(); }
    nodes.mode.value = mode;
    for (const option of nodes.mode.options) option.disabled = option.value !== "side_by_side" && !aligned;
    nodes.stage.dataset.mode = mode;
    nodes.stage.dataset.inspection = inspection;
    nodes.presentation.value = recordedPresentationEnabled() ? presentationView : "clip_srgb";
    nodes.presentation.disabled = !recordedPresentationEnabled();
    nodes.stage.dataset.presentation = viewFor(primary);
    nodes.figures[1].figure.classList.toggle("hidden", !comparison);
    nodes.figures[1].caption.classList.toggle("hidden", !comparison);
    nodes.wipeField.classList.toggle("hidden", mode !== "wipe");
    nodes.blink.classList.toggle("hidden", mode !== "blink");
    nodes.nativeControls.classList.toggle("hidden", inspection !== "native");
    nodes.x.value = String(Math.round(center.x * 100));
    nodes.y.value = String(Math.round(center.y * 100));
    nodes.nativeNotice.textContent = inspection === "native"
      ? "Native crops: 1 source pixel per CSS pixel after the server size and source-hash checks pass. Both views share the same normalized center; scroll and arrow controls pan together."
      : "Fit previews are reduced for the browser. Enlarging one does not show native detail; choose Native 1:1 crop to inspect source pixels.";
    nodes.nativeNotice.textContent += " Display pixels are an 8-bit encoded sRGB derivative. Native 1:1 describes geometry; master precision and highlights remain authoritative in the original file.";
    for (const [label, variant] of [["Selected", primary], ["Comparison", comparison]]) {
      if (!variant) continue;
      nodes.nativeNotice.textContent += viewFor(variant) === "recorded"
        ? ` ${label}: authoritative recorded presentation (${variant.presentation?.policy?.stage || "recorded"}); rendered deliveries receive no second tone transform.`
        : ` ${label}: explicit linear-to-sRGB clipping view${!recordedPresentationEnabled() ? "; recorded-presentation rollout is disabled" : presentationView === "recorded" ? "; recorded authority is unavailable, so this is a labelled clipping fallback" : ""}.`;
    }
    if (comparison && !aligned) nodes.nativeNotice.textContent += " Image dimensions differ; wipe and blink are unavailable because pixel alignment is unproven.";
    if (!currentScene()?.variants.some((variant) => variant.role === "original")) {
      nodes.nativeNotice.textContent += ` Original unavailable: ${currentScene()?.original_unavailable_reason || currentScene()?.unavailable_reason || "a verified retained input has not been bound to this run"}.`;
    }
  }

  async function loadImage(variant, native, edge = 1600) {
    const geometry = native ? nativeCropGeometry(variant, center, cropSize) : null;
    const thumbnail = edge === 256;
    const key = imageKey(variant, native, edge);
    if (!thumbnail && imageCache.has(key)) {
      const cached = imageCache.get(key);
      imageCache.delete(key); imageCache.set(key, cached);
      return cached;
    }
    if (!thumbnail && imageInFlight.has(key) && !imageControllers.get(key)?.signal.aborted) return imageInFlight.get(key);
    checkBackoff();
    const params = new URLSearchParams({ artifact_path: variant.artifact_path, sha256: variant.sha256, source_identity_sha256: identity(variant),
      ...(presentationAware(variant) ? { view: viewFor(variant) } : {}),
      ...(geometry ? Object.fromEntries(Object.entries(geometry).map(([name, value]) => [name, String(value)])) : { max_edge: String(edge) }) });
    const controller = new AbortController();
    const requestJob = jobId;
    const requestActor = actorScope;
    requests.add(controller);
    if (!thumbnail) imageControllers.set(key, controller);
    const timer = setTimeout(() => controller.abort(), 30000);
    const promise = (async () => {
      try {
        const response = await fetchImpl(`${photographicJobUrl(requestJob, native ? "/native-crop" : "/render")}?${params}`, {
          headers: _buildAuthHeaders({ Accept: "image/png" }, "GET"), credentials: "same-origin", signal: controller.signal,
        });
        checkRateLimit(response);
        if (!response.ok) throw new Error(`${native ? "Native crop" : "Fit preview"} unavailable (${response.status}); retained metadata remains available.`);
        if (!/^image\/png(?:;|$)/i.test(response.headers.get("Content-Type") || "")
          || response.headers.get("X-TP-Source-SHA256") !== variant.sha256
          || (variant.source_identity_sha256 && response.headers.get("X-TP-Source-Identity-SHA256") !== variant.source_identity_sha256)
          || (presentationAware(variant) && (response.headers.get("X-TP-Presentation-View") !== viewFor(variant)
          || response.headers.get("X-TP-Presentation-Policy-SHA256") !== (variant.presentation_policy_sha256 || "none")))
          || (native && (Number(response.headers.get("X-TP-Native-Width")) !== geometry.width
          || Number(response.headers.get("X-TP-Native-Height")) !== geometry.height))) throw new Error("Image source identity or native dimensions could not be verified.");
        const blob = await response.blob();
        if (requestJob !== jobId || requestActor !== actorScope || controller.signal.aborted) return null;
        if (thumbnail && blob.size > 1024 * 1024) throw new Error("Thumbnail exceeds its display limit.");
        const entry = { url: URL.createObjectURL(blob), geometry, bytes: blob.size };
        if (thumbnail) return entry;
        imageCache.set(key, entry);
        let bytes = [...imageCache.values()].reduce((total, image) => total + image.bytes, 0);
        while (imageCache.size > 12 || (bytes > 40 * 1024 * 1024 && imageCache.size > 1)) {
          const oldest = imageCache.keys().next().value;
          const image = imageCache.get(oldest);
          bytes -= image.bytes; URL.revokeObjectURL(image.url); imageCache.delete(oldest);
        }
        return entry;
      } finally {
        clearTimeout(timer); requests.delete(controller);
        if (imageControllers.get(key) === controller) imageControllers.delete(key);
        if (imageInFlight.get(key) === promise) imageInFlight.delete(key);
      }
    })();
    if (!thumbnail) imageInFlight.set(key, promise);
    return promise;
  }

  function imageMetadata(variant) {
    return `${photographicVariantLabel(variant)} · ${photographicProperties(variant)} · ${variant.artifact_path} · SHA-256 ${variant.sha256} · Source identity ${identity(variant)}${variant.identity_status ? ` · ${variant.identity_status}` : ""}`;
  }

  async function renderImages() {
    if (!data || !currentVariant()) return;
    renderStage();
    const variants = [currentVariant(), currentVariant(comparePath)];
    const keys = variants.filter(Boolean).map((variant) => imageKey(variant, inspection === "native"));
    const nextKey = JSON.stringify(keys);
    if (activeImageKey === nextKey) {
      variants.forEach((variant, index) => {
        if (variant) nodes.figures[index].caption.textContent = imageMetadata(variant) + (nodes.figures[index].cropCaption || "");
      });
      return;
    }
    activeImageKey = nextKey;
    displayedInspectionKey = "";
    stopBlink();
    const sequence = ++imageRequest;
    imageControllers.forEach((controller, key) => { if (!keys.includes(key)) controller.abort(); });
    if (imageTimer) clearTimeout(imageTimer);
    finishImageDelay?.();
    const delay = host.renderDebounceMs ?? 120;
    if (delay > 0) await new Promise((resolve) => { finishImageDelay = resolve; imageTimer = setTimeout(resolve, delay); });
    imageTimer = null; finishImageDelay = null;
    if (sequence !== imageRequest) return;
    await Promise.all(variants.map(async (variant, index) => {
      const figure = nodes.figures[index];
      figure.image.classList.add("hidden");
      figure.image.removeAttribute("src");
      if (!variant) { figure.caption.textContent = "No comparison selected."; return; }
      figure.cropCaption = "";
      figure.caption.textContent = imageMetadata(variant);
      figure.image.alt = `${photographicVariantLabel(variant)} ${inspection === "native" ? "native crop" : "reduced browser preview"}`;
      try {
        const source = await loadImage(variant, inspection === "native");
        if (sequence !== imageRequest || !source) return;
        if (!source.url) { figure.caption.textContent += " · Full-frame preview unavailable; use a native crop."; return; }
        figure.image.style.width = source.geometry ? `${source.geometry.width}px` : "";
        figure.image.style.height = source.geometry ? `${source.geometry.height}px` : "";
        figure.image.onload = () => {
          if (sequence !== imageRequest) return;
          if (source.geometry && (figure.image.naturalWidth !== source.geometry.width || figure.image.naturalHeight !== source.geometry.height)) {
            figure.image.classList.add("hidden");
            status("The decoded crop dimensions do not match native evidence; inspection is blocked.", true);
            return;
          }
          figure.image.classList.remove("hidden");
          if (index === 0) displayedInspectionKey = imageKey(variant, inspection === "native");
          if (source.geometry) { figure.cropCaption = ` · Crop x=${source.geometry.x}, y=${source.geometry.y}, ${source.geometry.width} × ${source.geometry.height}, native 1:1 verified`; figure.caption.textContent = imageMetadata(variant) + figure.cropCaption; }
        };
        figure.image.onerror = () => { if (sequence === imageRequest) status("Image display failed. Retained metadata and downloads remain available.", true); };
        figure.image.src = source.url;
        if (source.geometry) { figure.cropCaption = ` · Crop x=${source.geometry.x}, y=${source.geometry.y}, ${source.geometry.width} × ${source.geometry.height}, decoding native crop`; figure.caption.textContent = imageMetadata(variant) + figure.cropCaption; }
      } catch (error) { if (sequence === imageRequest) { activeImageKey = ""; status(error.message, true); } }
    }));
  }

  function pan(dx, dy) {
    const variant = currentVariant();
    if (!variant?.width || !variant?.height || inspection !== "native") return;
    center = { x: Math.max(0, Math.min(1, center.x + dx * cropSize / variant.width / 2)), y: Math.max(0, Math.min(1, center.y + dy * cropSize / variant.height / 2)) };
    void renderImages();
  }

  async function mutate(path, body, success, timeout = 30000) {
    if (pending || !data) return null;
    const expectedJob = jobId;
    setPending(true);
    try {
      const result = await request(path, { ...body, expected_version: data.state_version }, timeout);
      if (jobId !== expectedJob) return null;
      await load();
      status(success);
      return result;
    } catch (error) {
      if (jobId !== expectedJob) return null;
      if (error.code === "REVIEW_CONFLICT") { resetPreview(); await load(true); status("Review changed elsewhere. Current decisions were reloaded; inspect them before retrying.", true); }
      else status(error.name === "AbortError" ? "Review request timed out. Refresh before retrying; it may have completed on the server." : error.message, true);
      return null;
    } finally { if (jobId === expectedJob) { setPending(false); updateDeliveryButton(); } }
  }

  async function measureQuality() {
    const variant = currentVariant();
    const comparison = currentVariant(comparePath);
    if (!variant || !comparison) return;
    const expectedJob = jobId;
    nodes.measureQuality.disabled = true;
    nodes.quality.textContent = "Measuring advisory local color, halo, clipping, and banding signals…";
    const params = new URLSearchParams({ artifact_path: variant.artifact_path, sha256: variant.sha256, source_identity_sha256: identity(variant),
      reference_identity_sha256: identity(comparison), reference_path: comparison.artifact_path, reference_sha256: comparison.sha256 });
    try {
      const result = await request(`/quality?${params}`, null, 90000);
      if (jobId !== expectedJob || currentVariant() !== variant || currentVariant(comparePath) !== comparison) return;
      nodes.quality.replaceChildren(el("p", "Reference-relative diagnostics are advisory and do not establish photographic acceptance or grant approval."));
      nodes.quality.append(el("p", `Algorithm ${result.algorithm_revision || "not reported"} · ${result.scope || "scope not reported"}`));
      const table = el("table", "", "photo-review-metrics");
      const caption = el("caption", "Local image diagnostics for the selected image and explicit comparison");
      table.append(caption);
      Object.entries(result.metrics || {}).forEach(([key, value]) => {
        const row = el("tr");
        const label = el("th", key.replaceAll("_", " "));
        label.scope = "row";
        row.append(label, el("td", typeof value === "number" ? Number(value.toPrecision(6)).toString() : JSON.stringify(value)));
        table.append(row);
      });
      nodes.quality.append(table);
      status("Advisory quality signals measured. Human review and approval remain separate.");
    } catch (error) { if (jobId === expectedJob && currentVariant() === variant) nodes.quality.textContent = error.message; }
    finally { if (jobId === expectedJob) nodes.measureQuality.disabled = !currentVariant(comparePath); }
  }

  async function saveDecision(statusValue) {
    const variant = currentVariant();
    if (!variant || variant.role === "original") return;
    const inspected = presentationAware(variant) && displayedInspectionKey === imageKey(variant, inspection === "native");
    const result = await mutate("/decisions", { artifact_path: variant.artifact_path, sha256: variant.sha256, source_identity_sha256: identity(variant), status: statusValue, note: nodes.note.value,
      ...(inspected ? { inspection_presentation: { view: viewFor(variant), policy_sha256: variant.presentation_policy_sha256 || null } } : {}) }, "Decision saved for this exact artifact hash; presentation evidence does not change its approval scope.");
    if (result && statusValue === "approved") { deliverySelections.set(sceneId, variant.artifact_path); renderContactSheet(); updateDeliveryButton(); }
  }
  async function saveLightingGroup() {
    const variant = currentVariant();
    const name = nodes.groupName.value.trim();
    const reference = nodes.reference.value;
    if (!variant || !name || !reference) { status("Choose a lighting group name and a reference photograph.", true); return; }
    const groups = data.groups.map((group) => ({ ...group, members: group.members.filter((sha) => sha !== identity(variant)) })).filter((group) => group.members.length);
    const group = groups.find((item) => item.name === name);
    if (group) { group.members = [...new Set([...group.members, identity(variant), reference])]; group.reference_sha256 = reference; }
    else groups.push({ name, reference_sha256: reference, members: [identity(variant), ...(reference !== identity(variant) ? [reference] : [])] });
    await mutate("/lighting-groups", { groups }, "Lighting reference saved. Intentional warmth and twilight remain operator decisions.");
  }
  async function revise(previewOnly) {
    const variant = currentVariant();
    if (!variant) return;
    let recipe;
    try { recipe = photographicRecipe({ ...revisionDraft, materials: materialDrafts }); } catch (error) { status(error.message, true); return; }
    const key = JSON.stringify([variant.artifact_path, variant.sha256, identity(variant), variant.presentation_policy_sha256 || null, recipe, nodes.reference.value]);
    const expectedJob = jobId;
    if (!previewOnly && variantPreview?.key !== key) { status("Render and inspect the current adjustments before creating a revision.", true); return; }
    if (previewOnly) {
      if (pending) return;
      setPending(true);
      try {
        const result = await request("/revisions", { artifact_path: variant.artifact_path, sha256: variant.sha256, source_identity_sha256: identity(variant), scene_id: sceneId, recipe,
          ...(presentationAware(variant) ? { presentation_policy_sha256: variant.presentation_policy_sha256 || null } : {}),
          reference_sha256: nodes.reference.value || null, expected_version: data.state_version, preview_only: true }, 90000);
        if (currentVariant()?.artifact_path !== variant.artifact_path || key !== JSON.stringify([variant.artifact_path, currentVariant().sha256, identity(currentVariant()), currentVariant().presentation_policy_sha256 || null, photographicRecipe({ ...revisionDraft, materials: materialDrafts }), nodes.reference.value])) return;
        if (!/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(result.preview_data_url || "")
          || !photographicPreviewMatches(result.receipt, variant, recipe, nodes.reference.value || null)) throw new Error("Rendered variant preview identity or recipe could not be verified.");
        variantPreview = { key, result };
        nodes.variantPreview.src = result.preview_data_url;
        nodes.variantPreview.classList.remove("hidden");
        nodes.revisionReceipt.textContent = `Rendered preview from master ${variant.sha256}. Reduced preview only; approval is required separately after the full revision is created.`;
        status("Rendered variant ready for visual inspection. Create revision preserves the existing image.");
      } catch (error) { if (jobId === expectedJob) { resetPreview(); status(error.message, true); } }
      finally { if (jobId === expectedJob) { setPending(false); nodes.commitRevision.disabled = !variantPreview; updateDeliveryButton(); } }
    } else {
      const result = await mutate("/revisions", { artifact_path: variant.artifact_path, sha256: variant.sha256, source_identity_sha256: identity(variant), scene_id: sceneId, recipe,
        ...(presentationAware(variant) ? { presentation_policy_sha256: variant.presentation_policy_sha256 || null } : {}),
        reference_sha256: nodes.reference.value || null, preview_only: false }, "Revision created separately and remains unapproved.", 180000);
      if (result) { chooseVariants(); render(); resetPreview(); }
    }
  }
  function clearDownload() {
    downloadController?.abort(); downloadController = null;
    if (downloadRevoke) clearTimeout(downloadRevoke);
    downloadRevoke = null;
    if (downloadObjectUrl) URL.revokeObjectURL(downloadObjectUrl);
    downloadObjectUrl = "";
  }
  function downloadScope() { return JSON.stringify([jobId, actorScope, deliverySources]); }
  function bindDownload(link, url, filename, sha = "") {
    link.href = url; link.download = filename;
    if (state.auth?.mode !== "direct_debug") return;
    const scope = downloadScope();
    link.addEventListener("click", async (event) => {
      event.preventDefault();
      clearDownload();
      const controller = new AbortController();
      downloadController = controller; requests.add(controller);
      const timer = setTimeout(() => controller.abort(), 30 * 60 * 1000);
      let writable = null;
      try {
        checkBackoff();
        if (scope !== downloadScope() || !photographicAssetUrl(url, jobId)) throw new Error("Delivery selection changed. Create a current verified bundle.");
        const handle = globalThis.showSaveFilePicker ? await globalThis.showSaveFilePicker({ suggestedName: filename }) : null;
        if (scope !== downloadScope() || controller.signal.aborted) return;
        status(`Downloading ${filename} through authenticated access…`);
        const response = await fetchImpl(url, { method: "GET", credentials: "same-origin", cache: "no-store", signal: controller.signal,
          headers: _buildAuthHeaders({ Accept: "application/octet-stream" }, "GET") });
        checkRateLimit(response);
        const returnedSha = response.headers.get("X-TP-Artifact-SHA256");
        if (!response.ok || !SHA256.test(returnedSha || "") || (sha && returnedSha !== sha)) throw new Error("Verified download unavailable. Refresh review and check current approval.");
        if (!response.body) throw new Error("Download stream is unavailable.");
        if (handle) {
          writable = await handle.createWritable();
          await response.body.pipeTo(writable, { signal: controller.signal, preventClose: true });
          if (scope !== downloadScope() || controller.signal.aborted) throw new Error("Delivery changed during download; the file was not committed.");
          await writable.close(); writable = null;
          status(`Saved ${filename}.`);
        } else {
          const limit = 64 * 1024 * 1024;
          const length = Number(response.headers.get("Content-Length"));
          if (!Number.isSafeInteger(length) || length < 1 || length > limit) throw new Error("Direct debug fallback requires a known file size up to 64 MiB. Use Chrome's save picker or the managed front door for this download.");
          const reader = response.body.getReader();
          const chunks = [];
          let size = 0;
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            size += value.byteLength;
            if (size > length || size > limit) throw new Error("Download exceeded its bounded size; use the managed front door.");
            chunks.push(value);
          }
          if (size !== length || scope !== downloadScope() || controller.signal.aborted) throw new Error("Delivery changed or its download was incomplete.");
          downloadObjectUrl = URL.createObjectURL(new Blob(chunks, { type: response.headers.get("Content-Type") || "application/octet-stream" }));
          const anchor = doc.createElement("a"); anchor.href = downloadObjectUrl; anchor.download = filename;
          doc.body.append(anchor); anchor.click(); anchor.remove();
          downloadRevoke = setTimeout(() => { if (downloadObjectUrl === anchor.href) { URL.revokeObjectURL(downloadObjectUrl); downloadObjectUrl = ""; } }, 1000);
          status(`Download started for ${filename}.`);
        }
      } catch (error) {
        controller.abort();
        if (writable) await writable.abort().catch(() => {});
        if (downloadController === controller) {
          clearDownload();
          if (scope === downloadScope()) status(error.name === "AbortError" ? "Download canceled." : error.message, true);
        }
      } finally {
        clearTimeout(timer); requests.delete(controller);
        if (downloadController === controller) downloadController = null;
      }
    });
  }
  function deliveryRecoveryScope() { return JSON.stringify([jobId, actorScope]); }

  function saveDeliveryAttempt(attempt) {
    const saved = recoveryStorage.set(deliveryRecoveryScope(), attempt);
    if (saved || attempt === null) deliveryAttempt = attempt;
    return saved;
  }

  function renderDeliveryHistory() {
    nodes.deliveryHistory.replaceChildren();
    if (deliveryHistory.length) panel.classList.remove("hidden");
    for (const entry of deliveryHistory) {
      const row = el("article", "", "photo-review-help", "photographic-delivery-history-entry");
      row.dataset.deliveryId = String(entry.id || "");
      row.append(el("p", `${entry.intent?.property_name || "Delivery"} · ${entry.created_at ? new Date(entry.created_at * 1000).toLocaleString() : "Time not recorded"}`),
        el("p", `Approval at creation: ${entry.approval_at_creation || "unknown"}. Current recorded approval: ${entry.current_recorded_approval || "unknown"}. Source and file bytes have not been checked by history; downloads revalidate access, source identity, approval, and bytes.`));
      if (entry.current_recorded_approval === "approved") {
        for (const file of entry.files || []) {
          const url = photographicAssetUrl(file.url, jobId);
          if (!url || !SHA256.test(file.sha256 || "")) continue;
          const link = el("a", `Download ${file.filename}`, "operator-action-btn operator-action-btn-secondary", "photographic-delivery-history-download");
          bindDownload(link, url, file.filename, file.sha256);
          row.append(link);
        }
      }
      nodes.deliveryHistory.append(row);
    }
    nodes.moreDeliveries.disabled = !deliveryCursor;
  }

  async function loadDeliveryHistory(more = false) {
    if (!jobId) return;
    if (!deliveryHistoryEnabled()) {
      if (deliveryAttempt) status("Delivery history UI is disabled. An earlier export remains unknown; re-enable history or reconcile through the authenticated API before another export.", true);
      return;
    }
    const sequence = ++historyRequest;
    const expectedScope = deliveryRecoveryScope();
    if (deliveryAttempt) deliveryOutcome = "reconciling";
    updateDeliveryButton();
    nodes.deliveryHistoryStatus.textContent = "Loading committed delivery metadata…";
    try {
      const params = new URLSearchParams({ limit: "20" });
      if (more && deliveryCursor) params.set("cursor", deliveryCursor);
      const page = await request(`/deliveries?${params}`);
      if (sequence !== historyRequest || expectedScope !== deliveryRecoveryScope()) return;
      if (!Array.isArray(page.items)) throw new Error("Delivery history response is unavailable.");
      deliveryHistory = more ? [...deliveryHistory, ...page.items] : page.items;
      deliveryCursor = page.next_cursor || null;
      renderDeliveryHistory();
      nodes.deliveryHistoryStatus.textContent = deliveryHistory.length ? "Committed history loaded. Approval observations are current; file availability is checked on download." : "No committed deliveries in this snapshot.";
      if (deliveryAttempt) {
        const attempt = deliveryAttempt;
        const recovered = await request(`/deliveries?${new URLSearchParams({ limit: "64", request_correlation_id: attempt.body.request_correlation_id })}`);
        if (sequence !== historyRequest || expectedScope !== deliveryRecoveryScope() || attempt !== deliveryAttempt) return;
        const matches = Array.isArray(recovered.items) ? recovered.items : [];
        if (matches.length === 1 && photographicDeliveryRecovered(matches[0], attempt)) {
          deliveryOutcome = "committed";
          saveDeliveryAttempt(null);
          if (!deliveryHistory.some((entry) => entry.id === matches[0].id)) deliveryHistory.unshift(matches[0]);
          renderDeliveryHistory();
          nodes.deliveryHistoryStatus.textContent = "Recovered the committed delivery for the original request. No export was retried.";
        } else if (!matches.length && recovered.next_cursor === null && Number.isInteger(recovered.state_version)
          && recovered.state_version > attempt.body.expected_version) {
          deliveryOutcome = "idle";
          saveDeliveryAttempt(null);
          nodes.deliveryHistoryStatus.textContent = "No matching delivery committed at the original review version, which has now advanced. Inspect current review before explicitly creating a new export.";
        } else {
          deliveryOutcome = "outcome_unknown";
          nodes.deliveryHistoryStatus.textContent = matches.length ? "Delivery correlation is ambiguous or differs from the frozen request. Export remains blocked; review the saved history." : "Export outcome is still unknown. Reconcile again; no automatic export retry will occur.";
        }
      }
    } catch (error) {
      if (sequence !== historyRequest || expectedScope !== deliveryRecoveryScope()) return;
      if (deliveryAttempt) deliveryOutcome = "outcome_unknown";
      nodes.deliveryHistoryStatus.textContent = `Delivery history unavailable: ${error.message}${deliveryAttempt ? " Original export outcome remains unknown; sign in or reconcile again before another export." : ""}`;
    } finally { if (sequence === historyRequest && expectedScope === deliveryRecoveryScope()) updateDeliveryButton(); }
  }

  function renderCommittedDelivery(result, sources) {
    deliverySources = sources.map((source) => ({ ...source, status: "approved" }));
    const url = photographicAssetUrl(result.download_url, jobId);
    if (!url || !SHA256.test(result.sha256 || "")) throw new Error("Delivery response did not contain a verified job-bound download.");
    const link = el("a", `Download ${result.filename || "verified delivery ZIP"}`, "operator-action-btn operator-action-btn-primary", "photographic-delivery-download");
    bindDownload(link, url, result.filename || "delivery.zip", result.sha256);
    nodes.deliveryResult.append(link, el("p", `SHA-256 ${result.sha256} · ${result.size_bytes} bytes. Inventory and source-master links are retained inside the bundle.`));
    for (const file of result.files || []) {
      const fileUrl = photographicAssetUrl(file.url, jobId);
      if (!fileUrl || file.filename === result.filename) continue;
      const companion = el("a", file.filename, "operator-action-btn operator-action-btn-secondary", "photographic-delivery-companion");
      bindDownload(companion, fileUrl, file.filename);
      nodes.deliveryResult.append(companion);
    }
  }

  async function deliver() {
    if (pending || !data || ["submitting", "outcome_unknown", "reconciling"].includes(deliveryOutcome)) return;
    clearDownload();
    nodes.deliveryResult.replaceChildren();
    let presets;
    try { presets = deliveryPresets(); } catch (error) { status(error.message, true); return; }
    const sources = approvedVariants().map((variant) => ({ artifact_path: variant.artifact_path, sha256: variant.sha256, source_identity_sha256: identity(variant) }));
    const sourceHashes = [...new Set(sources.map((variant) => variant.sha256))];
    const scene_labels = Object.fromEntries(data.scenes.filter((scene) => scene.variants.some((variant) => sources.some((source) => source.artifact_path === variant.artifact_path)))
      .map((scene) => [scene.scene_id, sceneLabels.get(scene.scene_id) || scene.label || scene.scene_id]));
    if (!deliveryHistoryEnabled()) {
      const result = await mutate("/delivery", { property_name: nodes.property.value.trim(), presets,
        approved_sources: sources, approved_sha256: sourceHashes, scene_labels }, "Approved-set delivery verified and ready to download.", 300000);
      if (result) {
        try { renderCommittedDelivery(result, sources); if (createToast) createToast("Approved delivery is ready.", "success"); }
        catch (error) { status(error.message, true); }
      }
      return;
    }
    const correlation = globalThis.crypto?.randomUUID?.();
    if (!correlation) { status("A secure request correlation ID is unavailable; reload in a supported browser.", true); return; }
    const body = { property_name: nodes.property.value.trim(), presets, approved_sources: sources, approved_sha256: sourceHashes, scene_labels,
      expected_version: data.state_version, request_correlation_id: correlation };
    const attempt = photographicDeliveryAttempt(body, approvedVariants(), data.presets);
    const expectedScope = deliveryRecoveryScope();
    const expectedGeneration = scopeGeneration;
    if (!saveDeliveryAttempt(attempt)) {
      deliveryOutcome = "idle";
      status("Session recovery storage is unavailable or full. No export was submitted. Enable session storage or resolve retained attempts, then retry explicitly.", true);
      return;
    }
    deliveryOutcome = "submitting";
    setPending(true); updateDeliveryButton();
    try {
      const result = await request("/delivery", attempt.body, 300000);
      if (expectedScope !== deliveryRecoveryScope() || expectedGeneration !== scopeGeneration || deliveryAttempt !== attempt) return;
      renderCommittedDelivery(result, sources);
      deliveryOutcome = "committed"; saveDeliveryAttempt(null);
      if (Number.isInteger(result.state_version)) data.state_version = result.state_version;
      status("Approved-set delivery verified and ready to download.");
      if (createToast) createToast("Approved delivery is ready.", "success");
      void load(); void loadDeliveryHistory();
    } catch (error) {
      if (expectedScope !== deliveryRecoveryScope() || expectedGeneration !== scopeGeneration || deliveryAttempt !== attempt) return;
      if ([400, 422].includes(error.httpStatus)) { deliveryOutcome = "idle"; saveDeliveryAttempt(null); status(error.message, true); }
      else {
        deliveryOutcome = "outcome_unknown";
        status("Export outcome is unknown. Reconciling committed history; the original request will not be retried.", true);
        await loadDeliveryHistory();
      }
    } finally { if (expectedScope === deliveryRecoveryScope() && expectedGeneration === scopeGeneration) { setPending(false); updateDeliveryButton(); } }
  }

  function show(job, artifact) {
    if (!els.artifactShellContent || !doc) return;
    mount();
    nodes.deliveryHistoryControls.classList.toggle("hidden", !deliveryHistoryEnabled());
    const nextId = String(job?.id || "");
    const nextActorScope = JSON.stringify([state.auth?.mode, state.auth?.actor, state.auth?.tenant]);
    if (state.currentView && state.currentView !== "review") stopBlink();
    const photographic = /lux[-_]depth|photograph/i.test(String(job?.pipeline || job?.pipeline_name || ""))
      || (job?.artifacts || []).some((item) => /(^|\/)delivery\.tiff?$/.test(item.relative_path || item.path || ""));
    if (!nextId || !photographic) {
      requests.forEach((controller) => controller.abort()); requests.clear();
      stopBlink(); releaseImages(); imageRequest += 1;
      jobId = ""; inventoryKey = ""; data = null;
      historyRequest += 1; deliveryAttempt = null; deliveryOutcome = "idle"; deliveryHistory = []; deliveryCursor = null;
      scopeGeneration += 1;
      panel.classList.add("hidden"); return;
    }
    const key = JSON.stringify([nextActorScope, nextId, deliveryHistoryEnabled(), (job.artifacts || []).map((item) => [item.path, item.sha256])]);
    if (key === inventoryKey) return;
    inventoryKey = key;
    if (nextId !== jobId || nextActorScope !== actorScope) {
      scopeGeneration += 1;
      requests.forEach((controller) => controller.abort()); requests.clear();
      stopBlink(); releaseImages(); imageRequest += 1;
      jobId = nextId; actorScope = nextActorScope; pending = false; presetEdits.clear(); deliverySelections.clear(); sceneLabels.clear(); deliverySources = []; retryAt = 0; data = null; primaryPath = ""; comparePath = ""; sceneId = "";
      selectedPresets = new Set(); resetPreview();
      historyRequest += 1; deliveryHistory = []; deliveryCursor = null;
      deliveryAttempt = recoveryStorage.get(deliveryRecoveryScope());
      deliveryOutcome = deliveryAttempt ? "outcome_unknown" : "idle";
      nodes.deliveryResult.replaceChildren(); renderDeliveryHistory();
    }
    panel.classList.remove("hidden");
    void load();
    void loadDeliveryHistory();
  }
  doc?.addEventListener?.("visibilitychange", () => { if (doc.hidden) stopBlink(); });
  globalThis.addEventListener?.("pagehide", () => { requests.forEach((controller) => controller.abort()); releaseImages(); });
  return { show, refresh: () => load(true) };
}
