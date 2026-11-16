// Authoring identity and edit generations are separate from image inspection.
const copy = (value) => JSON.parse(JSON.stringify(value));
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);

export function photographicDraftSourceKey(owner, job, scene, variant) {
  return variant ? JSON.stringify([owner, job, scene, variant.artifact_path, variant.sha256,
    variant.source_identity_sha256 || variant.sha256]) : "";
}

export function createPhotographicDraftState() {
  let identity = "";
  const baselines = new Map();
  const generations = new Map();
  function bind(next) {
    if (identity === next) return false;
    identity = next; baselines.clear(); generations.clear(); return true;
  }
  function dirty(form, value) { return baselines.has(form) && !same(baselines.get(form), value); }
  return {
    bind,
    key: () => identity,
    dirty,
    edit(form) { generations.set(form, (generations.get(form) || 0) + 1); },
    hydrate(form, saved, current) {
      const retain = dirty(form, current);
      baselines.set(form, copy(saved));
      return copy(retain ? current : saved);
    },
    capture(form, value) { return { identity, form, generation: generations.get(form) || 0, value: copy(value) }; },
    acknowledge(token) {
      if (!token || token.identity !== identity) return false;
      baselines.set(token.form, copy(token.value));
      return token.generation === (generations.get(token.form) || 0);
    },
    clear() { identity = ""; baselines.clear(); generations.clear(); },
  };
}

export function createPhotographicDraftLifecycle(host) {
  let pendingNavigation = null, discarding = false, frozen = false, reauthorizing = false, saveState = "idle", notice = null;
  const nodes = host.nodes;
  function clearNavigation() {
    pendingNavigation = null;
    if (nodes.discardNavigation) nodes.discardNavigation.hidden = nodes.cancelNavigation.hidden = true;
  }
  function render() {
    if (!nodes.draftStatus) return;
    const dirty = host.hasEdits();
    nodes.draftStatus.dataset.state = frozen ? "frozen" : ["saving", "failed", "unknown"].includes(saveState) ? saveState : dirty ? "unsaved" : "saved";
    const text = frozen ? "Review paused. Unsaved changes remain in this tab while access is checked."
      : saveState === "saving" ? "Saving the submitted changes. Newer edits are retained."
      : saveState === "failed" ? "Save failed. Your unsaved changes are retained."
      : saveState === "unknown" ? "Save outcome unknown. Refresh to reconcile before trying again."
      : dirty ? "Unsaved changes" : "Saved";
    if (nodes.draftStatus.textContent !== text) nodes.draftStatus.textContent = text;
    host.onChange?.({ dirty, frozen });
  }
  function retain(action, options = {}) {
    if (discarding || !host.hasEdits()) return false;
    pendingNavigation = { action, leave: options.leave === true };
    nodes.discardNavigation.hidden = nodes.cancelNavigation.hidden = false;
    host.status("Unsaved edits are retained. Save or explicitly discard them to continue.", true);
    return true;
  }
  async function navigate(action, options = {}) {
    if (frozen && options.leave === true) {
      if (!host.hasEdits()) { purge(); return action(); }
      pendingNavigation = { action, leave: true };
      mask("Unsaved changes remain private. Discard them to leave, or keep waiting for access to recover."); return false;
    }
    if (host.blocked() || frozen) { host.status("Review is busy. Wait for the current operation before navigating.", true); return false; }
    return retain(action, options) ? false : action();
  }
  function removeNotice() { if (notice) { notice.hidden = true; notice.remove(); notice = null; } }
  function purge() { frozen = reauthorizing = false; saveState = "idle"; clearNavigation(); removeNotice(); host.purge(); }
  function mask(message) {
    host.hide();
    if (!notice) {
      notice = host.el("div", "", "photo-review-help", "photographic-authorization-status");
      notice.setAttribute("role", "status");
      notice.append(host.el("p"), host.button("Discard retained draft", "photographic-discard-frozen-draft", () => {
        const next = pendingNavigation;
        if (!next?.leave && host.authorization().status !== "ready") return;
        // A source retry closes over job/pager state that purge removes.
        // Restore its authorized context through show() instead of invoking it.
        purge(); if (next?.leave) void next.action(); else host.restoreContext();
      }), host.button("Keep waiting", "photographic-cancel-frozen-navigation", () => {
        clearNavigation(); mask("Review paused while sign-in is checked. Unsaved changes remain private in this tab.");
      }));
      host.parent()?.append(notice);
    }
    notice.children[0].textContent = message; notice.hidden = false;
  }
  function pause() {
    if (!frozen && host.scope().jobId) {
      frozen = true; reauthorizing = false;
      if (host.blocked()) saveState = "unknown";
      host.pause();
    }
    if (host.scope().jobId) mask("Review paused while sign-in is checked. Unsaved changes remain private in this tab.");
  }
  function syncAuthorization() {
    const current = host.authorization(), scope = host.scope();
    if (current.status === "revoked") { purge(); return false; }
    if (current.status !== "ready" || !current.ownerKey) { pause(); return false; }
    if (scope.owner && current.ownerKey !== scope.owner) purge();
    if (!frozen) return true;
    if (!reauthorizing) {
      const expected = host.scope();
      reauthorizing = true;
      void host.reload().then((loaded) => {
        const auth = host.authorization();
        if (expected.generation !== host.scope().generation || auth.status !== "ready" || auth.ownerKey !== expected.owner) return;
        if (!loaded) { mask("Review access or source changed. Your private draft remains withheld; refresh access or discard it to continue."); return; }
        frozen = false; removeNotice(); host.resume();
      }).finally(() => { if (expected.generation === host.scope().generation) reauthorizing = false; });
    }
    return false;
  }
  function mount() {
    nodes.draftStatus = host.el("p", "Saved", "photo-review-help", "photographic-draft-status");
    nodes.draftStatus.setAttribute("role", "status"); nodes.draftStatus.setAttribute("aria-live", "polite");
    nodes.discardNavigation = host.button("Discard unsaved edits and continue", "photographic-discard-navigation", async () => {
      if (host.blocked() || frozen || !pendingNavigation) return;
      const action = pendingNavigation.action;
      discarding = true;
      try { const result = await action(); if (result !== false) { host.discard?.(); clearNavigation(); } }
      finally { discarding = false; render(); }
    });
    nodes.cancelNavigation = host.button("Keep editing", "photographic-cancel-navigation", () => {
      clearNavigation(); host.status("Navigation canceled. Your unsaved edits are retained.");
    });
    clearNavigation();
  }
  return { mount, render, retain, navigate, clearNavigation, syncAuthorization, pause, revoke: purge,
    frozen: () => frozen, reauthorizing: () => reauthorizing, discarding: () => discarding,
    requiresReconciliation: () => saveState === "unknown",
    reconciled() { if (saveState === "unknown") { saveState = "idle"; render(); } },
    save(value) { saveState = value; render(); } };
}

export function photographicReviewAttribution(recorded) {
  const actor = typeof recorded?.actor === "object" ? recorded.actor?.display_name || recorded.actor?.username || recorded.actor?.id : recorded?.actor;
  const date = recorded?.created_at ? new Date(typeof recorded.created_at === "number" ? recorded.created_at * 1000 : recorded.created_at).toLocaleString() : "Date not recorded";
  return `${actor || "Reviewer not recorded"} · ${date}`;
}

export function photographicPageReview(loaded, page) {
  const paths = new Set((loaded.scenes || []).flatMap((scene) => scene.variants || []).map((variant) => variant.artifact_path));
  const recordedScenes = new Map();
  for (const row of page.decisions || []) {
    if (!row.scene_id || row.status !== "approved" || !/^[a-f0-9]{64}$/.test(row.sha256 || "")) continue;
    if (!recordedScenes.has(row.scene_id)) recordedScenes.set(row.scene_id, { scene_id: row.scene_id, label: row.label || row.scene_id, variants: [] });
    recordedScenes.get(row.scene_id).variants.push({ ...row, role: row.role || "previous_revision" });
  }
  const scenes = (loaded.scenes || []).map((scene) => ({ ...scene,
    railPaths: scene.variants.filter((variant) => ["final", "previous_revision"].includes(variant.role)).map((variant) => variant.artifact_path) }));
  return { review: { ...loaded, scenes, decisions: [...(page.decisions || []).filter((row) => !paths.has(row.artifact_path)), ...(loaded.decisions || [])] }, recordedScenes };
}

// Persist selectors and view settings only. The server still authorizes every
// detail, image and mutation; neither bytes nor credentials enter this store.

export function normalizedPhotographicReview(data) {
  const scenes = Array.isArray(data?.scenes) ? data.scenes.map((scene) => ({
    ...scene,
    scene_id: String(scene.scene_id || ""),
    variants: (Array.isArray(scene.variants) ? scene.variants : []).filter((variant) =>
      variant && typeof variant.artifact_path === "string" && /^[a-f0-9]{64}$/.test(variant.sha256 || "")),
  })).filter((scene) => scene.scene_id && scene.variants.length) : [];
  return { ...data, scenes, decisions: Array.isArray(data?.decisions) ? data.decisions : [],
    groups: Array.isArray(data?.groups) ? data.groups : [], presets: Array.isArray(data?.presets) ? data.presets : [] };
}

export function hydratePhotographicDraftForms({ nodes, scene, variant, savedDecision, groups, referenceScenes,
  paged, drafts, currentLighting, revision, identity, variantLabel, replaceOptions, cohesionData }) {
    nodes.decisionStatus.textContent = variant?.role === "original"
      ? "Original inspection only. Select a final or revision to record an approval."
      : savedDecision
      ? `${savedDecision.status === "needs_revision" ? "Needs revision" : savedDecision.status[0].toUpperCase() + savedDecision.status.slice(1)} · ${photographicReviewAttribution(savedDecision)} · exact SHA-256 ${variant.sha256}`
      : `Unreviewed · exact SHA-256 ${variant.sha256}`;
    nodes.decisions.querySelectorAll("[data-decision]").forEach((control) => control.setAttribute("aria-pressed", String(control.dataset.decision === savedDecision?.status)));
    nodes.note.value = drafts.hydrate("note", savedDecision?.note || "", nodes.note.value);
    const group = groups.find((item) => item.members?.includes(identity(variant)))
      || groups.find((item) => item.name === scene.lighting_group);
    const lighting = drafts.hydrate("lighting", { name: group?.name || scene.lighting_group || "", reference: group?.reference_sha256 || scene.reference_sha256 || "" }, currentLighting);
    nodes.groupName.value = lighting.name;
    const referenceVariants = referenceScenes.flatMap((item) => item.variants.filter((variant) => variant.role !== "original").map((variant) => ({ scene: item, variant })));
    if (paged && (group?.reference_sha256 || scene.reference_sha256) && !referenceVariants.some(({ variant }) => identity(variant) === (group?.reference_sha256 || scene.reference_sha256))) {
      referenceVariants.push({ scene: { label: "Recorded lighting reference" }, variant: { source_identity_sha256: group?.reference_sha256 || scene.reference_sha256, label: "Open its photograph to inspect", role: "reference" } });
    }
    const references = [["", "No reference selected"], ...referenceVariants.map(({ scene: item, variant }) => [identity(variant), `${item.label || item.scene_id} · ${variantLabel(variant)}`])];
    if (lighting.reference && !references.some(([key]) => key === lighting.reference)) references.push([lighting.reference, "Unsaved reference · refresh before saving"]);
    replaceOptions(nodes.reference, references, lighting.reference);
    drafts.hydrate("revision", { mode: "legacy", adjustments: { exposure: 0, warmth: 0, saturation: 1 }, regions: [] }, revision);
    const cohesion = (cohesionData?.groups || []).flatMap((group) => group.images || []).find((row) => (row.source_identity_sha256 || row.source_master_sha256) === identity(variant));
    nodes.outliers.textContent = (scene.quality_flags || scene.outlier_flags || cohesion?.flags || []).map((flag) => typeof flag === "string" ? flag : flag.message || flag.code || "").filter(Boolean).join(" · ")
      || "No automatic quality flags reported. Review sky, fine edges, foliage, texture, and intentional lighting manually.";
    if (cohesion) nodes.outliers.textContent += ` Reference-relative exposure ${Number(cohesion.exposure_delta_ev).toFixed(2)} EV; chromaticity difference ${Number(cohesion.chromaticity_delta).toFixed(3)}. Composition and intentional warmth or twilight can explain these advisory signals.`;
}

export function createPhotographicReviewElements(doc) {
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
  return { el, button, field, select, input };
}
