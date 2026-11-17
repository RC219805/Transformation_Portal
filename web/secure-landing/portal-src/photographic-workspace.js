export function mountPhotographicWorkspace(host) {
  const { panel, nodes, el, heading, controls, comparisonTools, quality, revisions, delivery, pagerControls } = host;
  const rail = el("aside", "", "photo-workspace-rail", "photographic-filmstrip");
  rail.setAttribute("aria-label", "Photographs in this catalogue page");
  rail.append(el("h4", "Photographs", "photo-review-subtitle"), pagerControls, nodes.contact);
  const evidence = el("details", "", "photo-review-artifact-summary", "photographic-image-evidence");
  evidence.append(el("summary", "Image evidence and source properties"), nodes.metadata, nodes.outliers);
  const canvas = el("section", "", "photo-workspace-canvas", "photographic-canvas");
  canvas.setAttribute("aria-label", "Photographic inspection");
  canvas.append(controls, nodes.nativeControls, nodes.stage, comparisonTools, nodes.nativeNotice, evidence);
  const inspector = el("aside", "", "photo-workspace-inspector", "photographic-inspector");
  inspector.setAttribute("aria-label", "Review decision and revision tools");
  const coherence = el("details", "", "photo-review-artifact-summary", "photographic-coherence-details");
  coherence.append(el("summary", "Lighting reference and quality signals"), nodes.groups, quality);
  inspector.append(el("h4", "Review decision", "photo-review-subtitle"), nodes.decisions,
    nodes.draftStatus, nodes.discardNavigation, nodes.cancelNavigation, coherence, revisions);
  const workspace = el("div", "", "photo-workspace", "photographic-workspace");
  workspace.append(rail, canvas, inspector);
  panel.append(heading, workspace, delivery, nodes.status);
}

export function photographicDeliverySummary({ total, selected, approved, products, property }) {
  const excluded = Math.max(0, total - approved);
  return `${approved} approved source${approved === 1 ? "" : "s"} of ${total}; ${excluded} excluded or not currently approved. `
    + `${selected} selected for revalidation. Products: ${products.length ? products.join(", ") : "choose at least one"}. `
    + `Delivery set: ${property || "name this property"}. Destination: verified ZIP download with contact sheet and inventory. `
    + "Every selected source is opened and revalidated before export.";
}

export function buildPhotographicDeliveryControls({ nodes, el, input, field, button, deliver, loadDeliveryHistory }) {
  const section = el("section", "", "photo-review-section", "photographic-delivery");
  section.append(el("h4", "Deliver approved set", "photo-review-subtitle"),
    el("p", "Choose destination requirements. The bundle includes meaningful filenames, a contact sheet, inventory, and verified source-master links. Approval is checked again at export.", "photo-review-help"));
  nodes.property = input("photographic-property-name", "text", "Property");
  nodes.property.maxLength = 100;
  nodes.presets = el("div", "", "photo-review-controls", "photographic-delivery-presets");
  nodes.deliverySelectionSummary = el("p", "", "photo-review-help", "photographic-delivery-selection-summary");
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
  section.append(field("Property name", nodes.property), nodes.presets, nodes.deliverySelectionSummary, nodes.deliver, nodes.deliveryResult, nodes.deliveryHistoryControls);
  return section;
}

export function photographicRegionChange(region, dx, dy, handle, width, height) {
  if (!Array.isArray(region) || region.length !== 4 || !region.every((value) => ["number", "string"].includes(typeof value)
    && String(value).trim() !== "" && Number.isFinite(Number(value)))
    || !Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) return null;
  const [left, top, right, bottom] = region.map(Number);
  if (left < 0 || top < 0 || right > 1 || bottom > 1 || left >= right || top >= bottom || !Number.isFinite(dx) || !Number.isFinite(dy)) return null;
  // Tiny numeric regions can leave no room for an anchored one-pixel resize.
  const bound = (value, low, high) => low > high ? null : Math.max(low, Math.min(high, value));
  let result;
  if (handle === "move") {
    dx = bound(dx, -left, 1 - right); dy = bound(dy, -top, 1 - bottom);
    result = [left + dx, top + dy, right + dx, bottom + dy];
  } else if (["nw", "ne", "sw", "se"].includes(handle)) {
    result = [handle.includes("w") ? bound(left + dx, 0, right - 1 / width) : left,
      handle.includes("n") ? bound(top + dy, 0, bottom - 1 / height) : top,
      handle.includes("e") ? bound(right + dx, left + 1 / width, 1) : right,
      handle.includes("s") ? bound(bottom + dy, top + 1 / height, 1) : bottom];
  } else return null;
  return result.includes(null) ? null : result;
}

export function createPhotographicRegionOverlay(host) {
  const { nodes, el, button, getMaterials, context, changed, status } = host;
  let enabled = true, operation = null;
  const layer = el("div", "", "photo-region-layer", "photographic-region-overlay");
  layer.dataset.interaction = "idle";
  layer.setAttribute("role", "group");
  layer.setAttribute("aria-label", "Human rectangle regions on the selected source");
  const toggle = button("Hide region overlays", "photographic-region-overlays-toggle", () => {
    enabled = !enabled; sync();
  });
  const help = el("p", "In Side by side inspection, move or resize human rectangles on the selected image. Arrow keys adjust one source pixel; Shift adjusts ten. Numeric Left, Top, Right and Bottom fields remain available. Escape cancels a pointer drag.", "photo-review-help", "photographic-region-help");
  toggle.setAttribute("aria-pressed", "true");
  const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  function apply(material, region) {
    if (!region) return;
    material.region = region; changed(); host.syncNumeric?.(); position();
  }
  function current(expected) {
    const value = context();
    return enabled && value?.ready && value.visible && !value.blocked && value.key === expected.key
      && getMaterials()[expected.index] === expected.material
      && photographicRegionChange(expected.material.region, 0, 0, "move", value.width, value.height) !== null;
  }
  function finish(event, restore = false) {
    if (!operation || (event.pointerId !== undefined && operation.pointerId !== event.pointerId)) return;
    const previous = operation; operation = null;
    layer.dataset.interaction = "idle";
    if (restore && current(previous)) apply(previous.material, previous.region);
    if (previous.control.hasPointerCapture?.(previous.pointerId)) previous.control.releasePointerCapture(previous.pointerId);
  }
  function position() {
    const value = context(), image = nodes.figures?.[0]?.image;
    layer.hidden = !enabled || !getMaterials().length || !value?.ready || !value.visible;
    if (layer.hidden || !image) return;
    layer.style.width = `${image.clientWidth}px`; layer.style.height = `${image.clientHeight}px`;
    for (const row of layer.children) {
      const material = getMaterials()[Number(row.dataset.regionIndex)];
      const rect = photographicRegionChange(material?.region, 0, 0, "move", value.width, value.height), crop = value.crop;
      row.hidden = !rect;
      if (!rect) continue;
      row.style.left = `${(rect[0] * value.width - crop.x) / crop.width * 100}%`;
      row.style.top = `${(rect[1] * value.height - crop.y) / crop.height * 100}%`;
      row.style.width = `${(rect[2] - rect[0]) * value.width / crop.width * 100}%`;
      row.style.height = `${(rect[3] - rect[1]) * value.height / crop.height * 100}%`;
      row.querySelectorAll("button").forEach((control) => { control.disabled = value.blocked; });
    }
  }
  function sync() {
    if (operation && !current(operation)) finish({});
    toggle.textContent = enabled ? "Hide region overlays" : "Show region overlays";
    toggle.setAttribute("aria-pressed", String(enabled));
    position();
  }
  function render() {
    if (operation) finish({}); layer.replaceChildren();
    getMaterials().forEach((material, index) => {
      const row = el("div", "", "photo-region-box"); row.dataset.regionIndex = String(index);
      row.setAttribute("role", "group"); row.setAttribute("aria-label", `Region ${index + 1}: ${material.label || material.name || "Human rectangle"}`);
      for (const [handle, label] of [["move", "Move"], ["nw", "Resize top left"], ["ne", "Resize top right"], ["sw", "Resize bottom left"], ["se", "Resize bottom right"]]) {
        const control = button("", `photographic-region-${handle}`, () => {});
        if (handle === "move") control.append(el("span", String(index + 1), "photo-region-label"));
        control.className = `photo-region-handle photo-region-${handle}`; control.dataset.regionIndex = String(index);
        control.setAttribute("aria-label", `${label} region ${index + 1}. Arrow keys adjust one source pixel; Shift adjusts ten.`);
        control.addEventListener("keydown", (event) => {
          if (event.key === "Escape") { event.preventDefault(); finish(event, true); return; }
          if (event.altKey || event.ctrlKey || event.metaKey) return;
          const direction = directions[event.key], value = context();
          if (!direction || !enabled || !value?.ready || !value.visible || value.blocked) return;
          event.preventDefault(); event.stopPropagation();
          const step = event.shiftKey ? 10 : 1;
          apply(material, photographicRegionChange(material.region, direction[0] * step / value.width, direction[1] * step / value.height, handle, value.width, value.height));
        });
        control.addEventListener("pointerdown", (event) => {
          const value = context(), image = nodes.figures?.[0]?.image;
          if (operation || event.isPrimary === false || event.button !== 0 || !enabled || !value?.ready || !value.visible || value.blocked || !image) return;
          const region = photographicRegionChange(material.region, 0, 0, "move", value.width, value.height);
          if (!region) return;
          const box = image.getBoundingClientRect(); if (!box.width || !box.height) return;
          event.preventDefault(); event.stopPropagation(); control.focus();
          operation = { key: value.key, material, index, handle, region,
            pointerId: event.pointerId, x: event.clientX, y: event.clientY, box, ...value, control };
          layer.dataset.interaction = "dragging";
          control.setPointerCapture(event.pointerId);
        });
        control.addEventListener("pointermove", (event) => {
          const active = operation;
          if (!active || active.control !== control || active.pointerId !== event.pointerId || !current(active)) return;
          event.preventDefault();
          apply(material, photographicRegionChange(active.region,
            (event.clientX - active.x) / active.box.width * active.crop.width / active.width,
            (event.clientY - active.y) / active.box.height * active.crop.height / active.height,
            handle, active.width, active.height));
        });
        control.addEventListener("pointerup", (event) => finish(event));
        control.addEventListener("pointercancel", (event) => finish(event, true));
        row.append(control);
      }
      layer.append(row);
    });
    sync();
  }
  const figure = nodes.figures?.[0];
  if (figure) {
    const surface = el("div", "", "photo-region-surface"); surface.append(figure.image, layer);
    figure.viewport.replaceChildren(surface);
    figure.image.addEventListener("load", sync);
    if (globalThis.ResizeObserver) new ResizeObserver(() => {
      if (operation) {
        const box = figure.image.getBoundingClientRect();
        if (box.width !== operation.box.width || box.height !== operation.box.height) finish({});
      }
      sync();
    }).observe(figure.image);
  }
  return { toggle, help, render, sync };
}
