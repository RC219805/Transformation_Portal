import { photographicRecipe, photographicPreviewMatches } from "./photographic-review-support.js";

const SHA256 = /^[a-f0-9]{64}$/;
const LABELS = ["glass", "stone", "water", "foliage", "sky"];

export function createPhotographicRevisionEditor(host) {
  const { nodes, el, input, select, field, button, status, resetPreview, revise, getDraft, getMaterials } = host;
  let native = false;
  let eligible = false;
  let blocked = false;
  let nextRegion = 1;
  const globalEdits = () => Number(getDraft().exposure) !== 0 || Number(getDraft().warmth) !== 0 || Number(getDraft().saturation) !== 1;
  const changed = () => { resetPreview(); syncControls(); };

  function syncControls() {
    if (!nodes.nativeMaterials) return;
    nodes.nativeMaterials.value = native ? "native" : "legacy";
    nodes.nativeMaterials.disabled = blocked;
    nodes.nativeMaterialsField.hidden = !eligible && !native;
    nodes.nativeMaterials.hidden = !eligible && !native;
    nodes.nativeMaterialsHelp.hidden = !native;
    nodes.nativeMaterialsHelp.textContent = eligible
      ? "Native masks use these human rectangles. Overlap and uncertain support are protected. Inspect the separate unapproved revision."
      : "Native Materials is unavailable for this source. Native intent is retained; clear regions before changing mode.";
    nodes.materialsHelp.textContent = native
      ? "Native material labels and rectangles are supplied by you. Masks follow native source coordinates; this mode asserts a display-referred SDR source."
      : "Local material regions are explicit feathered rectangles, not automatic semantic masks. Keep adjustments subtle; overlapping exposure is checked by the renderer.";
    Object.values(nodes.revisionInputs || {}).forEach((control) => { control.disabled = blocked || native; });
    if (nodes.reference) nodes.reference.disabled = blocked || native;
    nodes.materials?.querySelectorAll("input,select,button").forEach((control) => { control.disabled = blocked; });
    nodes.addRegion.disabled = blocked || (native && !eligible) || getMaterials().length >= (native ? 4 : 16);
    if (native && !eligible) { nodes.previewRevision.disabled = true; nodes.commitRevision.disabled = true; }
  }

  function build() {
    const section = el("section", "", "photo-review-section", "photographic-revision");
    section.append(el("h4", "Create a separate revision", "photo-review-subtitle"),
      el("p", "Bounded adjustments create a new unapproved artifact. Originals and existing deliveries stay unchanged. Inspect the rendered variant before creating it.", "photo-review-help"));
    const controls = el("div", "", "photo-review-controls");
    nodes.nativeMaterials = select("photographic-native-materials", [["legacy", "Local adjustments"], ["native", "Native Materials · 8/16-bit display SDR"]]);
    nodes.nativeMaterialsField = field("Revision mode", nodes.nativeMaterials);
    nodes.nativeMaterialsField.hidden = true;
    nodes.nativeMaterials.addEventListener("change", () => {
      const requested = nodes.nativeMaterials.value === "native";
      if (blocked || (requested && !eligible) || getMaterials().length || globalEdits()) {
        nodes.nativeMaterials.value = native ? "native" : "legacy";
        status("Clear global adjustments and remove local regions before changing revision mode.", true);
        return;
      }
      native = requested;
      changed(); renderMaterials();
    });
    controls.append(nodes.nativeMaterialsField);
    nodes.revisionInputs = {};
    for (const [key, label, min, max, step] of [["exposure", "Exposure (EV)", -1, 1, 0.05], ["warmth", "Warmth balance", -0.15, 0.15, 0.01], ["saturation", "Saturation", 0.8, 1.2, 0.05]]) {
      const control = input(`photographic-revision-${key}`, "number", String(getDraft()[key]));
      control.min = String(min); control.max = String(max); control.step = String(step);
      control.addEventListener("input", () => { getDraft()[key] = control.value; changed(); });
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
    nodes.addRegion = button("Add local material region", "photographic-add-material", () => {
      if (blocked || (native && !eligible)) return;
      if (getMaterials().length >= (native ? 4 : 16)) { status(`At most ${native ? 4 : 16} local regions are supported.`, true); return; }
      getMaterials().push(native
        ? { id: `region-${nextRegion++}`, label: "stone", region: [0.25, 0.25, 0.75, 0.75], strength: 0.03 }
        : { name: `Region ${getMaterials().length + 1}`, region: [0.25, 0.25, 0.75, 0.75], exposure_stops: 0, warmth: 0, feather: 0.03 });
      changed(); renderMaterials();
    });
    nodes.nativeMaterialsHelp = el("p", "", "photo-review-help", "photographic-native-materials-help");
    nodes.nativeMaterialsHelp.hidden = true;
    nodes.materialsHelp = el("p", "Local material regions are explicit feathered rectangles, not automatic semantic masks. Keep adjustments subtle; overlapping exposure is checked by the renderer.", "photo-review-help");
    nodes.materialsDiagnostics = el("div", "", "photo-review-help", "photographic-materials-diagnostics");
    section.append(controls, nodes.materialsHelp,
      nodes.nativeMaterialsHelp, nodes.addRegion, nodes.materials, nodes.variantPreview, nodes.revisionReceipt, nodes.materialsDiagnostics);
    return section;
  }

  function renderMaterials() {
    if (!nodes.materials) return;
    nodes.materials.replaceChildren();
    getMaterials().forEach((material, index) => {
      const controls = el("fieldset", "", "photo-review-material-region");
      controls.append(el("legend", `Local region ${index + 1}`));
      if (native) {
        const label = select("photographic-material-label", LABELS.map((value) => [value, value]));
        label.value = material.label;
        label.addEventListener("change", () => { material.label = label.value; changed(); });
        controls.append(field("Material", label));
      } else {
        const name = input("photographic-material-name", "text", material.name);
        name.maxLength = 80;
        name.addEventListener("input", () => { material.name = name.value; changed(); });
        controls.append(field("Material / region name", name));
      }
      for (const [position, label] of ["Left", "Top", "Right", "Bottom"].entries()) {
        const control = input(`photographic-material-${label.toLowerCase()}`, "number", String(material.region[position]));
        control.min = "0"; control.max = "1"; control.step = "0.01";
        control.addEventListener("input", () => { material.region[position] = control.value; changed(); });
        controls.append(field(`${label} (0–1)`, control));
      }
      const adjustments = native ? [["strength", "Material strength", 0, 0.05, 0.005]]
        : [["exposure_stops", "Local exposure (EV)", -0.5, 0.5, 0.05], ["warmth", "Local warmth", -0.1, 0.1, 0.01], ["feather", "Edge feather", 0.005, 0.1, 0.005]];
      for (const [key, label, min, max, step] of adjustments) {
        const control = input(`photographic-material-${key}`, "number", String(material[key]));
        control.min = String(min); control.max = String(max); control.step = String(step);
        control.addEventListener("input", () => { material[key] = control.value; changed(); });
        controls.append(field(label, control));
      }
      controls.append(button("Remove region", "photographic-remove-material", () => {
        if (blocked) return;
        getMaterials().splice(index, 1); changed(); renderMaterials();
      }));
      nodes.materials.append(controls);
    });
    syncControls();
  }

  function recipe() {
    if (!native) return photographicRecipe({ ...getDraft(), materials: getMaterials() });
    if (!eligible) throw new Error("Native Materials is unavailable; its selected mode cannot fall back to local adjustments.");
    if (globalEdits()) throw new Error("Native Materials requires neutral global adjustments.");
    const materials = getMaterials();
    if (!materials.length || materials.length > 4) throw new Error("Native Materials requires one to four explicit regions.");
    const number = (value, min, max) => {
      const result = Number(value);
      if (String(value).trim() === "" || !Number.isFinite(result) || result < min || result > max) throw new Error("Native material values are outside the supported range.");
      return Math.round(result * 10000) / 10000;
    };
    const strengths = new Map();
    const regions = materials.map((material) => {
      if (!LABELS.includes(material.label)) throw new Error("Choose a supported material label.");
      const region = material.region.map((value) => number(value, 0, 1));
      if (region.length !== 4 || region[0] >= region[2] || region[1] >= region[3]) throw new Error("Material regions require positive area.");
      const strength = number(material.strength, 0, 0.05);
      if (strengths.has(material.label) && strengths.get(material.label) !== strength) throw new Error("Regions with the same material label require the same strength.");
      strengths.set(material.label, strength);
      return { id: material.id, label: material.label, region, strength };
    });
    regions.sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0);
    return { schema: "tp.photo.materials_recipe.v1", input_stage: "display_referred", regions };
  }

  function matches(receipt, variant, requested, reference = null) {
    if (requested?.schema !== "tp.photo.materials_recipe.v1") return photographicPreviewMatches(receipt, variant, requested, reference);
    return Boolean(receipt && variant && receipt.schema === "tp.photo.materials_revision.v1"
      && receipt.processing_contract === "tp.photo.materials_revision_processing.v1"
      && receipt.source_master_sha256 === variant.sha256
      && receipt.source_identity_sha256 === (variant.source_identity_sha256 || variant.sha256)
      && receipt.width === variant.width && receipt.height === variant.height && receipt.output_bit_depth === 16
      && receipt.output_presentation?.schema === "tp.photo.revision_presentation.v1" && receipt.output_presentation.stage === "display_referred"
      && (receipt.input_presentation_policy_sha256 || null) === (variant.presentation_policy_sha256 || null)
      && SHA256.test(receipt.policy_sha256 || "") && receipt.review_status === "unreviewed" && receipt.approval_transferred === false
      && receipt.recipe?.schema === requested.schema && receipt.recipe.input_stage === requested.input_stage
      && Array.isArray(receipt.recipe.regions) && receipt.recipe.regions.length === requested.regions.length
      && requested.regions.every((region, index) => {
        const actual = receipt.recipe.regions[index];
        return actual?.id === region.id && actual.label === region.label && actual.strength === region.strength
          && JSON.stringify(actual.region) === JSON.stringify(region.region);
      }) && reference === null && (receipt.reference_sha256 || null) === null);
  }

  function sync(capability, variant, disabled) {
    const policy = variant?.presentation?.policy;
    const recorded = policy || variant?.presentation_policy_sha256 || variant?.presentation?.policy_sha256;
    eligible = capability?.enabled === true && [8, 16].includes(variant?.bit_depth)
      && (!recorded || (policy?.stage === "display_referred" && SHA256.test(variant.presentation_policy_sha256 || "")
        && variant.presentation?.policy_sha256 === variant.presentation_policy_sha256));
    blocked = disabled;
    syncControls();
  }

  function reset() {
    native = false; nextRegion = 1;
    Object.entries(nodes.revisionInputs || {}).forEach(([key, control]) => { control.value = String(getDraft()[key]); });
    renderMaterials();
  }
  return { build, renderMaterials, recipe, matches, sync, reset };
}
