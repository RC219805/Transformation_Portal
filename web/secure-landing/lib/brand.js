function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const FRONTDOOR_ASSETS = Object.freeze({
  brandDark: "/brand/dna-mark-dark.svg",
  brandLight: "/brand/dna-mark-light.svg",
  loopVideo: "/video/dna-loop.mp4"
});

function resolveBrandAssetPath(variant = "dark") {
  return variant === "light" ? FRONTDOOR_ASSETS.brandLight : FRONTDOOR_ASSETS.brandDark;
}

function renderBrandAsset({ variant = "dark", alt = "Dynamic Neural Access", className = "brand-asset" } = {}) {
  const safeAlt = escapeHtml(alt);
  const safeClassName = escapeHtml(className);
  const assetPath = resolveBrandAssetPath(variant);
  return `<img class="${safeClassName}" src="${assetPath}" alt="${safeAlt}" decoding="async" />`;
}

export { FRONTDOOR_ASSETS, escapeHtml, renderBrandAsset, resolveBrandAssetPath };
