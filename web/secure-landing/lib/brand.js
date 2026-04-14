function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const FRONTDOOR_ASSETS = Object.freeze({
  brand: Object.freeze({
    symbol: Object.freeze({
      dark: "/brand/dna-symbol-dark.svg",
      light: "/brand/dna-symbol-light.svg"
    }),
    lockup: Object.freeze({
      dark: "/brand/dna-lockup-dark.svg",
      light: "/brand/dna-lockup-light.svg"
    })
  }),
  loopVideo: "/video/dna-loop.mp4"
});

function resolveBrandAssetPath({ kind = "symbol", variant = "dark" } = {}) {
  const safeKind = kind === "lockup" ? "lockup" : "symbol";
  const safeVariant = variant === "light" ? "light" : "dark";
  return FRONTDOOR_ASSETS.brand[safeKind][safeVariant];
}

function renderBrandAsset({
  kind = "symbol",
  variant = "dark",
  alt = "Dynamic Neural Access",
  className = "brand-asset"
} = {}) {
  const safeAlt = escapeHtml(alt);
  const safeClassName = escapeHtml(className);
  const assetPath = resolveBrandAssetPath({ kind, variant });
  return `<img class="${safeClassName}" src="${assetPath}" alt="${safeAlt}" decoding="async" />`;
}

export { FRONTDOOR_ASSETS, escapeHtml, renderBrandAsset, resolveBrandAssetPath };
