export function createLazyPhotographicReview(host) {
  let api = null;
  let loading = null;
  let context = { job: null, artifact: null };
  let notice = null;
  let retryAfter = 0;
  const doc = host.document || globalThis.document;
  const origin = host.origin || globalThis.location?.origin;
  const importer = host.importPhotographicModule || ((url) => import(url));

  function isPhotographic(job) {
    return /lux[-_]depth|photograph/i.test(String(job?.pipeline || job?.pipeline_name || ""))
      || (job?.artifacts || []).some((item) => /(^|\/)delivery\.tiff?$/.test(item.relative_path || item.path || ""));
  }
  function assetUrl(raw, filename) {
    const parsed = new URL(raw, origin);
    if (!raw || parsed.origin !== origin || parsed.username || parsed.password
      || !parsed.pathname.endsWith(`/${filename}`)) throw new Error("Photographic review asset is unavailable.");
    return `${parsed.pathname}${parsed.search}`;
  }
  function loadCss() {
    if (host.loadPhotographicCss) return host.loadPhotographicCss();
    const href = assetUrl(String(doc?.body?.dataset?.photographicReviewCssUrl || ""), "portal-photographic.css");
    if (doc.querySelector('link[data-ui="photographic-review-css"]')) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const link = doc.createElement("link");
      link.rel = "stylesheet"; link.href = href; link.dataset.ui = "photographic-review-css";
      const timer = setTimeout(() => { link.remove(); reject(new Error("Photographic review styles timed out.")); }, 15000);
      link.onload = () => { clearTimeout(timer); resolve(); };
      link.onerror = () => { clearTimeout(timer); link.remove(); reject(new Error("Photographic review styles are unavailable.")); };
      doc.head.append(link);
    });
  }
  function show(job, artifact) {
    context = { job, artifact };
    if (api) { api.show(job, artifact); return; }
    if (!job || !isPhotographic(job)) { if (notice) notice.hidden = true; return; }
    if (loading || Date.now() < retryAfter) return;
    const raw = String(doc?.body?.dataset?.photographicReviewJsUrl || "");
    let url;
    try {
      url = assetUrl(raw, "portal-photographic.js");
    } catch { return; }
    loading = Promise.all([importer(url), loadCss()]).then(([module]) => {
      if (typeof module.createPhotographicReview !== "function") throw new Error("Photographic review module is invalid.");
      api = module.createPhotographicReview(host);
      notice?.remove();
      api.show(context.job, context.artifact);
    }).catch(() => {
      retryAfter = Date.now() + 10000;
      if (!context.job || !isPhotographic(context.job)) return;
      if (!notice && doc?.createElement && host.els.artifactShellContent) {
        notice = doc.createElement("p");
        notice.className = "photo-review-help";
        notice.dataset.ui = "photographic-review-load-error";
        notice.setAttribute("role", "status");
        host.els.artifactShellContent.append(notice);
      }
      if (notice) { notice.hidden = false; notice.textContent = "Photographic review could not load. Existing artifact inspection remains available; return to Review to retry."; }
    }).finally(() => { loading = null; });
  }
  return { show };
}
