const DEFAULT_PORTAL_RETURN_TO = "/portal";
const ALLOWED_PORTAL_VIEWS = new Set(["overview", "build", "operate", "review"]);
const ALLOWED_QUERY_KEYS = ["view", "job", "artifact", "compare"];

function _valueOrEmpty(rawValue) {
  return String(rawValue || "").trim();
}

export function validatePortalReturnTo(rawValue) {
  const value = _valueOrEmpty(rawValue);
  if (!value || !value.startsWith("/") || value.startsWith("//")) {
    return null;
  }

  let parsed;
  try {
    parsed = new URL(value, "https://portal.invalid");
  } catch {
    return null;
  }

  if (parsed.origin !== "https://portal.invalid" || parsed.pathname !== DEFAULT_PORTAL_RETURN_TO || parsed.hash) {
    return null;
  }

  const params = parsed.searchParams;
  const seenKeys = new Set();
  for (const key of params.keys()) {
    if (seenKeys.has(key)) {
      return null;
    }
    seenKeys.add(key);
    if (!ALLOWED_QUERY_KEYS.includes(key)) {
      return null;
    }
  }

  const view = params.get("view");
  const job = params.get("job");
  const artifact = params.get("artifact");
  const compare = params.get("compare");

  if (view !== null && !ALLOWED_PORTAL_VIEWS.has(view)) {
    return null;
  }
  if (job !== null && !_valueOrEmpty(job)) {
    return null;
  }
  if (artifact !== null && !_valueOrEmpty(artifact)) {
    return null;
  }
  if (compare !== null && compare !== "1") {
    return null;
  }

  const hasJobContext = job !== null || artifact !== null || compare !== null;
  if (hasJobContext) {
    if (!view || (view !== "operate" && view !== "review")) {
      return null;
    }
    if (!job) {
      return null;
    }
  }

  const normalized = new URL(DEFAULT_PORTAL_RETURN_TO, "https://portal.invalid");
  if (view) {
    normalized.searchParams.set("view", view);
  }
  if (job) {
    normalized.searchParams.set("job", _valueOrEmpty(job));
  }
  if (artifact) {
    normalized.searchParams.set("artifact", _valueOrEmpty(artifact));
  }
  if (compare === "1") {
    normalized.searchParams.set("compare", "1");
  }

  return `${normalized.pathname}${normalized.search}`;
}

export function resolvePortalReturnTo(rawValue, fallback = DEFAULT_PORTAL_RETURN_TO) {
  return validatePortalReturnTo(rawValue) || fallback;
}

export function applyPortalReturnTo(url, rawValue) {
  const validated = validatePortalReturnTo(rawValue);
  if (validated) {
    url.searchParams.set("returnTo", validated);
  } else {
    url.searchParams.delete("returnTo");
  }
  return url;
}

export function currentPortalReturnToFromRequest(request) {
  return validatePortalReturnTo(`${request.nextUrl.pathname || ""}${request.nextUrl.search || ""}`);
}
