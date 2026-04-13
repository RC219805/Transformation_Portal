const BASE_SECURITY_HEADERS = {
  "Referrer-Policy": "same-origin",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
};

export const FRONTDOOR_CSP = [
  "default-src 'self'",
  "style-src 'self'",
  "img-src 'self' data:",
  "media-src 'self'",
  "font-src 'self'",
  "script-src 'none'",
  "connect-src 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "object-src 'none'"
].join("; ");

export const LOGIN_CSP = FRONTDOOR_CSP;

function firstHeaderValue(value) {
  return String(value || "")
    .split(",")[0]
    .trim();
}

export function buildRequestUrl(request, pathname) {
  const url = new URL(pathname, request.url);
  const host = firstHeaderValue(request.headers.get("x-forwarded-host"))
    || firstHeaderValue(request.headers.get("host"))
    || request.nextUrl.host;
  const proto = firstHeaderValue(request.headers.get("x-forwarded-proto"))
    || String(request.nextUrl.protocol || "").replace(/:$/, "");

  if (host) {
    url.host = host;
  }
  if (proto) {
    url.protocol = proto.endsWith(":") ? proto : `${proto}:`;
  }
  return url;
}

export function applySecurityHeaders(response, { csp = null } = {}) {
  for (const [name, value] of Object.entries(BASE_SECURITY_HEADERS)) {
    if (!response.headers.has(name)) {
      response.headers.set(name, value);
    }
  }
  if (csp) {
    response.headers.set("Content-Security-Policy", csp);
  }
  return response;
}
