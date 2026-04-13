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

function normalizeProto(value) {
  const normalized = firstHeaderValue(value).replace(/:$/, "").trim().toLowerCase();
  return normalized === "http" || normalized === "https" ? normalized : "";
}

function parseHost(value) {
  const normalized = firstHeaderValue(value).toLowerCase();
  if (!normalized) return null;

  try {
    const url = new URL(`http://${normalized}`);
    return {
      host: url.host.toLowerCase(),
      hostname: url.hostname.toLowerCase(),
      port: url.port || ""
    };
  } catch {
    return null;
  }
}

function isLoopbackHostname(hostname) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1" || hostname === "[::1]";
}

function hostsAreEquivalent(candidateValue, fallbackValue) {
  const candidate = parseHost(candidateValue);
  const fallback = parseHost(fallbackValue);
  if (!candidate || !fallback) return false;
  if (candidate.host === fallback.host) return true;
  return (
    candidate.port === fallback.port
    && isLoopbackHostname(candidate.hostname)
    && isLoopbackHostname(fallback.hostname)
  );
}

function trustedHostValue(headerValue, fallbackValue) {
  const candidate = parseHost(headerValue);
  return candidate && hostsAreEquivalent(headerValue, fallbackValue) ? candidate.host : fallbackValue;
}

function trustedProtoValue(headerValue, fallbackValue) {
  const candidate = normalizeProto(headerValue);
  const fallback = normalizeProto(fallbackValue);
  return candidate && candidate === fallback ? candidate : fallback;
}

export function buildRequestUrl(request, pathname) {
  const url = new URL(pathname, request.url);
  const requestUrl = new URL(request.url);
  const fallbackHost = request.nextUrl.host || requestUrl.host;
  const fallbackProto = request.nextUrl.protocol || requestUrl.protocol || "";
  const host = trustedHostValue(
    firstHeaderValue(request.headers.get("x-forwarded-host"))
      || firstHeaderValue(request.headers.get("host")),
    fallbackHost
  );
  const proto = trustedProtoValue(request.headers.get("x-forwarded-proto"), fallbackProto);

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
