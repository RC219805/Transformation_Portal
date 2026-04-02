const BASE_SECURITY_HEADERS = {
  "Referrer-Policy": "same-origin",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
};

export const LOGIN_CSP = [
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
