import { isLocalAccessBypassEnabled } from "./config.js";
import { buildRequestUrl, isLoopbackHostname } from "./http.js";

const UNSAFE_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

function originFromReferrer(referrer) {
  try {
    return new URL(referrer).origin;
  } catch {
    return "";
  }
}

export function isUnsafeMethod(method) {
  return UNSAFE_METHODS.has(String(method || "GET").toUpperCase());
}

export function validateOriginAndReferrer(request) {
  if (!isUnsafeMethod(request.method)) return true;

  const requestUrl = buildRequestUrl(request, request.nextUrl.pathname || "/");
  const expectedOrigin = requestUrl.origin;
  const origin = request.headers.get("origin");
  if (origin) {
    return origin === expectedOrigin;
  }

  const referrer = request.headers.get("referer");
  if (referrer) {
    return originFromReferrer(referrer) === expectedOrigin;
  }

  return isLocalAccessBypassEnabled() && isLoopbackHostname(requestUrl.hostname);
}
