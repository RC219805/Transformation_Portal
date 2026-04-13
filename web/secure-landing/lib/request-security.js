import { isLocalAccessBypassEnabled } from "./config.js";
import { buildRequestUrl } from "./http.js";

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

  const expectedOrigin = buildRequestUrl(request, request.nextUrl.pathname || "/").origin;
  const origin = request.headers.get("origin");
  if (origin) {
    return origin === expectedOrigin;
  }

  const referrer = request.headers.get("referer");
  if (referrer) {
    return originFromReferrer(referrer) === expectedOrigin;
  }

  return isLocalAccessBypassEnabled();
}
