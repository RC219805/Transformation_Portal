import { getConfig } from "./config.js";

const STRIP_REQUEST_HEADERS = new Set([
  "authorization",
  "connection",
  "content-length",
  "cookie",
  "forwarded",
  "host",
  "x-forwarded-for",
  "x-forwarded-host",
  "x-forwarded-proto",
  "x-api-key",
  "x-csrf-token",
  "x-real-ip"
]);

const STRIP_RESPONSE_HEADERS = new Set([
  "connection",
  "content-encoding",
  "content-length",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade"
]);

export function buildUpstreamUrl(pathname, search = "") {
  return new URL(`${pathname}${search}`, getConfig().fastapiOrigin).toString();
}

export function buildUpstreamHeaders(
  sourceHeaders,
  { backendApiKey, actor, preferIdentityEncoding = false, forwarding = null, traceparent = "" }
) {
  const headers = new Headers();

  for (const [key, value] of sourceHeaders.entries()) {
    const normalizedKey = key.toLowerCase();
    if (STRIP_REQUEST_HEADERS.has(normalizedKey)) continue;
    headers.set(key, value);
  }

  if (preferIdentityEncoding) {
    headers.set("Accept-Encoding", "identity");
  }

  if (forwarding?.proto) headers.set("x-forwarded-proto", forwarding.proto);
  if (forwarding?.host) headers.set("x-forwarded-host", forwarding.host);
  if (forwarding?.clientIp && forwarding.clientIp !== "unknown") {
    headers.set("x-forwarded-for", forwarding.clientIp);
    headers.set("x-real-ip", forwarding.clientIp);
    if (forwarding.host && forwarding.proto) {
      headers.set("Forwarded", `for="${forwarding.clientIp}";host="${forwarding.host}";proto="${forwarding.proto}"`);
    }
  }

  headers.set("Authorization", `Bearer ${backendApiKey}`);
  headers.set("x-api-key", backendApiKey);
  if (traceparent) headers.set("traceparent", traceparent);

  if (actor?.username) headers.set("x-tp-actor", actor.username);
  if (actor?.accessEmail) headers.set("x-tp-actor-email", actor.accessEmail);
  if (actor?.role) headers.set("x-tp-actor-role", actor.role);

  return headers;
}

export function copyUpstreamResponseHeaders(sourceHeaders) {
  const headers = new Headers();
  for (const [key, value] of sourceHeaders.entries()) {
    if (STRIP_RESPONSE_HEADERS.has(key.toLowerCase())) continue;
    headers.set(key, value);
  }
  return headers;
}

export function isSsePath(pathname) {
  return /^\/v1\/jobs\/[^/]+\/events$/.test(pathname);
}
