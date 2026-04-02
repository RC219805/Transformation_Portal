import { getConfig } from "./config.js";

const STRIP_REQUEST_HEADERS = new Set([
  "authorization",
  "connection",
  "content-length",
  "cookie",
  "host",
  "x-api-key",
  "x-csrf-token"
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

export function buildUpstreamHeaders(sourceHeaders, { backendApiKey, actor, preferIdentityEncoding = false }) {
  const headers = new Headers();

  for (const [key, value] of sourceHeaders.entries()) {
    const normalizedKey = key.toLowerCase();
    if (STRIP_REQUEST_HEADERS.has(normalizedKey)) continue;
    headers.set(key, value);
  }

  if (preferIdentityEncoding) {
    headers.set("Accept-Encoding", "identity");
  }
  headers.set("Authorization", `Bearer ${backendApiKey}`);
  headers.set("x-api-key", backendApiKey);

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
