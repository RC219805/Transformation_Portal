export interface Env {
  FRONTDOOR_ORIGIN?: string;
}

const HOP_BY_HOP_HEADERS = [
  "connection",
  "upgrade",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "trailers",
  "transfer-encoding"
];

const SPOOFABLE_FORWARDING_HEADERS = [
  "forwarded",
  "x-forwarded-for",
  "x-real-ip"
];

function requireFrontdoorOrigin(env: Env, requestUrl: URL): URL {
  if (!env.FRONTDOOR_ORIGIN) {
    throw new Error("FRONTDOOR_ORIGIN is not configured");
  }

  const origin = new URL(env.FRONTDOOR_ORIGIN);

  if (origin.protocol !== "https:") {
    throw new Error("FRONTDOOR_ORIGIN must be https");
  }

  if (
    origin.username ||
    origin.password ||
    origin.pathname !== "/" ||
    origin.search ||
    origin.hash
  ) {
    throw new Error("FRONTDOOR_ORIGIN must be an origin without credentials, path, query, or fragment");
  }

  if (origin.hostname === requestUrl.hostname) {
    throw new Error("FRONTDOOR_ORIGIN must not point at the public Worker hostname");
  }

  return origin;
}

function buildUpstreamHeaders(request: Request, incomingUrl: URL): Headers {
  const headers = new Headers(request.headers);
  const connectionHeader = request.headers.get("connection");

  if (connectionHeader) {
    for (const token of connectionHeader.split(",")) {
      const header = token.trim();
      if (header) {
        headers.delete(header);
      }
    }
  }

  for (const header of HOP_BY_HOP_HEADERS) {
    headers.delete(header);
  }

  for (const header of SPOOFABLE_FORWARDING_HEADERS) {
    headers.delete(header);
  }

  const cfConnectingIp = request.headers.get("cf-connecting-ip");
  if (cfConnectingIp) {
    headers.set("x-forwarded-for", cfConnectingIp);
  }

  headers.delete("host");
  headers.set("x-forwarded-host", incomingUrl.host);
  headers.set("x-forwarded-proto", incomingUrl.protocol.replace(":", ""));

  return headers;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const incomingUrl = new URL(request.url);

    let frontdoorOrigin: URL;
    try {
      frontdoorOrigin = requireFrontdoorOrigin(env, incomingUrl);
    } catch (error) {
      return new Response(
        error instanceof Error ? error.message : "Invalid Worker configuration",
        { status: 500 }
      );
    }

    const upstreamUrl = new URL(
      incomingUrl.pathname + incomingUrl.search,
      frontdoorOrigin
    );

    const upstreamRequest = new Request(upstreamUrl, {
      method: request.method,
      headers: buildUpstreamHeaders(request, incomingUrl),
      body:
        request.method === "GET" || request.method === "HEAD"
          ? undefined
          : request.body,
      redirect: "manual"
    });

    return fetch(upstreamRequest);
  }
};
