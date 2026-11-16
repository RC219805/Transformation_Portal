import { getConfig } from "./config.js";
import { hasIdentitySecret } from "./frontdoor-identity.js";
import { buildUpstreamHeaders, buildUpstreamUrl } from "./proxy.js";

export async function backendRecoveryBootstrap(session, traceparent, { fetchImpl = globalThis.fetch, timeoutMs = 2500 } = {}) {
  const unavailable = { recoveryContext: null, recoveryContextStatus: "unavailable",
    features: { jobSubmissionReceipts: false, uploadReceipts: false } };
  const config = getConfig();
  if (!config.pilotControlPlaneEnabled || !config.backendApiKey || !hasIdentitySecret(config.frontdoorIdentitySecret)
    || config.frontdoorIdentitySecret === config.backendApiKey) return unavailable;
  const controller = new AbortController();
  let timer, reader;
  const deadline = new Promise((_, reject) => { timer = setTimeout(() => {
    controller.abort(); reject(new Error("recovery bootstrap timed out"));
  }, timeoutMs); });
  try {
    const url = buildUpstreamUrl("/portal/bootstrap");
    const headers = buildUpstreamHeaders(new Headers({ Accept: "application/json" }), {
      backendApiKey: config.backendApiKey, actor: session, traceparent,
      identity: { secret: config.frontdoorIdentitySecret, method: "GET", target: "/portal/bootstrap" }
    });
    const response = await Promise.race([fetchImpl(url, { method: "GET", headers, cache: "no-store", redirect: "manual", signal: controller.signal }), deadline]);
    if (!response.ok || !response.body || !/^application\/json(?:\s*;|$)/i.test(response.headers.get("content-type") || "")) return unavailable;
    reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8", { fatal: true });
    let body = "", bytes = 0;
    for (;;) {
      const { value, done } = await Promise.race([reader.read(), deadline]);
      if (done) break;
      bytes += value.byteLength;
      if (bytes > 16384) return unavailable;
      body += decoder.decode(value, { stream: true });
    }
    const parsed = JSON.parse(body + decoder.decode());
    const context = parsed?.recoveryContext;
    if (context?.schema !== "tp.portal.recovery_context.v1" || typeof context.ownerId !== "string" || !/^[a-f0-9]{64}$/.test(context.ownerId)
      || typeof context.tenantId !== "string" || !context.tenantId || context.tenantId.length > 256
      || /[\x00-\x1f\x7f]/.test(context.tenantId)) return unavailable;
    return { recoveryContext: { schema: context.schema, tenantId: context.tenantId, ownerId: context.ownerId }, recoveryContextStatus: "ready",
      features: { jobSubmissionReceipts: parsed.features?.jobSubmissionReceipts === true, uploadReceipts: parsed.features?.uploadReceipts === true } };
  } catch { return unavailable; }
  finally {
    clearTimeout(timer);
    if (reader) { void reader.cancel().catch(() => {}); try { reader.releaseLock(); } catch { /* A timed-out read can still be pending. */ } }
    controller.abort();
  }
}
