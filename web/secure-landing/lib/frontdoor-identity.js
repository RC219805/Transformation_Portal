import { createHmac } from "node:crypto";

export const ACTOR_ASSERTION_HEADER = "x-tp-actor-assertion";

export function hasIdentitySecret(secret) {
  return Buffer.byteLength(String(secret || ""), "utf8") >= 32;
}

// Call only after Access/session authentication and (for writes) CSRF checks.
// Tenant membership is intentionally absent: the backend owns the mapping.
export function signActorAssertion({ secret, actor, method, target, now = Date.now() }) {
  if (!hasIdentitySecret(secret)) throw new Error("TP_FRONTDOOR_IDENTITY_SECRET requires at least 32 UTF-8 bytes");
  const identity = Object.fromEntries(
    ["username", "accessEmail", "role"].map((key) => [key, String(actor?.[key] || "").trim().toLowerCase()])
  );
  if (Object.values(identity).some((value) => !value || value.length > 320)) {
    throw new Error("authenticated actor identity is incomplete");
  }
  const payload = { v: 1, iat: Math.floor(now / 1000), method: method.toUpperCase(), target, actor: identity };
  const encoded = Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
  if (encoded.length > 8127) throw new Error("actor assertion request target is too long");
  const signature = createHmac("sha256", secret).update(`tp.frontdoor.actor.v1\n${encoded}`, "ascii").digest("hex");
  return `${encoded}.${signature}`;
}
