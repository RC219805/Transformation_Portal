import { createHash } from "node:crypto";

export function stableRolloutBucket(key) {
  const normalized = String(key || "").trim().toLowerCase();
  if (!normalized) {
    return 100;
  }
  const digest = createHash("sha256").update(normalized).digest("hex");
  return Number.parseInt(digest.slice(0, 8), 16) % 100;
}
