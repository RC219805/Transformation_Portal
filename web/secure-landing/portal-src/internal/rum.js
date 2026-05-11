import { generateTraceparent, normalizeTraceparent } from "../../lib/trace.js";

export const PORTAL_RUM_EVENT_TYPES = Object.freeze([
  "portal_shell_rendered",
  "bootstrap_ready",
  "first_view_interactive",
  "core_web_vital",
  "queue_request",
  "sse_reconnect",
  "login_submit_success",
  "logout_submit_attempt",
  "logout_submit_success"
]);

export const PORTAL_RUM_VIEWS = Object.freeze(["overview", "build", "operate", "review", "login"]);

export function createPortalRumState() {
  return {
    enabled: false,
    observersStarted: false,
    bootstrapTraceparent: "",
    pageTraceparent: generateTraceparent(),
    queuedSamples: [],
    emittedMilestones: Object.create(null),
    firstInteractiveScheduled: false,
    vitals: {
      lcpMs: null,
      inpMs: null,
      clsScore: 0,
      finalized: false
    }
  };
}

export function normalizePortalRumView(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return PORTAL_RUM_VIEWS.includes(normalized) ? normalized : "overview";
}

export function normalizePortalRumTraceparent(value, fallback = "") {
  return normalizeTraceparent(value) || normalizeTraceparent(fallback) || generateTraceparent();
}

export function createChildTraceparent(parentTraceparent = "") {
  return generateTraceparent({ parentTraceparent });
}
