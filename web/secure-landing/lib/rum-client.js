// Server-side string builder that returns the JavaScript body injected into
// the rendered HTML via <script nonce="...">. The returned string is the JS
// body only — the calling route is responsible for the <script> wrapper and
// the matching CSP nonce.
//
// Mirrors the portal RUM emitter at portal-src/portal.template.js (LCP/CLS/INP
// observers, first_view_interactive, queued POSTs to /v1/portal/rum) so the
// front door produces samples in the same shape that the backend validator
// at app.py:_record_portal_rum already accepts.

import { normalizeTraceparent } from "./trace.js";
import { getConfig } from "./config.js";

export const LOGIN_SUBMIT_BREADCRUMB_KEY = "tpLoginSubmitStartedAt";
export const LOGIN_SUBMIT_FAILURE_MARKER_COOKIE = "tp_login_submit_failure";
export const LOGIN_SUBMIT_FAILURE_MARKER_MAX_AGE_SECONDS = 60;
// Server-set marker the portal bundle reads on first /portal load to
// emit login_submit_success (#1689 client-side mirror series). Path=/
// so the cookie crosses from the front-door /login response to the
// portal page; max-age caps stale-tab false positives. The cookie
// value is a fixed "1" presence marker — the actual elapsed time is
// computed from the sessionStorage breadcrumb, never from the cookie.
export const LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE = "tp_login_submit_success";
export const LOGIN_SUBMIT_SUCCESS_MARKER_VALUE = "1";
export const LOGIN_SUBMIT_SUCCESS_MARKER_MAX_AGE_SECONDS = 60;

// Marker descriptors that pin every variant of the RUM cookie pair to a
// single shape: { name, path, maxAgeSeconds }. The setRumMarkerCookie /
// clearRumMarkerCookie primitives below consume these descriptors so a
// future logout client-mirror lands by adding two more constants
// (LOGOUT_SUBMIT_FAILURE_MARKER, LOGOUT_SUBMIT_SUCCESS_MARKER) instead
// of duplicating four near-identical helpers.
//
// The failure marker is scoped to Path=/login so it never reaches
// /portal; the success marker is scoped to Path=/ so the portal bundle
// can read and clear it on first load.
export const LOGIN_SUBMIT_FAILURE_MARKER = Object.freeze({
  name: LOGIN_SUBMIT_FAILURE_MARKER_COOKIE,
  path: "/login",
  maxAgeSeconds: LOGIN_SUBMIT_FAILURE_MARKER_MAX_AGE_SECONDS,
});
export const LOGIN_SUBMIT_SUCCESS_MARKER = Object.freeze({
  name: LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE,
  path: "/",
  maxAgeSeconds: LOGIN_SUBMIT_SUCCESS_MARKER_MAX_AGE_SECONDS,
});

// Set a server-set RUM marker cookie. httpOnly=false because the
// portal bundle and the inline rum-client.js script both need to read
// these from JS; secure tracks config.sessionCookieSecure so local-dev
// HTTP loopbacks don't strip the cookie; sameSite="lax" lets the
// cookie ride the same-origin 303 redirect from POST /login (or
// future /logout) to the GET landing page. Nullish values are coerced
// to an empty string defensively so a caller passing null/undefined
// doesn't end up with the literal strings "null" / "undefined" as the
// cookie value, while explicit falsy marker values like 0/false remain
// observable.
export function setRumMarkerCookie(response, marker, value) {
  const config = getConfig();
  response.cookies.set(marker.name, String(value ?? ""), {
    httpOnly: false,
    secure: config.sessionCookieSecure,
    sameSite: "lax",
    path: marker.path,
    maxAge: marker.maxAgeSeconds,
  });
  return response;
}

// Clear a server-set RUM marker cookie by writing an empty value with
// expires=epoch on the same path the set helper uses. Path discipline
// matters: clearing has to match the original Path attribute or the
// browser keeps the prior cookie alive.
export function clearRumMarkerCookie(response, marker) {
  const config = getConfig();
  response.cookies.set(marker.name, "", {
    httpOnly: false,
    secure: config.sessionCookieSecure,
    sameSite: "lax",
    path: marker.path,
    expires: new Date(0),
  });
  return response;
}

function _serialize(value) {
  // JSON.stringify is safe inside a <script> body provided we escape the two
  // sequences that can prematurely close the script tag or open an HTML
  // comment. Both substitutions preserve JSON validity for the embedded JS
  // parser.
  return JSON.stringify(value)
    .replace(/<\//g, "<\\/")
    .replace(/<!--/g, "<\\!--");
}

export function renderRumClientScript({ route, view, traceparent }) {
  const routeLiteral = _serialize(String(route));
  const viewLiteral = _serialize(String(view));
  const traceparentLiteral = _serialize(normalizeTraceparent(traceparent));
  const eventTypeLiteral = _serialize(`${String(view)}_rendered`);
  const loginSubmitBreadcrumbKeyLiteral = _serialize(LOGIN_SUBMIT_BREADCRUMB_KEY);
  const loginSubmitFailureMarkerCookieLiteral = _serialize(LOGIN_SUBMIT_FAILURE_MARKER_COOKIE);
  const loginSubmitFailureFreshnessMs = LOGIN_SUBMIT_FAILURE_MARKER_MAX_AGE_SECONDS * 1000;

  return `(function () {
  try {
    var ROUTE = ${routeLiteral};
    var VIEW = ${viewLiteral};
    var TRACEPARENT = ${traceparentLiteral};
    var RENDERED_EVENT = ${eventTypeLiteral};
    var ENDPOINT = "/v1/portal/rum";
    var LOGIN_SUBMIT_BREADCRUMB_KEY = ${loginSubmitBreadcrumbKeyLiteral};
    var LOGIN_SUBMIT_FAILURE_MARKER_COOKIE = ${loginSubmitFailureMarkerCookieLiteral};
    var LOGIN_SUBMIT_FAILURE_FRESHNESS_MS = ${loginSubmitFailureFreshnessMs};
    var emitted = Object.create(null);
    var vitals = { lcpMs: null, inpMs: null, clsScore: 0, finalized: false };
    var queue = [];
    // Anchor for the deepest Date.now() fallback path. Used only when
    // neither performance.now() NOR performance.timeOrigin is available
    // (very rare). The other two branches normalize against
    // performance.timeOrigin so durations from all paths share the same
    // navigation-start origin and stay comparable across runtimes.
    var SCRIPT_LOAD_EPOCH = Date.now();

    function nowMs() {
      try {
        if (window.performance && typeof window.performance.now === "function") {
          // Returns time elapsed since performance.timeOrigin
          // (navigation start). This is the canonical fast path.
          return window.performance.now();
        }
      } catch (_err) {
        // fall through
      }
      try {
        if (
          window.performance
          && typeof window.performance.timeOrigin === "number"
          && isFinite(window.performance.timeOrigin)
        ) {
          // Match the performance.now() branch's origin so values from
          // either runtime path can be aggregated together.
          return Math.max(0, Date.now() - window.performance.timeOrigin);
        }
      } catch (_err) {
        // fall through
      }
      try {
        // Deepest fallback: anchor at the IIFE's first execution. Origin
        // here is script-load rather than navigation-start, so values
        // observed via this branch will read ~10-200ms LOWER than the
        // performance.now() branch on the same wall-clock moment.
        return Math.max(0, Date.now() - SCRIPT_LOAD_EPOCH);
      } catch (_err) {
        return 0;
      }
    }

    function readCookieValue(name) {
      try {
        var prefix = name + "=";
        var parts = String(document.cookie || "").split(";");
        for (var i = 0; i < parts.length; i += 1) {
          var part = parts[i].replace(/^\\s+/, "");
          if (part.indexOf(prefix) === 0) {
            var raw = part.slice(prefix.length);
            try {
              return decodeURIComponent(raw);
            } catch (_decodeErr) {
              return raw;
            }
          }
        }
      } catch (_cookieErr) {
        // cookie access unavailable; treat as missing.
      }
      return "";
    }

    function clearCookieValue(name) {
      try {
        document.cookie = name + "=; Max-Age=0; Path=/login; SameSite=Lax";
      } catch (_cookieErr) {
        // best-effort removal
      }
    }

    function postSample(sample, keepalive) {
      try {
        var headers = { "Content-Type": "application/json" };
        if (TRACEPARENT) {
          headers["traceparent"] = TRACEPARENT;
        }
        var body = JSON.stringify({
          event_type: sample.event_type,
          route: ROUTE,
          view: VIEW,
          metric: sample.metric,
          value: sample.value,
          unit: sample.unit,
          metadata: sample.metadata || {}
        });
        return fetch(ENDPOINT, {
          method: "POST",
          headers: headers,
          body: body,
          keepalive: Boolean(keepalive),
          credentials: "same-origin"
        }).catch(function () { /* best-effort telemetry */ });
      } catch (_err) {
        // best-effort telemetry only
      }
    }

    function flushQueue(keepalive) {
      var pending = queue.splice(0, queue.length);
      for (var i = 0; i < pending.length; i += 1) {
        postSample(pending[i], keepalive);
      }
    }

    function enqueue(sample, options) {
      if (!sample || typeof sample.value !== "number" || !isFinite(sample.value) || sample.value < 0) {
        return;
      }
      queue.push(sample);
      flushQueue(Boolean(options && options.keepalive));
    }

    function emitOnce(eventType, sampleBuilder) {
      if (emitted[eventType]) return;
      emitted[eventType] = true;
      enqueue(sampleBuilder());
    }

    function emitRendered() {
      emitOnce(RENDERED_EVENT, function () {
        return {
          event_type: RENDERED_EVENT,
          metric: "duration",
          unit: "ms",
          value: nowMs()
        };
      });
    }

    function scheduleFirstViewInteractive() {
      if (emitted.first_view_interactive) return;
      var emit = function () {
        emitOnce("first_view_interactive", function () {
          return {
            event_type: "first_view_interactive",
            metric: "duration",
            unit: "ms",
            value: nowMs()
          };
        });
      };
      if (typeof window.requestAnimationFrame === "function") {
        window.requestAnimationFrame(emit);
      } else {
        window.setTimeout(emit, 0);
      }
    }

    function finalizeVitals() {
      if (vitals.finalized) return;
      vitals.finalized = true;
      if (typeof vitals.lcpMs === "number" && isFinite(vitals.lcpMs)) {
        enqueue({
          event_type: "core_web_vital",
          metric: "lcp",
          unit: "ms",
          value: vitals.lcpMs
        }, { keepalive: true });
      }
      if (typeof vitals.inpMs === "number" && isFinite(vitals.inpMs)) {
        enqueue({
          event_type: "core_web_vital",
          metric: "inp",
          unit: "ms",
          value: vitals.inpMs
        }, { keepalive: true });
      }
      enqueue({
        event_type: "core_web_vital",
        metric: "cls",
        unit: "score",
        value: Number(vitals.clsScore.toFixed(4))
      }, { keepalive: true });
    }

    function startObservers() {
      if (typeof window.PerformanceObserver !== "function") return;
      var supported = Array.isArray(window.PerformanceObserver.supportedEntryTypes)
        ? new Set(window.PerformanceObserver.supportedEntryTypes)
        : new Set();
      if (supported.has("largest-contentful-paint")) {
        try {
          var lcpObserver = new PerformanceObserver(function (list) {
            var entries = list.getEntries();
            var latest = entries[entries.length - 1];
            if (latest) {
              vitals.lcpMs = latest.startTime;
            }
          });
          lcpObserver.observe({ type: "largest-contentful-paint", buffered: true });
          window.addEventListener("pagehide", function () { lcpObserver.disconnect(); }, { once: true });
        } catch (_err) { /* observer unsupported */ }
      }
      if (supported.has("layout-shift")) {
        try {
          var clsObserver = new PerformanceObserver(function (list) {
            list.getEntries().forEach(function (entry) {
              if (!entry.hadRecentInput) {
                vitals.clsScore = Number((vitals.clsScore + entry.value).toFixed(4));
              }
            });
          });
          clsObserver.observe({ type: "layout-shift", buffered: true });
          window.addEventListener("pagehide", function () { clsObserver.disconnect(); }, { once: true });
        } catch (_err) { /* observer unsupported */ }
      }
      if (supported.has("event")) {
        try {
          var inpObserver = new PerformanceObserver(function (list) {
            list.getEntries().forEach(function (entry) {
              if (Number(entry.interactionId) > 0) {
                var current = Number(vitals.inpMs) || 0;
                vitals.inpMs = Math.max(current, entry.duration || 0);
              }
            });
          });
          inpObserver.observe({ type: "event", buffered: true, durationThreshold: 16 });
          window.addEventListener("pagehide", function () { inpObserver.disconnect(); }, { once: true });
        } catch (_err) { /* observer unsupported */ }
      }
    }

    function scheduleLoginSubmitListener() {
      // Browser-side counterpart to the server-side login_submit_attempt
      // (#1684). Only the login surface has a submission form; the landing
      // surface short-circuits here.
      //
      // Dedup contract for aggregators: the server-side handler also emits
      // login_submit_attempt with metric=count value=1 (no metadata.source).
      // This client emission carries metadata.source="client" so dashboards
      // can either (a) filter to one source for an authoritative count, or
      // (b) sum both sources for an "all observed submissions" view that
      // includes browser-only signals (closed tab, network failure before
      // request reaches server, HTML5-validated submissions that never
      // dispatch). Naive sum-by-event_type aggregations will double-count;
      // dashboards must filter by metadata.source to avoid that.
      if (VIEW !== "login") return;
      if (typeof document === "undefined") return;
      var form = null;
      try {
        form = document.querySelector('[data-ui="login-form"]');
      } catch (_err) {
        return;
      }
      if (!form || typeof form.addEventListener !== "function") return;
      // The submit event only fires AFTER native HTML5 validation passes,
      // so an unsubmitted form (empty required field, etc.) produces no
      // telemetry. We never call preventDefault — the form proceeds.
      form.addEventListener("submit", function () {
        try {
          var elapsedMs = Math.max(0, Math.round(nowMs()));
          enqueue({
            event_type: "login_submit_attempt",
            metric: "count",
            unit: "count",
            value: 1,
            metadata: {
              source: "client",
              duration_ms: elapsedMs
            }
          }, { keepalive: true });
          // Persist a Date.now() epoch-ms breadcrumb so the redirect
          // target (/login?error=<code> on failure) can compute the
          // submit-to-failure latency. Date.now() is cross-navigation
          // safe; performance.now() resets on each new page load and
          // would always read 0 on the receiving page.
          try {
            window.sessionStorage.setItem(
              LOGIN_SUBMIT_BREADCRUMB_KEY,
              String(Date.now())
            );
          } catch (_storageErr) {
            // sessionStorage unavailable (private mode / blocked);
            // silently no-op so the failure mirror simply does not
            // emit on this submission.
          }
        } catch (_err) {
          // best-effort telemetry only
        }
      }, { once: true });
    }

    function scheduleLoginSubmitFailureListener() {
      // Browser-side counterpart to the server-side login_submit_failure
      // event (#1684). Emits when /login is loaded with a redirect-back
      // ?error=<code> AND a fresh sessionStorage breadcrumb (written by
      // scheduleLoginSubmitListener at submit time) proves we just
      // submitted from this tab. Manual visits to /login?error=… and
      // stale browser-back state therefore do NOT produce telemetry.
      //
      // Dedup contract: server-side already emits login_submit_failure
      // with metric=duration. This client emission carries
      // metadata.source="client" so dashboards can filter to one source
      // for an authoritative count or sum both for an "all observed
      // failures" view that includes browser-only signals (closed tab,
      // network drop after submit). Naive sum-by-event_type aggregations
      // will double-count; dashboards must filter by metadata.source.
      if (VIEW !== "login") return;
      if (typeof window === "undefined") return;
      // Always read AND clear the breadcrumb on every /login load,
      // regardless of whether we ultimately emit. Keeps stale entries
      // from a closed-tab submit out of a future visitor's session.
      var rawStart = null;
      try {
        rawStart = window.sessionStorage.getItem(LOGIN_SUBMIT_BREADCRUMB_KEY);
      } catch (_storageErr) {
        // sessionStorage unavailable; bail without emitting.
        return;
      }
      try {
        window.sessionStorage.removeItem(LOGIN_SUBMIT_BREADCRUMB_KEY);
      } catch (_storageErr) {
        // best-effort removal
      }
      var failureMarker = readCookieValue(LOGIN_SUBMIT_FAILURE_MARKER_COOKIE);
      clearCookieValue(LOGIN_SUBMIT_FAILURE_MARKER_COOKIE);
      if (!rawStart || !failureMarker) return;
      var errorCode = "";
      try {
        var url = new URL(window.location.href);
        errorCode = String(url.searchParams.get("error") || "")
          .trim()
          .toLowerCase();
      } catch (_urlErr) {
        return;
      }
      if (!errorCode) return;
      // Mirrors LOGIN_RUM_FAILURE_CODES at lib/rum-emitter.js. A drift
      // test pins both lists.
      var allowed = ["csrf", "configuration", "access", "throttled", "invalid"];
      var matched = false;
      for (var i = 0; i < allowed.length; i += 1) {
        if (allowed[i] === errorCode) {
          matched = true;
          break;
        }
      }
      if (!matched) return;
      if (failureMarker !== errorCode) return;
      var startedAt = Number(rawStart);
      if (!isFinite(startedAt) || startedAt <= 0) return;
      var elapsedMs = Math.max(0, Math.round(Date.now() - startedAt));
      // Freshness cap: production submit→failure latency is sub-second,
      // so 60s catches network drops while excluding stale-back-button
      // false positives.
      if (elapsedMs > LOGIN_SUBMIT_FAILURE_FRESHNESS_MS) return;
      try {
        enqueue({
          event_type: "login_submit_failure",
          metric: "duration",
          unit: "ms",
          value: elapsedMs,
          metadata: {
            source: "client",
            failure_code: errorCode
          }
        }, { keepalive: true });
      } catch (_err) {
        // best-effort telemetry only
      }
    }

    function bootstrap() {
      startObservers();
      emitRendered();
      scheduleFirstViewInteractive();
      scheduleLoginSubmitListener();
      scheduleLoginSubmitFailureListener();
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", bootstrap, { once: true });
    } else {
      bootstrap();
    }

    window.addEventListener("pagehide", function () {
      finalizeVitals();
      flushQueue(true);
    }, { once: true });
  } catch (_err) {
    // RUM is best-effort; never block the page on telemetry errors.
  }
})();`;
}
