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

  return `(function () {
  try {
    var ROUTE = ${routeLiteral};
    var VIEW = ${viewLiteral};
    var TRACEPARENT = ${traceparentLiteral};
    var RENDERED_EVENT = ${eventTypeLiteral};
    var ENDPOINT = "/v1/portal/rum";
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
        } catch (_err) {
          // best-effort telemetry only
        }
      }, { once: true });
    }

    function bootstrap() {
      startObservers();
      emitRendered();
      scheduleFirstViewInteractive();
      scheduleLoginSubmitListener();
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
