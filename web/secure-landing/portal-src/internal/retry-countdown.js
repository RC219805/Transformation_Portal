// Live countdown helper for the rate-limit retry banners.
//
// Pure JavaScript; no DOM access. Each tick computes the seconds remaining
// until ``retryAtMs`` and invokes ``onTick(secondsRemaining)``. Once the
// remaining seconds reach zero the helper invokes ``onComplete()`` and stops.
//
// The clock and interval scheduler are injectable so unit tests run without
// real timers. Default: ``Date.now`` and ``setInterval``/``clearInterval``.
//
// Caller responsibilities:
//   - Append the countdown DOM element via ``createElement`` (never inline
//     HTML strings); the helper only touches state via callbacks.
//   - Cancel the countdown when the banner is replaced or retry succeeds by
//     calling the returned ``cancel()`` function.
//
// Lifecycle guarantees:
//   - ``cancel()`` is idempotent: subsequent calls are no-ops.
//   - When ``retryAtMs`` is missing, NaN, zero, negative, or already in the
//     past, the helper invokes ``onComplete`` once (if provided) and returns
//     a no-op cancel without starting an interval.
//   - When ``onTick`` or ``onComplete`` throws, the error is swallowed so
//     a render-callback failure cannot stop subsequent ticks.

export function formatRetryCountdown(secondsRemaining) {
  const seconds = Math.max(0, Math.floor(Number(secondsRemaining) || 0));
  if (seconds <= 0) {
    return "Retrying now…";
  }
  return `Retry in ${seconds}s`;
}

export function startRetryCountdown({
  retryAtMs,
  onTick,
  onComplete,
  intervalMs = 1000,
  now = Date.now,
  scheduleInterval = (fn, ms) => setInterval(fn, ms),
  clearScheduledInterval = (handle) => clearInterval(handle),
} = {}) {
  const _safe = (fn, arg) => {
    if (typeof fn !== "function") return;
    try { fn(arg); } catch (_err) { /* best-effort */ }
  };
  const target = Number(retryAtMs);
  if (!Number.isFinite(target) || target <= 0 || target <= now()) {
    _safe(onComplete);
    return () => {};
  }

  let cancelled = false;
  let handle = null;

  const stop = () => {
    if (cancelled) return;
    cancelled = true;
    if (handle !== null) {
      _safe(clearScheduledInterval, handle);
      handle = null;
    }
  };

  const tick = () => {
    if (cancelled) return;
    const remainingSeconds = Math.ceil(Math.max(0, target - now()) / 1000);
    if (remainingSeconds <= 0) {
      stop();
      _safe(onTick, 0);
      _safe(onComplete);
      return;
    }
    _safe(onTick, remainingSeconds);
  };

  // Synchronous first emission so the placeholder reflects the current
  // remaining time without waiting an interval boundary.
  tick();
  if (cancelled) return stop;
  handle = scheduleInterval(tick, intervalMs);
  return stop;
}
