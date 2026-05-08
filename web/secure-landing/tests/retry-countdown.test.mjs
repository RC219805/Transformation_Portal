// Unit tests for the retry-countdown helper used by the portal's rate-limit
// banners. The helper has no DOM dependency; tests inject the clock and
// interval scheduler so timing is deterministic.

import test from "node:test";
import assert from "node:assert/strict";

import { formatRetryCountdown, startRetryCountdown } from "../portal-src/internal/retry-countdown.js";

function _stubClock(initialMs) {
  let nowMs = initialMs;
  const handles = new Map();
  let nextId = 1;
  // Advance the virtual clock to ``targetMs`` while firing scheduled intervals
  // in fire-event order. Each tick sees the virtual clock pinned at its own
  // scheduled fire time, not the final target — matching how a real timer
  // observes Date.now() at the moment its callback runs.
  const advance = (deltaMs) => {
    const targetMs = nowMs + deltaMs;
    while (true) {
      let earliest = null;
      for (const [id, entry] of handles) {
        if (entry.nextFireAt <= targetMs && (earliest === null || entry.nextFireAt < earliest.entry.nextFireAt)) {
          earliest = { id, entry };
        }
      }
      if (earliest === null) break;
      nowMs = earliest.entry.nextFireAt;
      earliest.entry.fn();
      if (handles.has(earliest.id)) {
        earliest.entry.nextFireAt += earliest.entry.intervalMs;
      }
    }
    nowMs = targetMs;
  };
  return {
    now: () => nowMs,
    scheduleInterval: (fn, intervalMs) => {
      const id = nextId;
      nextId += 1;
      handles.set(id, { fn, intervalMs, nextFireAt: nowMs + intervalMs });
      return id;
    },
    clearScheduledInterval: (handle) => {
      handles.delete(handle);
    },
    advance,
    handleCount: () => handles.size,
  };
}

test("formatRetryCountdown renders compact seconds and pivots to 'Retrying now…' at zero", () => {
  assert.equal(formatRetryCountdown(62), "Retry in 62s");
  assert.equal(formatRetryCountdown(1), "Retry in 1s");
  assert.equal(formatRetryCountdown(0), "Retrying now…");
  assert.equal(formatRetryCountdown(-3), "Retrying now…");
  assert.equal(formatRetryCountdown(Number.NaN), "Retrying now…");
  assert.equal(formatRetryCountdown(undefined), "Retrying now…");
  // Fractional seconds floor toward integer steps.
  assert.equal(formatRetryCountdown(5.9), "Retry in 5s");
});

test("startRetryCountdown emits a synchronous first tick at the supplied retryAtMs", () => {
  const clock = _stubClock(1000);
  const ticks = [];
  const completes = [];
  const cancel = startRetryCountdown({
    retryAtMs: 1000 + 5000,
    onTick: (s) => ticks.push(s),
    onComplete: () => completes.push(true),
    intervalMs: 1000,
    now: clock.now,
    scheduleInterval: clock.scheduleInterval,
    clearScheduledInterval: clock.clearScheduledInterval,
  });

  assert.deepEqual(ticks, [5], "first tick should fire synchronously with the initial seconds remaining");
  assert.deepEqual(completes, []);

  clock.advance(1000);
  clock.advance(1000);
  clock.advance(1000);
  // After 3 seconds, the interval has fired three times.
  assert.deepEqual(ticks.slice(1), [4, 3, 2]);

  clock.advance(2000);
  // Crossed zero — onComplete fires once and the interval is cleared.
  assert.deepEqual(ticks.slice(-2), [1, 0]);
  assert.deepEqual(completes, [true]);
  assert.equal(clock.handleCount(), 0, "interval should be cleared after onComplete");

  cancel(); // idempotent no-op
  cancel();
});

test("cancel() returned by startRetryCountdown halts further ticks", () => {
  const clock = _stubClock(0);
  const ticks = [];
  const completes = [];
  const cancel = startRetryCountdown({
    retryAtMs: 10000,
    onTick: (s) => ticks.push(s),
    onComplete: () => completes.push(true),
    intervalMs: 1000,
    now: clock.now,
    scheduleInterval: clock.scheduleInterval,
    clearScheduledInterval: clock.clearScheduledInterval,
  });

  assert.deepEqual(ticks, [10]);
  clock.advance(2000);
  assert.deepEqual(ticks.slice(1), [9, 8]);

  cancel();
  assert.equal(clock.handleCount(), 0);

  // Further clock advances must not push new ticks.
  clock.advance(20000);
  assert.deepEqual(ticks, [10, 9, 8]);
  assert.deepEqual(completes, []);
});

test("startRetryCountdown completes immediately when retryAtMs is missing, past, or invalid", () => {
  const clock = _stubClock(1_000_000);
  const records = [];

  const completeRecorder = (label) => {
    records.push(label);
  };

  // Missing
  startRetryCountdown({
    retryAtMs: undefined,
    onTick: () => assert.fail("missing retryAtMs should not tick"),
    onComplete: () => completeRecorder("missing"),
    now: clock.now,
    scheduleInterval: clock.scheduleInterval,
    clearScheduledInterval: clock.clearScheduledInterval,
  });

  // Past
  startRetryCountdown({
    retryAtMs: 500_000,
    onTick: () => assert.fail("past retryAtMs should not tick"),
    onComplete: () => completeRecorder("past"),
    now: clock.now,
    scheduleInterval: clock.scheduleInterval,
    clearScheduledInterval: clock.clearScheduledInterval,
  });

  // Zero
  startRetryCountdown({
    retryAtMs: 0,
    onTick: () => assert.fail("zero retryAtMs should not tick"),
    onComplete: () => completeRecorder("zero"),
    now: clock.now,
    scheduleInterval: clock.scheduleInterval,
    clearScheduledInterval: clock.clearScheduledInterval,
  });

  // NaN
  startRetryCountdown({
    retryAtMs: Number.NaN,
    onTick: () => assert.fail("NaN retryAtMs should not tick"),
    onComplete: () => completeRecorder("nan"),
    now: clock.now,
    scheduleInterval: clock.scheduleInterval,
    clearScheduledInterval: clock.clearScheduledInterval,
  });

  assert.deepEqual(records, ["missing", "past", "zero", "nan"]);
  assert.equal(clock.handleCount(), 0, "no intervals should be scheduled for short-circuit cases");
});

test("startRetryCountdown swallows errors thrown by onTick or onComplete callbacks", () => {
  const clock = _stubClock(0);
  const tickCalls = [];
  const completeCalls = [];
  const cancel = startRetryCountdown({
    retryAtMs: 2000,
    onTick: (s) => {
      tickCalls.push(s);
      throw new Error("simulated render failure");
    },
    onComplete: () => {
      completeCalls.push(true);
      throw new Error("simulated cleanup failure");
    },
    intervalMs: 1000,
    now: clock.now,
    scheduleInterval: clock.scheduleInterval,
    clearScheduledInterval: clock.clearScheduledInterval,
  });

  // Synchronous first tick threw, but the helper kept the interval alive.
  assert.deepEqual(tickCalls, [2]);
  assert.equal(clock.handleCount(), 1, "interval must still be scheduled after a thrown onTick");

  // Subsequent ticks continue to fire even though every callback throws.
  clock.advance(1000);
  clock.advance(1000);
  assert.deepEqual(tickCalls, [2, 1, 0]);
  assert.deepEqual(completeCalls, [true]);
  assert.equal(clock.handleCount(), 0, "interval cleared after onComplete (even though it threw)");

  cancel();
});
