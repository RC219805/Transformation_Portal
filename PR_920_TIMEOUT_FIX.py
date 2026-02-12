"""Proposed fix for PR #920 barrier timeout silent degradation.

ISSUE:
  If _ready.wait(timeout=...) expires, __enter__ returns without error.
  This resurrects the race condition the PR is trying to fix.

SOLUTION:
  Check wait() return value and raise RuntimeError on timeout.

IMPACT:
  - 7 lines of code
  - No new dependencies
  - Explicit failure mode (better than silent degradation)
  - Extremely unlikely to trigger (timeout is adaptive and generous)

STATUS:
  Architect-approved as REQUIRED for merge.
"""

# ==============================================================================
# CURRENT CODE (from origin/copilot/update-readme-benchmark-policy)
# ==============================================================================


def __enter__(self):
    self.peak_rss_bytes = self.process.memory_info().rss
    self.samples = 0
    self._stop.clear()
    self._ready.clear()
    self._thread = threading.Thread(target=self._poll, daemon=True)
    self._thread.start()
    self._ready.wait(timeout=max(0.05, self.interval * 10))  # <- Returns even if timeout!
    return self


# ==============================================================================
# PROPOSED FIX
# ==============================================================================


def __enter__(self):
    self.peak_rss_bytes = self.process.memory_info().rss
    self.samples = 0
    self._stop.clear()
    self._ready.clear()
    self._thread = threading.Thread(target=self._poll, daemon=True)
    self._thread.start()

    # Barrier timeout check - fail loudly if first sample takes too long
    timeout = max(0.05, self.interval * 10)
    if not self._ready.wait(timeout=timeout):
        # Timeout expired - clean up thread and raise error
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        raise RuntimeError(
            f"PeakRSSTracker first-sample barrier timed out after {timeout}s. "
            f"System may be under extreme load; measurement would be invalid."
        )

    return self


# ==============================================================================
# DIFF
# ==============================================================================

"""
 def __enter__(self):
     self.peak_rss_bytes = self.process.memory_info().rss
     self.samples = 0
     self._stop.clear()
     self._ready.clear()
     self._thread = threading.Thread(target=self._poll, daemon=True)
     self._thread.start()
-    self._ready.wait(timeout=max(0.05, self.interval * 10))
+
+    # Barrier timeout check - fail loudly if first sample takes too long
+    timeout = max(0.05, self.interval * 10)
+    if not self._ready.wait(timeout=timeout):
+        # Timeout expired - clean up thread and raise error
+        self._stop.set()
+        if self._thread is not None:
+            self._thread.join(timeout=1.0)
+        raise RuntimeError(
+            f"PeakRSSTracker first-sample barrier timed out after {timeout}s. "
+            f"System may be under extreme load; measurement would be invalid."
+        )
+
     return self
"""


# ==============================================================================
# JUSTIFICATION
# ==============================================================================

"""
Why this is required:

1. CURRENT BEHAVIOR IS INCORRECT
   - Timeout expiry defeats the race condition fix
   - PR claims to guarantee first sample before workload
   - Without check, guarantee is violated under load

2. SILENT FAILURES ARE UNACCEPTABLE IN BENCHMARKS
   - Invalid measurements poison baseline database
   - False positives/negatives erode developer trust
   - Architectural principle: fail fast and loud

3. TIMEOUT IS UNLIKELY TO TRIGGER
   - First sample typically completes in <1ms
   - Timeout is 50ms minimum (5000% safety margin)
   - Only triggers under pathological system load
   - If system is that loaded, benchmark is already invalid

4. EXPLICIT FAILURE IS BETTER THAN DEGRADATION
   - Clear error message explains what happened
   - CI will fail (correct behavior - load too high)
   - Prevents invalid data from entering baseline
   - Actionable: "System under extreme load"

5. MINIMAL COMPLEXITY
   - 7 lines of code
   - Standard Python idiom (check Event.wait() return value)
   - No new dependencies
   - Obvious correctness

ALTERNATIVES CONSIDERED:

A) Log warning instead of raising
   - Pro: Test doesn't fail
   - Con: Silent degradation still possible
   - Con: Log may not be noticed
   - REJECTED: Benchmarks must fail loudly on invalid data

B) Do nothing (accept silent degradation)
   - Pro: Zero code change
   - Con: Race condition returns under load
   - Con: Violates fix's stated guarantee
   - REJECTED: Defeats purpose of the PR

C) Increase timeout to infinity
   - Pro: Never times out
   - Con: CI can hang indefinitely
   - Con: Doesn't address root issue (system overload)
   - REJECTED: Creates different failure mode (hangs)

DECISION: Option from proposal (raise RuntimeError on timeout)
"""


# ==============================================================================
# TESTING
# ==============================================================================

"""
Test case that demonstrates the issue:

    def test_timeout_behavior():
        process = psutil.Process(os.getpid())

        # Monkey-patch to simulate slow first sample
        original_sample = PeakRSSTracker._sample
        def slow_sample(self):
            time.sleep(0.2)  # Longer than timeout
            original_sample(self)

        with patch.object(PeakRSSTracker, '_sample', slow_sample):
            # CURRENT CODE: No error raised
            with PeakRSSTracker(process) as tracker:
                pass  # Workload may start before first sample!

            # PROPOSED FIX: RuntimeError raised
            # (prevents invalid measurement)

Expected behavior after fix:
    RuntimeError: PeakRSSTracker first-sample barrier timed out after 0.05s.
                  System may be under extreme load; measurement would be invalid.
"""


# ==============================================================================
# DEPLOYMENT
# ==============================================================================

"""
1. Apply diff to tests/benchmarks/test_lux_depth_v3_perf_smoke.py
2. Run test suite: pytest tests/benchmarks/ -v
3. Verify CI passes
4. Merge PR #920
5. Invalidate existing baseline artifacts
6. Recapture baselines with correct semantics

FILE: tests/benchmarks/test_lux_depth_v3_perf_smoke.py
CLASS: PeakRSSTracker
METHOD: __enter__
LINES: ~169-172 (current), ~169-182 (after fix)
"""
