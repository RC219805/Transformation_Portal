from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


@dataclass
class StageProfiler:
    """
    Minimal-overhead stage timing collector.

    Designed to match the "optional profiling <5% overhead" requirement.
    """
    enabled: bool = True
    stages_ms: Dict[str, float] = field(default_factory=dict)
    _stack: list[tuple[str, float]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        self._stack.append((name, t0))
        try:
            yield
        finally:
            _, start = self._stack.pop()
            dt_ms = (time.perf_counter() - start) * 1000.0
            self.stages_ms[name] = self.stages_ms.get(name, 0.0) + dt_ms

    def summary(self) -> Dict[str, float]:
        return dict(sorted(self.stages_ms.items(), key=lambda kv: kv[0]))
