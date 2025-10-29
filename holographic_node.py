"""Pragmatic implementation of the "holographic" processing concept.

The original project description referenced a :class:`HolographicNode` that
"saw" the entire system at once through metaphysical constructs such as
``GlobalSystemState`` and ``EntanglementField``.  The production tooling in
this repository expects concrete, testable behaviour instead of abstract
metaphors, so this module provides lightweight implementations that capture
the intended ideas in plain Python.

* ``GlobalSystemState`` keeps a dictionary of shared values and allows callers
  to subscribe to change notifications.
* ``EntanglementField`` turns a set of candidate transforms into a
  :class:`Superposition`, essentially a scored list of possible outputs.
* ``HolographicNode`` orchestrates those pieces to process an input tensor by
  either running a full pipeline of transforms or collapsing the superposition
  according to an ``observer`` function (called ``client_aesthetic_profile`` in
  the repository's terminology).

These classes are intentionally small and dependency-free so they can be used
from both production scripts and unit tests without additional infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional


Observer = Callable[[Any], float]
Transform = Callable[[Any], Any]


class GlobalSystemState:
    """Store shared state and notify interested observers when it changes."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._observers: List[Callable[[Mapping[str, Any]], None]] = []

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* and broadcast the updated state."""

        self._state[key] = value
        snapshot = self.snapshot()
        for observer in list(self._observers):
            observer(snapshot)

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return the stored value for *key* or *default* when absent."""

        return self._state.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the current state."""

        return dict(self._state)

    def subscribe(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        """Register *callback* to be invoked whenever the state changes."""

        if callback not in self._observers:
            self._observers.append(callback)

    def unsubscribe(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        """Remove *callback* if it was previously registered."""

        try:
            self._observers.remove(callback)
        except ValueError:
            pass


@dataclass(frozen=True)
class Superposition:
    """Collection of candidate outcomes produced by multiple transforms."""

    source: Any
    outcomes: List[tuple[Transform, Any]]

    def collapse(self, observer: Optional[Observer] = None) -> Any:
        """Select and return the best candidate according to *observer*.

        If *observer* is ``None`` the first outcome is returned.  When no
        outcomes are available the original ``source`` is returned unchanged.
        """

        if not self.outcomes:
            return self.source

        if observer is None:
            return self.outcomes[0][1]

        best_score = float("-inf")
        best_value: Any = self.outcomes[0][1]
        for transform, value in self.outcomes:
            score = observer(value)
            if score > best_score:
                best_score = score
                best_value = value
        return best_value


class EntanglementField:
    """Generate :class:`Superposition` objects from candidate transforms."""

    def create_superposition(
        self, input_tensor: Any, transforms: Iterable[Transform]
    ) -> Superposition:
        outcomes: List[tuple[Transform, Any]] = []
        current = input_tensor
        for transform in transforms:
            current = transform(current)
            outcomes.append((transform, current))
        return Superposition(source=input_tensor, outcomes=outcomes)


class HolographicNode:
    """Process inputs using a pool of transforms and a scoring function."""

    def __init__(
        self,
        transforms: Optional[Iterable[Transform]] = None,
        *,
        client_aesthetic_profile: Optional[Observer] = None,
    ) -> None:
        self.local_state: Dict[str, Any] = {}
        self.global_consciousness = GlobalSystemState()
        self.quantum_field = EntanglementField()
        self._transforms: List[Transform] = list(transforms or [])
        self.client_aesthetic_profile: Observer = (
            client_aesthetic_profile if client_aesthetic_profile is not None else lambda _: 0.0
        )

    def add_transform(self, transform: Transform) -> None:
        """Register an additional candidate transform."""

        self._transforms.append(transform)
        self.local_state["expanded"] = False

    def enumerate_possible_transforms(self) -> Iterator[Transform]:
        """Yield the currently registered transforms."""

        yield from self._transforms

    def requires_expansion(self) -> bool:
        """Return ``True`` when the node should run the full pipeline."""

        return not self.local_state.get("expanded", False)

    def spawn_full_pipeline(self, input_tensor: Any) -> Any:
        """Run the transforms sequentially and mark the node as expanded."""

        result = input_tensor
        for transform in self._transforms:
            result = transform(result)
        self.local_state["expanded"] = True
        self.global_consciousness.set("last_result", result)
        return result

    def process(self, input_tensor: Any) -> Any:
        """Process *input_tensor* using either the full pipeline or scoring."""

        if self.requires_expansion():
            return self.spawn_full_pipeline(input_tensor)

        superposition = self.quantum_field.create_superposition(
            input_tensor, self.enumerate_possible_transforms()
        )
        return superposition.collapse(observer=self.client_aesthetic_profile)


__all__ = [
    "EntanglementField",
    "GlobalSystemState",
    "HolographicNode",
    "Superposition",
]
