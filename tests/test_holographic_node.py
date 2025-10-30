from __future__ import annotations

import math


from holographic_node import (
    EntanglementField,
    GlobalSystemState,
    HolographicNode,
    Superposition,
)


def test_global_state_notifies_subscribers():
    state = GlobalSystemState()
    snapshots: list[dict[str, int]] = []

    state.subscribe(lambda snapshot: snapshots.append(dict(snapshot)))
    state.set("exposure", 2)
    state.set("contrast", 3)

    assert snapshots == [
        {"exposure": 2},
        {"exposure": 2, "contrast": 3},
    ]


def test_superposition_collapse_without_observer_returns_first_outcome():
    sup = Superposition(source="raw", outcomes=[(lambda value: value, "primary"), (lambda v: v, "secondary")])

    assert sup.collapse() == "primary"


def test_entanglement_field_evaluates_all_transforms():
    field = EntanglementField()
    calls: list[int] = []

    def _double(value: int) -> int:
        calls.append(value)
        return value * 2

    superposition = field.create_superposition(3, [_double])

    assert superposition.outcomes[0][1] == 6
    assert calls == [3]


def test_holographic_node_pipeline_and_scoring():
    transforms = [
        lambda value: value + 1,
        lambda value: value * 10,
    ]

    def score(value: int) -> float:
        return -math.fabs(value - 25)

    node = HolographicNode(transforms=transforms, client_aesthetic_profile=score)

    # First call runs the sequential pipeline.
    assert node.process(1) == 20
    assert node.global_consciousness.get("last_result") == 20

    # Subsequent calls collapse the superposition using the scoring function.
    assert node.process(2) == 30


def test_requires_expansion_when_transforms_added():
    node = HolographicNode(transforms=[lambda value: value + 1])
    node.process(0)

    node.add_transform(lambda value: value * 2)

    assert node.requires_expansion() is True
