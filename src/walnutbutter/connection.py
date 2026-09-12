"""A Connection is a one-way, weighted pathway from one neuron to another.

It stores references to both neurons. In Python every variable holds a
reference (the equivalent of a pointer), so no neuron is ever copied: the
grid, the neurons' own lists, and this object all point at the same Neuron.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type hints; avoids a circular import at runtime
    from .neuron import Neuron


class Connection:
    def __init__(
        self,
        connection_id: int,
        source: Neuron,
        target: Neuron,
        weight: float = 1.0,
        is_active: bool = True,
        kind: str = "local",
    ):
        if source is target:
            raise ValueError(f"{source.name} cannot connect to itself")
        self.id = connection_id
        self.source = source  # signals travel from here...
        self.target = target  # ...to here, never the other way
        self.weight = float(weight)
        self.is_active = is_active
        self.kind = kind  # "local" (a neighbour), "local2" (a neighbour of a neighbour), "small_world" (a shortcut)
        self.last_signal: float | None = None  # clock time of the last signal the target integrated along here: the synapse's only trace

    def joins(self, source: Neuron, target: Neuron) -> bool:
        """True if this connection runs from `source` to `target` (direction matters)."""
        return source is self.source and target is self.target

    def __repr__(self) -> str:
        state = "active" if self.is_active else "inactive"
        kind = "" if self.kind == "local" else f", {self.kind}"
        return f"Connection({self.id}: {self.source.name} -> {self.target.name}, weight {self.weight:g}, {state}{kind})"
