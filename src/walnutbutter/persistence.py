"""Saving and restoring what the network has learned.

A checkpoint is a JSON file holding every connection's weight (by ID) together
with what is needed to rebuild the identical mesh: size, omega, threshold,
seed and the input permutation. Loading builds the mesh from those settings
and copies the weights back in, so a run that took hours can be continued or
inspected later.
"""

from __future__ import annotations

import json
from pathlib import Path

from .cartesian import CartesianNodes
from .grid import GridOfNeurons

FORMAT = 1


def checkpoint(grid: GridOfNeurons, path: str | Path, teacher=None) -> dict:
    """Write the grid's weights and settings to `path`. Returns what was written."""
    lattice = isinstance(grid, CartesianNodes)
    data = {
        "format": FORMAT,
        "container": "lattice" if lattice else "grid",
        "columns": grid.columns,
        "rows": grid.rows,
        "omega": getattr(grid, "omega", 0.0),
        "threshold": grid.threshold,
        "minimum_potential": grid.minimum_potential,
        "seed": grid.seed,
        "weight": getattr(grid, "weight", None),
        "weight_range": list(grid.weight_range),
        "permutation": grid.permutation,
        "ecc": grid.ecc,
        "random_weights": grid.weight is None if not lattice else True,
        "epoch": grid.epoch,
        "connections": len(grid.connections),
        # the shortcuts are the only random part of a grid's topology: record them so a load can verify the mesh
        "shortcuts": [] if lattice else [[c.source.name, c.target.name] for c in grid.small_world_connections()],
        "weights": [grid.connections[i].weight for i in range(1, len(grid.connections) + 1)],
        "thresholds": [n.threshold for n in grid.all_neurons()],
        "rates": [n.rate for n in grid.all_neurons()],
    }
    if lattice:
        index = {n: i for i, n in enumerate(grid.neurons)}
        data["layout"] = grid.layout
        data["reach"] = getattr(grid, "reach", None)
        data["receptive_field_sigma"] = getattr(grid, "receptive_field_sigma", None)  # the earlier Gaussian rule, if used
        data["positions"] = grid.positions()
        # the whole wiring, so a restore rebuilds it exactly without redrawing anything
        data["connection_list"] = [
            [index[c.source], index[c.target], c.kind] for c in (grid.connections[i] for i in range(1, len(grid.connections) + 1))
        ]
    if teacher is not None:
        data["learning"] = {
            "target": teacher.target,
            "lr": teacher.lr,
            "sigma": teacher.sigma,
            "eligibility": teacher.eligibility,
            "epochs": teacher.epochs,
            "homeostasis": teacher.homeostasis,
            "target_rate": teacher.target_rate,
            "threshold_range": list(teacher.threshold_range),
            "unstick": teacher.unstick,
            "unstick_target": teacher.unstick_target,
            "critic": teacher.critic,
            "history": teacher.history,
            "total_reward": teacher.total_reward,
            "baseline": teacher.baseline,
            "average": teacher.average,
        }
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half checkpoint
    return data


def read_checkpoint(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {data.get('format')!r}")
    return data


def restore(path: str | Path) -> tuple[GridOfNeurons, dict]:
    """Rebuild the mesh described by the checkpoint and load its weights into it.

    Returns the grid and the checkpoint data (so a Teacher can be resumed).
    """
    data = read_checkpoint(path)
    if data.get("container") == "lattice":
        return _restore_lattice(data), data
    if data["seed"] is None:
        raise ValueError(f"{path}: the mesh was built without a seed, so its shortcuts cannot be rebuilt")
    grid = GridOfNeurons(
        columns=data["columns"],
        rows=data["rows"],
        weight=None if data["random_weights"] else data["weight"],
        threshold=data["threshold"],
        seed=data["seed"],
        omega=data["omega"],
        permute=False,
        weight_range=tuple(data.get("weight_range", (-1.0, 1.0))),
        minimum_potential=data.get("minimum_potential", float("-inf")),  # older checkpoints had no floor
    )
    grid.permutation = list(data["permutation"])
    grid.ecc = _ecc_name(data)
    load_weights(grid, data)
    grid.epoch = data["epoch"]
    return grid, data


def load_weights(grid: GridOfNeurons, data: dict) -> None:
    """Copy a checkpoint's weights into `grid`, which must have the same mesh."""
    for key in ("columns", "rows", "omega", "seed"):
        if getattr(grid, key) != data[key]:
            raise ValueError(f"checkpoint {key} is {data[key]!r} but the mesh has {getattr(grid, key)!r}")
    if len(grid.connections) != data["connections"] or len(data["weights"]) != data["connections"]:
        raise ValueError(
            f"checkpoint has {data['connections']} connections but the mesh has {len(grid.connections)}"
        )
    shortcuts = [[c.source.name, c.target.name] for c in grid.small_world_connections()]
    if shortcuts != data["shortcuts"]:
        raise ValueError("checkpoint shortcuts differ from the mesh's: it was built from a different seed")
    for connection_id, weight in enumerate(data["weights"], start=1):
        grid.connections[connection_id].weight = weight
    neurons = list(grid.all_neurons())
    for neuron, threshold in zip(neurons, data.get("thresholds", [])):
        neuron.threshold = threshold
    for neuron, rate in zip(neurons, data.get("rates", [])):
        neuron.rate = rate


def resume_teacher(teacher, data: dict) -> None:
    """Continue a Teacher's running statistics from a checkpoint's learning record, if any."""
    record = data.get("learning")
    if not record:
        return
    teacher.epochs = record["epochs"]
    teacher.total_reward = record["total_reward"]
    teacher.baseline = record["baseline"]
    teacher.average = record["average"]
    teacher.history = list(record.get("history", []))


def _restore_lattice(data: dict) -> CartesianNodes:
    """Rebuild a CartesianNodes network from its checkpoint: positions, wiring, weights, thresholds."""
    nodes = CartesianNodes(
        columns=data["columns"],
        rows=data["rows"],
        layout=data.get("layout", "hex"),
        count=0 if data.get("layout") == "random" else None,
        seed=data["seed"],
        threshold=data["threshold"],
        minimum_potential=data.get("minimum_potential", -1.0),
        permute=False,
        weight_range=tuple(data.get("weight_range", (-1.0, 1.0))),
    )
    if data.get("layout") == "random":
        for x, y in data["positions"]:
            nodes.add(x, y)
    nodes.permutation = list(data["permutation"])
    nodes.ecc = _ecc_name(data)
    nodes.receptive_field_sigma = data.get("receptive_field_sigma")
    nodes.reach = data.get("reach")
    neurons = nodes.neurons
    for connection_id, ((s_idx, t_idx, kind), weight) in enumerate(zip(data["connection_list"], data["weights"]), start=1):
        nodes.connections[connection_id] = neurons[s_idx].connect(neurons[t_idx], connection_id, weight, kind=kind)
    for neuron, threshold in zip(neurons, data.get("thresholds", [])):
        neuron.threshold = threshold
    for neuron, rate in zip(neurons, data.get("rates", [])):
        neuron.rate = rate
    nodes.epoch = data["epoch"]
    return nodes


def _ecc_name(data: dict) -> str | None:
    value = data.get("ecc")
    if value is True:
        return "parity64"  # written when ecc was a flag and meant the (6, 4) code
    return value or None
