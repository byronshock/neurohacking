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

from .grid import GridOfNeurons

FORMAT = 1


def checkpoint(grid: GridOfNeurons, path: str | Path, teacher=None) -> dict:
    """Write the grid's weights and settings to `path`. Returns what was written."""
    data = {
        "format": FORMAT,
        "columns": grid.columns,
        "rows": grid.rows,
        "omega": grid.omega,
        "threshold": grid.threshold,
        "minimum_potential": grid.minimum_potential,
        "seed": grid.seed,
        "random_weights": grid.weight is None,
        "weight": grid.weight,
        "weight_range": list(grid.weight_range),
        "permutation": grid.permutation,
        "epoch": grid.epoch,
        "connections": len(grid.connections),
        # the shortcuts are the only random part of the topology: record them so a load can verify the mesh
        "shortcuts": [[c.source.name, c.target.name] for c in grid.small_world_connections()],
        "weights": [grid.connections[i].weight for i in range(1, len(grid.connections) + 1)],
        "thresholds": [n.threshold for n in grid.neurons.values()],
        "rates": [n.rate for n in grid.neurons.values()],
    }
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
    neurons = list(grid.neurons.values())
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
