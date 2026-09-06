# neurohacking

Builds a hexagonal grid of neurons, fires the one at the centre, and watches
the signal spread outward. Each neuron fires at most once per run.

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra includes pytest and pygame. To install only what the
visualizer needs, use `pip install -e ".[viz]"` instead.

## Usage

```bash
neurohacking                 # grid of radius 10 (331 neurons)
neurohacking --size 3        # smaller grid, 37 neurons
python -m neurohacking       # same thing without the installed command
```

`--interval` is accepted but not used yet.

## Seeing the grid

```bash
neurohacking --show                  # open a window; close it or press Esc/Q
neurohacking --size 6 --save grid.png
```

Fired neurons are coloured, shading from yellow at the origin to orange at
the rim, unfired neurons are grey, and the origin carries a white ring.

```python
from neurohacking import main, visualizer

grid = main(grid_size=6)
visualizer.save(grid, "grid.png")   # write a picture, no window needed
visualizer.show(grid)               # or open a window
```

## From Python

```python
from neurohacking import main

grid = main(grid_size=3)       # builds the grid and fires the origin
print(len(grid.fired_neurons()))
grid.reset()                   # allow every neuron to fire again
```

## How the grid works

Neurons sit on a hexagonal grid addressed by axial coordinates `(q, r)`. A
grid of size `n` contains every cell where the largest of `|q|`, `|r|` and
`|q + r|` is at most `n`, which gives `3n² + 3n + 1` neurons. Each neuron is
connected to its six neighbours.

Each neighbouring pair shares one `Connection` object, which holds references
to both neurons and an `is_active` flag. The grid keeps every connection in
`grid.connections`, a dictionary keyed by ID starting from 1, so a pathway
can be looked up and switched off from either end. Activating a neuron marks
it as fired and passes the signal along each active connection to a
neighbour that has not fired yet.

```python
grid = main(grid_size=3)
conn = grid.get_connection(1)          # Connection(1: Neuron_-3_0 <-> Neuron_-2_0, active)
conn.is_active = False                 # cut that pathway
grid.connection_between(grid.get_neuron(0, 0), grid.get_neuron(1, 0))
```

## Tests

```bash
pytest
```

## Layout

```
src/neurohacking/
  connection.py Connection: ID, references to two neurons, is_active flag
  neuron.py    Neuron: shared connections, has_fired flag, activate() and reset()
  grid.py      GridOfNeurons: builds the hexagon and wires up neighbours
  monitor.py   main(grid_size): build a grid, fire the origin, return the grid
  visualizer.py hex geometry and pygame drawing: show() and save()
  cli.py       argument parsing and the `neurohacking` command
  __main__.py  lets you run `python -m neurohacking`
tests/         pytest tests for the neuron, the grid, and the command line
pyproject.toml project metadata, dependencies, and the command definition
```
