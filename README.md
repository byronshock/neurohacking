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
neurohacking --weight 0.5 --threshold 1.0 --show   # signal dies at the origin
python -m neurohacking       # same thing without the installed command
```

`--interval` is accepted but not used yet.

## Seeing the grid

```bash
neurohacking --show                  # open a window; close it or press Esc/Q
neurohacking --size 6 --save grid.png
```

Fired neurons are coloured, shading from yellow in wave 0 to orange in the
last wave, unfired neurons are grey, and the origin carries a white ring.

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

Connections are one-way and weighted. Each neighbouring pair gets two
`Connection` objects, one in each direction, and each holds references to its
`source` and `target` neurons, a `weight` (default 1.0), and an `is_active`
flag. The grid keeps every connection in `grid.connections`, a dictionary
keyed by ID starting from 1. A neuron lists the connections it sends along in
`outgoing` and the ones it receives from in `incoming`.

**Firing rule.** Every neuron has a `threshold` (default 1.0) and a running
`potential`. When a neuron fires, each of its active outgoing connections adds
its weight to the target's potential. A neuron fires the moment its potential
reaches its threshold, and it fires at most once until the grid is reset.
Negative weights lower the potential, so they act as inhibitory connections.
The origin is fired directly as an external stimulus, which ignores the
threshold.

**Propagation** is not recursive. `propagation.propagate` keeps a first-in,
first-out queue of `Signal` messages, each tagged with a wave number. In each
wave it first delivers every queued signal, then fires every neuron that has
reached its threshold, queueing their outgoing signals for the next wave.
Delivering everything before deciding who fires means the outcome never
depends on the order neurons are stored in, and there is no recursion limit
on grid size. Each neuron records `fired_in_wave`, the grid keeps the list of
`Wave` objects from its last epoch in `grid.waves`, and the visualizer shades
fired neurons by wave.

```python
from neurohacking.propagation import propagate

grid = GridOfNeurons(size=3)
origin, corner = grid.get_neuron(0, 0), grid.get_neuron(3, 0)
waves = grid.propagate(fire=[origin, corner])          # two stimuli in one epoch
waves = grid.propagate(inputs={origin: 0.6, corner: 0.6})  # external input amounts instead
[len(w.fired) for w in waves]                          # neurons fired per wave
```

With the defaults (weight 1.0, threshold 1.0) one signal is enough and the
wave crosses the whole grid. Try `--weight 0.5` to see it stop at the origin,
because no neuron ever hears from more than one fired neighbour.

```python
grid = main(grid_size=3)
conn = grid.get_connection(1)   # Connection(1: Neuron_-3_0 -> Neuron_-2_0, weight 1, active)
conn.weight = 0.5
conn.is_active = False          # cut that direction only
origin, right = grid.get_neuron(0, 0), grid.get_neuron(1, 0)
grid.connection_between(origin, right)   # origin -> right
grid.connection_between(right, origin)   # right -> origin, a different connection
```

## Tests

```bash
pytest
```

## Layout

```
src/neurohacking/
  connection.py Connection: ID, source and target neurons, weight, is_active
  neuron.py    Neuron: threshold, potential, receive(), fire(), reset()
  propagation.py Signal queue and wave-by-wave propagate()
  grid.py      GridOfNeurons: builds the hexagon and wires up neighbours
  monitor.py   main(grid_size): build a grid, fire the origin, return the grid
  visualizer.py hex geometry and pygame drawing: show() and save()
  cli.py       argument parsing and the `neurohacking` command
  __main__.py  lets you run `python -m neurohacking`
tests/         pytest tests for the neuron, the grid, and the command line
pyproject.toml project metadata, dependencies, and the command definition
```
