# neurohacking

Builds a rectangular mesh of hexagonal neurons, fires the one at the centre,
and watches the signal spread outward. Each neuron fires at most once per run.

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
neurohacking                 # 24 x 20 mesh (480 neurons), random weights
neurohacking --columns 8 --rows 6
neurohacking --seed 42       # repeat a particular random mesh
neurohacking --weight 1      # a fixed weight on every connection instead
neurohacking --weight 0.2 --show   # signal dies at the origin
python -m neurohacking       # same thing without the installed command
```

`--interval` is accepted but not used yet.

## Seeing the grid

```bash
neurohacking --show                  # open an 800x600 window on the fresh mesh
neurohacking --columns 8 --rows 6 --save grid.png
neurohacking --window 1200 800 --show
```

With `--show` the window opens as soon as the mesh is built, before anything
has fired. Press **Space** to fire the origin, **R** to reset the mesh, and
**Esc** or **Q** to close the window. Fired neurons are coloured, shading from
yellow in wave 0 to orange in the last wave, unfired neurons are grey, and the
origin carries a white ring. Adding `--save PATH` writes whatever state the
mesh is in when the window closes.

```python
from neurohacking import main, visualizer

grid = main(columns=8, rows=6)
visualizer.save(grid, "grid.png")   # write a picture, no window needed
visualizer.show(grid, 1200, 800)    # or open a window; Space fires, R resets, Esc quits
```

## From Python

```python
from neurohacking import main

grid = main(columns=8, rows=6)  # builds the grid and fires the origin
print(len(grid.fired_neurons()))
grid.reset()                   # allow every neuron to fire again
```

## How the grid works

Neurons sit on a `columns x rows` rectangle of pointy-top hexagons. Every
odd row is shifted half a cell to the right ("odd-r" layout), which is what
lets whole hexagons fill a rectangle; the left and right edges are therefore
slightly jagged rather than cut. Internally each cell is addressed by axial
coordinates `(q, r)`, centred so the middle cell is `(0, 0)`, and each neuron
is connected to its six neighbours (fewer on the edges). `grid.get_neuron_at(column, row)`
looks a cell up by its position from the top-left corner; `grid.get_neuron(q, r)`
by axial coordinates.

Connections are one-way and weighted. Each neighbouring pair gets two
`Connection` objects, one in each direction, and each holds references to its
`source` and `target` neurons, a `weight` (default 1.0), and an `is_active`
flag. The grid keeps every connection in `grid.connections`, a dictionary
keyed by ID starting from 1. A neuron lists the connections it sends along in
`outgoing` and the ones it receives from in `incoming`.

**Firing rule.** Every neuron has a `threshold` (default 0.25) and a running
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

grid = GridOfNeurons(columns=8, rows=6)
origin, corner = grid.get_origin_neuron(), grid.get_neuron_at(0, 0)
waves = grid.propagate(fire=[origin, corner])          # two stimuli in one epoch
waves = grid.propagate(inputs={origin: 0.6, corner: 0.6})  # external input amounts instead
[len(w.fired) for w in waves]                          # neurons fired per wave
```

**Weights.** By default the command line gives every connection its own
random weight, drawn uniformly between -1 and 1, so each direction between a
pair of neurons gets an independent value. The seed is printed so a run can be
repeated with `--seed`. With the default threshold of 0.25, a typical random
mesh lets the signal reach somewhere between a tenth and a third of the
neurons before it dies out; raise `--threshold` to make it die sooner, lower
it to let it spread further. `--weight W` uses a fixed weight instead: with
`--weight 1` one signal is enough and the wave crosses the whole grid, while
`--weight 0.2` stops at the origin because no neuron ever hears from more
than one fired neighbour. From Python, `GridOfNeurons(weight=None, seed=...)`
or `grid.randomize_weights(low, high, seed)` do the same.

```python
grid = main(columns=8, rows=6)
conn = grid.get_connection(1)   # the first registered connection
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
  grid.py      GridOfNeurons: builds the rectangle of hexagons and wires up neighbours
  monitor.py   main(columns, rows): build a grid, fire the origin, return the grid
  visualizer.py hex geometry and pygame drawing: show() and save()
  cli.py       argument parsing and the `neurohacking` command
  __main__.py  lets you run `python -m neurohacking`
tests/         pytest tests for the neuron, the grid, and the command line
pyproject.toml project metadata, dependencies, and the command definition
```
