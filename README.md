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
neurohacking --columns 8 --rows 6         # 4 input bits, coded to 8
neurohacking --input 101100111000         # choose the 12 input bits
neurohacking --seed 42       # repeat a particular random mesh
neurohacking --weight 1      # a fixed weight on every connection instead
neurohacking --omega 0.1     # one connection in ten is a shortcut (default: 0.05)
neurohacking --omega 0       # plain mesh, no shortcuts
neurohacking --weight 0.2 --show   # signal dies at the origin
python -m neurohacking       # same thing without the installed command
```

**Input.** The network's input is its bottom row. Each epoch draws 12 random
bits (half the columns), complement-codes them by appending their negations,
and forces the bottom-row neurons whose bit is 1 to fire in wave 0. So
exactly half of the row fires every time. The raw bits and the coded row are
printed, `--input 101100111000` supplies specific bits for the first epoch,
and `--seed` reproduces the whole sequence of random inputs. Columns must be
even. From Python, `run_epoch(grid)` resets the mesh and presents the next input.

## Seeing the grid

```bash
neurohacking --show                  # open an 800x600 window on the fresh mesh
neurohacking --columns 8 --rows 6 --save grid.png
neurohacking --window 1200 800 --show
neurohacking --fast                       # free-run the system; the window monitors it at 30 Hz
neurohacking --quiet                      # no line per firing neuron
neurohacking --fast --learn               # reinforce after every epoch; accuracy in the title bar
neurohacking --learn --epochs 20000 -q    # headless training run, accuracy printed as it goes
```

With `--show` the window opens on the mesh after its first epoch has run.
Press **Space** to start a new epoch: every neuron is reset (weights,
shortcuts and thresholds are kept), a fresh random input is drawn, and the
bottom row is fired again. **Esc** or **Q** closes the window.

`--fast` (which implies `--show`) turns the window into a monitor. The system
runs epoch after epoch as fast as the machine allows, silently, with no
coupling to the display, and the window samples its state 30 times a second,
always showing a completed epoch. The title bar reports the epoch rate. The input
neurons are ringed in white. Fired neurons are coloured, shading from
yellow in wave 0 to orange in the last wave, unfired neurons are grey, and the
neurons that were forced in wave 0 carry a white ring. Adding `--save PATH`
writes whatever state the mesh is in when the window closes.

```python
from neurohacking import main, visualizer

grid = main(columns=8, rows=6)
visualizer.save(grid, "grid.png")   # write a picture, no window needed
visualizer.show(grid, 1200, 800)    # or open a window; Space runs a new epoch, Esc quits
```

## From Python

```python
from neurohacking import main
from neurohacking.monitor import run_epoch

grid = main(columns=8, rows=6)  # builds the grid and runs the first epoch on its bottom row
run_epoch(grid)                 # reset every neuron and present a new random input
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

**Small-world shortcuts.** `omega` (0 up to but not including 1, default 0.05)
is the proportion of all connections that are long-range shortcuts. After the local
mesh is built with L connections, `omega * L / (1 - omega)` extra connections
are added, each running one way from a random neuron to a random neuron that
is not one of its six neighbours and not already a target of it. Shortcuts
are marked `kind == "small_world"` (local ones are `"local"`), listed by
`grid.small_world_connections()`, and get weights like any other connection.
The same `seed` reproduces both the shortcuts and the weights.

**Firing rule.** Every neuron has a `threshold` (default 0.25) and a running
`potential`. When a neuron fires, each of its active outgoing connections adds
its weight to the target's potential. A neuron fires the moment its potential
reaches its threshold, and it fires at most once until the grid is reset.
Negative weights lower the potential, so they act as inhibitory connections.
The input neurons are fired directly as an external stimulus, which ignores
the threshold.

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

## Teaching the network

The top row is the output. For each epoch the network is told what the top
row should have shown, by default a **reversed** copy of the bottom-row
input (`--target reversed`; `copy`, `all-off` and `all-on` also exist).
Accuracy is the fraction of the 24 output neurons that match. Because the
input is complement-coded, exactly half the outputs should fire, so an output
row that never fires already scores 50%; that is the number to beat.

Learning is **global reinforcement**: a single scalar reward, the epoch's
accuracy, is broadcast to every connection. Nothing is traced back through
the network. Each epoch every neuron starts with a small random potential
(exploration, `--sigma`), the epoch runs, and the reward is compared with a
running average to give an *advantage*: better or worse than usual. Every
connection that carried a signal into a neuron that was not a forced input
then moves by `lr * advantage * eligibility`, where the eligibility is the
target neuron's exploration noise. A neuron that was nudged towards firing
in a better-than-usual epoch gets stronger inputs from whoever fed it. This
is the REINFORCE / node-perturbation estimator, a three-factor rule:
presynaptic activity x postsynaptic perturbation x global reward.
`--eligibility hebb` swaps the perturbation for a plain Hebbian term (+1 if
the target fired, -1 if not) with no noise. Forced inputs are never adjusted
and weights stay within [-1, 1]. `Teacher` wraps all this; use
`teacher.epoch()` instead of `run_epoch(grid)` so the exploration noise is
injected.

**Where this stands.** Input-independent targets are learned: `all-off`
passes 90% within a couple of thousand epochs on an 8x4 mesh, `all-on` more
slowly. For input-dependent targets learning is real but slow: on an 8x4
mesh over thirty thousand epochs, `copy` climbs from about 55% to 65-68% and
`reversed` from 50% to 57-68%, depending on the seed. On the default 24x20
mesh nothing measurable has happened within fifteen thousand epochs.
Reinforcement learning of this kind pays for its generality with variance,
and the variance grows with the number of neurons being perturbed.

```python
from neurohacking.learning import Teacher
grid = main(columns=24, rows=20, seed=1)
teacher = Teacher(grid, target="reversed", lr=0.03, sigma=0.1, seed=1)
for _ in range(10000):
    teacher.epoch(verbose=False)   # exploration noise, new input, propagate, reinforce
print(teacher.status())
```

## Tests

```bash
pytest
```

## Layout

```
src/neurohacking/
  connection.py Connection: ID, source and target neurons, weight, is_active, kind
  neuron.py    Neuron: threshold, potential, receive(), fire(), reset()
  propagation.py Signal queue and wave-by-wave propagate()
  grid.py      GridOfNeurons: builds the rectangle of hexagons and wires up neighbours
  inputs.py    random bits, complement coding, parsing and formatting
  monitor.py   main(): build a grid and run its first epoch; run_epoch(): reset and present a new input
  learning.py  output targets, reward, the global-reinforcement rule, and Teacher
  visualizer.py hex geometry and pygame drawing: show() and save()
  cli.py       argument parsing and the `neurohacking` command
  __main__.py  lets you run `python -m neurohacking`
tests/         pytest tests for the neuron, the grid, and the command line
pyproject.toml project metadata, dependencies, and the command definition
```
