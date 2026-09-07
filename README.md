# walnutbutter

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
walnutbutter                 # open the window, free-run, learn, report accuracy; close it to stop
walnutbutter --headless --epochs 20000 -q   # the same without a window, for a fixed number of epochs
walnutbutter --step          # window where each Space press runs one epoch
walnutbutter --no-learn      # just watch the untrained network
walnutbutter --columns 24 --rows 20       # a bigger mesh than the default 8 x 10: 12 input bits, coded to 24
walnutbutter --input 1011                 # choose the 4 input bits
walnutbutter --no-permute                 # coded bits in order on the bottom row
walnutbutter --seed 42       # repeat a particular random mesh
walnutbutter --weight 1      # a fixed weight on every connection instead
walnutbutter --omega 0.1     # one connection in ten is a shortcut (default: 0.05)
walnutbutter --positive-weights --threshold 2   # no inhibition: weights kept in [epsilon, 1]
walnutbutter --omega 0       # plain mesh, no shortcuts
walnutbutter --weight 0.2 --show   # signal dies at the origin
python -m walnutbutter       # same thing without the installed command
```

**Input.** The network's input is its bottom row. Each epoch draws 4 random
bits (half the columns), complement-codes them by appending their negations,
and scrambles the 8 coded bits with a random permutation of the columns that
is drawn once per run and never changes. The bottom-row neurons whose bit is
1 are forced to fire in wave 0, so exactly half of the row fires every time.
The raw bits, the coded bits and the permuted row are printed, and the
permutation is printed once at the start. `--input 1011` supplies specific
bits for the first epoch, `--no-permute` lays the coded bits down in order,
and `--seed` reproduces the permutation and the whole sequence of random
inputs. Columns must be even. From Python, `run_epoch(grid)` resets the
mesh and presents the next input.

## Seeing the grid

```bash
walnutbutter                              # the default: free-running window with learning
walnutbutter --window 1200 800            # a bigger window
walnutbutter --report 300                 # a progress line every 5 minutes instead of 30 s
walnutbutter --save-weights run1.json     # checkpoint the learned weights at every report and on exit
walnutbutter --load-weights run1.json     # continue from a checkpoint (same mesh, seed and permutation)
walnutbutter --step                       # one epoch per Space press
walnutbutter --columns 24 --rows 20 --headless --save grid.png
```

By default the window is a monitor on a free-running system. The network
runs epoch after epoch as fast as the machine allows, silently, learning
after every one, with no coupling to the display; the window samples its
state 30 times a second, always showing a completed epoch. The title bar
shows the epoch count, the epoch rate, the accuracy to date (the mean over
every epoch since the start) and the recent accuracy, and the same figures
go to the terminal every `--report` seconds with an elapsed-time stamp, so a
run can be left for hours and read back later. **Esc** or **Q** closes the
window; the final figures are printed on exit.

With `--step` nothing happens until you press **Space**, which resets every
neuron (weights, shortcuts and thresholds are kept), draws a fresh random
input, fires the bottom row and, unless `--no-learn`, teaches. The input
neurons are ringed in white. Fired neurons are coloured, shading from
yellow in wave 0 to orange in the last wave, unfired neurons are grey, and the
neurons that were forced in wave 0 carry a white ring. Adding `--save PATH`
writes whatever state the mesh is in when the window closes.

```python
from walnutbutter import main, visualizer

grid = main(columns=8, rows=10)
visualizer.save(grid, "grid.png")   # write a picture, no window needed
visualizer.show(grid, 1200, 800)    # or open a window; Space runs a new epoch, Esc quits
```

## From Python

```python
from walnutbutter import main
from walnutbutter.monitor import run_epoch

grid = main(columns=8, rows=10)  # builds the grid and runs the first epoch on its bottom row
run_epoch(grid)                 # reset every neuron and present a new random input
print(len(grid.fired_neurons()))
grid.reset()                   # allow every neuron to fire again
```

## How the grid works

Neurons sit on a `columns x rows` rectangle of pointy-top hexagons. Every
odd row is shifted half a cell to the right ("odd-r" layout), which is what
lets whole hexagons fill a rectangle; the left and right edges are therefore
slightly jagged rather than cut. Internally each cell is addressed by axial
coordinates `(q, r)`, centred so the middle cell is `(0, 0)`. Each neuron is
connected to its six neighbours and to the twelve neighbours of those
neighbours, so an interior neuron has 18 outgoing and 18 incoming local
connections (fewer on the edges), and a signal covers two cells per wave.
`grid.get_neuron_at(column, row)`
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
is more than two steps away and not already a target of it. Shortcuts are
marked `kind == "small_world"` (first-ring connections are `"local"`,
second-ring ones `"local2"`), listed by
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
from walnutbutter.propagation import propagate

grid = GridOfNeurons(columns=8, rows=10)
origin, corner = grid.get_origin_neuron(), grid.get_neuron_at(0, 0)
waves = grid.propagate(fire=[origin, corner])          # two stimuli in one epoch
waves = grid.propagate(inputs={origin: 0.6, corner: 0.6})  # external input amounts instead
[len(w.fired) for w in waves]                          # neurons fired per wave
```

**Weights.** By default the command line gives every connection its own
random weight, drawn uniformly between -1 and 1, so each direction between a
pair of neurons gets an independent value. `--positive-weights` restricts
the range to `[epsilon, 1]` (`--epsilon`, default 0.001) for both the
initial draw and the clipping applied during learning, which removes all
inhibition. With nothing to hold activity down, a positive-weight mesh at
threshold 0.25 fires every neuron every epoch; a threshold of about 2 gives
activity comparable to the signed default. The range is stored in
checkpoints and restored with them. The seed is printed so a run can be
repeated with `--seed`. With the default threshold of 0.25, a typical random
mesh lets the signal reach somewhere between a tenth and a third of the
neurons before it dies out; raise `--threshold` to make it die sooner, lower
it to let it spread further. `--weight W` uses a fixed weight instead: with
`--weight 1` one signal is enough and the wave crosses the whole grid, while
`--weight 0.2` stops at the origin because no neuron ever hears from more
than one fired neighbour. From Python, `GridOfNeurons(weight=None, seed=...)`
or `grid.randomize_weights(low, high, seed)` do the same.

```python
grid = main(columns=8, rows=10)
conn = grid.get_connection(1)   # the first registered connection
conn.weight = 0.5
conn.is_active = False          # cut that direction only
origin, right = grid.get_neuron(0, 0), grid.get_neuron(1, 0)
grid.connection_between(origin, right)   # origin -> right
grid.connection_between(right, origin)   # right -> origin, a different connection
```

## Neurons without a grid

`walnutbutter --nodes` builds a population of 64 such neurons from the seed
(`--nodes N` for another count) and shows it, or lists their positions with
`--headless`, or writes a picture with `--save`. Nothing is wired or learned
for nodes yet.

`CartesianNodes` is a second container. Each neuron gets an `(x, y)` position
inside a bounding box, by default the square from -1 to 1 on both axes.
`add(x, y)` places a neuron exactly; `add()` places it at random, each
coordinate drawn uniformly from the box using the container's `seed`.
Coordinates outside the box are rejected. The container makes no
connections; `nearest` and `within` help decide them. Propagation and the
neurons themselves work exactly as in the grid.

```python
from walnutbutter.cartesian import CartesianNodes

nodes = CartesianNodes(count=50, seed=1)     # 50 neurons scattered in [-1, 1] x [-1, 1]
centre = nodes.add(0.0, 0.0)                 # one placed by hand
for other in nodes.within(0.0, 0.0, radius=0.3, exclude=centre):
    centre.connect(other, weight=0.5)
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

**Saving what it learned.** `--save-weights FILE` writes a JSON checkpoint
at every progress report and on exit: every weight by connection ID, plus
the mesh size, omega, threshold, seed, input permutation, epoch count and
the learning statistics. `--load-weights FILE` rebuilds that exact mesh from
the stored seed and settings, restores the weights, and carries on counting
epochs and accuracy to date from where the file left off. The sequence of
random inputs starts afresh, so a resumed run is not epoch-for-epoch
identical to an uninterrupted one, but the learned weights are the same.
From Python: `persistence.checkpoint(grid, path, teacher)` and
`grid, data = persistence.restore(path)`.

**Stuck neurons and homeostasis.** A neuron whose input sits far from its
threshold is never flipped by the exploration noise, so it gets no learning
signal and stays "stuck" always on or always off; on a long run most hidden
neurons end up that way. The status line counts them. To counter it, every
neuron outside the input row tracks its own firing rate and slowly moves its
threshold toward a target rate (`--homeostasis`, default 1e-6 per epoch,
`--target-rate`, default 0.4); firing too often raises the threshold, too
rarely lowers it. `--homeostasis 0` switches it off. Per-neuron thresholds
are saved in checkpoints.

**Un-sticking the outputs.** A saturated output neuron, one that fires on
every input or on none, gets no learning signal at all, because the
exploration noise never changes what it does. `--unstick RATE` (default
0.001) moves the threshold of any output neuron that is stuck, firing more
than 99% or less than 1% of the time, toward `--unstick-target` (default
0.5), and stops the moment it is no longer stuck. Nothing else in the mesh
is touched, so what the network has already learned is preserved. On a
trained checkpoint this freed both stuck outputs without disturbing the
six correct ones; routing the missing bit to them additionally needs the
interior to loosen, which is the slow global homeostasis's job.

Accuracy **to date** is the mean over every epoch since the start; the
**recent** figure is an exponential average over roughly the last 200.

**Where this stands.** Input-independent targets are learned: `all-off`
passes 90% within a couple of thousand epochs on an 8x4 mesh. For
input-dependent targets learning is real but slow: on an 8x4 mesh with both
rings of neighbours, `reversed` climbs from 50% to the low 60s within ten
thousand epochs. On a 24x20 mesh nothing measurable happened within fifteen
thousand epochs. Reinforcement learning of this kind pays for its generality
with variance, and the variance grows with the number of neurons being
perturbed.

```python
from walnutbutter.learning import Teacher
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
src/walnutbutter/
  connection.py Connection: ID, source and target neurons, weight, is_active, kind
  neuron.py    Neuron: threshold, potential, receive(), fire(), reset()
  propagation.py Signal queue and wave-by-wave propagate()
  grid.py      GridOfNeurons: builds the rectangle of hexagons and wires up both rings of neighbours
  cartesian.py CartesianNodes: neurons at (x, y) positions in a box, placed or random; no grid
  inputs.py    random bits, complement coding, parsing and formatting
  monitor.py   main(): build a grid and run its first epoch; run_epoch(): reset and present a new input
  learning.py  output targets, reward, the global-reinforcement rule, and Teacher
  persistence.py checkpoint() and restore() for learned weights
  visualizer.py hex geometry and pygame drawing: show() and save()
  cli.py       argument parsing and the `walnutbutter` command
  __main__.py  lets you run `python -m walnutbutter`
tests/         pytest tests for the neuron, the grid, and the command line
pyproject.toml project metadata, dependencies, and the command definition
```
