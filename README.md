# walnutbutter

A living network of simple neurons on a plane, and the substance you build
it from.

Each neuron is a fire-once threshold unit: weighted input accumulates until
it crosses a threshold, the neuron fires exactly once per epoch, and its
signal travels one-way, wave by wave, along weighted connections. The
network's input is a complement-coded, permuted bit pattern forced onto its
bottom row; its output is the top row, which is taught to show that pattern
reversed. Learning is global reinforcement: every epoch a single scalar
reward, the output's accuracy, is broadcast to every connection and combined
with each neuron's exploration noise (node-perturbation REINFORCE), with a
slow homeostatic drift of thresholds and an un-sticking rule for saturated
outputs. There is no training run and no evaluation run, only one run that
keeps going: the window is a 30 Hz monitor on a free-running system that
learns, checkpoints itself, and reports as it goes.

Two containers build networks. The **hex grid** wires every cell to its two
rings of neighbours plus a few random small-world shortcuts. **Walnut butter**
is the substance the neurons are made of: spread it on the plane in smears of
a given density and neurons appear at that density, and butter spread near
other butter connects, so where you put it and how thick decides the whole
architecture. The default of each is an 8 x 10 field of 80 neurons.

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra includes pytest, pygame, numpy and scipy. To install only
what the visualizer needs, use `pip install -e ".[viz]"`; only what the
array engine needs, `pip install -e ".[arrays]"`. The object engine has no
dependencies at all.

## Usage

```bash
walnutbutter                 # open the window, free-run, learn, report accuracy; close it to stop
walnutbutter --headless --epochs 20000 -q   # the same without a window, for a fixed number of epochs
walnutbutter --step          # window where each Space press runs one epoch
walnutbutter --no-learn      # just watch the untrained network
walnutbutter --columns 24 --rows 20       # a bigger mesh than the default 8 x 10: 12 input bits, coded to 24
walnutbutter --ecc                        # 4 data bits -> Hamming (7, 4) -> 14 columns, on a 14 x 10 field
walnutbutter --ecc parity64               # the (6, 4) detect-only code on 12 columns instead
walnutbutter --input 1011                 # choose the 4 input bits
walnutbutter --no-permute                 # coded bits in order on the bottom row
walnutbutter --seed 42       # repeat a particular random mesh
walnutbutter --weight 1      # a fixed weight on every connection instead
walnutbutter --omega 0.1     # one connection in ten is a shortcut (default: 0.2)
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

## Comparing seeds

Outcomes vary a lot between seeds: with identical settings, some seeds reach
the high 80s within a million epochs while others sit in the low 60s. So
the best use of a many-core machine is to run several seeds at once and
keep the best:

```bash
walnutbutter --seeds 15 --epochs 1000000 --seed 1
```

runs seeds 1 to 15 in parallel, headless, one process per core (add `--nodes`
for the lattice instead of the grid), prints a table sorted best first (accuracy over each run's last tenth, and to date),
and checkpoints every run to `runs/` so the winner can be loaded with
`--load-weights`. Without `--seed` the base seed is random and printed.

## Seeing the grid

```bash
walnutbutter                              # the default: free-running window with learning
walnutbutter --window 1200 800            # a bigger window
walnutbutter --report 30                  # a progress line every 30 s instead of every second
walnutbutter --save-weights run1.json     # choose the checkpoint file (default: runs/<date>-<time>-seed<seed>.json)
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
go to the terminal every `--report` seconds (default 1) with an elapsed-time
stamp, so a run can be left for hours and read back later. Each report is
also appended to an accuracy history (epoch, elapsed seconds, accuracy to
date, recent accuracy, stuck counts, epoch rate) that `--save-weights`
stores in the checkpoint and `--load-weights` carries forward, so the
learning curve survives the window closing. **Esc** or **Q** closes the
window; the final figures are printed on exit.

With `--step` nothing happens until you press **Space**, which resets every
neuron (weights, shortcuts and thresholds are kept), draws a fresh random
input, fires the bottom row and, unless `--no-learn`, teaches. The input
neurons are ringed in white. Fired neurons are coloured, shading from
yellow in wave 0 to orange in the last wave, unfired neurons are grey, and the
neurons that were forced in wave 0 carry a white ring. Each neuron is drawn as a disc on its hexagonal cell, sized so neighbouring discs never touch. Adding `--save PATH`
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

**Small-world shortcuts.** `omega` (0 up to but not including 1, default 0.2)
is the proportion of all connections that are long-range shortcuts. After the local
mesh is built with L connections, `omega * L / (1 - omega)` extra connections
are added, each running one way from a random neuron to a random neuron that
is more than two steps away and not already a target of it. Shortcuts are
marked `kind == "small_world"` (first-ring connections are `"local"`,
second-ring ones `"local2"`), listed by
`grid.small_world_connections()`, and get weights like any other connection.
The same `seed` reproduces both the shortcuts and the weights.

**Error-correcting code.** With `--ecc` (or `network.use_ecc()`), the raw
input is 4 data bits, encoded before complement coding. The default code is
Hamming's (7, 4): three parity bits, each covering three of the four data
bits, giving every bit position a distinct syndrome, so any single flipped
bit is located and corrected (`Code.correct`, `Code.decode`); complement
coding then fills a 14-column bottom row, so the usual field is 14 columns
by 10 rows. `--ecc parity64` is the (6, 4) code instead: two parity bits,
minimum distance 2, single errors detected but not corrected, 12 columns.
`--ecc` sets the columns to fit unless told otherwise. The epoch line reads
`data 1011 -> hamming74 1011010 -> coded ... -> bottom row ...`, and
checkpoints remember which code is on.

**Critics.** The reward is a single number per epoch, and `--critic` chooses
how it is judged. `row` (the default) is the fraction of output neurons that
match the target, neuron by neuron. `decoded` reads the output row the way a
receiver would: it undoes the target's arrangement and the permutation,
resolves each complement pair to a bit (a pair whose neurons contradict each
other is unreadable), runs the word through the code's error correction, and
rewards the fraction of data bits that come out right; `decoded-exact` gives
1 only if all of them do. Under Hamming a single wrong output neuron costs
nothing with the decoding critics, because the code absorbs it: the network
is judged on the message, not the pixels, and the code's redundancy stands in
for a population of outputs.

**Firing rule.** Every neuron has a `threshold` (default 0.25) and a running
`potential`. When a neuron fires, each of its active outgoing connections adds
its weight to the target's potential. A neuron fires the moment its potential
reaches its threshold, and it fires at most once until the grid is reset.
Negative weights lower the potential, so they act as inhibitory connections.
The input neurons are fired directly as an external stimulus, which ignores
the threshold.

**Propagation** is not recursive. `propagation.propagate` keeps a queue of
waves: one list of connections whose signals are in flight, filled by the
neurons that fired in the previous wave. In each wave it first delivers every
queued signal, then fires every neuron that has reached its threshold,
queueing their outgoing connections for the next wave. Every neuron and
connection stays an object that receives and fires for itself; the queue
adds no allocation per signal, which nearly doubled the epoch rate.
Delivering everything before deciding who fires means the outcome never
depends on the order neurons are stored in, and there is no recursion limit
on grid size. The floor on a potential (`--minimum-potential`, default -1)
is applied once a wave's signals are all in, so it acts on the wave's total
and the order of arrival cannot matter there either. Each neuron records
`fired_in_wave`, the grid keeps the list of `Wave` objects from its last
epoch in `grid.waves`, and the visualizer shades fired neurons by wave.

**Two engines, one network.** The object engine above is the one you watch:
every neuron and connection is an object that receives and fires for
itself. `--engine arrays` runs the same network as numpy vectors and a scipy
sparse matrix (`arrays.py`): the neurons that fired in a wave, as a 0/1
vector, times the weight matrix gives every neuron its summed input in one
product, and the learning rule becomes a handful of elementwise operations
over the edges. Both engines build the mesh the same way, share connection
ids, read and write the same checkpoints (a loaded checkpoint keeps the
engine that wrote it unless `--engine` says otherwise), draw the same
exploration noise from the same seed, and are run side by side by
`tests/test_arrays.py`, which checks that they fire the same neurons wave
by wave and move the same weights. They can differ only in the order
floating-point additions happen, so on the rare epoch where a potential sits
within rounding of a threshold the two may decide differently and diverge
from there, like two seeds. On this machine the array engine runs an 8x10
mesh about twice as fast as the object engine, a 24x20 mesh five times as
fast and a 48x40 mesh seven times as fast; the object engine has no
dependencies and prints per neuron with `-v`, which the array engine does not.

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

## Walnut butter

Walnut butter is the substance the neurons are made of. It is spread over
the plane in **smears**: each is a shape (`Rect` or `Disc`, in unit
distances) with a **density**, and placing the butter packs neurons on a
hexagonal lattice inside each shape at the spacing that density implies.
Thick butter means many neurons close together, thin butter a few far
apart, bare plane none. **Butter that is spread near other butter
connects:** a neuron projects to every neuron within its `reach` (default 2
units), so density alone decides how richly a region is wired, and a gap in
the spread is a gap in the network. Nothing about the topology is random;
only the weights are drawn from the seed.

Butterspace has one scale, the **unit distance**. Density is measured in
neurons per unit cell, the hexagon a neuron owns in a lattice at unit
spacing, so unit density (`UNIT_DENSITY`, which is 1) means neighbours one
unit apart, and a density of 4 packs four neurons into each cell, half a
unit apart. Every distance in the substance is compared with that one unit
whatever the local density: a reach of 2 is two units everywhere, so butter
four times as thick has four times the neurons within reach.

The default network is one rectangular smear at unit density, which is the
8 x 10 hexagonal lattice at unit spacing: `CartesianNodes()` builds it
directly, and with a reach of 2 each interior neuron has eighteen neighbours,
six at distance 1, six at √3 and six at 2, the same as the hex grid's two
rings. Its bottom row is the input and its top row the output, addressed
like the grid with `get_neuron_at(column, row)`. A free spread has no rows,
so input and output zones for it are still to be defined.

```python
from walnutbutter.butter import Disc, Rect, WalnutButter, UNIT_DENSITY
from walnutbutter.cartesian import CartesianNodes

recipe = (WalnutButter()
          .spread(Rect(-4, -4, 4, 4), UNIT_DENSITY)        # a lattice-density slab (density 1)
          .spread(Disc(0, 0, 1.5), 3.0))                   # a dense knot in the middle
nodes = CartesianNodes.from_butter(recipe, seed=1)
nodes.connect_within(reach=2.0, weight=None)               # near butter connects
```

`walnutbutter --nodes` builds the default lattice, wires it with `--reach`
(default 2), and then does everything the grid does: learns, reports,
checkpoints to `runs/` (lattice checkpoints record the wiring and load back
with `--load-weights`), and shows in the window or runs headless with
`--epochs`. `--nodes N` scatters N neurons at random instead; a scatter has
no rows, so it is shown, not trained. The earlier Gaussian receptive-field
wiring (`connect_by_distance`) remains in the library for reference.

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
and weights stay within [-1, 1].

A signal that arrives after its target has already fired is dropped on
delivery and changes nothing in the epoch, yet by default its connection is
still reinforced: pre fired, post fired, and the global reward says whether
the coincidence was good. That is a local Hebbian term riding on the
perturbation estimator, strictly a bias with respect to the reward
gradient, but it is the biological shape of the rule (local eligibility,
global signal) and it learns faster: on the 8x10 reversed task at 100k
epochs, every seed tried did better with it (last tenth 0.76-0.91 against
0.61-0.75). `--late` chooses what a late signal earns: `count` (the
default), `ignore` (nothing: only the signals that landed, the
node-perturbation estimator proper) or `depress` (the opposite update, the
shape of spike-timing-dependent plasticity, where a presynaptic spike after
the postsynaptic one weakens the synapse). `Teacher` wraps all this; use
`teacher.epoch()` instead of `run_epoch(grid)` so the exploration noise is
injected.

**Output.** Runs are silent apart from the progress reports and the final
summary: printing is far slower than learning, and a headless run of millions
of epochs would otherwise spend its time writing to the terminal. `-v` /
`--verbose` prints a line for every epoch's input and every neuron that
fires, for short inspections. From Python the same switch is
`Neuron.verbose`, off by default; `run_epoch(grid)` prints its one input
line unless told `verbose=False`.

**Saving what it learned.** Every run writes a JSON checkpoint at every
progress report and on exit, by default to `runs/<date>-<time>-seed<seed>.json`
(the path is printed at the start; `runs/` is ignored by git). `--save-weights FILE`
chooses the file, `--no-save` skips it. A checkpoint holds: every weight by connection ID, plus
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
neurons end up that way. To counter it, every
neuron tracks its own firing rate and slowly moves its threshold toward a
target rate (`--homeostasis`, default 1e-6 per epoch,
`--target-rate`, default 0.5); firing too often raises the threshold, too
rarely lowers it. Neurons forced in wave 0 are left out for that epoch, as
reinforcement leaves them out; an unforced input neuron is treated like any
other. `--homeostasis 0` switches it off. Per-neuron thresholds are saved in
checkpoints.

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
  propagation.py wave-by-wave propagate(): one list of connections in flight per wave
  arrays.py    ArrayNetwork: the same network as numpy vectors and a scipy sparse matrix (--engine arrays)
  exploration.py the Box-Muller noise draws both engines share
  learning_rules.py constants shared by the learning code of both engines
  grid.py      GridOfNeurons: builds the rectangle of hexagons and wires up both rings of neighbours
  butter.py    WalnutButter: smears of neuron density (per unit cell) on the plane; shapes Rect and Disc
  cartesian.py CartesianNodes: the lattice, a scatter, or a placed recipe; reach wiring
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
