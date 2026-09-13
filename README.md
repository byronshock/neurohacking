# walnutbutter

A living network of simple neurons on a plane, and the substance you build
it from.

The system is specified in `AUTHORITY.md`; the code follows it. Each
neuron is an integrate-and-fire unit with no leak: weighted input
accumulates until it crosses a threshold, the neuron fires and resets, and
for an absolute refractory period it ignores everything. Its signal travels
one-way along weighted connections, one hop per connection, on a clock in
nominal milliseconds; a wave is everything that happens at one moment. A
neuron may fire as often as its refractory period allows, so a tight loop
can carry a neuron's own spike back to refire it, and activity sustains
itself. The network's input is a complement-coded, permuted bit pattern
forced onto its bottom row; its output is the top row. Learning is
**dopamine**: a neuron that fires again after its refractory period
releases dopamine, most when it refires the instant it may, into one
global pool; at that same refire its incoming synapses that carried a
signal since its previous spike move together in proportion to the pool
minus the network's expectation of it. There is no training run and no
evaluation run, only one run that keeps going: the window is a 30 Hz
monitor on a free-running system that learns, checkpoints itself, and
reports as it goes.

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
walnutbutter --problem sustain_inputs   # the 16 four-bit inputs as they are on 4 neurons, 20 ms epochs; score = which input neurons spiked again, against the pattern; learns by dopamine (AUTHORITY.md §8)
walnutbutter --problem improved_sustain # the same on a 10x7 grid wired to reach 3, the 4 inputs in the middle of the middle row
walnutbutter --trace runs/sustain.csv   # one line per epoch: epoch, time, dopamine, expected, score (default: next to the checkpoint)
walnutbutter --rule reinforce           # the pre-alpha's global-reward rule instead, run by the Teacher
walnutbutter --refractory-hops 3.7 --release-theta 2 --order update-first   # the clock and dopamine knobs (see --help)
walnutbutter --across 24 --rows 20       # a bigger mesh than the default 8 x 10: 12 input bits, coded to 24
walnutbutter --ecc                        # 4 data bits -> Hamming (7, 4) -> 14 across, on a 14 x 10 field
walnutbutter --ecc parity64               # the (6, 4) detect-only code on 12 across instead
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
bits (half the count across), complement-codes them by appending their negations,
and scrambles the 8 coded bits with a random permutation of the places along the row that
is drawn once per run and never changes. The bottom-row neurons whose bit is
1 are forced to fire at the input's time (refractory period permitting), so
up to half of the row fires every time.
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
walnutbutter --across 24 --rows 20 --headless --save grid.png
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
red for a spike at the end of the epoch cooling to blue over one epoch's
length, neurons that never fired are grey, and the
neurons that were forced this epoch carry a white ring. Each neuron is drawn as a disc on its hexagonal cell, sized so neighbouring discs never touch. Adding `--save PATH`
writes whatever state the mesh is in when the window closes.

```python
from walnutbutter import main, visualizer

grid = main(across=8, rows=10)
visualizer.save(grid, "grid.png")   # write a picture, no window needed
visualizer.show(grid, 1200, 800)    # or open a window; Space runs a new epoch, Esc quits
```

## From Python

```python
from walnutbutter import main
from walnutbutter.monitor import run_epoch

grid = main(across=8, rows=10)  # builds the grid and runs the first epoch on its bottom row
run_epoch(grid)                 # reset every neuron and present a new random input
print(len(grid.fired_neurons()))
grid.reset()                   # allow every neuron to fire again
```

## How the grid works

Neurons sit on a `across x rows` rectangle of pointy-top hexagons. Every
odd row is shifted half a cell to the right ("odd-r" layout), which is what
lets whole hexagons fill a rectangle; the left and right edges are therefore
slightly jagged rather than cut. Internally each cell is addressed by axial
coordinates `(q, r)`, centred so the middle cell is `(0, 0)`. Each neuron is
connected to its six neighbours and to the twelve neighbours of those
neighbours, so an interior neuron has 18 outgoing and 18 incoming local
connections (fewer on the edges), and a signal covers two cells per wave.
`grid.get_neuron_at(place, row)`
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
coding then fills a 14-place bottom row, so the usual field is 14 across
by 10 rows. `--ecc parity64` is the (6, 4) code instead: two parity bits,
minimum distance 2, single errors detected but not corrected, 12 across.
`--ecc` sets the count across to fit unless told otherwise. The epoch line reads
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

**Firing rule** (AUTHORITY.md §5). Every neuron has a `threshold` (default
0.25) and a running `potential`. When a neuron fires, each of its active
outgoing connections delivers its weight to the target's potential one hop
later. A neuron fires the moment its potential reaches its threshold, the
spike resets the potential, and the potential leaks with time constant
`--tau` (default 2 ms; `inf` switches it off), lazily: nothing happens to a
quiet neuron, and when a signal arrives the potential is first decayed for
the time since it was last brought up to date. Negative
weights lower the potential, so they act as inhibitory connections. The
input neurons are fired directly as an external stimulus, which ignores the
threshold. A neuron that fired within `--refractory` milliseconds (default
5) ignores every signal, forced stimulus included; that is the only thing
that limits how often it fires. A neuron's threshold also falls with its
silence, reaching zero `--bored-after` ms after its last spike (default
200), so a bored neuron fires on its own and resets (AUTHORITY.md §5.4).

**Time and the schedule** (§4). The network runs on a clock in nominal
milliseconds. A signal takes one **hop** to travel a connection,
`--refractory` / `--refractory-hops` (default 5 / 3 ms, and the ratio need
not be an integer). Firing is never recursive: `propagation.Schedule` is a
time-ordered queue of the signals in flight, and a **wave** is everything
due at one moment, the signals arriving and the stimulus if an input lands
then. In each wave every signal is delivered first (to targets that are not
refractory), the floor on a potential (`--minimum-potential`, default -1)
is applied to each touched neuron's total, and then every forced neuron and
every neuron that has reached its threshold fires, its outgoing connections
scheduled one hop later. Delivering everything before deciding who fires
means the outcome never depends on the order neurons are stored in, and
there is no recursion limit on grid size. An epoch is one input, stamped
`--interval` milliseconds (default 10) after the last unless given its own
time, and the schedule run up to the next input's time; signals still in
flight then join the next epoch. Each connection stamps the time of the
last signal its target integrated, the synapse's only trace. Each neuron
records `fired_in_wave`, the grid keeps this epoch's `Wave` objects in
`grid.waves`, the visualizer shades fired neurons by wave, and each output
neuron carries the time of its last spike (`grid.output_times()`).
`--discharge` zeroes every potential between inputs. Checkpoints save the
clock, every neuron's potential and last two spikes, every synapse stamp,
the signals in flight and the dopamine, so a resumed run continues
mid-cascade rather than restarts.

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

grid = GridOfNeurons(across=8, rows=10)
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
grid = main(across=8, rows=10)
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
like the grid with `get_neuron_at(place, row)`. A free spread has no rows,
so input and output zones for it are still to be defined.

## Hexagonal columns

The prespread butter. The plane is tiled with hexagonal cells, one neuron
per cell, and every cell is extruded into a **column** of `--layers`
neurons, so the network lives in R3. A cell is one unit across: neurons sit
half a unit apart in the plane, and the layers of a column an eighth of a
unit apart, a quarter of the cell spacing, so a column reads as a compact
stack of neurons much closer to each other than to their neighbours. The
connection rule is

    same position:                       never
    horizontal distance <= 1 + epsilon:  always
    otherwise:                           small-world shortcuts only (--omega)

measured in the plane only, so the guarantee holds at any height: a neuron
is wired to every neuron of its own column and of the eighteen columns
around it, in every layer, and everything further away is reached only by
shortcuts drawn as on the grid. The bottom layer is the **input butter**
and the top layer the **output butter**: a word is `across x rows` bits per
layer, complement-coded and permuted as before, and the targets and critics
read the top layer the same way they read the top row.

With one layer the two surfaces coincide and the stack keeps the grid's
convention (bottom row in, top row out). It is then the hex grid itself:
the same neurons in the same order, the same eighteen guaranteed
connections per neuron with the same ids, and with the same seed the same
shortcuts, weights and permutation, so `--layers 1` reproduces a plain run
epoch for epoch. That was the test that the rule changed nothing but the
ruler.

```bash
walnutbutter --layers 3                    # 8 x 10 cells, three deep: 80 input neurons, 80 outputs
walnutbutter --layers 3 --engine arrays    # the array engine wraps a stack like any mesh
walnutbutter --layers 2 -a 7 -r 2 --ecc    # a code needs 14 input neurons, however they are arranged
```

Fan-out grows with depth: an interior neuron of an L-layer stack has 19L - 1
guaranteed connections, so the array engine earns its keep quickly.

The free spread of variable density below is kept in the library but no
longer sets any default; the columns are where input and output live now.

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

## Learning

**Dopamine** (AUTHORITY.md §6, the default, `--rule dopamine`). A neuron
whose previous spike was at `t_prev` and which fires again at `t`, a delay
`d = t - t_prev - refractory` past the end of its refractory period,
releases the gamma density of that delay, `d^(alpha-1) exp(-d/theta) /
(Gamma(alpha) theta^alpha)`, averaged over the hop that follows it so that
it is finite for every alpha (`--release-alpha` 2, `--release-theta` 1 ms
by default: about 0.3 for an instant refire, a peak a little later, then a
decay). Releases pool into one
global value that decays with `--dopamine-tau` (default 20 ms) and is read
without being depleted. The network's expectation of it is an exponential
moving average of the value over `--expectation-tau` (default 10 minutes),
starting at 0. At the refire,
every incoming synapse that carried a signal the neuron integrated since
its previous spike moves together by `lr * (dopamine - expected) *
release`, clipped to the weight range: REINFORCE's shape, presynaptic
activity x postsynaptic eligibility x global signal, with the eligibility
the neuron's own release and nothing traced back through the network.
`--order` says whether a wave's releases join the pool before its updates
read it (`release-first`, default) or after. An input neuron whose bit is
0 this epoch, one that should not fire, has the sign of its update
reversed when it refires, so above-expected dopamine punishes it, and
`--punish-gain` (default 2) times as hard as a reward (`--no-punish`
switches that off). Every weight also moves toward zero by
`--weight-decay` per epoch (default 1e-4): synapses that forget on their
own, so that sustained activity has to be earned (AUTHORITY.md §6.8). Every neuron learns this way,
forced inputs included: input, hidden and output neurons differ only in
where external connections land. At every input each neuron's potential is
nudged by exploration noise (`--sigma`, default 0.1; 0 switches it off).
Both engines do the same arithmetic and agree to the last bit.

**The external teacher** (`--rule teacher`, the default for the sustain
problems, AUTHORITY.md §6.10). The teacher scores the read: +0.25 for each
input neuron that is on when its bit is 1 or off when its bit is 0, −0.25
for each one that is not, so four inputs give a score of −1, −0.5, 0, 0.5
or 1. During the epoch a refire adds its release to the `eligibility` of
each gated incoming synapse instead of moving it; at the read every synapse
moves by `lr * score * eligibility` and the trace is cleared. The dopamine
pool still runs and is reported but decides nothing, and the bit-0
punishment is off, since the score already accounts for those neurons.

**The reinforce rule** (`--rule reinforce`) is the pre-alpha's global
reinforcement, factored out and kept for comparison. On a trained problem
(`--problem reversal`) the `Teacher` tells the network each epoch what the
top row should have shown, by default a **reversed** copy of the bottom-row
input (`--target reversed`; `copy`, `all-off` and `all-on` also exist), and
scores the fraction of output neurons that match; under the dopamine rule
it only scores and reports. Under the reinforce rule a single scalar
reward, the epoch's accuracy, is compared with a running average to give an
*advantage*, and every connection that carried a signal into a neuron that
was not a forced input moves by `lr * advantage * eligibility`, where the
eligibility is the target neuron's exploration noise (node-perturbation
REINFORCE) or, with `--eligibility hebb`, +1 if the target fired and -1 if
not. Forced inputs are never adjusted and weights stay within [-1, 1].
Under the schedule the mesh reverberates on its own, so no performance is
claimed for this rule any more.

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

**The trace.** Every scored run appends one line per epoch to a CSV next
to its checkpoint (`--trace FILE` to choose it, `--no-trace` to skip it):
epoch, time in ms, the dopamine value, the expectation, and the score.
In the window every neuron is coloured by the time of its last spike, red
at the end of the epoch cooling to blue over one epoch's length, and Space
pauses the free run at the end of an epoch to show its trace, a raster of
every spike, time across and neuron down; Space resumes.

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
rarely lowers it. Neurons forced this epoch are left out for that epoch, as
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
grid = main(across=24, rows=20, seed=1)
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
AUTHORITY.md   the specification: constants, substance, connectivity, signalling, activation, learning; the code follows it
src/walnutbutter/
  connection.py Connection: ID, source and target neurons, weight, is_active, kind
  neuron.py    Neuron: threshold, potential, refractory period, receive(), fire(), reset(); no leak
  clock.py     the clock's tolerance: when two moments are the same moment
  propagation.py Schedule: the time-ordered queue of signals in flight, run a wave at a time; propagate()
  dopamine.py  Dopamine: the global pool, its expectation, and the learning at a refire
  arrays.py    ArrayNetwork: the same network as numpy vectors and a scipy sparse matrix (--engine arrays)
  exploration.py the Box-Muller noise draws both engines share
  constants.py every global constant: the default network, the neuron's clock, the learning rule's knobs
  grid.py      GridOfNeurons: builds the rectangle of hexagons and wires up both rings of neighbours
  columns.py   HexColumns: the cells extruded into layers in R3 (--layers); bottom layer in, top layer out
  butter.py    WalnutButter: smears of neuron density (per unit cell) on the plane; shapes Rect and Disc
  cartesian.py CartesianNodes: the lattice, a scatter, or a placed recipe; reach wiring
  inputs.py    random bits, complement coding, parsing and formatting
  monitor.py   main(): build a grid and run its first epoch; run_epoch(): reset and present a new input
  learning.py  output targets, reward, the global-reinforcement rule, and Teacher
  problems.py  the problems (--problem): layout, inputs, and whether a Teacher trains the network
  persistence.py checkpoint() and restore() for learned weights
  visualizer.py hex geometry and pygame drawing: show() and save()
  cli.py       argument parsing and the `walnutbutter` command
  __main__.py  lets you run `python -m walnutbutter`
tests/         pytest tests for the neuron, the grid, and the command line
pyproject.toml project metadata, dependencies, and the command definition
```
