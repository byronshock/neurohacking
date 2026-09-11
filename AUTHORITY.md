# The authority

This file specifies walnutbutter. The code follows it: where the two
disagree, this file is right and the code has a bug. A change to the system
is made here first, then carried into the code and its tests.

Byron and Cedric own the substance of it. Claude keeps it and the code in
step. §0 is **non-negotiable**. Every other section says whether it is
**kept** (the connectivity and signalling scaffolding, agreed on
September 10, 2026, refactored lightly as needed) or **open** (to be
respecified: the constants, the activation rule, the learning rule). An
open section currently describes what the code does today, so that editing
it is editing a working system; where it conflicts with §0, §0 wins and the
code is behind.

Notation: a neuron $j$ has potential $p_j$, threshold $\theta_j$, and firing
rate estimate $r_j$. A connection $i \to j$ has weight $w_{ij}$. Time $t$ is
in nominal milliseconds. Every symbol in small capitals is a constant from
§1 and from `src/walnutbutter/constants.py`.

## 0. The Strong Statement — non-negotiable

The following are non-negotiable in this system. Everything else is tunable
parameters with sweeps.

* The inputs are the outputs. An "input" neuron, a "hidden" neuron, and an
  "output" neuron are defined only with respect to where external
  connections exist. The neurons themselves operate identically.
* A neuron integrates delta functions to determine its potential and fires
  when the integral exceeds a threshold, at which point it resets. This is
  an integrate-and-fire neuron. Note that there is no leak, not even a lazy
  leak.
* A neuron firing is followed by an absolute refractory period during which
  the neuron ignores its inputs and does not integrate them. This is a
  feedback control mechanism and computational feature of the system.
* A neuron becomes eligible for dopamine release when it fires. Dopamine is
  the global reward used for reinforcement learning. It is produced locally
  and consumed globally. Local production initiates when the neuron returns
  online from its refractory period and fires again. Maximum dopamine is
  released when the neuron fires immediately after the refractory period.
  The amount of dopamine released exponentially decays with time from
  1 unit at time t_first_fired + refractory_period + epsilon.

## 1. Global constants — open

One value each, for the whole network. `constants.py` is their only home in
the code; the constructors, the clock, the Teacher and the command line all
read from it, and `tests/test_constants.py` fails if any path grows a
literal of its own.

### 1.1 The default network

| constant | value | meaning |
|---|---|---|
| ACROSS | 8 | cells across; the input row has one neuron per coded bit |
| ROWS | 10 | rows of cells; input at the bottom, output at the top |
| OMEGA | 0.2 | proportion of all connections that are small-world shortcuts, $0 \le \omega < 1$ |
| REACH | 2 | lattice wiring: every pair within this many unit distances connects |
| WEIGHT_RANGE | [−1, 1] | random weights are drawn uniformly from this range, and learning clips to it |
| WEIGHT_EPSILON | 0.001 | under `--positive-weights` the range becomes [ε, 1]: no inhibition |

### 1.2 The neuron and its clock

| constant | value | meaning |
|---|---|---|
| THRESHOLD | 0.25 | $\theta$ every neuron starts with |
| MINIMUM_POTENTIAL | −1 | floor on $p$: inhibition and carried-over charge go no lower |
| TAU | 2 ms | leak time constant; $\infty$ switches the leak off |
| REFRACTORY | 5 ms | absolute refractory period |
| INTERVAL | 10 ms | spacing of inputs when no time is given |

### 1.3 Learning

| constant | value | meaning |
|---|---|---|
| TARGET | reversed | what the output should show, derived from the input (§4.3) |
| CRITIC | row | how the reward is judged (§6.2) |
| ELIGIBILITY | perturb | what the reward acts on: the exploration noise, or a Hebbian ±1 |
| LATE | count | what a signal arriving after its target fired earns (§6.4) |
| LR | 0.03 | learning rate |
| SIGMA | 0.1 | exploration noise: standard deviation of each neuron's starting potential |
| BASELINE_RATE | 0.05 | per-epoch update of the running reward baseline |
| HOMEOSTASIS | 10⁻⁶ | per-epoch rate a threshold drifts toward its target firing rate; 0 = off |
| TARGET_RATE | 0.5 | firing rate homeostasis aims for |
| UNSTICK | 10⁻³ | per-epoch rate a stuck output's threshold moves toward UNSTICK_TARGET; 0 = off |
| UNSTICK_TARGET | 0.5 | firing rate the un-sticking aims for |
| THRESHOLD_RANGE | [−5, 5] | limits on where homeostasis and un-sticking may move a threshold |
| RATE_MEMORY | 0.01 | per-epoch update of $r_j$: about the last 100 epochs |
| STUCK_BELOW, STUCK_ABOVE | 0.01, 0.99 | a neuron with $r_j$ outside this band is stuck |
| WINDOW | 200 | epochs the reported moving-average accuracy spans (reporting only) |

## 2. The substance — kept

Walnut butter is a substance spread on the plane. Its density is neuron
density: butter spread at unit density puts one neuron per unit cell of a
hexagonal lattice. Butter spread near other butter connects (§3). The goal
is to create walnut butter, not to solve a particular task: a task is a way
of watching whether the substance is alive and learning.

Three containers build a network from it, all sharing the same neurons,
connections, signalling and learning:

- **The hex grid.** A rectangle of ACROSS × ROWS hexagonal cells, one neuron
  each, pointy-top, odd rows shifted half a cell. Row 0 is the top.
- **Hexagonal columns** (`--layers N`). The same field of cells extruded
  into columns of $N$ neurons in $\mathbb{R}^3$: a cell is one unit across,
  neighbouring cells are half a unit apart in the plane, the layers of a
  column an eighth of a unit apart. With one layer the stack *is* the hex
  grid, bit for bit.
- **The lattice / a spread** (`--nodes`). Neurons at $(x, y)$ positions in
  unit distances: a hexagonal lattice, a random scatter, or the positions a
  butter recipe describes.

A neuron may sit in several input and output zones at once; structures are
permissive, never artificially restricted.

## 3. Connectivity — kept

Every connection is one-way, from a source to a target, with its own weight.
Between two neurons $i$ and $j$ there are two connections, $i \to j$ and
$j \to i$, each with an independent weight. No neuron connects to itself.
Connections have ids from 1 in the order they are made, so a seed rebuilds
the same network.

### 3.1 The guaranteed neighbourhood

- **Hex grid.** Every neuron connects to its six neighbours (the first ring)
  and to the twelve neighbours of those neighbours (the second ring):
  eighteen local targets for an interior cell.
- **Columns.** Same position, never; horizontal distance $\le 1 + \varepsilon$
  (measured in the plane, ignoring height), always. The radius of one unit
  takes in a neuron's own column and the eighteen columns around it, in
  every layer.
- **Lattice / spread.** Every ordered pair within REACH units connects. At
  unit density with REACH = 2 that is the same eighteen neighbours.

### 3.2 Shortcuts

Everything further away is reached only by small-world shortcuts. With $L$
guaranteed connections, $S = \mathrm{round}\big(\omega L / (1 - \omega)\big)$
shortcuts are added so that they are the fraction OMEGA of all connections.
Each runs one way from a random neuron to a random neuron that is not
itself, not a guaranteed partner, and not already a target of the source.
The lattice and a spread have no shortcuts.

### 3.3 Weights

Every weight is drawn independently and uniformly from WEIGHT_RANGE, in
connection-id order from the seeded stream, unless a fixed weight is given
to all. Learning (§6) keeps every weight inside WEIGHT_RANGE.

## 4. Signalling — kept

### 4.1 The clock

Time is in nominal milliseconds. Each input has a time $t_e$; the first is
at 0 and by default each is INTERVAL after the last. Propagation is
instantaneous: every wave of a cascade happens at $t_e$, and the clock
advances only between inputs. The time component of signalling is carried
by the refractory period and the leak, not by delays.

### 4.2 An epoch

An epoch is one input and the cascade it causes. In order:

1. **Reset.** Every neuron's fired state is cleared. A neuron that did not
   fire keeps its sub-threshold potential; one that fired had its potential
   reset by the spike. (`--discharge` zeroes every potential instead.) The
   time of the last spike is kept: the refractory period outlives the
   cascade.
2. **Input.** The pattern is placed on the input neurons (§4.3).
3. **Explore.** Each potential is leaked to $t_e$ (§5.1) and then nudged by
   the exploration noise (§6.1), floored at MINIMUM_POTENTIAL.
4. **Cascade.** The clock is set to $t_e$ and the waves run (§4.4) until the
   queue is empty. The epoch ends there.
5. **Learn.** The output is read, scored, and every connection updated (§6).

### 4.3 Input and output

The network's input is its bottom row (the bottom layer, for columns); its
output is its top row (top layer). An input is $k$ = ACROSS / 2 raw bits,
drawn as fair coin flips from the seeded stream (or given). They are
**complement-coded**, the bits followed by their negations, so $2k$ bits
reach the row and exactly half of it fires whatever the raw bits. The coded
bits are then **permuted** by a random permutation drawn once per network
and fixed for its life: place $i$ along the row shows coded bit $\pi(i)$.
With an error-correcting code (`--ecc`), 4 data bits are first encoded to 7
(Hamming) or 6 (parity) before complement coding.

The neurons whose bit is 1 are **forced** to fire in wave 0. The target is
derived from the input row: by default its reverse (TARGET = reversed);
also copy, all-off, all-on.

### 4.4 Waves

Firing is queued, never recursive. Wave 0 is the stimulus: each forced
neuron fires unless it is refractory (§5.3). The signals the neurons firing
in wave $n$ send are delivered in wave $n+1$, and a wave has two phases:

1. **Deliver.** Every queued connection $i \to j$ delivers $w_{ij}$ to $j$
   (§5.1). Then every neuron touched this wave settles: its potential is
   floored at MINIMUM_POTENTIAL, so the floor acts on the wave's summed input
   and the result does not depend on the order the signals arrived in.
2. **Fire.** Every touched neuron whose potential now meets its threshold,
   and is not refractory, fires (§5.2), and its active outgoing connections
   are queued for the next wave.

A neuron fires at most once per cascade. Once it has fired, or while it is
refractory, it ignores every signal, forced stimulus included; the delivery
is still recorded (it matters to §6.4). A cascade ends when a wave queues
nothing.

## 5. Activation rule — open

Everything below is what the code does today. §0 overrides it: the neuron
is integrate-and-fire with **no leak** (§5.1 and TAU go), the refractory
period stays and is a feature, and a neuron becomes eligible for dopamine
release when it fires.

### 5.1 Integration, with a lazy leak

Nothing happens to a quiet neuron. When a signal of weight $w$ arrives at
time $t$, the potential is first decayed for the time since it was last
brought up to date, then the signal is added:

$$p \leftarrow p \, e^{-(t - t_{\text{last}}) / \tau}, \qquad t_{\text{last}} \leftarrow t, \qquad p \leftarrow p + w .$$

Within a cascade every arrival is at the same $t$, so the leak acts only
across the gap between inputs. TAU = $\infty$ switches it off. Inhibition
($w < 0$) pushes the potential down, and once per wave the floor applies:
$p \leftarrow \max(p, \text{MINIMUM\_POTENTIAL})$.

### 5.2 Firing

A neuron fires in a wave iff $p \ge \theta$, it has not fired in this
cascade, and it is not refractory. The spike resets it:

$$p \leftarrow 0, \qquad t_{\text{fired}} \leftarrow t, \qquad t_{\text{last}} \leftarrow t .$$

A forced input neuron fires in wave 0 regardless of $p$ and $\theta$.

### 5.3 Refractory period

A neuron that fired at $t_{\text{fired}}$ is refractory while
$t < t_{\text{fired}} + \text{REFRACTORY}$. While refractory it ignores every
signal and cannot be forced. With inputs INTERVAL = 10 ms apart and
REFRACTORY = 5 ms, a neuron that fired last epoch is free again by the next.

## 6. Learning rule — open

Global reinforcement: one scalar reward per epoch, broadcast to every
connection, combined with each neuron's own exploration noise. Nothing is
traced back through the network. Everything below is what the code does
today. §0 overrides the source of the reward: it is dopamine, produced
locally by neurons that fire again after their refractory period and
consumed globally (§6.2's critic is not the reward). How the consumed
dopamine moves a weight is not yet specified.

### 6.1 Exploration

Before the cascade, every neuron $j$ (inputs included) draws
$\xi_j \sim \mathcal{N}(0, \sigma^2)$ with $\sigma$ = SIGMA and adds it to
its (leaked) potential, floored. It remembers $\xi_j$ for the epoch. Both
engines draw from the same Box-Muller stream, so a seed gives the same noise
whichever engine runs.

### 6.2 Reward and advantage

The **row** critic scores the epoch as the fraction of output neurons whose
fired/not-fired state matches the target pattern: $R \in [0, 1]$, and firing
nothing scores 0.5. (The decoded critics read the row as a code word and
error-correct it first.) A running baseline $b$ tracks what usual looks
like: on the first epoch $b = R$, and after each update

$$b \leftarrow b + \text{BASELINE\_RATE}\,(R - b) .$$

The **advantage** is $A = R - b$, taken before that update.

### 6.3 The weight update

For every connection $i \to j$ that carried a signal this epoch ($i$ fired)
into a neuron $j$ that was not a forced input:

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR} \cdot A \cdot e_j,\ \text{WEIGHT\_RANGE}\big),
\qquad e_j = \xi_j / \sigma .$$

This is node-perturbation REINFORCE, a three-factor rule: presynaptic
activity × postsynaptic perturbation × global reward. A neuron nudged toward
firing in an epoch better than usual gets stronger inputs from the neurons
that fed it; in a worse epoch, weaker. With ELIGIBILITY = hebb, $e_j = +1$
if $j$ fired and $-1$ if not, and no noise is injected. Weights into forced
inputs are never touched: their firing was not the network's doing.

### 6.4 Late signals

A signal that arrives after its target has already fired changed nothing
in the cascade, but its connection was active with pre and post both fired
this epoch. LATE says what it earns: **count** (default) the same update as
a signal that landed, a local Hebbian term riding on the estimator, which
learned faster on every seed tried; **ignore** nothing, the estimator
proper; **depress** the opposite update, the shape of spike-timing-dependent
plasticity.

### 6.5 Firing rates, homeostasis, un-sticking

Every neuron tracks its own firing rate, starting at 0.5, skipping epochs it
was forced:

$$r_j \leftarrow r_j + \text{RATE\_MEMORY}\,(f_j - r_j), \qquad f_j \in \{0, 1\}.$$

Each epoch every unforced neuron's threshold drifts toward its target rate,
and every **stuck** output neuron ($r_j$ outside [STUCK_BELOW, STUCK_ABOVE])
is nudged faster, both clipped to THRESHOLD_RANGE:

$$\theta_j \leftarrow \theta_j + \text{HOMEOSTASIS}\,(r_j - \text{TARGET\_RATE}),
\qquad
\theta_j \leftarrow \theta_j + \text{UNSTICK}\,(r_j - \text{UNSTICK\_TARGET}) .$$

Firing too often raises the threshold; too rarely lowers it. A saturated
neuron gets no learning signal because the noise never changes whether it
fires; moving its threshold back to where the noise matters restores a
gradient there.

### 6.6 What is reported

The reward of every epoch, its mean to date, and an exponential moving
average over about WINDOW epochs. The network runs forever: these are read
as health, not convergence, and drift is normal.

## 7. Invariants the scaffolding guarantees — kept

- **Two engines, one network.** The object engine (neurons and a queue of
  waves) and the array engine (numpy vectors, a scipy sparse matrix) run the
  same network: same ids, same firing wave by wave, same weights to
  $10^{-12}$, and `tests/test_arrays.py` runs them side by side. A new rule
  is implemented in both and must pass the same tests.
- **A seed is the whole run.** Shortcuts, weights, permutation, inputs and
  exploration noise all come from the seed's stream, in both engines.
- **Checkpoints round-trip.** A checkpoint rebuilds the network from its
  seed and settings and reloads its weights, thresholds and clock, in
  either engine.
- **The network keeps living.** There is no training run and no evaluation
  run, only one run that keeps going; a rule may not assume an end.
