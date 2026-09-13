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
  leak. *Revised (Byron, September 12, 2026):* bring back the leaky
  integrate-and-fire neuron; the infinite impulse response without the
  leak is not wanted. The leak is lazy, with time constant TAU (§5.1).
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
| TAU | 2 ms | leak time constant of the potential, computed lazily on arrival; $\infty$ switches it off (§5.1) |
| MINIMUM_POTENTIAL | −1 | floor on $p$: inhibition and carried-over charge go no lower |
| REFRACTORY | 5 ms | absolute refractory period |
| REFRACTORY_HOPS | 3 | the refractory period divided by the time a signal takes to travel one hop; not an integer, started at 3 |
| INTERVAL | 10 ms | spacing of inputs when no time is given |
| BORED_AFTER | 200 ms | silence after which a neuron's threshold has fallen to zero and it fires on its own (§5.4) |

### 1.3 Learning

| constant | value | meaning |
|---|---|---|
| RULE | teacher | which learning rule runs: teacher (§6.10), dopamine (§6.2-6.6) or reinforce (§6.7) |
| TEACHER_CREDIT | 0.25 | what each input neuron read correctly adds to the teacher's score, and each one read wrongly subtracts |
| LR | 0.03 | learning rate |
| SIGMA | 0.1 | exploration noise: standard deviation added to each neuron's potential at every input |
| DOPAMINE_RELEASE_ALPHA, DOPAMINE_RELEASE_THETA | 2, 1 ms | shape and scale of the gamma density that gives the amount released against the refire delay past the refractory period |
| DOPAMINE_TAU | 20 ms | decay of the global dopamine value |
| DOPAMINE_EXPECTATION_TAU | 10 min | exponential window of the expected dopamine trace |
| DOPAMINE_EXPECTATION_START | 0 | where that trace starts; a high start holds early learning back (Byron, September 12, 2026) |
| DOPAMINE_ORDER | release-first | at a refire, release before the weight update (or update-first) |
| DOPAMINE_PUNISH | on | an input neuron whose bit is 0 has its update reversed when it refires |
| DOPAMINE_PUNISH_GAIN | 2 | and that reversed update is this many times a reward (§6.8) |
| WEIGHT_DECAY | 10⁻⁴ | every weight moves toward 0 by this fraction each epoch: synapses that forget (§6.8) |

The reinforce rule, factored out behind RULE = reinforce, keeps its own:

| constant | value | meaning |
|---|---|---|
| TARGET | reversed | what the output should show, derived from the input (§4.3) |
| CRITIC | row | how the reward is judged (§6.7) |
| ELIGIBILITY | perturb | what the reward acts on: the exploration noise, or a Hebbian ±1 |
| LATE | count | what a signal arriving after its target fired earns (§6.7) |
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

- **Hex grid.** Every neuron connects to every cell within `reach` hex
  steps: with the default reach 2, its six neighbours (the first ring) and
  the twelve neighbours of those neighbours (the second ring), eighteen
  local targets for an interior cell; reach 3 adds the eighteen cells of
  the third ring, thirty-six in all.
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

## 4. Signalling — kept, on a schedule

### 4.1 The clock

Time is in nominal milliseconds. A signal takes one **hop** to travel a
connection, $h = \text{REFRACTORY} / \text{REFRACTORY\_HOPS}$ (Byron,
September 11, 2026; not an integer, started at 3). Each input has a time
$t_e$; the first is at 0 and by default each is INTERVAL after the last.
There is no other delay: the time component of signalling is carried by
the hop and the refractory period.

*Decided (September 11, 2026):* a wave is "everything that happens at time
$t$", not "everything one hop after the last wave". With a non-integer
ratio, refractory recovery and hop arrivals fall at different times, and a
second input can land mid-cascade. The queue is therefore a time-ordered
**schedule** of signals, and a wave is the batch at the front with the same
timestamp, including any external signals stamped for that moment.
Same-time signals sum within a wave before anyone fires, which keeps the
floor and the firing decision order-independent; signals one hop apart are
separate waves. Epochs stop being the unit: a cascade from one input can
still be running when the next input's signals join the schedule, so "the
queue empties, then the clock advances" no longer holds. Learning has its
own cadence (§6). Times are rounded to a nanosecond so that two signals
computed to arrive at the same moment are the same moment in both engines.

### 4.2 An epoch

An epoch is one input and the schedule run up to the next input's time, the
**horizon**, $t_e + \text{INTERVAL}$ by default. Signals due at or after the
horizon wait for the next epoch. An input may not be given a time before
the horizon the schedule has already run to. In order:

1. **Reset.** Every neuron's fired-this-epoch state is cleared. Potentials
   are kept (`--discharge` zeroes them instead). Spike times are kept: the
   refractory period outlives the epoch. Signals in flight stay scheduled.
2. **Input.** The pattern is placed on the input neurons (§4.3), stamped
   $t_e$.
3. **Explore.** Each potential is nudged by the exploration noise (§6.1),
   floored at MINIMUM_POTENTIAL.
4. **Run.** The schedule runs wave by wave (§4.4) until the horizon; a
   neuron that refires learns as it fires (§6).
5. **Report.** The output is read (and, under the reinforce rule, scored
   and reinforced, §6.7).

### 4.3 Input and output

The network's input is its bottom row (the bottom layer, for columns); its
output is its top row (top layer). Per §0 the neurons are the same
everywhere; only the external connections differ. An input is $k$ =
ACROSS / 2 raw bits, drawn as fair coin flips from the seeded stream (or
given). They are **complement-coded**, the bits followed by their
negations, so $2k$ bits reach the row and exactly half of it fires whatever
the raw bits. The coded bits are then **permuted** by a random permutation
drawn once per network and fixed for its life: place $i$ along the row
shows coded bit $\pi(i)$. With an error-correcting code (`--ecc`), 4 data
bits are first encoded to 7 (Hamming) or 6 (parity) before complement
coding.

(A problem may instead lay the raw bits down as they are, `coding = raw`:
sustain_inputs does, §8. A problem also says what "on" means at the read:
fired this epoch, spiked again after the input's moment, or fired within a
window before the horizon.) The neurons whose bit is 1 are **forced** to fire
at $t_e$, refractory period permitting. A neuron forced this epoch is marked as such, which only
the reinforce rule (§6.7) consults.

### 4.4 Waves

Firing is queued, never recursive. A wave is every event scheduled for one
time $t$: the signals arriving, and the stimulus if an input lands then. It
has two phases:

1. **Deliver.** Every arriving signal $i \to j$ delivers $w_{ij}$ to $j$
   (§5.1). Then every neuron touched this wave settles: its potential is
   floored at MINIMUM_POTENTIAL, so the floor acts on the wave's summed input
   and the result does not depend on the order the signals arrived in.
2. **Fire.** Every forced neuron fires unless refractory; then every touched
   neuron whose potential now meets its threshold, and is not refractory,
   fires (§5.2). Each firing neuron's active outgoing connections are
   scheduled for $t + h$. The refires in the wave then learn (§6).

A neuron may fire any number of times, the refractory period permitting; a
tight loop of about REFRACTORY_HOPS hops can bring a neuron's own spike
back to refire it. A refractory neuron ignores every signal, forced
stimulus included; the delivery is still recorded as such. Each connection
stamps the time of the last signal its target actually integrated: that is
the only trace of activity the target has (§6.4).

## 5. Activation rule — open

Per §0 as revised, the neuron is leaky integrate-and-fire with an absolute
refractory period.

### 5.1 Integration, with a lazy leak

Nothing happens to a quiet neuron. When a signal of weight $w$ arrives at
time $t$, the potential is first decayed for the time since it was last
brought up to date, then the signal is added:

$$p \leftarrow p \, e^{-(t - t_{\text{last}}) / \tau}, \qquad t_{\text{last}} \leftarrow t, \qquad p \leftarrow p + w,$$

with $\tau$ = TAU (Byron, September 12, 2026, bringing the leak back:
"I don't like the infinite impulse response without the leak"; the default
is the 2 ms his earlier sweep chose, and a hop is now 1.7 ms, so the leak
acts within a cascade as well as between inputs). The exploration noise
(§6.1) is added on top of the leaked potential at the input's moment, and
a bored neuron's check (§5.4) compares the leaked potential against the
falling threshold. Inhibition ($w < 0$) pushes the potential down, and
once per wave the floor applies: $p \leftarrow \max(p, \text{MINIMUM\_POTENTIAL})$.

### 5.2 Firing

A neuron fires in a wave iff $p \ge \theta$ and it is not refractory. The
spike resets it and is remembered, along with the spike before it:

$$p \leftarrow 0, \qquad t_{\text{prev}} \leftarrow t_{\text{fired}}, \qquad t_{\text{fired}} \leftarrow t .$$

A forced input neuron fires at its input's time regardless of $p$ and
$\theta$, refractory period permitting.

### 5.3 Refractory period

A neuron that fired at $t_{\text{fired}}$ is refractory while
$t < t_{\text{fired}} + \text{REFRACTORY}$. While refractory it ignores every
signal, does not integrate it, and cannot be forced. This is a feedback
control mechanism and computational feature of the system (§0).

### 5.4 Threshold homeostasis: bored neurons — decided for now

*Decided (Byron, September 12, 2026):* we need a form of threshold
homeostasis so bored neurons tend to fire after about 200 ms all on their
own. *Claude's reading, built:* the threshold a neuron faces falls
linearly with the silence since its last spike,

$$\theta_j(t) = \theta_j \Big(1 - \frac{t - t_{\text{fired}}}{\text{BORED\_AFTER}}\Big),$$

reaching zero at BORED_AFTER and continuing below it, so a neuron whose
potential sits under zero fires later still; a spike resets it to
$\theta_j$. A neuron that has never fired counts as silent since the
clock started. Every wave now checks every neuron, touched or not, so a
bored neuron fires at the first wave after its threshold has fallen to
its potential: in a live mesh within a hop, in a dead one at the next
input. (Before this, a neuron was only checked when a signal reached it,
which also left a neuron recovered from its refractory period with enough
potential waiting to be touched; both engines now fire it at once.) The
spike it fires on its own is a spike like any other: it releases by its
delay, which at 200 ms is nothing, and it resets the clock on its
boredom.

## 6. Learning rule — open

Per §0: a neuron becomes eligible for dopamine release when it fires.
Dopamine is the global reward used for reinforcement learning. It is
produced locally and consumed globally.

*Decided (Byron, September 11, 2026):* Learning happens when a neuron that
previously fired fires again and is proportional to the global dopamine
value. We can take care of this when the neuron fires. The strategy of
REINFORCE remains the same as much as we can keep it with the global
dopamine reinforcer.

*Decided (Byron, September 12, 2026):* the dopamine eligibility is
calculated both locally and lazily. It is a property of a neuron. Signals
are NOT traced back to their ancestors. The only trace available to the
neuron is the activity across its synapses.

*Decided FOR NOW (Byron, September 12, 2026):* the eligibility is the same
for all of a neuron's weights. For now, dopamine is released and then all
weights move together. *Open question:* whether the dopamine release or the
weight update should happen first.

### 6.1 Exploration

At every input, every neuron $j$ (inputs included) draws
$\xi_j \sim \mathcal{N}(0, \sigma^2)$ with $\sigma$ = SIGMA and adds it to
its potential, floored. Tunable; 0 switches it off. Both engines draw from
the same Box-Muller stream, so a seed gives the same noise whichever engine
runs.

### 6.2 Release

A neuron whose previous spike was at $t_{\text{prev}}$ and which fires
again at $t$ (necessarily $t \ge t_{\text{prev}} + \text{REFRACTORY}$), a
delay $\delta = t - t_{\text{prev}} - \text{REFRACTORY}$ past the end of its
refractory period, releases the gamma density of that delay (Byron,
September 12, 2026, replacing the exponential
$e^{-\delta / \text{DOPAMINE\_RELEASE\_TAU}}$):

$$d_j = \frac{\delta^{\alpha - 1} e^{-\delta / \theta}}{\Gamma(\alpha)\,\theta^{\alpha}},
\qquad \alpha = \text{DOPAMINE\_RELEASE\_ALPHA} = 2,\ \theta = \text{DOPAMINE\_RELEASE\_THETA} = 1\text{ ms}.$$

*Claude's implementation note:* the density is averaged over the hop that
follows the delay, $d_j = \big[F(\delta + h) - F(\delta)\big] / h$ with $F$
the gamma distribution function and $h$ the hop. That is the density as the
hop shrinks, and it is finite for every $\alpha$: for $\alpha < 1$ the
density itself is infinite at zero delay, and an instant refire is the
common case here (the first sweep arm at $\alpha = 0.5$ read "dopamine
inf" within a minute). With $\alpha = 2$, $\theta = 1$ ms an instant refire
releases about 0.3, the release peaks a little later, and it decays from
there. *Tension with §0*, which says maximum dopamine is released when the
neuron fires immediately after the refractory period: for $\alpha > 1$ the
maximum comes later, and $\alpha \le 1$ recovers §0's shape. For Byron and
Cedric to settle in §0. A neuron's first ever spike releases nothing.
Forced neurons are neurons (§0): a forced refire releases and learns like
any other.

### 6.3 The global value and the expectation

The global dopamine value $D(t)$ is a pool every release adds to, decaying
lazily with DOPAMINE_TAU: $D(t) = D(t_0)\, e^{-(t - t_0)/\tau_D}$ between
events. Reading it does not deplete it.

*Decided (Byron, September 12, 2026):* instead of a baseline, actual
reinforcement learning with an internal expectation, `dopamine_expected`,
calculated globally under the assumption that dopamine is global. For now,
the actual counts (not biologically realistic) of dopamine to date divided
by the current time. The reinforcement is proportional to
(dopamine − dopamine_expected), not to dopamine.

*Decided (Byron, September 12, 2026), replacing the counts-over-time
estimator, which could only creep:* start with dopamine_expected = 0 and use
an exponential decay window of 10 minutes for the expected dopamine trace.
At every wave, after the wave's releases and updates, the expectation moves
toward the value by the fraction of the window that has elapsed since the
last wave:

$$E \leftarrow E + \big(1 - e^{-\Delta t / \tau_E}\big)\,(D(t) - E),
\qquad A(t) = D(t) - E ,$$

with $\tau_E$ = DOPAMINE_EXPECTATION_TAU and $A$ read before the move. This
may start off slow but may eventually get us where we are trying to get.

### 6.4 Eligibility — for now

One value per neuron, computed when it refires, from nothing but its own
spikes and the stamps on its synapses (§4.4): $e_j = d_j$, its own release
amount. A tight refire is both the biggest release and the biggest
eligibility.

### 6.5 The update — gated, for now

When neuron $j$ refires at $t$, every incoming connection $i \to j$ that
carried a signal $j$ integrated since its previous spike (the gate,
$x_i = 1$; otherwise $x_i = 0$) moves together:

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR} \cdot A(t) \cdot e_j \cdot x_i,\ \text{WEIGHT\_RANGE}\big).$$

This keeps REINFORCE's shape, presynaptic activity × postsynaptic
eligibility × global signal, with the global signal now $A(t)$.

*Decided (Byron, September 12, 2026):* the system has no knowledge of how
it is being scored: a neuron that should fire is forced, checked, and
rewarded or anti-rewarded by the rule, but a neuron that SHOULD NOT FIRE
(input bit 0) is free to be recruited into sustaining the others. For
neurons that should not fire, the learning rule is applied but the reward
is reversed upon firing: if the dopamine is greater than expected, the
neuron is punished for firing instead of rewarded. In the update above,
$\text{LR}$ takes a minus sign for a refiring input neuron whose bit this
epoch is 0. Its release, and hidden and forced neurons, are unchanged.
DOPAMINE_PUNISH switches it (`--no-punish`).

### 6.6 Order within a wave

All the refires in one wave are handled together: their releases are
computed, then (DOPAMINE_ORDER = release-first) added to the pool before
$A(t)$ is read and the updates applied, or (update-first) after. Release
first lets a neuron's own release count toward what it consumes; update
first keeps the signal strictly what others released. A switch, to be
swept.

### 6.7 The reinforce rule — factored out

RULE = reinforce is the rule of the pre-alpha, kept for comparison and run
by the Teacher on a trained problem (§8). One scalar reward per epoch: the
**row** critic scores the fraction of output neurons whose fired state this
epoch matches the target pattern (decoded critics read the row as a code
word first). A running baseline $b$ (BASELINE_RATE) gives the advantage
$A = R - b$. For every connection $i \to j$ that carried a signal this
epoch into a neuron $j$ not forced this epoch,
$w_{ij} \leftarrow \mathrm{clip}(w_{ij} + \text{LR} \cdot A \cdot e_j)$ with
$e_j = \xi_j/\sigma$ (ELIGIBILITY = perturb) or $\pm 1$ by whether $j$ fired
(hebb). LATE says what a signal arriving after its target fired earns:
count, ignore, or depress. The Teacher also keeps each unforced neuron's
firing rate ($r_j$, RATE_MEMORY) and drifts thresholds toward TARGET_RATE
(HOMEOSTASIS), nudging stuck outputs faster (UNSTICK), within
THRESHOLD_RANGE. Under RULE = dopamine the Teacher only scores and reports;
the network learns by §6.2–6.6.

### 6.8 Earned activity — decided for now

*Proposed by Claude, September 12, 2026, after the punishment rule (§6.5)
moved the weights into the input row but not the score; decided by Byron
the same day: "I do want synapses that forget on their own. For now."* The
diagnosis:
under these constants the mesh reverberates on its own, so every input
neuron is driven every 6 ms whatever its bit, and the same synapses carry
that drive in an epoch where it should fire and in one where it should
not. Reward on the one and punishment on the other land on the same
weights and cancel. Nothing a neuron can see distinguishes the two
epochs except its own forced spike, and the rule gives that spike no role
because the rest of the mesh fires the same either way. Two decisions
together would give it one; either alone does not.

1. **Punishment outweighs reward.** A refire that should not have
   happened is punished DOPAMINE_PUNISH_GAIN times as hard as a refire that
   should have is rewarded (proposed 2). For an input neuron driven by an
   autonomous loop, the net over its bit-1 and bit-0 epochs is then
   negative, and its incoming weights fall until the loop no longer fires
   it unforced. For a loop that only runs when the neuron's own forced
   spike starts it, there are no bit-0 refires to punish, the net is
   positive, and it grows. The fixed point of the rule is the boundary the
   task asks for: on when forced, off when not. Alone this fails, because
   in a fully reverberating mesh the drive does not depend on the forced
   spike, so the neuron falls silent in both cases and the score stays at
   chance.

2. **Weights decay toward zero.** Every weight moves toward 0 by a fraction
   WEIGHT_DECAY per epoch (proposed $10^{-4}$: about two hundred epochs of
   decay per learning step at LR 0.02). Activity that is not rewarded on
   net dissolves, the intrinsic reverberation included, so sustained
   activity has to be earned, and the only structures that earn it on net
   are loops that depend on a forced spike. Alone this fails too: a quiet
   mesh gives nothing to reward until exploration noise finds a refire, and
   without (1) an autonomous loop, once found, is net neutral and survives.

Together: decay empties the mesh of unearned reverberation, exploration
seeds refires, reward grows the loops that a forced spike starts, and the
asymmetric punishment dissolves any loop that learns to run without one.
Both are single constants and stay local: DOPAMINE_PUNISH_GAIN
(`--punish-gain`) and WEIGHT_DECAY (`--weight-decay`), the decay applied
to every weight once per epoch, after the epoch's waves, in both engines.
The alternative not taken: a global inhibition proportional to the mesh's
activity, which also makes reverberation something to be earned but
couples every neuron to the whole.

### 6.9 The external teacher — decided, the current rule

*Decided (Byron, September 12, 2026):* the dopamine student-as-teacher is
very hard to figure out for now, so revert to an external teacher. This
teacher assigns reinforcement globally according to a score:

- $+0.25$ for a forced-input neuron that sustains,
- $+0.25$ for an input neuron whose input is zero and does not fire,
- $-0.25$ for a forced-input neuron that fails to sustain,
- $-0.25$ for an input neuron whose input is zero and does fire.

Thus the possible scores for any input on reading its sustain behaviour are
$-1$, $-0.5$, $0$, $0.5$ and $1$ (four input neurons at TEACHER_CREDIT
each; a wider zone widens the range in the same steps).

*Claude's reading, built:* the score is known only at the read, so the
epoch's credit is carried by an **eligibility trace**. Everything in
§6.2–6.5 stands except who supplies the global signal and when the weights
move. During the epoch a refiring neuron's release (§6.2) is added to the
`eligibility` of each of its gated incoming synapses (the gate of §6.5: a
synapse that carried a signal the neuron integrated since its previous
spike) instead of moving that synapse. At the read the teacher computes the
score $S$ and every synapse moves by

$$w_{ij} \leftarrow \mathrm{clip}\big(w_{ij} + \text{LR} \cdot S \cdot e_{ij},\ \text{WEIGHT\_RANGE}\big),$$

with $e_{ij}$ the trace, which is then cleared: an epoch's credit never
carries into the next. The dopamine pool still runs and is still reported,
but it no longer decides anything. The §6.5 punishment of a bit-0 input
neuron is switched **off** under this rule, because the teacher's score
already knows which neurons should not have fired and reversing the sign
locally would double-count it; `--punish` turns it back on. Weight decay
(§6.8) and the bored neurons (§5.4) are unchanged. A problem names the rule
it is posed for: reversal the reinforce rule, sustain_inputs and
improved_sustain the teacher.

### 6.10 A cycle has been found — noted, not implemented

*Byron and Cedric, September 13, 2026:* we have a new learning rule. When a
forced-input neuron fires again within the epoch, a cycle has been found.
Not to be implemented right now; noted here as the next rule to try.

The finding it answers is the one the sweeps of September 12 and 13 kept
arriving at (`docs/findings-2026-09-13.md`): nothing a neuron can observe
differs between an epoch where it should fire and one where it should not,
because the mesh around it behaves the same either way and its own forced
spike reaches nothing but itself. A refire *within the epoch that forced
it* is the one event that cannot happen without that neuron's own spike
having travelled out and come back. It is pattern-dependent by
construction, it is local to the neuron that observes it, and it is exactly
what the task asks the substance to do.

### 6.11 What is reported

Spikes to date, the neurons fired this epoch, the dopamine value and its
expectation, releases and updates to date; the score of every epoch, its
mean to date, and an exponential moving average over about WINDOW epochs;
and a per-epoch trace file of epoch, time, dopamine, expected and score.
The window colours every neuron by the time of its last spike, hot (red)
at the end of the epoch cooling to blue over an epoch's length, and Space
pauses the free run at the end of an epoch to show its trace, a raster of
every spike. The network runs forever: these are read as health, not
convergence, and drift is normal.

## 7. Invariants the scaffolding guarantees — kept

- **Two engines, one network.** The object engine (neurons and a queue of
  waves) and the array engine (numpy vectors, a scipy sparse matrix) run the
  same network: same ids, same firing wave by wave, same weights to
  $10^{-12}$, and `tests/test_arrays.py` runs them side by side. A new rule
  is implemented in both and must pass the same tests.
- **A seed is the whole run.** Shortcuts, weights, permutation, inputs and
  exploration noise all come from the seed's stream, in both engines.
- **Checkpoints round-trip.** A checkpoint rebuilds the network from its
  seed and settings and reloads its weights, thresholds, clock, spike
  times, synapse stamps, signals in flight and dopamine, in either engine.
- **The network keeps living.** There is no training run and no evaluation
  run, only one run that keeps going; a rule may not assume an end.

## 8. Problems — open

A problem is what a network is asked to do and how it is watched doing it:
the layout, the inputs, and whether anything outside the network trains it.
`--problem` picks one; PROBLEM (§1) is the default.

- **reversal** (default today). The 8 × 10 hex grid; 4 random bits,
  complement-coded and permuted onto the bottom row; the top row is taught
  to show the bottom row reversed by a Teacher with a target and a critic
  (§6.2). The task of the pre-alpha, kept as the baseline.
- **sustain_inputs** (Byron, September 12, 2026). The same 16 inputs will
  be used across 8 neurons. However, this network is not trained
  externally: the neurons will utilize the new eligibility rule (§6).

  *The training epoch (Byron, September 12, 2026):* the sustain_inputs task
  receives an input on the same forced neurons, reverberates for 20
  milliseconds, and the same inputs (the inputs are the outputs) are read
  so we have a trace of the "score"; the previous Teacher metric computes
  the score even though it is not used for reinforcement. Time or epoch
  number, expected dopamine value, and score are recorded for each
  training epoch.

  *Claude's reading of the read:* the outputs are the input row, the target
  is the input pattern itself (copy), and a neuron is read as on if it
  fired within the last refractory period before the read at the end of
  the 20 ms, so the forced spike itself does not count and only a
  sustained neuron scores. Nothing outside the network moves a threshold
  (homeostasis and un-sticking off). The record goes to a CSV next to the
  checkpoint: epoch, time, dopamine, expected, score.

  *Decided (Byron, September 12, 2026), replacing the row critic:* what we
  are interested in is: did the neurons that were forced to fire sustain?
  Only the four neurons that were forced to fire are scored. The score is
  the **sustained** critic: of the input neurons the pattern forced, the
  fraction read as on at the end of the epoch. The unforced four are not
  scored at all. (Under the row critic a uniform row, all on or all off,
  scored 0.5 whatever was forced, which is why every sweep to this point
  read 0.500.)

  *Decided (Byron, September 12, 2026), simplifying the task:* rather than
  complement-coding the inputs, the sixteen inputs are presented as 4 bits
  each, laid down as they are on a 4-across, 10-row grid (no complement
  coding: 0000 forces nothing, 1111 forces all four). The score is the
  original row critic over all four input neurons, forced or not: the
  forced ones should be on at the read and the others off. The `sustained`
  critic stays available (`--critic sustained`).

  *Decided (Byron, September 12, 2026), replacing the read window:* the
  read asks whether the neuron fired again at all, rather than whether it
  fired in the last 5 ms. A neuron is on at the read if it spiked at any
  moment strictly after the input's moment $t_e$: for a forced neuron, a
  refire; for an unforced one, any spike in the epoch. (The 5 ms window read
  one phase in three of a fully sustaining neuron as off, on a 2 ms hop
  grid with a 6 ms cadence: the 2/3 ceiling of the low grid.) The window
  read stays available to a problem as `read = "window"`.

- **improved_sustain** (Byron, September 12, 2026). A different topology
  for this problem: the hex grid consists of 7 rows by 10 across. The input
  is presented in the MIDDLE, row 4, places 4 through 7 counted from 1
  (row 3, places 3 to 6 counted from 0: the middle four of the middle row).
  The reach is 3: every neuron is wired to every cell within three hex
  steps (§3.1). No permutation, the bits land where they are. Everything
  else as sustain_inputs: raw coding, 20 ms epochs, the same four neurons
  read back as spiked again, the row critic, learning by dopamine, nothing
  outside the network training it. A problem may name any input zone, a
  list of (place, row) cells (`Network.set_input_cells`), in place of the
  bottom row; a neuron may sit in several zones at once.
