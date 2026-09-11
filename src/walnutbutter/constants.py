"""Every global constant of walnutbutter, in one place.

These are the defaults the constructors, the neuron's clock, the Teacher and
the command line all read from: change a value here and every path follows.
They are the physics of the substance (what a neuron needs to fire, how fast
it leaks, how learning moves a weight) and the shape of the default network.
Lattice geometry (cell spacing, row spacing, the guaranteed radius of a
column) is not tunable and stays with the lattice that owns it; the format
fallbacks for old checkpoints stay in persistence.py, because they record
what those files meant when they were written, not what the default is now.

The clock values live here but run from `Neuron.tau` and `Neuron.refractory`,
the class attributes the command line sets (and restores) per run.
"""

# --- the default network: footprint and wiring ----------------------------------
ACROSS = 8  # cells across (the input row has one neuron per coded bit)
ROWS = 10  # rows of cells, input at the bottom, output at the top
OMEGA = 0.2  # proportion of all connections that are small-world shortcuts, 0 <= omega < 1
REACH = 2.0  # lattice wiring: every pair within this many unit distances connects (the two hex rings)
WEIGHT_RANGE = (-1.0, 1.0)  # random weights are drawn from this range, and learning clips to it
WEIGHT_EPSILON = 0.001  # --epsilon: the smallest weight allowed under --positive-weights, range (epsilon, 1)

# --- the neuron: activation and the clock (nominal milliseconds) ----------------
THRESHOLD = 0.25  # total weighted input a neuron needs before it fires
MINIMUM_POTENTIAL = -1.0  # floor on a potential: inhibition and carried-over charge can go no lower
TAU = 2.0  # leak time constant (swept September 10, 2026: see docs/tau-sweep.md); math.inf switches the leak off
REFRACTORY = 5.0  # absolute refractory period: a neuron that fired this recently ignores every signal
INTERVAL = 10.0  # spacing of inputs when no time is given; the clock only advances between inputs

# --- learning: the global-reinforcement rule and its housekeeping ---------------
TARGET = "reversed"  # what the top row should show, derived from the input row (learning.TARGETS)
CRITIC = "row"  # how the reward is judged (learning.CRITICS)
ELIGIBILITY = "perturb"  # what the global reward acts on (learning.ELIGIBILITIES)
LATE = "count"  # what a signal arriving after its target fired earns (learning.LATE_RULES)
LR = 0.03  # learning rate
SIGMA = 0.1  # exploration noise: std dev of each neuron's starting potential
BASELINE_RATE = 0.05  # per-epoch update of the running reward baseline the advantage is measured against
WINDOW = 200  # epochs the Teacher's moving-average accuracy spans
HOMEOSTASIS = 1e-6  # per-epoch rate at which a threshold moves toward the target firing rate; 0 = off
TARGET_RATE = 0.5  # firing rate homeostasis aims for, 0 to 1
UNSTICK = 1e-3  # per-epoch rate at which a stuck output neuron's threshold moves toward UNSTICK_TARGET; 0 = off
UNSTICK_TARGET = 0.5  # firing rate the output un-sticking aims for
THRESHOLD_RANGE = (-5.0, 5.0)  # limits on what homeostasis may move a threshold to
RATE_MEMORY = 0.01  # per-epoch update of a neuron's running firing rate (about the last 100 epochs)
STUCK_BELOW, STUCK_ABOVE = 0.01, 0.99  # a neuron firing less or more often than this is "stuck"
