"""Every global constant of walnutbutter, in one place.

These are the defaults the constructors, the neuron's clock, the Teacher and
the command line all read from: change a value here and every path follows.
They are the physics of the substance (what a neuron needs to fire, how fast
it leaks, how learning moves a weight) and the shape of the default network.
Lattice geometry (cell spacing, row spacing, the guaranteed radius of a
column) is not tunable and stays with the lattice that owns it; the format
fallbacks for old checkpoints stay in persistence.py, because they record
what those files meant when they were written, not what the default is now.

The clock values live here but run from `Neuron.refractory` and
`Neuron.refractory_hops`, the class attributes the command line sets (and
restores) per run.
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
TAU = 2.0  # ms: leak time constant of the potential, computed lazily on arrival (Byron, September 12, 2026, bringing the leak back; his earlier sweep chose 2); math.inf switches it off
REFRACTORY = 5.0  # absolute refractory period: a neuron that fired this recently ignores every signal
REFRACTORY_HOPS = 3.0  # the refractory period divided by the time a signal takes to travel one hop; not an integer (Byron, September 11, 2026)
INTERVAL = 10.0  # spacing of inputs when no time is given
BORED_AFTER = 200.0  # ms of silence after which a neuron's threshold has fallen to zero and it fires on its own (Byron, September 12, 2026); 0 = off

# --- the problem ------------------------------------------------------------------
PROBLEM = "reversal"  # what the network is asked to do and how it is watched (problems.PROBLEMS)

# --- learning: dopamine (AUTHORITY.md §6) --------------------------------------------
RULE = "teacher"  # which learning rule runs (learning.RULES): teacher (an external teacher scores the read, Byron, September 12, 2026),
# dopamine (the student as its own teacher), or the reinforce rule factored out below
TEACHER_CREDIT = 0.25  # what each input neuron read correctly adds to the teacher's score, and each one read wrongly subtracts
LR = 0.03  # learning rate, both rules
SIGMA = 0.1  # exploration noise: std dev added to each neuron's potential at every input; 0 switches it off
DOPAMINE_RELEASE_ALPHA = 2.0  # shape of the gamma density of the amount a refire releases against its delay past the refractory period (Byron, September 12, 2026)
DOPAMINE_RELEASE_THETA = 1.0  # ms: its scale; the release peaks at (alpha - 1) * theta past the end of the refractory period
DOPAMINE_TAU = 20.0  # ms: decay of the global dopamine value
DOPAMINE_EXPECTATION_TAU = 600_000.0  # ms (10 minutes): the exponential window of the expected dopamine trace (Byron, September 12, 2026)
DOPAMINE_EXPECTATION_START = 0.0  # where the expected dopamine trace starts; a high start holds early learning back (Byron, same day)
DOPAMINE_ORDER = "release-first"  # at a refire, release before the weight update, or update-first (dopamine.ORDERS)
DOPAMINE_PUNISH = True  # an input neuron whose bit is 0 has the sign of its update reversed when it refires (Byron, September 12, 2026)
DOPAMINE_PUNISH_GAIN = 2.0  # and that reversed update is this many times as large as a reward (Byron, September 12, 2026, 'for now')
WEIGHT_DECAY = 1e-4  # every weight moves toward 0 by this fraction each epoch: synapses that forget on their own (Byron, same day, 'for now')

# --- the reinforce rule of the pre-alpha, factored out behind RULE = "reinforce" ---
TARGET = "reversed"  # what the top row should show, derived from the input row (learning.TARGETS)
CRITIC = "row"  # how the reward is judged (learning.CRITICS)
ELIGIBILITY = "perturb"  # what the global reward acts on (learning.ELIGIBILITIES)
LATE = "count"  # what a signal arriving after its target fired earns (learning.LATE_RULES)
BASELINE_RATE = 0.05  # per-epoch update of the running reward baseline the advantage is measured against
WINDOW = 200  # epochs the Teacher's moving-average accuracy spans
HOMEOSTASIS = 1e-6  # per-epoch rate at which a threshold moves toward the target firing rate; 0 = off
TARGET_RATE = 0.5  # firing rate homeostasis aims for, 0 to 1
UNSTICK = 1e-3  # per-epoch rate at which a stuck output neuron's threshold moves toward UNSTICK_TARGET; 0 = off
UNSTICK_TARGET = 0.5  # firing rate the output un-sticking aims for
THRESHOLD_RANGE = (-5.0, 5.0)  # limits on what homeostasis may move a threshold to
RATE_MEMORY = 0.01  # per-epoch update of a neuron's running firing rate (about the last 100 epochs)
STUCK_BELOW, STUCK_ABOVE = 0.01, 0.99  # a neuron firing less or more often than this is "stuck"
