"""Constants shared by the learning code of both engines."""

RATE_MEMORY = 0.01  # per-epoch update of a neuron's running firing rate (about the last 100 epochs)
STUCK_BELOW, STUCK_ABOVE = 0.01, 0.99  # a neuron firing less or more often than this is "stuck"
THRESHOLD_RANGE = (-5.0, 5.0)  # default limits on what homeostasis may move a threshold to
