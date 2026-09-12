"""The problems: what a network is asked to do, and how it is watched doing it.

A problem names the layout, the inputs, the spacing of inputs, what is read
as the output and when, and whether the Teacher only scores it or may also
train it (the reinforce rule, homeostasis, un-sticking). The command line
picks one with --problem; every other option still applies on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import ACROSS, REFRACTORY, ROWS


@dataclass(frozen=True)
class Problem:
    name: str
    description: str
    across: int  # cells across, unless --across (or --ecc) says otherwise
    rows: int
    trained: bool  # True: the Teacher may train (reinforce rule, homeostasis, un-sticking); False: it only scores
    interval: float | None = None  # ms between inputs: the epoch's length (None: INTERVAL)
    readout: str = "top"  # which neurons are read as the output: "top" (the top row) or "input" (the inputs are the outputs)
    read_window: float | None = None  # None: an output is on if it fired this epoch; else if it fired within this many
    # ms before the epoch's end, the read
    target: str | None = None  # what the output should show (None: --target)


PROBLEMS: dict[str, Problem] = {
    "reversal": Problem(
        "reversal",
        "the top row learns to show the bottom row reversed, taught by a Teacher with a target and a critic",
        ACROSS, ROWS, trained=True,
    ),
    "sustain_inputs": Problem(
        "sustain_inputs",
        "the same 16 inputs (4 bits, complement-coded) across 8 neurons; an input is forced, the mesh reverberates "
        "for 20 ms, and the same neurons are read (the inputs are the outputs): on if they fired within the last "
        "refractory period before the read. Scored by the Teacher, not trained by it: the neurons learn by dopamine "
        "(AUTHORITY.md §6, §8)",
        8, ROWS, trained=False, interval=20.0, readout="input", read_window=REFRACTORY, target="copy",
    ),
}
