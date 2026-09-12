"""The problems: what a network is asked to do, and how it is watched doing it.

A problem names the layout, the inputs, the spacing of inputs, what is read
as the output and when, and whether the Teacher only scores it or may also
train it (the reinforce rule, homeostasis, un-sticking). The command line
picks one with --problem; every other option still applies on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import ACROSS, ROWS


@dataclass(frozen=True)
class Problem:
    name: str
    description: str
    across: int  # cells across, unless --across (or --ecc) says otherwise
    rows: int
    trained: bool  # True: the Teacher may train (reinforce rule, homeostasis, un-sticking); False: it only scores
    interval: float | None = None  # ms between inputs: the epoch's length (None: INTERVAL)
    readout: str = "top"  # which neurons are read as the output: "top" (the top row) or "input" (the inputs are the outputs)
    read: str = "fired"  # what "on" means at the read: "fired" this epoch, "again" (spiked after the input's moment), or
    # "window" (within read_window ms before the epoch's end)
    read_window: float | None = None  # the window for read == "window"
    target: str | None = None  # what the output should show (None: --target)
    critic: str | None = None  # how the read is scored (None: --critic)
    coding: str = "complement"  # how raw bits reach the input row: "complement" (bits then their negations) or "raw"


PROBLEMS: dict[str, Problem] = {
    "reversal": Problem(
        "reversal",
        "the top row learns to show the bottom row reversed, taught by a Teacher with a target and a critic",
        ACROSS, ROWS, trained=True,
    ),
    "sustain_inputs": Problem(
        "sustain_inputs",
        "the 16 four-bit inputs laid down as they are on 4 neurons (no complement coding, so 0000 forces nothing and "
        "1111 forces all four); an input is forced, the mesh reverberates for 20 ms, and the same neurons are read "
        "(the inputs are the outputs): on if they spiked again after the input's moment. The score "
        "is the fraction of the four whose read state matches the pattern: the forced ones on, the others off. "
        "Scored by the Teacher, not trained by it: the neurons learn by dopamine (AUTHORITY.md §6, §8)",
        4, ROWS, trained=False, interval=20.0, readout="input", read="again", target="copy", critic="row", coding="raw",
    ),
}
