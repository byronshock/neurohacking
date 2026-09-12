"""The problems: what a network is asked to do, and how it is watched doing it.

A problem names the layout and the inputs, and says whether the network is
trained externally (a Teacher with a target and a critic, the reversal
task) or left to the neurons' own rule (AUTHORITY.md §6). The command line
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
    trained: bool  # True: a Teacher scores every epoch against a target; False: no external training


PROBLEMS: dict[str, Problem] = {
    "reversal": Problem(
        "reversal",
        "the top row learns to show the bottom row reversed, taught by a Teacher with a target and a critic",
        ACROSS, ROWS, trained=True,
    ),
    "sustain_inputs": Problem(
        "sustain_inputs",
        "the same 16 inputs (4 bits, complement-coded) across 8 neurons; not trained externally: "
        "the neurons use the dopamine eligibility rule (AUTHORITY.md §6)",
        8, ROWS, trained=False,
    ),
}
