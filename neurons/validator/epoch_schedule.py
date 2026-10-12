"""When the validator scores an epoch.

The main loop scores epoch N at the boundary block that ends it. A cycle that
is still running when the NEXT boundary arrives used to read the block after
finishing, find it was not an exact multiple of the epoch length, and wait for
the boundary after that, so the epoch whose boundary passed during processing
was never scored at all (no ranking, no weights, no payout for any miner).

Cycle time is not constant: preparation, inference (bounded by a deadline) and
the post-processing (uploads, weights, payouts) add up to more than one epoch
whenever inference runs long. That happened on 2026-10-12: epoch 25676's cycle
took 4344 s against a 4332 s epoch and finished one block after the boundary,
so epoch 25677 was skipped.

``next_action`` keeps this decision pure so it can be tested without a chain.
"""
from __future__ import annotations

import os

#: A cycle finishing this many blocks (12 s each) after a boundary still scores
#: that epoch immediately. 30 blocks is 6 minutes: enough to absorb a modest
#: overrun, small enough that the epoch's data is still recent. Override with
#: NOVA_LATE_START_GRACE_BLOCKS; 0 restores the old skip-to-next-boundary rule.
DEFAULT_LATE_START_GRACE_BLOCKS = 30

SCORE = "score"
WAIT = "wait"


def late_start_grace_blocks() -> int:
    raw = os.environ.get("NOVA_LATE_START_GRACE_BLOCKS")
    if raw is None or raw.strip() == "":
        return DEFAULT_LATE_START_GRACE_BLOCKS
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_LATE_START_GRACE_BLOCKS


def next_action(
    current_block: int,
    epoch_length: int,
    last_scored_boundary: int | None,
    grace_blocks: int,
) -> tuple[str, int]:
    """Decide what the main loop does at ``current_block``.

    Returns ``(SCORE, boundary_block)`` to score the epoch ending at that
    boundary now, or ``(WAIT, boundary_block)`` to wait for that block first.

    A late start is only taken for the boundary immediately after the one this
    process last scored. That keeps a restart from re-scoring an epoch the
    previous process already scored (which would pay winners twice), and means
    a long outage still waits for the next clean boundary.
    """
    remainder = current_block % epoch_length
    if remainder == 0:
        return SCORE, current_block
    boundary = current_block - remainder
    if (
        grace_blocks > 0
        and last_scored_boundary is not None
        and boundary == last_scored_boundary + epoch_length
        and remainder <= grace_blocks
    ):
        return SCORE, boundary
    return WAIT, boundary + epoch_length
