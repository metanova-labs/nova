"""Scheduling of epoch scoring in the validator main loop (pure, no chain needed)."""
import pytest

from neurons.validator import epoch_schedule as sched

L = 361  # epoch length on SN68


def act(block, last=None, grace=30):
    return sched.next_action(block, L, last, grace)


def test_at_a_boundary_scores_now():
    assert act(9269397) == (sched.SCORE, 9269397)


def test_first_cycle_mid_epoch_waits_for_the_next_boundary():
    # a freshly started process must not score an epoch an earlier process may already have scored
    assert act(9269400, last=None) == (sched.WAIT, 9269758)


def test_overrunning_cycle_scores_the_missed_epoch_late():
    # 2026-10-12: epoch 25676 started at block 9269397, finished at 9269759, one block past 9269758.
    # The old rule waited until 9270119 and epoch 25677 was never scored.
    assert act(9269759, last=9269397) == (sched.SCORE, 9269758)


def test_late_by_more_than_the_grace_waits_for_the_next_boundary():
    assert act(9269758 + 31, last=9269397) == (sched.WAIT, 9270119)
    assert act(9269758 + 30, last=9269397) == (sched.SCORE, 9269758)


def test_only_the_boundary_right_after_the_last_scored_one_is_taken_late():
    # an outage that skipped a whole epoch is not caught up
    assert act(9270119 + 5, last=9269397) == (sched.WAIT, 9270480)


def test_reentering_after_an_error_does_not_rescore_the_same_boundary():
    assert act(9269397 + 3, last=9269397) == (sched.WAIT, 9269758)


def test_grace_zero_restores_the_old_behaviour():
    assert act(9269759, last=9269397, grace=0) == (sched.WAIT, 9270119)


def test_consecutive_late_starts_are_each_allowed_while_inside_the_grace():
    first = act(9269759, last=9269397)
    assert first == (sched.SCORE, 9269758)
    # the next cycle also overran slightly
    assert act(9270120 + 4, last=9269758) == (sched.SCORE, 9270119)


@pytest.mark.parametrize("raw,expected", [
    (None, sched.DEFAULT_LATE_START_GRACE_BLOCKS),
    ("", sched.DEFAULT_LATE_START_GRACE_BLOCKS),
    ("12", 12),
    ("0", 0),
    ("-5", 0),
    ("abc", sched.DEFAULT_LATE_START_GRACE_BLOCKS),
])
def test_grace_from_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("NOVA_LATE_START_GRACE_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("NOVA_LATE_START_GRACE_BLOCKS", raw)
    assert sched.late_start_grace_blocks() == expected
