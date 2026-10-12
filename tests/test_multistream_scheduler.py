"""Behavior checks for the deterministic multi-stream schedulers.

These are the properties the four-stream budget depends on: proportional service,
no starvation, determinism, bounded credit, per-stream rate control, and a bounded
drop-oldest queue. None of them needs a board or a model.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]


def _load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scheduler_module():
    return _load_module("ms_scheduler", "src/multistream_framework/scheduler.py")


def _full(stream_ids, length=1000):
    return {sid: {"queue_length": length} for sid in stream_ids}


def _serve(scheduler, states, count, now=None):
    """Run ``count`` selections and return the per-stream service counts."""
    counts = Counter()
    for _ in range(count):
        sid = scheduler.select(states, now)
        if sid < 0:
            break
        counts[sid] += 1
    return counts


# --------------------------------------------------------------------------- RR


def test_round_robin_rotates_and_skips_empty(scheduler_module) -> None:
    scheduler = scheduler_module.RoundRobinScheduler(4)

    assert scheduler.select({0: {"queue_length": 1}, 1: {"queue_length": 1}, 2: {"queue_length": 1}, 3: {"queue_length": 1}}) == 0
    assert scheduler.select({0: {"queue_length": 1}, 1: {"queue_length": 1}, 2: {"queue_length": 1}, 3: {"queue_length": 1}}) == 1
    # stream 2 is empty, so the rotation must skip to 3
    assert scheduler.select({0: {"queue_length": 1}, 1: {"queue_length": 1}, 2: {"queue_length": 0}, 3: {"queue_length": 1}}) == 3
    assert scheduler.select({0: {"queue_length": 0}, 1: {"queue_length": 0}, 2: {"queue_length": 0}, 3: {"queue_length": 0}}) == -1


def test_round_robin_rejects_a_bad_stream_count(scheduler_module) -> None:
    with pytest.raises(ValueError):
        scheduler_module.RoundRobinScheduler(0)


# -------------------------------------------------------------------------- WRR


def test_weighted_round_robin_is_proportional(scheduler_module) -> None:
    scheduler = scheduler_module.WeightedRoundRobinScheduler({0: 5, 1: 2, 2: 2, 3: 1})
    counts = _serve(scheduler, _full(range(4)), 1000)

    # weights 5:2:2:1 out of 10 -> 50% / 20% / 20% / 10%
    assert counts[0] == pytest.approx(500, abs=6)
    assert counts[1] == pytest.approx(200, abs=6)
    assert counts[2] == pytest.approx(200, abs=6)
    assert counts[3] == pytest.approx(100, abs=6)
    assert sum(counts.values()) == 1000


def test_weighted_round_robin_never_starves_a_positive_weight(scheduler_module) -> None:
    scheduler = scheduler_module.WeightedRoundRobinScheduler({0: 100, 1: 1, 2: 1, 3: 1})
    counts = _serve(scheduler, _full(range(4)), 400)

    # the class of failure this guards against is a low-weight stream never being
    # picked at all while a high-weight stream is always ready
    for sid in (1, 2, 3):
        assert counts[sid] >= 3, counts


def test_weighted_round_robin_skips_zero_weight_and_empty_queues(scheduler_module) -> None:
    scheduler = scheduler_module.WeightedRoundRobinScheduler({0: 1, 1: 0, 2: 1, 3: 1})

    counts = _serve(scheduler, _full(range(4)), 300)
    assert counts[1] == 0

    states = {0: {"queue_length": 0}, 1: {"queue_length": 5}, 2: {"queue_length": 0}, 3: {"queue_length": 0}}
    assert scheduler.select(states) == -1


def test_weighted_round_robin_credit_stays_bounded(scheduler_module) -> None:
    scheduler = scheduler_module.WeightedRoundRobinScheduler({0: 3, 1: 1})

    # stream 1 is starved of work for a long stretch; its credit must not grow
    # without limit, or it would monopolise the NPU when it comes back
    states = {0: {"queue_length": 1}, 1: {"queue_length": 0}}
    for _ in range(500):
        scheduler.select(states)
    assert scheduler.credit(1) <= 2 * 4

    states[1]["queue_length"] = 1
    counts = _serve(scheduler, states, 60)
    assert counts[0] > counts[1]


def test_weighted_round_robin_is_deterministic(scheduler_module) -> None:
    first = _serve(scheduler_module.WeightedRoundRobinScheduler({0: 2, 1: 1}), _full(range(2)), 50)
    second = _serve(scheduler_module.WeightedRoundRobinScheduler({0: 2, 1: 1}), _full(range(2)), 50)

    assert first == second


def test_weighted_round_robin_validates_weights(scheduler_module) -> None:
    with pytest.raises(ValueError):
        scheduler_module.WeightedRoundRobinScheduler({})
    with pytest.raises(ValueError):
        scheduler_module.WeightedRoundRobinScheduler({0: -1, 1: 2})
    with pytest.raises(ValueError):
        scheduler_module.WeightedRoundRobinScheduler({0: 0, 1: 0})


def test_front_priority_weights_put_the_most_on_front(scheduler_module) -> None:
    weights = scheduler_module.front_priority_weights()

    assert weights[0] > weights[1] == weights[2] > weights[3]


# -------------------------------------------------------------------- rate/deadline


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_rate_scheduler_returns_minus_one_when_nothing_is_due(scheduler_module) -> None:
    clock = _VirtualClock()
    scheduler = scheduler_module.RateDeadlineScheduler({0: 10.0, 1: 6.0}, clock=clock)
    states = _full(range(2))

    # a stream that has never been served is due immediately, which is what keeps
    # a newly started stream from waiting a full period behind an established one
    assert scheduler.select(states, now=0.0) == 0
    assert scheduler.select(states, now=0.050) == 1
    # now both have deadlines at 0.100 and 0.217, so nothing is due at 0.06
    assert scheduler.select(states, now=0.060) == -1


def test_rate_scheduler_honours_per_stream_target_fps(scheduler_module) -> None:
    clock = _VirtualClock()
    scheduler = scheduler_module.RateDeadlineScheduler({0: 10.0, 1: 6.0}, clock=clock)
    states = _full(range(2))

    counts = Counter()
    for step in range(600):  # 3 seconds at 5 ms resolution
        now = step * 0.005
        sid = scheduler.select(states, now=now)
        if sid >= 0:
            counts[sid] += 1
    # 3 s at 10 FPS / 6 FPS
    assert counts[0] == pytest.approx(30, abs=3)
    assert counts[1] == pytest.approx(18, abs=3)


def test_rate_scheduler_breaks_ties_by_priority(scheduler_module) -> None:
    clock = _VirtualClock()
    scheduler = scheduler_module.RateDeadlineScheduler(
        {0: 10.0, 3: 10.0}, priority={0: 3, 3: 1}, clock=clock
    )
    states = _full([0, 3])

    # both come due at t=0 with equal deadline, so FRONT must win
    assert scheduler.select(states, now=0.0) == 0


def test_rate_scheduler_does_not_accumulate_unbounded_lateness(scheduler_module) -> None:
    clock = _VirtualClock()
    scheduler = scheduler_module.RateDeadlineScheduler({0: 10.0}, clock=clock)
    states = _full([0])

    # a long stall: next due must be rebased on "now", not on the stale deadline
    assert scheduler.select(states, now=0.0) == 0
    assert scheduler.select(states, now=5.0) == 0
    assert scheduler.next_due(0) == pytest.approx(5.1)
    assert scheduler.select(states, now=5.05) == -1


def test_rate_scheduler_serves_the_most_overdue_stream(scheduler_module) -> None:
    clock = _VirtualClock()
    scheduler = scheduler_module.RateDeadlineScheduler({0: 5.0, 1: 5.0}, clock=clock)
    states = _full([0, 1])

    assert scheduler.select(states, now=0.0) == 0
    # 0 is next due at 0.2; make 1 overdue by now instead
    assert scheduler.select(states, now=0.3) == 1


def test_rate_scheduler_validates_rates(scheduler_module) -> None:
    with pytest.raises(ValueError):
        scheduler_module.RateDeadlineScheduler({})
    with pytest.raises(ValueError):
        scheduler_module.RateDeadlineScheduler({0: 0.0})


# ------------------------------------------------------------------ drop-oldest


@pytest.fixture(scope="module")
def drop_queue_module():
    return _load_module("ms_replay", "src/multistream_framework/replay.py")


def test_drop_oldest_queue_is_bounded_and_evicts_the_oldest(drop_queue_module) -> None:
    queue = drop_queue_module.DropOldestQueue(2)

    assert queue.put_drop_oldest("a") is None
    assert queue.put_drop_oldest("b") is None
    evicted = queue.put_drop_oldest("c")

    assert evicted == "a"
    assert queue.qsize() == 2
    assert queue.dropped == 1
    assert queue.get_nowait() == "b"


def test_drop_oldest_queue_counts_every_drop(drop_queue_module) -> None:
    queue = drop_queue_module.DropOldestQueue(1)

    for value in range(5):
        queue.put_drop_oldest(value)

    assert queue.dropped == 4
    assert queue.qsize() == 1
    assert queue.get_nowait() == 4


def test_drop_oldest_queue_raises_when_empty(drop_queue_module) -> None:
    import queue as queue_module

    with pytest.raises(queue_module.Empty):
        drop_queue_module.DropOldestQueue(2).get_nowait()
