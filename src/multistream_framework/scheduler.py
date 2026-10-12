"""Deterministic schedulers for the four-stream heterogeneous pipeline.

Two implementations, both simple and fully deterministic - no learned scheduling,
no RL, no ranking model:

``RoundRobinScheduler``
    The original uniform rotation. Kept for the homogeneous baseline.

``WeightedRoundRobinScheduler``
    Interleaved (smooth) weighted round-robin. Weights set the share of inference
    slots, so ``FRONT 5, LEFT 2, RIGHT 2, BACK 1`` spends half the NPU on the front
    camera. Any stream with a non-zero weight is guaranteed a slot in every
    super-cycle, so none can starve. Credit is clamped, which bounds the burst a
    stream can accumulate while its queue happened to be empty.

``RateDeadlineScheduler``
    Per-stream rate control. Each stream has its own target FPS and the scheduler
    serves whichever stream is most overdue. A stream that is behind by more than
    one period wins over a stream that is merely due, which is how the front
    camera keeps its priority when the NPU saturates. Returns ``-1`` when nothing
    is due, so the caller idles instead of over-serving.

Both take an optional ``now`` so tests can drive them with a virtual clock
instead of wall time.
"""
from __future__ import annotations

import time
from typing import Callable, Mapping


class Scheduler:
    """Selects which stream gets the next inference slot."""

    def select(self, stream_states: Mapping[int, Mapping[str, int]], now: float | None = None) -> int:
        """Return a stream id with queued work, or ``-1`` when there is nothing to serve."""
        raise NotImplementedError


class RoundRobinScheduler(Scheduler):
    def __init__(self, stream_count: int) -> None:
        if stream_count < 1:
            raise ValueError("stream_count must be positive")
        self.stream_count = stream_count
        self._cursor = 0

    def select(self, stream_states: Mapping[int, Mapping[str, int]], now: float | None = None) -> int:
        for offset in range(self.stream_count):
            stream_id = (self._cursor + offset) % self.stream_count
            if stream_states.get(stream_id, {}).get("queue_length", 0) > 0:
                self._cursor = (stream_id + 1) % self.stream_count
                return stream_id
        return -1


class WeightedRoundRobinScheduler(Scheduler):
    """Interleaved weighted round-robin with bounded credit."""

    def __init__(self, weights: Mapping[int, int]) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("weights must be non-negative")
        if sum(weights.values()) <= 0:
            raise ValueError("at least one weight must be positive")
        self.weights = {int(sid): int(weight) for sid, weight in weights.items()}
        self._credit = {sid: 0 for sid in self.weights}
        self._active: set[int] = set()
        self._total = sum(self.weights.values())
        self._limit = 2 * self._total

    def credit(self, stream_id: int) -> int:
        return self._credit[int(stream_id)]

    def select(self, stream_states: Mapping[int, Mapping[str, int]], now: float | None = None) -> int:
        candidates = [
            sid
            for sid in sorted(self.weights)
            if self.weights[sid] > 0 and stream_states.get(sid, {}).get("queue_length", 0) > 0
        ]
        if not candidates:
            self._active = set()
            return -1
        # Credit is only moved between streams that actually have work, and a
        # stream re-entering after being empty restarts from zero. Crediting
        # absent streams instead lets a starved stream build an unbounded deficit
        # and then monopolise the NPU when it comes back - which is exactly the
        # failure the bounded-credit test guards.
        for sid in candidates:
            if sid not in self._active:
                self._credit[sid] = 0
        self._active = set(candidates)
        active_total = sum(self.weights[sid] for sid in candidates)
        for sid in candidates:
            self._credit[sid] = min(self._credit[sid] + self.weights[sid], self._limit)
        # Highest credit wins; ties break towards the lower stream id so the
        # schedule is reproducible.
        chosen = max(candidates, key=lambda sid: (self._credit[sid], -sid))
        self._credit[chosen] -= active_total
        return chosen


class RateDeadlineScheduler(Scheduler):
    """Per-stream target rate, served most-overdue-first."""

    def __init__(
        self,
        target_fps: Mapping[int, float],
        priority: Mapping[int, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not target_fps:
            raise ValueError("target_fps must not be empty")
        if any(fps <= 0 for fps in target_fps.values()):
            raise ValueError("target_fps must be positive for every stream")
        self.target_fps = {int(sid): float(fps) for sid, fps in target_fps.items()}
        self.priority = {int(sid): int((priority or {}).get(sid, 0)) for sid in self.target_fps}
        self._clock = clock
        self._next_due: dict[int, float] = {}
        self._served: dict[int, int] = {sid: 0 for sid in self.target_fps}

    def count(self, stream_id: int) -> int:
        return self._served[int(stream_id)]

    def select(self, stream_states: Mapping[int, Mapping[str, int]], now: float | None = None) -> int:
        moment = self._clock() if now is None else float(now)
        candidates = [
            sid
            for sid in sorted(self.target_fps)
            if self.target_fps[sid] > 0 and stream_states.get(sid, {}).get("queue_length", 0) > 0
        ]
        if not candidates:
            return -1
        for sid in candidates:
            self._next_due.setdefault(sid, moment)
        due = [sid for sid in candidates if self._next_due[sid] <= moment]
        if not due:
            return -1
        # Most overdue first; then higher priority; then lower stream id.
        chosen = min(due, key=lambda sid: (self._next_due[sid], -self.priority[sid], sid))
        # Advance from the later of "now" and the previous deadline so a saturated
        # NPU cannot accumulate unbounded lateness credit.
        self._next_due[chosen] = max(moment, self._next_due[chosen]) + 1.0 / self.target_fps[chosen]
        self._served[chosen] += 1
        return chosen

    def next_due(self, stream_id: int) -> float | None:
        return self._next_due.get(int(stream_id))


def front_priority_weights(front: int = 5, side: int = 2, back: int = 1) -> dict[int, int]:
    """Convenience weights for CH0=FRONT, CH1=LEFT, CH2=RIGHT, CH3=BACK."""
    return {0: front, 1: side, 2: side, 3: back}


def front_priority_fps(front: float, side: float, back: float) -> dict[int, float]:
    """Convenience target rates for CH0=FRONT, CH1=LEFT, CH2=RIGHT, CH3=BACK."""
    return {0: front, 1: side, 2: side, 3: back}


def front_priority_ranks() -> dict[int, int]:
    return {0: 3, 1: 2, 2: 2, 3: 1}
