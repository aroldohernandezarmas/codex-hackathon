"""The event: the confirmed predicate state flips in the direction the user asked for."""

from typing import Optional


class Tracker:
    def __init__(self, direction: str, persist: int = 2) -> None:
        assert direction in ("rising", "falling"), direction
        self.direction = direction
        self.persist = persist
        self.state: Optional[bool] = None  # confirmed state; None until the baseline
        self._candidate: Optional[bool] = None
        self._streak = 0

    def update(self, state_now: bool) -> bool:
        """Feed one model answer. Returns True when the event fires on this answer."""
        if state_now == self.state:
            self._candidate, self._streak = None, 0
            return False
        if state_now == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = state_now, 1
        if self._streak < self.persist:
            return False
        previous, self.state = self.state, state_now
        self._candidate, self._streak = None, 0
        return previous is not None and state_now == (self.direction == "rising")
