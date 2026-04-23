import time
from collections import defaultdict
from typing import Dict, TypeAlias

TimeKey: TypeAlias = str
CountMap: TypeAlias = defaultdict[TimeKey, int]
TimeMap: TypeAlias = defaultdict[TimeKey, float]
StartTimeMap: TypeAlias = Dict[TimeKey, float]


class _TimerContextManager:

    def __init__(self, timer: "Timer", key: TimeKey) -> None:
        self.timer = timer
        self.key = key

    def __enter__(self) -> None:
        self.timer.tick(self.key)

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.timer.tock(self.key)


class Timer:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts: CountMap = defaultdict(int)
        self.times: TimeMap = defaultdict(float)
        self.start_times: StartTimeMap = {}

    def tick(self, key: TimeKey) -> None:
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key: TimeKey) -> None:
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def context(self, key: TimeKey) -> _TimerContextManager:
        return _TimerContextManager(self, key)

    def get_average_times(self, reset: bool = True) -> Dict[TimeKey, float]:
        """
        Calculate average elapsed time for each tracked key

        Args:
            reset: If True, reset all tracking metrics after calculation

        Returns:
            Dictionary mapping keys to their average elapsed time
        """
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret
