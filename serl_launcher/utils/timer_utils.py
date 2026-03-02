import time
from collections import defaultdict


class _TimerContextManager:
    # Initialize context manager with timer instance and target key
    def __init__(self, timer: "Timer", key: str):
        self.timer = timer
        self.key = key

    # Start timing when entering context
    def __enter__(self):
        self.timer.tick(self.key)

    # Stop timing when exiting context (even if exception occurs)
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.tock(self.key)


class Timer:
    # Initialize timer with empty tracking structures
    def __init__(self):
        self.reset()

    # Reset all tracking variables to initial state
    def reset(self):
        self.counts = defaultdict(int)  # Count of executions per key
        self.times = defaultdict(float)  # Total time spent per key
        self.start_times = {}  # Current start time per key

    # Record start time for a given key
    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    # Calculate elapsed time and update tracking for a given key
    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    # Return context manager for automatic tick/tock
    def context(self, key):
        return _TimerContextManager(self, key)

    # Calculate average time per key and optionally reset timer
    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret
