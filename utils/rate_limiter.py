import time


class RateLimiter:
    """Enforce consistent simulation rate for workspace exploration."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Maintain target simulation frequency."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration
