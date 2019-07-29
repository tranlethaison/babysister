import time


class FPSCounter:
    """Events counter.

    Args:
        interval (float): time interval (seconds) to count within.
    """

    def __init__(self, interval):
        self.interval = interval
        self.start()

    def start(self):
        """Start counter.
        
        Init stating time, event counter, result.
        """
        self.start_time = time.time()
        self.counter = 0
        self.fps = 0

    def tick(self):
        """Mark the end of an event.

        Returns:
            int: number of counted events in last `interval` interval.
        """
        self.counter += 1
        if (time.time() - self.start_time) > self.interval:
            self.fps = self.counter / (time.time() - self.start_time)
            self.counter = 0
            self.start_time = time.time()
        return self.fps

    def get(self):
        """Returns number of counted events in last `interval` interval."""
        return self.fps
