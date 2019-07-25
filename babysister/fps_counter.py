import time


class FPSCounter:
    """"""

    def __init__(self, limit):
        """"""
        self.limit = limit
        self.start()

    def start(self):
        """"""
        self.start_time = time.time()
        self.counter = 0
        self.fps = 0

    def tick(self):
        """"""
        self.counter += 1
        if (time.time() - self.start_time) > self.limit:
            self.fps = self.counter / (time.time() - self.start_time)
            self.counter = 0
            self.start_time = time.time()
        return self.fps

    def get(self):
        """"""
        return self.fps
