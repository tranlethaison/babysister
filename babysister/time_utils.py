"""¯\_(ツ)_/¯"""
import time
import datetime


class StopWatch():
    def __init__(self, precision=7):
        self.precision = precision

    def time(self):
        now = time.time()
        fmt = "{:." + str(self.precision) + "f}"
        return float(fmt.format(now))

    def start(self):
        self.start_ = self.split_ = self.time()
        self.is_paused = False
        return self.start_

    def pause(self):
        if not self.is_paused:
            self.pause_ = self.time()
            self.is_paused = True
        
    def resume(self):
        if self.is_paused:
            pause_elapsed = self.time() - self.pause_
            self.start_ += pause_elapsed
            self.split_ += pause_elapsed
            self.is_paused = False

    def elapsed(self):
        assert not self.is_paused
        return self.time() - self.start_

    def split_elapsed(self):
        assert not self.is_paused
        return self.time() - self.split_

    def split(self):
        assert not self.is_paused
        now = self.time()
        split_elapsed = now - self.split_
        self.split_ = now
        return split_elapsed


def get_str_localtime(time_fmt, epoch_time):
    """"""
    return time.strftime(time_fmt, time.localtime(epoch_time)) 


def get_epoch_time(time_fmt, str_time):
    """"""
    timetuple = datetime.datetime.strptime(str_time, time_fmt).timetuple()
    return time.mktime(timetuple)

