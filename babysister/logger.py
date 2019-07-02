""""""
import csv
import time


class Logger:
    def __init__(self, save_to='log.csv', delimiter=',', quotechar="'"):
        """"""
        self.header = ['roi_id', 'n_objs', 'timestamp']
        self.time_fmt = '%Y/%m/%d %H:%M:%S'
        self.save_to = save_to
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.open()

    def open(self):
        """"""
        self.fp = open(self.save_to, 'w+', newline='')
        self.writer = csv.DictWriter(
            self.fp, fieldnames=self.header,
            delimiter=self.delimiter, quotechar=self.quotechar,
            quoting=csv.QUOTE_NONNUMERIC)

    def close(self):
        """"""
        self.fp.close()

    def write_header(self):
        """"""
        self.info(self.header, do_format_time=False)

    def info(self, msg, do_format_time=True):
        """"""
        if type(msg) is list:
            row = {}
            for field_name, value in zip(self.header, msg):
                row[field_name] = value

        elif type(msg) is dict:
            row = msg

        if do_format_time:
            row['timestamp'] = self.format_time(row['timestamp'])

        self.writer.writerow(row)

    def format_time(self, seconds):
        """"""
        return time.strftime(self.time_fmt, time.localtime(seconds))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

