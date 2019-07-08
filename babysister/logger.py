""""""
import os
import csv
import time


class Logger:
    def __init__(self, log_file, header, delimiter=',', quotechar="'"):
        """"""
        self.header = header
        self.log_file = log_file
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.open()

    def open(self):
        """"""
        self.fo = open(self.log_file, 'w+', newline='')
        self.writer = csv.DictWriter(
            self.fo, fieldnames=self.header,
            delimiter=self.delimiter, quotechar=self.quotechar,
            quoting=csv.QUOTE_NONNUMERIC)

    def close(self):
        """"""
        self.fo.close()

    def save(self):
        """"""
        self.fo.flush()
        os.fsync(self.fo.fileno())

    def write_header(self):
        """"""
        self.info(self.header)

    def info(self, msg):
        """"""
        if type(msg) is list:
            row = {}
            for field_name, value in zip(self.header, msg):
                row[field_name] = value
        elif type(msg) is dict:
            row = msg

        self.writer.writerow(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

