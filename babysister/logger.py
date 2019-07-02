""""""
import os
import csv
import time


class Logger:
    def __init__(self, save_to, header, delimiter=',', quotechar="'"):
        """"""
        self.header = header
        self.save_to = save_to
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.open()

    def open(self):
        """"""
        self.file = open(self.save_to, 'w+', newline='')
        self.writer = csv.DictWriter(
            self.file, fieldnames=self.header,
            delimiter=self.delimiter, quotechar=self.quotechar,
            quoting=csv.QUOTE_NONNUMERIC)

    def close(self):
        """"""
        self.file.close()

    def save(self):
        """"""
        self.file.flush()
        os.fsync(self.file.fileno())

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

