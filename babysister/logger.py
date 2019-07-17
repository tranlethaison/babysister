""""""
import os
import csv
import time
from collections import OrderedDict


class Logger:
    def __init__(
        self, 
        log_file,
        header,
        delimiter=',',
        quotechar="'",
    ):
        """"""
        assert os.path.splitext(log_file)[-1] == ".csv"

        self.header = header
        self.log_file = log_file
        self.delimiter = delimiter
        self.quotechar = quotechar

    def open(self, mode="w+"):
        """open for writing"""
        self.csvfile = open(self.log_file, mode, newline='')
        self.writer = csv.DictWriter(
            self.csvfile, 
            fieldnames=self.header,
            delimiter=self.delimiter, 
            quotechar=self.quotechar,
            quoting=csv.QUOTE_NONNUMERIC)

    def close(self):
        """"""
        self.csvfile.close()

    def save(self):
        """"""
        self.csvfile.flush()
        os.fsync(self.csvfile.fileno())

    def write_header(self):
        """"""
        self.info(self.header)

    def info(self, msg):
        """"""
        if type(msg) is list:
            row = {}
            for field_name, value in zip(self.header, msg):
                row[field_name] = value
        elif (type(msg) is dict
        or type(msg) is OrderedDict):
            row = msg

        self.writer.writerow(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def read(self):
        """"""
        with open(self.log_file, newline='') as csvfile:
            reader = csv.DictReader(
                csvfile, 
                fieldnames=self.header,
                delimiter=self.delimiter, 
                quotechar=self.quotechar,
                quoting=csv.QUOTE_NONNUMERIC)
            return list(reader)

