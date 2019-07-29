import os
import csv
import time
from collections import OrderedDict


class Logger:
    """CSV logger.
    
    Args:
        header (list of str): column names.
        log_file (str): path to log file.
        delimiter (str): delimiter.
        quotechar (str): quote char.
        quoting (csv.QUOTE_xxx constant): quoting instruction.
    """

    def __init__(
        self,
        log_file,
        header,
        delimiter=",",
        quotechar="'",
        quoting=csv.QUOTE_NONNUMERIC,
    ):
        assert os.path.splitext(log_file)[-1] == ".csv"

        self.header = header
        self.log_file = log_file
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.quoting = quoting

    def open(self, mode="w+"):
        """Open `log_file`.

        Args:
            mode (str): Python open mode.
        """
        self.csvfile = open(self.log_file, mode, newline="")
        self.writer = csv.DictWriter(
            self.csvfile,
            fieldnames=self.header,
            delimiter=self.delimiter,
            quotechar=self.quotechar,
            quoting=self.quoting,
        )

    def close(self):
        """Close `log_file`."""
        self.csvfile.close()

    def save(self):
        """Save writing buffer to `log_file`."""
        self.csvfile.flush()
        os.fsync(self.csvfile.fileno())

    def write_header(self):
        """Write `header` to writing buffer."""
        self.info(self.header)

    def info(self, msg):
        """Write `msg` to writing buffer.

        Args:
            msg (list or dict): a line of log in either list of value format,
                or dict of {column: value} format. 
        """
        if type(msg) is list:
            row = {}
            for field_name, value in zip(self.header, msg):
                row[field_name] = value
        elif type(msg) is dict or type(msg) is OrderedDict:
            row = msg

        self.writer.writerow(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def read(self):
        """Return list of lines from `log_file`, with line in {column: value} format."""
        with open(self.log_file, newline="") as csvfile:
            reader = csv.DictReader(
                csvfile,
                fieldnames=self.header,
                delimiter=self.delimiter,
                quotechar=self.quotechar,
                quoting=self.quoting,
            )
            return list(reader)
