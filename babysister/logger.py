import csv
import time


class Logger:
    def __init__(
        self, field_names, save_to='log.csv', delimiter=',', quotechar="'"
    ):
        self.field_names = field_names
        self.save_to = save_to
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.open()

    def open(self):
        self.fp = open(self.save_to, 'w+', newline='')
        self.writer = csv.DictWriter(
            self.fp, fieldnames=self.field_names,
            delimiter=self.delimiter, quotechar=self.quotechar,
            quoting=csv.QUOTE_NONNUMERIC)

    def close(self):
        self.fp.close()

    def info(self, msg):
        if type(msg) is dict:
            self.writer.writerow(msg)
        elif type(msg) is list:
            row = {}
            for field_name, value in zip(self.field_names, msg):
                row[field_name] = value
            self.writer.writerow(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    

if __name__ == '__main__':
    time_fmt = '%Y/%m/%d %H:%M:%S'
    field_names = ['timestamp', 'msg']
    #with Logger(field_names, 'test.csv') as logger:
    #    for i in range(10):
    #        timestamp = time.strftime(time_fmt, time.localtime())
    #        logger.info([timestamp, i]) 
    #        time.sleep(2)

    logger = Logger(field_names, 'test.csv') 
    for i in range(10):
        timestamp = time.strftime(time_fmt, time.localtime())
        logger.info([timestamp, i]) 
        time.sleep(2)
    logger.close()

