"""
    Date: 2021.05.02
    Author: Chance Kim (https://github.com/tiratano)
    Purpose: Common Utility Functions
"""

import pickle
import time
import logging.handlers
from datetime import timedelta
import atexit


def benchmark(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        logging.info("[{}] execution time: {:.3f} sec".format(original_fn.__name__, end_time - start_time))
        return result

    return wrapper_fn


# https://stackoverflow.com/questions/25194864/python-logging-time-since-start-of-program/25198931
class ElapsedFormatter(logging.Formatter):
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = int(record.created - self.start_time)
        elapsed = timedelta(seconds=elapsed_seconds)
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
        return "[{}] [{}] [{}] {}".format(cur_time, elapsed, record.levelname, record.getMessage())


handler = logging.StreamHandler()
handler.setFormatter(ElapsedFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info('========================================================')
logger.info('start')
logger.info('========================================================')


def goodbye():
    logger.info('========================================================')
    logging.info("end")
    logger.info('========================================================')


atexit.register(goodbye)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj

    return None


if __name__ == "__main__":
    logging.info("Hello")

    @benchmark
    def train():
        time.sleep(1)

    train()


