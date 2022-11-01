__version__ = "0.1.0"

from time import time_ns
from enum import Enum
import base32_lib


class ColorMode(str, Enum):
    color = "color"
    grayscale = "gray"
    black_and_white = "b/w"


def experiment_id() -> str:
    """
    Generate a unique ID for an experiments such that when experiments
    are sorted sorted alpanumerically by id, the experiments will also
    be sorted by creation time.
    """
    ms = int(time_ns() / 1e6)
    return base32_lib.encode(ms)


class IDGenerator:
    def __init__(self):
        self.last = int(time_ns() / 1e6)
    def next(self) -> str:
        self.last += 1
        return base32_lib.encode(self.last)
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()