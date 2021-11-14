from datetime import datetime
from collections import namedtuple
import time

Record = namedtuple("Record", "name, ts")

class SimpleTimer:
    def __init__(self):
        self.clock = time.time()
        self.storage = []

    def checkpoint(self, name):
        self.storage.append(Record(name, time.time()))

    def report(self):
        total_time = self.storage[-1].ts - self.clock
        for ind, record in enumerate(self.storage):
            print("name: {}, elapsed time (s): {:.4f}, from_last: {:.4f} fraction: {:.4f}".format(
                record.name,
                record.ts - self.clock,
                (record.ts - self.storage[ind - 1].ts) if ind > 0 else 0.0,
                (record.ts - self.clock) / total_time))
