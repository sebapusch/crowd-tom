import numpy as np

class Exit:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)