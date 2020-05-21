import random
import numpy as np

def noiseadd(y, percent, rate):
    """
    Returns <y> with a noisy random <percent> number of points.
    Noisiness means the value which was y0 becomes a value between [0, 2 * y0]
    * Maybe it's not a good function (it's a completely random criteria).
    * I've read that the gaussian distribution is a good way to add noise to a signal.
    It'd great to try using it instead of this crappy method.
    """
    for i in range(len(y)):
        if random.random() < percent:
            y[i] *= random.uniform(1 - rate, 1 + rate)

