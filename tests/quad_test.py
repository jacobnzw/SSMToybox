from unittest import TestCase

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from transforms.quad import MonteCarlo
from models.ungm import UNGM


def sum_of_squares(x, pars, dx=False):
    """Sum of squares test function.

    If x is Gaussian random variable than x.T.dot(x) is chi-squared distributed with mean d and variance 2d,
    where d is the dimension of x.
    """
    if not dx:
        return np.atleast_1d(x.T.dot(x))
    else:
        return np.atleast_1d(2 * x)


class MonteCarloTest(TestCase):
    def test_crash(self):
        d = 1
        tmc = MonteCarlo(d, n=1e4)
        f = UNGM().dyn_eval
        mean = np.zeros(d)
        cov = np.eye(d)
        # does it crash ?
        tmc.apply(f, mean, cov, np.atleast_1d(1.0))

    def test_increasing_samples(self):
        d = 1
        tmc = (
            MonteCarlo(d, n=1e1),
            MonteCarlo(d, n=1e2),
            MonteCarlo(d, n=1e3),
            MonteCarlo(d, n=1e4),
            MonteCarlo(d, n=1e5),
        )
        f = sum_of_squares  # UNGM().dyn_eval
        mean = np.zeros(d)
        cov = np.eye(d)
        # does it crash ?
        for t in tmc:
            print t.apply(f, mean, cov, np.atleast_1d(1.0))
