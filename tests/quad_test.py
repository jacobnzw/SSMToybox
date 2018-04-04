from unittest import TestCase

import numpy as np
from transforms.quad import MonteCarlo, SphericalRadialTrunc, FullySymmetricStudent
from ssmod import UNGM


def sum_of_squares(x, pars, dx=False):
    """Sum of squares test function.

    If x is Gaussian random variable than x.T.dot(x) is chi-squared distributed with mean d and variance 2d,
    where d is the dimension of x.
    """
    if not dx:
        return np.atleast_1d(x.T.dot(x))
    else:
        return np.atleast_1d(2 * x)


def cartesian2polar(x, pars, dx=False):
    return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan2(x[1], x[0])])


class SigmaPointTruncTest(TestCase):
    def test_apply(self):
        d, d_eff = 5, 2
        t = SphericalRadialTrunc(d, d_eff)
        f = cartesian2polar
        mean, cov = np.zeros(d), np.eye(d)
        t.apply(f, mean, cov, None)


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
            print(t.apply(f, mean, cov, np.atleast_1d(1.0)))


class FullySymmetricStudentTest(TestCase):

    def test_symmetric_set(self):

        # 1D points
        dim = 1
        sp = FullySymmetricStudent.symmetric_set(dim, [])
        self.assertEqual(sp.ndim, 2)
        self.assertEqual(sp.shape, (dim, 1))
        sp = FullySymmetricStudent.symmetric_set(dim, [1])
        self.assertEqual(sp.shape, (dim, 2*dim))
        sp = FullySymmetricStudent.symmetric_set(dim, [1, 1])
        self.assertEqual(sp.shape, (dim, 2*dim*(dim-1)))

        # 2D points
        dim = 2
        sp = FullySymmetricStudent.symmetric_set(dim, [])
        self.assertEqual(sp.shape, (dim, 1))
        sp = FullySymmetricStudent.symmetric_set(dim, [1])
        self.assertEqual(sp.shape, (dim, 2*dim))
        sp = FullySymmetricStudent.symmetric_set(dim, [1, 1])
        self.assertEqual(sp.shape, (dim, 2 * dim * (dim - 1)))

        # 3D points
        dim = 3
        sp = FullySymmetricStudent.symmetric_set(dim, [1, 1])
        self.assertEqual(sp.shape, (dim, 2 * dim * (dim - 1)))

    def test_crash(self):
        dim = 1
        mt = FullySymmetricStudent(dim, degree=3)
        f = UNGM().dyn_eval
        mean = np.zeros(dim)
        cov = np.eye(dim)
        # does it crash ?
        mt.apply(f, mean, cov, np.atleast_1d(1.0))

        dim = 2
        mt = FullySymmetricStudent(dim, degree=5)
        f = sum_of_squares
        mean = np.zeros(dim)
        cov = np.eye(dim)
        # does it crash ?
        mt.apply(f, mean, cov, np.atleast_1d(1.0))
