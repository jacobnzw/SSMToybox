from unittest import TestCase
from kernel import RBF, Affine
import numpy as np
import numpy.linalg as la


class RBFKernelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hypers_1d = {'alpha': 1.0, 'el': 3.0 * np.ones(1, )}
        cls.hypers_2d = {'alpha': 1.0, 'el': 3.0 * np.ones(2, )}
        cls.kern_rbf_1d = RBF(1, cls.hypers_1d)
        cls.kern_rbf_2d = RBF(2, cls.hypers_2d)
        cls.data_1d = np.array([[1, -1, 0]], dtype=float)
        cls.data_2d = np.hstack((np.zeros((2, 1)), np.eye(2), -np.eye(2)))

    def test_eval(self):
        # check dimension, shape, symmetry and positive definiteness
        K_1d = self.kern_rbf_1d.eval(self.data_1d)
        self.assertTrue(K_1d.ndim == 2)
        self.assertTrue(K_1d.shape == (3, 3))
        self.assertTrue(np.array_equal(K_1d, K_1d.T))
        la.cholesky(K_1d)
        # higher-dimensional inputs
        K_2d = self.kern_rbf_2d.eval(self.data_2d)
        self.assertTrue(K_2d.ndim == 2)
        self.assertTrue(K_2d.shape == (5, 5))
        self.assertTrue(np.array_equal(K_2d, K_2d.T))
        la.cholesky(K_2d)

    def test_exp_x_kx(self):
        pass
