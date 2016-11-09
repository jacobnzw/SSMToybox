from unittest import TestCase

import numpy as np
import numpy.linalg as la

from transforms.bqkernel import RBF


class RBFKernelTest(TestCase):
    # TODO: could be made into general test class for arbitrary kernel
    @classmethod
    def setUpClass(cls):
        cls.hypers_1d = {'alpha': 1.0, 'el': 3.0 * np.ones(1, )}
        cls.hypers_2d = {'alpha': 1.0, 'el': 3.0 * np.ones(2, )}
        cls.kern_rbf_1d = RBF(1, cls.hypers_1d)
        cls.kern_rbf_2d = RBF(2, cls.hypers_2d)
        cls.data_1d = np.array([[1, -1, 0]], dtype=float)
        cls.data_2d = np.hstack((np.zeros((2, 1)), np.eye(2), -np.eye(2)))
        cls.test_data_1d = np.atleast_2d(np.linspace(-5, 5, 50))
        cls.test_data_2d = np.random.multivariate_normal(np.zeros((2,)), np.eye(2), 50).T

    def test_eval(self):
        # check dimension, shape, symmetry and positive definiteness
        K = self.kern_rbf_1d.eval(self.data_1d)
        self.assertTrue(K.ndim == 2)
        self.assertTrue(K.shape == (3, 3))
        self.assertTrue(np.array_equal(K, K.T))
        la.cholesky(K)
        # higher-dimensional inputs
        K = self.kern_rbf_2d.eval(self.data_2d)
        self.assertTrue(K.ndim == 2)
        self.assertTrue(K.shape == (5, 5))
        self.assertTrue(np.array_equal(K, K.T))
        la.cholesky(K)
        # kx, kxx
        kx = self.kern_rbf_1d.eval(self.test_data_1d, self.data_1d)
        kxx = self.kern_rbf_1d.eval(self.test_data_1d, self.test_data_1d)
        kxx_diag = self.kern_rbf_1d.eval(self.test_data_1d, self.test_data_1d, diag=True)
        self.assertTrue(kx.shape == (50, 3))
        self.assertTrue(kxx.shape == (50, 50))
        self.assertTrue(kxx_diag.shape == (50,))
        kx = self.kern_rbf_2d.eval(self.test_data_2d, self.data_2d)
        kxx = self.kern_rbf_2d.eval(self.test_data_2d, self.test_data_2d)
        kxx_diag = self.kern_rbf_2d.eval(self.test_data_2d, self.test_data_2d, diag=True)
        self.assertTrue(kx.shape == (50, 5))
        self.assertTrue(kxx.shape == (50, 50))
        self.assertTrue(kxx_diag.shape == (50,))

    def test_exp_x_kx(self):
        q = self.kern_rbf_1d.exp_x_kx(self.data_1d)
        self.assertTrue(q.shape == (3,))
        self.assertTrue(np.alltrue(q >= 0))
        q = self.kern_rbf_2d.exp_x_kx(self.data_2d)
        self.assertTrue(q.shape == (5,))
        self.assertTrue(np.alltrue(q >= 0))

    def test_exp_x_kxx(self):
        self.kern_rbf_1d.exp_x_kxx()
        self.kern_rbf_2d.exp_x_kxx()

    def test_exp_xy_kxy(self):
        self.kern_rbf_1d.exp_xy_kxy()
        self.kern_rbf_2d.exp_xy_kxy()

    def test_exp_x_xkx(self):
        r = self.kern_rbf_1d.exp_x_xkx(self.data_1d)
        self.assertTrue(r.shape == (1, 3))
        r = self.kern_rbf_2d.exp_x_xkx(self.data_2d)
        self.assertTrue(r.shape == (2, 5))

    def test_exp_x_kxkx(self):
        q = self.kern_rbf_1d.exp_x_kxkx(self.data_1d)
        self.assertTrue(q.shape == (3, 3))
        self.assertTrue(np.array_equal(q, q.T))
        la.cholesky(q)
        q = self.kern_rbf_2d.exp_x_kxkx(self.data_2d)
        self.assertTrue(q.shape == (5, 5))
        self.assertTrue(np.array_equal(q, q.T))
        la.cholesky(q)

    def test_exp_model_variance(self):
        emv = self.kern_rbf_1d.exp_model_variance(self.data_1d)
        self.assertTrue(emv >= 0)
        emv = self.kern_rbf_2d.exp_model_variance(self.data_2d)
        self.assertTrue(emv >= 0)

    def test_integral_variance(self):
        ivar = self.kern_rbf_1d.integral_variance(self.data_1d)
        self.assertTrue(ivar >= 0)
        ivar = self.kern_rbf_2d.integral_variance(self.data_2d)
        self.assertTrue(ivar >= 0)

# class AffineKernelTest(TestCase):
#     def test_eval(self):
#         pass
