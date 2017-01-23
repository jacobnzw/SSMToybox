from unittest import TestCase

import numpy as np
import numpy.linalg as la

from transforms.bqkernel import RBF


class RBFKernelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.par_1d = np.array([[1, 3]])
        cls.par_2d = np.array([[1, 3, 3]])
        cls.kern_rbf_1d = RBF(1, cls.par_1d)
        cls.kern_rbf_2d = RBF(2, cls.par_2d)
        cls.data_1d = np.array([[1, -1, 0]], dtype=float)
        cls.data_2d = np.hstack((np.zeros((2, 1)), np.eye(2), -np.eye(2)))
        cls.test_data_1d = np.atleast_2d(np.linspace(-5, 5, 50))
        cls.test_data_2d = np.random.multivariate_normal(np.zeros((2,)), np.eye(2), 50).T

    def test_eval(self):

        # check dimension, shape, symmetry and positive definiteness
        K = self.kern_rbf_1d.eval(self.par_1d, self.data_1d)
        self.assertTrue(K.ndim == 2)
        self.assertTrue(K.shape == (3, 3))
        self.assertTrue(np.array_equal(K, K.T))
        la.cholesky(K)

        # higher-dimensional inputs
        K = self.kern_rbf_2d.eval(self.par_2d, self.data_2d)
        self.assertTrue(K.ndim == 2)
        self.assertTrue(K.shape == (5, 5))
        self.assertTrue(np.array_equal(K, K.T))
        la.cholesky(K)

        # check computation of cross-covariances kx, kxx
        kx = self.kern_rbf_1d.eval(self.par_1d, self.test_data_1d, self.data_1d)
        kxx = self.kern_rbf_1d.eval(self.par_1d, self.test_data_1d, self.test_data_1d)
        kxx_diag = self.kern_rbf_1d.eval(self.par_1d, self.test_data_1d, self.test_data_1d, diag=True)
        self.assertTrue(kx.shape == (50, 3))
        self.assertTrue(kxx.shape == (50, 50))
        self.assertTrue(kxx_diag.shape == (50,))

        kx = self.kern_rbf_2d.eval(self.par_2d, self.test_data_2d, self.data_2d)
        kxx = self.kern_rbf_2d.eval(self.par_2d, self.test_data_2d, self.test_data_2d)
        kxx_diag = self.kern_rbf_2d.eval(self.par_2d, self.test_data_2d, self.test_data_2d, diag=True)
        self.assertTrue(kx.shape == (50, 5))
        self.assertTrue(kxx.shape == (50, 50))
        self.assertTrue(kxx_diag.shape == (50,))

    def test_exp_x_kx(self):

        q = self.kern_rbf_1d.exp_x_kx(self.par_1d, self.data_1d)
        self.assertTrue(q.shape == (3,))
        self.assertTrue(np.alltrue(q >= 0))

        q = self.kern_rbf_2d.exp_x_kx(self.par_2d, self.data_2d)
        self.assertTrue(q.shape == (5,))
        self.assertTrue(np.alltrue(q >= 0))

    def test_exp_x_kxx(self):
        self.kern_rbf_1d.exp_x_kxx(self.par_1d)
        self.kern_rbf_2d.exp_x_kxx(self.par_2d)

    def test_exp_xy_kxy(self):
        self.kern_rbf_1d.exp_xy_kxy(self.par_1d)
        self.kern_rbf_2d.exp_xy_kxy(self.par_2d)

    def test_exp_x_xkx(self):
        r = self.kern_rbf_1d.exp_x_xkx(self.par_1d, self.data_1d)
        self.assertTrue(r.shape == (1, 3))

        r = self.kern_rbf_2d.exp_x_xkx(self.par_2d, self.data_2d)
        self.assertTrue(r.shape == (2, 5))

    def test_exp_x_kxkx(self):
        q = self.kern_rbf_1d.exp_x_kxkx(self.par_1d, self.par_1d, self.data_1d)
        self.assertTrue(q.shape == (3, 3))
        self.assertTrue(np.array_equal(q, q.T), 'Result not symmetric.')
        la.cholesky(q)

        q = self.kern_rbf_2d.exp_x_kxkx(self.par_2d, self.par_2d, self.data_2d)
        self.assertTrue(q.shape == (5, 5))
        self.assertTrue(np.array_equal(q, q.T), 'Result not symmetric.')
        la.cholesky(q)

