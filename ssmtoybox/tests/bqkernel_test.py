from unittest import TestCase

import numpy as np
import numpy.linalg as la
from numpy import newaxis as na

from ssmtoybox.bq.bqkern import RBF, RBFStudent


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

    def test_exp_x_xpx(self):
        mi_1d = np.array([[0, 1, 2]])
        ke = self.kern_rbf_1d.exp_x_xpx(mi_1d)
        self.assertTrue(ke.shape == mi_1d.shape)
        self.assertTrue(np.array_equal(ke, np.array([[0, 1, 0]])))

        mi_2d = np.array([[0, 1, 0, 1, 0, 2],
                          [0, 0, 1, 1, 2, 0]])
        ke_true = np.array([[0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0]])
        ke = self.kern_rbf_2d.exp_x_xpx(mi_2d)
        self.assertTrue(ke.shape == mi_2d.shape)
        self.assertTrue(np.array_equal(ke, ke_true))

    def test_exp_x_pxpx(self):
        mi_1d = np.array([[0, 1, 2]])
        ke = self.kern_rbf_1d.exp_x_pxpx(mi_1d)
        ke_true = np.array([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 3]])
        self.assertTrue(ke.shape == (mi_1d.shape[1], mi_1d.shape[1]))
        self.assertTrue(np.array_equal(ke, ke_true))

        mi_2d = np.array([[0, 1, 0, 1, 0, 2],
                          [0, 0, 1, 1, 2, 0]])
        ke_true = np.array([[1, 0, 0, 0, 1, 1],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [1, 0, 0, 0, 3, 1],
                            [1, 0, 0, 0, 1, 3]])
        ke = self.kern_rbf_2d.exp_x_pxpx(mi_2d)
        self.assertTrue(ke.shape == (mi_2d.shape[1], mi_2d.shape[1]))
        self.assertTrue(np.array_equal(ke, ke_true))

    def test_mc_verification(self):
        dim = 2

        q = self.kern_rbf_2d.exp_x_kx(self.par_2d, self.data_2d)
        Q = self.kern_rbf_2d.exp_x_kxkx(self.par_2d, self.par_2d, self.data_2d)
        R = self.kern_rbf_2d.exp_x_kx(self.par_2d, self.data_2d)

        # approximate expectations using MC
        num_samples = 10000
        x_samples = np.random.multivariate_normal(np.zeros((dim, )), np.eye(dim), size=num_samples).T
        k = self.kern_rbf_2d.eval(self.par_2d, x_samples, self.data_2d, scaling=False)
        q_mc = (1/num_samples) * k.sum(axis=0)
        Q_mc = (1/num_samples) * (k[:, na, :] * k[..., na]).sum(axis=0)
        R_mc = (1/num_samples) * (x_samples[..., na] * k[na, ...]).sum(axis=1)

        # compare MC approximates with analytic expressions
        self.assertTrue(np.allclose(q, q_mc), 'q diff {:.4f}'.format(np.abs(q - q_mc).max()))
        self.assertTrue(np.allclose(Q, Q_mc), 'Q diff {:.4f}'.format(np.abs(Q - Q_mc).max()))
        self.assertTrue(np.allclose(R, R_mc), 'R diff {:.4f}'.format(np.abs(Q - Q_mc).max()))

    def test_par_gradient(self):
        dim = 2
        x = np.hstack((np.zeros((dim, 1)), np.eye(dim), -np.eye(dim)))
        y = x[0, :]

        par = np.array([[1, 1, 3]], dtype=float)
        kernel = RBF(dim, par)
        dK_dpar = kernel.der_par(par.squeeze(), x)


class RBFStudentKernelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        from ssmtoybox.mtran import FullySymmetricStudent
        cls.points = FullySymmetricStudent.unit_sigma_points(2)
        cls.num_pts = cls.points.shape[1]

    def test_expectations_dim(self):
        dim = 2
        par = np.array([[1.5, 3.0, 3.0]])
        ker = RBFStudent(dim, par)

        q = ker.exp_x_kx(par, self.points)
        self.assertTrue(q.shape == (self.num_pts, ))

        Q = ker.exp_x_kxkx(par, par, self.points)
        self.assertTrue(Q.shape == (self.num_pts, self.num_pts))
        self.assertTrue(np.array_equal(Q, Q.T), 'Q not symmetric')

        R = ker.exp_x_xkx(par, self.points)
        self.assertTrue(R.shape == (dim, self.num_pts))

        kbar = ker.exp_x_kxx(par)
        self.assertTrue(kbar.shape == ())

        kbarbar = ker.exp_xy_kxy(par)
        self.assertTrue(kbarbar.shape == ())