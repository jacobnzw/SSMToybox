from unittest import TestCase

import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
import numba as nb

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
        def kern_eval(x1, x2, par):
            # define straightforward implementation to check easily
            dim, num_pts_1 = x1.shape
            dim, num_pts_2 = x2.shape
            assert dim == par.shape[1]-1
            alpha = par[0, 0]
            Li = np.linalg.inv(np.diag(par[0, 1:] ** 2))
            K = np.zeros((num_pts_1, num_pts_2))
            for i in range(num_pts_1):
                for j in range(num_pts_2):
                    dx = x1[:, i] - x2[:, j]
                    K[i, j] = np.exp(-0.5 * (dx.T.dot(Li).dot(dx)))
            return alpha**2 * K

        # check dimension, shape, symmetry and positive definiteness
        K = self.kern_rbf_1d.eval(self.par_1d, self.data_1d)
        self.assertTrue(K.ndim == 2)
        self.assertTrue(K.shape == (3, 3))
        self.assertTrue(np.array_equal(K, K.T))
        la.cholesky(K)
        # same result as the obvious implementation?
        K_true = kern_eval(self.data_1d, self.data_1d, self.par_1d)
        self.assertTrue(np.array_equal(K, K_true))

        # higher-dimensional inputs
        K = self.kern_rbf_2d.eval(self.par_2d, self.data_2d)
        self.assertTrue(K.ndim == 2)
        self.assertTrue(K.shape == (5, 5))
        self.assertTrue(np.array_equal(K, K.T))
        la.cholesky(K)
        # same result as the obvious implementation?
        K_true = kern_eval(self.data_2d, self.data_2d, self.par_2d)
        self.assertTrue(np.array_equal(K, K_true))

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
        def kx_eval(x, par):
            # simple straightforward easy to check implementation
            dim, num_pts = x.shape
            assert dim == par.shape[1]-1
            alpha = par[0, 0]
            L = np.diag(par[0, 1:] ** 2)
            A = np.linalg.inv(L + np.eye(dim))
            c = alpha**2 * np.linalg.det(np.linalg.inv(L) + np.eye(dim)) ** (-0.5)
            q = np.zeros((num_pts, ))
            for i in range(num_pts):
                q[i] = c * np.exp(-0.5*(x[:, i].T.dot(A).dot(x[:, i])))
            return q

        q = self.kern_rbf_1d.exp_x_kx(self.par_1d, self.data_1d)
        q_true = kx_eval(self.data_1d, self.par_1d)
        self.assertTrue(q.shape == (3,))
        self.assertTrue(np.alltrue(q >= 0))
        self.assertTrue(np.array_equal(q, q_true))

        q = self.kern_rbf_2d.exp_x_kx(self.par_2d, self.data_2d)
        q_true = kx_eval(self.data_2d, self.par_2d)
        self.assertTrue(q.shape == (5,))
        self.assertTrue(np.alltrue(q >= 0))
        self.assertTrue(np.array_equal(q, q_true))

    def test_exp_x_kxx(self):
        self.kern_rbf_1d.exp_x_kxx(self.par_1d)
        self.kern_rbf_2d.exp_x_kxx(self.par_2d)

    def test_exp_xy_kxy(self):
        self.kern_rbf_1d.exp_xy_kxy(self.par_1d)
        self.kern_rbf_2d.exp_xy_kxy(self.par_2d)

    def test_exp_x_xkx(self):
        def xkx_eval(x, par):
            # simple straightforward easy to check implementation
            dim, num_pts = x.shape
            assert dim == par.shape[1]-1
            alpha = par[0, 0]
            L = np.diag(par[0, 1:] ** 2)
            A = np.linalg.inv(L + np.eye(dim))
            c = alpha**2 * np.linalg.det(np.linalg.inv(L) + np.eye(dim)) ** (-0.5)
            R = np.zeros(x.shape)
            for i in range(num_pts):
                R[:, i] = c * np.exp(-0.5*(x[:, i].T.dot(A).dot(x[:, i]))) * (A.dot(x[:, i]))
            return R

        r = self.kern_rbf_1d.exp_x_xkx(self.par_1d, self.data_1d)
        self.assertTrue(r.shape == (1, 3))
        r_true = xkx_eval(self.data_1d, self.par_1d)
        self.assertTrue(np.allclose(r, r_true))

        r = self.kern_rbf_2d.exp_x_xkx(self.par_2d, self.data_2d)
        r_true = xkx_eval(self.data_2d, self.par_2d)
        self.assertTrue(r.shape == (2, 5))
        self.assertTrue(np.allclose(r, r_true))

    def test_exp_x_kxkx(self):
        q = self.kern_rbf_1d.exp_x_kxkx(self.par_1d, self.par_1d, self.data_1d)
        self.assertTrue(q.shape == (3, 3))
        self.assertTrue(np.array_equal(q, q.T), 'Result not symmetric.')
        la.cholesky(q)

        q = self.kern_rbf_2d.exp_x_kxkx(self.par_2d, self.par_2d, self.data_2d)
        self.assertTrue(q.shape == (5, 5))
        self.assertTrue(np.array_equal(q, q.T), 'Result not symmetric.')
        la.cholesky(q)

    def test_mc_verification(self):
        dim = 2

        q = self.kern_rbf_2d.exp_x_kx(self.par_2d, self.data_2d)
        Q = self.kern_rbf_2d.exp_x_kxkx(self.par_2d, self.par_2d, self.data_2d)
        R = self.kern_rbf_2d.exp_x_xkx(self.par_2d, self.data_2d)

        # approximate expectations using cumulative moving average MC
        def cma_mc(new_samples, old_avg, old_avg_size, axis=0):
            b_size = new_samples.shape[axis]
            return (new_samples.sum(axis=axis) + old_avg_size * old_avg) / (old_avg_size + b_size)

        batch_size = 100000
        num_iter = 100
        q_mc, Q_mc, R_mc = 0, 0, 0
        for i in range(num_iter):
            # sample from standard Gaussian
            x_samples = np.random.multivariate_normal(np.zeros((dim, )), np.eye(dim), size=batch_size).T
            k = self.kern_rbf_2d.eval(self.par_2d, x_samples, self.data_2d, scaling=False)
            q_mc = cma_mc(k, q_mc, i*batch_size, axis=0)
            Q_mc = cma_mc(k[:, na, :] * k[..., na], Q_mc, i*batch_size, axis=0)
            R_mc = cma_mc(x_samples[..., na] * k[na, ...], R_mc, i*batch_size, axis=1)

        # compare MC approximates with analytic expressions
        tol = 2e-3
        print('Norm of the difference using {:d} samples.'.format(batch_size*num_iter))
        print('q {:.2e}'.format(np.linalg.norm(q - q_mc)))
        print('Q {:.2e}'.format(np.linalg.norm(Q - Q_mc)))
        print('R {:.2e}'.format(np.linalg.norm(R - R_mc)))
        self.assertLessEqual(np.linalg.norm(q - q_mc), tol)
        self.assertLessEqual(np.linalg.norm(Q - Q_mc), tol)
        self.assertLessEqual(np.linalg.norm(R - R_mc), tol)

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
        from ssmtoybox.mtran import FullySymmetricStudentTransform
        cls.points = FullySymmetricStudentTransform.unit_sigma_points(2)
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