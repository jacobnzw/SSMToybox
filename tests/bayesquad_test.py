from unittest import TestCase

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.stats import multivariate_normal

from models.ungm import UNGM
from models.pendulum import Pendulum
from transforms.bayesquad import GPQuad, TPQuad, GPQuadDerAffine, GPQuadDerHermite, GPQuadDerRBF
from transforms.quad import Unscented, GaussHermite
import time


class GPQuadTest(TestCase):
    def test_weights_rbf(self):
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        hypers = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((n,)), 'noise_var': 1e-8}
        tf = GPQuad(n, unit_sp, hypers)
        wm, wc, wcc = tf.weights_rbf(unit_sp, hypers)
        # print 'wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc)
        # print 'GP model variance: {}'.format(tf.model_var)

    def test_min_var_sigmas(self):
        d = 2
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (d + kappa) - d
        # unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        unit_sp = np.random.rand(d, 5)
        hypers = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((d,)), 'noise_var': 1e-8}
        tf = GPQuad(d, unit_sp, hypers)
        # find minimum variance sigmas
        x_opt = tf._min_var_sigmas().reshape((tf.d, tf.n))
        # plot minvar sigmas
        # ax = plt.figure().add_axes([-2, -2, 2, 2])
        # plt.scatter(x_opt[0, :], x_opt[1, :])
        # plt.show()

    def test_min_var_hypers(self):
        d = 2
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (d + kappa) - d
        unit_sp = Unscented.unit_sigma_points(d, np.sqrt(d + lam))
        # unit_sp = np.random.rand(d, 5)
        hypers = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((d,)), 'noise_var': 1e-8}
        tf = GPQuad(d, unit_sp, hypers)
        # find minimum variance sigmas
        hyp_opt = tf._min_var_hypers()
        # print "s2 = {0}\nel = {1}".format(hyp_opt[0], hyp_opt[1:])

    def test_plot_gp_model(self):
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        # unit_sp = GaussHermite.unit_sigma_points(n, 10)
        hypers = {'sig_var': 1.0, 'lengthscale': 1.0 * np.ones((n,)), 'noise_var': 1e-8}
        tf = GPQuad(n, unit_sp, hypers)
        sys = UNGM()
        tf.plot_gp_model(sys.dyn_eval, unit_sp, np.atleast_1d(1.0), test_range=(-5, 5, 50), plot_dims=(0, 0))
        print "Expected model variance: {}".format(tf.model_var[0])
        # tf.plot_gp_model(sys.meas_eval, unit_sp, np.atleast_1d(1.0), test_range=(-5, 5, 50), plot_dims=(0, 0))

    def test_expected_gp_var(self):
        # compares expected gp variance computed numerically with the closed form expression
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        # unit_sp = GaussHermite.unit_sigma_points(n, 10)
        hypers = {'sig_var': 1.0, 'lengthscale': 1.0 * np.ones((n,)), 'noise_var': 1e-8}
        tf = GPQuad
        al, el = 1.0, 100.0
        res_cf = tf.expected_gp_var(unit_sp, alpha=al, el=el)
        # MC approximation of the expected gp var
        samples = 1e4
        x_mc = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=int(samples)).T
        f_mc = tf.gp_var(x_mc, unit_sp, alpha=al, el=el)
        res_mc = f_mc.sum() / samples
        # Gauss-Hermite quadrature approximation
        order_gh = 50
        x_gh, w_gh = sp.special.he_roots(order_gh)
        f_gh = tf.gp_var(x_gh[na, :], unit_sp, alpha=al, el=el)
        res_gh = w_gh.dot(f_gh)
        print "Closed-form: {:.4e} (RE: {:.2f}%)".format(res_cf, 100 * np.abs(1 - (res_mc / res_cf)))
        print "Monte Carlo: {:.4e} (RE: {:.2f}%)".format(res_mc, 100 * np.abs(1 - (res_cf / res_mc)))
        print "Gauss-Hermi: {:.4e} (RE: {:.2f}%)".format(res_gh, 100 * np.abs(1 - (res_cf / res_gh)))


class GPQuadDerRBFTest(TestCase):
    def test_weights(self):
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        # unit_sp = GaussHermite.unit_sigma_points(n, 10)
        hypers = {'sig_var': 10.0, 'lengthscale': 0.7 * np.ones((n,)), 'noise_var': 1e-8}
        sys = UNGM()
        tf = GPQuadDerRBF(n, unit_sp, hypers, which_der=np.arange(unit_sp.shape[1]))
        wm, wc, wcc = tf.weights_rbf(unit_sp, hypers)
        wmd, wcd, wccd = tf.weights_rbf_der(unit_sp, hypers)
        self.assertTrue(np.array_equal(wm, wmd))
        self.assertTrue(np.array_equal(wc, wcd))
        self.assertTrue(np.array_equal(wcc, wccd))

    def test_plot_gp_model(self):
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        dmask = np.array([0, 1, 2])
        hypers = {'sig_var': 10.0, 'lengthscale': 1.0 * np.ones((n,)), 'noise_var': 1e-8}
        sys = UNGM()
        tf = GPQuadDerRBF(n, unit_sp, hypers, which_der=dmask)
        fig = tf.plot_gp_model(sys.meas_eval, unit_sp, np.atleast_1d(1.0))

    def test_expected_gp_var(self):
        # compares expected gp variance computed numerically with the closed form expression
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        # unit_sp = GaussHermite.unit_sigma_points(n, 10)
        tf = GPQuadDerRBF
        al, el = 10.0, 0.7
        res_cf = tf.expected_gp_var(unit_sp, alpha=al, el=el, jitter=1e-8)
        # MC approximation of the expected gp var
        samples = 1e4
        x_mc = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=int(samples)).T
        f_mc = tf.gp_var(x_mc, unit_sp, alpha=al, el=el)
        res_mc = f_mc.sum() / samples
        # Gauss-Hermite quadrature approximation
        order_gh = 50
        x_gh, w_gh = sp.special.he_roots(order_gh)
        f_gh = tf.gp_var(x_gh[na, :], unit_sp, alpha=al, el=el)
        res_gh = w_gh.dot(f_gh)
        print "Closed-form: {:.4e} (RE: {:.2f}%)".format(res_cf, 100 * np.abs(1 - (res_mc / res_cf)))
        print "Monte Carlo: {:.4e} (RE: {:.2f}%)".format(res_mc, 100 * np.abs(1 - (res_cf / res_mc)))
        print "Gauss-Hermi: {:.4e} (RE: {:.2f}%)".format(res_gh, 100 * np.abs(1 - (res_cf / res_gh)))


class TPQuadTest(TestCase):
    def test_plot_tp_model(self):
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        # unit_sp = GaussHermite.unit_sigma_points(n, 10)
        hypers = {'sig_var': 1.0, 'lengthscale': 1.0 * np.ones((n,)), 'noise_var': 1e-8}
        tf = TPQuad(n, unit_sp, hypers)
        sys = UNGM()
        tf.plot_tp_model(sys.dyn_eval, unit_sp, np.atleast_1d(1.0), test_range=(-5, 5, 50), plot_dims=(0, 0))


class GPQuadDerAffineTest(TestCase):
    def test_kern_affine(self):
        d = 2
        kappa, alpha, beta = 0, 1.0, 2.0
        unit_sp = Unscented.unit_sigma_points(d, kappa, alpha)
        hypers = {'bias': 1.0, 'variance': 1.0 * np.ones((d,)), 'noise_var': 1e-8}
        kern_mat = GPQuadDerAffine.kern_affine_der(unit_sp, hypers)
        d, n = unit_sp.shape
        self.assertEqual(kern_mat.shape, (n + n * d, n + n * d))

    def test_weights_affine(self):
        d = 2
        kappa, alpha, beta = 0, 1.0, 2.0
        unit_sp = Unscented.unit_sigma_points(d, kappa, alpha)
        hypers = {'bias': 1.0, 'variance': 1.0 * np.ones((d,)), 'noise_var': 1e-8}
        tf = GPQuadDerAffine(d, unit_sp=unit_sp, hypers=hypers)
        wm, wc, wcc = tf.weights_affine(unit_sp, hypers)


class GPQuadDerHermiteTest(TestCase):
    plots = False

    def test_ind_sum(self):
        res = GPQuadDerHermite.ind_sum(1, 0)
        self.assertTrue(np.array_equal(res, np.zeros((1, 1))))
        res = GPQuadDerHermite.ind_sum(3, 1)
        self.assertTrue(np.array_equal(res, np.eye(3)))
        res = GPQuadDerHermite.ind_sum(2, 2)
        correct_res = np.array([[2, 1, 1, 0], [0, 1, 1, 2]])
        self.assertTrue(np.array_equal(res, correct_res))

    def test_multihermite(self):
        x = np.array([0, 1, -1])
        res = GPQuadDerHermite.multihermite(x, 3)
        correct_res = np.array([0, -2, 2])
        self.assertTrue(np.array_equal(res, correct_res))
        x = np.array([[1, 2], [1, 2]])
        res = GPQuadDerHermite.multihermite(x, [1, 2])
        correct_res = np.array([0, 6])
        self.assertTrue(np.array_equal(res, correct_res))
        # test handling of negative degrees
        x = np.array([[0, 1, 0, -1, 0], [0, 0, 1, 0, -1]])
        res0 = GPQuadDerHermite.multihermite(x, [0, 0])
        res1 = GPQuadDerHermite.multihermite(x, [-1, 0])
        self.assertTrue(np.array_equal(res0, res1))
        self.assertTrue(np.array_equal(res0, np.ones(5)))

    def test_multihermite_grad(self):
        # one-dimensional inputs
        x = np.array([[0, 1, -1]])
        res = np.zeros((x.shape[1]))
        for i in range(x.shape[1]):
            res[i] = GPQuadDerHermite.multihermite_grad(x[:, i, na], 3)
        correct_res = np.array([-3, 0, 0])
        self.assertTrue(np.array_equal(res, correct_res))
        # multi-dimensional inputs
        x = np.array([[1, 2, 1], [1, 2, 2]])
        res = np.zeros((x.shape))
        for i in range(x.shape[1]):
            res[:, i] = GPQuadDerHermite.multihermite_grad(x[:, i, na], [1, 2])
        correct_res = np.array([[0, 3, 3], [2, 8, 4]])
        self.assertTrue(np.array_equal(res, correct_res))
        # test handling of negative degrees
        x = np.array([[0, 1, 0, -1, 0], [0, 0, 1, 0, -1]])
        res0 = np.zeros((x.shape))
        res1 = np.zeros((x.shape))
        for i in range(x.shape[1]):
            res0[:, i] = GPQuadDerHermite.multihermite_grad(x[:, i, na], [0, 0])
            res1[:, i] = GPQuadDerHermite.multihermite_grad(x[:, i, na], [-1, 0])
        self.assertTrue(np.array_equal(res0, res1))
        self.assertTrue(np.array_equal(res0, np.zeros(x.shape)))

    def test_kernel_ut(self):
        c = GPQuadDerHermite(1)
        x = np.hstack((np.zeros((2, 1)), np.eye(2), -np.eye(2)))
        kff = c._kernel_ut(x, x)
        self.assertEqual(kff.shape, (5, 5))  # check shape
        self.assertTrue(np.array_equal(kff, kff.T))  # check symmetry
        la.cholesky(kff + 1e-16 * np.eye(5))  # positive definite?
        print "cond(Kff): {0:.2e}, cond(Kff_norm): {1:.2e}".format(la.cond(kff),
                                                                   la.cond(self.normalize_kernel_matrix(kff)))
        if self.plots:
            x = np.atleast_2d(np.linspace(-3, 3, 50).T)
            kff = c._kernel_ut(x, x)
            self.plot_matrix(kff)

    def test_kernel_ut_dx(self):
        c = GPQuadDerHermite(1)
        # test one-dimensional symmetric inputs
        x = np.array([[0, -1, 1]])
        # x = np.atleast_2d(np.linspace(-5, 5, 20))
        d, n = x.shape
        kfd, kdd = c._kernel_ut_dx(x, x)
        self.assertEqual(kfd.shape, (n, d * n))  # check shapes
        self.assertEqual(kdd.shape, (d * n, d * n))
        self.assertTrue(np.array_equal(kdd, kdd.T))  # check symmetry
        print "{}D: cond(Kfd): {:.2e}, cond(Kdd): {:.2e}".format(d, la.cond(kfd), la.cond(kdd))
        # test multi-dimensional inputs
        # x = np.hstack((np.zeros((2, 1)), np.eye(2), -np.eye(2)))
        x = Unscented.unit_sigma_points(2)
        d, n = x.shape
        kfd, kdd = c._kernel_ut_dx(x, x)
        self.assertEqual(kfd.shape, (n, d * n))  # check shapes
        self.assertEqual(kdd.shape, (d * n, d * n))
        self.assertTrue(np.array_equal(kdd, kdd.T))  # check symmetry
        la.cholesky(kdd + 1e-8 * np.eye(d * n))  # posdef ?
        print "{}D: cond(Kfd): {:.2e}, cond(Kdd): {:.2e}".format(d, la.cond(kfd), la.cond(kdd))
        if self.plots:
            x = np.atleast_2d(np.linspace(-3, 3, 50).T)
            kfd, kdd = c._kernel_ut_dx(x, x)
            self.plot_matrix(kdd)

    def test_kern_hermite_der(self):
        x = Unscented.unit_sigma_points(2)
        # x = np.hstack((np.zeros((2, 1)), np.eye(2), -np.eye(2)))
        d, n = x.shape
        hyp = {'lambda': np.ones(4)}
        c = GPQuadDerHermite(1)
        kmat = c.kern_hermite_der(x, hyp)
        self.assertEqual(kmat.shape, (n + d * n, n + d * n))
        self.assertTrue(np.array_equal(kmat, kmat.T))  # symmetric ?
        print "cond(K): {:.2e}, cond(K_norm): {:.2e}".format(la.cond(kmat), la.cond(self.normalize_kernel_matrix(kmat)))
        la.inv(kmat + 1e-8 * np.eye(n + d * n))  # invertible ?
        la.cholesky(kmat + 1e-8 * np.eye(n + d * n))  # posdef ?
        if self.plots:
            x = np.atleast_2d(np.linspace(-3, 3, 25).T)
            kmat = c.kern_hermite_der(x, hyp)
            self.plot_matrix(kmat)

    def test_weights_hermite(self):
        hyp = {'lambda': np.ones(4), 'noise_var': 1e-8}
        c = GPQuadDerHermite(1, hypers=hyp)
        d = 2
        n = 2 * d + 1
        usp = np.hstack((np.zeros((d, 1)), np.eye(d), -np.eye(d)))
        wm, wc, wcc = c.weights_hermite(unit_sp=usp, hypers=hyp)

    def plot_matrix(self, kmat):
        evals = la.eigvals(kmat + 1e-8 * np.eye(50))
        plt.subplot(121)
        plt.imshow(kmat, interpolation='none')
        plt.colorbar()
        plt.subplot(122)
        plt.plot(evals, ls='-', marker='.', ms=15)
        plt.title('Eigvals range: [{0:.2e}, {1:.2e}]'.format(evals.min(), evals.max()))
        plt.show()

    def normalize_kernel_matrix(self, kmat):
        kmat_diag = np.diag(kmat)
        return kmat * (1.0 / (np.sqrt(np.outer(kmat_diag, kmat_diag))))
